"""Checkpoint loading, deterministic PNG I/O and overlap-tiled inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .model import NCWNAFNet, build_from_checkpoint_config

TILE = 256
OVERLAP = 64
HALO = 64


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path, device="cpu") -> tuple[NCWNAFNet, dict]:
    """Build the model from the config stored inside the checkpoint and load it.

    The checkpoint is self-describing: it carries the resolved configuration
    and the validation metrics alongside the weights, so no external YAML is
    needed. The load is strict -- every tensor must match.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"{path} is not an NCW checkpoint (missing 'model'/'config')")

    model = build_from_checkpoint_config(checkpoint["config"])
    state = checkpoint["model"]
    # fp16-stored checkpoints are upcast; inference always runs in float32.
    state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, checkpoint


def read_rgb(path: Path, device="cpu") -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def write_png(path: Path, tensor: torch.Tensor) -> None:
    """Quantise round-half-up and write 8-bit RGB.

    ``floor(x * 255 + 0.5)`` rather than ``torch.round``, which rounds halves to
    even and would shift a small fraction of pixels by one level.
    """
    array = tensor.detach().clamp(0, 1).mul(255).add(0.5).floor().byte().cpu()[0]
    array = array.permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG",
                                            compress_level=6, optimize=False)


def _tile_starts(total: int, tile: int, overlap: int) -> list[int]:
    """Evenly spaced tile origins covering ``total`` with at least ``overlap``."""
    if tile >= total:
        return [0]
    count = max(2, int(np.ceil((total - overlap) / (tile - overlap))))
    return [int(round(value)) for value in np.linspace(0, total - tile, count)]


def _blend_window(height: int, width: int, overlap: int) -> torch.Tensor:
    """Separable raised-cosine window, strictly positive at the edges."""

    def axis(length: int) -> torch.Tensor:
        values = torch.ones(length, dtype=torch.float32)
        ramp_length = min(overlap, length // 2)
        if ramp_length:
            positions = torch.arange(ramp_length, dtype=torch.float32)
            ramp = 0.5 - 0.5 * torch.cos(torch.pi * (positions + 1.0) / (ramp_length + 1.0))
            values[:ramp_length] = ramp
            values[-ramp_length:] = ramp.flip(0)
        return values

    return axis(height)[:, None] * axis(width)[None, :]


@torch.no_grad()
def predict_tiled(model, image_cpu: torch.Tensor, device,
                  tile: int = TILE, overlap: int = OVERLAP,
                  halo: int = HALO) -> torch.Tensor:
    """Denoise a full image tile by tile, accumulating on CPU.

    Each tile is forwarded with ``halo`` pixels of context on every side, which
    is then discarded -- this keeps the receptive field intact across seams.
    ``overlap`` controls tile spacing and the width of the blending ramp; the
    two are independent. Keeping the accumulators on CPU bounds device memory
    to one crop at a time.
    """
    _, channels, height, width = image_cpu.shape
    if channels != 3:
        raise ValueError(f"expected RGB input, got {channels} channels")

    accumulator = torch.zeros((1, 3, height, width), dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, height, width), dtype=torch.float32)
    windows: dict[tuple[int, int], torch.Tensor] = {}

    for y0 in _tile_starts(height, tile, overlap):
        for x0 in _tile_starts(width, tile, overlap):
            tile_height = min(tile, height - y0)
            tile_width = min(tile, width - x0)

            ey0, ex0 = y0 - halo, x0 - halo
            ey1, ex1 = y0 + tile_height + halo, x0 + tile_width + halo
            pad_top, pad_left = max(0, -ey0), max(0, -ex0)
            pad_bottom, pad_right = max(0, ey1 - height), max(0, ex1 - width)

            crop = image_cpu[:, :, max(0, ey0):min(height, ey1),
                             max(0, ex0):min(width, ex1)]
            if pad_top or pad_left or pad_bottom or pad_right:
                crop = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom),
                             mode="reflect")

            crop_device = crop.to(device)
            output_device = model(crop_device).clamp_(0.0, 1.0)
            output = output_device[:, :, halo:halo + tile_height,
                                   halo:halo + tile_width].float().cpu()
            del crop_device, output_device

            shape = (tile_height, tile_width)
            if shape not in windows:
                windows[shape] = _blend_window(tile_height, tile_width, overlap)
            window = windows[shape][None, None]

            accumulator[:, :, y0:y0 + tile_height, x0:x0 + tile_width].add_(output * window)
            weight_sum[:, :, y0:y0 + tile_height, x0:x0 + tile_width].add_(window)
            del output

    if float(weight_sum.min()) <= 0:
        raise RuntimeError("tiled inference left uncovered pixels")
    return (accumulator / weight_sum).clamp_(0.0, 1.0)


@torch.no_grad()
def denoise_image(model, noisy_path: Path, output_path: Path, device,
                  tile: int = TILE) -> None:
    image = read_rgb(noisy_path, "cpu")
    if tile:
        prediction = predict_tiled(model, image, device, tile=tile)
    else:
        prediction = model(image.to(device)).clamp(0.0, 1.0)
    write_png(output_path, prediction)
