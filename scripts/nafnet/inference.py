"""Strict checkpoint loading and memory-bounded RGB inference."""

from collections.abc import Callable, Mapping
from pathlib import Path
import hashlib
import re

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn

from .model import NAFNet


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(checkpoint: Path, device: torch.device) -> tuple[NAFNet, dict]:
    """Infer architecture from tensor names, then require a complete weight match."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run python scripts/setup_nafnet.py first."
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint must contain a PyTorch state dictionary.")
    state = payload
    for key in ("params_ema", "params", "state_dict"):
        if key in payload:
            state = payload[key]
            break
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint has no model weights.")
    state = {name.removeprefix("module."): value for name, value in state.items()}
    if not all(isinstance(value, Tensor) for value in state.values()):
        raise ValueError("The model state dictionary must contain tensors only.")
    if "intro.weight" not in state or state["intro.weight"].shape[1] != 3:
        raise ValueError("Expected an RGB NAFNet checkpoint.")

    def stage_depths(prefix: str) -> tuple[int, ...]:
        stages: dict[int, set[int]] = {}
        for name in state:
            match = re.match(rf"{prefix}\.(\d+)\.(\d+)\.", name)
            if match:
                stage, block = map(int, match.groups())
                stages.setdefault(stage, set()).add(block)
        if not stages or sorted(stages) != list(range(len(stages))):
            raise ValueError(f"Checkpoint contains invalid {prefix} stages.")
        for blocks in stages.values():
            if sorted(blocks) != list(range(len(blocks))):
                raise ValueError(f"Checkpoint contains missing {prefix} blocks.")
        return tuple(len(stages[index]) for index in range(len(stages)))

    middle = {int(name.split(".")[1]) for name in state if name.startswith("middle_blks.")}
    if not middle or sorted(middle) != list(range(len(middle))):
        raise ValueError("Checkpoint contains invalid middle blocks.")
    config = {
        "width": int(state["intro.weight"].shape[0]),
        "encoder_blocks": stage_depths("encoders"),
        "middle_blocks": len(middle),
        "decoder_blocks": stage_depths("decoders"),
    }
    model = NAFNet(**config)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device), config


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("Require tile > 0 and 0 <= overlap < tile.")
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def predict(
    model: nn.Module,
    image: Tensor,
    device: torch.device,
    tile: int = 256,
    overlap: int = 32,
    progress: Callable[[int, int], None] | None = None,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> Tensor:
    """Return CPU float32 RGB; feather tile boundaries before clipping/rounding.

    The input and accumulation buffers stay on CPU. Only one tile is sent to
    the selected device at a time. Tiled channel attention differs from a
    whole-image forward pass; tile settings are part of the run configuration.
    """
    if image.ndim != 4 or image.shape[:2] != (1, 3):
        raise ValueError("Expected an image tensor with shape [1, 3, height, width].")
    if tile < 0 or overlap < 0 or (tile > 0 and overlap >= tile):
        raise ValueError("Tile must be zero (full image) or larger than overlap.")
    height, width = image.shape[-2:]
    if tile == 0 or (height <= tile and width <= tile):
        result = model(image.to(device, memory_format=memory_format)).float().cpu()
        if progress:
            progress(1, 1)
        return result

    rows = tile_starts(height, tile, overlap)
    columns = tile_starts(width, tile, overlap)
    output = torch.zeros((1, 3, height, width), dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), dtype=torch.float32)
    tile_height, tile_width = min(tile, height), min(tile, width)
    # A positive floor protects image borders, including 1-pixel-wide inputs.
    y_weights = torch.hann_window(tile_height, periodic=False).clamp_min(0.05)
    x_weights = torch.hann_window(tile_width, periodic=False).clamp_min(0.05)
    window = (y_weights[:, None] * x_weights[None, :])[None, None]
    count = 0
    for top in rows:
        for left in columns:
            region = (..., slice(top, top + tile_height), slice(left, left + tile_width))
            prediction = model(image[region].to(device, memory_format=memory_format)).float().cpu()
            output[region] += prediction * window
            weights[region] += window
            count += 1
            if progress:
                progress(count, len(rows) * len(columns))
    return output / weights


def read_rgb(path: Path) -> tuple[Tensor, tuple[int, int]]:
    with Image.open(path) as source:
        if source.mode != "RGB":
            raise ValueError(f"{path.name}: expected RGB input, found {source.mode}.")
        array = np.array(source, dtype=np.float32) / 255.0
        size = source.size
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous(), size


def to_image(prediction: Tensor) -> Image.Image:
    if not torch.isfinite(prediction).all():
        raise ValueError("Model produced non-finite pixels; no output was saved.")
    pixels = prediction[0].clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
