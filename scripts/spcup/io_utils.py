"""
io_utils.py -- image loading, deterministic PNG writing, and filename handling.

Deliberately tolerant on input and strict on output: the organizer's invocation may
differ from ours in ways we cannot see (case, extension, whether the `_noise` suffix
is present at all), and any crash forfeits the entire image score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NOISE_SUFFIX = "_noise"


def strip_noise_suffix(stem: str) -> str:
    """``461_noise`` -> ``461``; a stem without the suffix is returned unchanged.

    Case-insensitive, matching baseline/denoise.py, because the organizer's files
    could plausibly arrive as ``461_NOISE.PNG``.
    """
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[: -len(NOISE_SUFFIX)]
    return stem


def list_inputs(input_dir: Path):
    """All image files in a directory, in a fixed order.

    ``sorted`` matters: it makes logs, timings and any accumulation order identical
    between runs, which is part of reproducing the submission exactly.
    """
    return sorted(p for p in Path(input_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in VALID_EXT)


def load_rgb_uint8(path) -> np.ndarray:
    """Read as uint8 HxWx3 RGB.

    ``.convert("RGB")`` is what evaluation/evaluate.py does, and it also silently
    rescues grayscale, palette, RGBA and 16-bit inputs.
    """
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def quantize_u8(x: np.ndarray) -> np.ndarray:
    """Float [0,1] -> uint8 with round-half-UP.

    NOT ``np.round``: that is round-half-to-even, so np.round(0.5) == 0. The two
    conventions disagree on a meaningful fraction of pixels, and validation and the
    submitted PNGs must use the same one.
    """
    return np.floor(np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png_deterministic(path, u8: np.ndarray) -> None:
    """Write RGB uint8 as PNG with pinned encoder settings, so the BYTES reproduce."""
    if u8.dtype != np.uint8 or u8.ndim != 3 or u8.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3, got {u8.dtype} {u8.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8, mode="RGB").save(path, format="PNG", compress_level=6, optimize=False)
