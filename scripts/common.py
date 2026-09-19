"""
common.py -- shared utilities for the Mora SP Cup 2026 denoising solution (Team Voxel).

Holds the pieces that MUST agree with the official scorer:

  * ``ssim_torch``  -- a torch re-implementation of ``skimage.metrics.structural_similarity``
                       with EXACTLY the settings used by ``evaluation/evaluate.py``
                       (win_size=7, gaussian_weights=False, use_sample_covariance=True,
                        K1=0.01, K2=0.03, data_range=1.0, channel_axis=-1).
  * ``psnr_torch``  -- matches ``skimage.metrics.peak_signal_noise_ratio(data_range=1.0)``.
  * ``official_composite`` -- 0.6*clip(dPSNR/15,0,1) + 0.4*max(dSSIM,0), per image.

Every number the project reports is produced by these functions, and
``scripts/tools/check_metric.py`` asserts them against scikit-image to < 1e-6.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARTIFACTS_DIR = SCRIPTS_DIR / "artifacts"
WEIGHTS_DIR = SCRIPTS_DIR / "weights"
RUNS_DIR = SCRIPTS_DIR / "_runs"

PUBLIC_NOISY = REPO_ROOT / "competition_data" / "public" / "noisy"
PUBLIC_GT = REPO_ROOT / "competition_data" / "public" / "ground_truth"
SUBMISSION_NOISY = REPO_ROOT / "competition_data" / "submissions" / "noisy"
SUBMISSION_DENOISED = REPO_ROOT / "competition_data" / "submissions" / "denoised"

ANALYSIS_DIR = SCRIPTS_DIR / "analysis" / "noise_characterization"

N_PUBLIC = 460
IMG_SIZE = 992

# Measured noise model (8-bit units), from scripts/analysis/noise_characterization/REPORT.md
# var(residual) = A_COEF[c] * I + B_COEF[c]   with I the clean intensity in [0, 255]
A_COEF = (4.35921, 4.21573, 4.85656)     # R, G, B
B_COEF = (413.721, 369.608, 357.141)     # R, G, B


# ---------------------------------------------------------------- misc

def seed_everything(seed: int = 1337) -> None:
    """Seed python / numpy / torch. Import torch lazily so this module stays importable
    in environments without torch (e.g. a pure-numpy analysis run)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def load_split() -> dict:
    """Return {'train': [...], 'val': [...], 'fast_val': [...]} of 1-based image ids."""
    return read_json(ARTIFACTS_DIR / "split.json")


# ---------------------------------------------------------------- image IO

def load_rgb_uint8(path) -> np.ndarray:
    """Read an image as uint8 HxWx3 RGB. Mirrors evaluate.py's ``img.convert("RGB")``."""
    from PIL import Image
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def quantize_u8(x: np.ndarray) -> np.ndarray:
    """Float [0,1] -> uint8, using round-half-UP.

    ``np.round`` is banker's rounding (round-half-to-even): np.round(0.5) == 0.
    That silently shifts a fraction of a percent of pixels relative to the obvious
    convention, so we fix floor(x + 0.5) here and use it everywhere -- training
    validation, ablations and the submitted PNGs all quantize identically.
    """
    return np.floor(np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_png_deterministic(path, u8: np.ndarray) -> None:
    """Write an RGB uint8 array as PNG with fixed encoder settings.

    Fixed ``compress_level``/``optimize`` plus a pinned Pillow makes the PNG *bytes*
    reproducible, not merely the pixels. cv2.imwrite is deliberately avoided (BGR
    ordering footgun, and zlib parameters vary between builds).
    """
    from PIL import Image
    assert u8.dtype == np.uint8 and u8.ndim == 3 and u8.shape[2] == 3, u8.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8, mode="RGB").save(path, format="PNG", compress_level=6, optimize=False)


NOISE_SUFFIX = "_noise"


def strip_noise_suffix(stem: str) -> str:
    """``001_noise`` -> ``001``. Same semantics as baseline/denoise.py (case-insensitive)."""
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[: -len(NOISE_SUFFIX)]
    return stem


# ---------------------------------------------------------------- metrics (torch)

def _uniform_filter_valid(x, size: int = 7):
    """Mean filter over a size x size window, 'valid' region only.

    scikit-image applies ``scipy.ndimage.uniform_filter`` then crops ``(win-1)//2``
    pixels from every side, so the boundary mode never affects the score. Cropping
    first via an unpadded average pool reproduces exactly that interior region.
    """
    import torch.nn.functional as F
    return F.avg_pool2d(x, kernel_size=size, stride=1, padding=0)


def ssim_torch(a, b, win_size: int = 7, data_range: float = 1.0,
               K1: float = 0.01, K2: float = 0.03):
    """SSIM matching evaluation/evaluate.py exactly.

    Args:
        a, b: float tensors, shape (N, C, H, W), values in [0, 1].
    Returns:
        A tensor of shape (N,): the per-image SSIM, averaged over channels
        exactly as ``channel_axis=-1`` does in scikit-image.
    """
    import torch

    a = a.double()
    b = b.double()

    NR = win_size ** 2
    cov_norm = NR / (NR - 1.0)          # use_sample_covariance=True  ->  49/48

    ux = _uniform_filter_valid(a, win_size)
    uy = _uniform_filter_valid(b, win_size)
    uxx = _uniform_filter_valid(a * a, win_size)
    uyy = _uniform_filter_valid(b * b, win_size)
    uxy = _uniform_filter_valid(a * b, win_size)

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    # mean over channels and space == skimage's mean over the cropped volume
    return S.mean(dim=(1, 2, 3))


def psnr_torch(a, b, data_range: float = 1.0):
    """PSNR matching skimage.peak_signal_noise_ratio(data_range=1.0). Shape (N,)."""
    import torch
    mse = ((a.double() - b.double()) ** 2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / mse.clamp_min(1e-20))


def normalize_delta_psnr(delta_psnr: float) -> float:
    return float(np.clip(delta_psnr / 15.0, 0.0, 1.0))


def official_composite(delta_psnr: float, delta_ssim: float) -> float:
    """0.6*clip(dPSNR/15,0,1) + 0.4*max(dSSIM,0) -- copied from evaluation/evaluate.py.

    Applied PER IMAGE and then averaged; the clip and the max do not commute with
    averaging, so computing it on mean deltas would give a different (wrong) number.
    """
    return float(0.6 * normalize_delta_psnr(delta_psnr) + 0.4 * max(delta_ssim, 0.0))


def composite_from_tensors(pred, gt, noisy_psnr, noisy_ssim):
    """Per-image composite for a batch of full images.

    ``pred`` is quantized to uint8 precision first: predictions ship as 8-bit PNGs,
    so scoring un-quantized floats would over-report.

    Args:
        pred, gt: (N, C, H, W) float tensors in [0, 1].
        noisy_psnr, noisy_ssim: (N,) tensors -- the cached scores of the noisy inputs.
    Returns:
        (composite, psnr, ssim) each shape (N,).
    """
    import torch
    pred_q = torch.floor(pred.clamp(0, 1) * 255.0 + 0.5) / 255.0
    p = psnr_torch(pred_q, gt)
    s = ssim_torch(pred_q, gt)
    dp = (p - noisy_psnr) / 15.0
    comp = 0.6 * dp.clamp(0.0, 1.0) + 0.4 * (s - noisy_ssim).clamp_min(0.0)
    return comp, p, s
