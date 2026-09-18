"""Reuse the supplied evaluator's metric functions without changing its code."""

import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[2] / "evaluation" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("competition_evaluator", _path)
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def score_arrays(prediction: np.ndarray, clean: np.ndarray, noisy: np.ndarray) -> dict:
    """Score the exact rounded RGB pixels that will be written to PNG."""
    prediction = np.round(np.clip(prediction, 0, 1) * 255).astype(np.float32) / 255
    psnr = float(official.sk_psnr(clean, prediction, data_range=1.0))
    ssim = official.ssim_value(clean, prediction)
    noisy_psnr = float(official.sk_psnr(clean, noisy, data_range=1.0))
    noisy_ssim = official.ssim_value(clean, noisy)
    delta_psnr, delta_ssim = psnr - noisy_psnr, ssim - noisy_ssim
    return {"psnr": psnr, "ssim": ssim, "noisy_psnr": noisy_psnr,
            "noisy_ssim": noisy_ssim, "delta_psnr": delta_psnr,
            "delta_ssim": delta_ssim,
            "composite_score": official.official_composite(delta_psnr, delta_ssim)}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot report metrics for an empty set.")
    keys = ("psnr", "ssim", "noisy_psnr", "noisy_ssim", "delta_psnr", "delta_ssim", "composite_score")
    return {"images": len(rows), **{key: float(np.mean([row[key] for row in rows])) for key in keys}}
