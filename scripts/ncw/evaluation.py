"""Scoring against the organizers' unmodified evaluator.

``evaluation/evaluate.py`` is loaded from disk and used as-is, so the numbers
reported here are produced by exactly the code the organizers run. Predictions
are written to PNG and scored from the file, not from the float tensor -- the
8-bit quantisation is part of what gets measured.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from .inference import predict_tiled, read_rgb, write_png
from .paths import OFFICIAL_EVALUATOR, PUBLIC_GT, PUBLIC_NOISY


def load_official_evaluator():
    spec = importlib.util.spec_from_file_location("official_evaluator", OFFICIAL_EVALUATOR)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official evaluator from {OFFICIAL_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def validate_official(model, ids, device, prediction_dir,
                      tile_size: int = 256, progress: bool = False,
                      noisy_dir: Path = PUBLIC_NOISY,
                      clean_dir: Path = PUBLIC_GT) -> dict:
    """Denoise and score the given public image ids.

    Returns the mean PSNR, SSIM and official composite. The composite is
    averaged per image -- the clip and max in the formula are applied before
    averaging, which is what the organizers' script does.
    """
    evaluator = load_official_evaluator()
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    model.eval()
    for index, image_id in enumerate(ids, 1):
        stem = f"{int(image_id):03d}"
        noisy_path = Path(noisy_dir) / f"{stem}_noise.png"
        clean_path = Path(clean_dir) / f"{stem}.png"
        prediction_path = prediction_dir / f"{stem}.png"

        image = read_rgb(noisy_path, "cpu")
        if tile_size:
            prediction = predict_tiled(model, image, device, tile=tile_size)
        else:
            prediction = model(image.to(device)).clamp(0.0, 1.0)
        write_png(prediction_path, prediction)

        rows.append(evaluator.evaluate_image(stem, prediction_path, clean_path, noisy_path))
        del prediction

        if progress and (index == 1 or index % 5 == 0 or index == len(ids)):
            print(f"[validation] {index}/{len(ids)} full images scored "
                  f"(tile={tile_size or 'whole'})", flush=True)

    return {
        "count": len(rows),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "composite": float(np.mean([row["composite_score"] for row in rows])),
        "rows": rows,
    }
