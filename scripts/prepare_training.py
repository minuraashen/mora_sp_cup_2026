"""Create a reproducible 410/50 split and fit synthetic noise on training data."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import nnls

from nafnet.classical import estimate_noise


def read(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (992, 992):
            raise ValueError(f"Invalid competition image: {path}")
        return np.asarray(image, dtype=np.float32) / 255


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("competition_data/public"))
    parser.add_argument("--output", type=Path, default=Path("scripts/runs/nafnet_docs_20260918"))
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    destination = args.output / "split.json"
    if destination.exists():
        raise FileExistsError(f"The fixed split already exists: {destination}")
    rng = np.random.default_rng(args.seed)
    ids = [f"{index:03d}" for index in range(1, 461)]
    severity = []
    for index, image_id in enumerate(ids, 1):
        noisy = read(args.data / "noisy" / f"{image_id}_noise.png")
        severity.append(float(np.mean(estimate_noise(noisy))))
        if index % 50 == 0:
            print(f"Blind severity: {index}/460", flush=True)
    # One random held-out image per adjacent severity band, covering the continuum.
    bands = np.array_split(np.argsort(severity, kind="stable"), 50)
    validation = sorted(ids[int(rng.choice(band))] for band in bands)
    training = sorted(set(ids) - set(validation))
    split = {"seed": args.seed, "method": "50 equal-frequency noisy-only Haar-MAD strata",
             "train": training, "validation": validation,
             "severity": dict(zip(ids, severity)), "fit_uses_training_targets_only": True}
    args.output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    sums = np.zeros((3, 256, 3), dtype=np.float64)
    for index, image_id in enumerate(training, 1):
        clean = read(args.data / "ground_truth" / f"{image_id}.png").reshape(-1, 3)
        noisy = read(args.data / "noisy" / f"{image_id}_noise.png").reshape(-1, 3)
        chosen = rng.choice(len(clean), 4096, replace=False)
        intensities = np.round(clean[chosen] * 255).astype(int)
        residual = noisy[chosen] - clean[chosen]
        for channel in range(3):
            bins = intensities[:, channel]
            values = residual[:, channel]
            sums[channel, :, 0] += np.bincount(bins, minlength=256)
            sums[channel, :, 1] += np.bincount(bins, weights=values, minlength=256)
            sums[channel, :, 2] += np.bincount(bins, weights=values**2, minlength=256)
        if index % 50 == 0:
            print(f"Training-only noise fit: {index}/410", flush=True)
    slopes, floors, diagnostics = [], [], []
    for channel in range(3):
        count, total, square = sums[channel].T
        mean = total / np.maximum(count, 1)
        variance = square / np.maximum(count, 1) - mean**2
        intensity = np.arange(256) / 255
        valid = (count > 100) & (intensity >= 0.06) & (intensity <= 0.94)
        weights = np.sqrt(count[valid])
        matrix = np.stack((intensity[valid], np.ones(valid.sum())), axis=-1)
        fit, _ = nnls(matrix * weights[:, None], variance[valid] * weights)
        slopes.append(float(fit[0])); floors.append(float(fit[1]))
        diagnostics.append({"intensity": intensity[valid].tolist(),
                            "variance": variance[valid].tolist(),
                            "sample_count": count[valid].astype(int).tolist()})
    train_sigma = [split["severity"][image_id] for image_id in training]
    noise = {"units": "normalized RGB [0,1]", "model": "variance = a * clean + b",
             "a": slopes, "b": floors, "training_sigma": train_sigma,
             "median_sigma": float(np.median(train_sigma)), "diagnostics": diagnostics,
             "fit_ids": training, "validation_targets_used": False,
             "caveat": "Pooled sRGB approximation; clipping and scene mix affect the fit."}
    (args.output / "noise_model.json").write_text(json.dumps(noise, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fixed split: {len(training)} train / {len(validation)} validation.")
    print(f"Training-only noise parameters: a={slopes}, b={floors}")


if __name__ == "__main__":
    main()
