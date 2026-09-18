"""Reproduce the starter metrics and compare a blind classical fallback on val."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np

from nafnet.classical import repair_impulses, wavelet_denoise
from nafnet.scoring import official, score_arrays, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("scripts/runs/nafnet_docs_20260918"))
    parser.add_argument("--data", type=Path, default=Path("competition_data/public"))
    args = parser.parse_args()
    output = args.run / "classical_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(2)
    split = json.loads((args.run / "split.json").read_text(encoding="utf-8"))
    baseline_path = Path(__file__).resolve().parents[1] / "baseline" / "denoise.py"
    spec = importlib.util.spec_from_file_location("starter_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    settings = SimpleNamespace(skip_defect_correction=False, denoise_method="nlm", nlm_h=10.0)
    rows = {"starter_all": [], "starter_validation": [], "adaptive_wavelet_validation": []}
    validation = set(split["validation"])
    started = time.perf_counter()
    for index, image_id in enumerate(sorted(split["train"] + split["validation"]), 1):
        noisy = official.load_rgb_float(args.data / "noisy" / f"{image_id}_noise.png")
        clean = official.load_rgb_float(args.data / "ground_truth" / f"{image_id}.png")
        image_started = time.perf_counter()
        pred = baseline.process_image(noisy.copy(), settings)
        # Match the starter's truncating PNG conversion exactly.
        pred = (np.clip(pred, 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255
        row = {"image_id": image_id, "seconds": time.perf_counter() - image_started,
               **score_arrays(pred, clean, noisy)}
        rows["starter_all"].append(row)
        if image_id in validation:
            rows["starter_validation"].append(row)
            image_started = time.perf_counter()
            repaired, _ = repair_impulses(noisy)
            classical = wavelet_denoise(repaired)
            rows["adaptive_wavelet_validation"].append({
                "image_id": image_id, "seconds": time.perf_counter() - image_started,
                **score_arrays(classical, clean, noisy),
            })
        if index % 20 == 0:
            print(f"Starter reproduction {index}/460 | {time.perf_counter() - started:.0f}s", flush=True)
    results = {name: {"summary": summarize(values), "images": values,
                      "mean_seconds": float(np.mean([row["seconds"] for row in values]))}
               for name, values in rows.items()}
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(name, result["summary"], flush=True)


if __name__ == "__main__":
    main()
