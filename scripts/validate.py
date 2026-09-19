"""Reproduce the reported validation numbers on the held-out public images.

    python scripts/validate.py

Scores the 46 held-out images with the organizers' unmodified
``evaluation/evaluate.py`` and prints mean PSNR, SSIM and official composite.
The split is regenerated deterministically, so this needs no stored state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from ncw.evaluation import validate_official
from ncw.inference import load_checkpoint, sha256_file
from ncw.paths import DEFAULT_CHECKPOINT
from ncw.split import make_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N images")
    parser.add_argument("--subset", choices=("val", "fast_val"), default="val")
    parser.add_argument("--prediction_dir", type=Path,
                        default=Path(__file__).resolve().parent / "_validation" / "pred")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "_validation" / "metrics.json")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    split = make_split()
    ids = split[args.subset]
    print(f"[split] {len(split['train'])} train / {len(split['val'])} val; "
          f"val sigma {split['val_sigma_min']:.2f}..{split['val_sigma_max']:.2f}")
    if args.limit:
        ids = ids[: args.limit]

    print(f"[model] {args.weights.name}")
    digest = sha256_file(args.weights)
    print(f"[model] sha256 {digest}")
    model, checkpoint = load_checkpoint(args.weights, args.device)

    start = time.time()
    result = validate_official(model, ids, torch.device(args.device),
                               args.prediction_dir, tile_size=args.tile, progress=True)
    elapsed = time.time() - start

    print(f"\n[result] {result['count']} images in {elapsed / 60:.1f} min")
    print(f"  PSNR      {result['psnr']:.4f} dB")
    print(f"  SSIM      {result['ssim']:.6f}")
    print(f"  composite {result['composite']:.8f}")

    reference = checkpoint.get("metrics")
    if reference and args.limit == 0 and args.subset == "val":
        print("\n  metric      checkpoint      this run        delta")
        for key in ("psnr", "ssim", "composite"):
            print(f"  {key:<10}  {reference[key]:<15.10f} {result[key]:<15.10f} "
                  f"{result[key] - reference[key]:+.3e}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "checkpoint": args.weights.name,
        "sha256": digest,
        "subset": args.subset,
        "count": result["count"],
        "psnr": result["psnr"],
        "ssim": result["ssim"],
        "composite": result["composite"],
        "elapsed_seconds": elapsed,
        "per_image": result["rows"],
    }, indent=2), encoding="utf-8")
    print(f"\n[report] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
