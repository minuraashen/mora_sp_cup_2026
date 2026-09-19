"""Competition entry point: denoise a directory of noisy images.

    python scripts/denoise.py --noise_dir <in> --denoised_dir <out>

Reads every ``<id>_noise.png`` from the input directory and writes ``<id>.png``
to the output directory, which is the submission naming convention. Runs on CPU
by default and makes no network requests.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from ncw.inference import denoise_image, load_checkpoint, sha256_file
from ncw.paths import DEFAULT_CHECKPOINT, SUBMISSION_DENOISED, SUBMISSION_NOISY

VALID_EXT = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # The handbook's example uses --noise_dir/--denoised_dir; the provided
    # baseline uses --input_dir/--output_dir. Both are accepted.
    parser.add_argument("--noise_dir", "--input_dir", dest="noise_dir", type=Path,
                        default=SUBMISSION_NOISY,
                        help="directory of <id>_noise.png inputs")
    parser.add_argument("--denoised_dir", "--output_dir", dest="denoised_dir", type=Path,
                        default=SUBMISSION_DENOISED,
                        help="directory to write <id>.png outputs")
    parser.add_argument("--weights", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tile", type=int, default=256,
                        help="tile size for overlap-tiled inference; 0 = whole image")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads; 0 leaves the default")
    return parser


def collect_inputs(noise_dir: Path) -> list[tuple[Path, str]]:
    jobs = []
    for path in sorted(noise_dir.iterdir()):
        if path.suffix.lower() not in VALID_EXT:
            continue
        stem = path.stem
        if stem.endswith(NOISE_SUFFIX):
            stem = stem[: -len(NOISE_SUFFIX)]
        jobs.append((path, stem))
    return jobs


def main() -> int:
    args = build_parser().parse_args()

    if not args.noise_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.noise_dir}")
    if args.noise_dir.resolve() == args.denoised_dir.resolve():
        raise SystemExit("Input and output directories must differ.")
    if not args.weights.is_file():
        raise SystemExit(f"Checkpoint not found: {args.weights}\n"
                         f"See scripts/README.md for the download and expected path.")
    if args.threads:
        torch.set_num_threads(args.threads)

    jobs = collect_inputs(args.noise_dir)
    if not jobs:
        raise SystemExit(f"No images found in {args.noise_dir}")

    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[model] {args.weights.name}")
    print(f"[model] sha256 {sha256_file(args.weights)}")
    model, checkpoint = load_checkpoint(args.weights, device)
    metrics = checkpoint.get("metrics", {})
    if metrics:
        print(f"[model] validation: {metrics.get('psnr', float('nan')):.4f} dB, "
              f"SSIM {metrics.get('ssim', float('nan')):.6f}, "
              f"composite {metrics.get('composite', float('nan')):.8f} "
              f"on {metrics.get('count', '?')} held-out images")

    start = time.time()
    for index, (noisy_path, stem) in enumerate(jobs, 1):
        denoise_image(model, noisy_path, args.denoised_dir / f"{stem}.png",
                      device, tile=args.tile)
        print(f"[denoise] {index}/{len(jobs)} {noisy_path.name} -> {stem}.png", flush=True)

    elapsed = time.time() - start
    print(f"[done] {len(jobs)} images in {elapsed:.1f}s "
          f"({elapsed / len(jobs):.1f}s per image) -> {args.denoised_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
