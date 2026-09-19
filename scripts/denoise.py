"""
denoise.py -- Mora SP Cup 2026 preliminary-round submission (Team Voxel).

Blind low-light sRGB denoising with a NAFNet restoration core fine-tuned on the 460
public pairs, using a synthetic-noise generator fitted to the measured degradation.

A classical noise-model-gated adaptive median is implemented and available
(``--prepass``), but is OFF for the network path: a gate-threshold sweep showed the
composite rising monotonically as fewer pixels are filtered, so the trained network
repairs impulses better than a median does. It is always applied in ``--mode
classical``, the offline fallback, where it is worth +0.0152 composite.

Reads every ``<id>_noise.png`` from the input directory and writes ``<id>.png`` to the
output directory. Runs fully offline; CPU is the default device.

USAGE (both flag spellings are accepted):
    python scripts/denoise.py --noise_dir  <in> --denoised_dir <out>
    python scripts/denoise.py --input_dir  <in> --output_dir   <out>

No brightness, gamma or tone correction is applied anywhere. The noisy inputs and the
ground truth share the same exposure -- they differ only by noise -- so brightening
would move the output away from the reference and lower both PSNR and SSIM.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# determinism.configure() must run before torch is imported anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent / "spcup"))
import determinism  # noqa: E402

_PRE_ARGS = argparse.ArgumentParser(add_help=False)
_PRE_ARGS.add_argument("--num_threads", type=int, default=0)
_PRE_ARGS.add_argument("--seed", type=int, default=0)
_pre, _ = _PRE_ARGS.parse_known_args()
N_THREADS = determinism.configure(seed=_pre.seed, threads=_pre.num_threads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import classical  # noqa: E402
import infer as infer_mod  # noqa: E402
import io_utils  # noqa: E402
import pipeline  # noqa: E402

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "voxel_nafnet.pth"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)

    # The handbook shows --noise_dir/--denoised_dir; the organizer baseline uses
    # --input_dir/--output_dir. Both are accepted so either invocation works.
    ap.add_argument("--noise_dir", "--input_dir", dest="input_dir", type=Path,
                    help="folder of noisy input images")
    ap.add_argument("--denoised_dir", "--output_dir", dest="output_dir", type=Path,
                    help="folder to write denoised <id>.png into")

    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                    help="default cpu: this is the configuration the submitted images "
                         "were generated with, and it reproduces on any machine")
    ap.add_argument("--tile", type=int, default=0,
                    help="0 = whole image (default); >0 enables overlap-tiled inference")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4, 8],
                    help="self-ensemble size; 1 = off (default)")
    ap.add_argument("--prepass", action="store_true",
                    help="enable the gated adaptive-median impulse repair before the "
                         "network. Off by default: measured to cost composite, because "
                         "the network repairs impulses better than a median does. It is "
                         "always applied in --mode classical, where it helps.")
    ap.add_argument("--gate_k", type=float, default=classical.DEFAULT_GATE_K)
    ap.add_argument("--tlc", type=int, default=0,
                    help="test-time local converter window; 0 = off")
    ap.add_argument("--mode", choices=["net", "classical"], default="net")
    ap.add_argument("--on_missing_weights", choices=["classical", "error"], default="classical",
                    help="a missing checkpoint should degrade the score, not zero it")
    ap.add_argument("--num_threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is unavailable; using CPU.")
        return "cpu"
    return choice


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.input_dir is None or args.output_dir is None:
        build_parser().print_usage()
        print("\nERROR: provide --noise_dir/--denoised_dir (or --input_dir/--output_dir).")
        return 2
    if not args.input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {args.input_dir}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    paths = io_utils.list_inputs(args.input_dir)
    if not paths:
        print(f"ERROR: no images found in {args.input_dir}")
        return 2

    model = None
    mode = args.mode
    if mode == "net":
        if args.weights.exists():
            model, arch = infer_mod.load_model(args.weights, device=device, tlc_kernel=args.tlc)
            if not args.quiet:
                print(f"model: {args.weights.name} (arch={arch}) on {device}, "
                      f"{N_THREADS} threads")
        elif args.on_missing_weights == "classical":
            print(f"WARNING: weights not found at {args.weights}; "
                  f"falling back to the classical pipeline.")
            mode = "classical"
        else:
            print(f"ERROR: weights not found at {args.weights}")
            return 2

    cfg = pipeline.Config(
        weights=str(args.weights), device=device, tile=args.tile, tta=args.tta,
        prepass=(args.prepass or mode == "classical"),
        gate_k=args.gate_k, tlc=args.tlc, mode=mode,
    )

    n_ok = n_fallback = 0
    total = 0.0
    for path in paths:
        out_id = io_utils.strip_noise_suffix(path.stem)
        out_path = args.output_dir / f"{out_id}.png"
        t0 = time.time()
        try:
            img = io_utils.load_rgb_uint8(path)
        except Exception as exc:
            print(f"ERROR: cannot read {path}: {exc}")
            continue

        try:
            out = pipeline.denoise_image(img, cfg, model)
            n_ok += 1
        except Exception as exc:
            # Any failure must still produce a valid file: a crash forfeits the
            # entire image score, whereas a weaker output only reduces it.
            print(f"WARNING: {path.name} failed ({type(exc).__name__}: {exc}); "
                  f"using the classical fallback.")
            try:
                out = pipeline.denoise_image(
                    img, pipeline.Config(mode="classical", prepass=True), None)
            except Exception:
                out = img
            n_fallback += 1

        io_utils.save_png_deterministic(out_path, out)
        dt = time.time() - t0
        total += dt
        if not args.quiet:
            print(f"  {path.name} -> {out_path.name}  ({dt:.2f}s)", flush=True)

    n = n_ok + n_fallback
    print(f"\nProcessed {n} images in {total:.1f}s ({total / max(1, n):.2f}s/image) on {device}.")
    if n_fallback:
        print(f"  {n_fallback} image(s) used the classical fallback.")
    print(f"Results written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
