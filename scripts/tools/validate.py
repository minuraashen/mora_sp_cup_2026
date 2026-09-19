"""
validate.py -- score the full pipeline on the held-out split with the OFFICIAL metric.

The formulas are never reimplemented here. evaluation/evaluate.py is used two ways:

  (a) as a subprocess -- literally the official code path, parsed from its stdout;
  (b) imported by file path, to get per-image rows (its main() is __name__-guarded,
      so importing it executes nothing).

Every run cross-checks (a) against (b) and hard-fails if they disagree, which makes a
silent drift between our reported numbers and the official ones impossible.

    python scripts/tools/validate.py --weights scripts/_runs/run1/best_ema.pth --tag run1_ema
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "spcup"))

import common  # noqa: E402
import determinism  # noqa: E402

determinism.configure(seed=0)

import infer as infer_mod  # noqa: E402
import io_utils  # noqa: E402
import pipeline  # noqa: E402


def load_official():
    spec = importlib.util.spec_from_file_location(
        "official_eval", common.REPO_ROOT / "evaluation" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def run_subprocess_eval(noisy_dir, pred_dir, gt_dir) -> float:
    out = subprocess.run(
        [sys.executable, str(common.REPO_ROOT / "evaluation" / "evaluate.py"),
         "--noisy_dir", str(noisy_dir), "--pred_dir", str(pred_dir), "--gt_dir", str(gt_dir)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"official evaluate.py failed:\n{out.stdout}\n{out.stderr}")
    m = re.search(r"Composite Score:\s*([0-9.]+)", out.stdout)
    if not m:
        raise RuntimeError(f"could not parse composite from:\n{out.stdout}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=SCRIPTS / "weights" / "voxel_nafnet.pth")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--tlc", type=int, default=0)
    ap.add_argument("--prepass", action="store_true")
    ap.add_argument("--no_prepass", action="store_true", help="deprecated; pre-pass is off by default")
    ap.add_argument("--mode", choices=["net", "classical"], default="net")
    ap.add_argument("--subset", default="val", choices=["val", "fast_val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_subprocess", action="store_true")
    # ablation-only switches (see spcup/pipeline.Config)
    ap.add_argument("--chroma_smooth", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--gat_prenet", action="store_true")
    ap.add_argument("--classical_method", default="wavelet", choices=["wavelet", "gat"])
    args = ap.parse_args()

    split = common.load_split()
    ids = split[args.subset]
    if args.limit:
        ids = ids[:args.limit]

    work = common.RUNS_DIR / "val" / args.tag
    gt_dir, noisy_dir, pred_dir = work / "gt", work / "noisy", work / "pred"
    for d in (gt_dir, noisy_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in ids:
        link_or_copy(common.PUBLIC_GT / f"{i:03d}.png", gt_dir / f"{i:03d}.png")
        link_or_copy(common.PUBLIC_NOISY / f"{i:03d}_noise.png", noisy_dir / f"{i:03d}_noise.png")

    model = None
    if args.mode == "net":
        if not args.weights.exists():
            print(f"ERROR: weights not found: {args.weights}")
            return 2
        model, arch = infer_mod.load_model(args.weights, device=args.device, tlc_kernel=args.tlc)
        print(f"model {args.weights} (arch={arch}) device={args.device} "
              f"tile={args.tile} tta={args.tta} tlc={args.tlc} prepass={args.prepass}")

    cfg = pipeline.Config(device=args.device, tile=args.tile, tta=args.tta,
                          prepass=(args.prepass or args.mode == "classical") and not args.no_prepass,
                          tlc=args.tlc, mode=args.mode,
                          chroma_smooth=args.chroma_smooth, gamma=args.gamma,
                          gat_prenet=args.gat_prenet,
                          classical_method=args.classical_method)

    times = []
    for i in ids:
        img = io_utils.load_rgb_uint8(common.PUBLIC_NOISY / f"{i:03d}_noise.png")
        t0 = time.time()
        out = pipeline.denoise_image(img, cfg, model)
        times.append(time.time() - t0)
        io_utils.save_png_deterministic(pred_dir / f"{i:03d}.png", out)
    ms = 1000.0 * float(np.mean(times))

    ev = load_official()
    rows = []
    for i in ids:
        rows.append(ev.evaluate_image(f"{i:03d}", pred_dir / f"{i:03d}.png",
                                      gt_dir / f"{i:03d}.png",
                                      noisy_dir / f"{i:03d}_noise.png"))
    comp_import = float(np.mean([r["composite_score"] for r in rows]))

    if not args.skip_subprocess:
        comp_sub = run_subprocess_eval(noisy_dir, pred_dir, gt_dir)
        if abs(comp_sub - comp_import) > 1e-8:
            print(f"FATAL: import ({comp_import:.10f}) and subprocess ({comp_sub:.10f}) disagree")
            return 1
        print(f"cross-check OK: subprocess == import ({comp_sub:.8f})")

    with open(work / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "tag": args.tag, "n": len(rows), "composite": comp_import,
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "noisy_psnr": float(np.mean([r["noisy_psnr"] for r in rows])),
        "noisy_ssim": float(np.mean([r["noisy_ssim"] for r in rows])),
        "delta_psnr": float(np.mean([r["delta_psnr"] for r in rows])),
        "delta_ssim": float(np.mean([r["delta_ssim"] for r in rows])),
        "ms_per_image": ms,
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    common.write_json(work / "summary.json", summary)

    worst = sorted(rows, key=lambda r: r["composite_score"])[:5]
    print(f"\n=== {args.tag}  (n={len(rows)}, {args.subset}) ===")
    print(f"  PSNR {summary['psnr']:.4f}  (noisy {summary['noisy_psnr']:.4f}, "
          f"delta {summary['delta_psnr']:+.4f})")
    print(f"  SSIM {summary['ssim']:.6f}  (noisy {summary['noisy_ssim']:.6f}, "
          f"delta {summary['delta_ssim']:+.6f})")
    print(f"  COMPOSITE {summary['composite']:.8f}")
    print(f"  {ms:.0f} ms/image on {args.device}")
    print("  worst 5: " + ", ".join(f"{r['image_id']}({r['composite_score']:.3f})" for r in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
