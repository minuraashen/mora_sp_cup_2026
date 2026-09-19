"""
ablate.py -- run the named configurations and emit the results table.

Each row is produced by the SAME pipeline.denoise_image used for the submission, on
the same held-out split, scored by the unmodified official evaluator. The rejected
configurations are included deliberately: a design decision backed by a measured
negative result is worth more than one asserted in prose.

    python scripts/tools/ablate.py --weights scripts/weights/voxel_nafnet.pth --emit latex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
REPO = SCRIPTS.parent

# name -> (validate.py flags, human-readable label)
ABLATIONS = {
    "classical_only":   (["--mode", "classical"], "Classical only (VST + wavelet + median)"),
    "classical_noprep": (["--mode", "classical", "--no_prepass"], "Classical, no impulse pre-pass"),
    "classical_gat":    (["--mode", "classical", "--classical_method", "gat"],
                         "Classical with Anscombe VST (rejected variant)"),
    "ours":             ([], "NAFNet, fine-tuned (submitted)"),
    "net_prepass":      (["--prepass"], "Ours + gated adaptive median pre-pass (rejected)"),
    "ours_tlc":         (["--tlc", "384"], "Ours + TLC(384)"),
    "ours_tta4":        (["--tta", "4"], "Ours + TTA x4"),
    "ours_tta8":        (["--tta", "8"], "Ours + TTA x8"),
    "ours_tiled":       (["--tile", "512"], "Ours, tiled 512 (identity check)"),
    # --- rejected alternatives, measured rather than argued away
    "ours_chroma":      (["--chroma_smooth", "3"], "Ours + chroma smoothing (rejected)"),
    "ours_gamma":       (["--gamma", "1.05"], "Ours + gamma 1.05 brightening (rejected)"),
    "ours_gat":         (["--gat_prenet"], "Ours + Anscombe VST before net (rejected)"),
}


def run_one(name, flags, weights, device, subset, limit):
    cmd = [sys.executable, str(SCRIPTS / "tools" / "validate.py"),
           "--weights", str(weights), "--tag", f"abl_{name}",
           "--device", device, "--subset", subset, "--skip_subprocess"] + flags
    if limit:
        cmd += ["--limit", str(limit)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED {name}:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        return None
    return json.loads((SCRIPTS / "_runs" / "val" / f"abl_{name}" / "summary.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=SCRIPTS / "weights" / "voxel_nafnet.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--subset", default="val", choices=["val", "fast_val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated subset of ablation names")
    ap.add_argument("--emit", default="text", choices=["text", "latex"])
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(ABLATIONS)
    rows = {}
    for name in names:
        flags, label = ABLATIONS[name]
        print(f"running {name} ...", flush=True)
        s = run_one(name, flags, args.weights, args.device, args.subset, args.limit)
        if s:
            s["label"] = label
            rows[name] = s
            print(f"  composite={s['composite']:.4f}  psnr={s['psnr']:.3f}  "
                  f"{s['ms_per_image']:.0f} ms/img")

    # Merge into any previous results rather than overwriting them: rows are often
    # collected in several passes (some on GPU for speed, some on CPU for runtime),
    # and an overwrite silently drops the ones not in this invocation.
    out = SCRIPTS / "_runs" / "ablation.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(rows)
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    rows = merged

    ref = rows.get("ours", {}).get("composite")
    print(f"\n{'config':<44} {'PSNR':>7} {'SSIM':>8} {'dPSNR':>7} {'dSSIM':>7} "
          f"{'comp':>7} {'vs ours':>8} {'ms/img':>8}")
    for name, s in rows.items():
        delta = f"{s['composite'] - ref:+.4f}" if ref is not None else "--"
        print(f"{s['label']:<44} {s['psnr']:7.3f} {s['ssim']:8.4f} {s['delta_psnr']:+7.3f} "
              f"{s['delta_ssim']:+7.4f} {s['composite']:7.4f} {delta:>8} {s['ms_per_image']:8.0f}")

    if args.emit == "latex":
        print("\n% --- ablation table ---")
        print("\\begin{tabular}{lrrrrr}\n\\toprule")
        print("Configuration & PSNR & SSIM & $\\Delta$PSNR & Composite & ms/img \\\\\n\\midrule")
        for name, s in rows.items():
            print(f"{s['label']} & {s['psnr']:.2f} & {s['ssim']:.4f} & "
                  f"{s['delta_psnr']:+.2f} & {s['composite']:.4f} & {s['ms_per_image']:.0f} \\\\")
        print("\\bottomrule\n\\end{tabular}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
