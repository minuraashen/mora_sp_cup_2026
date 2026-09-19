"""
select_checkpoint.py -- pick which checkpoint ships, using the official evaluator.

This exists because in-training "best" tracking is not trustworthy enough to decide the
submission on its own. During run1 the validation subset alternated between a 23-image
fast subset and the full 46-image set, and a 23-image score is not comparable to a
46-image one (the full set scores systematically lower on this data), so the recorded
"best" could reflect which subset a checkpoint happened to land on. The final step is
also the fully-annealed model and the most likely winner, yet it is not necessarily a
multiple of the full-validation interval.

So every candidate -- both weight sets inside every checkpoint on disk -- is re-scored
here on the full held-out split by scripts/tools/validate.py, i.e. by the unmodified
evaluation/evaluate.py, and the winner is reported.

    python scripts/tools/select_checkpoint.py --run run1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
import common  # noqa: E402


def candidates(run_dir: Path):
    """Every (label, checkpoint path, which-weights) triple worth scoring."""
    out = []
    for name in sorted(p.name for p in run_dir.glob("*.pth")):
        path = run_dir / name
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        step = ck.get("step")
        for key in ("model", "ema"):
            if key in ck and isinstance(ck[key], dict):
                out.append((f"{name}:{key}", path, key, step))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0,
                    help="images per candidate; 0 = the full held-out split")
    ap.add_argument("--only", default="", help="substring filter on candidate names")
    args = ap.parse_args()

    run_dir = common.RUNS_DIR / args.run
    cands = candidates(run_dir)
    if args.only:
        cands = [c for c in cands if args.only in c[0]]
    if not cands:
        print(f"no checkpoints found in {run_dir}")
        return 2

    print(f"scoring {len(cands)} candidate weight sets from {run_dir}\n")
    results = []
    with tempfile.TemporaryDirectory() as td:
        for label, path, key, step in cands:
            # validate.py loads whichever of model/ema/params it finds, so write a
            # single-key temporary checkpoint to force the one we want.
            ck = torch.load(path, map_location="cpu", weights_only=False)
            tmp = Path(td) / f"{label.replace(':', '_')}.pth"
            torch.save({"model": ck[key], "arch": ck.get("arch", "w32")}, tmp)

            cmd = [sys.executable, str(SCRIPTS / "tools" / "validate.py"),
                   "--weights", str(tmp), "--tag", "sel_" + label.replace(":", "_"),
                   "--device", args.device, "--skip_subprocess"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {label:32s} FAILED\n{r.stdout[-400:]}{r.stderr[-400:]}")
                continue
            s = json.loads((common.RUNS_DIR / "val" / ("sel_" + label.replace(":", "_"))
                            / "summary.json").read_text())
            results.append((s["composite"], label, step, s))
            print(f"  {label:32s} step={str(step):>6s}  composite={s['composite']:.6f}  "
                  f"psnr={s['psnr']:.3f}  ssim={s['ssim']:.4f}", flush=True)

    if not results:
        return 1
    results.sort(reverse=True)
    best = results[0]
    print(f"\nWINNER: {best[1]} (step {best[2]}) composite={best[0]:.6f}")
    print(f"  spread over {len(results)} candidates: "
          f"{results[-1][0]:.6f} .. {results[0][0]:.6f}")
    common.write_json(common.ARTIFACTS_DIR / "checkpoint_selection.json", {
        "run": args.run, "n_candidates": len(results),
        "winner": {"label": best[1], "step": best[2], **best[3]},
        "all": [{"label": lb, "step": st, "composite": c} for c, lb, st, _ in results],
    })
    print(f"\nexport it with:\n  python scripts/train/export_model.py --ckpt "
          f"{run_dir / best[1].split(':')[0]}")
    if best[1].split(":")[1] == "ema":
        print("  (note: the winner is the EMA weight set -- export_model.py prefers "
              "'model' when both are present, so pass the EMA explicitly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
