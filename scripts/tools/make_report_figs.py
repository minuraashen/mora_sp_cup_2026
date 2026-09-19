"""
make_report_figs.py -- generate every report figure and table from real run outputs.

Nothing here is hand-authored: each artefact is derived from per_image.csv /
ablation.json produced by validate.py and ablate.py, so the report cannot drift from
the measurements.

    python scripts/tools/make_report_figs.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "spcup"))

import common  # noqa: E402
import io_utils  # noqa: E402

FIGS = SCRIPTS / "report" / "figs"

# Order and grouping of the ablation table; anything after "REJECTED" goes below a rule.
# Must stay in sync with ABLATIONS in ablate.py.
TABLE_ORDER = [
    # what we ship, and the classical tiers beneath it
    "classical_noprep", "classical_only", "ours",
    # available options, measured
    "ours_tta4", "ours_tta8", "ours_tiled",
    # measured and rejected
    "REJECTED", "net_prepass", "ours_tlc", "ours_chroma", "ours_gamma",
    "ours_gat", "classical_gat",
]


def ablation_table(out: Path) -> None:
    src = SCRIPTS / "_runs" / "ablation.json"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    rows = json.loads(src.read_text())
    ref = rows.get("ours", {}).get("composite")

    # Scores only -- deliberately no runtime column. Rows were collected on different
    # devices (network rows on GPU for speed, classical rows on CPU), and while the
    # SCORES are device-independent -- CPU and GPU agree to 5e-6 on the submitted
    # configuration -- the TIMINGS are not comparable across devices. Runtime is
    # reported separately from scripts/tools/bench_cpu.py, all on one CPU.
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Configuration & PSNR & SSIM & Composite \\", r"\midrule"]
    for name in TABLE_ORDER:
        if name == "REJECTED":
            if any(n in rows for n in TABLE_ORDER[TABLE_ORDER.index("REJECTED") + 1:]):
                lines.append(r"\midrule")
            continue
        s = rows.get(name)
        if not s:
            continue
        label = s["label"].replace("&", r"\&")
        cells = [label,
                 format(s["psnr"], ".2f"),
                 format(s["ssim"], ".4f"),
                 format(s["composite"], ".4f")]
        if name == "ours":
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out}")


def composite_vs_sigma(tag: str, out: Path) -> None:
    csv_path = SCRIPTS / "_runs" / "val" / tag / "per_image.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigma_all = np.load(common.ANALYSIS_DIR / "_sigma_blind.npy")
    xs, ys, ps = [], [], []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = int(r["image_id"])
            xs.append(sigma_all[i - 1])
            ys.append(float(r["composite_score"]))
            ps.append(float(r["delta_psnr"]))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
    ax[0].scatter(xs, ys, s=18, alpha=0.8, color="#2b6cb0")
    ax[0].set_xlabel(r"blind $\sigma$ (8-bit)"); ax[0].set_ylabel("composite")
    ax[0].set_title("Composite vs. noise severity"); ax[0].grid(alpha=0.3)
    ax[1].scatter(xs, ps, s=18, alpha=0.8, color="#c05621")
    ax[1].set_xlabel(r"blind $\sigma$ (8-bit)"); ax[1].set_ylabel(r"$\Delta$PSNR (dB)")
    ax[1].set_title("Improvement vs. noise severity"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  wrote {out}  (n={len(xs)})")


def qualitative_crops(tag: str, out: Path, size: int = 200) -> None:
    """Two severity levels x four panels (noisy | baseline | ours | ground truth).

    Deliberately wide and short: the report has a hard 5-page limit, and a tall figure
    is the most expensive thing that can be put in it. The baseline column comes from
    competition_data/public/denoised, which holds the organizer baseline's own outputs,
    so the comparison is against their actual result rather than our reimplementation.
    """
    csv_path = SCRIPTS / "_runs" / "val" / tag / "per_image.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows.sort(key=lambda r: float(r["noisy_psnr"]))
    picks = [rows[0]]          # the hardest image by input noise level

    pred_dir = SCRIPTS / "_runs" / "val" / tag / "pred"
    base_dir = common.REPO_ROOT / "competition_data" / "public" / "denoised"

    fig, axes = plt.subplots(len(picks), 4, figsize=(8.0, 2.15 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(picks):
        i = int(row["image_id"])
        panels = [
            (io_utils.load_rgb_uint8(common.PUBLIC_NOISY / f"{i:03d}_noise.png"), "noisy input"),
            (io_utils.load_rgb_uint8(base_dir / f"{i:03d}.png") if (base_dir / f"{i:03d}.png").exists()
             else None, "organizer baseline"),
            (io_utils.load_rgb_uint8(pred_dir / f"{i:03d}.png"), "ours"),
            (io_utils.load_rgb_uint8(common.PUBLIC_GT / f"{i:03d}.png"), "ground truth"),
        ]
        y = x = (common.IMG_SIZE - size) // 2
        sl = (slice(y, y + size), slice(x, x + size))
        for c, (img, name) in enumerate(panels):
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img[sl])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
        axes[r, 0].set_ylabel(f"id {i:03d}\ninput {float(row['noisy_psnr']):.1f} dB", fontsize=7)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, left=0.045, right=0.995,
                        top=0.92, bottom=0.01)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"  wrote {out}  (ids {[int(p['image_id']) for p in picks]})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ours")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fig", default="")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    want = (lambda n: args.all or args.fig == n)

    if want("ablation"):
        ablation_table(FIGS / "ablation_table.tex")
    if want("sigma"):
        composite_vs_sigma(args.tag, FIGS / "composite_vs_sigma.png")
    if want("crops"):
        qualitative_crops(args.tag, FIGS / "crops.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
