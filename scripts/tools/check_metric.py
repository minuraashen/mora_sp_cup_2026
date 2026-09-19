"""
check_metric.py -- HARD GATE.

Asserts that the torch metrics in scripts/common.py agree with the official
scorer (evaluation/evaluate.py, via scikit-image) to well under 1e-6.

Every training decision -- checkpoint selection, ablations, the numbers in the
report -- is made with common.ssim_torch / psnr_torch. If they disagree with
skimage even slightly, every one of those decisions is made on a lie. So this
runs before training starts, and it exits non-zero on any mismatch.

    python scripts/tools/check_metric.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402

# The official implementations, imported from the unmodified organizer file.
sys.path.insert(0, str(common.REPO_ROOT / "evaluation"))
from skimage.metrics import peak_signal_noise_ratio as sk_psnr  # noqa: E402
from skimage.metrics import structural_similarity as sk_ssim  # noqa: E402


def official_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Byte-for-byte the call in evaluation/evaluate.py::ssim_value."""
    return float(sk_ssim(
        a, b,
        channel_axis=-1,
        data_range=1.0,
        win_size=7,
        gaussian_weights=False,
        use_sample_covariance=True,
        K1=0.01,
        K2=0.03,
    ))


def main() -> int:
    ids = [1, 37, 150, 288, 455]
    tol = 1e-6
    worst_ssim = 0.0
    worst_psnr = 0.0
    failures = 0

    print(f"{'id':>5} {'skimage SSIM':>14} {'torch SSIM':>14} {'|d|':>10} "
          f"{'skimage PSNR':>14} {'torch PSNR':>14} {'|d|':>10}")

    for i in ids:
        gt_path = common.PUBLIC_GT / f"{i:03d}.png"
        noisy_path = common.PUBLIC_NOISY / f"{i:03d}_noise.png"
        if not gt_path.exists() or not noisy_path.exists():
            print(f"SKIP {i}: missing {gt_path if not gt_path.exists() else noisy_path}")
            continue

        gt = common.load_rgb_uint8(gt_path).astype(np.float32) / 255.0
        noisy = common.load_rgb_uint8(noisy_path).astype(np.float32) / 255.0

        ref_s = official_ssim(gt, noisy)
        ref_p = float(sk_psnr(gt, noisy, data_range=1.0))

        t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
        t_no = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0)
        got_s = float(common.ssim_torch(t_gt, t_no)[0])
        got_p = float(common.psnr_torch(t_gt, t_no)[0])

        ds, dp = abs(ref_s - got_s), abs(ref_p - got_p)
        worst_ssim = max(worst_ssim, ds)
        worst_psnr = max(worst_psnr, dp)
        if ds > tol or dp > tol:
            failures += 1

        print(f"{i:5d} {ref_s:14.9f} {got_s:14.9f} {ds:10.2e} "
              f"{ref_p:14.9f} {got_p:14.9f} {dp:10.2e}")

    # Also check the composite arithmetic against the organizer's own function.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "official_eval", common.REPO_ROOT / "evaluation" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    for dp_, ds_ in [(4.5, 0.21), (20.0, 0.5), (-1.0, -0.2), (15.0, 0.0), (0.0, 0.44)]:
        ref = ev.official_composite(dp_, ds_)
        got = common.official_composite(dp_, ds_)
        if abs(ref - got) > 1e-12:
            print(f"COMPOSITE MISMATCH at ({dp_}, {ds_}): {ref} vs {got}")
            failures += 1
    print("composite formula matches evaluate.py on 5 probe points")

    print(f"\nworst |dSSIM| = {worst_ssim:.3e}   worst |dPSNR| = {worst_psnr:.3e}   tol = {tol:.0e}")
    if failures:
        print("FAIL: torch metrics do not match the official scorer.")
        return 1
    print("PASS: torch metrics match evaluation/evaluate.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
