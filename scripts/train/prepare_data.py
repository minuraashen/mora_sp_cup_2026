"""
prepare_data.py -- one-time data preparation.

Produces, into scripts/artifacts/:

  noisy_u8.npy / clean_u8.npy   memmap-able uint8 arrays (460, 992, 992, 3)
  split.json                    train / val / fast_val image ids
  noisy_baseline.json           per-image PSNR & SSIM of the NOISY inputs vs GT
  hotpixels.npy                 (K, 2) coordinates of the fixed hot pixels

Why memmaps rather than an in-RAM python object: the pair cache is 2.72 GB, which
fits in RAM, but on Windows a DataLoader worker is *spawned*, so every worker would
copy it. Two on-disk memmaps let the workers (and the main process) share the OS
page cache instead. Even at num_workers=0 this keeps startup at a few seconds
instead of a 920-image decode on every run.

    python scripts/train/prepare_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

N = common.N_PUBLIC
S = common.IMG_SIZE

# Validation split: sort by blind sigma, take every VAL_STRIDE-th starting at VAL_OFFSET.
# Deterministic by construction -- no RNG, no seed, reproducible from the sigma array
# alone, and it spans the whole severity continuum (sigma 5.9 -> 42.8) rather than
# sampling it randomly. This is the split stated in the report.
VAL_STRIDE = 10
VAL_OFFSET = 5
FAST_VAL_STRIDE = 2      # every 2nd val image -> the in-training fast-val subset

# A pixel is treated as a fixed sensor defect if it fired as an impulse in more than
# this fraction of the sampled images. The measured distribution is sharply bimodal
# with a total gap between 0.02 and 0.50, so any threshold in that gap gives the
# same 492 pixels.
HOTPIX_RATE_THRESHOLD = 0.5


def pack_images() -> None:
    noisy_path = common.ARTIFACTS_DIR / "noisy_u8.npy"
    clean_path = common.ARTIFACTS_DIR / "clean_u8.npy"
    if noisy_path.exists() and clean_path.exists():
        print(f"[pack] already present: {noisy_path.name}, {clean_path.name}")
        return

    common.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    noisy = np.lib.format.open_memmap(noisy_path, mode="w+", dtype=np.uint8, shape=(N, S, S, 3))
    clean = np.lib.format.open_memmap(clean_path, mode="w+", dtype=np.uint8, shape=(N, S, S, 3))

    t0 = time.time()
    for k in range(N):
        i = k + 1
        noisy[k] = common.load_rgb_uint8(common.PUBLIC_NOISY / f"{i:03d}_noise.png")
        clean[k] = common.load_rgb_uint8(common.PUBLIC_GT / f"{i:03d}.png")
        if (k + 1) % 50 == 0:
            print(f"[pack] {k + 1}/{N}  ({time.time() - t0:.0f}s)", flush=True)
    noisy.flush()
    clean.flush()
    print(f"[pack] done in {time.time() - t0:.0f}s -> {noisy_path.name}, {clean_path.name}")


def make_split() -> dict:
    sigma_path = common.ANALYSIS_DIR / "_sigma_blind.npy"
    sigma = np.load(sigma_path)
    assert sigma.shape == (N,), f"expected {N} sigmas, got {sigma.shape}"

    order = np.argsort(sigma, kind="stable")          # ascending severity
    val_pos = np.arange(VAL_OFFSET, N, VAL_STRIDE)
    val_idx = np.sort(order[val_pos])
    val_ids = [int(i) + 1 for i in val_idx]
    train_ids = [i + 1 for i in range(N) if (i + 1) not in set(val_ids)]
    fast_val_ids = val_ids[::FAST_VAL_STRIDE]

    split = {
        "train": train_ids,
        "val": val_ids,
        "fast_val": fast_val_ids,
        "method": (f"ids sorted by blind sigma (_sigma_blind.npy); every {VAL_STRIDE}th "
                   f"starting at index {VAL_OFFSET} held out. Deterministic, no RNG, "
                   f"severity-stratified across the full sigma continuum."),
        "val_sigma_min": float(sigma[val_idx].min()),
        "val_sigma_max": float(sigma[val_idx].max()),
        "all_sigma_min": float(sigma.min()),
        "all_sigma_max": float(sigma.max()),
    }
    common.write_json(common.ARTIFACTS_DIR / "split.json", split)
    print(f"[split] {len(train_ids)} train / {len(val_ids)} val "
          f"({len(fast_val_ids)} fast-val); val sigma "
          f"{split['val_sigma_min']:.2f}..{split['val_sigma_max']:.2f} "
          f"(full set {split['all_sigma_min']:.2f}..{split['all_sigma_max']:.2f})")
    return split


def compute_noisy_baseline(split: dict) -> None:
    """Per-image PSNR/SSIM of the noisy inputs vs GT, for every val image.

    These are the denominators of every delta in the competition metric. Computed
    once here and cached, never recomputed during training.
    """
    out_path = common.ARTIFACTS_DIR / "noisy_baseline.json"
    if out_path.exists():
        print(f"[baseline] already present: {out_path.name}")
        return

    import torch
    noisy = np.load(common.ARTIFACTS_DIR / "noisy_u8.npy", mmap_mode="r")
    clean = np.load(common.ARTIFACTS_DIR / "clean_u8.npy", mmap_mode="r")

    rows = {}
    t0 = time.time()
    for i in split["val"]:
        k = i - 1
        n = torch.from_numpy(np.ascontiguousarray(noisy[k])).permute(2, 0, 1)[None].float() / 255.0
        g = torch.from_numpy(np.ascontiguousarray(clean[k])).permute(2, 0, 1)[None].float() / 255.0
        rows[str(i)] = {
            "psnr": float(common.psnr_torch(n, g)[0]),
            "ssim": float(common.ssim_torch(n, g)[0]),
        }
    common.write_json(out_path, rows)
    mp = float(np.mean([r["psnr"] for r in rows.values()]))
    ms = float(np.mean([r["ssim"] for r in rows.values()]))
    print(f"[baseline] {len(rows)} val images in {time.time() - t0:.0f}s -- "
          f"noisy mean PSNR {mp:.4f} dB, mean SSIM {ms:.6f}")


def extract_hotpixels() -> None:
    """The fixed-location sensor defects, from the cached per-pixel outlier rates."""
    out_path = common.ARTIFACTS_DIR / "hotpixels.npy"
    if out_path.exists():
        print(f"[hotpix] already present: {out_path.name}")
        return
    rate = np.load(common.ANALYSIS_DIR / "_per_pixel_outlier_rate.npy")
    ys, xs = np.nonzero(rate > HOTPIX_RATE_THRESHOLD)
    coords = np.stack([ys, xs], axis=1).astype(np.int16)
    np.save(out_path, coords)
    print(f"[hotpix] {len(coords)} fixed defect pixels "
          f"({100.0 * len(coords) / (S * S):.4f}% of the frame) -> {out_path.name}")


def main() -> int:
    print(f"repo root: {common.REPO_ROOT}")
    pack_images()
    split = make_split()
    extract_hotpixels()
    compute_noisy_baseline(split)
    print("\nprepare_data complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
