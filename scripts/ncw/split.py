"""The immutable train/validation split.

Deterministic and RNG-free: image ids are ordered by blind noise estimate and
every tenth is held out starting at index 5. Stratifying by severity means the
validation set spans the full sigma continuum rather than over-representing
whichever end a random draw happened to favour.
"""

from __future__ import annotations

import numpy as np

from .paths import ANALYSIS_DIR, N_PUBLIC

VAL_STRIDE = 10
VAL_OFFSET = 5
FAST_VAL_STRIDE = 2


def make_split() -> dict:
    sigma_path = ANALYSIS_DIR / "_sigma_blind.npy"
    sigma = np.load(sigma_path)
    if sigma.shape != (N_PUBLIC,):
        raise ValueError(f"expected {N_PUBLIC} sigmas, got {sigma.shape}")

    order = np.argsort(sigma, kind="stable")
    val_positions = np.arange(VAL_OFFSET, N_PUBLIC, VAL_STRIDE)
    val_index = np.sort(order[val_positions])

    val_ids = [int(i) + 1 for i in val_index]
    held_out = set(val_ids)
    train_ids = [i + 1 for i in range(N_PUBLIC) if i + 1 not in held_out]
    fast_val_ids = val_ids[::FAST_VAL_STRIDE]

    return {
        "train": train_ids,
        "val": val_ids,
        "fast_val": fast_val_ids,
        "method": (f"ids sorted by blind sigma (_sigma_blind.npy); every {VAL_STRIDE}th "
                   f"starting at index {VAL_OFFSET} held out. Deterministic, no RNG, "
                   f"severity-stratified across the full sigma continuum."),
        "val_sigma_min": float(sigma[val_index].min()),
        "val_sigma_max": float(sigma[val_index].max()),
        "all_sigma_min": float(sigma.min()),
        "all_sigma_max": float(sigma.max()),
    }
