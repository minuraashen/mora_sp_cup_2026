"""
dataset.py -- paired crops from the packed memmaps, with D4 augmentation.

Returns the REAL noisy crop, the clean crop, the crop offset, and a synthetic flag.
When the flag is set the training loop discards the real noisy crop and synthesizes
noise from the clean crop on-GPU instead (see noise_model.py); the offset lets it
index the fixed hot-pixel mask at the correct absolute coordinates.

Augmentation is D4 only (flip + rot90). Nothing photometric: the noise is
intensity-dependent (var = a*I + b), so brightness or gamma jitter would put the
image on a different point of that curve than the noise it carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402


class PairedPatches(Dataset):
    def __init__(self, ids, patch: int = 256, p_synth: float = 0.4,
                 seed: int = 0, length: int | None = None):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.patch = int(patch)
        self.p_synth = float(p_synth)
        self.seed = int(seed)
        self.length = int(length) if length is not None else len(self.ids)
        self._noisy = None
        self._clean = None
        self.epoch = 0

    def _ensure_open(self):
        """Open the memmaps lazily so the object stays picklable for spawned workers."""
        if self._noisy is None:
            self._noisy = np.load(common.ARTIFACTS_DIR / "noisy_u8.npy", mmap_mode="r")
            self._clean = np.load(common.ARTIFACTS_DIR / "clean_u8.npy", mmap_mode="r")

    def set_patch(self, patch: int) -> None:
        self.patch = int(patch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        self._ensure_open()
        # Seeded per (epoch, index) so a run is reproducible and resumable.
        rng = np.random.default_rng((self.seed, self.epoch, idx))
        k = int(self.ids[rng.integers(0, len(self.ids))]) - 1

        S = common.IMG_SIZE
        p = self.patch
        y = int(rng.integers(0, S - p + 1))
        x = int(rng.integers(0, S - p + 1))

        clean = np.ascontiguousarray(self._clean[k, y:y + p, x:x + p])
        noisy = np.ascontiguousarray(self._noisy[k, y:y + p, x:x + p])

        # D4: the same transform must be applied to both members of the pair.
        rot = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        if rot:
            clean = np.rot90(clean, rot, axes=(0, 1))
            noisy = np.rot90(noisy, rot, axes=(0, 1))
        if flip:
            clean = clean[:, ::-1]
            noisy = noisy[:, ::-1]

        is_synth = bool(rng.random() < self.p_synth)

        return (
            torch.from_numpy(np.ascontiguousarray(noisy)),
            torch.from_numpy(np.ascontiguousarray(clean)),
            torch.tensor([y, x, rot, int(flip)], dtype=torch.int64),
            torch.tensor(is_synth),
        )


def transform_mask(mask: torch.Tensor, meta: torch.Tensor, patch: int) -> torch.Tensor:
    """Crop the full-frame hot-pixel mask to a patch and apply the same D4 transform.

    The defects live at fixed sensor coordinates, so a synthetic sample must place them
    where they would actually be for that crop -- not at random.
    """
    out = torch.empty((meta.shape[0], patch, patch), dtype=torch.bool, device=mask.device)
    for i in range(meta.shape[0]):
        y, x, rot, flip = (int(v) for v in meta[i])
        m = mask[y:y + patch, x:x + patch]
        if rot:
            m = torch.rot90(m, rot, dims=(0, 1))
        if flip:
            m = torch.flip(m, dims=(1,))
        out[i] = m
    return out
