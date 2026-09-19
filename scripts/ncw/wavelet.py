"""Fixed orthonormal Haar analysis -- the classical multiresolution front end.

No learned parameters: the transform is a fixed filter bank, so the detail
bands it exposes are a deterministic function of the input. High-frequency
corruption concentrates in LH/HL/HH, which is exactly what the noise-conditioned
adapters need to see.
"""

from __future__ import annotations

import torch


def haar_dwt(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One level of 2-D Haar decomposition.

    Returns ``(ll, lh, hl, hh)``, each at half resolution. The 0.5 factor makes
    the 2-D transform orthonormal (two 1-D passes of 1/sqrt(2) each). Odd
    dimensions are truncated by the strided slices.
    """
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_pyramid(x: torch.Tensor, levels: int = 4) -> list[torch.Tensor]:
    """Recursively decompose the LL band, collecting detail bands per level.

    Entry ``i`` holds ``cat([lh, hl, hh])`` at resolution ``H / 2**(i+1)`` --
    nine channels for an RGB input. The final LL band is not returned; the
    backbone already carries the low-frequency content.
    """
    features = []
    ll = x
    for _ in range(levels):
        ll, lh, hl, hh = haar_dwt(ll)
        features.append(torch.cat((lh, hl, hh), dim=1))
    return features
