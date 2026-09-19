"""
losses.py -- training objectives.

PSNRLoss is the primary objective and the choice is a correctness argument, not a
stylistic one. The competition metric averages PER-IMAGE PSNR, and PSNR is
per-image-normalised log-MSE. A plain L1 or L2 loss weights every image by its
absolute error, so the noisiest images dominate the gradient even though the metric
gives them no extra credit -- worse, those are exactly the images where the PSNR term
is least likely to saturate. Optimising log-MSE per image matches what is scored.

The SSIM term (Run 2 only) uses the same skimage-exact SSIM as validation, so the
number being optimised and the number being reported cannot drift apart. Its weight
follows from the metric itself: the composite pays 0.6/15 = 0.04 per dB and 0.4 per
SSIM unit, so 0.1 SSIM is worth about 1 dB. A small positive weight pays; a large one
would trade away the dominant term.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402


class PSNRLoss(nn.Module):
    """-PSNR, averaged over the batch. Minimising this maximises mean per-image PSNR."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return torch.log10(mse + self.eps).mean() * 10.0


class SSIMLoss(nn.Module):
    """1 - SSIM, using the exact windowing of evaluation/evaluate.py."""

    def forward(self, pred, target):
        return 1.0 - common.ssim_torch(pred.clamp(0, 1), target).mean()


class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.0):
        super().__init__()
        self.psnr = PSNRLoss()
        self.ssim_weight = float(ssim_weight)
        self.ssim = SSIMLoss() if ssim_weight > 0 else None

    def forward(self, pred, target):
        loss = self.psnr(pred, target)
        if self.ssim is not None:
            # SSIM is computed in float64 inside common.ssim_torch; keep it out of
            # autocast so the loss scale stays well-conditioned.
            with torch.autocast(device_type=pred.device.type, enabled=False):
                loss = loss + self.ssim_weight * self.ssim(pred.float(), target.float())
        return loss
