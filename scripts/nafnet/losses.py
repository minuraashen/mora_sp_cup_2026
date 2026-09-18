"""Restoration losses; checkpoint selection still uses the official composite."""

import torch
from torch.nn import functional as F


def multiscale_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Three-scale SSIM surrogate with local 7x7 moments (not the official metric)."""
    values = []
    for scale in range(3):
        mean_x = F.avg_pool2d(prediction, 7, stride=1)
        mean_y = F.avg_pool2d(target, 7, stride=1)
        var_x = F.avg_pool2d(prediction.square(), 7, stride=1) - mean_x.square()
        var_y = F.avg_pool2d(target.square(), 7, stride=1) - mean_y.square()
        cov = F.avg_pool2d(prediction * target, 7, stride=1) - mean_x * mean_y
        luminance = (2 * mean_x * mean_y + 0.01**2) / (mean_x.square() + mean_y.square() + 0.01**2)
        contrast = (2 * cov + 0.03**2) / (var_x + var_y + 0.03**2)
        values.append((luminance * contrast if scale == 2 else contrast).mean().clamp(1e-5, 1))
        if scale < 2:
            prediction = F.avg_pool2d(prediction, 2)
            target = F.avg_pool2d(target, 2)
    return values[0] ** 0.25 * values[1] ** 0.35 * values[2] ** 0.4


def haar_bands(image: torch.Tensor) -> tuple[torch.Tensor, ...]:
    a, b = image[..., ::2, ::2], image[..., ::2, 1::2]
    c, d = image[..., 1::2, ::2], image[..., 1::2, 1::2]
    return ((a + b - c - d) * 0.5, (a - b + c - d) * 0.5,
            (a - b - c + d) * 0.5)


def restoration_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    charbonnier = torch.sqrt((prediction - target).square() + 1e-6).mean()
    structural = 1 - multiscale_ssim(prediction, target)
    wavelet = sum((a - b).abs().mean() for a, b in zip(haar_bands(prediction), haar_bands(target))) / 3
    return charbonnier + 0.1 * structural + 0.05 * wavelet
