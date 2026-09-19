"""Noise conditioning: analytical sensor model, impulse detection, FiLM injection.

The first two are classical signal processing operating directly on the noisy
observation -- a fitted Poisson-Gaussian variance law and an exact 3x3 median
filter. Their outputs modulate the network rather than replacing the input, so
the restoration backbone keeps working on unmodified RGB.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .nafnet import NAFBlock


class AnalyticalNoiseMap(nn.Module):
    """Per-pixel sigma from the measured variance law ``var(I) = a*I + b``.

    ``a`` (shot noise) and ``b`` (read-noise floor) are per-channel constants
    fitted on the 460 public pairs, in normalised [0,1] intensity units. They
    are buffers, not parameters: the sensor model is measured, not learned.
    """

    def __init__(self, a_norm, b_norm, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("a", torch.tensor(a_norm).view(1, 3, 1, 1).float())
        self.register_buffer("b", torch.tensor(b_norm).view(1, 3, 1, 1).float())
        self.eps = float(eps)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        variance = self.a * image.clamp(0.0, 1.0) + self.b
        return variance.clamp_min(self.eps).sqrt()


class NoiseRefiner(nn.Module):
    """Small NAF branch predicting a bounded correction to the analytical sigma.

    The output conv is zero-initialised and the correction is tanh-bounded to
    +/- ``max_delta``, so at initialisation the refiner contributes nothing and
    the analytical map passes through untouched.
    """

    def __init__(self, width: int = 16, blocks: int = 2, max_delta: float = 0.1):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, 1, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(width) for _ in range(blocks)])
        self.ending = nn.Conv2d(width, 3, 3, 1, padding=1)
        self.max_delta = float(max_delta)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.ending(self.blocks(self.intro(image)))) * self.max_delta


def impulse_feature(image: torch.Tensor, scale: float = 8.0) -> torch.Tensor:
    """Confidence map for impulse-like corruption (hot pixels, sparse spikes).

    Computes an exact 3x3 median with a sorting network -- 36 pairwise
    min/max passes over the nine replicate-padded neighbours -- then measures
    how far the centre pixel deviates from it. A sorting network is used
    instead of ``torch.median`` so the whole thing stays differentiable and
    runs on backends without a median kernel.
    """
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    height, width = image.shape[-2:]
    values = [padded[..., dy:dy + height, dx:dx + width]
              for dy in range(3) for dx in range(3)]

    # Bubble sort the nine neighbour tensors elementwise; index 4 is the median.
    for end in range(8, 0, -1):
        for index in range(end):
            low = torch.minimum(values[index], values[index + 1])
            high = torch.maximum(values[index], values[index + 1])
            values[index], values[index + 1] = low, high

    median = values[4]
    deviation = (image - median).abs().amax(1, keepdim=True)
    return (deviation * scale).clamp(0.0, 1.0)


class ConditioningAdapter(nn.Module):
    """FiLM modulation of one backbone stage, gated to start at identity.

    ``gamma`` is tanh-bounded so the multiplicative path cannot explode;
    ``beta`` is unbounded. Both scalar gates start at zero, so the stage is
    untouched until training moves them.
    """

    def __init__(self, cond_channels: int, feature_channels: int, hidden: int = 32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.to_gamma = nn.Conv2d(hidden, feature_channels, 1)
        self.to_beta = nn.Conv2d(hidden, feature_channels, 1)
        self.gamma_gate = nn.Parameter(torch.zeros(()))
        self.beta_gate = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        z = F.interpolate(conditioning, feature.shape[-2:],
                          mode="bilinear", align_corners=False)
        z = self.shared(z)
        gamma = torch.tanh(self.to_gamma(z))
        beta = self.to_beta(z)
        return (1.0 + self.gamma_gate * gamma) * feature + self.beta_gate * beta


class WaveletAdapter(nn.Module):
    """Project nine Haar detail channels into a stage and add through a zero gate."""

    def __init__(self, feature_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(9, feature_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
        if wavelet.shape[-2:] != feature.shape[-2:]:
            wavelet = F.interpolate(wavelet, feature.shape[-2:],
                                    mode="bilinear", align_corners=False)
        return feature + self.gate * self.project(wavelet)
