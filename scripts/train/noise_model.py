"""
noise_model.py -- the measured low-light sensor-noise synthesizer (GPU, seeded).

Every constant here was measured on the 460 public pairs; see
analysis/noise_characterization/REPORT.md. Nothing is guessed.

  var(residual) = s^2 * (a_c * I + b_c)      I in [0,255], per channel c
      a = [4.359, 4.216, 4.857]  (shot / signal-dependent term)
      b = [413.7, 369.6, 357.1]  (read-noise floor, sqrt(b) ~ 19-20 levels)

  severity  s ~ U(0.26, 1.5874) -> per-image sigma spans the measured ~6..43 (8-bit)
            continuum. The bounds are not free: the measured a and b are pooled over
            the whole dataset and therefore already average over severity, so the
            severity distribution must satisfy E[s^2] = 1 or the synthetic variance is
            biased by exactly that factor. For s ~ U(lo,hi) that means
            lo^2 + lo*hi + hi^2 = 3, which fixes hi once lo is chosen.
  impulses  p = 3.3e-4 per pixel per channel, magnitude +-U(60,180)
  hot pixels  the 492 measured fixed locations, fired with p=0.6 per image
  spatial   WHITE -- the measured ACF is 0.027-0.032 at lag 1 and flat to lag 3,
            so there is no correlation structure worth modelling.
  chroma    NOT inflated. Measured Cb/Y = 0.84, Cr/Y = 0.92, i.e. chroma is quieter
            than luma (this is ISP-processed sRGB, not raw). Applying per-channel
            RGB Poisson-Gaussian reproduces that split on its own.

The heavy tail (measured excess kurtosis 1.86-2.36) comes from two places, and in
neither is sigma inflated -- inflating sigma would widen the bulk and break the
variance-vs-intensity fit:
  (a) a two-component Gaussian SCALE MIXTURE on the per-pixel variance, with weights
      solved so E[t]=1 (variance preserved exactly) and E[t^2]=(K+3)/3 for a target
      excess kurtosis K;
  (b) the sparse impulses, which supply the extreme tail beyond ~6 sigma.
Component (a) carries most of the kurtosis: impulses at 0.033% of pixels are far too
rare to produce an excess kurtosis near 2 on their own.
"""

from __future__ import annotations

import torch

A_COEF = (4.35921, 4.21573, 4.85656)
B_COEF = (413.721, 369.608, 357.141)

SEVERITY_MIN = 0.26
SEVERITY_MAX = 1.5874        # solved from lo^2 + lo*hi + hi^2 = 3  ->  E[s^2] = 1

# Two-component Gaussian scale mixture reproducing the measured leptokurtosis.
KURT_MIX_Q = 0.10            # fraction of pixels drawn from the wide component
KURT_TARGET = 0.85           # excess kurtosis from the mixture alone; the sparse
                             # impulses supply the rest of the measured 1.86-2.36

IMPULSE_PROB = 3.3e-4
IMPULSE_MIN = 60.0
IMPULSE_MAX = 180.0

HOTPIX_FIRE_PROB = 0.6
HOTPIX_MIN = 80.0
HOTPIX_MAX = 200.0


class NoiseSynthesizer:
    """Applies the measured degradation to clean crops, on-GPU.

    Args:
        device: torch device.
        hotpix_mask: optional (H, W) bool tensor of fixed defect locations, already
            cropped to the patch geometry by the caller. Passing None disables the
            fixed-defect component (the random impulses still apply).
    """

    def __init__(self, device, generator: torch.Generator | None = None):
        self.device = device
        self.gen = generator
        self.a = torch.tensor(A_COEF, device=device).view(1, 3, 1, 1)
        self.b = torch.tensor(B_COEF, device=device).view(1, 3, 1, 1)

    @staticmethod
    def _mixture_scales(q: float, kurt: float):
        """Per-pixel variance multipliers (t1, t2) with E[t]=1 and E[t^2]=(kurt+3)/3.

        E[t]=1 keeps the total variance exactly on the measured a*I+b; E[t^2] sets the
        excess kurtosis. Solving the two moment equations for a two-point distribution
        with mass q on the wide component gives the closed form below.
        """
        m2 = (kurt + 3.0) / 3.0
        t2 = 1.0 + (1.0 - (1.0 - m2 * (1.0 - q)) / q) ** 0.5
        t1 = (1.0 - q * t2) / (1.0 - q)
        return t1, t2

    def _rand(self, shape, low=0.0, high=1.0):
        r = torch.rand(shape, device=self.device, generator=self.gen)
        return r * (high - low) + low

    def __call__(self, clean01: torch.Tensor, hotpix_mask: torch.Tensor | None = None) -> torch.Tensor:
        """clean01: (N,3,H,W) float in [0,1]  ->  noisy (N,3,H,W) float in [0,1]."""
        n, _, h, w = clean01.shape
        I = clean01 * 255.0

        # --- heteroscedastic Poisson-Gaussian term, per-image severity
        s = self._rand((n, 1, 1, 1), SEVERITY_MIN, SEVERITY_MAX)
        var = (s ** 2) * (self.a * I + self.b)

        # per-pixel variance scale mixture -> heavy tails at unchanged total variance
        t1, t2 = self._mixture_scales(KURT_MIX_Q, KURT_TARGET)
        wide = torch.rand(I.shape, device=self.device, generator=self.gen) < KURT_MIX_Q
        tscale = torch.where(wide, torch.full_like(I, t2), torch.full_like(I, t1))

        eps = torch.randn(I.shape, device=self.device, generator=self.gen)
        out = I + eps * (var * tscale).sqrt()

        # --- sparse random impulses (the measured heavy tail)
        imp = torch.rand(I.shape, device=self.device, generator=self.gen) < IMPULSE_PROB
        if imp.any():
            mag = self._rand(I.shape, IMPULSE_MIN, IMPULSE_MAX)
            sign = torch.where(
                torch.rand(I.shape, device=self.device, generator=self.gen) < 0.5, -1.0, 1.0)
            out = torch.where(imp, out + sign * mag, out)

        # --- fixed-location hot pixels, fired per image
        if hotpix_mask is not None and hotpix_mask.any():
            fire = (torch.rand((n, 1, h, w), device=self.device, generator=self.gen)
                    < HOTPIX_FIRE_PROB)
            # accept either a shared (H,W) frame mask or a per-sample (N,H,W) crop mask
            hm = hotpix_mask.view(1, 1, h, w) if hotpix_mask.dim() == 2 \
                else hotpix_mask.view(n, 1, h, w)
            hp = hm & fire
            mag = self._rand((n, 3, h, w), HOTPIX_MIN, HOTPIX_MAX)
            sign = torch.where(
                torch.rand((n, 3, h, w), device=self.device, generator=self.gen) < 0.5, -1.0, 1.0)
            out = torch.where(hp.expand(-1, 3, -1, -1), out + sign * mag, out)

        # 8-bit quantisation, exactly as the real data was produced
        out = torch.floor(out.clamp(0.0, 255.0) + 0.5)
        return out / 255.0
