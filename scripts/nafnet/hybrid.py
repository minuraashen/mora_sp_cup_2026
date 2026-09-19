"""Ten-channel NAFNet with identity-initialized Haar skip corrections."""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter
import torch
from torch import nn
from torch.nn import functional as F

from .classical import estimate_noise
from .model import NAFNet


def condition_image(image: np.ndarray, fit: dict) -> np.ndarray:
    """Original RGB, softly repaired RGB, RGB sigma, impulse confidence.

    Parameters a/b are fitted on training pairs only. All image-dependent
    statistics use the noisy observation. No stored defect coordinates.
    Compute on a complete image before tiling to avoid tile-boundary estimates.
    """
    sigma = estimate_noise(image)
    median = median_filter(image, size=(3, 3, 1), mode="reflect")
    difference = np.abs(image - median)
    local_mad = median_filter(difference, size=(3, 3, 1), mode="reflect")
    threshold = np.maximum(4.5 * sigma, np.maximum(5.9304 * local_mad, 0.08))
    confidence = np.clip((difference - threshold) / np.maximum(threshold * 0.5, 1e-6), 0, 1)
    repaired = image + confidence * (median - image)
    signal = uniform_filter(repaired, size=(7, 7, 1), mode="reflect")
    a, b = np.asarray(fit["a"], np.float32), np.asarray(fit["b"], np.float32)
    variance = np.maximum(a * signal + b, 1e-8)
    # Use the fitted brightness dependence, calibrated to this image's severity.
    maps = np.sqrt(variance / np.maximum(variance.mean(axis=(0, 1)), 1e-8)) * sigma
    return np.ascontiguousarray(np.concatenate(
        (image, repaired, maps, confidence.max(axis=2, keepdims=True)), axis=2), dtype=np.float32)


def haar(image):
    a, b = image[..., ::2, ::2], image[..., ::2, 1::2]
    c, d = image[..., 1::2, ::2], image[..., 1::2, 1::2]
    return ((a + b + c + d) * 0.5, (a + b - c - d) * 0.5,
            (a - b + c - d) * 0.5, (a - b - c + d) * 0.5)


def inverse_haar(bands):
    ll, lh, hl, hh = bands
    a = (ll + lh + hl + hh) * 0.5
    b = (ll + lh - hl - hh) * 0.5
    c = (ll - lh + hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    # Pixel shuffle expects four spatial positions grouped per feature channel.
    packed = torch.stack((a, b, c, d), dim=2).flatten(1, 2)
    return F.pixel_shuffle(packed, 2)


class WaveletSkip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.threshold = nn.Conv2d(3, channels * 3, 1)
        nn.init.zeros_(self.threshold.weight)
        nn.init.constant_(self.threshold.bias, -3.0)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features, noise):
        ll, *details = haar(features)
        noise = F.interpolate(noise, size=ll.shape[-2:], mode="nearest")
        thresholds = F.softplus(self.threshold(noise)).chunk(3, dim=1)
        shrunk = [torch.sign(band) * F.relu(torch.abs(band) - threshold)
                  for band, threshold in zip(details, thresholds)]
        corrected = inverse_haar((ll, *shrunk))
        return features + torch.tanh(self.gate) * (corrected - features)


class HybridNAFNet(NAFNet):
    input_channels = 10

    def __init__(self, **config):
        super().__init__(**config)
        width = self.intro.out_channels
        self.intro = nn.Conv2d(10, width, 3, padding=1)
        self.wavelet_skips = nn.ModuleList([WaveletSkip(width), WaveletSkip(width * 2)])

    def forward(self, image):
        height, width = image.shape[-2:]
        ph, pw = -height % self.padder_size, -width % self.padder_size
        if ph or pw:
            image = F.pad(image, (0, pw, 0, ph))
        noise = image[:, 6:9]
        features, skips = self.intro(image), []
        for index, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            features = encoder(features)
            skip = self.wavelet_skips[index](features, noise) if index < 2 else features
            skips.append(skip)
            features = down(features)
        features = self.middle_blks(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)
        return (self.ending(features) + image[:, :3])[..., :height, :width]


def upgrade_model(model, config, device):
    """Preserve the original RGB path exactly at initialization."""
    upgraded = HybridNAFNet(**config)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    intro = torch.zeros_like(upgraded.intro.weight)
    intro[:, :3] = state["intro.weight"]
    state["intro.weight"] = intro
    missing, unexpected = upgraded.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("wavelet_skips.") for key in missing):
        raise ValueError("Unexpected backbone mismatch during hybrid initialization.")
    return upgraded.to(device)
