"""NAFNet with state-dict names compatible with the official SIDD checkpoints.

Adapted from megvii-research/NAFNet (MIT, copyright 2022 megvii-model).
See ../licenses/NAFNet-LICENSE.txt. Layer normalization uses native PyTorch
operations; BasicSR and custom CUDA extensions are not required.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    """Normalize channels independently at each spatial location."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        expanded = channels * 2
        self.conv1 = nn.Conv2d(channels, expanded, 1)
        self.conv2 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(channels, expanded, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        branch = self.sg(self.conv2(self.conv1(self.norm1(x))))
        branch = self.conv3(branch * self.sca(branch))
        intermediate = x + branch * self.beta
        branch = self.conv5(self.sg(self.conv4(self.norm2(intermediate))))
        return intermediate + branch * self.gamma


class NAFNet(nn.Module):
    """RGB restoration network; defaults match NAFNet-SIDD-width32."""

    def __init__(
        self,
        width: int = 32,
        encoder_blocks: Sequence[int] = (2, 2, 4, 8),
        middle_blocks: int = 12,
        decoder_blocks: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if width < 1 or not encoder_blocks or len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("Use a positive width and equal, nonempty encoder/decoder depths.")
        if any(count < 1 for count in (*encoder_blocks, middle_blocks, *decoder_blocks)):
            raise ValueError("Every stage must contain at least one block.")
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        channels = width
        for count in encoder_blocks:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(*(NAFBlock(channels) for _ in range(middle_blocks)))
        for count in decoder_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)
            ))
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
        self.padder_size = 2 ** len(encoder_blocks)

    def forward(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        pad_height, pad_width = -height % self.padder_size, -width % self.padder_size
        # Skip no-op padding: legacy torch-directml can incorrectly zero it.
        if pad_height or pad_width:
            image = F.pad(image, (0, pad_width, 0, pad_height))
        features = self.intro(image)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = down(features)
        features = self.middle_blks(features)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(up(features) + skip)
        return (self.ending(features) + image)[..., :height, :width]
