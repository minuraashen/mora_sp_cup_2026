"""
nafnet.py -- NAFNet (Chen et al., ECCV 2022, "Simple Baselines for Image Restoration").

Self-contained: no dependency on the upstream repository, so inference needs nothing
but torch. Parameter names match the official implementation exactly, which is what
lets the released NAFNet-SIDD-width32 checkpoint load with ``strict=True``.

Also implements TLC (Test-time Local Converter, Chu et al. 2022). NAFNet's Simplified
Channel Attention aggregates with a GLOBAL average pool. Trained on 256x256 crops and
then run on a whole 992x992 image, those channel statistics are computed over 15x more
pixels than the network ever saw, and the resulting distribution shift costs real dB.
TLC swaps the global pool for a local box average at inference only -- no retraining,
no parameter change. ``convert_to_tlc`` performs that swap in place.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over (C,) for each spatial position."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        y = (x - mu) / torch.sqrt(var + self.eps)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """The activation-free nonlinearity: split channels in half and multiply."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LocalAvgPool2d(nn.Module):
    """Box average over a k x k window, same output size (TLC's replacement for a global pool).

    Computed with an integral image so the cost is O(HW) and independent of k -- a
    direct ``avg_pool2d`` with a 576-wide kernel would be ruinous. Degrades to a true
    global pool when the window covers the whole feature map, so a TLC-converted model
    is numerically identical to the original on inputs at or below the window size.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}"

    def forward(self, x):
        h, w = x.shape[-2:]
        k1, k2 = min(h, self.kernel_size), min(w, self.kernel_size)
        if k1 >= h and k2 >= w:
            return F.adaptive_avg_pool2d(x, 1)

        s = x.cumsum(dim=-1).cumsum(dim=-2)
        s = F.pad(s, (1, 0, 1, 0))                      # prepend a zero row and column
        s1 = s[:, :, :-k1, :-k2]
        s2 = s[:, :, :-k1, k2:]
        s3 = s[:, :, k1:, :-k2]
        s4 = s[:, :, k1:, k2:]
        out = (s4 + s1 - s2 - s3) / (k1 * k2)           # 'valid' box mean
        return F.pad(out, [(k2 - 1) // 2, k2 // 2, (k1 - 1) // 2, k1 // 2], mode="replicate")


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)

        # Simplified Channel Attention: global context -> per-channel scale.
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """U-shaped NAFNet. Predicts a residual which is added to the input."""

    def __init__(self, img_channel: int = 3, width: int = 32, middle_blk_num: int = 12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x):
        """Reflect-pad up to a multiple of the total downsampling factor.

        992 = 2^5 * 31 is already a multiple of 16, so the competition images need no
        padding at all; this exists so the model accepts arbitrary sizes without crashing.
        """
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x

    def forward(self, inp):
        _, _, H, W = inp.shape
        x_in = self.check_image_size(inp)

        x = self.intro(x_in)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + x_in
        return x[:, :, :H, :W]


# ------------------------------------------------------------------ presets

ARCHS = {
    # The official SIDD width32 configuration -- this is the one the released
    # checkpoint matches (29.16 M params, 116,861,841 bytes on disk).
    "w32": dict(width=32, middle_blk_num=12, enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)),
    # From-scratch fallback: ~2x the training throughput, ~1.6x faster on CPU.
    # Used only when the pretrained checkpoint cannot be obtained.
    "w32lite": dict(width=32, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 14), dec_blk_nums=(1, 1, 1, 1)),
}


def build(arch: str = "w32") -> NAFNet:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    return NAFNet(img_channel=3, **ARCHS[arch])


def convert_to_tlc(model: nn.Module, kernel_size: int) -> nn.Module:
    """Replace every global pool in the SCA branches with a local box average.

    Inference-time only: no parameters change, so this is applied after loading
    weights and is reversible by rebuilding the model.
    """
    for module in model.modules():
        if isinstance(module, NAFBlock):
            if isinstance(module.sca[0], (nn.AdaptiveAvgPool2d, LocalAvgPool2d)):
                module.sca[0] = LocalAvgPool2d(kernel_size)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name in ARCHS:
        m = build(name)
        n = count_params(m)
        print(f"{name:10s} {n:12,d} params   fp32 {n * 4 / 1e6:7.1f} MB   fp16 {n * 2 / 1e6:7.1f} MB")
    x = torch.randn(1, 3, 992, 992)
    with torch.no_grad():
        y = build("w32")(x)
    print("forward 992x992 ->", tuple(y.shape))
