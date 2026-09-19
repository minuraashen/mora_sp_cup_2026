"""NCW-NAFNet: noise-conditioned, wavelet-informed NAFNet.

The NAFNet backbone is wrapped, never modified. Conditioning enters through
FiLM at every encoder stage and the middle block; fixed Haar details enter
after each downsample. Every added path is scaled by a scalar gate initialised
to zero, so an untrained wrapper reproduces ``backbone(input)`` exactly.
"""

from __future__ import annotations

import torch
from torch import nn

from . import nafnet
from .conditioning import (
    AnalyticalNoiseMap,
    ConditioningAdapter,
    NoiseRefiner,
    WaveletAdapter,
    impulse_feature,
)
from .wavelet import haar_pyramid


class NCWNAFNet(nn.Module):
    """Encoder conditioning at H, H/2, H/4, H/8 and the middle at H/16.

    Haar levels 1..4 inject at H/2..H/16, matching the resolution after each
    downsample, so no resampling is needed in practice.
    """

    def __init__(self,
                 width: int = 32,
                 middle_blk_num: int = 12,
                 enc_blk_nums=(2, 2, 4, 8),
                 dec_blk_nums=(2, 2, 2, 2),
                 a_norm=(0.017094939, 0.0165322929, 0.0190453349),
                 b_norm=(0.006362492, 0.0056840836, 0.0054923607),
                 refiner_width: int = 16,
                 refiner_blocks: int = 2,
                 condition_hidden: int = 32,
                 sigma_scale: float = 4.0,
                 impulse_scale: float = 8.0,
                 use_analytical_noise_map: bool = True,
                 use_learned_noise_refiner: bool = True,
                 use_impulse_feature: bool = True,
                 use_noise_conditioning: bool = True,
                 use_wavelet_features: bool = True):
        super().__init__()
        self.backbone = nafnet.NAFNet(img_channel=3, width=width,
                                      middle_blk_num=middle_blk_num,
                                      enc_blk_nums=tuple(enc_blk_nums),
                                      dec_blk_nums=tuple(dec_blk_nums))
        self.analytical = AnalyticalNoiseMap(a_norm, b_norm)
        self.noise_refiner = NoiseRefiner(refiner_width, refiner_blocks)

        channels = [width * 2 ** i for i in range(5)]
        # One conditioner per encoder stage plus one for the middle block.
        self.conditioners = nn.ModuleList(
            [ConditioningAdapter(4, c, condition_hidden) for c in channels])
        # Wavelet details are injected post-downsample, so skip the full-res width.
        self.wavelet_adapters = nn.ModuleList(
            [WaveletAdapter(c) for c in channels[1:]])

        self.sigma_scale = float(sigma_scale)
        self.impulse_scale = float(impulse_scale)
        self.use_analytical_noise_map = bool(use_analytical_noise_map)
        self.use_learned_noise_refiner = bool(use_learned_noise_refiner)
        self.use_impulse_feature = bool(use_impulse_feature)
        self.use_noise_conditioning = bool(use_noise_conditioning)
        self.use_wavelet_features = bool(use_wavelet_features)

    def conditioning_features(self, image: torch.Tensor) -> torch.Tensor:
        """Four channels: three sigma maps plus one impulse confidence map."""
        if self.use_analytical_noise_map:
            sigma = self.analytical(image)
        else:
            sigma = torch.zeros_like(image)
        if self.use_learned_noise_refiner:
            sigma = (sigma + self.noise_refiner(image)).clamp_min(1e-6)
        sigma = (sigma * self.sigma_scale).clamp(0.0, 1.0)

        if self.use_impulse_feature:
            impulse = impulse_feature(image, self.impulse_scale)
        else:
            impulse = image[:, :1] * 0
        return torch.cat((sigma, impulse), dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, height, width = inp.shape
        x_in = self.backbone.check_image_size(inp)

        # Conditioning is derived from the padded tensor so every stage lines up.
        condition = self.conditioning_features(x_in)
        wavelets = haar_pyramid(x_in, 4) if self.use_wavelet_features else None

        x = self.backbone.intro(x_in)
        encs = []
        for level, (encoder, down) in enumerate(zip(self.backbone.encoders,
                                                    self.backbone.downs)):
            x = encoder(x)
            if self.use_noise_conditioning:
                x = self.conditioners[level](x, condition)
            # The skip carries conditioned features, so the decoder sees them too.
            encs.append(x)
            x = down(x)
            if wavelets is not None:
                x = self.wavelet_adapters[level](x, wavelets[level])

        x = self.backbone.middle_blks(x)
        if self.use_noise_conditioning:
            x = self.conditioners[4](x, condition)

        for decoder, up, enc_skip in zip(self.backbone.decoders,
                                         self.backbone.ups,
                                         reversed(encs)):
            x = decoder(up(x) + enc_skip)

        x = self.backbone.ending(x) + x_in
        return x[:, :, :height, :width]

    def adapter_parameters(self):
        """Every parameter outside the wrapped backbone."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def gate_values(self) -> dict:
        """Current scalar gate values, for logging identity drift."""
        values = {}
        for index, adapter in enumerate(self.conditioners):
            values[f"condition_{index}_gamma"] = float(adapter.gamma_gate)
            values[f"condition_{index}_beta"] = float(adapter.beta_gate)
        for index, adapter in enumerate(self.wavelet_adapters, 1):
            values[f"wavelet_{index}"] = float(adapter.gate)
        return values


def build_from_checkpoint_config(config: dict) -> NCWNAFNet:
    """Instantiate the model described by a checkpoint's embedded config."""
    model_config = dict(config.get("model", {}))
    noise = config.get("noise", {})
    if "a_norm" in noise:
        model_config["a_norm"] = noise["a_norm"]
    if "b_norm" in noise:
        model_config["b_norm"] = noise["b_norm"]
    return NCWNAFNet(**model_config)
