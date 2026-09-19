"""
pipeline.py -- THE SINGLE SEAM.

``denoise_image`` is the only definition in this project of what happens to an image.
denoise.py (the submission), validate.py, ablate.py and preflight.py all call it, so
there is structurally no way for the pipeline we validate to differ from the pipeline
we submit.

Order: gated impulse repair -> NAFNet -> clamp -> quantise. Nothing brightens the
image at any point; the measured residual mean is +1.5/255, i.e. there is no exposure
gap, and the metric compares against ground truth at the original brightness.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

import classical
import infer as infer_mod
import io_utils


@dataclass(frozen=True)
class Config:
    weights: str = ""
    device: str = "cpu"
    tile: int = 0                 # 0 = whole image
    tta: int = 1                  # 1 = off, 4 = flips, 8 = full D4
    # Gated adaptive median before the network. OFF for the network path: a threshold
    # sweep (k = 4.5/6/8/12) showed the composite rising monotonically as fewer pixels
    # are touched, converging on the no-pre-pass score -- the trained network repairs
    # impulses better than a median does, so any median pass strictly costs. It remains
    # ON for the classical fallback, where it is worth +0.0152 composite.
    prepass: bool = False
    gate_k: float = classical.DEFAULT_GATE_K
    tlc: int = 0                  # 0 = off, else the local-pool kernel size
    mode: str = "net"             # 'net' or 'classical'

    # --- ablation-only switches. All default off; none are used by the submission.
    # They exist so that rejected alternatives are rejected on measured evidence.
    chroma_smooth: int = 0        # >0 = box radius applied to Cb/Cr after the network
    gamma: float = 0.0            # >0 = global tone adjustment (expected to hurt)
    gat_prenet: bool = False      # variance-stabilise before the network
    classical_method: str = "wavelet"   # fallback variant: "wavelet" or "gat"

    def with_tile(self, tile: int) -> "Config":
        return replace(self, tile=tile)


def denoise_image(img_u8: np.ndarray, cfg: Config, model=None) -> np.ndarray:
    """uint8 HxWx3 -> uint8 HxWx3."""
    img01 = img_u8.astype(np.float32) / 255.0

    if cfg.prepass:
        img01, _ = classical.repair_impulses(img01, k=cfg.gate_k)

    if cfg.mode == "classical" or model is None:
        out01 = classical.classical_denoise(img01, method=cfg.classical_method)
        return io_utils.quantize_u8(out01)

    gat_scale = None
    if cfg.gat_prenet:
        img01, gat_scale = classical.gat_forward(img01)

    x = torch.from_numpy(np.ascontiguousarray(img01)).permute(2, 0, 1)[None]
    x = x.to(cfg.device)
    with torch.inference_mode():
        y, _ = infer_mod.forward_with_oom_ladder(model, x, tile=cfg.tile, tta=cfg.tta)
    out01 = y[0].permute(1, 2, 0).float().cpu().numpy()

    if gat_scale is not None:
        out01 = classical.gat_inverse(out01, gat_scale)
    if cfg.chroma_smooth:
        out01 = classical.chroma_smooth(out01, radius=cfg.chroma_smooth)
    if cfg.gamma:
        out01 = classical.apply_gamma(out01, cfg.gamma)

    return io_utils.quantize_u8(out01)
