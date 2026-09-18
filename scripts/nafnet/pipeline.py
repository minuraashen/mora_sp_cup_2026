"""Adaptive impulse repair, tiled NAFNet, D4 self-ensemble and optional chroma DSP."""

import numpy as np
import torch

from .augmentation import inverse_transform, transform
from .classical import refine_chroma, repair_impulses
from .inference import predict


@torch.inference_mode()
def restore(model, image, device, tile=256, overlap=32, progress=None,
            memory_format=torch.contiguous_format, preprocess="none", tta=False,
            chroma_refine=0.0):
    """Return a CPU RGB tensor and input-derived preprocessing diagnostics."""
    if preprocess not in ("none", "adaptive"):
        raise ValueError("Preprocessing must be 'none' or 'adaptive'.")
    if not 0 <= chroma_refine <= 1:
        raise ValueError("Chroma refinement must be between zero and one.")
    array = image[0].permute(1, 2, 0).cpu().numpy()
    diagnostics = {}
    if preprocess == "adaptive":
        array, diagnostics = repair_impulses(array)
    count = 8 if tta else 1
    result = np.zeros_like(array, dtype=np.float32)
    for symmetry in range(count):
        augmented = np.ascontiguousarray(transform(array, symmetry).transpose(2, 0, 1))
        batch = torch.from_numpy(augmented).unsqueeze(0)
        prediction = predict(model, batch, device, tile, overlap, progress, memory_format)
        pixels = prediction[0].permute(1, 2, 0).numpy()
        result += inverse_transform(pixels, symmetry) / count
    result = refine_chroma(result, chroma_refine)
    return torch.from_numpy(np.ascontiguousarray(result.transpose(2, 0, 1))).unsqueeze(0), diagnostics
