"""Paired D4 augmentation and a fitted, continuously varying noise model."""

import numpy as np


def transform(array: np.ndarray, index: int) -> np.ndarray:
    """Eight unique square symmetries on HWC arrays."""
    if not 0 <= index < 8:
        raise ValueError("D4 index must be in [0, 7].")
    result = np.rot90(array, index % 4, axes=(0, 1))
    return np.flip(result, axis=1) if index >= 4 else result


def inverse_transform(array: np.ndarray, index: int) -> np.ndarray:
    result = np.flip(array, axis=1) if index >= 4 else array
    return np.rot90(result, -(index % 4), axes=(0, 1))


def synthesize(clean: np.ndarray, fit: dict, rng: np.random.Generator) -> np.ndarray:
    """Shot/read approximation plus sparse random impulses, sampled on CPU.

    The intensity-dependent Gaussian variance approximates Poisson-Gaussian
    sensor noise in quantized sRGB. Parameters are fitted on training IDs only.
    This is an augmentation model, not a claim to recover the organizers' ISP.
    """
    sigma = float(rng.choice(fit["training_sigma"]))
    sigma *= float(rng.uniform(0.85, 1.15))
    scale = sigma / fit["median_sigma"]
    a, b = np.asarray(fit["a"]), np.asarray(fit["b"])
    variance = np.maximum((a * clean + b) * scale**2, 1e-8)
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * np.sqrt(variance)
    impulses = rng.random(clean.shape[:2]) < rng.uniform(0.0003, 0.001)
    if np.any(impulses):
        noisy[impulses] = rng.integers(0, 2, size=(int(impulses.sum()), 1))
    return np.round(np.clip(noisy, 0, 1) * 255).astype(np.float32) / 255
