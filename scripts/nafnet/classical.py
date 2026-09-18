"""Image-adaptive signal processing; no image IDs or stored defect locations."""

import numpy as np
from scipy.ndimage import median_filter


def estimate_noise(image: np.ndarray) -> np.ndarray:
    """Per-channel Haar-HH MAD in [0, 1] units, using only the noisy image."""
    height, width = image.shape[:2]
    if min(height, width) < 2:
        return np.full(image.shape[-1], 1 / 255, dtype=np.float32)
    even = image[:height // 2 * 2, :width // 2 * 2]
    hh = (even[::2, ::2] - even[::2, 1::2]
          - even[1::2, ::2] + even[1::2, 1::2]) * 0.5
    center = np.median(hh, axis=(0, 1))
    return np.maximum(np.median(np.abs(hh - center), axis=(0, 1)) / 0.67448975,
                      1 / 255).astype(np.float32)


def repair_impulses(image: np.ndarray, threshold: float = 4.5) -> tuple[np.ndarray, dict]:
    """Reject strong isolated outliers using noise MAD and local rank statistics.

    A 3x3 median supplies the replacement when it agrees with the 5x5 median;
    otherwise the larger support is used. A local MAD gate protects edges.
    Neither clean targets nor a fixed sensor-defect map are used at inference.
    """
    if threshold <= 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB HWC data and a positive impulse threshold.")
    sigma = estimate_noise(image)
    median3 = median_filter(image, size=(3, 3, 1), mode="reflect")
    local_mad = median_filter(np.abs(image - median3), size=(3, 3, 1), mode="reflect")
    gate = np.maximum(threshold * sigma, np.maximum(4 * 1.4826 * local_mad, 0.08))
    mask = np.abs(image - median3) > gate
    result = image.copy()
    if np.any(mask):
        median5 = median_filter(image, size=(5, 5, 1), mode="reflect")
        replacement = np.where(np.abs(median3 - median5) < 2 * sigma, median3, median5)
        result[mask] = replacement[mask]
    return result, {"sigma_rgb": sigma.tolist(), "repaired_fraction": float(mask.mean())}


def wavelet_denoise(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Blind BayesShrink reference; balanced RGB noise, no exposure adjustment."""
    from skimage.restoration import denoise_wavelet

    if strength <= 0:
        raise ValueError("Wavelet strength must be positive.")
    sigma = estimate_noise(image) * strength
    return np.clip(denoise_wavelet(
        image, sigma=sigma.tolist(), wavelet="db2", method="BayesShrink",
        mode="soft", channel_axis=-1, convert2ycbcr=False, rescale_sigma=True,
    ), 0, 1).astype(np.float32)


def refine_chroma(image: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Optional mild wavelet refinement of Cb/Cr; retain the original luminance."""
    if not 0 <= amount <= 1:
        raise ValueError("Chroma refinement must be between zero and one.")
    if amount == 0:
        return image
    from skimage.restoration import denoise_wavelet

    red, green, blue = np.moveaxis(image, -1, 0)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = (blue - y) / 1.772
    cr = (red - y) / 1.402
    chroma = np.stack((cb, cr), axis=-1)
    # Explicit finite sigma also handles perfectly flat chroma; the library's
    # automatic estimator can return NaN when every detail coefficient is zero.
    filtered = denoise_wavelet(chroma, sigma=estimate_noise(chroma).tolist(),
                              wavelet="db2", method="BayesShrink",
                              channel_axis=-1, rescale_sigma=True)
    cb, cr = np.moveaxis(chroma + amount * (filtered - chroma), -1, 0)
    red = y + 1.402 * cr
    blue = y + 1.772 * cb
    green = (y - 0.299 * red - 0.114 * blue) / 0.587
    return np.clip(np.stack((red, green, blue), axis=-1), 0, 1).astype(np.float32)
