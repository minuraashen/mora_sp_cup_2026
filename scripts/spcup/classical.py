"""
classical.py -- the classical signal-processing stages.

Two things live here:

  1. ``repair_impulses`` -- a noise-model-gated adaptive median. This is the classical
     contribution claimed in the report, and it is specifically NOT the baseline's
     filter. The baseline gates on a FIXED absolute threshold |x - med3| > 0.25
     (= 64/255). Across this dataset the per-image sigma runs from 6 to 43 (8-bit), so
     64 levels is ~10 sigma on the cleanest images (the filter never fires) but only
     ~1.5 sigma on the noisiest (it fires on ordinary noise and smears texture). The
     correct threshold is not a constant: the measured noise is heteroscedastic,
     var(I) = a*I + b, so the gate must scale with sqrt(a*I + b) -- per pixel, per
     channel. That is what this does, and it is why it helps where the baseline's hurts.

  2. ``classical_denoise`` -- a complete, self-contained fallback denoiser
     (generalised Anscombe VST -> wavelet BayesShrink -> exact unbiased inverse).
     It never runs in the normal path; it exists so that a missing checkpoint or an
     unexpected exception degrades the score instead of zeroing it.

Both are pure numpy/torch on the hot path. ``pywt`` is imported lazily inside the
fallback so that a missing optional dependency cannot break the graded path.
"""

from __future__ import annotations

import numpy as np

# Measured per-channel noise model, 8-bit units (analysis/noise_characterization).
A_COEF = np.array([4.35921, 4.21573, 4.85656], dtype=np.float32)
B_COEF = np.array([413.721, 369.608, 357.141], dtype=np.float32)

DEFAULT_GATE_K = 4.5
DEFAULT_MAX_WINDOW = 7


def sigma_of_intensity(img01: np.ndarray) -> np.ndarray:
    """Per-pixel, per-channel noise sigma in [0,1] units from the measured model."""
    I255 = np.clip(img01, 0.0, 1.0) * 255.0
    var = A_COEF[None, None, :] * I255 + B_COEF[None, None, :]
    return np.sqrt(var) / 255.0


def estimate_sigma_blind(img01: np.ndarray) -> float:
    """Immerkaer 3x3-Laplacian noise estimate, in [0,1] units.

    O(N), no wavelet dependency, fully deterministic. Used as a robustness fallback
    when an image is far off the measured model, and reported as a diagnostic.
    """
    from scipy.signal import convolve2d
    K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    vals = []
    for c in range(img01.shape[2]):
        conv = convolve2d(img01[..., c].astype(np.float64), K, mode="valid")
        # Immerkaer: sigma = sqrt(pi/2) / (6 * N) * sum|I * K|
        vals.append(np.sqrt(np.pi / 2.0) / (6.0 * conv.size) * np.abs(conv).sum())
    return float(np.mean(vals))


def _median3(img01: np.ndarray) -> np.ndarray:
    """3x3 median per channel, edge-replicated, without an OpenCV dependency.

    np.partition at the middle rank is used instead of np.median: for 9 elements it
    does the same work but skips median's generality, and this runs on every pixel of
    every image, so the constant matters.
    """
    p = np.pad(img01, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = img01.shape[:2]
    stack = np.stack([p[dy:dy + h, dx:dx + w, :] for dy in range(3) for dx in range(3)], axis=0)
    return np.partition(stack, 4, axis=0)[4]


def _median_at(img01: np.ndarray, k: int, ys: np.ndarray, xs: np.ndarray,
               cs: np.ndarray) -> np.ndarray:
    """Median of a k x k neighbourhood, evaluated ONLY at the given pixel locations.

    The adaptive median only ever needs the larger windows at pixels that were already
    flagged, which is ~0.05% of the frame. Computing a full-image 7x7 median to use a
    few thousand of its values would build a 49-plane stack over 992x992x3 -- about two
    orders of magnitude more work than the result requires.
    """
    r = k // 2
    p = np.pad(img01, ((r, r), (r, r), (0, 0)), mode="edge")
    # +r maps a source coordinate into the padded array; the window is then [y, y+k)
    yy = ys[:, None] + np.arange(k)[None, :]
    xx = xs[:, None] + np.arange(k)[None, :]
    patch = p[yy[:, :, None], xx[:, None, :], cs[:, None, None]]
    return np.median(patch.reshape(len(ys), -1), axis=1)


def repair_impulses(img01: np.ndarray, k: float = DEFAULT_GATE_K,
                    max_window: int = DEFAULT_MAX_WINDOW) -> tuple[np.ndarray, float]:
    """Noise-model-gated adaptive median.

    A pixel is flagged only when it deviates from its local median by more than
    k * sigma(I) for the measured heteroscedastic model. Flagged pixels are replaced
    by the median of a window that grows 3 -> 5 -> 7 until that median is itself not
    flagged; unflagged pixels are passed through completely untouched, so ordinary
    texture is never filtered.

    Returns (repaired, flagged_fraction).
    """
    out = img01.copy()
    med = _median3(img01)
    sigma = sigma_of_intensity(med)
    thresh = k * sigma
    flagged = np.abs(img01 - med) > thresh

    frac = float(flagged.mean())
    if frac == 0.0:
        return out, 0.0

    out[flagged] = med[flagged]

    ys, xs, cs = np.nonzero(flagged)
    w = 5
    while w <= max_window and len(ys):
        m_w = _median_at(img01, w, ys, xs, cs)
        base = med[ys, xs, cs]
        thr = thresh[ys, xs, cs]
        still = np.abs(m_w - base) > thr
        out[ys, xs, cs] = m_w
        ys, xs, cs = ys[still], xs[still], cs[still]
        w += 2

    return out, frac


# ---------------------------------------------------------------- VST fallback

def _gat(x255: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generalised Anscombe transform: maps var = a*I + b to approximately unit variance."""
    inner = np.maximum(x255 * a + 0.375 * a * a + b, 0.0)
    return (2.0 / a) * np.sqrt(inner)


def _inv_gat(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closed-form unbiased inverse of the generalised Anscombe transform."""
    yy = np.maximum(y, 0.0)
    z = (yy * a / 2.0) ** 2
    return (z - 0.125 - b) / a


# ------------------------------------------------- ablation-only transforms
# These exist to be MEASURED and rejected, not to be shipped. Each corresponds to a
# technique the low-light denoising literature would suggest by default and that the
# measurements on this dataset do not support; see the report's alternatives section.

_RGB2YCC = np.array([[0.299, 0.587, 0.114],
                     [-0.168736, -0.331264, 0.5],
                     [0.5, -0.418688, -0.081312]], dtype=np.float32)
_YCC2RGB = np.linalg.inv(_RGB2YCC).astype(np.float32)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur via cumulative sums, edge-replicated."""
    p = np.pad(x, ((r, r), (r, r)), mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    k = 2 * r + 1
    h, w = x.shape
    return (c[k:k + h, k:k + w] - c[0:h, k:k + w]
            - c[k:k + h, 0:w] + c[0:h, 0:w]) / (k * k)


def chroma_smooth(img01: np.ndarray, radius: int = 3) -> np.ndarray:
    """Smooth only Cb/Cr. ABLATION ONLY -- measured chroma sigma is BELOW luma here."""
    ycc = img01.reshape(-1, 3) @ _RGB2YCC.T
    ycc = ycc.reshape(img01.shape)
    ycc[..., 1] = _box(ycc[..., 1] + 0.5, radius) - 0.5
    ycc[..., 2] = _box(ycc[..., 2] + 0.5, radius) - 0.5
    out = ycc.reshape(-1, 3) @ _YCC2RGB.T
    return np.clip(out.reshape(img01.shape), 0.0, 1.0)


def apply_gamma(img01: np.ndarray, gamma: float = 1.05) -> np.ndarray:
    """Global tone adjustment. ABLATION ONLY -- there is no exposure gap to correct."""
    return np.clip(img01, 0.0, 1.0) ** (1.0 / gamma)


def gat_forward(img01: np.ndarray) -> np.ndarray:
    """Variance-stabilise into a [0,1]-ish range. ABLATION ONLY (train/test mismatch)."""
    x255 = np.clip(img01, 0.0, 1.0) * 255.0
    out = np.empty_like(x255)
    for c in range(3):
        y = _gat(x255[..., c], np.float32(A_COEF[c]), np.float32(B_COEF[c]))
        out[..., c] = y
    return out / max(out.max(), 1e-6), float(max(out.max(), 1e-6))


def gat_inverse(y01: np.ndarray, scale: float) -> np.ndarray:
    x = np.empty_like(y01)
    for c in range(3):
        x[..., c] = _inv_gat(y01[..., c] * scale, np.float32(A_COEF[c]), np.float32(B_COEF[c]))
    return np.clip(x / 255.0, 0.0, 1.0)


def classical_denoise(img01: np.ndarray, method: str = "wavelet") -> np.ndarray:
    """The offline, no-model fallback denoiser.

    Two variants were implemented and measured on held-out images:

      'wavelet' (default) -- estimate the image's noise level blind, then apply
          BayesShrink soft thresholding in a luma/chroma decorrelated basis.
      'gat' -- generalised Anscombe VST -> BayesShrink at unit sigma -> exact
          unbiased inverse. Theoretically the better-motivated option, since the
          measured noise is heteroscedastic (var = a*I + b) and a VST is precisely
          the classical remedy for that.

    Measured on validation images, 'gat' scored 0.2204 against 0.3104 for 'wavelet',
    so the theoretically-preferred transform is NOT what ships. The likely reason is
    that BayesShrink adapts its threshold per sub-band from the data, so it already
    absorbs much of the variance non-stationarity, while the VST costs an extra
    forward/inverse rounding and mis-matches BayesShrink's noise assumption. This is
    reported as a measured result rather than argued from first principles.
    """
    try:
        from skimage.restoration import denoise_wavelet, estimate_sigma
    except ImportError:
        # skimage is an optional dependency; never let its absence crash the run.
        return _median_k_full(img01, 3)

    if method == "gat":
        x255 = np.clip(img01, 0.0, 1.0) * 255.0
        out = np.empty_like(x255)
        for c in range(3):
            a, b = np.float32(A_COEF[c]), np.float32(B_COEF[c])
            y = _gat(x255[..., c], a, b)
            scale = max(float(y.max()), 1e-6)
            den = denoise_wavelet(y / scale, sigma=1.0 / scale, mode="soft",
                                  method="BayesShrink", rescale_sigma=True) * scale
            out[..., c] = _inv_gat(den, a, b)
        return np.clip(out / 255.0, 0.0, 1.0)

    sigma = float(np.mean(estimate_sigma(img01, channel_axis=-1)))
    out = denoise_wavelet(img01, sigma=sigma, channel_axis=-1, convert2ycbcr=True,
                          method="BayesShrink", mode="soft", rescale_sigma=True)
    return np.clip(out, 0.0, 1.0)


def _median_k_full(img01: np.ndarray, k: int) -> np.ndarray:
    """Whole-image k x k median. Only used as a last-resort fallback."""
    r = k // 2
    p = np.pad(img01, ((r, r), (r, r), (0, 0)), mode="edge")
    h, w = img01.shape[:2]
    stack = np.stack([p[dy:dy + h, dx:dx + w, :] for dy in range(k) for dx in range(k)], axis=0)
    return np.median(stack, axis=0)
