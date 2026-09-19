# Empirical Noise Characterization -- Mora SP Cup 2026 Low-Light sRGB Denoising

**Date:** 2026-09-18 | **Env:** conda `spcup26` (python 3.11.16, numpy 1.26.4, scikit-image 0.26.0, scipy 1.17.1, opencv 4.10.0, Pillow 10.4.0) | **Seed:** 0
**Data:** 460 paired public images, `competition_data/public/{ground_truth/NNN.png, noisy/NNN_noise.png}`, 992x992 RGB 8-bit (read-only).
**Reproduce:** from repo root, `python scripts/analysis/noise_characterization/characterize.py`. Regenerates all figures and `metrics.json`.

## Package deviations (noted per instructions)
- Installed `matplotlib==3.11.2` (allowed by task) -- figures only.
- Installed `PyWavelets==1.8.0` into `spcup26`. It was NOT pre-installed and is a hard dependency of `skimage.restoration.estimate_sigma` (the sanctioned sigma estimator). No other packages changed/upgraded.

## Methodology & conventions
- **Residual** `R = noisy_uint8 - clean_uint8`, a float in [-255, 255] -- **8-bit units**. To convert any sigma/slope to normalized [0,1] units, divide sigma by 255 (variance by 255^2, intensity-slope a by 255). This is the domain the graded `denoise.py` operates in.
- **Sampling.** Task 3 uses all 460 images. Heavier per-pixel tasks (1,2,4,6) use a stratified sample of 60 images spanning the full severity range (sorted by blind sigma, 60 evenly-spaced picks). Task 5 (defects) uses a random sample of 200 images (seed 0). Sample id lists are in `metrics.json`.
- **Robust sigma** = MAD-based, 1.4826 * median(|x - median(x)|).

---

## 1. Residual distribution per channel  (`fig1_residual_histograms.png`)

Aggregated over 60 images, 40k random pixels each (2.4M samples/channel).

| Channel | mean | std (8-bit) | skew | excess kurtosis | 1%-99% | frac >3sigma |
|---|---|---|---|---|---|---|
| R | 1.39 | 29.33 | 0.031 | **1.86** | [-76, 77] | 0.95% |
| G | 1.49 | 28.19 | 0.018 | **2.04** | [-73, 73] | 0.94% |
| B | 1.84 | 27.96 | 0.085 | **2.36** | [-72, 75] | 1.00% |

**Finding:** residual is **symmetric (skew ~ 0) but distinctly heavy-tailed/leptokurtic** -- excess kurtosis ~1.9-2.4 (Gaussian=0). On log-y the tails sit clearly above the matched Gaussian beyond +-40. Bulk is Gaussian-like but a real heavy-tail/impulse component exists (~1% of pixels beyond 3sigma vs 0.27% for a Gaussian). Kurtosis rises R->G->B. Small positive mean (1.4-1.8) is a slight black-level offset, negligible vs sigma~28 -- this is denoising, not brightening.

## 2. Signal-dependent (Poisson-like) noise  (`fig2_variance_vs_intensity.png`)

Binned residual variance vs clean intensity, weighted fit var(R) = a*I + b (I in 8-bit units).

| Channel | a (slope 8-bit) | b (intercept 8-bit^2) | read sigma=sqrt(b) | a (norm) | b (norm) |
|---|---|---|---|---|---|
| R | 4.36 | 413.7 | 20.3 | 0.0171 | 0.00636 |
| G | 4.22 | 369.6 | 19.2 | 0.0165 | 0.00568 |
| B | 4.86 | 357.1 | 18.9 | 0.0190 | 0.00549 |

**Finding:** noise is **signal-dependent (Poisson+Gaussian / heteroscedastic)**. a>0 for all channels => variance grows with intensity (shot noise through ISP). There is also a large **read-noise floor** b (sqrt(b)~19-20 levels) dominating the dark night-scene content. **Do NOT assume pure AWGN** -- a single global sigma is a poor model; variance ~doubles from black to mid-tones. B has the steepest slope.

## 3. Per-image noise sigma distribution  (`fig3_sigma_distribution.png`)

Blind `estimate_sigma` (wavelet MAD) on each of the 460 noisy images (as at test time), cross-checked vs true residual std and residual MAD.

| Estimator | min | p05 | median | mean | p95 | max |
|---|---|---|---|---|---|---|
| Blind sigma (noisy) | 5.9 | 10.5 | **24.9** | 24.4 | 35.7 | 42.8 |
| True residual std | 7.8 | 12.2 | 27.8 | 27.1 | 38.2 | 43.0 |
| Residual MAD | 5.9 | 10.4 | 25.2 | 25.0 | 37.1 | 44.5 |

**Finding:** severity is a **wide continuum, NOT discrete levels.** sigma ~ 6 -> 43 (8-bit) = normalized ~0.023 -> 0.169, median ~25 (~0.098), distribution std ~7.5. Single broad, mildly right-skewed hump -- no clustering. Blind estimator tracks true residual std well (slight ~10% under-estimate, expected for MAD vs heavy tail), so it is usable as a blind/adaptive front-end. **Synthesize sigma continuously over ~6-43 8-bit (0.02-0.17 norm), weighted toward the 15-35 body.**

## 4. Spatial autocorrelation  (`fig4_autocorrelation.png`)

FFT 2D autocorrelation on 379 flat 64x64 patches, normalized so lag-0 = 1.

| Channel | ACF lag1 (h/v) | lag2 (h/v) | lag3 (h/v) |
|---|---|---|---|
| R | 0.027 / 0.027 | 0.025 / 0.026 | 0.024 / 0.026 |
| G | 0.030 / 0.032 | 0.029 / 0.032 | 0.030 / 0.029 |
| B | 0.029 / 0.030 | 0.027 / 0.029 | 0.026 / 0.025 |

**Finding:** noise is **essentially white / very weakly correlated.** ACF drops 1.0 -> ~0.03 at lag 1 and stays flat (~0.025-0.03) through lag 3 -- no decaying structure, isotropic (h~v). The small ~0.03 floor is mild residual ISP/demosaicing correlation. Classical AWGN denoisers are NOT badly broken by correlation, and a synthetic white-noise generator (with the sec.2 signal-dependent variance) is a faithful match.

## 5. Defect / hot-pixel analysis  (`fig5_defects.png`)

Impulse = |R - median| > 6*(robust sigma) per channel per image. Over 200 images, tallied per pixel location.

- **Impulse fraction per image:** mean **0.033%**, median 0.030%, p95 0.081%, max 0.159%.
- **Fixed-location check (key result):** per-pixel outlier-rate distribution is **sharply bimodal with a total gap between 0.02 and 0.50**:
  - Random band: ~37,620 pixels fire in <=2% of images (per-image random impulses).
  - **Defect band: exactly 492 pixels fire in 50-70% of images**, ZERO pixels in between.
  - **492 fixed hot/defect pixels = 0.050% of the frame.** Under a random model at the mean rate (~0.00050), the chance of any pixel firing in >50% of 200 frames is astronomically small => these are unambiguously **fixed-location sensor defects (hot/dead pixels)**, scattered across the frame (`_per_pixel_outlier_rate.npy`).

**Finding:** two components -- (a) a fixed set of ~492 hot pixels present in most images, and (b) sparse random per-image impulses (~0.03%). A **median / adaptive-median impulse pre-pass** removes both; the fixed set can additionally be repaired by a precomputed defect map.

## 6. Luma vs chroma noise (YCbCr, BT.601 full-range)  (`fig6_luma_chroma.png`)

Robust sigma of the YCbCr residual per image, over 60 images.

| Channel | median sigma (8-bit) | mean sigma | ratio to Y |
|---|---|---|---|
| Y (luma) | **17.98** | 17.30 | 1.00 |
| Cb (chroma) | 15.14 | 14.78 | **0.84** |
| Cr (chroma) | 16.49 | 16.10 | **0.92** |

**Finding (surprise):** chroma sigma is **NOT higher than luma** here; Cb/Y=0.84, Cr/Y=0.92. The 'low-light => dominant chroma noise' rule applies to raw/linear data; this dataset is **already ISP-processed 8-bit sRGB** with chroma denoising/subsampling applied, so chroma sigma is slightly below luma. **Aggressive chroma smoothing is NOT needed** and would risk color bleeding for little gain. Noise energy is fairly balanced across Y/Cb/Cr.

---

## Implications for implementation

### (a) Classical pipeline parameters
- **Signal-dependent, not pure AWGN.** Use a variance-stabilizing transform (generalized Anscombe with a~4.4, b~380 8-bit, per-channel) OR a variance-adaptive/local-sigma denoiser rather than one global sigma. The read floor sqrt(b)~19-20 dominates dark content, so even flat dark areas need real denoising.
- **Impulse/hot-pixel pre-pass: YES.** Adaptive median (or 3x3 median gated on |R|>k*sigma) before the main denoiser kills the ~492 fixed hot pixels + ~0.03% random impulses. This is the 'classical SP beyond baseline' story and protects the Gaussian-oriented main stage from the heavy tail (sec.1).
- **Chroma smoothing: mild only.** Chroma sigma <= luma sigma (sec.6). Luma-matched or slightly lighter chroma filtering; no large chroma prize.
- **Spatial handling:** noise ~white (sec.4) -> standard AWGN denoisers (BM3D/NLM/wavelet) are appropriate; no correlated-noise variants needed. Set strength per image from the blind sigma estimate (median ~25, range 6-43); blind/adaptive strength is essential given the continuum.
- **Do not brighten** -- residual mean ~+1.5/255, negligible; pure denoising.

### (b) Synthetic-noise generator spec (deep model training)
1. **Noise model: Poisson-Gaussian (heteroscedastic)**, var(pixel) = a*I + b, per-channel 8-bit a~{R:4.36,G:4.22,B:4.86}, b~{R:414,G:370,B:357} (norm a~0.017, b~0.0057). Sample shot term ~sqrt(a*I) plus read term ~N(0,sqrt(b)). Reproduces measured variance-vs-intensity, not flat AWGN.
2. **Severity sampling (blind):** draw severity so per-image sigma spans the real continuum ~6-43 8-bit (0.023-0.169 norm), sampled continuously (uniform-in-sigma or log-normal centered ~25, std ~7.5), NOT discrete levels. Enables generalizing to unseen severities in 481-500.
3. **Heavy tail / impulse:** add ~0.03% random impulse pixels per image (matching sec.5 / the ~1%-beyond-3sigma tail), and optionally a fixed hot-pixel map of ~0.05% of pixels across all synthetic frames (or rely on the classical pre-pass and train on de-impulsed input).
4. **Spatial correlation:** white noise is sufficient (ACF ~0.03 at lag 1). Optional very mild (sigma~0.3px) blur of the noise field to reproduce the ~0.03 floor; second-order.
5. **Per-channel/chroma:** apply per-channel a,b in RGB. Do NOT inflate chroma noise -- measured chroma sigma <= luma; RGB Poisson-Gaussian already gives the correct balanced YCbCr split.
6. **Best practice:** train on the real 460 pairs as anchor, augment with this synthetic generator for unlimited severity coverage; always validate on real noisy images to avoid overfitting the noise model.

## Output files
- `characterize.py` -- reproducible script (seed 0).
- `metrics.json` -- all numbers (machine-readable) + sample id lists.
- `fig1..fig6_*.png` -- the six figures.
- `_sigma_blind.npy`, `_sigma_resid_std.npy`, `_per_pixel_outlier_rate.npy` -- cached arrays.
