# 02 — Problem & Metric Analysis

This is the strategic heart of the project. If you read one doc before building, read this.

## 1. What kind of problem is this, *precisely*?

- **Blind sRGB denoising**, not raw denoising. We get 8-bit RGB PNGs, not Bayer raw. So most "raw low-light" literature (SID/ELD/Poisson-Gaussian in raw domain) applies only *conceptually*, not directly — our noise has already been pushed through an (unknown) ISP and quantized to 8-bit sRGB.
- **Blind** = the noise **severity varies across the set** and we are never told the level per image. The model must self-adapt (either an intrinsically blind network, or an explicit noise-estimation front-end à la CBDNet).
- **Denoising, not exposure enhancement.** The baseline author states it explicitly: *"input and ground truth are both naturally dim evening/night images, differing mainly by noise, so there is no exposure gap to fix."* The metric confirms it — it compares denoised vs. GT at the *same* brightness. **Do not add global brightening/gamma/Retinex tone-mapping**; it would move the output away from GT and *lower* both PSNR and SSIM. (This is the single most common way teams will hurt themselves given the "Enhancement" in the title.)
- **Artifacts present** (per the handbook): read noise, thermal noise, grain, **defect/hot pixels**, and "visual distortion." The defect pixels are why the baseline has an explicit outlier-repair stage — impulse-like outliers are not handled well by Gaussian-oriented denoisers, so an **impulse/hot-pixel pre-pass (e.g., median / adaptive median)** is genuinely useful and doubles as our "classical SP beyond baseline" story.

> **Action item once data is downloaded:** empirically characterize the noise (see §6). Do *not* assume pure AWGN — verify whether it is signal-dependent (Poisson-like), whether there are impulse outliers, chroma vs. luma noise balance, and whether severity clusters into discrete levels.

## 2. The exact scoring formula and what it rewards

```
ΔPSNR = PSNR(denoised, GT) − PSNR(noisy, GT)
ΔSSIM = SSIM(denoised, GT) − SSIM(noisy, GT)
N     = clip(ΔPSNR / 15, 0, 1)
S     = max(ΔSSIM, 0)
Composite = 0.6·N + 0.4·S
```
Verified against `evaluation/evaluate.py` (skimage `peak_signal_noise_ratio`, `data_range=1.0`; `structural_similarity` with `channel_axis=-1, win_size=7, gaussian_weights=False, use_sample_covariance=True, K1=0.01, K2=0.03`). **We must validate with these exact SSIM settings** — SSIM is very sensitive to windowing/gaussian choices, and a mismatch would make our local numbers lie.

### Marginal value of effort (the key insight)
- **PSNR term** is linear until it saturates at ΔPSNR = 15 dB. Each **+1 dB** of ΔPSNR (while below 15) adds **0.6/15 = 0.040** to the composite.
- **SSIM term** each **+0.01** of ΔSSIM adds **0.4·0.01 = 0.004** to the composite.
- ⇒ **Per natural unit, PSNR is ~10× more valuable than SSIM.** But SSIM improvements come "for free" with good denoising and there's no clip until you exceed the (large) headroom, so **maximize PSNR first, but don't sacrifice structure**.
- **Saturation matters:** once denoised PSNR ≥ ~34.8 dB (ΔPSNR ≥ 15), extra PSNR is worthless. Pushing SSIM then becomes the only lever. Most realistic solutions will sit *below* saturation, so PSNR remains the primary target.

## 3. Reference points (baseline-derived)

| Quantity | Value | Source |
|---|---|---|
| Noisy input PSNR | **≈ 19.83 dB** | 24.336 − 4.510 |
| Noisy input SSIM | **≈ 0.464** | 0.6717 − 0.2073 |
| Baseline denoised PSNR | 24.336 dB | baseline README |
| Baseline denoised SSIM | 0.6717 | baseline README |
| Baseline ΔPSNR / ΔSSIM | +4.510 / +0.2073 | baseline README |
| Baseline N / S | 0.3007 / 0.2073 | computed |
| **Baseline composite** | **0.2633** | matches PDF "≈0.26" |

## 4. Score projections (targets)

Assuming noisy reference ≈ 19.83 dB / 0.464 SSIM stays roughly constant across splits:

| Scenario | Denoised PSNR | Denoised SSIM | ΔPSNR | N | ΔSSIM→S | **Composite** |
|---|---|---|---|---|---|---|
| Noisy (do nothing) | 19.83 | 0.464 | 0 | 0 | 0 | 0.000 |
| **Baseline (NLM)** | 24.34 | 0.672 | 4.51 | 0.301 | 0.207 | **0.263** |
| Strong classical (BM3D/wavelet, tuned) | ~26–27 | ~0.74 | ~6.5 | 0.43 | 0.28 | **~0.37** |
| Good deep net (blind, moderate) | ~30 | ~0.85 | ~10.2 | 0.68 | 0.39 | **~0.56** |
| SOTA deep (Restormer/NAFNet-class, well-trained) | ~33–35 | ~0.90 | ~13–15 | 0.87–1.0 | ~0.44 | **~0.70–0.78** |
| Theoretical max | ≥34.8 | →1.0 | ≥15 | 1.0 | 0.536 | **0.814** |

**Takeaways**
- Beating baseline (0.263) is *easy* — even a well-tuned classical pipeline gets there.
- The real prize is the **0.55–0.75 band**, which **requires a trained deep model**. Classical-only tops out around ~0.35–0.40 on this kind of data.
- Target: **≥0.60 composite** on our held-out validation, aiming for the PSNR term to approach saturation.

## 5. Constraints that shape the architecture

1. **CPU fallback + reasonable runtime on mid-range CPU** (part of the 15% code score, and required for the *live final round* which may not have a strong GPU). ⇒ favor an **efficient** network (tile-able, modest params) over a giant transformer. NAFNet-class and lightweight Restormer variants are attractive; consider tiling the 992×992 image and FP32/FP16, and offering `--tile` and `--device` flags.
2. **992×992 inference** — too big for one forward pass on modest memory. Plan **overlap-tiled inference** (e.g., 256/512 tiles with 16–32 px overlap, feathered blending) to avoid seams. This is mandatory engineering, not optional.
3. **Offline, self-contained** — bundle weights (or Drive + SHA-256). No downloads at inference.
4. **Reproducibility** — outputs in `TeamName.zip` must be *bit-for-bit reproducible* from the frozen code+model (fixed seeds, deterministic ops, no test-time randomness unless seeded). Otherwise DQ.
5. **No hardcoding / per-image lookups.** The pipeline must be a single generalizable procedure.
6. **10% classical-SP credit** — we must include *significant* classical DSP beyond the baseline's NLM, and justify it. Good candidates that also help the score: adaptive-median hot-pixel removal, **wavelet/BM3D** stage, guided/bilateral edge-preserving refinement, luma/chroma (YCbCr) separation, self-ensemble (geometric TTA — flips/rotations averaged, a classic SP trick worth ~0.1–0.3 dB).

## 6. Empirical noise characterization plan (do this first, once data lands)

Before choosing hyperparameters, measure the actual degradation on the 460 public pairs:
1. **Residual = noisy − clean** per image. Inspect histograms per channel. Gaussian? Heavy-tailed (impulse)? Signal-dependent (variance vs. intensity)?
2. **Per-image noise σ estimate** (e.g., MAD of wavelet HH sub-band, or robust std of flat patches). Plot the distribution → confirm "different severity levels" and whether they cluster (this tells us the σ range to synthesize during training).
3. **Spatial correlation** of the residual (autocorrelation) — is the noise white or spatially correlated (demosaicing/ISP-correlated)? Correlated noise breaks classical AWGN denoisers and favors learned models trained on matching synthetic noise.
4. **Defect-pixel fraction** — how many pixels are impulse outliers, and are they fixed-location (hot pixels) or random?
5. **Luma vs. chroma** split in YCbCr — low light usually has stronger chroma noise, which argues for stronger chroma smoothing.

The output of this study directly parameterizes both the classical pipeline and, crucially, the **synthetic-noise generator** used to train the deep model (see [doc 04](04_solution_strategy.md)). Because we have 460 *paired* images, we can also train **directly on the real pairs** — but modeling the noise well lets us augment with unlimited synthetic data and generalize to the unseen severities in `481–500`.

## 7. Validation strategy (must be stated in the report)

- **Split** the 460 public pairs into train / val (e.g., 410 / 50, or 5-fold), stratified by estimated noise severity so val covers all levels.
- **Score with the official metric** via `evaluation/evaluate.py` on the held-out val set — never tune on train.
- Track **PSNR, SSIM, and composite** (not just loss). Watch for the PSNR/SSIM trade-off.
- **Never touch** `461–480` GT (we don't have it) or `481–500`. Our val composite is our proxy for the hidden prelim score.
- Guard against **overfitting to synthetic noise**: always validate against the *real* noisy public images, not just synthetic ones.
