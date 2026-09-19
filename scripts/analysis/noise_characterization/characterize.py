#!/usr/bin/env python
"""
Empirical noise characterization for the Mora SP Cup 2026 low-light sRGB
denoising competition.

Analyzes the residual R = noisy - clean over the 460 paired public images and
produces per-channel distribution stats, a signal-dependent-noise (Poisson+read)
fit, a per-image sigma distribution, spatial autocorrelation, hot/defect-pixel
analysis, and a luma-vs-chroma (YCbCr) sigma split.

Run (from repo root, env spcup26):
    python analysis/noise_characterization/characterize.py

All outputs (figures, metrics.json) are written next to this script.
Read-only on competition_data/**. Fixed seed = 0 for all sampling.
"""
import os
import json
import numpy as np
from PIL import Image
from scipy import stats as sstats
from skimage.restoration import estimate_sigma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)

ROOT = "C:/Users/ADMIN/Desktop/mora_sp_cup_2026"
GT_DIR = os.path.join(ROOT, "competition_data/public/ground_truth")
NOISY_DIR = os.path.join(ROOT, "competition_data/public/noisy")
OUT = os.path.join(ROOT, "analysis/noise_characterization")
os.makedirs(OUT, exist_ok=True)

N_IMAGES = 460
IDS = list(range(1, N_IMAGES + 1))
CH = ["R", "G", "B"]


def load_pair(i):
    """Return (clean_uint8, noisy_uint8) HxWx3 for image id i."""
    c = np.asarray(Image.open(os.path.join(GT_DIR, f"{i:03d}.png")).convert("RGB"))
    n = np.asarray(Image.open(os.path.join(NOISY_DIR, f"{i:03d}_noise.png")).convert("RGB"))
    return c, n


def residual8(c, n):
    """Residual in 8-bit units (float): noisy - clean, range approx [-255,255]."""
    return n.astype(np.float64) - c.astype(np.float64)


# ---------------------------------------------------------------------------
# Convention note:
#   We report the residual in 8-BIT UNITS (noisy_uint8 - clean_uint8), a float
#   in [-255, 255]. To convert any sigma to normalized [0,1] float units,
#   divide by 255. Both are given in the report where useful.
# ---------------------------------------------------------------------------


def rand_sample(k):
    """Deterministic sample of k image ids spanning the set."""
    return sorted(rng.choice(IDS, size=min(k, len(IDS)), replace=False).tolist())


metrics = {"convention": "residual R = noisy_uint8 - clean_uint8 (8-bit units, [-255,255]); divide by 255 for [0,1] float", "seed": SEED}


# ===========================================================================
# TASK 3 (compute first, needed for stratified sampling everywhere):
#   Per-image noise sigma via skimage.restoration.estimate_sigma (wavelet MAD).
#   Estimated on the NOISY image (blind, as at test time) and also on residual.
# ===========================================================================
def task3_sigma_distribution():
    print("[Task 3] Per-image sigma over all 460 images ...")
    sig_noisy = np.zeros(N_IMAGES)          # blind estimate on noisy (0..255 units)
    sig_resid_std = np.zeros(N_IMAGES)      # robust std of full residual (8-bit)
    sig_resid_mad = np.zeros(N_IMAGES)      # MAD-based robust sigma of residual
    for idx, i in enumerate(IDS):
        c, n = load_pair(i)
        # skimage estimate_sigma expects float; use 0..1 then scale to 8-bit units
        s = estimate_sigma(n.astype(np.float64) / 255.0, channel_axis=-1,
                           average_sigmas=True)
        sig_noisy[idx] = s * 255.0
        r = residual8(c, n)
        sig_resid_std[idx] = r.std()
        med = np.median(r)
        sig_resid_mad[idx] = 1.4826 * np.median(np.abs(r - med))
        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{N_IMAGES}")

    def q(a):
        return dict(min=float(a.min()), p05=float(np.percentile(a, 5)),
                    median=float(np.median(a)), mean=float(a.mean()),
                    p95=float(np.percentile(a, 95)), max=float(a.max()),
                    std=float(a.std()))

    metrics["task3_sigma"] = {
        "sigma_blind_noisy_8bit": q(sig_noisy),
        "sigma_residual_std_8bit": q(sig_resid_std),
        "sigma_residual_mad_8bit": q(sig_resid_mad),
        "note": "blind estimate_sigma on noisy image tracks true residual std; MAD is more robust to impulses",
    }

    # Clustering check: histogram of blind sigma
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(sig_noisy, bins=60, color="#3366cc", alpha=0.85)
    ax[0].set_title("Blind sigma (estimate_sigma on noisy), 8-bit units")
    ax[0].set_xlabel("sigma (8-bit)"); ax[0].set_ylabel("# images")
    ax[0].axvline(np.median(sig_noisy), color="k", ls="--",
                  label=f"median={np.median(sig_noisy):.2f}")
    ax[0].legend()
    ax[1].scatter(sig_resid_std, sig_noisy, s=8, alpha=0.5, color="#cc3333")
    lim = [0, max(sig_resid_std.max(), sig_noisy.max()) * 1.05]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_title("Blind sigma vs true residual std")
    ax[1].set_xlabel("residual std (8-bit)"); ax[1].set_ylabel("blind sigma (8-bit)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_sigma_distribution.png"), dpi=110)
    plt.close(fig)

    # simple clustering diagnostic: kmeans-lite via histogram modality
    # (report the histogram; note continuum vs discrete in REPORT)
    np.save(os.path.join(OUT, "_sigma_blind.npy"), sig_noisy)
    np.save(os.path.join(OUT, "_sigma_resid_std.npy"), sig_resid_std)
    return sig_noisy, sig_resid_std


# ===========================================================================
# TASK 1: per-channel residual distribution + moments
# ===========================================================================
def task1_histograms(sample_ids):
    print(f"[Task 1] Residual histograms/moments over {len(sample_ids)} images ...")
    # accumulate residual samples per channel (subsample pixels to bound memory)
    per_ch = [[] for _ in range(3)]
    px_per_img = 40000
    for i in sample_ids:
        c, n = load_pair(i)
        r = residual8(c, n)
        H, W, _ = r.shape
        flat = r.reshape(-1, 3)
        sel = rng.choice(flat.shape[0], size=px_per_img, replace=False)
        for ch in range(3):
            per_ch[ch].append(flat[sel, ch])
    per_ch = [np.concatenate(x) for x in per_ch]

    stats_out = {}
    for ch in range(3):
        a = per_ch[ch]
        stats_out[CH[ch]] = dict(
            n=int(a.size), mean=float(a.mean()), std=float(a.std()),
            skew=float(sstats.skew(a)), kurtosis_excess=float(sstats.kurtosis(a)),
            p01=float(np.percentile(a, 1)), p99=float(np.percentile(a, 99)),
            frac_abs_gt_3std=float(np.mean(np.abs(a - a.mean()) > 3 * a.std())),
        )
    metrics["task1_residual_moments"] = stats_out

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#cc3333", "#33aa33", "#3366cc"]
    for ch in range(3):
        a = per_ch[ch]
        ax = axes[ch]
        lo, hi = -100, 100
        ax.hist(a, bins=200, range=(lo, hi), density=True, color=colors[ch],
                alpha=0.6, label="residual")
        # overlay matched Gaussian
        xs = np.linspace(lo, hi, 400)
        g = sstats.norm.pdf(xs, a.mean(), a.std())
        ax.plot(xs, g, "k-", lw=1.3, label="Gaussian fit")
        ax.set_yscale("log")
        ax.set_title(f"{CH[ch]}: mu={a.mean():.2f} sd={a.std():.2f} "
                     f"skew={sstats.skew(a):.2f} kurt={sstats.kurtosis(a):.2f}")
        ax.set_xlabel("residual (8-bit)"); ax.set_ylabel("density (log)")
        ax.legend(fontsize=8)
    fig.suptitle("Per-channel residual distribution (log-y; Gaussian overlay)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_residual_histograms.png"), dpi=110)
    plt.close(fig)


# ===========================================================================
# TASK 2: signal-dependent noise: var(R) vs clean intensity, fit var = a*I + b
# ===========================================================================
def task2_signal_dependent(sample_ids):
    print(f"[Task 2] Signal-dependent variance fit over {len(sample_ids)} images ...")
    n_bins = 32
    edges = np.linspace(0, 255, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # accumulate sum, sumsq, count per (bin, channel)
    ssum = np.zeros((n_bins, 3)); ssq = np.zeros((n_bins, 3)); cnt = np.zeros((n_bins, 3))
    for i in sample_ids:
        c, n = load_pair(i)
        r = residual8(c, n)
        for ch in range(3):
            ci = c[..., ch].ravel().astype(np.float64)
            ri = r[..., ch].ravel()
            b = np.clip(np.digitize(ci, edges) - 1, 0, n_bins - 1)
            np.add.at(ssum, (b, ch), ri)
            np.add.at(ssq, (b, ch), ri * ri)
            np.add.at(cnt, (b, ch), 1.0)
    mean_r = ssum / np.maximum(cnt, 1)
    var_r = ssq / np.maximum(cnt, 1) - mean_r ** 2  # per-bin residual variance

    fit = {}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#cc3333", "#33aa33", "#3366cc"]
    for ch in range(3):
        valid = cnt[:, ch] > 5000  # enough pixels for a stable variance
        x = centers[valid]; y = var_r[valid, ch]
        # weighted linear fit var = a*I + b (I in 8-bit units)
        w = cnt[valid, ch]
        A = np.vstack([x, np.ones_like(x)]).T
        Wd = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * Wd[:, None], y * Wd, rcond=None)
        a_coef, b_coef = float(coef[0]), float(coef[1])
        # normalized-units versions (I in [0,1], var in [0,1]^2):
        # var01 = var8/255^2 ; I8 = I01*255 => a01 = a8/255, b01 = b8/255^2
        fit[CH[ch]] = dict(a_slope_8bit=a_coef, b_intercept_8bit=b_coef,
                           a_slope_norm01=a_coef / 255.0,
                           b_intercept_norm01=b_coef / (255.0 ** 2),
                           read_sigma_8bit=float(np.sqrt(max(b_coef, 0))))
        ax.scatter(x, y, s=20, color=colors[ch], alpha=0.7, label=f"{CH[ch]} data")
        ax.plot(centers, a_coef * centers + b_coef, color=colors[ch], lw=1.5,
                label=f"{CH[ch]} fit a={a_coef:.3f} b={b_coef:.1f}")
    ax.set_xlabel("clean intensity I (8-bit)")
    ax.set_ylabel("residual variance (8-bit^2)")
    ax.set_title("Signal-dependent noise: var(R) vs intensity, var = a*I + b")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_variance_vs_intensity.png"), dpi=110)
    plt.close(fig)
    metrics["task2_noise_model"] = {
        "form": "var(R) = a*I + b   (I in 8-bit intensity units)",
        "fit": fit,
        "interpretation": "a>0 => shot-noise-like signal dependence; b => read-noise variance floor",
    }
    return centers, var_r, cnt


# ===========================================================================
# TASK 4: spatial autocorrelation of residual on flat regions
# ===========================================================================
def _flat_patches(c, r, patch=64, grad_thresh=6.0, max_patches=8):
    """Yield residual patches from spatially flat clean regions (low gradient)."""
    H, W, _ = c.shape
    gray = c.mean(axis=2)
    gy, gx = np.gradient(gray)
    gmag = np.hypot(gx, gy)
    coords = []
    step = patch
    for y in range(0, H - patch, step):
        for x in range(0, W - patch, step):
            if gmag[y:y+patch, x:x+patch].mean() < grad_thresh:
                coords.append((y, x))
    rng.shuffle(coords)
    for (y, x) in coords[:max_patches]:
        yield r[y:y+patch, x:x+patch, :]


def task4_autocorr(sample_ids, patch=64, maxlag=6):
    print(f"[Task 4] Spatial autocorrelation over {len(sample_ids)} images ...")
    acc = np.zeros((3, 2 * maxlag + 1, 2 * maxlag + 1))
    nacc = 0
    for i in sample_ids:
        c, n = load_pair(i)
        r = residual8(c, n)
        for p in _flat_patches(c, r, patch=patch):
            for ch in range(3):
                pr = p[..., ch]
                pr = pr - pr.mean()
                var = pr.var()
                if var < 1e-6:
                    continue
                # FFT-based autocorr, take central lags
                F = np.fft.fft2(pr)
                ac = np.fft.ifft2(F * np.conj(F)).real
                ac = np.fft.fftshift(ac) / (pr.size * var)
                cy, cx = pr.shape[0] // 2, pr.shape[1] // 2
                acc[ch] += ac[cy - maxlag:cy + maxlag + 1,
                              cx - maxlag:cx + maxlag + 1]
            nacc += 1
    acc /= max(nacc, 1)

    lags = {}
    for ch in range(3):
        center = acc[ch, maxlag, maxlag]
        lags[CH[ch]] = {
            "acf_lag1_h": float(acc[ch, maxlag, maxlag + 1] / center),
            "acf_lag1_v": float(acc[ch, maxlag + 1, maxlag] / center),
            "acf_lag2_h": float(acc[ch, maxlag, maxlag + 2] / center),
            "acf_lag2_v": float(acc[ch, maxlag + 2, maxlag] / center),
            "acf_lag3_h": float(acc[ch, maxlag, maxlag + 3] / center),
            "acf_lag3_v": float(acc[ch, maxlag + 3, maxlag] / center),
        }
    metrics["task4_autocorrelation"] = {
        "n_patches": int(nacc), "patch_px": patch,
        "acf_normalized_lags": lags,
        "note": "1.0 at lag0 by construction; ~0 at lag>=1 => white noise; large => spatially correlated",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ch in range(3):
        im = axes[ch].imshow(acc[ch] / acc[ch, maxlag, maxlag],
                             extent=[-maxlag, maxlag, maxlag, -maxlag],
                             cmap="viridis", vmin=-0.05, vmax=0.3)
        axes[ch].set_title(f"{CH[ch]} normalized 2D ACF")
        axes[ch].set_xlabel("lag x (px)"); axes[ch].set_ylabel("lag y (px)")
        plt.colorbar(im, ax=axes[ch], fraction=0.046)
    fig.suptitle("Residual spatial autocorrelation on flat patches")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_autocorrelation.png"), dpi=110)
    plt.close(fig)


# ===========================================================================
# TASK 5: defect / hot-pixel analysis
# ===========================================================================
def task5_defects(sample_ids, k=6.0):
    print(f"[Task 5] Impulse/hot-pixel analysis over {len(sample_ids)} images ...")
    # impulse fraction per image (|R| > k * robust sigma of that image)
    impulse_frac = []
    H = W = 992
    outlier_count = np.zeros((H, W), dtype=np.int32)   # per-location outlier tally
    for i in sample_ids:
        c, n = load_pair(i)
        r = residual8(c, n)
        # per-channel robust sigma
        mask_any = np.zeros((H, W), dtype=bool)
        fr = 0.0
        for ch in range(3):
            ri = r[..., ch]
            med = np.median(ri)
            sig = 1.4826 * np.median(np.abs(ri - med))
            m = np.abs(ri - med) > k * max(sig, 1e-6)
            mask_any |= m
            fr += m.mean()
        impulse_frac.append(fr / 3.0)
        outlier_count += mask_any.astype(np.int32)
    impulse_frac = np.array(impulse_frac)
    ni = len(sample_ids)

    # Fixed-location check: fraction of images in which each pixel is an outlier.
    # If defects are random, max per-pixel rate ~ binomial around the mean impulse
    # rate. If hot pixels exist, some locations fire in an anomalously high fraction.
    per_pixel_rate = outlier_count / ni
    mean_any_rate = float((outlier_count.sum()) / (ni * H * W))
    # expected max rate under independence (Bonferroni-ish): mean + margin
    # count locations that are outliers in > 50% of images -> strong hot-pixel signal
    hot_50 = int((per_pixel_rate > 0.5).sum())
    hot_25 = int((per_pixel_rate > 0.25).sum())
    hot_10 = int((per_pixel_rate > 0.10).sum())
    # baseline: under independence at rate p=mean_any_rate over ni images,
    # P(pixel fires >50%) is astronomically small, so any hot_50>0 = fixed defects
    metrics["task5_defects"] = {
        "k_threshold_robust_sigma": k,
        "impulse_fraction_mean": float(impulse_frac.mean()),
        "impulse_fraction_median": float(np.median(impulse_frac)),
        "impulse_fraction_max": float(impulse_frac.max()),
        "per_image_impulse_fraction_p95": float(np.percentile(impulse_frac, 95)),
        "n_images_used": ni,
        "mean_per_pixel_outlier_rate": mean_any_rate,
        "n_pixels_outlier_gt_50pct_images": hot_50,
        "n_pixels_outlier_gt_25pct_images": hot_25,
        "n_pixels_outlier_gt_10pct_images": hot_10,
        "max_per_pixel_outlier_rate": float(per_pixel_rate.max()),
        "interpretation": ("hot_50/hot_25 >> 0 indicates FIXED-LOCATION hot pixels; "
                           "if ~0 and max rate near mean rate => random per-image impulses"),
    }

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].hist(impulse_frac * 100, bins=40, color="#aa33aa", alpha=0.85)
    ax[0].set_title(f"Per-image impulse fraction (|R|>{k}*sigma)")
    ax[0].set_xlabel("impulse fraction (%)"); ax[0].set_ylabel("# images")
    im = ax[1].imshow(per_pixel_rate, cmap="hot", vmax=max(per_pixel_rate.max(), 0.1))
    ax[1].set_title("Per-pixel outlier rate across images\n(bright fixed spots = hot pixels)")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_defects.png"), dpi=110)
    plt.close(fig)
    np.save(os.path.join(OUT, "_per_pixel_outlier_rate.npy"), per_pixel_rate)


# ===========================================================================
# TASK 6: luma vs chroma noise (YCbCr)
# ===========================================================================
def _rgb_to_ycbcr(img_u8):
    """BT.601 full-range YCbCr, float, same convention as PIL 'YCbCr'."""
    arr = img_u8.astype(np.float64)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return np.stack([Y, Cb, Cr], axis=-1)


def task6_luma_chroma(sample_ids):
    print(f"[Task 6] Luma vs chroma sigma over {len(sample_ids)} images ...")
    sig = {"Y": [], "Cb": [], "Cr": []}
    keys = ["Y", "Cb", "Cr"]
    for i in sample_ids:
        c, n = load_pair(i)
        rc = _rgb_to_ycbcr(n) - _rgb_to_ycbcr(c)  # residual in YCbCr space
        for j, key in enumerate(keys):
            ch = rc[..., j]
            med = np.median(ch)
            sig[key].append(1.4826 * np.median(np.abs(ch - med)))
    agg = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                   std=float(np.std(v))) for k, v in sig.items()}
    y_med = agg["Y"]["median"]
    agg["ratio_Cb_over_Y_median"] = agg["Cb"]["median"] / y_med
    agg["ratio_Cr_over_Y_median"] = agg["Cr"]["median"] / y_med
    metrics["task6_ycbcr"] = agg

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [sig["Y"], sig["Cb"], sig["Cr"]]
    bp = ax.boxplot(data, tick_labels=["Y (luma)", "Cb (chroma)", "Cr (chroma)"],
                    showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], ["#888888", "#3366cc", "#cc3333"]):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    ax.set_ylabel("robust sigma (8-bit units)")
    ax.set_title(f"Noise sigma by YCbCr channel\n"
                 f"Cb/Y={agg['ratio_Cb_over_Y_median']:.2f}  "
                 f"Cr/Y={agg['ratio_Cr_over_Y_median']:.2f} (median)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_luma_chroma.png"), dpi=110)
    plt.close(fig)


def main():
    print("=== Noise characterization (seed=%d) ===" % SEED)
    # Task 3 uses ALL images (fast estimator).
    sig_noisy, sig_resid = task3_sigma_distribution()

    # Build a stratified sample of 60 images spanning severity for the heavier
    # tasks (histograms, variance fit, autocorr, chroma). Stratify by blind sigma.
    order = np.argsort(sig_noisy)
    strat = [IDS[order[j]] for j in np.linspace(0, N_IMAGES - 1, 60).astype(int)]
    strat = sorted(set(strat))
    metrics["stratified_sample_ids"] = strat
    metrics["stratified_sample_size"] = len(strat)

    # Defect analysis benefits from many images -> use a larger sample (200).
    defect_ids = rand_sample(200)
    metrics["defect_sample_size"] = len(defect_ids)

    task1_histograms(strat)
    task2_signal_dependent(strat)
    task4_autocorr(strat)
    task5_defects(defect_ids)
    task6_luma_chroma(strat)

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nWrote metrics.json and figures to", OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
