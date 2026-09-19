"""
check_noise_model.py -- HARD GATE on the synthetic-noise generator.

40% of every training batch is synthetic. If the generator does not reproduce the
measured degradation, that 40% is actively teaching the network the wrong inverse
problem. So before training we synthesize noise onto real clean images and check the
residual statistics against scripts/analysis/noise_characterization/metrics.json:

  * per-channel residual std            (target: the real residual's own std)
  * excess kurtosis                     (target 1.86 / 2.04 / 2.36, heavy tailed)
  * variance-vs-intensity slope a       (target 4.359 / 4.216 / 4.857)
  * variance-vs-intensity intercept b   (target 413.7 / 369.6 / 357.1)
  * per-image sigma range               (target ~6..43 in 8-bit units)

    python scripts/tools/check_noise_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

import common  # noqa: E402
import noise_model  # noqa: E402

TOL_REL = 0.15      # 15% relative tolerance on a and b


def fit_var_vs_intensity(clean255: np.ndarray, resid: np.ndarray, nbins: int = 32):
    """Weighted least-squares fit of var(resid) = a*I + b, exactly as characterize.py does."""
    out = []
    for c in range(3):
        I = clean255[..., c].ravel()
        R = resid[..., c].ravel()
        edges = np.linspace(0, 255, nbins + 1)
        idx = np.clip(np.digitize(I, edges) - 1, 0, nbins - 1)
        xs, ys, ws = [], [], []
        for bi in range(nbins):
            m = idx == bi
            cnt = int(m.sum())
            if cnt <= 5000:
                continue
            xs.append(I[m].mean())
            ys.append(R[m].var())
            ws.append(cnt)
        xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, dtype=np.float64)
        W = np.diag(ws / ws.sum())
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        coef = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ ys, rcond=None)[0]
        out.append((float(coef[0]), float(coef[1])))
    return out


def excess_kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64).ravel()
    m = x.mean()
    s = x.std()
    return float(((x - m) ** 4).mean() / (s ** 4) - 3.0)


def main() -> int:
    metrics = common.read_json(common.ANALYSIS_DIR / "metrics.json")
    split = common.load_split()
    clean_mm = np.load(common.ARTIFACTS_DIR / "clean_u8.npy", mmap_mode="r")
    hot = np.load(common.ARTIFACTS_DIR / "hotpixels.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(0)
    synth = noise_model.NoiseSynthesizer(device, gen)

    S = common.IMG_SIZE
    mask = torch.zeros((S, S), dtype=torch.bool, device=device)
    mask[torch.from_numpy(hot[:, 0].astype(np.int64)), torch.from_numpy(hot[:, 1].astype(np.int64))] = True

    ids = split["train"][::40][:12]
    print(f"synthesizing on {len(ids)} clean images (device={device})\n")

    resid_all, clean_all, sigmas = [], [], []
    for i in ids:
        g_u8 = np.ascontiguousarray(clean_mm[i - 1])
        g = torch.from_numpy(g_u8).permute(2, 0, 1)[None].to(device).float() / 255.0
        n = synth(g, mask)
        r = ((n - g) * 255.0)[0].permute(1, 2, 0).cpu().numpy()
        sigmas.append(float(r.std()))
        # subsample so the aggregate fit stays cheap
        sel = np.random.default_rng(i).integers(0, S * S, 40000)
        resid_all.append(r.reshape(-1, 3)[sel])
        clean_all.append(g_u8.reshape(-1, 3)[sel].astype(np.float64))

    resid = np.concatenate(resid_all, axis=0)[:, None, :].reshape(-1, 1, 3)
    clean = np.concatenate(clean_all, axis=0)[:, None, :].reshape(-1, 1, 3)

    ok = True
    print(f"{'ch':>3} {'stat':>10} {'synthetic':>12} {'measured':>12} {'rel err':>9}")

    ref_kurt = [metrics["task1_residual"][c]["excess_kurtosis"] for c in ("R", "G", "B")] \
        if "task1_residual" in metrics else [1.859, 2.044, 2.358]
    fits = fit_var_vs_intensity(clean, resid)
    ref_ab = [(4.35921, 413.721), (4.21573, 369.608), (4.85656, 357.141)]

    for ci, cname in enumerate("RGB"):
        a, b = fits[ci]
        ra, rb = ref_ab[ci]
        k = excess_kurtosis(resid[:, 0, ci])
        for label, got, ref, tol in [("a (slope)", a, ra, TOL_REL),
                                     ("b (interc)", b, rb, TOL_REL),
                                     ("exc kurt", k, ref_kurt[ci], 0.60)]:
            rel = abs(got - ref) / abs(ref)
            flag = "" if rel <= tol else "   <-- OUT OF TOLERANCE"
            if rel > tol:
                ok = False
            print(f"{cname:>3} {label:>10} {got:12.4f} {ref:12.4f} {rel:8.1%}{flag}")

    smin, smax = min(sigmas), max(sigmas)
    ref_s = metrics["task3_sigma"]["sigma_residual_std_8bit"]
    print(f"\nper-image residual sigma (8-bit): synthetic {smin:.1f}..{smax:.1f}  "
          f"measured {ref_s['min']:.1f}..{ref_s['max']:.1f}")
    if smin < 4.0 or smax > 55.0:
        print("  <-- sigma range is implausible")
        ok = False

    print("\nPASS: synthetic noise matches the measured model." if ok
          else "\nFAIL: synthetic generator does not match the measured model. Fix before training.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
