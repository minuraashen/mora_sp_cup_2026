# 05 — Implementation Log

Running record of what was built, what was decided, and what the measurements said.
Written as the work proceeded (2026-09-18 → 2026-09-19), so the reasoning is recorded
at the point the decision was actually made rather than reconstructed afterwards.

Companion documents: [04 Solution Strategy](04_solution_strategy.md) (the plan),
[06 Results](06_results.md) (final numbers).

---

## 0. Environment (measured, not assumed)

| Item | Value |
|---|---|
| GPU | RTX 4050 Laptop, 6141 MiB total, **5920 MiB free** |
| CPU / RAM | Intel Core 5 210H, 12 logical cores / 16.78 GB |
| Training env | conda `vision_env` — torch 2.9.0+cu128, CUDA available |
| **Shipped env** | conda `spcup26` — Python 3.11.16, **torch 2.14.0+cpu**, numpy 1.26.4, Pillow 10.4.0 |
| Data | 460 public pairs + 20 preliminary, all verified 992×992 RGB uint8 |

Training runs in `vision_env` because it already had a working CUDA torch; the
**submission** targets `spcup26` with a CPU-only torch, which is the configuration the
graded images are produced in. A GPU is never required to reproduce the submission.

## 1. Decisions taken, with the evidence

### 1.1 Fine-tune pretrained NAFNet rather than train from scratch

The deciding measurement is that training throughput on this GPU is **pixel-bound at
~0.9 Mpx/s** regardless of how batch and patch are arranged. A 12-hour budget is
therefore ≈ 38,900 Mpx, against NAFNet's official SIDD recipe of 400k × bs32 × 256²
= **838,000 Mpx**. We can afford about **4.6 % of the original training compute**, so
matching that quality from a random initialisation is not reachable in the time
available.

The "the corruption is synthetic so there is no domain gap" argument actually
*reinforces* this: zero domain gap means training **data** is free (we can synthesize
unlimited perfectly-matched pairs), so data is not the bottleneck — compute is. A
pretrained initialisation buys compute, which is the scarce resource.

Source used: `https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width32.pth`
(the upstream repo publishes only Google Drive links, which are not scriptable).
Provenance was verified three ways rather than trusted:
- size is exactly **116,861,841 bytes**;
- that equals 29,159,715 params × 4 B plus container overhead, and our independent
  implementation has **29,159,715 parameters**;
- it loads with `strict=True` into our architecture — no missing or unexpected keys.

NAFNet is MIT/Apache-2.0, so competition use is unrestricted.

**Zero-shot result before any training: composite 0.3367** on the full 46-image
held-out split (PSNR 24.80, SSIM 0.7996, +4.98 dB over the noisy input) — already above
the organizer baseline of 0.2633 with no fine-tuning at all. Fine-tuning on the
competition's own data then took it to **0.6433**.

Rejected: **width64** (116 M params, 0.39 Mpx/s ≈ 43 % of the pixel budget, 4× the CPU
cost); **Restormer** (attention at 992² is minutes per image on this CPU, against a
runtime criterion worth 15 %); **SCUNet as primary** (Swin blocks are CPU-unfriendly
and it targets a different degradation family) — kept as a no-training fallback.

### 1.1b Outcome

Training ran 48,000 iterations in 289.7 minutes on the 4050 (0.362 s/it, peak 4.88 GB).
Final submitted model: **composite 0.6433** (PSNR 31.63, SSIM 0.9195) on the held-out
46, scored by the unmodified official evaluator on CPU. That is 2.44x the organizer
baseline and above the 0.60 target set in [04](04_solution_strategy.md).

### 1.2 Three-channel input, not four

The plan had considered concatenating a blind-σ map as a 4th input channel
(CBDNet-style conditioning). That was dropped: it changes the shape of `intro.weight`
and so forfeits `strict=True` loading of the pretrained checkpoint. Given decision 1.1,
preserving the initialisation is worth more than explicit conditioning — and NAFNet
trained across the full σ continuum is intrinsically blind anyway.

### 1.3 PSNRLoss, not L1/Charbonnier

The competition metric averages **per-image** PSNR. PSNR is per-image-normalised
log-MSE, so `PSNRLoss` optimises literally the scored quantity. L1 or L2 weight each
image by its absolute error, which lets the noisiest images dominate the gradient even
though the metric awards them no extra credit.

### 1.4 CPU is the default device

CPU and CUDA kernels differ numerically, and "the submitted code must reproduce the
submitted images" is a disqualification rule. The handbook *requires* a CPU fallback,
so the reviewer's machine certainly has one; making `--device cpu` the default means
the plain documented command reproduces the zip anywhere. `--device cuda` remains
available and is documented as *not* the submitted configuration.

### 1.5 Whole-image inference; tiling only as an OOM fallback

`992 = 2⁵ × 31` is a multiple of 32, so a 4-level NAFNet needs **zero padding** at full
resolution. Measured peak was **1.28 GB** for a full 992² forward. Tiling would process
1.3–2.4× more pixels (slower, against the 15 % runtime score) and adds a second code
path that can fail, so it is kept only behind an automatic out-of-memory ladder
(`0 → 512 → 256 → classical`).

### 1.6 No hardcoded hot-pixel map

The characterization identified 492 fixed defect pixels, and shipping those coordinates
would measurably help. It is not shipped: a per-dataset coordinate table is a lookup
derived from the competition data, which sits on the wrong side of the "no hardcoding
or dataset exploitation" rule. Instead the synthesizer injects them during training, so
the network learns to repair them from data rather than from a lookup — and, per §3b,
it does so better than a median filter would. The map is used **only** to generate
training data, never at inference.

## 2. Bugs the verification gates caught

Both gates were written *before* the thing they check, and both caught real defects
that would otherwise have silently degraded the result.

### 2.1 Synthetic-noise generator: variance inflated by 31 %

`scripts/tools/check_noise_model.py` compares synthesized residuals against
`metrics.json`. First run:

| ch | a (synthetic) | a (measured) | err | b (synthetic) | b (measured) | err |
|---|---|---|---|---|---|---|
| R | 5.168 | 4.359 | 18.6 % | 524.4 | 413.7 | 26.8 % |
| G | 4.922 | 4.216 | 16.8 % | 487.5 | 369.6 | 31.9 % |
| B | 6.627 | 4.857 | 36.4 % | 384.9 | 357.1 | 7.8 % |

Cause: severity was drawn as `s ~ U(0.55, 1.65)`, for which `E[s²] = 1.31`. The
measured `a` and `b` are pooled over the whole dataset and therefore **already average
over severity**, so the severity distribution must satisfy `E[s²] = 1` or every
synthetic variance is inflated by exactly that factor. For `s ~ U(lo,hi)` that
constraint is `lo² + lo·hi + hi² = 3`, which fixes `hi` once `lo` is chosen →
**`s ~ U(0.26, 1.5874)`**.

### 2.2 Synthetic-noise generator: tails too light

After the variance fix, excess kurtosis was 1.15–1.55 against a measured 1.86–2.36.
Sparse impulses at 0.033 % of pixels are far too rare to produce an excess kurtosis
near 2 on their own. Added a **two-component Gaussian scale mixture** on the per-pixel
variance, with weights solved in closed form so that `E[t] = 1` (total variance exactly
preserved, so the `a,b` fit is untouched) and `E[t²] = (K+3)/3` for a target excess
kurtosis `K`:

```
t2 = 1 + sqrt(1 - (1 - E[t²](1-q))/q),    t1 = (1 - q·t2)/(1-q)
```

Final state of the gate (mixture target K = 0.85, q = 0.10; the impulses supply the
remainder):

| ch | a err | b err | kurtosis (synth vs measured) |
|---|---|---|---|
| R | 12.7 % | 13.4 % | 2.05 vs 1.86 (10.1 %) |
| G | 7.0 % | 12.9 % | 2.19 vs 2.04 (7.2 %) |
| B | 5.6 % | 2.5 % | 2.56 vs 2.36 (8.6 %) |

### 2.3 Blind σ estimator off by 6×

`estimate_sigma_blind` returned 173.7 (8-bit) where the plausible range is 6–43 — the
factor 6 in the Immerkaer formula had been applied twice (divided then multiplied).
After the fix it tracks the wavelet-MAD reference within ~4 %:

| image | Immerkaer | reference |
|---|---|---|
| 009 | 28.95 | 27.83 |
| 042 | 31.20 | 29.81 |
| 459 | 39.68 | 39.35 |

### 2.4 Impulse pre-pass 40× slower than necessary

First implementation took ~9.7 s/image because the adaptive median computed **full
image** 5×5 and 7×7 medians (a 49-plane stack over 992×992×3) in order to use a few
thousand of their values. Since only the already-flagged pixels — about 0.05 % of the
frame — ever need the larger windows, the larger medians are now evaluated only at
those coordinates. **226 ms/image**, flagging 0.041 % of pixels, consistent with the
measured 0.033–0.05 % impulse rate.

## 3. Validation methodology

- **Split**: the 460 ids are sorted by the pre-computed blind σ
  (`scripts/analysis/noise_characterization/_sigma_blind.npy`) and every 10th starting at index
  5 is held out → **414 train / 46 validation**, of which every 2nd (23 images) forms
  the in-training fast-validation subset. Deterministic, no RNG, and severity-stratified
  by construction: the held-out σ range is 7.57–40.66 against a full-set range of
  5.86–42.83. Written once to `scripts/artifacts/split.json` and loaded from there by
  every script, so it cannot drift.
- **Metric**: never reimplemented. `scripts/tools/validate.py` uses the unmodified
  `evaluation/evaluate.py` two ways — as a subprocess (the literal official path) and
  imported by file path for per-image rows — and **hard-fails if the two disagree by
  more than 1e-8**. Confirmed agreeing exactly (0.30401935).
- **Training-time metric**: `scripts/common.py` reimplements PSNR/SSIM in torch for
  speed; `scripts/tools/check_metric.py` asserts it against scikit-image to < 1e-6 so
  checkpoint selection cannot be made on a number that disagrees with the official one.
- **Quantisation**: predictions are quantised to uint8 *before* scoring, because they
  ship as 8-bit PNGs; scoring un-quantised floats would over-report.
- Selection is on **validation composite**, never on loss.

## 3b. The classical impulse pre-pass was removed from the network path

The plan assumed the sigma(I)-gated adaptive median would help everywhere. It helps the
**classical** pipeline (+0.0152 composite, 0.3730 vs 0.3578) but *costs* score in front
of the trained network. A gate-threshold sweep settled why: the composite rises
monotonically as fewer pixels are filtered (k=4.5 -> 0.5664, 6 -> 0.5671, 8 -> 0.5687,
12 -> 0.5753, off -> 0.5762), converging on the no-filter result. That is a dose-response
curve, not a mis-tuned threshold: the network trains on impulses and hot pixels injected
by the synthesizer, so it learns to repair them, and its prior beats a median.

Consequence: `prepass` now defaults to **off** for the network and **on** for the
classical fallback, and `--prepass` is opt-in. The classical contribution to the shipped
system is therefore the measured noise model that generates 40% of the training data,
the classical fallback denoiser, COLA overlap-add tiling, and the set of measured
rejections -- not a filter in the scored path.

## 4. Reproducibility measures

- No stochasticity exists in the inference pipeline at all (no random crops, dropout or
  sampling), so seeds are a safeguard rather than the mechanism.
- `spcup/determinism.py` is configured *before* torch is imported (`CUBLAS_WORKSPACE_CONFIG`
  and the thread-count variables must precede it), then sets seeds,
  `use_deterministic_algorithms`, `cudnn.deterministic`, disables `cudnn.benchmark` and
  TF32, and pins the thread count.
- PNG writing uses `floor(x·255 + 0.5)` — **not** `np.round`, which is round-half-to-even
  and disagrees on a meaningful fraction of pixels — with fixed `compress_level` and a
  pinned Pillow, so the file *bytes* reproduce rather than just the pixels.
- `scripts/tools/preflight.py` re-runs the pipeline in a clean subprocess and compares
  SHA-256 against the files being submitted, and performs a **rename test** (`461_noise.png`
  copied to `999_noise.png` must produce a byte-identical output) as positive evidence
  that no per-image logic exists.
