# 04 — Solution Strategy & Roadmap to SOTA

Goal: **maximize the preliminary grade** (not just the image metric). That means hitting all four components — 35% image score, 40% report, 10% classical-SP, 15% code/runtime — with a single coherent, well-justified, reproducible pipeline. Target **composite ≥ 0.60** on held-out validation (vs. baseline 0.263).

## 0. Guiding principles (from [doc 02](02_problem_and_metric_analysis.md))

- **Denoise, don't brighten.** No global exposure/gamma/Retinex. Match GT brightness.
- **PSNR is the primary lever** (0.6 weight, ~10× SSIM per unit) until ΔPSNR≈15 dB; keep SSIM high too.
- **Blind:** handle varying, unknown severity — either an intrinsically-blind net or explicit noise conditioning.
- **CPU-runnable + reproducible + offline** — non-negotiable engineering constraints (and part of the grade).

## 1. Recommended architecture — a justified hybrid

```
noisy 992×992 RGB
   │
   ├─(1) Classical pre-pass:  adaptive-median hot/defect-pixel repair (impulse outliers)
   │         └─ optional: convert to YCbCr for separate luma/chroma handling
   │
   ├─(2) Learned core (the workhorse):  NAFNet  (efficient CNN U-Net)
   │         - trained on 460 real pairs + parametric-synthetic noise augmentation
   │         - blind (severity baked into training distribution); optionally a
   │           CBDNet-style noise-map conditioning input
   │         - run via OVERLAP-TILED inference (e.g. 512 tiles, 32px feather)
   │
   ├─(3) Self-ensemble TTA:  average the 8 D4 flips/rotations  (toggle: --tta)
   │
   ├─(4) Optional classical refinement:  light wavelet/BM3D or guided-filter pass
   │         on chroma only, to clean residual color noise without softening luma
   │
   └─→ denoised, written as <id>.png
```

**Why this shape:**
- **NAFNet** = best quality-per-FLOP → satisfies both the 35% (quality) and 15% (CPU runtime) at once. It beats Restormer/U-Net in several head-to-heads at a fraction of the compute. [doc 03 A2]
- The **classical stages (1) & (4)** are not decoration — impulse/hot-pixel repair genuinely helps (Gaussian-type nets smear impulses), and chroma-only refinement targets the dominant low-light chroma noise. Together they *earn the 10% classical-SP credit with real justification*, and give the report a proper "hybrid signal-processing + learning" story.
- **Self-ensemble** is a pure classical DSP trick worth ~0.1–0.3 dB for free.

**Fallback / alternative cores to have ready (state as "alternatives considered" in report):**
- **SCUNet** pretrained blind sRGB denoiser [doc 03 A7] — strongest *zero-training* option; fine-tune on the 460 pairs. Good insurance if training-from-scratch underperforms in the time budget.
- **Restormer** or **Restormer+NAFNet hybrid** [doc 03 C1] — higher ceiling, higher CPU cost. Consider only if runtime budget allows / if we ensemble for the image score.

## 2. Data & training strategy

1. **Download** the dataset (Drive link in [doc 01](01_competition_overview.md)); place per the required layout. Verify counts (460/460/20).
2. **Characterize noise** ([doc 02](02_problem_and_metric_analysis.md) §6): residual stats, per-image σ, signal-dependence `σ²=a·x+b`, spatial correlation, defect fraction, luma/chroma split.
3. **Train primarily on the 460 REAL pairs** — this is the highest-signal data and matches the test distribution exactly. Split e.g. **410 train / 50 val**, stratified by estimated severity.
4. **Augment with parametric-synthetic noise** fit to the measured model (Poisson-Gaussian + read + impulse) applied to the clean GTs → unlimited extra pairs, and — importantly — coverage of severities that may appear in the hidden `481–500`. Mix real+synthetic (e.g., 50/50) so we don't overfit to synthetic artifacts.
5. **Training recipe** (borrowed from NTIRE-2025 winner, [doc 03 C1]):
   - Loss = Charbonnier(L1) + λ·(1−MS-SSIM) + optional wavelet-domain L1. (Directly optimizes the scored metric.)
   - **Progressive patch size** 128→256→(384); random crop from 992² tiles; D4 flip/rotate augmentation.
   - AdamW, cosine LR, EMA weights, mixed precision on GPU (Colab T4 / Kaggle).
   - Early-stop / checkpoint-select on **val composite**, not loss.
6. **Guard against synthetic overfit:** always report val composite on *real* noisy public images.

## 3. Inference engineering (the `scripts/denoise.py` deliverable)

Must satisfy the [doc 01](01_competition_overview.md) §4 contract:
- **CLI:** accept `--noise_dir/--denoised_dir` (PDF form) **and** `--input_dir/--output_dir` (baseline form) as aliases. Add `--device {auto,cpu,cuda}`, `--tile`, `--tta`.
- **`_noise` suffix stripping** → `###.png`.
- **Overlap-tiled** forward pass with feathered blending (no seams) for 992².
- **CPU fallback**: auto-detect CUDA, else CPU; ensure it finishes on a few images in reasonable time (test explicitly). Provide a lighter default (tiling + TTA off) so the *default* run is fast, TTA opt-in.
- **Deterministic**: fixed seeds, `torch.use_deterministic_algorithms` where feasible → outputs reproduce the submitted `TeamName.zip` exactly (DQ risk otherwise).
- **Offline**: weights loaded from a local path; if hosted on Drive, document filename + path + SHA-256 in `scripts/README.md`.
- Bundle `requirements.txt`; keep code clean/commented (15% score).

## 4. Milestones (8-day sprint — deadline Sept 19, today Sept 11)

| Day | Deliverable |
|-----|-------------|
| **1** | Download data; verify layout; **reproduce baseline 0.263** with `evaluate.py`; noise characterization study ([doc 02] §6). |
| **1–2** | Build a strong **classical pipeline** (adaptive-median + BM3D/wavelet + chroma handling, tuned) → target ~0.35–0.40. This is our safe, always-submittable fallback and earns the 10% classical credit. |
| **2–3** | Stand up **NAFNet** training on Colab/Kaggle using the 460 pairs; get first checkpoint beating the classical pipeline. Also evaluate **pretrained SCUNet** as a baseline/fallback. |
| **3–5** | Add synthetic-noise augmentation, progressive training, SSIM+wavelet loss; iterate to **≥0.55–0.60** val composite. |
| **5–6** | Integrate the full hybrid `denoise.py` (tiled inference, TTA, classical pre/post, CPU fallback); verify reproducibility and CPU runtime. |
| **6–7** | Run on `461–480`, produce `461.png–480.png`, self-sanity-check formats/dimensions. Freeze model, compute SHA-256. |
| **7–8** | Write the **<5-page report** (problem, dataset observations, architecture diagram, justification, alternatives, methodology, validation, results). Set up private repo, add organizer collaborator, record commit SHA, assemble `TeamName.zip`. |

> **De-risking:** always keep a *working, submittable* version. The classical pipeline (Day 1–2) guarantees we beat baseline even if training slips. Improve on top of it, never replace the safety net until the deep model is verified end-to-end.

## 5. Report outline (maps to the 40% + reuses these docs)

1. **Problem intro** (1 para) — blind low-light sRGB sensor-noise denoising.
2. **Dataset observations** — the noise characterization results (histograms, σ distribution, signal-dependence, defect pixels, chroma noise). *This is where "observations on the dataset" points are won.*
3. **Architecture** — the hybrid diagram from §1, each stage justified.
4. **Justification** — why NAFNet (quality/CPU trade-off), why blind training, why classical stages, tie to the metric.
5. **Alternatives considered** — Restormer/hybrid, SCUNet, MambaIR, pure-classical BM3D; why not chosen (cite [doc 03]).
6. **Methodology** — data split, synthetic-noise fitting, progressive training, loss, validation with official metric.
7. **Results** — val PSNR/SSIM/composite table vs. baseline; ablations (classical-only, deep-only, +TTA); CPU runtime.
8. Diagrams preferred; ≤5 pages, 12-pt, 1-inch margins, no cover.

## 6. Things that will lose points / DQ (checklist)

- [ ] Do **not** brighten/tone-map (metric penalizes it).
- [ ] Do **not** modify `baseline/`, `evaluation/`, or `competition_data/` code/structure — only `scripts/`.
- [ ] Do **not** hardcode per-image outputs or lookups → instant DQ.
- [ ] Ensure `denoise.py` **reproduces** the submitted images exactly (seeds, determinism).
- [ ] Ensure it **runs offline on CPU** without error (35% forfeited if it errors).
- [ ] Correct output: **RGB PNG, 992×992, named `461.png…480.png`**, only those 20 in the zip.
- [ ] Freeze: commit SHA in README, organizer added as collaborator, model SHA-256 documented, no post-deadline commits.
- [ ] Report ≤5 pages, correct formatting.

## 7. Open questions to confirm with organizers / from the data

1. **Exact CLI flags** the organizer will use (`--noise_dir/--denoised_dir` per PDF vs. `--input_dir/--output_dir` per baseline). Support both to be safe.
2. Is the noise **signal-dependent** and **spatially correlated**? (Drives synthetic model + whether AWGN classical methods suffice.) — answered by the noise study.
3. Are defect pixels **fixed-location** or random per image? — affects the pre-pass.
4. What CPU/time budget is "reasonable" for the live final round? (Keep default inference well under a minute/image on CPU if possible.)
