# 01 — Competition Overview (Mora SP Cup 2026)

*Source: `Participant_Handbook.pdf` (official competition document) + starter-repo READMEs. Every fact below is traceable to those.*

## 1. Identity & Theme

- **Event:** Mora SP Cup 2026 — **Low-Light Image Denoising & Enhancement Challenge**.
- **Host:** University of Moratuwa IEEE Student Branch, Signal Processing Society Student Chapter.
- **Premise:** Under low light, camera sensors show **read noise, thermal noise, unwanted grain, and visual distortion**. We must design a denoiser that removes these artifacts while **preserving detail, structure, and visual quality**.
- **Approach freedom:** Classical DSP, deep learning, or a **hybrid** — all allowed. Python is mandatory; external libraries allowed.

## 2. The Data

- **Resolution:** every image is **992 × 992** pixels, PNG, RGB.
- **Synthetic corruption:** clean mobile-phone photos corrupted by a **synthetic low-light noise model** reproducing several sensor-related artifacts. **Different severity levels of noise** are mixed into the set → the denoiser must be **blind** (handle unknown, varying noise strength).
- **Split (500 images total):**

| Split | IDs | What we get | Purpose |
|-------|-----|-------------|---------|
| **Public** | `001–460` | 460 noisy (`###_noise.png`) **+** 460 ground-truth (`###.png`) | Develop, train, validate. |
| **Preliminary** | `461–480` | 20 noisy only (`###_noise.png`); GT hidden | We denoise → submit `461.png–480.png`. Scored against hidden GT. |
| **Final round** | `481–500` | Nothing until the physical final | Finalists denoise live using the *frozen* submitted code + model. |

- **Dataset link (Google Drive):** `https://drive.google.com/drive/folders/1kxh5m5Rm_3EB9oNK0nfHOMeTqXx9zyNq`
- **Official repo:** `https://github.com/moraspcup-official/mora_sp_cup_2026.git`
- **Rules on data:** noisy images **must not be renamed**; do **not** redistribute/mirror the dataset; do **not** attempt to obtain/reconstruct hidden GT or final-round data.

## 3. Required Repository Structure

```text
baseline/            # reference example (defect correction + NLM). Do NOT modify.
competition_data/
  public/
    ground_truth/    001.png … 460.png
    noisy/           001_noise.png … 460_noise.png
    denoised/        (our outputs on public set)
  submissions/
    noisy/           461_noise.png … 480_noise.png
    denoised/        461.png … 480.png   <-- our official prelim outputs
evaluation/
  evaluate.py        # self-check scorer (same metric as official). Do NOT modify.
scripts/             # <-- OUR CODE LIVES HERE ONLY
  denoise.py         # main deliverable
  README.md          # run instructions + freeze info (SHA, model link/checksum)
  requirements.txt
```

- **Hard rule:** *All* custom code goes in `scripts/`. **Teams may not change files in any other directory.**
- **Rename only outputs:** `461_noise.png → 461.png`, …, `480_noise.png → 480.png`. `denoise.py` must perform this renaming itself.

## 4. The `denoise.py` Contract

- Must be `scripts/denoise.py`. Running it denoises a given noisy dir → writes to a given denoised dir.
- **Example invocation (from the PDF):**
  ```bash
  python scripts/denoise.py --noise_dir x --denoised_dir y
  ```
  ⚠️ Note the PDF uses `--noise_dir` / `--denoised_dir`, while the **baseline** uses `--input_dir` / `--output_dir`. **Support the PDF's argument names** (and ideally accept both) — the organizer will likely invoke exactly the PDF form. Confirm with organizers if unsure.
- Must strip the `_noise` suffix on output (`###_noise.png → ###.png`).
- **Must run fully offline** (no network at inference) once model files are placed. **Must include a CPU fallback** and run in reasonable time on a mid-range CPU with a few images.
- `scripts/README.md` must give exact run instructions + a `requirements.txt`.

## 5. Scoring — Two Layers

### 5.1 Image quality score (the composite)
Computed by `evaluation/evaluate.py`, identical to official scoring:

```
ΔPSNR = PSNR(denoised) − PSNR(noisy)
ΔSSIM = SSIM(denoised) − SSIM(noisy)
N = clip(ΔPSNR / 15, 0, 1)
S = max(ΔSSIM, 0)
Composite = 0.6·N + 0.4·S
```
- **Baseline composite ≈ 0.263.** A submission **must beat the baseline** to be ranked/shortlisted.
- Full metric analysis + ceiling in [doc 02](02_problem_and_metric_analysis.md).

### 5.2 Preliminary-round grade breakdown (100%)

| Weight | Component |
|-------:|-----------|
| **40%** | Report and technical analysis. |
| **10%** | Use of **significant classical signal-processing techniques beyond the baseline**, with justification. |
| **35%** | Composite score on the 20 prelim images (`461–480`) vs. hidden GT. *Zero if the submission errors out.* |
| **15%** | Clear code structure, guideline adherence, and **better runtime on a standard mid-range CPU**. |

**Strategic reading:** the image score is only 35%. **60% of the grade is report + classical-SP + code/runtime.** A pure black-box deep net that ignores the 10% classical-SP requirement and is slow on CPU leaves ~25% on the table. The winning profile is a **well-justified hybrid** (deep core + meaningful classical DSP) with a clean, CPU-runnable, well-documented codebase and a sharp report. See [doc 04](04_solution_strategy.md).

## 6. Submission Package (Preliminary Round)

One Google Drive folder containing:
1. **`TeamName.zip`** — *exactly* the 20 denoised PNGs `461.png–480.png` (no noisy images, nothing else).
2. **`TeamName_Report.pdf`** — the report.

Plus:
- A **private GitHub repo** mirroring the starter structure, with the **official Mora SP Cup organizer account added as a collaborator** before the deadline.
- The **exact Git commit SHA** of the submitted version written in the repo README.
- If model weights are too large for git: upload to Drive, and document in `scripts/README.md` the **filename, expected local path, load/setup steps, and SHA-256 checksum**.

### Report must include (kept **under 5 pages**, 12-pt font, 1-inch margins, no cover page):
- Brief intro to the problem.
- Problem + **observations on the dataset**.
- Solution architecture (**diagrams preferred**).
- Justification of the chosen approach.
- **Alternatives considered.**
- Methodology for building the model using the 460 pairs (incl. **validation strategy** — must be stated explicitly).
- Any other necessary info.
- ⚠️ Over 5 pages / wrong formatting ⇒ **penalty or disqualification.**

## 7. Freeze Rules (read carefully — DQ risk)

- **Code freeze:** the README-stated commit SHA is the official version; **no commits after the deadline**. The recorded SHA must remain the latest commit.
- **Model freeze:** organizers record the **SHA-256** of every submitted checkpoint; it must not be modified/replaced/re-uploaded afterward (even at the same Drive link).
- **Final round uses the same frozen code + model** to denoise `481–500` live at the University of Moratuwa. Any mismatch ⇒ penalty/DQ.

### Instant-disqualification traps
- **Hardcoding / direct lookups / image-specific mappings / any dataset exploitation** that isn't a generalizable denoising procedure.
- **If the submitted code does not reproduce the submitted denoised images** → disqualified. (So the outputs in `TeamName.zip` must be *exactly* what `denoise.py` produces from the frozen code+model.)
- Submission that **errors while running** forfeits the 35% image score.

## 8. Phases & Dates

| Phase | When | What |
|-------|------|------|
| **Registration** | before **Sept 2, 2026, 23:59** local | Google Form (link in PDF §4). Team of 1–3, same Sri Lankan university, undergrad. |
| **Preliminary Round** | **Aug 29 – Sept 19, 2026** | Submit code + report + 20 denoised images. |
| **Final Round** | TBA | Physical, at University of Moratuwa. Top **5 teams**. Live denoise `481–500` + presentation + viva. Judges' decision final. |

> ⚠️ **Timeline note:** today is **2026-09-11**. The preliminary deadline (**Sept 19, 2026**) is ~8 days away. Registration (Sept 2) has passed — confirm the team is already registered. Prioritize a working, submittable end-to-end pipeline first, then push quality.

## 9. Resources & Allowed Tooling

- **Compute:** Google Colab (free T4 GPU), Kaggle (30 GPU-hrs/week). Training may use GPU; **inference must have a CPU fallback**.
- **Explicitly suggested libraries:** PyTorch + **Kornia** (differentiable image processing), **OpenCV** + **scikit-image** (classical bilateral, non-local means, wavelet denoising).
- **Contacts:** Heshan Ranasinghe (hranasinghe505@gmail.com), Maneth Perera (pereragmb.23@uom.lk).

## 10. Baseline (what we must beat)

- `baseline/denoise.py`: per-channel **median-outlier defect-pixel repair** (threshold 0.25) → **OpenCV `fastNlMeansDenoisingColored`** (h=10). **CPU-only, no training, no exposure correction.**
- Reported on the 460-image public set: **PSNR 24.336 dB, SSIM 0.6717, ΔPSNR +4.510, ΔSSIM +0.2073, composite 0.2633.**
- Implies **noisy-input baseline: PSNR ≈ 19.83 dB, SSIM ≈ 0.464.** These are the reference points every ΔPSNR/ΔSSIM is measured against.
