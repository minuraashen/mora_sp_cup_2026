# Mora SP Cup 2026 — Low-Light Image Denoising: Project Knowledge Base

This `docs/` folder is the single source of truth for our understanding of the
competition and our path to a **state-of-the-art blind low-light denoising**
solution. Read it in order.

| # | Document | What it covers |
|---|----------|----------------|
| 01 | [Competition Overview](01_competition_overview.md) | The full rulebook, dataset, submission format, scoring weights, phases, deadlines, hard constraints, and disqualification traps. |
| 02 | [Problem & Metric Analysis](02_problem_and_metric_analysis.md) | Deep analysis of the noise model, the exact scoring formula, the baseline numbers, what the metric *actually* rewards, and the score ceiling. |
| 03 | [Literature Review](03_literature_review.md) | Current SOTA for real/low-light/blind image denoising — deep-learning, classical DSP, and hybrid — with references and how each maps to *our* problem. |
| 04 | [Solution Strategy](04_solution_strategy.md) | The recommended roadmap: architecture choice, synthetic-noise training strategy, validation, CPU-fallback plan, and how to maximize the composite score. |

## TL;DR — What is this competition?

- **Organizer:** University of Moratuwa IEEE Student Branch (IEEE Signal Processing Society Student Chapter), *Mora SP Cup 2026*.
- **Task:** **Blind denoising** of sRGB mobile-phone photos corrupted by a **synthetic low-light sensor-noise model** (read noise, thermal noise, grain, defect pixels). Images are `992×992` PNG.
- **Not** an exposure/brightness enhancement task in practice — the noisy input and clean ground truth share the same (dim) exposure; only noise differs. The word "Enhancement" in the title is aspirational; the metric only rewards noise removal. See [doc 02](02_problem_and_metric_analysis.md).
- **Data:** 460 public pairs (train/val), 20 preliminary noisy-only images (`461–480`), 20 fully-hidden final-round images (`481–500`).
- **Metric:** `Score = 0.6·clip(ΔPSNR/15, 0, 1) + 0.4·max(ΔSSIM, 0)` — rewards *improvement over the noisy input*, not absolute quality.
- **Baseline to beat:** composite **≈ 0.263** (classical NLM + defect correction).
- **Deliverable:** a `scripts/denoise.py` that denoises a folder and renames outputs, a `<5 page` report, a frozen private GitHub repo (organizer added as collaborator), and a `TeamName.zip` of the 20 denoised preliminary images.

## Current repo state (as of setup)

- Starter repo is present: `baseline/`, `evaluation/`, `competition_data/`, `scripts/`.
- **Dataset is NOT downloaded yet** — `competition_data/public/` and `submissions/` are gitignored and empty. Download from the Drive link in [doc 01](01_competition_overview.md) and place per the layout there.
- `scripts/denoise.py` is **empty** — this is our main deliverable to build.
- Baseline (`baseline/denoise.py`) = defect-pixel median repair + OpenCV Non-Local Means, CPU-only.
