# NAFNet redo guided by the project documentation

This experiment follows all five Markdown files in `docs/`, read in their
specified order, together with the existing empirical noise study. It revises
the earlier pretrained-only experiment. The documented **0.60 held-out composite
target is a goal, not a measured result or a guarantee**.

## What the documents require and how it is implemented

| Source | Requirement or guidance | Implementation |
|---|---|---|
| `docs/00_INDEX.md` | Blind sRGB denoising; original exposure | No gamma, brightening, Retinex, or tone mapping |
| `docs/01_competition_overview.md` | Custom code under `scripts/`; preserve starter baseline and evaluator | All new code and reports live under `scripts/`; originals are imported/read only |
| `docs/01_competition_overview.md` | Offline folder CLI, both flag conventions, output suffix stripping | Existing `denoise.py` contract retained |
| `docs/02_problem_and_metric_analysis.md` | Characterize actual noise before setting parameters | Existing full-set study reviewed; augmentation parameters refitted using training targets only |
| `docs/02_problem_and_metric_analysis.md` | Stratified holdout; official PSNR/SSIM/composite | Fixed 410/50 split, using noisy-only Haar-MAD severity; exact official scoring functions reused |
| `docs/03_literature_review.md` | Efficient NAFNet core with practical training tricks | Fine-tuning the verified SIDD-width32 checkpoint; progressive crops, EMA, D4, structural/wavelet losses |
| `docs/04_solution_strategy.md` | Classical impulse pre-pass | Noise-adaptive 3x3/5x5 median repair with a local edge-protection gate |
| `docs/04_solution_strategy.md` | Real pairs plus fitted synthetic corruption | 50/50 real and heteroscedastic-Gaussian sRGB augmentation; continuous severity; random impulses |
| `docs/04_solution_strategy.md` | Select checkpoints on real validation composite | Full 50-image validation, rounded PNG-equivalent pixels, EMA checkpoint selection |
| `docs/04_solution_strategy.md` | Overlap tiling, TTA option, optional classical refinement | Feathered tiling; `--tta` averages all eight D4 transforms; optional mild chroma wavelet refinement |
| `docs/04_solution_strategy.md` | Classical reference and ablations | `benchmark_classical.py` reproduces the unchanged starter and evaluates an adaptive wavelet fallback |
| `docs/04_solution_strategy.md` | CPU fallback and reproducibility | PyTorch CPU path retained; model/input/output hashes, fixed seeds, explicit run settings |

## Decisions made from the actual data

- The full-set noise study is exploratory evidence. Validation targets are
  excluded from the new fitted augmentation parameters and from gradient updates.
- The 410 training images show positive signal-dependent variance slopes in all
  RGB channels. `noise_model.json` records the fit and its limitations in sRGB.
- The study found chroma noise slightly **below** luminance noise. Strong chroma
  smoothing is therefore not a default. Its optional mild variant must earn its
  place through validation.
- Fixed-location defects reported in the study are **not** encoded as a lookup
  mask. The inference filter identifies outliers from the current input only.
- Alternative architectures in the literature are alternatives, not mandatory
  ensembles. SCUNet weights are not interchangeable with a NAFNet architecture.
- The documents' score projections and September 11 sprint schedule are planning
  assumptions. Only actual measured results will be reported as achieved.
- The user requested a new experiment output folder and empty reserved
  `competition_data/**/denoised/` directories; that instruction takes precedence
  over the example output paths in the documents.

## Experiment files

```text
scripts/
  prepare_training.py          Fixed split and training-only synthetic-noise fit
  train_nafnet.py              Paired/synthetic fine-tuning, EMA, validation
  check_training_backend.py   Numerical forward/gradient comparison
  benchmark_classical.py      Starter reproduction and classical reference
  compare_variants.py         Fixed validation ablation grid and ranking
  evaluate_run.py             Strict complete-set metrics, JSON/CSV/text
  nafnet/
    augmentation.py           D4 transforms and synthetic corruption
    classical.py              Blind sigma estimation and classical filtering
    losses.py                 Charbonnier, multiscale SSIM surrogate, Haar L1
    pipeline.py               Hybrid inference and geometric self-ensemble
  runs/nafnet_docs_20260918/
    split.json                Immutable 410 train / 50 validation IDs
    noise_model.json          Fit from training targets only
    training_config.json      Created when real training starts
    training_history.jsonl    Full-image validation history
    best_validation.json      Selected checkpoint's held-out scores
    validation/               Ablation predictions (when run)
    public/                   New 460-image predictions (when run)
```

The previous images remain at `scripts/runs/nafnet/public_rerun/`.

## Reproduction commands

Run from the repository root. Inference dependencies remain in
`requirements.txt`; the optional native Windows training runtime is pinned in
`requirements-training-directml.txt`. The prepared Python 3.11 evaluation
environment also contains these training packages, with the official NumPy,
Pillow and scikit-image versions preserved.

```powershell
python scripts/prepare_training.py
.\scripts\.venv-evaluation\Scripts\python.exe scripts/check_training_backend.py
.\scripts\.venv-evaluation\Scripts\python.exe scripts/train_nafnet.py --device directml --benchmark --benchmark-patch 256
.\scripts\.venv-evaluation\Scripts\python.exe scripts/benchmark_classical.py
.\scripts\.venv-evaluation\Scripts\python.exe scripts/train_nafnet.py --device directml --steps 1500 --batch 2 --validate-every 300
```

The preparation command refuses to replace an existing split. Reuse that split
for comparisons. Training refuses to replace an earlier experiment checkpoint.
CUDA training uses mixed precision; the DirectML path uses float32 because CUDA
AMP is not available there. The three-scale SSIM training term is a surrogate,
not the official single-scale evaluation metric.

After training, export the selected checkpoint for the existing fast ONNX
inference environment and generate the new public outputs:

```powershell
.\scripts\.venv-directml\Scripts\python.exe scripts/export_nafnet.py --weights scripts/weights/NAFNet-Mora-docs-best.pth --output scripts/weights/NAFNet-Mora-docs-best-256.onnx
.\scripts\.venv-directml\Scripts\python.exe scripts/denoise.py --noise_dir competition_data/public/noisy --denoised_dir scripts/runs/nafnet_docs_20260918/public --weights scripts/weights/NAFNet-Mora-docs-best.pth --onnx-model scripts/weights/NAFNet-Mora-docs-best-256.onnx --backend directml --preprocess adaptive --threads 2 --memory-format contiguous --report scripts/runs/nafnet_docs_20260918/public_run.json
.\scripts\.venv-evaluation\Scripts\python.exe scripts/evaluate_run.py --pred-dir scripts/runs/nafnet_docs_20260918/public --subset validation --output scripts/runs/nafnet_docs_20260918/validation_metrics.json
.\scripts\.venv-evaluation\Scripts\python.exe evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir scripts/runs/nafnet_docs_20260918/public --gt_dir competition_data/public/ground_truth
```

The final inference options, including whether to add `--tta`, will be selected
from measured validation ablations. The 460-image metric includes the training
images and must be labeled as a full-public-set result. The held-out 50-image
metric is the checkpoint-selection result, not an untouched hidden test score.

## Verification and execution status

- All 21 offline correctness tests pass in both the training/evaluation and
  final inference environments, including flat-chroma and D4 regressions.
- Native DirectML's zero-size padding produced zero images. The network now
  skips that no-op. Verified CPU/GPU maximum forward difference: 0.00000244;
  maximum gradient difference: 0.00000008.
- The 256-pixel training benchmark passed: five updates completed in 8.03 seconds
  with finite losses. Benchmark weights were not saved as trained checkpoints.
- The earlier pretrained outputs score 24.6077 dB PSNR, 0.796498 SSIM and
  0.32981700 composite on the fixed 50-image validation split. These are the
  before-training reference, not new trained results.
- The unchanged starter baseline was reproduced on all 460 images: PSNR
  24.3360 dB, SSIM 0.671659, composite 0.26329211, matching its published result.
  Its 50-image validation composite is 0.24952984. The adaptive wavelet fallback
  achieves 0.33884956 on that same validation set.
- Full fine-tuning has started after additional disk space was made available.
  Ablations, final output verification and CPU timing follow checkpoint selection.
- Native DirectML also required regular no-grad tensors for tiled validation.
  A full-image smoke test now passes; training saves portable model, EMA,
  optimizer and random-generator state before validation for recovery.
- Cleanup removed 111 superseded trial/duplicate images (206,824,147 bytes).
  The completed pretrained 460-image run and all original data are preserved;
  historical trial records are archived under `runs/archive/pretrained_trials/`.
- No official submission, collaborator invitation, dataset redistribution,
  report publication or model/code freeze has been performed.

Native DirectML training setup follows the
[Microsoft instructions](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows).
