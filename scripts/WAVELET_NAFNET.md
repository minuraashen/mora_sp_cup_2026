# Noise-conditioned NAFNet with gated Haar skips

This implementation follows the requested ten-channel architecture. It starts
from the competition-fine-tuned width-32 model and preserves the fixed 410/50
split. The current experiment is `runs/wavelet_nafnet_20260919`.

Training status and results are machine-recorded, not inferred from file names:

- `job_status.json`: running, failed, or complete for the entire workflow.
- `training.log`: live optimizer progress and validation summaries.
- `training_history.jsonl`: PSNR, SSIM, deltas and official composite every 500 steps.
- `training_state.pth`: resumable state every 100 steps and around validation.
- `best_validation.json`: all 50 per-image scores for the selected checkpoint.
- `training_finished.json`: actual completed steps and selected checkpoint hash.
- `RESULTS.md`, `public_metrics.json`, `public_metrics.csv`: final exported-image metrics.
- `verification.json`: source checksums, output formats, repeatability and CPU checks.

## Architecture

The ten channels are original noisy RGB (3), softly repaired RGB (3), estimated
RGB standard-deviation maps (3), and maximum per-channel impulse confidence (1).
All preprocessing uses the current noisy image. The variance calibration comes
only from the 410 training pairs; no fixed defect coordinates are stored.

The input convolution expands from 3 to 10 channels. Original RGB weights are
copied from the old model; added weights start at zero. The backbone remains
width 32, encoder depths 2/2/4/8, 12 middle blocks, decoder depths 2/2/2/2.

At the two finest encoder scales, orthonormal Haar analysis separates LL and
three detail bands. LL is preserved. A learned map from RGB noise estimates
predicts nonnegative feature-domain soft thresholds for the three detail bands.
Inverse Haar reconstructs the features. Each skip is corrected once:

`corrected = original + tanh(gate) * (reconstructed - original)`

Gates start at zero, preserving the backbone initially. Noise maps condition
learned thresholds; they are not asserted to be exact feature-space variances.
Corrected skips feed the matching decoder scale; deeper skips remain original.
The RGB head predicts a signed correction added to original noisy RGB.
There is no variance-stabilizing transform, exposure change or external output lookup.

## Training and recovery

The run uses 5,000 additional steps, 256-pixel crops, batch 1, 75% real pairs and
25% fitted synthetic corruption, D4 augmentation, AdamW (2e-5 cosine decay),
gradient clipping and EMA (0.99). Loss remains Charbonnier + 0.1 structural
surrogate + 0.05 Haar detail L1. Validation uses the unchanged official scorer
on rounded PNG-equivalent RGB, not the training SSIM surrogate.

Native DirectML uses the existing Python 3.11 environment. The CNN runs on GPU;
one AdamW operation falls back to CPU. CPU/DirectML forward and gradient parity
were tested before training. Do not launch two training jobs into the same run.

From the repository root:

```powershell
.\scripts\.venv-evaluation\Scripts\python.exe -u scripts/run_wavelet_experiment.py
```

To recover an interrupted run (same configuration):

```powershell
.\scripts\.venv-evaluation\Scripts\python.exe -u scripts/run_wavelet_experiment.py --resume
```

Recovery contains model, EMA, AdamW moments, RNG state, step, configuration,
elapsed time, best score and pending-validation status. Writes use a temporary
file followed by replacement. Validation occurs at 500/1000/.../5000. Recovery
saves at intervening 100-step intervals have no new measured PSNR.

The best checkpoint is `weights/Wavelet-NAFNet-Mora-best.pth`; it embeds the fitted
noise model for offline inference. Best weights are retained if a later score
regresses. Original backbone checkpoints are preserved for comparison/recovery.

## Inference and outputs

After successful training, the runner automatically exports ONNX, generates
test and public images, scores public images, checks CPU fallback and a repeated
GPU output, and verifies all 940 original image hashes. It then replaces old
pretrained prediction images with the verified new public outputs, recording a
replacement manifest. Earlier numerical results remain historical references.

Final output locations:

```text
runs/wavelet_nafnet_20260919/
  public/                 001.png ... 460.png
  submissions/            461.png ... 480.png
  test_outputs_461-480.zip exactly the 20 test output PNGs
```

The reserved `competition_data/**/denoised/` folders remain empty as requested.
Only noisy test inputs are downloaded to `competition_data/submissions/noisy/`.
Test ground truth is hidden; test PSNR/SSIM cannot be calculated locally.
Full-public metrics include training images; validation metrics use the same
50 images that selected checkpoints and are not an untouched final test score.

GPU inference after training/export:

```powershell
.\scripts\.venv-directml\Scripts\python.exe scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir scripts/runs/my_test_outputs --backend directml
```

CPU inference uses the same entry point and trained checkpoint:

```powershell
python scripts/denoise.py --input_dir competition_data/submissions/noisy --output_dir scripts/runs/my_cpu_outputs --device cpu --threads 8
```

Inference is offline. Install `scripts/requirements.txt` for CPU; Windows ONNX
DirectML additionally uses `scripts/requirements-directml.txt`. The tested local
Python 3.14 DirectML environment contains both sets of inference dependencies.
No GPU is required for the default CPU fallback. Default inference uses
256-pixel tiles, 32-pixel overlap and deterministic feathering. Preprocessing is
performed on complete images before tiling. Backend float differences can change
rounded pixels; reproduce a run with its recorded backend and model checksums.

Use `--resume` to continue interrupted inference. Existing outputs are reused
only if input/output hashes and model/inference provenance match the journal.
To repeat only the post-training workflow after a failure:

```powershell
.\scripts\.venv-evaluation\Scripts\python.exe -u scripts/run_wavelet_experiment.py --finish-only
```

The prior RGB-only inference utility remains available as `run_nafnet.py`.
Neither a 34-35 dB score nor an improvement from wavelet corrections is assumed;
the final metrics must establish the result.
