# Mora SP Cup denoising

The current implementation is a **ten-channel, noise-conditioned NAFNet-32 with
gated Haar-wavelet corrections at the two finest encoder skip scales**.
See [WAVELET_NAFNET.md](WAVELET_NAFNET.md) for architecture, training, recovery,
dependencies and reproducibility details.

The 5,000-step hybrid experiment is running. Check
`runs/wavelet_nafnet_20260919/job_status.json` and `training.log` for actual status.
`training_finished.json` confirms completed training; `verification.json` and
`RESULTS.md` are produced after final image generation and evaluation.

## Data and results

- Training: 410 public noisy/ground-truth pairs; fixed validation: 50 pairs.
- Preliminary test inputs: `competition_data/submissions/noisy/461_noise.png` through `480_noise.png`.
- New public outputs: `scripts/runs/wavelet_nafnet_20260919/public/001.png` through `460.png`.
- New test outputs: `scripts/runs/wavelet_nafnet_20260919/submissions/461.png` through `480.png`.
- Test ZIP: `scripts/runs/wavelet_nafnet_20260919/test_outputs_461-480.zip`.
- Final metrics and verification: `scripts/runs/wavelet_nafnet_20260919/`.

Original noisy images and ground truth remain unchanged. Reserved
`competition_data/**/denoised/` directories remain empty as requested. Old
pretrained output images are removed only after new outputs pass verification;
the replacement manifest records their hashes and new locations.

Test ground truth is hidden, so test PSNR/SSIM cannot be measured locally.
Final reports distinguish the 50 validation images from all 460 public images,
which include training data. No 34-35 dB result is assumed.

## Run after training

Run from the repository root. Inference uses local weights and works offline.
Default weights: `scripts/weights/Wavelet-NAFNet-Mora-best.pth` (includes the
training-only noise calibration). Exact hashes appear in the final results.

```powershell
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir scripts/runs/my_outputs --device cpu --threads 8
```

Windows GPU inference with the matching ONNX export:

```powershell
.\scripts\.venv-directml\Scripts\python.exe scripts/denoise.py --input_dir competition_data/submissions/noisy --output_dir scripts/runs/my_gpu_outputs --backend directml
```

Each output is named `<id>.png`. Reports are saved beside image directories.
Use a fresh output folder, or `--resume` to reuse previously verified outputs
from the same model and inference settings. CPU requires `requirements.txt`;
ONNX GPU additionally requires `requirements-directml.txt`.

## Train and recover

```powershell
.\scripts\.venv-evaluation\Scripts\python.exe -u scripts/run_wavelet_experiment.py
```

The runner performs training and then automatic export, test/public inference,
official evaluation, CPU fallback checks, output verification and replacement.
Do not start another copy while it is running.

Resume an interrupted run:

```powershell
.\scripts\.venv-evaluation\Scripts\python.exe -u scripts/run_wavelet_experiment.py --resume
```

Recovery checkpoints are saved every 100 steps; all 50 validation images are
scored every 500 steps. The best EMA checkpoint is selected by official
composite. Validation PSNR, SSIM and composite are recorded in
`training_history.jsonl` and printed in `training.log`.

## Previous measured result

The earlier RGB NAFNet completed 1,500 fine-tuning steps on the same split:
**29.4123 dB PSNR, 0.894395 SSIM, 0.55905854 composite** on 50 validation images.
The matching starter baseline scored **24.0066 dB, 0.657887, 0.24952984**.
That checkpoint is retained as `weights/NAFNet-Mora-docs-best.pth`.

[DOCS_IMPLEMENTATION.md](DOCS_IMPLEMENTATION.md) records the first training
experiment; [PRETRAINED_RUN.md](PRETRAINED_RUN.md) records the original SIDD-only
inference run. `run_nafnet.py` remains the legacy RGB-model inference utility.

All custom implementation lives in `scripts/`. The baseline, evaluator and
project knowledge-base files are unchanged. The NAFNet MIT license is retained
in `licenses/NAFNet-LICENSE.txt`.
