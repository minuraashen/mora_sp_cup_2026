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

## Frozen model downloads and checksums

The model files are stored in this Google Drive folder because each complete
checkpoint is larger than GitHub's regular-file limit:

<https://drive.google.com/drive/folders/17kgPjzU7mZB9a68rZt4CCm7TQoWH12Ej?usp=sharing>

### Competition CLI checkpoint

The default `scripts/denoise.py` entry point requires:

```text
Filename:      Wavelet-NAFNet-Mora-best.pth
Expected path: scripts/weights/Wavelet-NAFNet-Mora-best.pth
Size:          116892024 bytes
SHA-256:       1ddd26c8cfd66d1a9f50fdcd96f6e0ba57caa9354bc0d7dae863692644907c31
```

Download both Drive parts into the repository root and reconstruct the frozen
checkpoint without modifying either part:

```powershell
New-Item -ItemType Directory -Force scripts/weights | Out-Null
python -c "from pathlib import Path; ps=[Path('Wavelet-NAFNet-Mora-best.pth.part01'),Path('Wavelet-NAFNet-Mora-best.pth.part02')]; o=Path('scripts/weights/Wavelet-NAFNet-Mora-best.pth'); f=o.open('wb'); [f.write(p.read_bytes()) for p in ps]; f.close()"
Get-FileHash -Algorithm SHA256 scripts/weights/Wavelet-NAFNet-Mora-best.pth
```

Part checksums:

```text
bdf6c359b1fd16ea1689d410321aeac3f6f74161c96759e34709630e47d895d7  Wavelet-NAFNet-Mora-best.pth.part01
4a2211ed48fe416409cea10903b1ad6ef2ffb86df53cea82533239e8bb7126c7  Wavelet-NAFNet-Mora-best.pth.part02
```

### Final NCW experiment checkpoint

The best-composite checkpoint documented in the final NCW report is supplied
separately for exact experiment reproduction:

```text
Filename:      ncw_1h_best_composite.pth
Expected path: experiments/ncw_1hour_33db_v5/checkpoints/ncw_1h_best_composite.pth
Size:          177937000 bytes
SHA-256:       30bbb067337916eaf938e1f3349ea347061f1a44e106ac505407c35eac42035f
```

Reconstruct it after downloading both Drive parts into the repository root:

```powershell
New-Item -ItemType Directory -Force experiments/ncw_1hour_33db_v5/checkpoints | Out-Null
python -c "from pathlib import Path; ps=[Path('ncw_1h_best_composite.pth.part01'),Path('ncw_1h_best_composite.pth.part02')]; o=Path('experiments/ncw_1hour_33db_v5/checkpoints/ncw_1h_best_composite.pth'); f=o.open('wb'); [f.write(p.read_bytes()) for p in ps]; f.close()"
Get-FileHash -Algorithm SHA256 experiments/ncw_1hour_33db_v5/checkpoints/ncw_1h_best_composite.pth
```

Part checksums:

```text
dc41cdbee41d6971765972fb4e6423a2bfa672df0591595bf4835f526fd62d40  ncw_1h_best_composite.pth.part01
391e813c7776913aac28e62a6718e8cc41cf5ace1b96e9113a23e8a68e994684  ncw_1h_best_composite.pth.part02
```

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
