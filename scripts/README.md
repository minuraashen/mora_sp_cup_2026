# NAFNet denoising

The new **docs-guided training and hybrid experiment** is described in
[DOCS_IMPLEMENTATION.md](DOCS_IMPLEMENTATION.md). Its fixed train/validation
split and training code are prepared; full fine-tuning is now running after
additional disk space was made available. The measurements below belong to the earlier pretrained
run and must not be presented as results of the trained redo.

Run the authors' pretrained **NAFNet-SIDD-width32** model on Mora SP Cup RGB
images. The model has not been trained or fine-tuned on the competition data.
The starter baseline and official evaluator are unchanged.

## Current output locations

- Original inputs: `competition_data/public/noisy/001_noise.png ... 460_noise.png`.
- Public evaluation targets: `competition_data/public/ground_truth/001.png ... 460.png`.
- Full rerun outputs: **`scripts/runs/nafnet/public_rerun/001.png ... 460.png`**.
- Full rerun settings, timings and hashes: `scripts/runs/public_nafnet_rerun.json`.
- Official evaluation results: `scripts/runs/public_nafnet_metrics.txt`.

Both `competition_data/public/denoised/` and
`competition_data/submissions/denoised/` are reserved and remain empty.
Superseded CPU trial images were removed during cleanup; their small historical
records are retained under `scripts/runs/archive/pretrained_trials/`.
Noisy inputs are never renamed or overwritten.

## Code layout

```text
scripts/
  run_nafnet.py              Experimental inference CLI
  denoise.py                 CLI using the required <id>.png output names
  setup_nafnet.py            Verified checkpoint download
  export_nafnet.py           Optional ONNX export of the same checkpoint
  download_public_data.py   Verified public noisy/ground-truth downloads
  verify_run.py              Verify saved outputs and unchanged original inputs
  requirements.txt          PyTorch inference dependencies
  requirements-directml.txt Optional Windows GPU dependencies
  nafnet/
    model.py                Checkpoint-compatible NAFNet architecture
    inference.py            Weight loading, RGB conversion and tiled inference
    directml.py             Optional ONNX Runtime GPU adapter
  tests/                    Offline correctness and output-safety checks
  licenses/                 Upstream license and attribution
  weights/                  Local checkpoints/exports, ignored by Git
  runs/                     Local images, manifests and reports, ignored by Git
```

## 1. Prepare dependencies and inputs

All commands run from the repository root. Core inference requires Python
3.11+, PyTorch, NumPy and Pillow:

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/setup_nafnet.py
```

Setup downloads approximately 117 MB from the authors' published Drive link,
checks its SHA-256 and reuses an existing verified file. Inference runs offline
and fails if the checkpoint is absent; it never substitutes random weights.

Place the original competition files according to the root README:

```text
competition_data/
  public/
    ground_truth/    001.png ... 460.png
    noisy/           001_noise.png ... 460_noise.png
    denoised/        Reserved for finalized results
  submissions/
    noisy/           461_noise.png ... 480_noise.png
    denoised/        Reserved for finalized results
```

The verified Drive manifests saved during this session can reproduce the
public downloads:

```powershell
python scripts/download_public_data.py --kind noisy --workers 12
python scripts/download_public_data.py --kind ground_truth --workers 12
```

The downloader verifies names, byte sizes, RGB PNG format and 992 x 992
dimensions before exposing each final filename. Valid existing originals are
reused. It records SHA-256 hashes in `scripts/runs/public_*_download.json`.
Manifests and images are local ignored data; a fresh checkout needs the
original dataset or newly verified Drive listings. Do not redistribute the
competition dataset. Preliminary and hidden data are not fetched by this tool.

## 2. Run NAFNet

CPU, with evaluator-compatible names in a separate experimental folder:

```powershell
python scripts/denoise.py --noise_dir competition_data/public/noisy --denoised_dir scripts/runs/nafnet/public_cpu --device cpu --report scripts/runs/public_cpu.json
```

For a one-image trial with a descriptive suffix:

```powershell
python scripts/run_nafnet.py --input competition_data/public/noisy/001_noise.png --output_dir scripts/runs/nafnet/trial
```

That trial writes `001_nafnet.png`. Folder mode selects `*_noise.png` by
default, and `--limit 3` restricts a trial to three inputs.

Existing outputs cause a clear error. Add `--overwrite` only to deliberately
regenerate output files. The runner rejects destinations that overwrite inputs
or weights. Output is RGB, 8-bit PNG at the original resolution. No brightness
adjustment, image-specific lookup or resizing is applied.

### Optional fast Windows GPU run

The full rerun uses ONNX Runtime DirectML, which executes the same float32
NAFNet graph on a Windows GPU. The optional runtime is much smaller than a
CUDA PyTorch installation. Core PyTorch inference remains available on CPU or
CUDA with a compatible PyTorch build.

```powershell
python -m venv --system-site-packages scripts/.venv-directml
.\scripts\.venv-directml\Scripts\python.exe -m pip install -r scripts/requirements-directml.txt
.\scripts\.venv-directml\Scripts\python.exe scripts/export_nafnet.py
.\scripts\.venv-directml\Scripts\python.exe scripts/denoise.py --noise_dir competition_data/public/noisy --denoised_dir scripts/runs/nafnet/public_rerun --backend directml --threads 2 --memory-format contiguous --report scripts/runs/public_nafnet_rerun.json
```

Skip the export command when that verified export already exists. The export
contains its source-checkpoint hash, architecture and tile size. The runner
checks these before inference. DirectML does not silently fall back to a
CPU-only session when GPU initialization fails.

This export uses fixed 256 x 256 tiles. DirectML inputs must be at least that
size in both dimensions; use the PyTorch backend for smaller images. On three
real 256 x 256 input tiles, the maximum absolute float difference between
DirectML and PyTorch was below 0.0000005.

## 3. Compute the official metrics

Use the exact dependencies in `evaluation/requirements.txt` with Python 3.11.
The isolated evaluation environment prepared for this run can be reproduced:

```powershell
py -3.11 -m venv scripts/.venv-evaluation
.\scripts\.venv-evaluation\Scripts\python.exe -m pip install -r evaluation/requirements.txt
.\scripts\.venv-evaluation\Scripts\python.exe evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir scripts/runs/nafnet/public_rerun --gt_dir competition_data/public/ground_truth
```

Verify that all 460 predictions and targets exist before scoring: the supplied
evaluator can otherwise report a mean over a partial set.

The unchanged evaluator reports mean PSNR, SSIM, Delta PSNR, Delta SSIM and
the mean per-image composite:

```text
composite = 0.6 * clip(Delta PSNR / 15, 0, 1) + 0.4 * max(Delta SSIM, 0)
```

### Completed public rerun: 18 September 2026

The unchanged official evaluator scored all **460/460 images** using the exact
packages in `evaluation/requirements.txt`. These are public-set development
results from pretrained SIDD weights, without competition-specific fine-tuning.

| Official metric | NAFNet rerun | Published starter baseline |
|---|---:|---:|
| Mean PSNR | 24.7258 dB | 24.3360 dB |
| Mean SSIM | 0.802140 | 0.671659 |
| Mean Delta PSNR | +4.9001 dB | +4.5103 dB |
| Mean Delta SSIM | +0.337756 | +0.207275 |
| Composite score | **0.33179406** | 0.26329211 |

Delta metrics measure improvement over the original noisy inputs. The composite
improves by **0.06850195** over the baseline published in `baseline/README.md`;
that baseline was not rerun during this experiment.

GPU inference and image saving took **288.09 seconds (4 minutes 48 seconds)**,
excluding setup, downloads and scoring. All 460 outputs passed format, dimensions
and hash verification; all original noisy inputs retained their download hashes.
All 13 unit tests passed. A repeated full-image prediction was byte-identical.

The evaluator's exact output is saved in `scripts/runs/public_nafnet_metrics.txt`;
inference settings and per-image timings/hashes are in
`scripts/runs/public_nafnet_rerun.json`.

## Settings and verification

| Option | Default | Purpose |
|---|---|---|
| `--backend` | `pytorch` | Native PyTorch or optional `directml` |
| `--device` | `auto` | PyTorch CPU/CUDA selection |
| `--tile` | `256` | Tile edge; 0 means full-image PyTorch inference |
| `--overlap` | `32` | Feather overlapping tiles |
| `--threads` | Up to 8 | CPU worker threads |
| `--memory-format` | `channels_last` | Tensor storage layout; RGB order is unchanged |
| `--preprocess` | `none` | Optional blind adaptive-median impulse repair |
| `--tta` | Off | Average eight D4 transforms; increases inference work about eightfold |
| `--chroma-refine` | `0` | Optional mild wavelet chroma refinement, chosen on validation |
| `--weights` | SIDD-width32 checkpoint | Local compatible model weights |
| `--report` | None | JSON settings, versions, timings and file hashes |

Both the handbook's `--noise_dir / --denoised_dir` flags and the baseline's
`--input_dir / --output_dir` aliases work. The competition entry point
requires a separate output directory.

Only one tile occupies accelerator memory at a time. This is standard NAFNet,
without the optional NAFNetLocal/TLC conversion. Tiled channel attention differs
from full-image attention; keep tile settings fixed when comparing results.

PyTorch uses a fixed seed, float32 and deterministic operations. DirectML
executes an equivalent exported float32 graph. Identical files across devices,
backends or package versions are not guaranteed. Reports record the backend,
versions, checkpoint/export hashes and output hashes for reproducibility.

```powershell
python -m unittest discover -s scripts/tests -v
python scripts/verify_run.py --report scripts/runs/public_nafnet_rerun.json --download-report scripts/runs/public_noisy_download.json
```

Tests cover normalization, odd/narrow tile coverage, RGB conversion, missing
weights, non-finite outputs, filename collisions, manifest paths and input
protection. Full-image format and hash checks also verify the completed run.

## Model provenance and submission freeze

- Upstream: <https://github.com/megvii-research/NAFNet>.
- Paper: Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022.
- Model: `NAFNet-SIDD-width32.pth`.
- Drive: <https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view>.
- Local path: `scripts/weights/NAFNet-SIDD-width32.pth`.
- SHA-256: `89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4`.
- Git commit SHA: **not frozen / not submitted**.

The model code is adapted from the upstream MIT implementation with native
PyTorch layer normalization. The upstream license is retained under
`licenses/NAFNet-LICENSE.txt`.

Record the final commit SHA and freeze the model before official submission.
The submitted repository must not receive post-deadline commits, and submitted
weights must not be replaced. No submission or publication has been performed.
