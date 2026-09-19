# Team Voxel — Mora SP Cup 2026 Preliminary Submission

Blind low-light sRGB denoising with a NAFNet restoration core fine-tuned on **414 of the
460 public pairs** — the remaining **46 are held out for validation and are never trained
on**, nor used by the synthetic-noise generator — against a noise model fitted to the
measured degradation of this dataset.

> **Important — Submission Freeze**
>
> After the preliminary-round submission deadline, no new commits may be
> made to the submitted private GitHub repository.
>
> The Git commit SHA stated in this README and recorded by the organizers
> will be treated as the official submitted code version.
>
> If external model weights/checkpoints are used, their Google Drive link,
> expected local path, and SHA-256 checksum must also be stated here.
> The submitted model/checkpoint must not be modified or replaced after
> the deadline.

## How to run

### Requirements

Python 3.11 (tested on 3.11.16, Windows 11). Install with:

```bash
pip install -r scripts/requirements.txt
```

Inference needs only `torch`, `numpy` and `Pillow`. `torch` is pinned to the **CPU**
build: the submitted images were produced on CPU and `--device cpu` is the default, so
**no GPU and no CUDA installation are required** to reproduce the submission. `scipy`
and `scikit-image` are used only by an optional diagnostic and by the offline classical
fallback; both are imported lazily, so their absence cannot break the main path.

No network access is required or attempted at any point. The model weights are in the
repository at `scripts/weights/voxel_nafnet.pth`.

### Usage

```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy \
                          --denoised_dir competition_data/submissions/denoised
```

`--input_dir` / `--output_dir` are accepted as aliases of `--noise_dir` /
`--denoised_dir`, so either calling convention works.

**This exact command, with all other flags left at their defaults, produced the
submitted images.**

Useful optional flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device {cpu,cuda,auto}` | `cpu` | `cpu` is the submitted configuration. `cuda` is faster but CPU and GPU kernels differ numerically, so it is *not* what the submission was generated with. |
| `--tile N` | `0` | `0` runs the whole image in one pass (992 is a multiple of 32, so no padding is needed). `>0` enables overlap-tiled inference; this also engages automatically if memory runs out. |
| `--tta {1,4,8}` | `1` | D4 self-ensemble. Off by default on measured evidence: `+0.0031` composite for `8.67×` the CPU runtime (84.5 s/image instead of 9.74 s). `--tta 4` is `+0.0023` for `4.37×`. |
| `--prepass` | off | Enables the classical gated adaptive-median impulse repair before the network. Off by default because it was **measured** to cost composite: a gate sweep (k = 4.5/6/8/12) showed the score rising monotonically as fewer pixels are filtered. It is always applied in `--mode classical`, where it is worth +0.0152. |
| `--mode {net,classical}` | `net` | `classical` runs the self-contained fallback denoiser with no model. |
| `--num_threads N` | `min(8, cpu_count)` | CPU thread count, pinned for reproducibility. |

### Output

One RGB PNG per input, `992×992`, 8-bit, written as `<id>.png` with the `_noise`
suffix stripped (`461_noise.png` → `461.png`).

### Reproducibility

The pipeline contains no randomness at inference. Outputs are quantised with
`floor(255·x + 0.5)` and written with fixed PNG encoder settings, so re-running the
command above in this environment reproduces the submitted files **byte for byte**.
Across different CPUs or thread counts the results are metrically identical
(ΔPSNR < 0.001 dB) though the last bit of a few pixels may differ; the documented
environment is the one the submission was made in.

Verify with:

```bash
python scripts/tools/preflight.py
```

which re-runs the pipeline in a clean subprocess, compares SHA-256 against the
submitted PNGs, and performs a rename test (`461_noise.png` copied to
`999_noise.png` must yield a byte-identical output) demonstrating that no per-image
logic exists.

## Repository layout

```
scripts/
  denoise.py            the submission entry point (CLI + per-image loop only)
  common.py             shared paths, IO, and metrics verified against the official scorer
  spcup/                the inference package
    determinism.py      seeds and torch flags, configured before torch is imported
    io_utils.py         image IO, deterministic PNG writing, filename handling
    classical.py        gated adaptive median, VST, classical fallback denoiser
    nafnet.py           NAFNet architecture (self-contained) + TLC
    infer.py            model loading, whole-image / tiled forward, D4 TTA
    pipeline.py         denoise_image() -- the single definition of the pipeline
  train/                training code (not used at inference)
  tools/                validation, ablation, preflight, submission packaging
  weights/              model checkpoint
  report/               report source
```

`spcup/pipeline.py::denoise_image` is the only definition of what happens to an image;
the entry point, the validator and the ablation runner all call it, so the pipeline that
was measured and the pipeline that was submitted cannot differ.

## Reproducing the training

```bash
python scripts/train/prepare_data.py                    # pack data, build the split
python scripts/tools/check_metric.py                    # assert metrics match the official scorer
python scripts/tools/check_noise_model.py               # assert the synthetic noise matches measurement
python scripts/train/fetch_weights.py --model nafnet_sidd_w32
python scripts/train/train.py --run run1 --init scripts/weights/NAFNet-SIDD-width32.pth \
    --arch w32 --iters 48000 --patch_schedule 0:256x4 --lr 2e-4 --lr_min 1e-6 \
    --warmup 500 --wd 1e-3 --ema 0.9995 --ema_start 2000 --p_synth 0.4 --seed 1337
python scripts/tools/select_checkpoint.py --run run1     # re-scores every checkpoint
python scripts/train/export_model.py --ckpt scripts/_runs/run1/hour_04.pth --which model
```

The exported model is whichever weight set `select_checkpoint.py` ranks first on the
held-out split, **not** whichever the training loop labelled "best": validation during
training alternated between a 23-image subset and the full 46, and those scores are not
comparable. For this run the winner was `hour_04.pth:model` (step 38,928, composite
0.643318); the top six candidates spanned only 0.0012.

Validation and ablations:

```bash
python scripts/tools/validate.py --weights scripts/weights/voxel_nafnet.pth --tag ours
python scripts/tools/ablate.py --emit latex
```

## Model provenance

The restoration core is initialised from the publicly released **NAFNet-SIDD-width32**
checkpoint (Chen et al., ECCV 2022; `megvii-research/NAFNet`, MIT/Apache-2.0) and then
fine-tuned on the competition's own public data (414 training ids). The initialisation was obtained
from the mirror `https://huggingface.co/nyanko7/nafnet-models` (the upstream project
distributes only Google Drive links) and verified by exact file size
(116,861,841 bytes), parameter count (29,159,715) and a `strict=True` load.

## Official Submission Information

Git Commit SHA:
`see below` — the official submitted version is **this repository at `HEAD`**, i.e. the
output of `git rev-parse HEAD` on the submitted private repository. A commit cannot
contain its own hash, so the literal SHA is supplied to the organizers on the submission
form; `git log -1` on this repository confirms it, and no commits follow it.

Model Checkpoint:
`voxel_nafnet.pth`

Model Drive Link:
`N/A — committed to this repository at scripts/weights/voxel_nafnet.pth`

Expected Model Path:
`scripts/weights/voxel_nafnet.pth`

Model SHA-256:
`6acdff27227bbde63386f37fb07f06d3db08656bb76b38ae3f3f2c240c903ae8`
