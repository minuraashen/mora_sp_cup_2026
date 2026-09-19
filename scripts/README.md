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
logic exists. It also checks the freeze record below is not stale.

### Building and freezing the submission bundle

```bash
python scripts/tools/make_submission.py   # -> Voxel.zip, exactly the 20 PNGs
python scripts/tools/freeze.py            # -> the freeze record, printed + recorded
```

`freeze.py` writes `scripts/artifacts/submission_manifest.json`, pinning the SHA-256
of every submitted PNG, of `Voxel.zip` and of the checkpoint, and prints the block to
paste into the organizers' submission form. It exits non-zero if the working tree is
dirty, because the commit SHA it prints would then not describe what is on disk.
Run it as the **last** step, after the final commit, and report the SHA it prints.

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
  tools/                validation, ablation, preflight, submission packaging, freeze
  weights/              model checkpoint (voxel_nafnet.pth, committed -- no download)
  artifacts/            provenance: split, checkpoint selection, freeze manifest
  analysis/             the measured noise characterisation the noise model is fitted to
  docs/                 our working analysis of the brief, the metric and the strategy
  report/               report source (main.tex + figs; build the PDF on Overleaf)
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

<!-- FROZEN-COMMIT-BEGIN -->
`0000000000000000000000000000000000000000`
<!-- FROZEN-COMMIT-END -->

This is the frozen preliminary-round code submission.

*On the one-commit offset, stated plainly so nothing looks inconsistent:* a git commit
is named by a hash taken over its own contents, so no file inside a commit can contain
that commit's own SHA — it is a hash fixed point, not an oversight. The SHA above is
therefore the last **content** commit, and the current `HEAD` is the commit that wrote
this line and nothing else. `scripts/tools/freeze.py` verifies exactly that: it exits
non-zero unless `HEAD` is a direct child of the SHA above whose only change is this
block. Both values are printed by

```bash
python scripts/tools/freeze.py
```

and both are given on the submission form, so the code identified either way is
identical. Re-stamp after any further commit with `python scripts/tools/freeze.py
--stamp`, then commit.

Model Checkpoint:
`voxel_nafnet.pth`

Model Drive Link:
`N/A` — the checkpoint is **committed to this repository**, so nothing has to be
downloaded and there is no external link to go stale. It is 58,525,221 bytes
(fp16 on disk), well inside GitHub's 100 MB per-file limit, and is stored as an
ordinary git blob — **no Git LFS**, so a plain `git clone` is sufficient.

Expected Model Path:
`scripts/weights/voxel_nafnet.pth`

Nothing needs to be moved or unpacked after cloning: `denoise.py` resolves this path
relative to its own location and loads it by default. Only if the checkpoint were
obtained separately (it should not need to be) would it go at exactly that path,
under that exact filename, before running the command in *Usage* above.

Model SHA-256:
`6acdff27227bbde63386f37fb07f06d3db08656bb76b38ae3f3f2c240c903ae8`

Verify after cloning with `sha256sum scripts/weights/voxel_nafnet.pth` (or
`certutil -hashfile scripts\weights\voxel_nafnet.pth SHA256` on Windows).

Submission bundle:
`Voxel.zip` — exactly the 20 PNGs `461.png`–`480.png`, flat, no directory prefix.
Rebuild it with `python scripts/tools/make_submission.py`; its SHA-256 and the
SHA-256 of each individual image are recorded in
`scripts/artifacts/submission_manifest.json`.
