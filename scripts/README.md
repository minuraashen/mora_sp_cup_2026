# NCW-NAFNet — Team Voxel

Blind denoising of 992x992 8-bit sRGB low-light photographs.

**NCW-NAFNet** is a NAFNet-w32 restoration backbone wrapped in noise
conditioning driven by classical signal processing: a fitted Poisson-Gaussian
variance model, an exact 3x3 median impulse detector, and a four-level
orthonormal Haar pyramid. Every added path is scaled by a scalar gate
initialised to zero, so the wrapper begins as an exact identity over the
backbone and only departs from it where training demonstrably helps.

---

## Quick start

```bash
pip install -r scripts/requirements.txt
python scripts/denoise.py --noise_dir competition_data/submissions/noisy \
                          --denoised_dir competition_data/submissions/denoised
```

Reads every `<id>_noise.png` and writes `<id>.png`. Runs on CPU, makes no
network requests, and is deterministic — the same input always produces
byte-identical output.

`--input_dir` / `--output_dir` are accepted as aliases for `--noise_dir` /
`--denoised_dir`, matching the provided baseline's flag names.

| Flag | Default | Meaning |
|---|---|---|
| `--noise_dir` | `competition_data/submissions/noisy` | input directory |
| `--denoised_dir` | `competition_data/submissions/denoised` | output directory |
| `--weights` | `scripts/weights/ncw_1h_best_composite_fp16.pth` | checkpoint |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--tile` | `256` | tile size; `0` forwards the whole image |
| `--threads` | `0` | torch CPU threads; `0` keeps the default |

## Required model file

The checkpoint ships in the repository, so no download is needed:

```text
Filename:      ncw_1h_best_composite_fp16.pth
Expected path: scripts/weights/ncw_1h_best_composite_fp16.pth
Size:          59585807 bytes
SHA-256:       be1515a1afe066bbd445b6a5b73f04d0141d3ed08e22eaa4cb9fd145df86155b
```

It is loaded by `ncw.inference.load_checkpoint`, which reads the resolved
configuration embedded in the checkpoint, rebuilds the model from it, and loads
the weights with `strict=True` — all 766 tensors must match, with no missing,
unexpected or shape-mismatched key. Weights are stored in float16 and upcast to
float32 at load; all arithmetic is float32.

The float32 original is identical in architecture and is available on Google
Drive for verification. It is not committed because it exceeds GitHub's 100 MB
regular-file limit:

```text
Filename:      ncw_1h_best_composite.pth
Expected path: scripts/weights/ncw_1h_best_composite.pth
Size:          177937000 bytes
SHA-256:       30bbb067337916eaf938e1f3349ea347061f1a44e106ac505407c35eac42035f
Drive:         https://drive.google.com/drive/folders/17kgPjzU7mZB9a68rZt4CCm7TQoWH12Ej?usp=sharing
```

Download the two parts into the repository root and reassemble:

```powershell
New-Item -ItemType Directory -Force scripts/weights | Out-Null
python -c "from pathlib import Path; ps=[Path('ncw_1h_best_composite.pth.part01'),Path('ncw_1h_best_composite.pth.part02')]; o=Path('scripts/weights/ncw_1h_best_composite.pth'); f=o.open('wb'); [f.write(p.read_bytes()) for p in ps]; f.close()"
Get-FileHash -Algorithm SHA256 scripts/weights/ncw_1h_best_composite.pth
```

Then pass `--weights scripts/weights/ncw_1h_best_composite.pth`.

## Results

Measured on the 46 held-out public images with the unmodified
`evaluation/evaluate.py`:

| Metric | Value |
|---|---|
| PSNR | **31.3054** dB |
| SSIM | **0.919342** |
| Mean improvement over noisy | **+11.4887** dB PSNR, **+0.463358** SSIM |
| Official composite | **0.634645** |

The provided starter baseline scores approximately 0.26, so this is roughly a
2.4x improvement on the official metric.

```bash
python scripts/validate.py
```

Regenerates the held-out split, denoises those images and scores them with the
official evaluator. Results are written to `scripts/_validation/metrics.json`.
The run takes about 35 minutes on 12 CPU threads and reproduces the table above
to seven decimal places; the residual difference is float accumulation order
across tiles.

The same numbers can be confirmed by calling the official evaluator directly:

```bash
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy \
                              --pred_dir scripts/_validation/pred \
                              --gt_dir <the 46 held-out ground-truth images>
```

## Layout

```text
scripts/
  denoise.py          competition entry point
  validate.py         reproduce the held-out metrics
  make_submission.py  build Voxel.zip from the denoised outputs
  requirements.txt
  ncw/
    nafnet.py         NAFNet backbone (LayerNorm2d, SimpleGate, NAFBlock, NAFNet)
    wavelet.py        fixed orthonormal Haar analysis
    conditioning.py   noise model, impulse detector, FiLM and wavelet adapters
    model.py          NCWNAFNet — the wrapper that ties them together
    inference.py      checkpoint loading, PNG I/O, overlap-tiled inference
    evaluation.py     scoring through the organizers' evaluator
    split.py          the deterministic held-out split
    paths.py          repository paths
  analysis/           noise characterization of the 460 public pairs
  docs/               problem analysis, literature review, solution strategy
  report/             Voxel_Report.tex and figures
  tests/              parity and shape tests
  weights/            model checkpoint
  licenses/           NAFNet MIT license
```

## Method

**Degradation model.** Residual variance across the 460 public pairs is
heteroscedastic, `var(I) = a*I + b`, with per-channel coefficients measured in
`analysis/noise_characterization/`:

```text
a = (0.017095, 0.016532, 0.019045)     shot noise
b = (0.006362, 0.005684, 0.005492)     read-noise floor
```

Severity forms a continuum rather than discrete levels, the read floor dominates
dark regions, and the residual is heavy-tailed with sparse impulses. These are
the observations the architecture is built around.

**Classical signal processing.** Three components operate directly on the noisy
observation, with no learned parameters:

1. *Analytical noise map* — `sqrt(a*I + b)` gives a per-pixel, per-channel
   standard-deviation estimate from the fitted sensor model, so the network
   knows how much corruption to expect at each intensity rather than inferring
   it from scratch.
2. *Exact 3x3 median* — implemented as a 36-comparison min/max sorting network
   over nine replicate-padded neighbours. The deviation of each pixel from its
   local median, channel-max and clipped, isolates impulse defects that a
   Gaussian model cannot describe.
3. *Four-level orthonormal Haar pyramid* — recursive decomposition of the LL
   band; the LH/HL/HH details at each scale expose high-frequency corruption
   without altering the RGB input the backbone sees.

**Conditioning.** The three sigma channels and the impulse channel form a
four-channel field. A shared 3x3 projection with two 1x1 heads produces FiLM
parameters at every encoder stage and the middle block:

```text
F' = (1 + g_gamma * tanh(gamma(C))) * F + g_beta * beta(C)
```

Haar details are projected by a `3x3 -> GELU -> 1x1` adapter and added after
each downsample. All scalar gates start at zero.

**Parameters.** 29,664,484 total — 29,159,715 in the backbone, 504,769 in the
conditioning and wavelet modules.

**Inference.** 256-pixel tiles with a 64-pixel discarded halo for context and a
separable raised-cosine blend across a 64-pixel overlap, so seams are invisible
and the receptive field is preserved. Predictions are quantised round-half-up,
`floor(x*255 + 0.5)`.

## Notes

- The model predicts a residual at the original exposure. It performs no gamma
  correction, tone mapping or global brightness adjustment.
- No image-specific lookup, no test-time metadata, no external training data.
  Image IDs 461-480 were excluded from all fitting, checkpoint selection and
  metric computation.
- `baseline/`, `evaluation/` and the provided data under `competition_data/`
  are unmodified.
- The NAFNet backbone derives from megvii-research/NAFNet (MIT); the license is
  retained in `licenses/NAFNet-LICENSE.txt`.
