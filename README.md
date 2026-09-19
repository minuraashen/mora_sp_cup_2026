# Welcome to the Low-Light Image Denoising & Enhancement Challenge

**Mora SP Cup 2026**

## About the Challenge

Low-light photography is challenging because camera sensors become more
susceptible to noise and other artifacts under limited illumination.

In this competition, participants will work with a custom dataset prepared
from clean mobile-phone photographs and corrupted using a synthetic low-light
noise model that reproduces several common sensor-related artifacts.

Your task is to design an effective denoising solution that removes these
artifacts while preserving image details, structure, and visual quality.

Whether you are new to image processing or testing novel approaches, this is
a hands-on opportunity to solve a practical digital imaging problem.

## The Task

- Denoise a collection of noisy images.
- The dataset contains different severity levels of noise.
- All images have a resolution of **992 × 992 pixels**.
- The complete dataset consists of 500 noisy images and 500 corresponding
  ground-truth images, divided as follows:

  - **Public set:** 460 noisy images
    (`001_noise.png`–`460_noise.png`) together with their ground-truth
    images (`001.png`–`460.png`). Use these images to develop, train,
    and validate your approach.

  - **Preliminary submission set:** 20 noisy images
    (`461_noise.png`–`480_noise.png`) are provided without their
    corresponding ground-truth images. Teams must denoise these images
    and submit the outputs as `461.png`–`480.png`.

  - **Final-round set:** A further 20 image pairs (`481`–`500`) are kept
    fully hidden by the organizers. Finalists will receive only the
    corresponding noisy images during the physical final round.

- The provided noisy images must **not** be renamed.

## Project Layout

Place the images in the correct directories before running the baseline:

```text
competition_data/
|-- public/
|   |-- ground_truth/
|   |   |-- 001.png
|   |   |-- ...
|   |   `-- 460.png
|   |
|   |-- noisy/
|   |   |-- 001_noise.png
|   |   |-- ...
|   |   `-- 460_noise.png
|   |
|   `-- denoised/
|
`-- submissions/
    |-- noisy/
    |   |-- 461_noise.png
    |   |-- ...
    |   `-- 480_noise.png
    |
    `-- denoised/

---

# Team Voxel — Preliminary Round Submission

**Frozen submission commit:** `07c1e2d3da10fafa2ff0cf519af0ec3cc55d7a24`

That commit contains the complete submitted solution: code, model checkpoint,
report and the 20 denoised outputs. The only commit after it is this one, which
exists solely to record the SHA above (a commit cannot contain its own hash).
The tip of `main` is therefore the submitted version.

## Solution

**NCW-NAFNet** — a NAFNet-w32 restoration backbone wrapped in noise
conditioning driven by classical signal processing: a fitted Poisson-Gaussian
variance model `var(I) = a*I + b`, an exact 3x3 median impulse detector built
from a min/max sorting network, and a four-level orthonormal Haar pyramid.
Every added path is gated by a scalar initialised to zero, so the wrapper
starts as an exact identity over the backbone.

Full method, architecture and reproduction instructions:
[`scripts/README.md`](scripts/README.md).
Report: [`scripts/report/Voxel_Report.pdf`](scripts/report/Voxel_Report.pdf).

## Run

```bash
pip install -r scripts/requirements.txt
python scripts/denoise.py --noise_dir competition_data/submissions/noisy \
                          --denoised_dir competition_data/submissions/denoised
```

CPU only, fully offline, deterministic.

## Model file

```text
Filename:      ncw_1h_best_composite_fp16.pth
Expected path: scripts/weights/ncw_1h_best_composite_fp16.pth
Size:          59585807 bytes
SHA-256:       be1515a1afe066bbd445b6a5b73f04d0141d3ed08e22eaa4cb9fd145df86155b
```

Committed to this repository — no download required. The float32 original
(`ncw_1h_best_composite.pth`, 177937000 bytes, SHA-256
`30bbb067337916eaf938e1f3349ea347061f1a44e106ac505407c35eac42035f`) is
available on Google Drive and documented in
[`scripts/README.md`](scripts/README.md); it exceeds GitHub's 100 MB
regular-file limit and so is not committed.

## Results

Measured on the 46 held-out public images with the unmodified
`evaluation/evaluate.py`. Image IDs 461-480 were excluded from all fitting,
checkpoint selection and metric computation.

| Metric | Value |
|---|---|
| PSNR | **31.3054** dB |
| SSIM | **0.919342** |
| Mean improvement over noisy | **+11.4887** dB PSNR, **+0.463358** SSIM |
| Official composite | **0.634645** |

Reproduce with `python scripts/validate.py`.

## Compliance

- All custom code lives in `scripts/`.
- `baseline/`, `evaluation/` and the provided data are unmodified.
- No image-specific lookup, no test-time metadata, no external training data.
- Inference is offline and deterministic.
