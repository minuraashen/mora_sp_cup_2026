# 03 — Literature Review: SOTA Blind / Low-Light Image Denoising

Scope: methods relevant to **blind sRGB denoising of low-light sensor noise**, split into (A) deep learning, (B) classical DSP, (C) hybrid, plus (D) noise modeling and (E) training/inference tricks. Each entry notes **relevance to our problem**. References at the bottom.

---

## A. Deep-learning restoration backbones (the SOTA core)

### A1. Restormer (CVPR 2022) — efficient transformer
- Multi-Dconv-head **transposed (channel) attention** + gated-Dconv FFN; attention over channels not spatial, so it scales to high-res. First to pass **40 dB on SIDD**; strong on real denoising, deraining, deblurring. [R1]
- **Relevance:** top-tier quality; a natural backbone. Heavier on CPU, but tile-able. The NTIRE-2025 winner used a **Restormer+NAFNet hybrid** (see C1).

### A2. NAFNet (ECCV 2022) — nonlinear-activation-free CNN
- Shows a simple CNN U-Net with **no activation functions** (uses SimpleGate + simplified channel attention) matches/beats transformers on SIDD/GoPro at **much lower compute**. Community consensus baseline for denoising competitions. [R2]
- **Relevance:** *best quality-per-FLOP* → ideal given our **CPU-runtime** constraint. Strong default backbone for us. In head-to-head comparisons (U-Net vs. Restormer vs. NAFNet) NAFNet often wins.

### A3. SwinIR (ICCV 2021 W) — Swin-transformer restoration
- Window-attention transformer; strong SR/denoising. Superseded on efficiency by NAFNet/Mamba but still a solid, well-supported reference. [R3]

### A4. Uformer / MPRNet / MIRNet / HINet — the "modern classics"
- U-shaped transformer (Uformer), multi-stage progressive (MPRNet), multi-scale feature (MIRNet), half-instance-norm (HINet). All strong real-denoising performers; frequently the ingredients recombined in challenge entries. [R4]

### A5. MambaIR / MambaIRv2 (ECCV 2024 / CVPR 2025) — state-space models
- Selective state-space (Mamba) with **linear complexity** and global receptive field; adds local-enhancement + channel attention to fix Mamba's "local pixel forgetting." **Beats SwinIR by up to 0.45 dB** at similar cost. MambaIRv2 adds attentive state-space restoration. [R5]
- **Relevance:** promising efficiency/quality trade-off; but tooling is less mature than NAFNet and CPU kernels for Mamba are not as optimized — **higher risk** for a time-boxed, CPU-constrained submission.

### A6. CBDNet (CVPR 2019) — the canonical *blind* real denoiser
- Two subnets: a **noise-estimation** subnet producing a per-pixel noise-level map, then a **non-blind** denoiser conditioned on it. **Asymmetric loss** discourages under-estimation of noise. Trained on realistic synthetic noise + real pairs. [R6]
- **Relevance:** the reference design for *blind* denoising — exactly our setting (unknown, varying severity). The "estimate then condition" idea is directly reusable, even bolted onto a NAFNet/Restormer backbone.

### A7. SCUNet (MIR 2023) — practical blind denoising via data synthesis
- **Swin-Conv** blocks in a UNet, trained with a **rich degradation synthesis pipeline** (Gaussian + Poisson + speckle + JPEG + processed camera-sensor noise, plus resize, random-shuffle, double-degradation). Emphasis on *practical* generalization to unknown real noise. Public code + pretrained weights. [R7]
- **Relevance:** arguably the **best off-the-shelf starting point** for us — designed for blind sRGB noise of unknown type, ships weights, and its degradation pipeline is a template for generating our own training data. Strong candidate for fine-tuning on the 460 pairs.

---

## B. Classical DSP denoisers (the 10% "beyond baseline" credit + hybrid stages)

### B1. BM3D / CBM3D (2007) — collaborative filtering
- Block-matching + 3D transform-domain shrinkage; **still the strongest classical denoiser** and a common non-learned reference. `bm3d` PyPI package exists. [R8]
- **Relevance:** a legitimate "significant classical technique beyond the baseline's NLM." Can serve as a strong classical baseline (~+1–2 dB over NLM when tuned) *and* as a refinement/fusion stage. CPU-friendly (though not fast at 992²).

### B2. Wavelet-domain denoising (BayesShrink / VisuShrink, dual-tree)
- Thresholding in a multiresolution transform; `skimage.restoration.denoise_wavelet`. Explicitly named in the handbook's key-libraries list. Great for the classical-SP narrative and cheap. [R9]

### B3. Non-Local Means (baseline) & bilateral / guided filtering
- NLM (Buades 2005) is the baseline. **Bilateral** and **guided filter** (He 2013) are edge-preserving smoothers — useful as **detail-preserving post-refinement** or as the "structure-transfer" step in a hybrid. Kornia provides differentiable versions. [R10]

### B4. Adaptive median / rank-order filters for impulse & hot pixels
- Standard fix for **defect/hot pixels** (impulse outliers) which Gaussian-type denoisers smear rather than remove. An **adaptive median** or outlier-rejection pre-pass is well-justified given the handbook explicitly lists sensor defects. Cheap, and a clean classical-SP contribution.

### B5. Anscombe / VST for signal-dependent (Poisson) noise
- **Variance-stabilizing transform** (generalized Anscombe) converts Poisson-Gaussian noise to approximately homoscedastic Gaussian, denoise with an AWGN method, then inverse-transform. [R11]
- **Relevance:** if §6 of [doc 02](02_problem_and_metric_analysis.md) shows signal-dependent noise, VST-then-denoise is a principled classical stage and an excellent report talking-point.

---

## C. Hybrid transformer+CNN (what actually wins challenges now)

### C1. NTIRE-2025 Image Denoising winner (team SRC-B) — **Restormer + NAFNet hybrid**
- Global features via the **transformer (Restormer)**, local detail recovery via the **CNN (NAFNet)**; rationale: pure transformers are weak at local detail, pure CNNs weak at global context. Combined with **data selection, wavelet-transform loss, progressive learning, and self-ensemble**. Reached ~31.2 dB on the σ=50 AWGN sRGB task. [R12]
- **Relevance:** this is the current *recipe* for a top score. Even if we don't build the exact hybrid, we should borrow the **training strategy** (progressive patch growth, wavelet loss, self-ensemble/TTA, careful data curation).

### C2. Xformer (2023) — X-shaped hybrid transformer
- Combines spatial-wise and channel-wise transformer branches for denoising; another point on the "hybridize to get both local+global" trend. [R13]

### C3. General trend (AIM/NTIRE 2025)
- Winners lean on **NAFNet-derived** cores + incremental attention, **masking/MAE-style pretraining for generalization** (AIM-2025 winner MR-CAS), and heavy **ensembling/TTA**. Model design matters less than **data-centric training + ensembling**. [R12][R14]

---

## D. Noise modeling (critical because our noise is *synthetic* and we can regenerate it)

### D1. Physics-based sensor noise (Poisson-Gaussian / heteroscedastic Gaussian)
- Shot + read noise ≈ **compound Poisson-Gaussian**, commonly approximated by a **heteroscedastic Gaussian** with signal-dependent variance `σ²(x) = a·x + b`. Parameters vary by device/ISO. [R15]
- **Relevance:** the right first-order model for our synthetic corruption if it's signal-dependent. We can *fit* `(a, b)` from the 460 residuals and synthesize matching noise for unlimited augmentation.

### D2. ELD / physics-based noise formation for extreme low light (CVPR 2020)
- Adds read-noise fine structure (banding, quantization, fixed-pattern) beyond Poisson-Gaussian; shows realistic noise modeling ≈ real paired data for training. [R16]

### D3. Learned / diffusion noise synthesis (2023–2025)
- Physics-informed noise models from dark frames; **Dark Noise Diffusion** and adversarial/flow noise generators outperform naive synthetic noise and rival real pairs. [R17]
- **Relevance:** likely overkill for an 8-day sprint, but a good "alternatives considered" citation. Our pragmatic path: fit a parametric model to the 460 residuals (D1) rather than learn a generator.

---

## E. Training & inference tricks that reliably add score

- **Self-ensemble / geometric TTA** — average predictions over the 8 flips/rotations (D4 group). ~+0.1–0.3 dB, no training cost. Classic SP trick; cheap on GPU, adds CPU time (make it a toggle). [R1][R12]
- **Progressive training** — grow patch size (e.g., 128→256→384) during training; Restormer/NTIRE-winner practice. [R1][R12]
- **Loss:** L1 (Charbonnier) primary; add **SSIM / MS-SSIM** and/or **wavelet-domain** loss terms to directly push the metric we're scored on. [R12]
- **Overlap-tiled inference** with feathered blending for the 992² images (see [doc 02](02_problem_and_metric_analysis.md) §5).
- **Model ensembling** (average a few checkpoints / architectures) — reliable at the top of leaderboards, but costs CPU time; weigh against the 15% runtime score.

---

## F. How the literature maps to our decision

| Need | Best-fit from literature |
|------|--------------------------|
| Blind, unknown severity | CBDNet-style noise conditioning [R6]; or intrinsically-blind SCUNet [R7] |
| Best quality/FLOP on CPU | **NAFNet** [R2] |
| Best raw quality | Restormer [R1] / Restormer+NAFNet hybrid [R12] |
| Fast strong-classical stage (10% credit) | **BM3D** [R8], wavelet [R9], adaptive-median hot-pixel [B4], VST [R11] |
| Regenerate training noise | Poisson-Gaussian fit [R15] / ELD-style [R16] |
| Free score bump | self-ensemble TTA, SSIM+wavelet loss, progressive training [R12] |
| Off-the-shelf head start with weights | **SCUNet** [R7] (pretrained blind sRGB) |

**Recommended direction (detailed in [doc 04](04_solution_strategy.md)):** an **efficient learned core (NAFNet, optionally SCUNet-initialized) trained on real 460 pairs + parametric-synthetic augmentation**, wrapped with **classical DSP** (adaptive-median hot-pixel pre-pass + optional wavelet/BM3D refinement + YCbCr chroma handling) and **self-ensemble TTA**, served with **overlap-tiled, CPU-fallback inference**. This simultaneously targets the 35% image score, the 10% classical-SP credit, and the 15% runtime/code score.

---

## References

- [R1] Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022. arXiv:2111.09881.
- [R2] Chen et al., *Simple Baselines for Image Restoration (NAFNet)*, ECCV 2022. arXiv:2204.04676.
- [R3] Liang et al., *SwinIR: Image Restoration Using Swin Transformer*, ICCV Workshops 2021. arXiv:2108.10257.
- [R4] Wang et al., *Uformer*, CVPR 2022; Zamir et al., *MPRNet*, CVPR 2021; Zamir et al., *MIRNet*, ECCV 2020; Chen et al., *HINet*, CVPRW 2021.
- [R5] Guo et al., *MambaIR: A Simple Baseline for Image Restoration with State-Space Model*, ECCV 2024; *MambaIRv2*, CVPR 2025.
- [R6] Guo et al., *Toward Convolutional Blind Denoising of Real Photographs (CBDNet)*, CVPR 2019. arXiv:1807.04686.
- [R7] Zhang et al., *Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis (SCUNet)*, Machine Intelligence Research 2023. arXiv:2203.13278. Code: github.com/cszn/SCUNet.
- [R8] Dabov et al., *Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering (BM3D)*, IEEE TIP 2007.
- [R9] Chang et al., *Adaptive Wavelet Thresholding (BayesShrink)*, IEEE TIP 2000; `skimage.restoration.denoise_wavelet`.
- [R10] Buades et al., *Non-Local Means*, CVPR 2005; He et al., *Guided Image Filtering*, IEEE TPAMI 2013; Kornia.
- [R11] Makitalo & Foi, *Optimal Inversion of the Generalized Anscombe Transformation for Poisson-Gaussian Noise*, IEEE TIP 2013.
- [R12] Sun, Guo et al., *The Tenth NTIRE 2025 Image Denoising Challenge Report*, CVPRW 2025. arXiv:2504.12276. (Winner: SRC-B, Restormer+NAFNet hybrid, ~31.2 dB.)
- [R13] Zhang et al., *Xformer: Hybrid X-Shaped Transformer for Image Denoising*, 2023. arXiv:2303.06440.
- [R14] AIM 2025 Challenge on Real-World RAW Image Denoising, arXiv:2510.06601 (winner MR-CAS, MAE-style masking for generalization).
- [R15] Foi et al., *Practical Poissonian-Gaussian Noise Modeling*, IEEE TIP 2008.
- [R16] Wei et al., *A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising (ELD)*, CVPR 2020. arXiv:2003.12751.
- [R17] *Dark Noise Diffusion: Noise Synthesis for Low-Light Image Denoising*, 2025, arXiv:2503.11262; *Learning Physics-Informed Noise Models from Dark Frames*, arXiv:2310.09126.
