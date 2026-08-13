# Learning Plan — CNS Medical Image Segmentation for CFD

> **Goal:** Build all technical + mathematical foundations needed to independently execute the CNS segmentation pipeline.
> **Structure:** 6 tracks, 4 weeks intensive (can overlap with development). Each topic has: what to learn, why you need it, and exactly where to learn it.
> **Priority:** ⚡ = Critical (blocks development), 🔶 = Important (needed within 2 weeks), 🔷 = Good-to-have (deepens quality)

---

## Track 1: Mathematics for 3D Medical Deep Learning

### 1.1 Linear Algebra (for understanding 3D transforms & architectures) ⚡

| Topic | Why You Need It | Resource |
|-------|----------------|----------|
| Matrix operations, transpose, inverse | Every image transform (rotation, scaling) is a matrix multiply | 3Blue1Brown "Essence of Linear Algebra" (YouTube, 15 videos, ~3hrs) |
| Affine transformations (4×4 matrices) | NIfTI files store a 4×4 affine matrix mapping voxels → real-world mm coordinates. You'll manipulate these directly. | [NiBabel coordinate systems docs](https://nipy.org/nibabel/coordinate_systems.html) |
| Eigenvalues/eigenvectors | PCA for dimensionality, understanding covariance in uncertainty | 3Blue1Brown Chapter 14 |
| Tensor operations (3D, 4D, 5D) | PyTorch tensors: (Batch, Channel, Depth, Height, Width) — you must think in 5D | PyTorch docs: "Tensor Basics" tutorial |

**Practice exercise:** Load a NIfTI file with nibabel, extract its affine matrix, understand what each row/column means, apply a rotation matrix to reorient it.

---

### 1.2 Calculus & Optimization (for understanding training) 🔶

| Topic | Why You Need It | Resource |
|-------|----------------|----------|
| Partial derivatives, chain rule | Backpropagation IS the chain rule applied repeatedly | 3Blue1Brown "Essence of Calculus" (YouTube) + "Neural Networks" series |
| Gradient descent (SGD, Adam, AdamW) | You're using AdamW with weight decay 1e-5. Know why. | Karpathy "Intro to Neural Networks" (YouTube, 2hrs) |
| Learning rate schedules (Cosine Annealing) | You're using CosineAnnealingLR — know the math behind warm restarts | [PyTorch LR scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) |
| Loss function gradients | DiceCE loss: understand why Dice is differentiable (soft Dice trick) | Read: Milletari "V-Net" paper Section 3 (2 pages) |

**Practice exercise:** Implement Dice loss from scratch in NumPy. Compute its gradient manually for a 2×2 prediction vs target. Verify with PyTorch autograd.

---

### 1.3 Probability & Statistics (for uncertainty quantification) ⚡

| Topic | Why You Need It | Resource |
|-------|----------------|----------|
| Bayesian probability basics | MC-Dropout is a Bayesian approximation. Must understand prior/posterior/predictive. | StatQuest "Bayes Theorem" + "Prior and Posterior" (YouTube, 30min) |
| Entropy (information theory) | Predictive entropy = your primary uncertainty measure: H = -Σ p·log(p) | StatQuest "Entropy" (YouTube) |
| Calibration & Expected Calibration Error | ECE measures if "90% confidence" actually means "correct 90% of the time" | Read: Guo et al. 2017 "On Calibration of Modern Neural Networks" (ICML) — Sections 1-3 only |
| Monte Carlo methods | MC-Dropout: run N forward passes, average = prediction, variance = uncertainty | Read: Gal & Ghahramani 2016 (arxiv.org/abs/1506.02142) — Sections 1-3 |
| Coefficient of variation (CV) | You report inter-site CV < 5%. CV = (std/mean) × 100. | Any intro statistics reference |

**Practice exercise:** Generate 100 random predictions (0-1). Bin them into 10 groups by confidence. Compute ECE by hand. Plot reliability diagram.

---

### 1.4 Topology Basics (for clDice and mesh validation) 🔶

| Topic | Why You Need It | Resource |
|-------|----------------|----------|
| Connected components (β₀) | Betti-0 = number of disconnected pieces. Your segmentation should have β₀=1 (one cord). | Wikipedia: "Connected component (graph theory)" + scipy.ndimage.label |
| Euler number | Euler = V - E + F (for meshes). Watertight sphere = 2. Quick topology check. | Wikipedia: "Euler characteristic" |
| Skeletonization (morphological) | clDice extracts the "centerline" of a structure via iterative erosion. Must understand morphological operations. | scikit-image docs: `skimage.morphology.skeletonize_3d` |
| Manifold vs non-manifold | A manifold mesh: every edge shared by exactly 2 faces. Non-manifold = CFD solver crash. | [Trimesh docs on mesh properties](https://trimesh.org/) |

**Practice exercise:** Create a 3D binary sphere in NumPy. Run `scipy.ndimage.label` to verify β₀=1. Poke a hole — verify β₀ changes. Run marching cubes and check `mesh.is_watertight`.

---

## Track 2: 3D Medical Image Processing

### 2.1 NIfTI Format & Coordinate Systems ⚡

| Topic | Resource | Time |
|-------|----------|------|
| NIfTI file structure (header, affine, data) | [NiBabel documentation](https://nipy.org/nibabel/nifti1.html) | 1-2 hrs |
| RAS vs LPS coordinate systems | [MONAI: "3D image transforms" tutorial](https://github.com/Project-MONAI/tutorials) | 1 hr |
| Voxel spacing (anisotropic resolution) | Load a Spine-Generic T2w, print `nii.header.get_zooms()` — understand what 0.5×0.5×3.0mm means | 30 min |
| Resampling & interpolation (bilinear for images, nearest for labels) | MONAI `Spacingd` docs | 30 min |

**Practice exercise:** Load `sub-amu01_T2w.nii.gz`, print shape, affine, spacing. Resample to isotropic 1mm³ using MONAI. Compare file sizes and visual quality.

---

### 2.2 BIDS Format ⚡

| Topic | Resource | Time |
|-------|----------|------|
| BIDS structure (subjects, sessions, derivatives) | [BIDS Specification](https://bids-specification.readthedocs.io/) — skim Sections 1-4 | 1 hr |
| How Spine-Generic organizes images vs labels | Look at `data-multi-subject/` structure + README | 30 min |
| participants.tsv, dataset_description.json | Open these files from Spine-Generic, understand metadata | 15 min |

---

### 2.3 MRI Physics (enough to understand your data) 🔶

| Topic | Why You Need It | Resource |
|-------|----------------|----------|
| T1 vs T2 weighting | T2w: CSF = bright, cord = dark. This is WHY T2w works for CSF space segmentation. | YouTube: "MRI Made Easy — T1 vs T2" (any radiology channel, 10-15 min) |
| What causes intensity differences across vendors | Different coils, field homogeneity, sequence parameters → intensity normalization is essential | MONAI tutorial on intensity normalization |
| Resolution vs field-of-view trade-off | Higher resolution = smaller FOV = partial cord coverage. Affects patch size decisions. | Any intro MRI textbook Chapter 1 |

**You do NOT need:** Full Bloch equations, k-space theory, or pulse sequence design. Just enough to understand why your data looks the way it does.

---

## Track 3: Deep Learning for Segmentation

### 3.1 PyTorch Fundamentals ⚡

| Topic | Resource | Time |
|-------|----------|------|
| Tensors, autograd, `nn.Module` | [PyTorch 60-min blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 2 hrs |
| Custom Dataset & DataLoader | PyTorch "Data Loading" tutorial | 1 hr |
| Training loop (forward → loss → backward → step) | Write one from scratch for MNIST, then understand MONAI's version | 2 hrs |
| `model.train()` vs `model.eval()` | Critical — dropout behaves differently. MC-Dropout exploits this. | PyTorch docs |
| Device management (CPU/MPS/CUDA) | `tensor.to(device)`, model placement | 30 min |

---

### 3.2 MONAI Specifics ⚡

| Topic | Resource | Time |
|-------|----------|------|
| MONAI Transforms (dictionary-based `*d` variants) | [MONAI Transforms tutorial](https://github.com/Project-MONAI/tutorials/blob/main/modules/transforms_demo_2d.ipynb) | 2 hrs |
| `CacheDataset` vs `Dataset` | CacheDataset pre-applies transforms and caches in RAM — essential for 3D | MONAI docs |
| `sliding_window_inference` | How it works: overlapping patches + Gaussian blending | MONAI docs + [tutorial](https://github.com/Project-MONAI/tutorials/blob/main/modules/sliding_window_inference.ipynb) | 1 hr |
| `RandCropByPosNegLabeld` | Patches guaranteed to contain foreground — solves class imbalance | MONAI docs |
| `DiceMetric`, `DiceCELoss` | Know the difference between Dice as metric vs Dice as loss | MONAI docs |
| `AsDiscrete` (post-processing) | Converts logits → one-hot for metric computation | MONAI docs |

**Best single resource:** Run the [MONAI spleen segmentation tutorial](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb) end-to-end. It covers 80% of what you need.

---

### 3.3 Segmentation Architectures 🔶

| Topic | Resource | Time |
|-------|----------|------|
| U-Net (the grandfather) | Read Ronneberger et al. 2015 — just look at Figure 1 + understand skip connections | 30 min |
| Encoder-decoder concept | Encoder: downsample (context). Decoder: upsample (localization). Skips: preserve detail. | Any U-Net explainer video |
| SegResNet (what you're using) | Residual blocks in encoder + VAE-regularized decoder. Read MONAI source or paper. | [MONAI SegResNet docs](https://docs.monai.io/en/stable/networks.html#segresnet) | 1 hr |
| nnU-Net (conceptual — not implementing) | Self-configuring: auto-selects patch size, batch size, architecture. Read Isensee 2021 Sections 1-2. | 1 hr |

**Key insight:** SegResNet ≈ ResNet encoder + transposed-conv decoder + skip connections + instance normalization. It's a U-Net with residual blocks.

---

### 3.4 Loss Functions (Deep Dive) ⚡

| Loss | Formula Intuition | When to Use | Resource |
|------|-------------------|-------------|----------|
| **Cross-Entropy** | -log(correct class probability). Penalizes wrong predictions hard. | General classification, good gradients | PyTorch `nn.CrossEntropyLoss` docs |
| **Dice Loss** | 1 - (2·intersection)/(sum). Directly optimizes overlap. | Imbalanced classes (small foreground) | Milletari V-Net paper (2016) |
| **DiceCE** | α·Dice + β·CE. Best of both. | Default for medical segmentation | MONAI `DiceCELoss` |
| **clDice** | Dice computed on skeletonized predictions/targets. | Tubular/elongated structures (cord, vessels) | Shit et al. 2021 (CVPR) — read Section 3 |

**Practice exercise:** Implement binary Dice loss in PyTorch. Create a 10×10 prediction and target. Compute loss. Change one pixel. See how loss changes. Then implement soft Dice (with softmax predictions).

---

## Track 4: Mesh Processing & CFD Basics

### 4.1 Marching Cubes Algorithm ⚡

| Topic | Resource | Time |
|-------|----------|------|
| What marching cubes does | Converts a 3D scalar field (your binary mask) into a triangle mesh (surface) | YouTube: "Marching Cubes Explained" (Sebastian Lague) | 15 min |
| `skimage.measure.marching_cubes` | API: input=3D array, output=vertices+faces+normals | scikit-image docs | 30 min |
| `level` parameter | level=0.5 for binary masks. This is the iso-surface threshold. | Experiment with different values |
| `spacing` parameter | Must match your voxel spacing or mesh will be distorted | Use `nii.header.get_zooms()` |

**Practice exercise:** Create a 3D binary sphere (32³). Run marching cubes. Visualize with trimesh. Verify it's watertight.

---

### 4.2 Mesh Repair & Validation ⚡

| Topic | Resource | Time |
|-------|----------|------|
| Trimesh basics (load, inspect, export) | [Trimesh docs](https://trimesh.org/) | 1 hr |
| `mesh.is_watertight` — what makes a mesh watertight | No holes, consistent winding (normals all point outward) | Trimesh docs |
| `mesh.is_winding_consistent` — manifold check | Every edge shared by exactly 2 faces | Trimesh docs |
| Hole filling, normal fixing | `mesh.fill_holes()`, `mesh.fix_normals()` | Trimesh docs |
| Laplacian smoothing | Moves each vertex toward average of neighbors. Reduces staircase. | `trimesh.smoothing.filter_laplacian` |
| Decimation (face count reduction) | Simplify mesh for faster CFD without losing shape | `mesh.simplify_quadric_decimation(target)` |

**Practice exercise:** Take a marching-cubes output from a real segmentation. Introduce a hole (delete faces). Verify `is_watertight=False`. Repair it. Verify `is_watertight=True`.

---

### 4.3 CFD Context (Conceptual Only) 🔷

| Topic | Why | Resource |
|-------|-----|----------|
| What CFD solves (Navier-Stokes equations) | Understand WHY mesh quality matters — bad mesh = wrong flow = wrong drug prediction | YouTube: "CFD Explained in 10 Minutes" |
| Mesh quality for CFD (skewness, aspect ratio) | Staircase artifacts create artificial turbulence at boundaries | Any Ansys Fluent "meshing best practices" guide (free) |
| What intrathecal drug delivery simulation looks like | Watch the CSF flow animations from Sass/Khani papers | Supplementary videos of DOI: 10.1186/s12987-017-0085-y |
| Boundary conditions | Inlet flow from cardiac pulsation, walls = no-slip. Your mesh defines these walls. | Conceptual understanding only |

**You do NOT need to:** Run CFD simulations, learn OpenFOAM, or solve Navier-Stokes. Just understand what the CFD team needs from your mesh output.

---

## Track 5: Uncertainty Quantification

### 5.1 MC-Dropout Theory ⚡

| Topic | Resource | Time |
|-------|----------|------|
| Dropout as regularization | Random zeroing of neurons during training prevents co-adaptation | Any DL course (dropout section) |
| Dropout as Bayesian approximation | Keeping dropout ON at inference ≈ sampling from posterior distribution over weights | Gal & Ghahramani 2016, Sections 1-2 (arxiv.org/abs/1506.02142) | 1 hr |
| N forward passes → mean + variance | mean = best prediction, variance = how much model disagrees with itself | Implement yourself |
| Predictive entropy vs mutual information | Entropy = total uncertainty. MI = epistemic (model) uncertainty. Difference = aleatoric (data noise). | Gal PhD thesis Chapter 2 (optional, deep) |

**Practice exercise:** Train a simple CNN on MNIST. Add dropout. Run 20 forward passes on one image. Plot the distribution of predictions. Compute variance. Try on an ambiguous/OOD image — variance should be higher.

---

### 5.2 Calibration ⚡

| Topic | Resource | Time |
|-------|----------|------|
| What calibration means | "When model says 80% confident, it should be correct 80% of the time" | Guo et al. 2017 "On Calibration" — Section 2 |
| ECE formula | ECE = Σ (|bin_accuracy - bin_confidence| × bin_proportion) over all bins | Same paper, Section 3 |
| Reliability diagram | X-axis: confidence, Y-axis: actual accuracy. Perfect = diagonal line. | Plot one yourself |
| Why calibration matters for medical AI | Clinicians need to trust confidence scores. Overconfident = dangerous. | Any medical AI deployment paper |

---

## Track 6: Software Engineering & Tools

### 6.1 Git-Annex (for Spine-Generic data) ⚡

| Topic | Resource | Time |
|-------|----------|------|
| What git-annex does | Version-controls large files by storing metadata in git, actual data in a remote | [git-annex quickstart](https://git-annex.branchable.com/walkthrough/) | 30 min |
| `git annex get <path>` | Downloads actual file content from remote | Practice on Spine-Generic |
| Detecting pointer stubs vs real files | Check file size: pointer = ~100 bytes, real NIfTI = megabytes | `os.path.getsize()` in Python |

---

### 6.2 MLflow (experiment tracking) 🔶

| Topic | Resource | Time |
|-------|----------|------|
| Logging metrics, parameters, artifacts | `mlflow.log_metric("dice", 0.95)` | [MLflow quickstart](https://mlflow.org/docs/latest/quickstarts/mlflow-tracing-quickstart.html) | 1 hr |
| Comparing runs | MLflow UI: `mlflow ui` → localhost:5000 | Practice |
| Saving/loading model artifacts | `mlflow.pytorch.log_model(model, "model")` | MLflow docs |

---

### 6.3 Config-Driven Training (YAML) 🔶

| Topic | Resource | Time |
|-------|----------|------|
| YAML syntax | [YAML tutorial](https://learnxinyminutes.com/docs/yaml/) | 15 min |
| Why config files > CLI args | Reproducibility, experiment tracking, ablation studies | Best practice |
| PyYAML or OmegaConf | `pip install pyyaml` / `pip install omegaconf` | Docs |

---

## Recommended Learning Sequence (4-Week Intensive)

### Week 1: Foundations (before or alongside Phase 1 development)

| Day | Morning (2-3 hrs) | Afternoon (2-3 hrs) |
|-----|-------------------|---------------------|
| Mon | 3Blue1Brown: Linear Algebra (videos 1-7) | PyTorch 60-min blitz tutorial |
| Tue | 3Blue1Brown: Linear Algebra (videos 8-15) | NIfTI format + NiBabel tutorial |
| Wed | MONAI spleen tutorial (full run-through) | BIDS format + Spine-Generic structure exploration |
| Thu | MONAI transforms deep-dive (all `*d` variants) | Implement: load Spine-Generic subject, visualize slices |
| Fri | Loss functions: implement Dice from scratch | MRI physics basics (T1 vs T2, 1 hr video) |

### Week 2: Architecture & Training (alongside Phase 1-2)

| Day | Morning (2-3 hrs) | Afternoon (2-3 hrs) |
|-----|-------------------|---------------------|
| Mon | U-Net architecture deep-dive (paper + diagram) | SegResNet: read MONAI source, understand residual blocks |
| Tue | Training loop: write one from scratch (no MONAI) | Sliding window inference: understand overlapping + blending |
| Wed | clDice paper (Shit et al. 2021) — Sections 1-4 | Implement: soft erosion/dilation via max-pooling |
| Thu | Morphological operations: erosion, dilation, skeletonization | Practice: skeletonize a tubular structure in 3D |
| Fri | Class imbalance strategies (pos/neg sampling, loss weighting) | CosineAnnealing LR: plot schedule, understand warm restarts |

### Week 3: Mesh & Uncertainty (alongside Phase 2-3)

| Day | Morning (2-3 hrs) | Afternoon (2-3 hrs) |
|-----|-------------------|---------------------|
| Mon | Marching cubes algorithm (video + implementation) | Trimesh: load, inspect, repair, export workflows |
| Tue | Mesh topology: watertight, manifold, Euler number | Practice: break and repair meshes programmatically |
| Wed | Probability: Bayes theorem, posterior, predictive | MC-Dropout theory (Gal 2016, Sections 1-3) |
| Thu | Implement MC-Dropout on a toy model (MNIST) | Entropy, mutual information — compute by hand |
| Fri | ECE: implement from scratch | Reliability diagrams: plot for your toy model |

### Week 4: Integration & Domain (alongside Phase 3-4)

| Day | Morning (2-3 hrs) | Afternoon (2-3 hrs) |
|-----|-------------------|---------------------|
| Mon | CFD concepts: Navier-Stokes intuition, boundary conditions | Watch Sass/Khani CSF flow videos |
| Tue | Spinal cord anatomy: where is SAS, what are rootlets | Read Montoya/Teli 2024 (Lilly's paper) in full |
| Wed | MLflow: log a complete experiment | Streamlit: build a simple image viewer |
| Thu | Read: Gros et al. 2019 (SCT DeepSeg) — understand baseline | Read: Cohen-Adad 2021 (Spine-Generic) — dataset paper |
| Fri | Review all concepts. Identify gaps. Fill them. | End-to-end mental walkthrough: MRI → segment → mesh → CFD |

---

## Quick-Reference: Mathematical Formulas You'll Use

### Dice Coefficient
```
Dice = (2 × |P ∩ G|) / (|P| + |G|)

Where:
  P = predicted voxels (binary)
  G = ground truth voxels (binary)
  |P ∩ G| = intersection (true positives)
  Range: 0 (no overlap) to 1 (perfect)
```

### Soft Dice Loss (differentiable)
```
SoftDice = 1 - (2 × Σᵢ pᵢ·gᵢ + ε) / (Σᵢ pᵢ² + Σᵢ gᵢ² + ε)

Where:
  pᵢ = predicted probability for voxel i (continuous, 0-1)
  gᵢ = ground truth for voxel i (0 or 1)
  ε = smoothing term (1e-5) to prevent division by zero
```

### clDice (Centerline Dice)
```
Tprec = |S(P) ∩ G| / |S(P)|     (skeleton of prediction covered by ground truth)
Tsens = |S(G) ∩ P| / |S(G)|     (skeleton of ground truth covered by prediction)
clDice = 2 × (Tprec × Tsens) / (Tprec + Tsens)

Where S(·) = soft skeletonization operator (iterative erosion/dilation)
```

### Predictive Entropy (Uncertainty)
```
H[y|x] = -Σ_c  p̄_c · log(p̄_c)

Where:
  p̄_c = (1/T) Σₜ pₜ(y=c|x)    (mean prediction over T MC-Dropout passes)
  c = class index
  T = number of MC samples (we use 8)
  Range: 0 (certain) to log(C) (maximum uncertainty for C classes)
```

### Expected Calibration Error (ECE)
```
ECE = Σ_b (nₘ/N) × |acc(b) - conf(b)|

Where:
  b = confidence bin (we use 15 bins from 0 to 1)
  nₘ = number of samples in bin b
  N = total number of samples
  acc(b) = accuracy of predictions in bin b
  conf(b) = average confidence of predictions in bin b
  Target: ECE < 0.05
```

### Euler Number (Mesh Topology)
```
χ = V - E + F

Where:
  V = vertices, E = edges, F = faces
  Watertight sphere: χ = 2
  Watertight torus: χ = 0
  If χ ≠ 2 for a single closed surface → topology issue
```

---

## Essential YouTube Playlist (in order)

1. **3Blue1Brown — Essence of Linear Algebra** (full series, 3 hrs)
2. **3Blue1Brown — Neural Networks** (4 videos, 1 hr)
3. **StatQuest — Probability fundamentals** (Bayes, distributions, entropy — pick relevant ones)
4. **Andrej Karpathy — "Let's build GPT"** (for understanding training loops at a deep level)
5. **Sebastian Lague — Marching Cubes** (15 min, visual intuition)
6. **Any radiology channel — "MRI T1 vs T2 explained"** (10 min)
7. **MONAI bootcamp recordings** (YouTube, Project MONAI channel — "3D Segmentation" session)

---

## Books (Reference, Not Cover-to-Cover)

| Book | Use For | How Much to Read |
|------|---------|------------------|
| **Deep Learning** (Goodfellow, Bengio, Courville) | Math foundations: Ch 2 (Linear Algebra), Ch 3 (Probability), Ch 6 (DL basics) | 3 chapters |
| **Hands-On Medical Image Analysis with Python** (Dey) | MONAI, NIfTI, medical DL workflows | Skim relevant chapters |
| **Digital Image Processing** (Gonzalez & Woods) | Morphological operations (Ch 9) — erosion, dilation, skeletonization | 1 chapter |

---

## Validation Checkpoints (Test Yourself)

After each week, you should be able to:

### After Week 1:
- [ ] Load a NIfTI file, explain its affine matrix, resample to isotropic
- [ ] Write a PyTorch training loop from scratch (forward/backward/step)
- [ ] Explain why T2w MRI shows CSF as bright
- [ ] Navigate BIDS structure to find image + its label

### After Week 2:
- [ ] Draw SegResNet architecture (encoder blocks, skip connections, decoder)
- [ ] Implement Dice loss and clDice loss from scratch
- [ ] Explain why `RandCropByPosNegLabel` solves class imbalance
- [ ] Explain what CosineAnnealing does and why

### After Week 3:
- [ ] Run marching cubes on a binary mask and get a watertight mesh
- [ ] Repair a non-manifold mesh programmatically
- [ ] Implement MC-Dropout: run 8 passes, compute mean + variance
- [ ] Calculate ECE for a set of predictions and plot reliability diagram

### After Week 4:
- [ ] Explain end-to-end: MRI → preprocess → segment → mesh → CFD (what each step does)
- [ ] Explain why rootlets matter for CSF flow (+60% steady streaming)
- [ ] Set up MLflow experiment tracking for a training run
- [ ] Build a simple Streamlit app that displays a NIfTI volume slice-by-slice

---

*This plan is designed to be executed alongside development — not as a prerequisite. Start building from Day 1; learn what you need just-in-time.*
