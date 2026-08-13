# CNS Segmentation Project — Development Context

> **Purpose:** This file is the single source of truth for starting development in a fresh chat.
> **Status:** Approved for development. Team Lead has signed off on dataset + abstract.
> **Timeline:** 8 weeks (2 months) from today.
> **Developer:** Solo — using Claude Code as implementation partner.

---

## 1. What This Project Is

An **end-to-end AI/ML pipeline** for Eli Lilly's CNS Drug Delivery program that:
1. Takes T2-weighted spinal MRI as input
2. Segments spinal cord + CSF spaces (subarachnoid space)
3. Exports watertight, manifold STL meshes suitable for CFD simulation
4. Provides calibrated per-voxel uncertainty

**The business problem:** Intrathecal drug delivery (lumbar injection into CSF to reach brain) requires accurate 3D geometry of the spinal subarachnoid space for Computational Fluid Dynamics (CFD) simulation. Currently this is done manually (1-5 hrs/case) by external vendor DigiM. We're building an automated in-house alternative that runs in <5 minutes.

**Lilly's published paper:** Montoya & Teli et al. "Computational Modeling and Simulation of the Cerebrospinal Flow and Drug Delivery." *Alzheimer's & Dementia* 2024. DOI: 10.1002/alz.094612 — this is the existing manual workflow we're automating.

---

## 2. Approved Dataset

### Primary: Spine-Generic Multi-Subject Database

| Attribute | Value |
|-----------|-------|
| Paper | Cohen-Adad et al., *Nature Scientific Data* 8:219, 2021 |
| DOI | 10.1038/s41597-021-00941-8 |
| Subjects | 267 healthy adults |
| Sites | 42 centers, 3 continents |
| Vendors | GE, Philips, Siemens (all 3 major) |
| Contrasts | T1w, T2w, T2*, DWI, MT, MTS |
| Inter-site CV | < 5% for all metrics (from abstract) |
| Format | BIDS-compliant, NIfTI, ~26 GB |
| License | CC-BY 4.0 |
| Download | `git clone https://github.com/spine-generic/data-multi-subject.git` then `git annex get .` |
| Labels | `derivatives/labels/sub-*/anat/*_T2w_label-SC_seg.nii.gz` (spinal cord binary) |
| Rootlets | `derivatives/labels/sub-*/anat/*_T2w_label-rootlets_dseg.nii.gz` (C2-C8, stretch goal) |

### Supplementary (for architecture validation):
- **Medical Segmentation Decathlon Task04 (Hippocampus):** 260 MRI volumes, 3 classes. For proving multi-anatomy capability.
- **Sass 2017 CAD Model:** STL/OBJ of spinal SAS with 31 rootlet pairs. Ground-truth reference mesh for CFD geometry validation. CC-BY-SA 4.0. From supplementary of DOI: 10.1186/s12987-017-0085-y

---

## 3. Approved Technical Stack

| Layer | Tool | Version |
|-------|------|---------|
| Language | Python | 3.10+ |
| DL Framework | PyTorch | 2.x |
| Medical DL | MONAI | latest (`pip install "monai[all]"`) |
| Architecture | SegResNet (3D) | MONAI built-in |
| Segmentation Loss | DiceCE + Soft clDice | Custom implementation |
| Uncertainty | MC-Dropout (8 passes) | Custom on SegResNet |
| Image I/O | NiBabel, SimpleITK | latest |
| Mesh Generation | scikit-image (marching cubes) | latest |
| Mesh Processing | Trimesh, PyMeshLab | latest |
| Mesh Validation | Trimesh (watertight/manifold checks) | latest |
| Data Format | BIDS + NIfTI | Standard |
| Experiment Tracking | MLflow | latest |
| Demo | Streamlit | latest |
| Dev Compute | Apple MPS | macOS |
| Prod Compute | NVIDIA A100 (CUDA) | When available |

---

## 4. Success Criteria (Approved Targets)

| Metric | Target | Reference |
|--------|--------|-----------|
| Spinal cord Dice | ≥ 0.93 | SCT DeepSeg: 0.95 median (Gros et al. 2019) |
| Mesh watertight rate | 100% | CFD solver requirement |
| Mesh manifold rate | 100% | CFD solver requirement |
| Uncertainty ECE | < 0.05 | Medical AI calibration benchmark |
| Inter-vendor variance | CV < 5% | Matches dataset's own published metric |
| Inference time | < 5 min/case | vs 1-5 hrs manual |

---

## 5. Development Phases (8 Weeks)

### Phase 1 — Baseline & Data (Weeks 1–2)
- Set up environment + dependencies
- Download Spine-Generic data (git-annex)
- Build BIDS-aware data loader (discovers subjects, handles annex stubs)
- Implement MONAI preprocessing pipeline (RAS orientation, normalization, patching)
- Train SegResNet on T2w spinal cord binary segmentation
- Establish baseline Dice on site-stratified validation

### Phase 2 — Topology-Aware Training + Mesh Export (Weeks 3–4)
- Implement Soft clDice loss (3D morphological skeletonization)
- Build CombinedLoss (DiceCE + clDice, configurable weights)
- Run 3-way ablation: DiceCE vs clDice vs Combined
- Build mesh pipeline: mask → marching cubes → repair → smooth → validate → STL
- Achieve 100% watertight pass rate

### Phase 3 — Uncertainty Quantification (Weeks 5–6)
- Implement MC-Dropout inference wrapper
- Compute predictive entropy + mutual information + variance
- Build ECE computation (15-bin reliability diagram)
- Save uncertainty maps as NIfTI (overlayable in 3D Slicer)
- Target: ECE < 0.05

### Phase 4 — Demo, Evaluation & Documentation (Weeks 7–8)
- Full evaluation across held-out subjects
- Comparison table: our pipeline vs SCT vs TotalSegmentator-MRI
- Streamlit dashboard (load MRI → segment → mesh → uncertainty)
- Technical report
- Phase 0 transition plan

---

## 6. Key Technical Decisions (Already Made)

1. **Architecture: SegResNet** — Not nnU-Net (too heavy for iterative dev). SegResNet from MONAI is lighter, runs on MPS, competitive performance.
2. **Patch size for spine: [48, 160, 160]** — Empirically determined. Captures enough axial context while fitting in memory.
3. **Loss: DiceCE + clDice** — DiceCE for gradient stability + Dice optimization; clDice for topology (needed for CFD mesh quality).
4. **Validation split: by site** — NOT random. Critical for proving cross-vendor generalizability.
5. **Uncertainty: MC-Dropout** — Simplest approach that works. No architecture change needed; just enable dropout at inference.
6. **Mesh pipeline: scikit-image marching cubes + trimesh** — VTK/PyMeshLab for repair if trimesh alone isn't sufficient.
7. **Git-annex handling:** Data loader checks file size (>1000 bytes) to skip annex pointer stubs gracefully.

---

## 7. Project Structure (Target)

```
CNS/
├── README.md
├── pyproject.toml                # or requirements.txt
├── configs/
│   ├── train_spine.yaml
│   ├── train_hippocampus.yaml
│   └── inference.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── spine_generic.py     # BIDS-aware loader + git-annex handling
│   │   ├── msd_loader.py        # MSD format loader (hippocampus)
│   │   └── transforms.py        # MONAI transform pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segresnet.py         # Model factory + SegResNet config
│   │   └── uncertainty.py       # MC-Dropout wrapper
│   ├── losses/
│   │   ├── __init__.py
│   │   └── topology.py          # SoftClDice + CombinedLoss
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training loop (handles both tasks)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Dice, HD95, Betti, volume
│   │   └── calibration.py       # ECE, reliability diagrams
│   ├── mesh/
│   │   ├── __init__.py
│   │   └── export.py            # mask → watertight STL pipeline
│   └── demo/
│       └── app.py               # Streamlit dashboard
├── scripts/
│   ├── train.py                 # CLI entry: python scripts/train.py --config configs/train_spine.yaml
│   ├── evaluate.py              # CLI entry: runs full eval + mesh export
│   ├── export_mesh.py           # Standalone mesh export
│   └── uncertainty.py           # Standalone uncertainty analysis
├── data/                        # Symlinks or download scripts (NOT committed)
│   └── README.md                # Instructions for data download
├── experiments/                  # MLflow artifacts + saved models
├── notebooks/                    # Exploratory work
└── tests/
    ├── test_transforms.py
    ├── test_losses.py
    └── test_mesh.py
```

---

## 8. Data Download & Setup Instructions

```bash
# 1. Spine-Generic (primary — requires git-annex)
brew install git-annex  # macOS
git clone https://github.com/spine-generic/data-multi-subject.git data/spine-generic
cd data/spine-generic
# Download a subset first (faster iteration):
git annex get sub-amu01 sub-amu05 sub-balgrist01 sub-balgrist02 sub-stanford02 sub-stanford05 sub-mgh01 sub-mgh02 sub-tehranS01 sub-ubc03 sub-ubc04 sub-ucdavis03 sub-unf07
# Full download when ready: git annex get .

# 2. MSD Hippocampus (secondary)
# Download from: http://medicaldecathlon.com/ → Task04_Hippocampus.tar
# Extract to: data/msd_hippocampus/Task04_Hippocampus/

# 3. Sass 2017 geometry (validation reference)
# Download STL/OBJ from supplementary of DOI: 10.1186/s12987-017-0085-y
# Place in: data/reference_geometry/
```

---

## 9. Environment Setup

```bash
# Python environment
python3.10 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or cu121 for CUDA
pip install "monai[all]"
pip install nibabel SimpleITK
pip install scikit-image trimesh pymeshlab
pip install mlflow streamlit plotly
pip install pyyaml rich typer  # CLI utilities

# Optional but useful
pip install napari  # 3D visualization
pip install gudhi   # Persistent homology (for Betti numbers)
```

---

## 10. Key Domain Knowledge

### Anatomy
- **Spinal cord:** The central nervous tissue inside the vertebral column. ~45cm long, ~1cm diameter.
- **Subarachnoid space (SAS):** CSF-filled space between pia mater and arachnoid mater surrounding the cord. This is where intrathecal drugs flow.
- **Nerve rootlets:** C2-C8 dorsal rootlets branch off the cord. They significantly affect CSF flow patterns (+60% steady streaming per Khani 2018).
- **T2-weighted MRI:** CSF appears bright (hyperintense), cord appears dark. Ideal for segmenting CSF spaces.

### Why CFD needs watertight meshes
- CFD solvers (Ansys Fluent, OpenFOAM) simulate fluid flow inside a closed volume
- If the mesh has holes → fluid "leaks" → simulation diverges
- If non-manifold → solver can't determine inside/outside → crash
- Staircase artifacts from voxel boundaries → spurious turbulence in simulation

### Clinical context
- Lumbar puncture injects drug into CSF at L3-L4
- Drug must travel rostrally (upward) to reach brain
- CSF flow is oscillatory (cardiac + respiratory driven)
- CFD predicts: how much drug reaches the brain, how fast, distribution pattern
- This informs dosing, injection protocol, needle gauge selection

---

## 11. Relevant Spine-Generic Data Structure

```
data-multi-subject/
├── participants.tsv              # Subject demographics
├── dataset_description.json      # BIDS metadata
├── sub-amu01/
│   └── anat/
│       ├── sub-amu01_T2w.nii.gz          # ← INPUT: T2-weighted MRI volume
│       ├── sub-amu01_T2w.json            # Acquisition parameters
│       ├── sub-amu01_T1w.nii.gz
│       └── ...
├── derivatives/
│   └── labels/
│       ├── sub-amu01/
│       │   └── anat/
│       │       ├── sub-amu01_T2w_label-SC_seg.nii.gz      # ← LABEL: spinal cord binary mask
│       │       ├── sub-amu01_T2w_label-rootlets_dseg.nii.gz  # ← LABEL: rootlets (if available)
│       │       └── sub-amu01_T2w_label-disc_dseg.nii.gz   # Disc labels
│       └── ...
└── README.md
```

---

## 12. Prior PoC Results (Reference Only)

These numbers were achieved in the previous exploration (May 2026). They serve as reference/targets, NOT as code to reuse:

| Experiment | Result | Notes |
|-----------|--------|-------|
| Spine cord seg (SegResNet, 30 epochs, 24 train / 6 val) | Dice 0.951 ± 0.012 | Multi-site val (Stanford, Tehran, UBC, UCDavis, UNF) |
| Hippocampus (SegResNet, 30 epochs, MSD Task04) | Dice 0.880 ± 0.026 | 3-class, 15 test cases |
| Mesh quality (hippocampus) | 100% watertight, 100% manifold | 30/30 meshes |
| Mesh quality (spine) | 100% watertight, 100% manifold | 6/6 meshes |
| Uncertainty (MC-Dropout, 8 samples) | ECE 0.005 | 10 hippocampus cases |
| Loss ablation | DiceCE: 0.871, Combined: 0.863, Dice: 0.868 | Hippocampus task |
| Rootlets (9-class) | Dice 0.0 | Failed — insufficient data/epochs |

---

## 13. Critical Implementation Notes

1. **Apple MPS quirks:** Some MONAI ops don't work on MPS. Use `device = "mps" if torch.backends.mps.is_available() else "cpu"` but test each operation. Sliding window inference works on MPS.

2. **Git-annex data:** Files appear to exist but are 100-byte pointer stubs until `git annex get` downloads them. Always check `os.path.getsize(path) > 1000` before trying to load.

3. **Spine-Generic label location:** Labels are NOT next to images. They're in `derivatives/labels/sub-*/anat/`. Subject folder name is the link between image and label.

4. **Patch-based training for spine:** Full T2w volumes are too large. Use `RandCropByPosNegLabel` with pos:neg ratio ≥ 2:1 to ensure patches contain cord.

5. **clDice iteration count:** `iter_=3` is sufficient for spinal cord. More iterations = slower + diminishing returns. Use 3 for training, evaluate topology post-hoc.

6. **Mesh repair order matters:** (1) fill holes → (2) fix normals → (3) remove degenerate faces → (4) smooth. If you smooth first, you may introduce new non-manifold edges.

7. **Validation by site:** Never mix subjects from the same site across train/val. Split at the site level for honest generalization metrics.

---

## 14. References (Key Papers)

| # | Paper | DOI | Relevance |
|---|-------|-----|-----------|
| 1 | Cohen-Adad et al. 2021 (Spine-Generic) | 10.1038/s41597-021-00941-8 | Primary dataset |
| 2 | Montoya/Teli et al. 2024 (Lilly CSF model) | 10.1002/alz.094612 | Our anchor paper |
| 3 | Gros et al. 2019 (SCT DeepSeg) | 10.1016/j.neuroimage.2018.09.081 | Baseline to match |
| 4 | Shit et al. 2021 (clDice) | arxiv.org/abs/2003.07311 | Topology loss |
| 5 | Gal & Ghahramani 2016 (MC-Dropout) | arxiv.org/abs/1506.02142 | Uncertainty method |
| 6 | Sass et al. 2017 (SAS CAD model) | 10.1186/s12987-017-0085-y | Reference geometry |
| 7 | Khani et al. 2022 (Human CFD) | 10.1186/s12987-022-00304-4 | CFD context |
| 8 | Isensee et al. 2021 (nnU-Net) | 10.1038/s41592-020-01008-z | Architecture reference |
| 9 | Valošek et al. 2024 (Rootlets) | 10.1162/imag_a_00218 | Stretch goal benchmark |
| 10 | Wasserthal et al. 2023 (TotalSegmentator) | github.com/wasserth/TotalSegmentator | Comparison baseline |

---

*End of context. Start development from Phase 1.*
