# CNS Segmentation Pipeline

Automated spinal cord and CSF segmentation from T2-weighted MRI for computational fluid dynamics (CFD) simulation of intrathecal drug delivery.

## Overview

This pipeline:

1. Takes T2-weighted spinal MRI as input
2. Segments spinal cord + CSF spaces using SegResNet (3D)
3. Exports watertight, manifold STL meshes suitable for CFD
4. Provides calibrated per-voxel uncertainty via MC-Dropout

## Quick Start

```bash
# Setup environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Download data (requires git-annex)
brew install git-annex
git clone https://github.com/spine-generic/data-multi-subject.git data/spine-generic
cd data/spine-generic && git annex get sub-amu01 sub-amu05 sub-balgrist01 sub-balgrist02

# Train
python scripts/train.py --config configs/train_spine.yaml

# Evaluate
python scripts/evaluate.py --config configs/inference.yaml

# Export mesh
python scripts/export_mesh.py --input prediction.nii.gz --output mesh.stl

# Run demo
streamlit run src/demo/app.py
```

## Project Structure

```
src/data/       — BIDS-aware data loaders, MONAI transforms
src/models/     — SegResNet factory, MC-Dropout wrapper
src/losses/     — Soft clDice topology loss
src/training/   — Training loop with MLflow tracking
src/evaluation/ — Metrics (Dice, HD95, ECE)
src/mesh/       — Mask → watertight STL mesh pipeline
src/demo/       — Streamlit dashboard
configs/        — YAML training/inference configs
scripts/        — CLI entry points
```

## Key Metrics

| Metric           | Target  | Achieved |
| ---------------- | ------- | -------- |
| Spinal cord Dice | ≥ 0.93 | —       |
| Mesh watertight  | 100%    | —       |
| Uncertainty ECE  | < 0.05  | —       |
| Inference time   | < 5 min | —       |

## References

- Cohen-Adad et al. 2021 — Spine-Generic dataset (DOI: 10.1038/s41597-021-00941-8)
- Montoya/Teli et al. 2024 — Lilly CSF modeling (DOI: 10.1002/alz.094612)
- Shit et al. 2021 — clDice topology loss (arXiv: 2003.07311)
