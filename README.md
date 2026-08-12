# cns-segmentation

Automated spinal cord + CSF segmentation from T2-weighted MRI, producing watertight STL meshes for CFD simulation. Part of the **CNS Digital Twin** pipeline.

## Overview

```
T2w MRI (NIfTI) → SegResNet → binary mask → marching cubes → repaired STL
```

Downstream consumer: [`cns-cfd-simulation`](../cns-cfd-simulation/) ingests the exported STL meshes.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train
python scripts/train.py --config configs/train_spine.yaml

# Evaluate
python scripts/evaluate.py --config configs/inference.yaml

# Export mesh from a segmentation prediction
python scripts/export_mesh.py --input prediction.nii.gz --output mesh.stl
```

## Project Layout

```
src/cns_segmentation/
├── data/         — BIDS data loaders, MONAI transforms
├── models/       — SegResNet factory, MC-Dropout wrapper
├── losses/       — Soft clDice, CombinedLoss
├── training/     — Training loop with MLflow tracking
├── evaluation/   — Dice/HD95 metrics, ECE calibration
└── mesh/         — Mask → watertight STL export
configs/          — YAML configs (train, inference)
scripts/          — CLI entry points
tests/            — Unit tests
```

## Mesh Output Contract

Meshes exported by this repo satisfy:
- Watertight (no open boundaries)
- Manifold (no non-manifold edges/vertices)
- Euler number χ = 2
- Format: binary STL, coordinate space: RAS, units: mm

These guarantees are required by `cns-cfd-simulation`.

## Targets

| Metric | Target |
|---|---|
| Spinal cord Dice | ≥ 0.93 |
| Mesh watertight | 100% |
| Mesh manifold | 100% |
| Uncertainty ECE | < 0.05 |
| Inter-vendor CV | < 5% |
| Inference time | < 5 min/case |

## Tech Stack

- Python 3.10+, PyTorch 2.x, MONAI
- Architecture: SegResNet (3D)
- Loss: DiceCE + Soft clDice
- Uncertainty: MC-Dropout (8 passes)
- Mesh repair: marching cubes + trimesh
- Tracking: MLflow
