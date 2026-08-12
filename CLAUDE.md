# cns-segmentation — Agent Guide

## What This Repo Does
Automated spinal cord + CSF segmentation from T2-weighted MRI → watertight STL meshes for CFD simulation.
Part of the [CNS Digital Twin](../) project. Downstream: `cns-cfd-simulation` consumes the STL output.

## Quick Commands
```bash
source .venv/bin/activate

python scripts/train.py --config configs/train_spine.yaml
python scripts/evaluate.py --config configs/inference.yaml
python scripts/export_mesh.py --input <nifti> --output <stl>
pytest tests/ -v
```

## Project Layout
```
src/cns_segmentation/
├── data/         — BIDS data loaders, MONAI transforms
├── models/       — SegResNet factory, MC-Dropout wrapper
├── losses/       — Soft clDice, CombinedLoss
├── training/     — Training loop with MLflow tracking
├── evaluation/   — Dice/HD95 metrics, ECE calibration
└── mesh/         — Mask → watertight STL pipeline
configs/          — YAML configs (train, inference)
scripts/          — CLI entry points
tests/            — Unit tests
```

## Tech Stack
- Python 3.10+, PyTorch 2.x, MONAI
- Architecture: SegResNet (3D)
- Loss: DiceCE + Soft clDice
- Uncertainty: MC-Dropout (8 passes)
- Mesh: marching cubes + trimesh repair
- Tracking: MLflow
- Dev compute: Apple MPS (fallback CPU)

## Critical Rules
1. **Device handling:** Always `"mps" if torch.backends.mps.is_available() else "cpu"`. Test MPS ops individually — some MONAI ops fail on MPS.
2. **Git-annex data:** Check `os.path.getsize(path) > 1000` before loading any NIfTI — annex pointer stubs are ~100 bytes.
3. **Validation splits:** Always split by SITE, never randomly.
4. **Spine labels location:** `derivatives/labels/sub-*/anat/*_T2w_label-SC_seg.nii.gz` — NOT next to images.
5. **Patch-based training:** Use `RandCropByPosNegLabel` with pos:neg ≥ 2:1 for spine.
6. **Mesh repair order:** fill holes → fix normals → remove degenerate → smooth. Never smooth first.
7. **clDice iterations:** `iter_=3` is sufficient for spinal cord.

## Mesh Output Contract
Meshes exported satisfy: watertight, manifold, Euler χ=2, binary STL, RAS coords, units mm.
These guarantees are required by `cns-cfd-simulation`.

## Targets
| Metric | Target |
|--------|--------|
| Spinal cord Dice | ≥ 0.93 |
| Mesh watertight | 100% |
| Mesh manifold | 100% |
| Uncertainty ECE | < 0.05 |
| Inter-vendor CV | < 5% |
| Inference time | < 5 min/case |
