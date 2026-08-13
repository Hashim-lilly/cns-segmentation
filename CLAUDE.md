# CNS Segmentation Pipeline — Claude Agent Guide

## What This Project Is
Automated spinal cord + CSF segmentation from T2-weighted MRI → watertight STL meshes for CFD simulation. Eli Lilly CNS Drug Delivery program.

## Quick Commands
```bash
# Activate env
source .venv/bin/activate

# Train (Phase 1 baseline)
python scripts/train.py --config configs/train_spine.yaml

# Evaluate
python scripts/evaluate.py --config configs/inference.yaml

# Export mesh from prediction
python scripts/export_mesh.py --input <nifti> --output <stl>

# Run tests
pytest tests/ -v

# Run demo
streamlit run src/demo/app.py
```

## Project Layout
```
src/data/       — BIDS data loaders, MONAI transforms
src/models/     — SegResNet factory, MC-Dropout wrapper
src/losses/     — Soft clDice, CombinedLoss
src/training/   — Training loop with MLflow tracking
src/evaluation/ — Dice/HD95 metrics, ECE calibration
src/mesh/       — Mask → watertight STL pipeline
src/demo/       — Streamlit app
configs/        — YAML configs (train, inference)
scripts/        — CLI entry points
tests/          — Unit tests
```

## Tech Stack
- Python 3.10+, PyTorch 2.x, MONAI
- Architecture: SegResNet (3D)
- Loss: DiceCE + Soft clDice
- Uncertainty: MC-Dropout (8 passes)
- Mesh: marching cubes + trimesh repair
- Tracking: MLflow
- Compute: auto-detects best available device (MPS > CUDA > CPU)

## Critical Rules
1. **Device handling:** Use `get_device()` from `src.models.segresnet` everywhere — it auto-selects MPS > CUDA > CPU. Never hardcode a device string. Use `empty_cache(device)` (same module) for accelerator memory cleanup instead of calling `torch.mps.empty_cache()` / `torch.cuda.empty_cache()` directly — it's a no-op on CPU. Test MPS ops individually — some MONAI ops fail on MPS.
2. **Git-annex data:** Check `os.path.getsize(path) > 1000` before loading any NIfTI — annex pointer stubs are ~100 bytes.
3. **Validation splits:** Always split by SITE, never randomly. This proves cross-vendor generalization.
4. **Spine labels location:** `derivatives/labels/sub-*/anat/*_T2w_label-SC_seg.nii.gz` — NOT next to images.
5. **Patch-based training:** Use `RandCropByPosNegLabel` with pos:neg ≥ 2:1 for spine.
6. **Mesh repair order:** fill holes → fix normals → remove degenerate → smooth. Never smooth first.
7. **clDice iterations:** `iter_=3` is sufficient for spinal cord.

## Development Phases
Each phase has a self-contained doc in `docs/phases/`:
- `docs/phases/phase1.md` — Baseline & Data (Weeks 1-2)
- `docs/phases/phase2.md` — Topology-Aware Training + Mesh (Weeks 3-4)
- `docs/phases/phase3.md` — Uncertainty Quantification (Weeks 5-6)
- `docs/phases/phase4.md` — Demo, Evaluation & Documentation (Weeks 7-8)

**Start a new session for a phase?** Read only that phase's doc + this file. No need to read CONTEXT.md.

## Coding Style
- Type hints on all function signatures
- Docstrings (Google style) on public functions
- Config-driven (YAML) — no hardcoded hyperparameters
- Use `rich` for CLI progress output
- Logging via Python `logging` module, not print statements
- All paths via `pathlib.Path`

## Targets
| Metric | Target |
|--------|--------|
| Spinal cord Dice | ≥ 0.93 |
| Mesh watertight | 100% |
| Mesh manifold | 100% |
| Uncertainty ECE | < 0.05 |
| Inter-vendor CV | < 5% |
| Inference time | < 5 min/case |
