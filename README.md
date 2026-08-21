# CNS Segmentation Pipeline

Automated spinal cord and CSF segmentation from T2-weighted MRI for computational fluid dynamics (CFD) simulation of intrathecal drug delivery.

## Overview

This pipeline:

1. Takes T2-weighted spinal MRI as input
2. Segments spinal cord, spinal canal, CSF, and nerve rootlets using SegResNet (3D)
3. Exports watertight, manifold STL meshes suitable for CFD (via `cns_segmentation.mesh.export`,
   consumed directly by the sibling `cns-cfd-simulation` repo)
4. Uncertainty quantification (MC-Dropout, ECE calibration) is planned but not yet implemented —
   see [Not Yet Implemented](#not-yet-implemented-planned) below

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

# Train a per-structure model — canal/rootlets/csf (configs/train_spine.yaml alone is a
# cord-only baseline; see CLAUDE.md's Quick Commands for the full set)
python scripts/train.py --config configs/train_spine_canal.yaml

# Predict on held-out spine-generic sites
python scripts/predict.py --config configs/inference_canal.yaml --checkpoint <path>

# Score a checkpoint against an external validation dataset (SPIDER, Al-Kafri, OpenNeuro
# ds004507 — see data/ATTRIBUTIONS.md)
python scripts/evaluate_external.py --dataset spider_canal --structure canal \
    --train-config configs/train_spine_canal.yaml --inference-config configs/inference_canal.yaml \
    --checkpoint <path>

# Run tests
pytest tests/ -v
```

### Not Yet Implemented (planned)

- `scripts/export_mesh.py` CLI wrapper — the underlying `export_cfd_mesh()` function it would
  wrap already exists and is used directly by `cns-cfd-simulation/scripts/run_full_pipeline.py`;
  only the standalone CLI entry point in this repo is missing.
- `scripts/evaluate.py` (full evaluation pipeline) and `src/cns_segmentation/demo/app.py`
  (Streamlit dashboard) — Phase 4 scope, not started.
- MC-Dropout uncertainty quantification and ECE calibration — Phase 3 scope, not started.

## Project Structure

```
src/cns_segmentation/data/       — BIDS-aware data loaders, MONAI transforms
src/cns_segmentation/models/     — SegResNet factory, MC-Dropout wrapper
src/cns_segmentation/losses/     — Soft clDice topology loss
src/cns_segmentation/training/   — Training loop with MLflow tracking
src/cns_segmentation/evaluation/ — Metrics (Dice, HD95, ECE)
src/cns_segmentation/mesh/       — Mask → watertight STL mesh pipeline
src/cns_segmentation/demo/       — Streamlit dashboard
configs/                         — YAML training/inference configs
scripts/                         — CLI entry points
```

## Key Metrics

| Metric                 | Target       | Achieved                                                                  |
| ---------------------- | ------------ | -------------------------------------------------------------------------- |
| Cord Dice              | ≥ 0.93      | **0.9507** (Phase 0/1, `spine_generic_cord`, held-out sites)               |
| Canal Dice             | ≥ 0.93      | **0.9501** (Phase 1, `spine_generic_canal`, held-out sites)                |
| CSF Dice               | ≥ 0.93      | **0.5512** (Phase 1, `spine_generic_csf`) — below target, known gap        |
| Rootlets Dice          | ≥ 0.93      | **0.6121** (Phase 1, `spine_generic_rootlets`) — below target, known gap   |
| Cross-dataset canal Dice (`spider_canal`, lumbar) | n/a | **0.0905** — real generalization test, out-of-region (cervical-thoracic → lumbar); see ROADMAP.md |
| Mesh watertight        | 100%         | — not yet measured (`scripts/export_mesh.py` CLI not built; underlying `export_cfd_mesh()` is exercised only via `cns-cfd-simulation`) |
| Uncertainty ECE        | < 0.05       | — not started (Phase 3)                                                    |
| Inference time         | < 5 min      | — not yet benchmarked                                                      |

See `ROADMAP.md` for full per-run provenance (checkpoint paths, MLflow run IDs, SLURM job IDs).

## References

- Cohen-Adad et al. 2021 — Spine-Generic dataset (DOI: 10.1038/s41597-021-00941-8)
- Montoya/Teli et al. 2024 — Lilly CSF modeling (DOI: 10.1002/alz.094612)
- Shit et al. 2021 — clDice topology loss (arXiv: 2003.07311)
- van der Graaf et al. 2024 — SPIDER spinal disorders dataset (DOI: 10.5281/zenodo.10159290)
- Al-Kafri, Sudirman et al. 2019 — Lumbar spine thecal-sac segmentation (DOI: 10.1109/ACCESS.2019.2908002)
- OpenNeuro ds004507 — Spinal Cord Head Positions (https://openneuro.org/datasets/ds004507)

Full attribution/license text for the three datasets above: `data/ATTRIBUTIONS.md`.
