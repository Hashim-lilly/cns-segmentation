# Phase 1 (CFD) — Automated Spinal SAS Segmentation for CFD

## Goal
Produce watertight, manifold 3D meshes of the spinal subarachnoid space
(foramen magnum to thecal sac) from T2w MRI using pre-trained models,
suitable for OpenFOAM pulsatile CSF flow simulation.

## Strategy: Leverage Existing Models
Use battle-tested pre-trained models as the primary pipeline.
Fine-tune with nnU-Net only where gaps exist.

## Pipeline
```
T2w MRI → TotalSpineSeg (cord + canal) 
        → model-canal-seg (dural sac)
        → RootletSeg (nerve rootlets)
        → Boolean: CSF = canal − cord − rootlets
        → Mesh export (marching cubes + repair + smooth)
        → Validate against Sass 2017 reference
```

## Key Models
| Model | Structures | Source |
|-------|-----------|--------|
| TotalSpineSeg | cord, canal, vertebrae, IVDs | sct_deepseg totalspineseg |
| model-canal-seg | dural sac (outer boundary) | ivadomed/model-canal-seg |
| RootletSeg | dorsal/ventral rootlets C2-T1 | ivadomed/model-spinal-rootlets |
| SCT contrast-agnostic | spinal cord | sct_deepseg seg_sc_contrast_agnostic |

## Data Sources
- **spine-generic** — cervical cord training/validation (260 subjects)
- **SPIDER** — lumbar injection region (218 patients, T2 SPACE)
- **Sass 2017 model** — reference geometry for validation (STL/OBJ)

## Validation Targets
| Metric | Target | Reference |
|--------|--------|-----------|
| Cord Dice | ≥ 0.93 | spine-generic published |
| Canal Dice | ≥ 0.85 | TotalSpineSeg published |
| Rootlet Dice | ≥ 0.65 | Valošek 2024: 0.67±0.16 |
| Mesh watertight | 100% | trimesh validation |
| Mesh manifold | 100% | trimesh validation |
| CSF volume | ±10% of Sass (97.3 cm³) | Sass 2017 Table 3 |
| Euler χ | = 2 | topological sphere |

## CFD Framework
- **Solver:** OpenFOAM pimpleFoam (transient laminar pulsatile)
- **BCs:** Sass 2017 published CSF waveforms (C2-C3, C7-T1, T10-T11)
- **Transport:** scalarTransportFoam (passive tracer)
- **Validation:** Re and Womersley profiles vs Sass Fig 7

## Files Created
```
src/segmentation/__init__.py        — Pipeline package
src/segmentation/model_registry.py  — Pre-trained model registry
src/segmentation/pipeline.py        — Pipeline orchestrator
src/mesh/export.py                  — CFD-grade mesh export
src/mesh/cfd_domain.py              — Boolean CSF domain extraction
src/evaluation/geometry_metrics.py  — Sass reference validation
scripts/run_pipeline.py             — CLI entry point (single + batch)
```

## Device Handling
Segmentation: runs on CPU (SCT models are CPU-compatible).
MPS for any custom nnU-Net fine-tuning later.

## Success Criteria
- End-to-end: MRI → segmentation → CFD mesh in < 30 min per case
- First watertight SAS mesh produced from fully automated pipeline
- Geometry validated against Sass 2017 reference model
