# Phase 4 — Demo, Evaluation & Documentation (Weeks 7–8)

## Goal
Full end-to-end evaluation, comparison with baselines, Streamlit dashboard, and technical documentation.

## Prerequisites
- Phases 1-3 complete: trained model, mesh pipeline, uncertainty quantification all working
- All target metrics achieved

## Deliverables
1. Full evaluation on held-out test subjects
2. Comparison table: Our pipeline vs SCT DeepSeg vs TotalSegmentator-MRI
3. Streamlit dashboard (load MRI → segment → mesh → uncertainty)
4. Technical report (methods, results, limitations)
5. Phase 0 transition plan

## Evaluation Protocol
- Hold-out test set: subjects from val sites not used during development
- Metrics per subject: Dice, HD95 (Hausdorff 95%), volume error
- Metrics per site: mean ± std, to show cross-vendor consistency
- Inter-vendor CV < 5%
- Mesh metrics: watertight rate, manifold rate, vertex count, smoothness

## Comparison Baselines
1. **SCT DeepSeg** (Gros et al. 2019): Install via `pip install spinalcordtoolbox`, run `sct_deepseg_sc`
2. **TotalSegmentator-MRI** (Wasserthal et al. 2023): Run on same test cases
3. Report: Dice, HD95, inference time for each

## Streamlit Dashboard Features
- Upload NIfTI or select from test subjects
- Run segmentation (with progress bar)
- 3D volume rendering (axial/sagittal/coronal slices)
- Uncertainty overlay visualization
- Mesh preview (if possible via plotly)
- Export: segmentation NIfTI, uncertainty NIfTI, STL mesh
- Per-case metrics display

## Technical Report Outline
1. Introduction & Clinical Motivation
2. Methods (Architecture, Loss, Training, Mesh, Uncertainty)
3. Results (tables + figures)
4. Discussion (limitations, failure modes)
5. Future Work (rootlets, CSF segmentation, multi-contrast)

## Files to Create/Modify
```
src/demo/app.py              — Streamlit dashboard
scripts/evaluate.py          — Full evaluation pipeline
src/evaluation/metrics.py    — Dice, HD95, volume metrics
docs/report.md               — Technical report
docs/transition_plan.md      — Phase 0 handoff doc
```

## Success Criteria
- All metrics meet targets from CLAUDE.md
- Dashboard runs end-to-end on a test case
- Report is complete and figures are publication-ready
- Transition plan identifies next steps (CSF seg, rootlets, production deployment)
