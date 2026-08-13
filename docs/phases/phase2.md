# Phase 2 — Topology-Aware Training + Mesh Export (Weeks 3–4)

## Goal
Add Soft clDice loss for topological correctness; build full mesh export pipeline producing watertight, manifold STLs.

## Prerequisites
- Phase 1 complete: SegResNet trained, baseline Dice ≥ 0.93
- Trained model checkpoint available in `experiments/`

## Deliverables
1. Soft clDice loss implementation (3D morphological skeletonization)
2. CombinedLoss (DiceCE + clDice, configurable weights)
3. 3-way ablation: DiceCE-only vs clDice-only vs Combined
4. Mesh pipeline: mask → marching cubes → repair → smooth → validate → STL
5. 100% watertight + manifold pass rate

## Soft clDice Implementation
```python
# Key algorithm: 3D soft skeletonization via iterative erosion
# 1. Apply 3D min-pool (erosion) iteratively
# 2. Subtract erosion from original → approximate skeleton
# 3. clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
#    where Tprec = skeleton(pred) masked by GT
#          Tsens = skeleton(GT) masked by pred
# iter_=3 is sufficient for spinal cord
```

## CombinedLoss
```python
loss = alpha * DiceCELoss + beta * SoftClDiceLoss
# Default: alpha=1.0, beta=0.5
# Ablation tests: (1,0), (0,1), (1,0.5)
```

## Mesh Pipeline Steps
1. **Threshold** prediction at 0.5
2. **Largest connected component** (remove floating artifacts)
3. **Marching cubes** (scikit-image) → vertices + faces
4. **Fill holes** (trimesh)
5. **Fix normals** (consistent winding)
6. **Remove degenerate faces** (zero-area)
7. **Laplacian smooth** (5 iterations, λ=0.5)
8. **Validate:** is_watertight + is_manifold (trimesh)
9. **Export STL**

Order matters: fill holes → fix normals → remove degenerate → smooth. Never smooth first.

## Mesh Validation Criteria
- `mesh.is_watertight == True`
- `mesh.is_volume == True` (implies manifold)
- No degenerate faces (area > 0)
- Euler characteristic χ = 2 (for genus-0 surface, i.e., topological sphere)

## Files to Create/Modify
```
src/losses/topology.py       — SoftClDice3D + CombinedLoss
src/mesh/export.py           — Full mesh pipeline
tests/test_losses.py         — clDice unit tests
tests/test_mesh.py           — Mesh validation tests
configs/train_spine.yaml     — Update with combined loss params
scripts/export_mesh.py       — CLI mesh export
```

## Success Criteria
- Combined loss Dice ≥ DiceCE-only Dice (no regression)
- 100% watertight rate across all validation predictions
- 100% manifold rate
- Mesh export < 30 seconds per volume
