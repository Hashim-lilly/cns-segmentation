# Phase 3 — Uncertainty Quantification (Weeks 5–6)

## Goal
Implement MC-Dropout uncertainty estimation with calibrated confidence maps. Target ECE < 0.05.

## Prerequisites
- Phase 2 complete: CombinedLoss trained model, mesh pipeline working
- Best model checkpoint with dropout_prob=0.2 in architecture

## Deliverables
1. MC-Dropout inference wrapper (N forward passes with dropout enabled)
2. Uncertainty maps: predictive entropy, mutual information, variance
3. ECE computation with 15-bin reliability diagrams
4. Uncertainty maps saved as NIfTI (overlayable in 3D Slicer)
5. ECE < 0.05

## MC-Dropout Method
```python
# Key concept: Enable dropout at inference time
# Run N stochastic forward passes → get distribution of predictions
# Uncertainty = disagreement across passes

model.train()  # keeps dropout active
predictions = []
for _ in range(N):  # N=8
    with torch.no_grad():
        pred = model(x)
        predictions.append(torch.softmax(pred, dim=1))

stacked = torch.stack(predictions)  # [N, B, C, D, H, W]
mean_pred = stacked.mean(dim=0)     # [B, C, D, H, W]
```

## Uncertainty Metrics
1. **Predictive entropy:** H[ȳ] = -Σ ȳ_c * log(ȳ_c) — total uncertainty
2. **Mutual information:** I[y, θ|x] = H[ȳ] - E[H[y|θ]] — epistemic (model) uncertainty
3. **Variance:** Var across MC samples per voxel — simple spread measure

## ECE (Expected Calibration Error)
```python
# 15-bin reliability diagram
# For each bin:
#   accuracy = fraction of voxels in bin that are correctly classified
#   confidence = mean predicted probability in bin
#   ECE += (n_bin / n_total) * |accuracy - confidence|
```

## NIfTI Export
- Save uncertainty maps with same affine as input MRI
- Channels: entropy, mutual_info, variance (separate files or 4D NIfTI)
- Viewable as overlay in 3D Slicer, FSLeyes, or ITK-SNAP

## Files to Create/Modify
```
src/models/uncertainty.py     — MCDropoutWrapper class
src/evaluation/calibration.py — ECE, reliability diagrams
scripts/uncertainty.py        — CLI: run uncertainty analysis
configs/inference.yaml        — Update with MC params (n_samples=8)
tests/test_uncertainty.py     — Unit tests
```

## Success Criteria
- ECE < 0.05
- Uncertainty maps correctly highlight boundaries and ambiguous regions
- High uncertainty correlates with actual segmentation errors
- Runs in < 5 min per case (8 passes × sliding window)
