# Phase 1 — Baseline & Data (Weeks 1–2)

## Goal
Train SegResNet on T2w spinal cord binary segmentation. Establish baseline Dice on site-stratified validation.

## Deliverables
1. ✅ Project scaffold + environment
2. ✅ BIDS-aware data loader with git-annex handling
3. ✅ MONAI preprocessing pipeline
4. ✅ SegResNet training on spinal cord segmentation
5. ✅ Baseline Dice ≥ 0.93 on multi-site validation

## Data
- **Dataset:** Spine-Generic multi-subject (`data/spine-generic/`)
- **Input:** `sub-*/anat/sub-*_T2w.nii.gz` (T2-weighted MRI)
- **Labels:** `derivatives/labels/sub-*/anat/*_T2w_label-SC_seg.nii.gz` (binary spinal cord mask)
- **Format:** NIfTI, BIDS-compliant
- **Annex handling:** Files <1000 bytes are pointer stubs → skip them

## Architecture: SegResNet
```python
# MONAI SegResNet config
SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,          # background + cord
    init_filters=32,
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    dropout_prob=0.2,        # needed for Phase 3 MC-Dropout
)
```

## Training Config
```yaml
# Key parameters
patch_size: [48, 160, 160]
batch_size: 2
epochs: 100
lr: 1e-4
optimizer: AdamW
scheduler: CosineAnnealingLR
loss: DiceCELoss (MONAI built-in, Phase 1 only)
pos_neg_ratio: 2  # for RandCropByPosNegLabel
num_samples: 4    # patches per volume per epoch
```

## Validation Strategy
Split by SITE (not random):
- **Train sites:** amu, balgrist, barcelona, beijing, brno, cardiff, chiba, cmrra, douglas, geneva, juntendo, hamburg, mgh, mountsinai, nottingham, oxford, pavia, perform, sherbrooke, strasbourg, tokyo, ucl, vuiis
- **Val sites:** stanford, tehranS, ubc, ucdavis, unf (5 sites, diverse vendors)

## MONAI Transforms Pipeline
### Training:
1. LoadImaged → load NIfTI
2. EnsureChannelFirstd
3. Orientationd(axcodes="RAS")
4. Spacingd(pixdim=[1.0, 0.5, 0.5]) — resample to consistent spacing
5. NormalizeIntensityd(nonzero=True)
6. RandCropByPosNegLabeld(spatial_size=[48,160,160], pos=2, neg=1, num_samples=4)
7. RandFlipd(prob=0.5, spatial_axis=[1,2])
8. RandRotate90d(prob=0.3, spatial_axis=(1,2))
9. RandGaussianNoised(prob=0.2, std=0.05)

### Validation:
1-5 same as train (deterministic)
6. SlidingWindowInferer(roi_size=[48,160,160], overlap=0.5)

## Device Handling
```python
from src.models.segresnet import get_device, empty_cache

device = get_device()  # MPS > CUDA > CPU, whichever is best available
# Some MONAI ops need CPU fallback — wrap in try/except
# Use empty_cache(device) instead of torch.mps/cuda.empty_cache() directly
```

## Success Criteria
- Dice ≥ 0.93 on validation sites
- Training completes without OOM on the available device (MPS/CUDA/CPU)
- Data loader handles missing/stub files gracefully
- Reproducible via config YAML

## Files to Create
```
src/data/spine_generic.py    — BIDS data discovery + git-annex check
src/data/transforms.py       — MONAI transform pipelines
src/models/segresnet.py      — Model factory
src/training/trainer.py      — Training loop
configs/train_spine.yaml     — Training config
scripts/train.py             — CLI entry point
tests/test_transforms.py     — Transform unit tests
```

## Reference Metrics (PoC)
Prior PoC achieved Dice 0.951 ± 0.012 (30 epochs, 24 train / 6 val subjects).
With more epochs and proper augmentation, should be achievable.
