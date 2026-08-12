"""MONAI transform pipelines for training and validation."""

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    ToTensord,
)


def build_train_transforms(
    patch_size: list[int],
    spacing: list[float],
    num_samples: int = 4,
    pos_neg_ratio: float = 2.0,
) -> Compose:
    """Build training augmentation pipeline.

    Args:
        patch_size: 3-D patch dimensions [D, H, W].
        spacing: Target voxel spacing [z, y, x] in mm.
        num_samples: Patches sampled per volume per batch call.
        pos_neg_ratio: Ratio of foreground to background patches.

    Returns:
        MONAI Compose transform.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=pos_neg_ratio,
                neg=1.0,
                num_samples=num_samples,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )


def build_val_transforms(spacing: list[float]) -> Compose:
    """Build deterministic validation transform pipeline.

    Args:
        spacing: Target voxel spacing [z, y, x] in mm.

    Returns:
        MONAI Compose transform.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
