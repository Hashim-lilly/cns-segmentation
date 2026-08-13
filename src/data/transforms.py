"""MONAI transform pipelines for spinal cord segmentation training and validation.

Provides factory functions that build deterministic and augmented transform
chains driven by values in the training configuration dictionary.
"""

import logging
from typing import Any

from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    Spacingd,
    SpatialPadd,
)

logger = logging.getLogger(__name__)


def get_train_transforms(config: dict[str, Any]) -> Compose:
    """Build the training transform pipeline with augmentation.

    Args:
        config: Training configuration dictionary. Relevant keys:
            - spacing (list[float]): Target voxel spacing. Default [1.0, 0.5, 0.5].
            - patch_size (list[int]): Random crop spatial size. Default [48, 160, 160].
            - num_samples (int): Crops per volume. Default 4.

    Returns:
        A MONAI Compose pipeline suitable for training dataloaders.
    """
    spacing = config.get("spacing", [1.0, 0.5, 0.5])
    patch_size = config.get("patch_size", [48, 160, 160])
    num_samples = config.get("num_samples", 4)

    logger.info(
        "Building train transforms: spacing=%s, patch_size=%s, num_samples=%d",
        spacing,
        patch_size,
        num_samples,
    )

    keys = ["image", "label"]

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True),
            SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant"),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=patch_size,
                pos=2,
                neg=1,
                num_samples=num_samples,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=[1, 2]),
            RandRotate90d(keys=keys, prob=0.3, spatial_axes=(1, 2)),
            RandGaussianNoised(keys="image", prob=0.2, std=0.05),
        ]
    )


def get_val_transforms(config: dict[str, Any]) -> Compose:
    """Build the validation transform pipeline (deterministic only).

    Args:
        config: Training configuration dictionary. Relevant keys:
            - spacing (list[float]): Target voxel spacing. Default [1.0, 0.5, 0.5].

    Returns:
        A MONAI Compose pipeline suitable for validation dataloaders.
    """
    spacing = config.get("spacing", [1.0, 0.5, 0.5])

    logger.info("Building val transforms: spacing=%s", spacing)

    keys = ["image", "label"]

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True),
        ]
    )


def get_post_transforms(num_classes: int = 2) -> dict[str, Compose]:
    """Build post-processing transforms for predictions and labels.

    Args:
        num_classes: Number of segmentation classes (including background).
            Default 2 (background + spinal cord).

    Returns:
        Dictionary with keys ``"pred"`` and ``"label"``, each mapping to a
        Compose pipeline that discretizes model outputs appropriately.
    """
    logger.info("Building post transforms: num_classes=%d", num_classes)

    pred_transforms = Compose([AsDiscrete(argmax=True)])
    label_transforms = Compose([AsDiscrete(to_onehot=num_classes)])

    return {"pred": pred_transforms, "label": label_transforms}
