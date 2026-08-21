"""MONAI transform pipelines for spinal cord segmentation training and validation.

Provides factory functions that build deterministic and augmented transform
chains driven by values in the training configuration dictionary.
"""

import logging
from typing import Any, Optional

from monai.transforms import (
    AsDiscrete,
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
    SpatialPadd,
)

from cns_segmentation.data.label_compositing import CompositeLabeld

logger = logging.getLogger(__name__)


def _build_io_keys(structures: Optional[list[str]]) -> tuple[list[str], list[str]]:
    """Determine the raw label keys to load and the image+label key list.

    Args:
        structures: Structure names for a multi-class run, or None for the
            legacy single-"label" cord-only shape.

    Returns:
        Tuple of (label_raw_keys, load_keys) where `load_keys` includes
        "image" plus every raw label key to hand to `LoadImaged`.
    """
    if structures:
        label_raw_keys = [f"label_{s}" for s in structures]
    else:
        label_raw_keys = ["label"]
    return label_raw_keys, ["image"] + label_raw_keys


def get_train_transforms(
    config: dict[str, Any],
    structures: Optional[list[str]] = None,
) -> Compose:
    """Build the training transform pipeline with augmentation.

    Args:
        config: Training configuration dictionary. Relevant keys:
            - spacing (list[float]): Target voxel spacing. Default [1.0, 0.5, 0.5].
            - patch_size (list[int]): Random crop spatial size. Default [48, 160, 160].
            - num_samples (int): Crops per volume. Default 4.
            - augmentation (dict): Per-transform probabilities/params (see
              below). Any key not present falls back to a default that
              reproduces this function's pre-existing hardcoded behavior,
              so an empty/missing "augmentation" block is a no-op change
              from before this parameter existed.
        structures: For a multi-class run, the list of structure names
            (e.g. ["cord", "canal"]) whose separately-loaded
            "label_<structure>" keys should be composited into one "label"
            volume via `CompositeLabeld`. None (default) preserves the
            legacy cord-only single-"label" pipeline shape exactly.

    Returns:
        A MONAI Compose pipeline suitable for training dataloaders.
    """
    spacing = config.get("spacing", [1.0, 0.5, 0.5])
    patch_size = config.get("patch_size", [48, 160, 160])
    num_samples = config.get("num_samples", 4)
    aug = config.get("augmentation", {})

    logger.info(
        "Building train transforms: spacing=%s, patch_size=%s, num_samples=%d, structures=%s",
        spacing,
        patch_size,
        num_samples,
        structures,
    )

    label_raw_keys, load_keys = _build_io_keys(structures)

    transforms: list[Any] = [
        LoadImaged(keys=load_keys),
        EnsureChannelFirstd(keys=load_keys),
    ]
    if structures:
        transforms.append(
            CompositeLabeld(
                structure_keys={s: f"label_{s}" for s in structures},
                output_key="label",
            )
        )
        transforms.append(DeleteItemsd(keys=label_raw_keys))

    keys = ["image", "label"]
    transforms += [
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
            pos=aug.get("rand_crop_pos", 2),
            neg=aug.get("rand_crop_neg", 1),
            num_samples=num_samples,
        ),
        RandFlipd(
            keys=keys,
            prob=aug.get("rand_flip_prob", 0.5),
            spatial_axis=aug.get("rand_flip_axes", [1, 2]),
        ),
        RandRotate90d(
            keys=keys,
            prob=aug.get("rand_rotate90_prob", 0.3),
            spatial_axes=tuple(aug.get("rand_rotate90_axes", [1, 2])),
        ),
        # Contrast-agnostic augmentation (Bedard/Cohen-Adad 2025-style domain
        # randomization). Off by default (prob=0.0) so the cord-only
        # baseline config is unaffected; multi-class configs enable these.
        RandAdjustContrastd(
            keys="image",
            prob=aug.get("rand_adjust_contrast_prob", 0.0),
            gamma=tuple(aug.get("rand_adjust_contrast_gamma", [0.7, 1.5])),
        ),
        RandBiasFieldd(
            keys="image",
            prob=aug.get("rand_bias_field_prob", 0.0),
            coeff_range=tuple(aug.get("rand_bias_field_coeff_range", [0.0, 0.3])),
        ),
        RandGaussianSmoothd(
            keys="image",
            prob=aug.get("rand_gaussian_smooth_prob", 0.0),
            sigma_x=tuple(aug.get("rand_gaussian_smooth_sigma", [0.25, 1.5])),
            sigma_y=tuple(aug.get("rand_gaussian_smooth_sigma", [0.25, 1.5])),
            sigma_z=tuple(aug.get("rand_gaussian_smooth_sigma", [0.25, 1.5])),
        ),
        RandScaleIntensityd(
            keys="image",
            prob=aug.get("rand_scale_intensity_prob", 0.0),
            factors=aug.get("rand_scale_intensity_factors", 0.1),
        ),
        RandShiftIntensityd(
            keys="image",
            prob=aug.get("rand_shift_intensity_prob", 0.0),
            offsets=aug.get("rand_shift_intensity_offsets", 0.1),
        ),
        RandGaussianNoised(
            keys="image",
            prob=aug.get("rand_gaussian_noise_prob", 0.2),
            std=aug.get("rand_gaussian_noise_std", 0.05),
        ),
    ]

    return Compose(transforms)


def get_val_transforms(
    config: dict[str, Any],
    structures: Optional[list[str]] = None,
) -> Compose:
    """Build the validation transform pipeline (deterministic only).

    Args:
        config: Training configuration dictionary. Relevant keys:
            - spacing (list[float]): Target voxel spacing. Default [1.0, 0.5, 0.5].
        structures: Same meaning as in `get_train_transforms`.

    Returns:
        A MONAI Compose pipeline suitable for validation dataloaders.
    """
    spacing = config.get("spacing", [1.0, 0.5, 0.5])

    logger.info("Building val transforms: spacing=%s, structures=%s", spacing, structures)

    label_raw_keys, load_keys = _build_io_keys(structures)

    transforms: list[Any] = [
        LoadImaged(keys=load_keys),
        EnsureChannelFirstd(keys=load_keys),
    ]
    if structures:
        transforms.append(
            CompositeLabeld(
                structure_keys={s: f"label_{s}" for s in structures},
                output_key="label",
            )
        )
        transforms.append(DeleteItemsd(keys=label_raw_keys))

    keys = ["image", "label"]
    transforms += [
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True),
    ]

    return Compose(transforms)


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
