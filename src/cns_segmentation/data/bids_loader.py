"""BIDS data loader for the spine-generic multi-site T2w dataset."""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from monai.data import CacheDataset, DataLoader, Dataset

logger = logging.getLogger(__name__)

# Minimum file size to distinguish real NIfTI from git-annex pointer stubs
_MIN_NIFTI_BYTES = 1000


def _is_real_nifti(path: Path) -> bool:
    """Return True if path is a real NIfTI file (not a git-annex pointer stub)."""
    return path.exists() and os.path.getsize(path) > _MIN_NIFTI_BYTES


def _collect_subjects(
    root: Path,
    sites: list[str],
    contrast: str,
    label_key: str,
) -> list[dict]:
    """Collect image/label pairs for the given sites.

    Args:
        root: BIDS root directory of the spine-generic dataset.
        sites: List of site prefixes (e.g. ["amu", "balgrist"]).
        contrast: MRI contrast key (e.g. "T2w").
        label_key: Label suffix to look for under derivatives/labels.

    Returns:
        List of dicts with "image" and "label" Path entries.
    """
    records = []
    for sub_dir in sorted(root.glob("sub-*")):
        subject = sub_dir.name  # e.g. "sub-amu01"
        site = subject.split("-")[1].rstrip("0123456789")
        if site not in sites:
            continue

        image = sub_dir / "anat" / f"{subject}_{contrast}.nii.gz"
        label = (
            root
            / "derivatives"
            / "labels"
            / subject
            / "anat"
            / f"{subject}_{contrast}_{label_key}.nii.gz"
        )

        if not _is_real_nifti(image):
            logger.debug("Skipping %s: image not available (annex stub or missing)", subject)
            continue
        if not _is_real_nifti(label):
            logger.debug("Skipping %s: label not available (annex stub or missing)", subject)
            continue

        records.append({"image": str(image), "label": str(label)})

    return records


class SpineGenericDataset:
    """Spine-generic BIDS dataset split by acquisition site.

    Attributes:
        train_files: List of train image/label dicts.
        val_files: List of validation image/label dicts.
    """

    def __init__(
        self,
        root_dir: Path | str,
        train_sites: list[str],
        val_sites: list[str],
        contrast: str = "T2w",
        label_key: str = "label-SC_seg",
    ) -> None:
        """Initialize dataset from BIDS root.

        Args:
            root_dir: Path to the BIDS root (contains sub-* directories).
            train_sites: Site prefixes for the training split.
            val_sites: Site prefixes for the validation split.
            contrast: MRI contrast identifier.
            label_key: Label file suffix used in derivatives/labels.
        """
        root = Path(root_dir)
        self.train_files = _collect_subjects(root, train_sites, contrast, label_key)
        self.val_files = _collect_subjects(root, val_sites, contrast, label_key)
        logger.info(
            "SpineGenericDataset: %d train, %d val subjects",
            len(self.train_files),
            len(self.val_files),
        )


def build_dataloaders(
    dataset: SpineGenericDataset,
    train_transforms,
    val_transforms,
    batch_size: int = 2,
    num_workers: int = 4,
    cache_rate: float = 0.0,
) -> tuple[DataLoader, DataLoader]:
    """Build MONAI DataLoaders for training and validation.

    Args:
        dataset: SpineGenericDataset with train/val file lists.
        train_transforms: MONAI Compose transform for training.
        val_transforms: MONAI Compose transform for validation.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker processes.
        cache_rate: Fraction of data to cache in memory (0 = no cache).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    ds_cls = CacheDataset if cache_rate > 0 else Dataset

    train_ds = (
        ds_cls(dataset.train_files, transform=train_transforms, cache_rate=cache_rate)
        if cache_rate > 0
        else Dataset(dataset.train_files, transform=train_transforms)
    )
    val_ds = (
        ds_cls(dataset.val_files, transform=val_transforms, cache_rate=cache_rate)
        if cache_rate > 0
        else Dataset(dataset.val_files, transform=val_transforms)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
