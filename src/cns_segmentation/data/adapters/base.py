"""Shared contract for dataset adapters.

Each adapter module (`spider.py`, `alkafri_mendeley.py`, etc.) downloads or
locates a foreign-format dataset and materializes it into a spine-generic-
shaped BIDS-derivatives tree — `sub-<tag><NNNN>/anat/sub-<tag><NNNN>_T2w.nii.gz`
plus `derivatives/labels/sub-<tag><NNNN>/anat/sub-<tag><NNNN>_T2w_label-<SUFFIX>.nii.gz`
— then returns a `DatasetSpec` pointing at that root. This is what lets
`spine_generic.create_datalist()` and `dataset_registry.label_path()` consume
the new dataset with zero changes to either function: both are hardcoded to
this exact directory shape, not to spine-generic specifically.

Subject dirnames use a per-dataset tag prefix (e.g. "spider", "alkafri") so
they can never collide with spine-generic's own `sub-<hospital><n>` subjects,
and still match `spine_generic._SITE_PATTERN` (letters then digits only).
"""

import logging
from pathlib import Path
from typing import Protocol

from cns_segmentation.data.dataset_registry import DatasetSpec

logger = logging.getLogger(__name__)


class DatasetAdapter(Protocol):
    """Protocol every `data/adapters/<name>.py` module implements at module level."""

    def prepare(self, force: bool = False) -> DatasetSpec:
        """Materialize the dataset on disk and return its `DatasetSpec`.

        Args:
            force: If True, re-download/re-materialize even if `is_prepared()`
                already reports the target root as complete. If False
                (default), skip that work and just return the spec.

        Returns:
            DatasetSpec describing the materialized dataset root.
        """
        ...


def subject_dirname(tag: str, index: int) -> str:
    """Build a collision-safe BIDS subject dirname.

    Args:
        tag: Dataset-specific alphabetic tag, e.g. "spider".
        index: 1-based subject index within the dataset.

    Returns:
        BIDS subject dirname, e.g. "sub-spider0001".
    """
    return f"sub-{tag}{index:04d}"


def write_bids_subject(
    root: Path,
    subject_id: str,
    image_bytes: bytes,
    labels: dict[str, bytes],
    contrast: str = "T2w",
) -> None:
    """Materialize one subject's image + per-structure label files.

    Args:
        root: Dataset root directory (matches the eventual `DatasetSpec.root`).
        subject_id: BIDS subject dirname, e.g. "sub-spider0001".
        image_bytes: Raw NIfTI (.nii.gz) bytes for the subject's image.
        labels: Maps BIDS label suffix (e.g. "canal_seg") to raw NIfTI label
            bytes. May be empty for images-only / comparison_only datasets.
        contrast: Image contrast tag used in filenames. Defaults to "T2w".
    """
    anat_dir = root / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / f"{subject_id}_{contrast}.nii.gz").write_bytes(image_bytes)

    if not labels:
        return
    label_dir = root / "derivatives" / "labels" / subject_id / "anat"
    label_dir.mkdir(parents=True, exist_ok=True)
    for suffix, data in labels.items():
        (label_dir / f"{subject_id}_{contrast}_label-{suffix}.nii.gz").write_bytes(data)


def is_prepared(root: Path, expected_subject_count: int, min_file_size: int = 1000) -> bool:
    """Check whether `root` already looks like a complete materialized dataset.

    Lets an adapter's `prepare(force=False)` skip redundant re-download/
    re-materialization work. Counts subject directories under
    `derivatives/labels/` (not under `root` itself) with at least one
    label file above `min_file_size`, since a subject with only an image
    and no real label is not usable by `create_datalist()`.

    Args:
        root: Dataset root directory to check.
        expected_subject_count: Subject count the adapter expects to produce.
        min_file_size: Byte threshold below which a file is treated as a
            stub/placeholder rather than real data.

    Returns:
        True if `root` already contains at least `expected_subject_count`
        subjects with real label data.
    """
    labels_dir = root / "derivatives" / "labels"
    if not labels_dir.is_dir():
        return False

    count = 0
    for subject_dir in labels_dir.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
            continue
        anat_dir = subject_dir / "anat"
        if not anat_dir.is_dir():
            continue
        if any(f.stat().st_size >= min_file_size for f in anat_dir.glob("*.nii.gz")):
            count += 1

    return count >= expected_subject_count
