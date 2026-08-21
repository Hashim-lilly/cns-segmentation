"""BIDS-aware data loader for the Spine-Generic multi-subject dataset.

Discovers T2w images and spinal cord segmentation labels following BIDS
conventions, handles git-annex pointer stubs, and supports site-based
train/val splitting.
"""

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Regex to extract site prefix from subject ID.
# Examples: "sub-amu01" -> "amu", "sub-tehranS01" -> "tehranS"
_SITE_PATTERN = re.compile(r"^sub-([a-zA-Z]+)\d+$")


def get_site_from_subject(subject_id: str) -> str:
    """Extract the acquisition site from a BIDS subject identifier.

    Args:
        subject_id: BIDS subject directory name, e.g. "sub-amu01".

    Returns:
        Site string extracted from the subject ID, e.g. "amu".

    Raises:
        ValueError: If the subject ID does not match expected pattern.
    """
    match = _SITE_PATTERN.match(subject_id)
    if match is None:
        raise ValueError(
            f"Cannot extract site from subject ID '{subject_id}'. "
            "Expected pattern: sub-<site><number>, e.g. sub-amu01"
        )
    return match.group(1)


def create_datalist(
    root_dir: Path | str,
    sites: Optional[list[str]] = None,
    min_file_size: int = 1000,
    label_keys: Optional[dict[str, str]] = None,
    require_all_labels: bool = False,
) -> list[dict]:
    """Create a list of image/label pairs from the Spine-Generic dataset.

    Scans the BIDS root directory for T2w images and their corresponding
    segmentation labels. Filters by site if specified and skips git-annex
    pointer stubs based on file size.

    Args:
        root_dir: Path to the BIDS dataset root directory.
        sites: Optional list of site names to include. If None, all
            discovered sites are included.
        min_file_size: Minimum file size in bytes to consider a file
            as real data (not a git-annex pointer stub). Defaults to 1000.
        label_keys: Optional mapping of structure name to BIDS derivative
            suffix (e.g. {"cord": "SC_seg", "canal": "canal_seg"}). When
            None (default), only the spinal cord label ("SC_seg") is looked
            up and the legacy single-"label" output shape is preserved.
            When provided, each subject's found labels are collected into a
            "labels" dict keyed by structure name.
        require_all_labels: When `label_keys` is provided and this is True,
            a subject is only included if every requested structure's label
            is present. Ignored when `label_keys` is None.

    Returns:
        List of dicts with keys "image", "subject", "site", and either
        "label" (legacy shape, when `label_keys` is None) or "labels"
        (a dict of structure -> Path, when `label_keys` is provided — plus
        "label" too if "cord" is one of the requested structures).
        Paths are returned as Path objects.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        logger.error("Root directory does not exist: %s", root_dir)
        return []

    # Discover all subject directories
    subject_dirs = sorted(
        d for d in root_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )

    datalist: list[dict] = []
    skipped_annex = 0
    skipped_missing_image = 0
    skipped_missing_label = 0
    skipped_site_filter = 0

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        # Extract site from subject ID
        try:
            site = get_site_from_subject(subject_id)
        except ValueError:
            logger.warning("Skipping subject with unparseable ID: %s", subject_id)
            continue

        # Filter by site if sites list is provided
        if sites is not None and site not in sites:
            skipped_site_filter += 1
            continue

        # Locate T2w image
        anat_dir = subject_dir / "anat"
        image_path = anat_dir / f"{subject_id}_T2w.nii.gz"

        if not image_path.is_file():
            logger.debug(
                "Missing T2w image for %s: %s", subject_id, image_path
            )
            skipped_missing_image += 1
            continue

        # Git-annex check: skip pointer stubs (very small files)
        image_size = image_path.stat().st_size
        if image_size < min_file_size:
            logger.debug(
                "Skipping %s: image is git-annex stub (%d bytes)",
                subject_id,
                image_size,
            )
            skipped_annex += 1
            continue

        label_dir = root_dir / "derivatives" / "labels" / subject_id / "anat"

        if label_keys is None:
            # Legacy path: single spinal cord label lookup.
            label_path = label_dir / f"{subject_id}_T2w_label-SC_seg.nii.gz"

            if not label_path.is_file():
                logger.debug(
                    "Missing label for %s: %s", subject_id, label_path
                )
                skipped_missing_label += 1
                continue

            if label_path.stat().st_size < min_file_size:
                logger.debug(
                    "Skipping %s: label is git-annex stub", subject_id
                )
                skipped_annex += 1
                continue

            datalist.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "subject": subject_id,
                    "site": site,
                }
            )
            continue

        # Multi-structure path: look up every requested label.
        found: dict[str, str] = {}
        for structure, suffix in label_keys.items():
            candidate = label_dir / f"{subject_id}_T2w_label-{suffix}.nii.gz"
            if candidate.is_file() and candidate.stat().st_size >= min_file_size:
                found[structure] = str(candidate)

        if not found:
            logger.debug("No requested labels found for %s", subject_id)
            skipped_missing_label += 1
            continue

        if require_all_labels and found.keys() != label_keys.keys():
            logger.debug(
                "Skipping %s: missing labels %s",
                subject_id,
                set(label_keys) - found.keys(),
            )
            skipped_missing_label += 1
            continue

        entry = {
            "image": str(image_path),
            "labels": found,
            "subject": subject_id,
            "site": site,
        }
        if "cord" in found:
            entry["label"] = found["cord"]
        datalist.append(entry)

    # Summary logging
    total_subjects = len(subject_dirs)
    logger.info(
        "Spine-Generic dataset: %d subjects found, %d usable pairs",
        total_subjects,
        len(datalist),
    )
    if skipped_annex > 0:
        logger.info("  Skipped (git-annex stubs): %d", skipped_annex)
    if skipped_missing_image > 0:
        logger.info("  Skipped (missing image): %d", skipped_missing_image)
    if skipped_missing_label > 0:
        logger.info("  Skipped (missing label): %d", skipped_missing_label)
    if skipped_site_filter > 0:
        logger.info("  Skipped (site filter): %d", skipped_site_filter)

    return datalist


def flatten_structure_labels(datalist: list[dict]) -> list[dict]:
    """Flatten each entry's "labels" dict into top-level "label_<structure>" keys.

    `create_datalist()` returns multi-structure entries with a nested
    `"labels"` dict (plus a legacy `"label"` convenience key when "cord" is
    among the found structures). Downstream MONAI dict-transforms need flat
    keys instead — `CompositeLabeld` reads `"label_<structure>"` keys and
    rebuilds `"label"` from scratch, so the stale legacy `"label"` key is
    dropped here rather than left to be silently overwritten later.

    Args:
        datalist: Output of `create_datalist(..., label_keys=<not None>)`.

    Returns:
        New list of dicts with `"labels"`/`"label"` replaced by
        `"label_<structure>"` keys for every structure found on that
        subject.
    """
    flattened = []
    for entry in datalist:
        entry = dict(entry)
        labels = entry.pop("labels", {})
        entry.pop("label", None)
        for structure, path in labels.items():
            entry[f"label_{structure}"] = path
        flattened.append(entry)
    return flattened


class SpineGenericDataset:
    """BIDS-aware dataset for Spine-Generic multi-subject spinal cord data.

    Discovers T2w NIfTI images and their corresponding spinal cord
    segmentation labels, handles git-annex pointer stubs, and supports
    site-based train/validation splitting.

    Args:
        root_dir: Path to the BIDS dataset root.
        train_sites: List of site names for training split. Mutually
            exclusive with val_sites (provide one or neither).
        val_sites: List of site names for validation split. Mutually
            exclusive with train_sites (provide one or neither).
        min_file_size: Minimum file size in bytes to distinguish real
            data from git-annex pointer stubs. Defaults to 1000.

    Raises:
        ValueError: If both train_sites and val_sites are provided.

    Example:
        >>> dataset = SpineGenericDataset(
        ...     root_dir="/data/spine-generic",
        ...     train_sites=["amu", "barcelona", "beijing"],
        ... )
        >>> sample = dataset[0]
        >>> sample["image"]  # Path to T2w NIfTI
        PosixPath('/data/spine-generic/sub-amu01/anat/sub-amu01_T2w.nii.gz')
    """

    def __init__(
        self,
        root_dir: Path | str,
        train_sites: Optional[list[str]] = None,
        val_sites: Optional[list[str]] = None,
        min_file_size: int = 1000,
    ) -> None:
        if train_sites is not None and val_sites is not None:
            raise ValueError(
                "Provide either train_sites or val_sites, not both. "
                "Create separate dataset instances for train and val."
            )

        self.root_dir = Path(root_dir)
        self.min_file_size = min_file_size

        # Determine which sites to include
        sites = train_sites if train_sites is not None else val_sites

        self.datalist = create_datalist(
            root_dir=self.root_dir,
            sites=sites,
            min_file_size=self.min_file_size,
        )

        split_name = "all"
        if train_sites is not None:
            split_name = "train"
        elif val_sites is not None:
            split_name = "val"

        logger.info(
            "SpineGenericDataset [%s]: %d samples from %d sites",
            split_name,
            len(self.datalist),
            len(set(item["site"] for item in self.datalist)),
        )

    def __len__(self) -> int:
        """Return the number of valid image/label pairs."""
        return len(self.datalist)

    def __getitem__(self, index: int) -> dict:
        """Retrieve a sample by index.

        Args:
            index: Integer index into the dataset.

        Returns:
            Dict with keys "image", "label", "subject", "site".
        """
        return self.datalist[index]

    def get_sites(self) -> list[str]:
        """Return sorted list of unique sites in this dataset subset.

        Returns:
            Sorted list of site name strings.
        """
        return sorted(set(item["site"] for item in self.datalist))

    def get_subjects(self) -> list[str]:
        """Return sorted list of subject IDs in this dataset subset.

        Returns:
            Sorted list of subject ID strings.
        """
        return sorted(item["subject"] for item in self.datalist)
