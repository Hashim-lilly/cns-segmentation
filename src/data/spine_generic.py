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
) -> list[dict]:
    """Create a list of image/label pairs from the Spine-Generic dataset.

    Scans the BIDS root directory for T2w images and their corresponding
    spinal cord segmentation labels. Filters by site if specified and
    skips git-annex pointer stubs based on file size.

    Args:
        root_dir: Path to the BIDS dataset root directory.
        sites: Optional list of site names to include. If None, all
            discovered sites are included.
        min_file_size: Minimum file size in bytes to consider a file
            as real data (not a git-annex pointer stub). Defaults to 1000.

    Returns:
        List of dicts with keys "image", "label", "subject", "site".
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

        # Locate spinal cord segmentation label in derivatives
        label_dir = root_dir / "derivatives" / "labels" / subject_id / "anat"
        label_path = label_dir / f"{subject_id}_T2w_label-SC_seg.nii.gz"

        if not label_path.is_file():
            logger.debug(
                "Missing label for %s: %s", subject_id, label_path
            )
            skipped_missing_label += 1
            continue

        # Git-annex check: skip pointer stubs (very small files)
        image_size = image_path.stat().st_size
        label_size = label_path.stat().st_size

        if image_size < min_file_size:
            logger.debug(
                "Skipping %s: image is git-annex stub (%d bytes)",
                subject_id,
                image_size,
            )
            skipped_annex += 1
            continue

        if label_size < min_file_size:
            logger.debug(
                "Skipping %s: label is git-annex stub (%d bytes)",
                subject_id,
                label_size,
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
