"""Materializes a canonical rootlets label for spine-generic subjects that
only ship rater-variant (desc-<rater>) annotations.

`create_datalist()` looks for the plain `<subject>_T2w_label-rootlets_dseg.nii.gz`
filename. Three spine-generic subjects — sub-amu02, sub-barcelona01,
sub-brnoUhb03 — never got a plain file, only `desc-rater1`..`desc-rater4` /
`desc-staple` variants (multiple independent raters plus a STAPLE consensus
segmentation), so `create_datalist()` silently treats them as missing
rootlets labels (21/24 spine-generic rootlet subjects reachable). This picks
one canonical rater per subject — `desc-staple` preferred, falling back to
`desc-rater1` — and copies it to the plain filename `spine_generic.py`
expects, without changing `spine_generic.py` itself (same "materialize to
the expected shape" approach as the Phase 2 external-dataset adapters).
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RATER_PREFERENCE = ["staple", "rater1"]
"""Preferred rater order: STAPLE consensus first, then rater1 as fallback."""

DEFERRED_ROOTLET_SUBJECTS = ["sub-amu02", "sub-barcelona01", "sub-brnoUhb03"]
"""spine-generic subjects that ship only rater-variant rootlets labels."""


def resolve_rootlet_rater_labels(
    root_dir: Path,
    subjects: Optional[list[str]] = None,
    force: bool = False,
) -> list[str]:
    """Materialize a canonical rootlets_dseg label for rater-variant-only subjects.

    Idempotent: subjects that already have a canonical file are skipped
    unless `force=True`.

    Args:
        root_dir: spine-generic BIDS dataset root (matches `DatasetSpec.root`).
        subjects: Subject IDs to resolve. Defaults to `DEFERRED_ROOTLET_SUBJECTS`.
        force: If True, overwrite an existing canonical file with the
            preferred rater's content. If False (default), leave it as-is.

    Returns:
        Subject IDs that have a canonical rootlets label after this call
        (whether just written or already present).
    """
    root_dir = Path(root_dir)
    subjects = subjects if subjects is not None else DEFERRED_ROOTLET_SUBJECTS
    resolved: list[str] = []

    for subject_id in subjects:
        anat_dir = root_dir / "derivatives" / "labels" / subject_id / "anat"
        canonical = anat_dir / f"{subject_id}_T2w_label-rootlets_dseg.nii.gz"

        if canonical.is_file() and not force:
            resolved.append(subject_id)
            continue

        source = next(
            (
                anat_dir / f"{subject_id}_T2w_desc-{rater}_label-rootlets_dseg.nii.gz"
                for rater in RATER_PREFERENCE
                if (anat_dir / f"{subject_id}_T2w_desc-{rater}_label-rootlets_dseg.nii.gz").is_file()
            ),
            None,
        )
        if source is None:
            logger.warning(
                "No rater variant %s found for %s under %s — cannot resolve "
                "canonical rootlets label",
                RATER_PREFERENCE,
                subject_id,
                anat_dir,
            )
            continue

        anat_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, canonical)

        source_json = source.with_suffix("").with_suffix(".json")
        if source_json.is_file():
            shutil.copyfile(source_json, canonical.with_suffix("").with_suffix(".json"))

        logger.info("Resolved %s rootlets label from %s", subject_id, source.name)
        resolved.append(subject_id)

    return resolved
