"""Adapter for OpenNeuro ds004507 ("Spinal Cord Head Positions", CC0).

Materializes a small supplemental rootlets-validation subset — 7 of the
dataset's 10 subjects ship a real volumetric `label-rootlets_dseg.nii.gz`
derivative (as opposed to the dataset's older single-voxel point-label
files, e.g. `labels-spinalroots-manual.nii.gz`, which this adapter ignores).
Each subject has up to 3 sessions (`ses-headDown`/`ses-headNormal`/
`ses-headUp`); rootlets-label coverage is per-session and incomplete, so one
session per subject is picked by ordered preference — same "prefer
canonical, fall back" pattern as `rootlet_rater_selection.py` uses for
spine-generic's rater-variant subjects.

Source subject IDs (`sub-002`, `sub-003`, ...) are digits-only and would
fail spine_generic's `_SITE_PATTERN` (letters then digits), so they are
remapped to `sub-headpos<NNNN>` via `subject_dirname("headpos", i)`,
i = 1..7 in sorted numeric order of the original ID — see
`SOURCE_SUBJECTS`/`build_subject_mapping()`.

Network access (S3, no credentials needed) is isolated to `_download_bytes`
and `_remote_exists`; every other function here — `select_session`,
`build_subject_mapping`, `materialize_subject` — is pure and takes
already-fetched bytes or plain data, so tests can exercise the BIDS
materialization logic with small synthetic byte strings and no network
access, mirroring `write_bids_subject`'s own test style in
`tests/test_adapters.py`.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import requests

from cns_segmentation.data.adapters.base import is_prepared, subject_dirname, write_bids_subject
from cns_segmentation.data.dataset_registry import DatasetSpec

logger = logging.getLogger(__name__)

# .../data/adapters/openneuro_ds004507.py -> .../cns-segmentation
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TARGET_ROOT = _REPO_ROOT / "data" / "openneuro_ds004507"

_S3_BUCKET = "openneuro.org"
_DATASET_PREFIX = "ds004507"
_HTTPS_BASE = f"https://s3.amazonaws.com/{_S3_BUCKET}/{_DATASET_PREFIX}"

SOURCE_SUBJECTS = ["sub-002", "sub-003", "sub-004", "sub-005", "sub-007", "sub-010", "sub-011"]
"""Original ds004507 subject IDs confirmed (prior investigation) to ship a
real volumetric `label-rootlets_dseg.nii.gz`, in sorted numeric order — this
order is what `build_subject_mapping()` assigns `sub-headpos0001`.. against."""

SESSION_PREFERENCE = ["ses-headNormal", "ses-headUp", "ses-headDown"]
"""Ordered session preference: canonical head position first, then the two
tilted positions as fallbacks, matching whichever session actually shipped
a rootlets label for a given subject."""

CONTRAST = "T2w"
LABEL_SUFFIX = "rootlets_dseg"


def _label_relpath(subject: str, session: str) -> str:
    """Dataset-relative path to a subject/session's plain rootlets label.

    Deliberately targets only the plain `label-rootlets_dseg.nii.gz` name —
    NOT the `desc-<rater>_label-rootlets_dseg.nii.gz` rater-variant files
    that sub-007 and sub-010 additionally ship for some sessions, and NOT
    the older single-voxel `labels-spinalroots-manual.nii.gz` point labels.
    """
    return (
        f"derivatives/labels/{subject}/{session}/anat/"
        f"{subject}_{session}_{CONTRAST}_label-{LABEL_SUFFIX}.nii.gz"
    )


def _image_relpath(subject: str, session: str) -> str:
    """Dataset-relative path to a subject/session's raw T2w image."""
    return f"{subject}/{session}/anat/{subject}_{session}_{CONTRAST}.nii.gz"


def _remote_exists(relpath: str) -> bool:
    """Check whether `relpath` exists in the ds004507 S3 bucket.

    Tries `aws s3api head-object --no-sign-request` first (no credentials
    needed for this public bucket); falls back to a plain HTTPS HEAD request
    if the aws CLI is unavailable.
    """
    key = f"{_DATASET_PREFIX}/{relpath}"
    try:
        result = subprocess.run(
            [
                "aws", "s3api", "head-object",
                "--no-sign-request",
                "--bucket", _S3_BUCKET,
                "--key", key,
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.info("aws CLI not found; falling back to HTTPS HEAD for %s", relpath)
        response = requests.head(f"{_HTTPS_BASE}/{relpath}", timeout=30)
        return response.status_code == 200


def _download_bytes(relpath: str) -> bytes:
    """Fetch `relpath`'s raw bytes from the ds004507 S3 bucket.

    Tries `aws s3 cp --no-sign-request ... -` (streamed to stdout, no
    credentials needed) first; falls back to a plain HTTPS GET via
    `requests` if the aws CLI is unavailable.
    """
    s3_uri = f"s3://{_S3_BUCKET}/{_DATASET_PREFIX}/{relpath}"
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", s3_uri, "-"],
            capture_output=True,
            timeout=300,
            check=True,
        )
        return result.stdout
    except FileNotFoundError:
        logger.info("aws CLI not found; falling back to HTTPS GET for %s", relpath)
        response = requests.get(f"{_HTTPS_BASE}/{relpath}", timeout=300)
        response.raise_for_status()
        return response.content


def select_session(available_sessions: set[str]) -> Optional[str]:
    """Pick one session per `SESSION_PREFERENCE`, given which sessions have a label.

    Pure function — takes the set of sessions already confirmed (by
    whatever means, remote or synthetic) to have a rootlets label, and
    returns the first of `SESSION_PREFERENCE` present in that set.

    Args:
        available_sessions: Session names that have a rootlets label for
            some subject.

    Returns:
        The preferred session name, or None if `available_sessions` matches
        none of `SESSION_PREFERENCE`.
    """
    for session in SESSION_PREFERENCE:
        if session in available_sessions:
            return session
    return None


def build_subject_mapping(source_subjects: list[str] = SOURCE_SUBJECTS) -> dict[str, str]:
    """Map original ds004507 subject IDs to collision-safe BIDS dirnames.

    Original IDs (`sub-002`, ...) are digits-only after the `sub-` prefix
    and would fail spine_generic's letters-then-digits site-tag pattern, so
    each is remapped to `sub-headpos<NNNN>` in the order given (expected to
    already be sorted numerically by the caller).

    Args:
        source_subjects: Original subject IDs, in the order they should
            receive `sub-headpos0001`, `sub-headpos0002`, ... .

    Returns:
        Mapping from original subject ID to new BIDS subject dirname.
    """
    return {
        original: subject_dirname("headpos", i)
        for i, original in enumerate(source_subjects, start=1)
    }


def materialize_subject(
    root: Path,
    new_subject_id: str,
    image_bytes: bytes,
    label_bytes: bytes,
) -> None:
    """Write one subject's already-fetched image + rootlets label to `root`.

    Thin, network-free wrapper around `write_bids_subject` that fixes this
    adapter's single label key (`rootlets_dseg`, matching
    `spine_generic_rootlets`'s label scheme) so tests can call it directly
    with synthetic bytes.

    Args:
        root: Dataset root directory (matches the eventual `DatasetSpec.root`).
        new_subject_id: Remapped BIDS subject dirname, e.g. "sub-headpos0001".
        image_bytes: Raw `.nii.gz` bytes for the subject's T2w image.
        label_bytes: Raw `.nii.gz` bytes for the subject's rootlets label.
    """
    write_bids_subject(
        root,
        new_subject_id,
        image_bytes=image_bytes,
        labels={LABEL_SUFFIX: label_bytes},
        contrast=CONTRAST,
    )


def prepare(force: bool = False) -> DatasetSpec:
    """Download and materialize the ds004507 rootlets-validation subset.

    For each of `SOURCE_SUBJECTS`, checks `SESSION_PREFERENCE` in order for
    the first session that has a plain `label-rootlets_dseg.nii.gz`
    derivative, downloads that session's label + raw T2w image, and writes
    them under the remapped `sub-headpos<NNNN>` subject ID via
    `materialize_subject`.

    Args:
        force: If True, re-download/re-materialize even if `is_prepared()`
            already reports the target root as complete. If False
            (default), skip that work and just return the spec.

    Returns:
        DatasetSpec describing the materialized dataset root.
    """
    expected_count = len(SOURCE_SUBJECTS)
    if not force and is_prepared(_TARGET_ROOT, expected_count):
        logger.info(
            "%s already prepared with >= %d subjects; skipping", _TARGET_ROOT, expected_count
        )
        return _build_spec(expected_count)

    mapping = build_subject_mapping(SOURCE_SUBJECTS)
    materialized = 0

    for original_id, new_id in mapping.items():
        available = {
            session for session in SESSION_PREFERENCE
            if _remote_exists(_label_relpath(original_id, session))
        }
        session = select_session(available)
        if session is None:
            logger.warning(
                "No session in %s has a rootlets label for %s — skipping",
                SESSION_PREFERENCE, original_id,
            )
            continue

        logger.info("Fetching %s (session %s) -> %s", original_id, session, new_id)
        label_bytes = _download_bytes(_label_relpath(original_id, session))
        image_bytes = _download_bytes(_image_relpath(original_id, session))
        materialize_subject(_TARGET_ROOT, new_id, image_bytes, label_bytes)
        materialized += 1

    logger.info(
        "Materialized %d/%d ds004507 subjects at %s", materialized, expected_count, _TARGET_ROOT
    )
    return _build_spec(materialized)


def _build_spec(subject_count: int) -> DatasetSpec:
    """Build this adapter's `DatasetSpec` for the given reachable subject count."""
    return DatasetSpec(
        name="openneuro_ds004507",
        root=_TARGET_ROOT,
        format="bids-derivatives",
        label_keys={"rootlets": LABEL_SUFFIX},
        spinal_region="cervical-thoracic-junction",
        subject_count=subject_count,
        sites=1,
        license="CC0",
        role="validation",
        adapter=prepare,
    )
