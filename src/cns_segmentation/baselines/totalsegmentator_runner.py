"""Subprocess wrapper for TotalSegmentator-MRI as a cord-only comparison baseline.

TotalSegmentator's `total_mr` task has a `spinal_cord` class but no
canal/thecal-sac/rootlet class (confirmed via its README), so this is only
usable as a third cord baseline alongside SCT's cord model — not a
canal/csf/rootlets baseline.

TotalSegmentator is NOT part of this repo's main environment (its own torch/
nnU-Net pin would conflict with this repo's MONAI/torch versions) — it must
be installed into an isolated venv. This module shells out to that venv's
`TotalSegmentator` CLI via subprocess, the same way `sct_runner.py` shells
out to `sct_deepseg`; it never imports totalsegmentator as a library.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from cns_segmentation.evaluation.metrics import evaluate_subject

logger = logging.getLogger(__name__)

# TotalSegmentator-MRI's total_mr task output filename for the spinal cord class.
_SPINAL_CORD_FILENAME = "spinal_cord.nii.gz"


def check_totalsegmentator_available(venv_python: Path) -> bool:
    """Check whether TotalSegmentator's CLI is importable inside `venv_python`'s venv."""
    if not venv_python.exists():
        logger.warning("TotalSegmentator venv python not found: %s", venv_python)
        return False
    try:
        result = subprocess.run(
            [str(venv_python), "-c", "import totalsegmentator"],
            capture_output=True, text=True, timeout=60,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_totalsegmentator_cord(
    venv_python: Path, input_path: Path, output_dir: Path, timeout: int = 1800
) -> Optional[Path]:
    """Run TotalSegmentator's `total_mr` task and extract the spinal_cord mask.

    Args:
        venv_python: Path to the isolated venv's `python` binary that has
            totalsegmentator installed (e.g. `/tmp/ts_venv/bin/python`).
        input_path: Path to input T2w NIfTI.
        output_dir: Directory TotalSegmentator writes its per-class masks into
            (one file per anatomical class, multilabel=False layout).
        timeout: Max seconds to allow the subprocess to run (model download +
            inference can be slow on CPU).

    Returns:
        Path to the extracted binary spinal_cord mask, or None if the run
        failed or produced no spinal_cord output — the exact failure is
        logged, never fabricated as a result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # `totalsegmentator` is a package with no `__main__.py`, so `-m
    # totalsegmentator` fails with "cannot be directly executed" — the real
    # entry point is the `TotalSegmentator` console script installed
    # alongside `venv_python` in the same venv's bin/ directory.
    totalsegmentator_bin = venv_python.parent / "TotalSegmentator"
    # `--fast` (3mm) is TotalSegmentator's own documented lower-resolution
    # mode. Full-resolution total_mr on CPU exceeded a 1800s (30min) timeout
    # without finishing on a real held-out subject (see baselines_report's
    # provenance notes) — `--fast` is used here to make CPU-only baseline
    # scoring tractable at all, at the documented cost of resolution/accuracy
    # relative to TotalSegmentator's default full-res mode. This is disclosed
    # as a caveat wherever this baseline's numbers are reported, not silently
    # assumed equivalent to full-res.
    cmd = [
        str(totalsegmentator_bin),
        "-i", str(input_path), "-o", str(output_dir),
        "--task", "total_mr",
        "--fast", "-d", "cpu",
    ]
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("TotalSegmentator timed out after %ds on %s", timeout, input_path)
        return None
    except FileNotFoundError:
        logger.error("TotalSegmentator venv python not found: %s", venv_python)
        return None

    if result.returncode != 0:
        logger.error(
            "TotalSegmentator failed (exit %d) on %s\nstdout: %s\nstderr: %s",
            result.returncode, input_path, result.stdout[-1000:], result.stderr[-1000:],
        )
        return None

    cord_path = output_dir / _SPINAL_CORD_FILENAME
    if not cord_path.exists():
        logger.error(
            "TotalSegmentator ran but produced no %s in %s (contents: %s)",
            _SPINAL_CORD_FILENAME, output_dir, [p.name for p in output_dir.glob("*")],
        )
        return None
    return cord_path


def score_totalsegmentator_cord(
    venv_python: Path,
    subject_id: str,
    site: str,
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    timeout: int = 1800,
) -> dict:
    """Run TotalSegmentator's cord class on one subject and score it against ground truth.

    Ground truth's grid was verified (Step 0 pattern) to need no resampling
    against SCT's native output; TotalSegmentator's `total_mr` task also runs
    at native resolution, so the same file-path-direct scoring applies. If
    the shapes don't match for some subject, `evaluate_subject()` raises
    `ValueError` and that is surfaced as an "error" result rather than
    silently skipped.

    Returns:
        A result dict matching `evaluate_subject()`'s flat shape, or
        `{"subject", "site", "error"}` on any failure.
    """
    subject_output_dir = output_dir / subject_id
    cord_path = run_totalsegmentator_cord(venv_python, image_path, subject_output_dir, timeout=timeout)
    if cord_path is None:
        return {"subject": subject_id, "site": site, "error": "totalsegmentator_failed"}

    try:
        result = evaluate_subject(cord_path, label_path)
    except ValueError as exc:
        logger.error("Shape mismatch scoring TotalSegmentator output for %s: %s", subject_id, exc)
        return {"subject": subject_id, "site": site, "error": f"shape_mismatch: {exc}"}

    result["subject"] = subject_id
    result["site"] = site
    return result
