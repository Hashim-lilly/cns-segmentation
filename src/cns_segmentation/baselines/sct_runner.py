"""Subprocess wrapper for SCT (Spinal Cord Toolbox) baseline segmentation.

Ports the `_run_sct_command` subprocess pattern from
`cns-cfd-simulation/src/cns_cfd/segmentation_bridge/pipeline.py` (same repo
family, not imported cross-repo — kept local so this repo has no dependency
on the sibling one). Scores baseline predictions through the existing
`evaluate_subject`/`aggregate_metrics` in `evaluation/metrics.py` — no new
metrics code.
"""

import logging
import subprocess
from pathlib import Path

from cns_segmentation.baselines.model_registry import ModelSpec
from cns_segmentation.evaluation.metrics import evaluate_subject

logger = logging.getLogger(__name__)


def run_sct_command(model: ModelSpec, input_path: Path, output_path: Path, timeout: int = 600) -> bool:
    """Run an SCT `sct_deepseg` baseline command.

    Args:
        model: ModelSpec with a `command_template` containing `{input}`/`{output}`.
        input_path: Path to input T2w NIfTI.
        output_path: Path for the output segmentation NIfTI.
        timeout: Max seconds to allow the subprocess to run.

    Returns:
        True if the command exited 0 and produced `output_path`.
    """
    cmd = model.command_template.format(input=str(input_path), output=str(output_path))
    logger.info("Running: %s", cmd)

    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error("Command failed (exit %d): %s\nstderr: %s", result.returncode, cmd, result.stderr[:500])
            return False
        return output_path.exists()
    except subprocess.TimeoutExpired:
        logger.error("Command timed out after %ds: %s", timeout, cmd)
        return False
    except FileNotFoundError:
        logger.error("Command not found — is sct_deepseg installed and on PATH? %s", cmd)
        return False


def score_sct_baseline(
    model_key: str,
    subject_id: str,
    site: str,
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    timeout: int = 600,
) -> dict:
    """Run an SCT baseline on one subject and score it against ground truth.

    `image_path`/`label_path` must be the dataset's raw (unresampled) NIfTI
    files — SCT's native output grid was confirmed (Step 0 spot-check) to
    match the raw ground-truth label grid exactly, so no resampling step is
    needed before calling `evaluate_subject()`.

    Args:
        model_key: Key into `baselines.model_registry.MODELS`.
        subject_id: BIDS subject ID, e.g. "sub-stanford01".
        site: Site name for per-site aggregation, e.g. "stanford".
        image_path: Raw input T2w NIfTI path.
        label_path: Raw ground-truth label NIfTI path (same grid as `image_path`).
        output_dir: Directory to write the baseline's predicted NIfTI into.
        timeout: Max seconds to allow the SCT subprocess to run.

    Returns:
        A result dict matching `evaluate_subject()`'s flat shape (subject,
        site, dice, hausdorff95_mm, volume_error_mm3, surface_dice) — or, on
        failure, `{"subject", "site", "error"}`.
    """
    from cns_segmentation.baselines.model_registry import get_model

    model = get_model(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{subject_id}_{model_key}_pred.nii.gz"

    if not run_sct_command(model, image_path, pred_path, timeout=timeout):
        return {"subject": subject_id, "site": site, "error": "sct_command_failed"}

    result = evaluate_subject(pred_path, label_path)
    result["subject"] = subject_id
    result["site"] = site
    return result
