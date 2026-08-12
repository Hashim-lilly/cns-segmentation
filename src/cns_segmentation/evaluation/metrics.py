"""Segmentation evaluation metrics: Dice, HD95, ECE."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute binary Dice coefficient.

    Args:
        pred: Binary prediction array.
        target: Binary ground-truth array.

    Returns:
        Dice score in [0, 1].
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def compute_hd95(pred: np.ndarray, target: np.ndarray, spacing_mm: tuple[float, ...] = (1.0, 0.5, 0.5)) -> float:
    """Compute 95th-percentile Hausdorff distance.

    Args:
        pred: Binary prediction array.
        target: Binary ground-truth array.
        spacing_mm: Physical voxel spacing in mm.

    Returns:
        HD95 in mm, or inf if either mask is empty.
    """
    try:
        from monai.metrics.hausdorff_distance import compute_hausdorff_distance
        import torch
    except ImportError:
        logger.warning("MONAI not available; returning inf for HD95")
        return float("inf")

    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
    result = compute_hausdorff_distance(pred_t, target_t, percentile=95)
    return result.item()


def compute_ece(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        confidence: Predicted probability for the positive class, shape [N].
        correct: Binary array indicating whether prediction was correct, shape [N].
        n_bins: Number of confidence bins.

    Returns:
        ECE scalar in [0, 1]. Target: < 0.05.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
