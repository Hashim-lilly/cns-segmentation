"""Segmentation evaluation metrics for spinal cord segmentation.

Provides Dice coefficient, Hausdorff distance (95th percentile),
volume error, and Normalized Surface Dice (NSD) computations.
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute the Dice similarity coefficient for binary masks.

    Args:
        pred: Binary prediction mask.
        target: Binary ground truth mask.

    Returns:
        Dice coefficient in [0, 1]. Returns 1.0 if both masks are empty,
        0.0 if only one mask is empty.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0

    intersection = np.logical_and(pred, target).sum()
    return float(2.0 * intersection / (pred_sum + target_sum))


def _get_surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels from a binary mask using morphological erosion.

    Args:
        mask: Binary 3D mask.

    Returns:
        Binary mask containing only surface voxels.
    """
    struct = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=struct)
    return np.logical_and(mask, np.logical_not(eroded))


def _compute_surface_distances(
    pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric surface distances between prediction and target.

    Args:
        pred: Binary prediction mask.
        target: Binary ground truth mask.
        spacing: Voxel spacing in mm for each dimension.

    Returns:
        Tuple of (distances from pred surface to target,
        distances from target surface to pred) in mm.
    """
    pred_surface = _get_surface_voxels(pred)
    target_surface = _get_surface_voxels(target)

    # Compute distance transform of the complement of each surface
    # dt gives distance from each voxel to nearest surface voxel
    target_dt = ndimage.distance_transform_edt(
        np.logical_not(target_surface), sampling=spacing
    )
    pred_dt = ndimage.distance_transform_edt(
        np.logical_not(pred_surface), sampling=spacing
    )

    # Distances from pred surface to nearest target surface point
    pred_to_target = target_dt[pred_surface]
    # Distances from target surface to nearest pred surface point
    target_to_pred = pred_dt[target_surface]

    return pred_to_target, target_to_pred


def compute_hausdorff95(
    pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> float:
    """Compute the 95th percentile Hausdorff distance in mm.

    Args:
        pred: Binary prediction mask.
        target: Binary ground truth mask.
        spacing: Voxel spacing in mm for each dimension.

    Returns:
        95th percentile Hausdorff distance in mm. Returns 0.0 if both
        masks are empty, inf if only one mask is empty.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        logger.warning("One mask is empty; Hausdorff distance is undefined.")
        return float("inf")

    pred_to_target, target_to_pred = _compute_surface_distances(
        pred, target, spacing
    )

    if pred_to_target.size == 0 or target_to_pred.size == 0:
        logger.warning("Surface extraction yielded no voxels.")
        return float("inf")

    all_distances = np.concatenate([pred_to_target, target_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_volume_error(
    pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> float:
    """Compute absolute volume difference in mm cubed.

    Args:
        pred: Binary prediction mask.
        target: Binary ground truth mask.
        spacing: Voxel spacing in mm for each dimension.

    Returns:
        Absolute volume difference in mm^3.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    voxel_volume = float(np.prod(spacing))
    pred_volume = pred.sum() * voxel_volume
    target_volume = target.sum() * voxel_volume

    return float(abs(pred_volume - target_volume))


def compute_surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, ...],
    tolerance: float = 1.0,
) -> float:
    """Compute Normalized Surface Dice (NSD) at a given tolerance.

    The NSD measures the fraction of surface voxels in both the prediction
    and target that lie within a specified tolerance distance of the
    opposing surface.

    Args:
        pred: Binary prediction mask.
        target: Binary ground truth mask.
        spacing: Voxel spacing in mm for each dimension.
        tolerance: Distance tolerance in mm for surface matching.

    Returns:
        NSD value in [0, 1]. Returns 1.0 if both masks are empty,
        0.0 if only one mask is empty.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0

    pred_to_target, target_to_pred = _compute_surface_distances(
        pred, target, spacing
    )

    if pred_to_target.size == 0 or target_to_pred.size == 0:
        logger.warning("Surface extraction yielded no voxels for NSD.")
        return 0.0

    pred_within_tol = (pred_to_target <= tolerance).sum()
    target_within_tol = (target_to_pred <= tolerance).sum()

    total_surface = pred_to_target.size + target_to_pred.size
    return float((pred_within_tol + target_within_tol) / total_surface)


def evaluate_subject(pred_path: Path, label_path: Path) -> dict:
    """Evaluate all segmentation metrics for a single subject.

    Loads prediction and label NIfTI files, then computes Dice,
    Hausdorff95, volume error, and surface Dice.

    Args:
        pred_path: Path to the predicted segmentation NIfTI file.
        label_path: Path to the ground truth segmentation NIfTI file.

    Returns:
        Dictionary containing:
            - subject: subject identifier derived from filename
            - site: site identifier (prefix before underscore in filename)
            - dice: Dice coefficient
            - hausdorff95_mm: 95th percentile Hausdorff distance in mm
            - volume_error_mm3: absolute volume difference in mm^3
            - surface_dice: Normalized Surface Dice at 1.0 mm tolerance

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If the shapes of pred and label do not match.
    """
    pred_path = Path(pred_path)
    label_path = Path(label_path)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    pred_nii = nib.load(pred_path)
    label_nii = nib.load(label_path)

    pred_data = np.asarray(pred_nii.dataobj).astype(np.uint8)
    label_data = np.asarray(label_nii.dataobj).astype(np.uint8)

    if pred_data.shape != label_data.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_data.shape} vs label {label_data.shape}"
        )

    # Extract voxel spacing from the NIfTI header
    spacing = tuple(float(s) for s in label_nii.header.get_zooms()[:3])

    # Binarize masks
    pred_binary = (pred_data > 0).astype(bool)
    label_binary = (label_data > 0).astype(bool)

    # Derive subject and site identifiers from filename
    subject_id = pred_path.stem.replace(".nii", "")
    # Assume site is the prefix before the first underscore
    parts = subject_id.split("_")
    site_id = parts[0] if len(parts) > 1 else "unknown"

    logger.info("Evaluating subject: %s (site: %s)", subject_id, site_id)

    dice = compute_dice(pred_binary, label_binary)
    hd95 = compute_hausdorff95(pred_binary, label_binary, spacing)
    vol_err = compute_volume_error(pred_binary, label_binary, spacing)
    nsd = compute_surface_dice(pred_binary, label_binary, spacing, tolerance=1.0)

    return {
        "subject": subject_id,
        "site": site_id,
        "dice": dice,
        "hausdorff95_mm": hd95,
        "volume_error_mm3": vol_err,
        "surface_dice": nsd,
    }


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate evaluation metrics across subjects.

    Computes mean and standard deviation for each metric, both overall
    and broken down by site.

    Args:
        results: List of per-subject metric dictionaries as returned
            by evaluate_subject.

    Returns:
        Dictionary containing:
            - overall: dict with mean and std for each metric
            - per_site: dict mapping site_id to mean/std for each metric
            - n_subjects: total number of subjects
    """
    if not results:
        logger.warning("No results to aggregate.")
        return {"overall": {}, "per_site": {}, "n_subjects": 0}

    metric_keys = ["dice", "hausdorff95_mm", "volume_error_mm3", "surface_dice"]

    # Overall aggregation
    overall = {}
    for key in metric_keys:
        values = [r[key] for r in results if np.isfinite(r[key])]
        if values:
            overall[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "n": len(values),
            }
        else:
            overall[key] = {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "n": 0}

    # Per-site aggregation
    sites: dict[str, list[dict]] = {}
    for r in results:
        site = r.get("site", "unknown")
        if site not in sites:
            sites[site] = []
        sites[site].append(r)

    per_site = {}
    for site, site_results in sites.items():
        per_site[site] = {}
        for key in metric_keys:
            values = [r[key] for r in site_results if np.isfinite(r[key])]
            if values:
                per_site[site][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "n": len(values),
                }
            else:
                per_site[site][key] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "n": 0,
                }

    n_subjects = len(results)
    logger.info(
        "Aggregated metrics for %d subjects across %d sites.",
        n_subjects,
        len(sites),
    )

    return {
        "overall": overall,
        "per_site": per_site,
        "n_subjects": n_subjects,
    }
