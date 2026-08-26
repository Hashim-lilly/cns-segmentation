"""Shared inference/evaluation core for scripts/predict.py, evaluate_external.py, evaluate.py.

Holds the sliding-window inference loop, checkpoint loading, and structure/
class-map resolution that `scripts/predict.py` (held-out spine-generic sites)
and `scripts/evaluate_external.py` (independent validation datasets) each
had duplicated inline. Both scripts keep their own Typer CLI and just call
into `run_predict()`/`run_evaluate_external()` here; `scripts/evaluate.py`
calls the same two functions directly to build one merged report.

Raises plain `ValueError` (not `typer.BadParameter`) on bad input, so this
module has no Typer dependency and is usable from non-CLI callers.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.progress import track
from scipy import ndimage

from cns_segmentation.data.dataset_registry import get_dataset, merge_label_keys
from cns_segmentation.data.label_compositing import DEFAULT_LABEL_PRIORITY
from cns_segmentation.data.spine_generic import create_datalist, flatten_structure_labels
from cns_segmentation.data.transforms import get_val_transforms
from cns_segmentation.evaluation.metrics import aggregate_metrics, evaluate_subject
from cns_segmentation.models.segresnet import create_segresnet, empty_cache, get_device

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class InferenceResult:
    """Artifacts and metrics from a `run_predict()`/`run_evaluate_external()` call."""

    out_dir: Path
    predictions_dir: Path
    labels_dir: Path
    results: list[dict]
    summary: dict
    structures: Optional[list[str]] = None
    overlays_dir: Optional[Path] = None
    extra: dict = field(default_factory=dict)


def load_yaml(path: Path) -> dict:
    """Load a YAML file, raising `ValueError` if missing/invalid."""
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}")


def load_checkpoint_state_dict(checkpoint_path: Path, device) -> dict:
    """Load a checkpoint's state dict, handling both wrapped and raw shapes.

    Args:
        checkpoint_path: Path to the `.pth` checkpoint.
        device: Target device for `torch.load`'s `map_location`.

    Returns:
        The model's state dict, unwrapping `checkpoint["model_state_dict"]`
        if the checkpoint was saved with training metadata alongside it.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    return checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint


def load_model(model_cfg: dict, checkpoint_path: Path, device) -> torch.nn.Module:
    """Build a SegResNet from config and load trained weights onto it."""
    model = create_segresnet(model_cfg)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    model.to(device)
    model.eval()
    return model


def build_inferer(sw_cfg: dict):
    """Build a MONAI `SlidingWindowInferer` from an inference config's `sliding_window` block."""
    from monai.inferers import SlidingWindowInferer  # local import: heavy, only needed here

    return SlidingWindowInferer(
        roi_size=sw_cfg["roi_size"],
        overlap=sw_cfg["overlap"],
        mode=sw_cfg.get("mode", "gaussian"),
        sw_batch_size=1,
    )


def resolve_structures(
    dataset_names: Union[str, list[str]],
) -> tuple[Optional[list[str]], Optional[dict[str, int]]]:
    """Resolve a train config's `data.dataset` into structures + class_map.

    Mirrors `SegmentationTrainer.setup_data()`/the original inline logic in
    `scripts/predict.py`: `data.dataset` may be a single registry key
    (legacy cord-only, `out_channels=2`) or a list of keys (multi-structure,
    joint model), which drives label compositing and the checkpoint's output
    channel ordering via `DEFAULT_LABEL_PRIORITY`.

    Args:
        dataset_names: `train_config["data"]["dataset"]` value.

    Returns:
        `(None, None)` for the legacy cord-only case. Otherwise
        `(structures, class_map)` where `structures` is
        `DEFAULT_LABEL_PRIORITY` filtered to this checkpoint's classes and
        `class_map` is `{structure: 1-indexed channel id}`.
    """
    names = [dataset_names] if isinstance(dataset_names, str) else list(dataset_names)
    if names == ["spine_generic_cord"]:
        return None, None
    specs = [get_dataset(name) for name in names]
    label_keys = merge_label_keys(*specs)
    structures = [s for s in DEFAULT_LABEL_PRIORITY if s in label_keys]
    class_map = {s: i + 1 for i, s in enumerate(structures)}
    return structures, class_map


def checkpoint_class_id(train_config: dict, structure: str) -> int:
    """Recover a checkpoint's output class id for `structure` from its train config.

    Args:
        train_config: Parsed training config the checkpoint came from.
        structure: Structure name to look up, e.g. "canal".

    Returns:
        The 1-indexed class id `structure` occupies in the checkpoint's output.

    Raises:
        ValueError: If `structure` was not one of this checkpoint's trained classes.
    """
    structures, _ = resolve_structures(train_config["data"]["dataset"])
    if structures is None or structure not in structures:
        raise ValueError(
            f"'{structure}' is not a class this checkpoint was trained on. "
            f"Trains on: {structures}"
        )
    return structures.index(structure) + 1


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Zero out every connected component except the largest."""
    binary = mask > 0
    if not binary.any():
        return mask
    labeled, n_components = ndimage.label(binary)
    if n_components <= 1:
        return mask
    sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    out = np.where(labeled == largest_label, mask, 0)
    return out.astype(mask.dtype)


def save_overlay_png(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    dice: float,
    subject_id: str,
    out_path: Path,
) -> None:
    """Save a 4-panel mid-slice overlay (input / GT / prediction / overlay).

    `label`/`pred` are expected to already be binarized (any-nonzero).
    """
    label_sum = label.sum(axis=(0, 1))
    best_slice = int(np.argmax(label_sum)) if label_sum.any() else label.shape[-1] // 2

    img_slice = image[:, :, best_slice].T
    lbl_slice = label[:, :, best_slice].T
    pred_slice = pred[:, :, best_slice].T

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_slice, cmap="gray", origin="lower")
    axes[0].set_title("Input MRI")
    axes[0].axis("off")

    axes[1].imshow(lbl_slice, cmap="Reds", origin="lower")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_slice, cmap="Blues", origin="lower")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(img_slice, cmap="gray", origin="lower")
    gt_mask = np.ma.masked_where(lbl_slice == 0, lbl_slice)
    pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[3].imshow(gt_mask, cmap="Greens", alpha=0.5, origin="lower")
    axes[3].imshow(pred_mask, cmap="Reds", alpha=0.3, origin="lower")
    axes[3].set_title("Green=GT, Red=Pred")
    axes[3].axis("off")

    fig.suptitle(f"{subject_id} (Dice={dice:.4f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def flatten_result(result: dict) -> dict:
    """Flatten a nested per-structure `evaluate_subject()` result for CSV export."""
    if "dice" in result:
        return result
    flat = {"subject": result["subject"], "site": result["site"]}
    for structure, metrics in result.items():
        if structure in ("subject", "site"):
            continue
        for metric_key, value in metrics.items():
            flat[f"{structure}_{metric_key}"] = value
    return flat


def list_heldout_subjects(
    train_config: dict,
    structure: str,
    project_root: Path,
) -> list[dict]:
    """Enumerate held-out ('val' split) subjects with raw image/label paths for one structure.

    Mirrors `run_predict()`'s datalist construction above, but returns raw
    (unresampled) file paths only — no MONAI transforms, no model. Shared by
    `scripts/run_baselines.py` (baseline scoring) and the Streamlit demo
    (per-case metrics), both of which need ground truth without running this
    repo's own model.

    Args:
        train_config: Parsed training config (dataset root/site splits).
        structure: Structure name to filter to, e.g. "canal".
        project_root: Repo root, used to resolve relative paths in configs.

    Returns:
        List of dicts with "subject", "site", "image", "label" raw file paths.
    """
    data_cfg = train_config["data"]
    structures, _ = resolve_structures(data_cfg.get("dataset", "spine_generic_cord"))
    multi_structure = structures is not None

    label_keys = None
    if multi_structure:
        names = [data_cfg["dataset"]] if isinstance(data_cfg["dataset"], str) else data_cfg["dataset"]
        specs = [get_dataset(n) for n in names]
        label_keys = merge_label_keys(*specs)

    datalist = create_datalist(
        root_dir=project_root / Path(data_cfg["root_dir"]),
        sites=data_cfg["val_sites"],
        min_file_size=data_cfg.get("min_file_size", 1000),
        label_keys=label_keys,
        require_all_labels=multi_structure and len(structures) > 1,
    )
    if multi_structure:
        datalist = flatten_structure_labels(datalist)
        label_key = f"label_{structure}"
        datalist = [item for item in datalist if label_key in item]
        for item in datalist:
            item["label"] = item[label_key]
    return datalist


def predict_volume(
    image_path: Path,
    model: torch.nn.Module,
    inferer,
    spacing: list,
    device,
    structures: Optional[list[str]] = None,
    uncertainty: bool = False,
    n_mc_samples: int = 8,
) -> dict:
    """Run single-volume inference for interactive use (e.g. the Streamlit demo).

    Unlike `run_predict()`/`run_evaluate_external()` (batch, dataset-driven,
    ground truth required), this loads exactly one image with no label, and
    optionally layers MC-Dropout uncertainty on top of the deterministic
    argmax prediction — the two are separate forward passes, so headline
    "prediction" and "uncertainty" outputs come from different inference
    modes (same provenance distinction `scripts/evaluate.py` discloses for
    the MC-Dropout-mean-based held-out numbers).

    Args:
        image_path: Path to a raw NIfTI image.
        model: Loaded model (see `load_model()`), already `.eval()`'d.
        inferer: Sliding-window inferer (see `build_inferer()`).
        spacing: Target voxel spacing, matching the checkpoint's training config.
        device: Torch device the model lives on.
        structures: Multi-class structure names in checkpoint output-channel
            order, or None for the legacy binary cord-only case. Carried
            through to the result for labeling only.
        uncertainty: If True, also run MC-Dropout and return entropy/variance/
            mutual_information maps (adds `n_mc_samples` extra forward passes).
        n_mc_samples: MC-Dropout forward passes, only used if uncertainty=True.

    Returns:
        Dict with "image" (normalized, oriented+resampled input volume),
        "pred" (argmax class-id volume, 0=background), "affine" (4x4,
        post-resample), "structures". If uncertainty=True, additionally
        "mean_probs" (MC-Dropout mean softmax, shape [C, ...]), "entropy",
        "variance", "mutual_information" (same spatial shape as "pred").
    """
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd, Orientationd, Spacingd

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True),
        ]
    )
    data = transforms({"image": str(image_path)})
    image_t = data["image"]
    inputs = image_t.unsqueeze(0).to(device)
    affine = image_t.affine.numpy() if hasattr(image_t, "affine") else np.eye(4)

    result: dict = {
        "image": image_t[0].cpu().numpy(),
        "affine": affine,
        "structures": structures,
    }

    with torch.no_grad():
        logits = inferer(inputs, model)
    result["pred"] = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    del logits
    empty_cache(device)

    if uncertainty:
        from cns_segmentation.models.uncertainty import MCDropoutWrapper

        mc = MCDropoutWrapper(model, n_samples=n_mc_samples)
        unc = mc.predict_with_uncertainty(inputs, inferer=inferer)
        result["mean_probs"] = unc["mean_probs"][0].cpu().numpy()
        result["entropy"] = unc["entropy"][0].cpu().numpy()
        result["variance"] = unc["variance"][0].cpu().numpy()
        result["mutual_information"] = unc["mutual_information"][0].cpu().numpy()

    del inputs
    empty_cache(device)
    return result


def run_predict(
    config: dict,
    train_config: dict,
    checkpoint_path: Path,
    project_root: Path,
    split: str = "val",
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    save_overlays: bool = True,
) -> InferenceResult:
    """Run sliding-window inference over a held-out site split and score it.

    Ported from `scripts/predict.py`'s inline body — behavior and output
    artifact schema (predictions/labels/overlays dirs, dice_per_subject.csv,
    metrics_summary.yaml) are unchanged.

    Args:
        config: Parsed inference YAML config.
        train_config: Parsed training config (dataset root/site splits/preprocessing).
        checkpoint_path: Resolved path to the trained checkpoint.
        project_root: Repo root, used to resolve relative paths in configs.
        split: 'val' or 'train'.
        output_dir: Resolved output directory. Defaults to `config["output"]["output_dir"]`.
        limit: Only process the first N subjects.
        save_overlays: Whether to render mid-slice overlay PNGs (skip for speed in reuse contexts).

    Returns:
        InferenceResult with per-subject results, aggregate summary, and artifact paths.

    Raises:
        ValueError: On bad split, missing checkpoint, or an empty datalist.
    """
    if split not in ("val", "train"):
        raise ValueError("split must be 'val' or 'train'")
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    sites = train_config["data"][f"{split}_sites"]
    out_dir = output_dir if output_dir is not None else project_root / Path(config["output"]["output_dir"])
    predictions_dir = out_dir / "predictions"
    labels_dir = out_dir / "labels"
    overlays_dir = out_dir / "overlays"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)

    data_cfg = train_config["data"]
    structures, class_map = resolve_structures(data_cfg.get("dataset", "spine_generic_cord"))
    multi_structure = structures is not None
    require_all_labels = multi_structure and len(structures) > 1
    label_keys = None
    if multi_structure:
        specs = [get_dataset(n) for n in (
            [data_cfg["dataset"]] if isinstance(data_cfg["dataset"], str) else data_cfg["dataset"]
        )]
        label_keys = merge_label_keys(*specs)

    datalist = create_datalist(
        root_dir=project_root / Path(data_cfg["root_dir"]),
        sites=sites,
        min_file_size=data_cfg.get("min_file_size", 1000),
        label_keys=label_keys,
        require_all_labels=require_all_labels,
    )
    if multi_structure:
        datalist = flatten_structure_labels(datalist)

    if not datalist:
        raise ValueError(f"No subjects found for split='{split}' sites={sites}")
    if limit is not None:
        datalist = datalist[:limit]
    logger.info(
        "Running inference on %d subjects (split=%s, structures=%s)", len(datalist), split, structures
    )

    val_transforms = get_val_transforms({"spacing": config["preprocessing"]["spacing"]}, structures=structures)
    model = load_model(config["model"], checkpoint_path, device)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    inferer = build_inferer(config["inference"]["sliding_window"])
    keep_largest = config["inference"].get("keep_largest_component", False)

    results = []
    for item in track(datalist, description="Predicting", console=console):
        subject_id = item["subject"]
        if multi_structure:
            transform_input = {
                "image": item["image"],
                **{f"label_{s}": item[f"label_{s}"] for s in structures},
            }
        else:
            transform_input = {"image": item["image"], "label": item["label"]}
        data = val_transforms(transform_input)

        image_t = data["image"]
        label_np = data["label"][0].numpy().astype(np.uint8)
        inputs = image_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = inferer(inputs, model)
        pred_np = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        del inputs, logits
        empty_cache(device)

        if keep_largest:
            pred_np = _keep_largest_component(pred_np)

        affine = image_t.affine.numpy() if hasattr(image_t, "affine") else np.eye(4)
        pred_path = predictions_dir / f"{subject_id}_pred.nii.gz"
        label_path = labels_dir / f"{subject_id}_label.nii.gz"
        nib.save(nib.Nifti1Image(pred_np, affine), pred_path)
        nib.save(nib.Nifti1Image(label_np, affine), label_path)

        result = evaluate_subject(pred_path, label_path, class_map=class_map)
        result["subject"] = subject_id
        result["site"] = item["site"]
        results.append(result)

        if save_overlays:
            overall_dice = result["overall"]["dice"] if multi_structure else result["dice"]
            image_np = image_t[0].cpu().numpy()
            save_overlay_png(
                image_np,
                (label_np > 0).astype(np.uint8),
                (pred_np > 0).astype(np.uint8),
                overall_dice,
                subject_id,
                overlays_dir / f"{subject_id}_overlay.png",
            )

    results_df = pd.DataFrame([flatten_result(r) for r in results])
    results_df.to_csv(out_dir / "dice_per_subject.csv", index=False)

    summary = aggregate_metrics(results)
    with open(out_dir / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    return InferenceResult(
        out_dir=out_dir,
        predictions_dir=predictions_dir,
        labels_dir=labels_dir,
        overlays_dir=overlays_dir if save_overlays else None,
        results=results,
        summary=summary,
        structures=structures,
    )


def run_evaluate_external(
    dataset: str,
    structure: str,
    train_config: dict,
    inference_config: dict,
    checkpoint_path: Path,
    project_root: Path,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> InferenceResult:
    """Score a trained checkpoint's `structure` channel against an external validation dataset.

    Ported from `scripts/evaluate_external.py`'s inline body — behavior and
    output artifact schema unchanged.

    Args:
        dataset: Registry key of the external validation dataset, e.g. 'spider_canal'.
        structure: Structure name to score, e.g. 'canal'.
        train_config: Parsed training config the checkpoint came from.
        inference_config: Parsed inference config (model architecture, sliding window params).
        checkpoint_path: Resolved path to the trained checkpoint.
        project_root: Repo root, used to resolve relative paths in configs.
        output_dir: Defaults to `experiments/external_eval_<dataset>/`.
        limit: Only process the first N subjects.

    Returns:
        InferenceResult with per-subject results, aggregate summary, and artifact paths.

    Raises:
        ValueError: On missing checkpoint, unknown structure, or an empty datalist.
    """
    class_id = checkpoint_class_id(train_config, structure)

    spec = get_dataset(dataset)
    if structure not in spec.label_keys:
        raise ValueError(f"'{structure}' is not in {dataset}'s label_keys: {list(spec.label_keys)}")

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = output_dir if output_dir is not None else project_root / Path(f"experiments/external_eval_{dataset}")
    predictions_dir = out_dir / "predictions"
    labels_dir = out_dir / "labels"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)

    datalist = create_datalist(
        root_dir=spec.root if spec.root.is_absolute() else project_root / spec.root,
        sites=None,
        min_file_size=1000,
        label_keys=spec.label_keys,
        require_all_labels=False,
    )
    datalist = flatten_structure_labels(datalist)
    datalist = [item for item in datalist if f"label_{structure}" in item]
    if not datalist:
        raise ValueError(f"No subjects with a '{structure}' label found in {dataset}")
    if limit is not None:
        datalist = datalist[:limit]
    logger.info(
        "Evaluating %d subjects from %s (structure=%s, checkpoint class id=%d)",
        len(datalist), dataset, structure, class_id,
    )

    val_transforms = get_val_transforms(
        {"spacing": inference_config["preprocessing"]["spacing"]}, structures=[structure]
    )
    model = load_model(inference_config["model"], checkpoint_path, device)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    inferer = build_inferer(inference_config["inference"]["sliding_window"])

    results = []
    for item in track(datalist, description="Evaluating", console=console):
        subject_id = item["subject"]
        data = val_transforms({"image": item["image"], f"label_{structure}": item[f"label_{structure}"]})

        image_t = data["image"]
        label_np = (data["label"][0].numpy() > 0).astype(np.uint8)
        inputs = image_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = inferer(inputs, model)
        pred_np = (logits.argmax(dim=1)[0].cpu().numpy() == class_id).astype(np.uint8)
        del inputs, logits
        empty_cache(device)

        affine = image_t.affine.numpy() if hasattr(image_t, "affine") else np.eye(4)
        pred_path = predictions_dir / f"{subject_id}_pred.nii.gz"
        label_path = labels_dir / f"{subject_id}_label.nii.gz"
        nib.save(nib.Nifti1Image(pred_np, affine), pred_path)
        nib.save(nib.Nifti1Image(label_np, affine), label_path)

        result = evaluate_subject(pred_path, label_path)
        result["subject"] = subject_id
        result["site"] = item["site"]
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "dice_per_subject.csv", index=False)

    summary = aggregate_metrics(results)
    summary["dataset"] = dataset
    summary["structure"] = structure
    summary["checkpoint"] = str(checkpoint_path)
    with open(out_dir / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    return InferenceResult(
        out_dir=out_dir,
        predictions_dir=predictions_dir,
        labels_dir=labels_dir,
        results=results,
        summary=summary,
        structures=[structure],
        extra={"dataset": dataset, "structure": structure, "class_id": class_id},
    )
