#!/usr/bin/env python
"""CLI entry point for batch inference and CPU-viewable artifact export.

Runs sliding-window inference with a trained checkpoint over a site split
(validation subjects by default) and writes prediction NIfTIs, mid-slice
overlay PNGs, and per-subject/aggregate Dice metrics to disk. Meant to run
once on a GPU node (see scripts/predict.slurm) so the results can be loaded
and visualized afterwards from a CPU-only notebook kernel.

Usage:
    python scripts/predict.py --config configs/inference.yaml
    python scripts/predict.py --checkpoint experiments/spine_segresnet_phase1_20260811_120000/checkpoints/best_model.pth
    python scripts/predict.py --split train --limit 5
    python scripts/predict.py --config configs/inference_canal.yaml --train-config configs/train_spine_canal.yaml
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from scipy import ndimage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from cns_segmentation.data.dataset_registry import get_dataset, merge_label_keys
from cns_segmentation.data.label_compositing import DEFAULT_LABEL_PRIORITY
from cns_segmentation.data.spine_generic import create_datalist, flatten_structure_labels
from cns_segmentation.data.transforms import get_val_transforms
from cns_segmentation.evaluation.metrics import aggregate_metrics, evaluate_subject
from cns_segmentation.models.segresnet import create_segresnet, empty_cache, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="predict",
    help="Run batch inference and export CPU-viewable artifacts.",
    add_completion=False,
)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, raising a typer-friendly error if missing/invalid.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.
    """
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {path}")
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"Invalid YAML in {path}: {exc}")


def _resolve(path: Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Zero out every connected component except the largest.

    Args:
        mask: Binary (or label) 3D array.

    Returns:
        Array of the same shape and dtype with only the largest foreground
        connected component retained. Returned unchanged if the mask is
        empty.
    """
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


def _save_overlay_png(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    dice: float,
    subject_id: str,
    out_path: Path,
) -> None:
    """Save a 4-panel mid-slice overlay (input / GT / prediction / overlay).

    Picks the axial slice with the most ground-truth label voxels, matching
    the convention used in notebooks/04_model_training.ipynb. `label`/`pred`
    are expected to already be binarized (any-nonzero) — multi-class color
    overlays are deferred to a later phase.

    Args:
        image: 3D input MRI volume.
        label: 3D ground truth binary mask.
        pred: 3D predicted binary mask.
        dice: Dice score for this subject, shown in the title.
        subject_id: Subject identifier, shown in the title.
        out_path: Destination PNG path.
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


def _flatten_result(result: dict) -> dict:
    """Flatten a nested per-structure evaluate_subject() result for CSV export.

    Args:
        result: Output of `evaluate_subject()` — flat (no class_map) or
            nested (class_map provided: structure name -> metrics dict,
            plus "overall").

    Returns:
        Flat dict suitable for a DataFrame row. Nested structure metrics
        are renamed "<structure>_<metric>". Byte-identical passthrough for
        the flat (cord-only) shape.
    """
    if "dice" in result:
        return result
    flat = {"subject": result["subject"], "site": result["site"]}
    for structure, metrics in result.items():
        if structure in ("subject", "site"):
            continue
        for metric_key, value in metrics.items():
            flat[f"{structure}_{metric_key}"] = value
    return flat


def _print_metrics_table(stats_by_metric: dict, title: str) -> None:
    """Print one rich table of mean/std/n for the four standard metrics."""
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Mean")
    table.add_column("Std")
    table.add_column("N")

    for key in ["dice", "hausdorff95_mm", "volume_error_mm3", "surface_dice"]:
        stats = stats_by_metric.get(key, {})
        table.add_row(
            key,
            f"{stats.get('mean', float('nan')):.4f}",
            f"{stats.get('std', float('nan')):.4f}",
            str(stats.get("n", 0)),
        )

    console.print()
    console.print(table)


def _print_summary(summary: dict, target_dice: float = 0.93) -> None:
    """Print a rich summary table of aggregate Dice/HD95/NSD metrics.

    Args:
        summary: Output of aggregate_metrics(). May be the flat shape or
            the nested per-structure shape — detected automatically.
        target_dice: Phase 1 target Dice score for comparison.
    """
    overall = summary.get("overall", {})
    nested = "dice" not in overall

    if nested:
        for structure, stats_by_metric in overall.items():
            _print_metrics_table(stats_by_metric, title=f"Overall Metrics — {structure}")
        dice_source = overall.get("overall", {})
    else:
        _print_metrics_table(overall, title="Overall Metrics")
        dice_source = overall

    dice_mean = dice_source.get("dice", {}).get("mean", float("nan"))
    status = "[green]MEETS[/green]" if dice_mean >= target_dice else "[red]BELOW[/red]"
    console.print(
        Panel(
            f"Mean Dice {dice_mean:.4f} vs. Phase 1 target {target_dice:.2f} — {status} target",
            border_style="blue",
            expand=False,
        )
    )

    site_table = Table(title="Per-Site Dice", show_header=True)
    site_table.add_column("Site", style="bold cyan")
    site_table.add_column("Mean Dice")
    site_table.add_column("N")
    for site, stats in summary.get("per_site", {}).items():
        site_dice_stats = (stats.get("overall", {}) if nested else stats).get("dice", {})
        site_table.add_row(site, f"{site_dice_stats.get('mean', float('nan')):.4f}", str(site_dice_stats.get("n", 0)))
    console.print(site_table)


@app.command()
def predict(
    config: Path = typer.Option(
        Path("configs/inference.yaml"),
        "--config",
        "-c",
        help="Path to YAML inference configuration file.",
    ),
    train_config: Path = typer.Option(
        Path("configs/train_spine.yaml"),
        "--train-config",
        help="Training config, used for dataset root/site splits/preprocessing.",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        "-m",
        help="Override model.checkpoint from the inference config.",
    ),
    split: str = typer.Option(
        "val",
        "--split",
        help="Which site split to run inference on: 'val' or 'train'.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Override output.output_dir from the inference config.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Only process the first N subjects (useful for a quick smoke test).",
        min=1,
    ),
) -> None:
    """Run sliding-window inference and export CPU-viewable artifacts.

    For each subject: runs sliding-window inference, computes Dice/HD95/
    volume-error/surface-Dice against the ground truth (per-structure plus
    "overall" for multi-class models, matching train_config's data.dataset),
    saves a prediction NIfTI, saves the composited label NIfTI, and saves a
    4-panel mid-slice overlay PNG. Writes a per-subject CSV and an aggregate
    metrics YAML at the end.
    """
    console.print(Panel("[bold]CNS Batch Inference[/bold]", border_style="blue", expand=False))

    cfg = _load_yaml(_resolve(config))
    train_cfg = _load_yaml(_resolve(train_config))

    if split not in ("val", "train"):
        raise typer.BadParameter("--split must be 'val' or 'train'")
    sites = train_cfg["data"][f"{split}_sites"]

    checkpoint_path = _resolve(checkpoint) if checkpoint is not None else _resolve(Path(cfg["model"]["checkpoint"]))
    if not checkpoint_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    out_dir = _resolve(output_dir) if output_dir is not None else _resolve(Path(cfg["output"]["output_dir"]))
    predictions_dir = out_dir / "predictions"
    labels_dir = out_dir / "labels"
    overlays_dir = out_dir / "overlays"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)

    # Resolve dataset registry entry/entries the same way SegmentationTrainer
    # does: data.dataset may be a single registry key (legacy cord-only) or
    # a list of keys (multi-structure), which drives label compositing.
    data_cfg = train_cfg["data"]
    dataset_name = data_cfg.get("dataset", "spine_generic_cord")
    dataset_names = [dataset_name] if isinstance(dataset_name, str) else list(dataset_name)
    multi_structure = dataset_names != ["spine_generic_cord"]

    label_keys = None
    require_all_labels = False
    structures: Optional[list[str]] = None
    class_map: Optional[dict[str, int]] = None
    if multi_structure:
        specs = [get_dataset(name) for name in dataset_names]
        label_keys = merge_label_keys(*specs)
        structures = [s for s in DEFAULT_LABEL_PRIORITY if s in label_keys]
        require_all_labels = len(structures) > 1
        class_map = {s: i + 1 for i, s in enumerate(structures)}

    datalist = create_datalist(
        root_dir=_resolve(Path(data_cfg["root_dir"])),
        sites=sites,
        min_file_size=data_cfg.get("min_file_size", 1000),
        label_keys=label_keys,
        require_all_labels=require_all_labels,
    )
    if multi_structure:
        datalist = flatten_structure_labels(datalist)

    if not datalist:
        console.print(f"[bold red]No subjects found for split='{split}' sites={sites}[/bold red]")
        raise typer.Exit(code=1)
    if limit is not None:
        datalist = datalist[:limit]
    logger.info(
        "Running inference on %d subjects (split=%s, structures=%s)",
        len(datalist),
        split,
        structures,
    )

    val_transforms = get_val_transforms({"spacing": cfg["preprocessing"]["spacing"]}, structures=structures)

    model = create_segresnet(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    sw_cfg = cfg["inference"]["sliding_window"]
    from monai.inferers import SlidingWindowInferer  # local import: heavy, only needed here

    inferer = SlidingWindowInferer(
        roi_size=sw_cfg["roi_size"],
        overlap=sw_cfg["overlap"],
        mode=sw_cfg.get("mode", "gaussian"),
        sw_batch_size=1,
    )
    keep_largest = cfg["inference"].get("keep_largest_component", False)

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
        overall_dice = result["overall"]["dice"] if multi_structure else result["dice"]

        image_np = image_t[0].cpu().numpy()
        _save_overlay_png(
            image_np,
            (label_np > 0).astype(np.uint8),
            (pred_np > 0).astype(np.uint8),
            overall_dice,
            subject_id,
            overlays_dir / f"{subject_id}_overlay.png",
        )

    results_df = pd.DataFrame([_flatten_result(r) for r in results])
    results_df.to_csv(out_dir / "dice_per_subject.csv", index=False)

    summary = aggregate_metrics(results)
    with open(out_dir / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    _print_summary(summary)
    console.print(f"\n[bold green]Artifacts written to:[/bold green] {out_dir}")
    console.print(f"  Predictions: {predictions_dir}")
    console.print(f"  Labels:      {labels_dir}")
    console.print(f"  Overlays:    {overlays_dir}")
    console.print(f"  Per-subject: {out_dir / 'dice_per_subject.csv'}")
    console.print(f"  Summary:     {out_dir / 'metrics_summary.yaml'}")


if __name__ == "__main__":
    app()
