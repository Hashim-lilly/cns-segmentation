#!/usr/bin/env python
"""CLI entry point for MC-Dropout uncertainty estimation and calibration.

Runs N stochastic sliding-window inference passes per subject (dropout kept
active via MCDropoutWrapper), exports entropy/mutual-information/variance
NIfTI maps alongside the usual prediction/label/Dice artifacts from
scripts/predict.py, and accumulates Expected Calibration Error (overall
top-1 plus per-structure one-vs-rest for multi-class checkpoints). This is
the code path that finally reads the `inference.yaml: uncertainty.*` block
(`enabled`, `n_samples`, `metrics`) that predict.py has always ignored.

Usage:
    python scripts/uncertainty.py --config configs/inference_rootlets.yaml \\
        --train-config configs/train_spine_rootlets.yaml --enabled --limit 1

    python scripts/uncertainty.py --config configs/inference_canal.yaml \\
        --train-config configs/train_spine_canal.yaml \\
        --checkpoint experiments/spine_segresnet_canal_20260819_053543/checkpoints/best_model.pth \\
        --enabled
"""

import logging
from pathlib import Path
from typing import Optional

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
from cns_segmentation.evaluation.calibration import ECEAccumulator, plot_reliability_diagram
from cns_segmentation.evaluation.metrics import aggregate_metrics, evaluate_subject
from cns_segmentation.models.segresnet import create_segresnet, empty_cache, get_device
from cns_segmentation.models.uncertainty import MCDropoutWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="uncertainty",
    help="Run MC-Dropout uncertainty estimation and calibration analysis.",
    add_completion=False,
)

# Config metric name -> (MCDropoutWrapper result key, NIfTI subdir/filename suffix).
_METRIC_MAP = {
    "predictive_entropy": ("entropy", "entropy"),
    "mutual_information": ("mutual_information", "mutual_info"),
    "variance": ("variance", "variance"),
}


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, raising a typer-friendly error if missing/invalid."""
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


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    """Mean of `arr` over voxels where `mask` is True, or NaN if `mask` is empty."""
    if not mask.any():
        return float("nan")
    return float(arr[mask].mean())


def _print_calibration_table(calibration: dict) -> None:
    """Print a rich table of ECE per key (overall + per-structure)."""
    table = Table(title="Expected Calibration Error (15 bins)", show_header=True)
    table.add_column("Key", style="bold cyan")
    table.add_column("ECE")
    table.add_column("N voxels")

    table.add_row("overall", f"{calibration['overall']['ece']:.4f}", str(calibration["overall"]["n_total"]))
    for structure, bin_result in calibration.get("per_structure", {}).items():
        table.add_row(structure, f"{bin_result['ece']:.4f}", str(bin_result["n_total"]))

    console.print()
    console.print(table)


@app.command()
def uncertainty(
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
    enabled: Optional[bool] = typer.Option(
        None,
        "--enabled/--no-enabled",
        help="Override inference.uncertainty.enabled from the config. "
        "Required (config or this flag) since the config default is false.",
    ),
) -> None:
    """Run MC-Dropout inference and export uncertainty/calibration artifacts.

    For each subject: runs N stochastic sliding-window passes (dropout kept
    active), derives the mean prediction plus entropy/mutual-information/
    variance maps, computes Dice/HD95/volume-error/surface-Dice against the
    ground truth (per-structure plus "overall" for multi-class models),
    saves prediction/label NIfTIs and the configured uncertainty maps, and
    folds per-voxel confidence/correctness into streaming ECE accumulators
    (overall top-1 plus per-structure one-vs-rest). Writes a per-subject
    CSV, an aggregate Dice metrics YAML, a calibration summary YAML, and one
    reliability-diagram PNG per calibration key at the end.
    """
    console.print(Panel("[bold]CNS MC-Dropout Uncertainty & Calibration[/bold]", border_style="blue", expand=False))

    cfg = _load_yaml(_resolve(config))
    train_cfg = _load_yaml(_resolve(train_config))

    uncertainty_cfg = cfg["inference"].get("uncertainty", {})
    is_enabled = enabled if enabled is not None else uncertainty_cfg.get("enabled", False)
    if not is_enabled:
        console.print(
            "[bold red]Uncertainty analysis is disabled.[/bold red] "
            "Set inference.uncertainty.enabled: true in the config, or pass --enabled."
        )
        raise typer.Exit(code=1)

    n_samples = uncertainty_cfg.get("n_samples", 8)
    configured_metrics = uncertainty_cfg.get("metrics", list(_METRIC_MAP))
    metrics_to_export = [m for m in configured_metrics if m in _METRIC_MAP]
    for unknown in set(configured_metrics) - set(metrics_to_export):
        logger.warning("Ignoring unknown uncertainty metric in config: %s", unknown)

    if split not in ("val", "train"):
        raise typer.BadParameter("--split must be 'val' or 'train'")
    sites = train_cfg["data"][f"{split}_sites"]

    checkpoint_path = _resolve(checkpoint) if checkpoint is not None else _resolve(Path(cfg["model"]["checkpoint"]))
    if not checkpoint_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    out_dir = _resolve(output_dir) if output_dir is not None else _resolve(Path(cfg["output"]["output_dir"]))
    predictions_dir = out_dir / "predictions"
    labels_dir = out_dir / "labels"
    reliability_dir = out_dir / "reliability_diagrams"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    reliability_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dirs = {}
    for metric_name in metrics_to_export:
        _, suffix = _METRIC_MAP[metric_name]
        metric_dir = out_dir / "uncertainty" / suffix
        metric_dir.mkdir(parents=True, exist_ok=True)
        uncertainty_dirs[metric_name] = metric_dir

    device = get_device()
    logger.info("Using device: %s", device)

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
        "Running MC-Dropout inference (N=%d) on %d subjects (split=%s, structures=%s, metrics=%s)",
        n_samples,
        len(datalist),
        split,
        structures,
        metrics_to_export,
    )

    val_transforms = get_val_transforms({"spacing": cfg["preprocessing"]["spacing"]}, structures=structures)

    model = create_segresnet(cfg["model"])
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = raw_checkpoint["model_state_dict"] if "model_state_dict" in raw_checkpoint else raw_checkpoint
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
    wrapper = MCDropoutWrapper(model, n_samples=n_samples)

    overall_acc = ECEAccumulator(n_bins=15)
    structure_accs = {s: ECEAccumulator(n_bins=15) for s in structures} if multi_structure else {}

    results = []
    csv_rows = []
    for item in track(datalist, description="MC-Dropout inference", console=console):
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

        unc = wrapper.predict_with_uncertainty(inputs, inferer=inferer)
        mean_probs_np = unc["mean_probs"][0].cpu().numpy()
        raw_pred_np = unc["mean_probs"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        entropy_np = unc["entropy"][0].cpu().numpy()
        mi_np = unc["mutual_information"][0].cpu().numpy()
        variance_np = unc["variance"][0].cpu().numpy()
        del inputs, unc
        empty_cache(device)

        pred_np = raw_pred_np.copy()
        if keep_largest:
            pred_np = _keep_largest_component(pred_np)

        affine = image_t.affine.numpy() if hasattr(image_t, "affine") else np.eye(4)
        pred_path = predictions_dir / f"{subject_id}_pred.nii.gz"
        label_path = labels_dir / f"{subject_id}_label.nii.gz"
        nib.save(nib.Nifti1Image(pred_np, affine), pred_path)
        nib.save(nib.Nifti1Image(label_np, affine), label_path)

        for metric_name, arr in (
            ("predictive_entropy", entropy_np),
            ("mutual_information", mi_np),
            ("variance", variance_np),
        ):
            if metric_name not in uncertainty_dirs:
                continue
            _, suffix = _METRIC_MAP[metric_name]
            out_path = uncertainty_dirs[metric_name] / f"{subject_id}_{suffix}.nii.gz"
            nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), out_path)

        result = evaluate_subject(pred_path, label_path, class_map=class_map)
        result["subject"] = subject_id
        result["site"] = item["site"]
        results.append(result)

        foreground = (raw_pred_np > 0) | (label_np > 0)
        flat_row = _flatten_result(result)
        flat_row["mean_entropy"] = _masked_mean(entropy_np, foreground)
        flat_row["mean_mutual_info"] = _masked_mean(mi_np, foreground)
        flat_row["mean_variance"] = _masked_mean(variance_np, foreground)
        csv_rows.append(flat_row)

        conf_overall = mean_probs_np.max(axis=0)
        correct_overall = raw_pred_np == label_np
        overall_acc.update(conf_overall.ravel(), correct_overall.ravel())

        if multi_structure:
            for s in structures:
                cid = class_map[s]
                conf_s = mean_probs_np[cid]
                correct_s = label_np == cid
                structure_accs[s].update(conf_s.ravel(), correct_s.ravel())

    results_df = pd.DataFrame(csv_rows)
    results_df.to_csv(out_dir / "dice_per_subject.csv", index=False)

    summary = aggregate_metrics(results)
    with open(out_dir / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    calibration = {"overall": overall_acc.compute()}
    if multi_structure:
        calibration["per_structure"] = {s: acc.compute() for s, acc in structure_accs.items()}
    with open(out_dir / "calibration_summary.yaml", "w") as f:
        yaml.dump(calibration, f, default_flow_style=False, sort_keys=False)

    plot_reliability_diagram(
        calibration["overall"], reliability_dir / "overall.png", title="Reliability — overall"
    )
    for structure, bin_result in calibration.get("per_structure", {}).items():
        plot_reliability_diagram(
            bin_result, reliability_dir / f"{structure}.png", title=f"Reliability — {structure}"
        )

    _print_calibration_table(calibration)
    console.print(f"\n[bold green]Artifacts written to:[/bold green] {out_dir}")
    console.print(f"  Predictions:          {predictions_dir}")
    console.print(f"  Labels:               {labels_dir}")
    for metric_name, metric_dir in uncertainty_dirs.items():
        console.print(f"  {metric_name}: {metric_dir}")
    console.print(f"  Per-subject:          {out_dir / 'dice_per_subject.csv'}")
    console.print(f"  Dice summary:         {out_dir / 'metrics_summary.yaml'}")
    console.print(f"  Calibration summary:  {out_dir / 'calibration_summary.yaml'}")
    console.print(f"  Reliability diagrams: {reliability_dir}")


if __name__ == "__main__":
    app()
