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
"""

import logging
import sys
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
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.spine_generic import create_datalist  # noqa: E402
from src.data.transforms import get_val_transforms  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_dice,
    compute_hausdorff95,
    compute_surface_dice,
    compute_volume_error,
)
from src.models.segresnet import create_segresnet, empty_cache, get_device  # noqa: E402

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
    the convention used in notebooks/04_model_training.ipynb.

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


def _print_summary(summary: dict, target_dice: float = 0.93) -> None:
    """Print a rich summary table of aggregate Dice/HD95/NSD metrics.

    Args:
        summary: Output of aggregate_metrics().
        target_dice: Phase 1 target Dice score for comparison.
    """
    overall = summary.get("overall", {})

    table = Table(title="Overall Metrics", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Mean")
    table.add_column("Std")
    table.add_column("N")

    for key in ["dice", "hausdorff95_mm", "volume_error_mm3", "surface_dice"]:
        stats = overall.get(key, {})
        table.add_row(
            key,
            f"{stats.get('mean', float('nan')):.4f}",
            f"{stats.get('std', float('nan')):.4f}",
            str(stats.get("n", 0)),
        )

    console.print()
    console.print(table)

    dice_mean = overall.get("dice", {}).get("mean", float("nan"))
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
        dice_stats = stats.get("dice", {})
        site_table.add_row(site, f"{dice_stats.get('mean', float('nan')):.4f}", str(dice_stats.get("n", 0)))
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
    volume-error/surface-Dice against the ground truth, saves a prediction
    NIfTI, and saves a 4-panel mid-slice overlay PNG. Writes a per-subject
    CSV and an aggregate metrics YAML at the end.
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
    overlays_dir = out_dir / "overlays"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)

    datalist = create_datalist(
        root_dir=_resolve(Path(train_cfg["data"]["root_dir"])),
        sites=sites,
        min_file_size=train_cfg["data"].get("min_file_size", 1000),
    )
    if not datalist:
        console.print(f"[bold red]No subjects found for split='{split}' sites={sites}[/bold red]")
        raise typer.Exit(code=1)
    if limit is not None:
        datalist = datalist[:limit]
    logger.info("Running inference on %d subjects (split=%s)", len(datalist), split)

    val_transforms = get_val_transforms({"spacing": cfg["preprocessing"]["spacing"]})

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
    spacing = tuple(float(s) for s in cfg["preprocessing"]["spacing"])

    results = []
    for item in track(datalist, description="Predicting", console=console):
        subject_id = item["subject"]
        data = val_transforms({"image": item["image"], "label": item["label"]})

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

        dice = compute_dice(pred_np, label_np)
        hd95 = compute_hausdorff95(pred_np, label_np, spacing)
        vol_err = compute_volume_error(pred_np, label_np, spacing)
        nsd = compute_surface_dice(pred_np, label_np, spacing)
        results.append(
            {
                "subject": subject_id,
                "site": item["site"],
                "dice": dice,
                "hausdorff95_mm": hd95,
                "volume_error_mm3": vol_err,
                "surface_dice": nsd,
            }
        )

        affine = image_t.affine.numpy() if hasattr(image_t, "affine") else np.eye(4)
        nib.save(nib.Nifti1Image(pred_np, affine), predictions_dir / f"{subject_id}_pred.nii.gz")

        image_np = image_t[0].cpu().numpy()
        _save_overlay_png(
            image_np, label_np, pred_np, dice, subject_id, overlays_dir / f"{subject_id}_overlay.png"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "dice_per_subject.csv", index=False)

    summary = aggregate_metrics(results)
    with open(out_dir / "metrics_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    _print_summary(summary)
    console.print(f"\n[bold green]Artifacts written to:[/bold green] {out_dir}")
    console.print(f"  Predictions: {predictions_dir}")
    console.print(f"  Overlays:    {overlays_dir}")
    console.print(f"  Per-subject: {out_dir / 'dice_per_subject.csv'}")
    console.print(f"  Summary:     {out_dir / 'metrics_summary.yaml'}")


if __name__ == "__main__":
    app()
