#!/usr/bin/env python
"""CLI entry point for external cross-dataset validation of a trained checkpoint.

Phase 2 registers three new `role="validation"` datasets (`spider_canal`,
`alkafri_mendeley_thecal_sac`, `openneuro_ds004507`) precisely so they are
never mixed into a training run (see `trainer.setup_data()`'s role guard) but
can still be used to check whether a checkpoint trained only on spine-generic
data generalizes to an independent source. This script is that check: it
scans every subject in a target registry entry (ignoring train/val site
splits entirely, since a validation-role dataset is held out in full), runs
the same sliding-window inference as `scripts/predict.py`, and scores one
structure's channel from the checkpoint's multi-class output against that
dataset's ground truth for that same structure.

Usage:
    python scripts/evaluate_external.py --dataset spider_canal --structure canal \\
        --train-config configs/train_spine_canal.yaml \\
        --inference-config configs/inference_canal.yaml \\
        --checkpoint experiments/spine_segresnet_canal_20260819_053543/checkpoints/best_model.pth

    python scripts/evaluate_external.py --dataset openneuro_ds004507 --structure rootlets \\
        --train-config configs/train_spine_rootlets.yaml \\
        --inference-config configs/inference_rootlets.yaml \\
        --checkpoint experiments/spine_segresnet_rootlets_20260819_053526/checkpoints/best_model.pth
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
    name="evaluate-external",
    help="Score a trained checkpoint against a held-out external validation dataset.",
    add_completion=False,
)


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


def _checkpoint_class_id(train_config: dict, structure: str, train_config_path: Path) -> int:
    """Recover a checkpoint's output class id for `structure` from its train config.

    Mirrors exactly how `SegmentationTrainer.setup_data()`/`scripts/predict.py`
    derive `class_map` from `data.dataset`'s merged `label_keys`, filtered
    and ordered by `DEFAULT_LABEL_PRIORITY`: `class_map = {s: i + 1 for i, s
    in enumerate(structures)}`. Recomputing it here (rather than hardcoding
    per-config numbers) keeps this script correct if `DEFAULT_LABEL_PRIORITY`
    or a train config's `dataset:` list ever changes.

    Args:
        train_config: Parsed training config the checkpoint came from.
        structure: Structure name to look up, e.g. "canal".
        train_config_path: Only used for the error message.

    Returns:
        The 1-indexed class id `structure` occupies in the checkpoint's output.

    Raises:
        typer.BadParameter: If `structure` was not one of this checkpoint's
            trained classes.
    """
    dataset_names = train_config["data"]["dataset"]
    dataset_names = [dataset_names] if isinstance(dataset_names, str) else list(dataset_names)
    specs = [get_dataset(name) for name in dataset_names]
    label_keys = merge_label_keys(*specs)
    structures = [s for s in DEFAULT_LABEL_PRIORITY if s in label_keys]
    if structure not in structures:
        raise typer.BadParameter(
            f"'{structure}' is not a class this checkpoint was trained on. "
            f"{train_config_path} trains on: {structures}"
        )
    return structures.index(structure) + 1


def _print_summary(summary: dict, dataset: str, structure: str) -> None:
    """Print a rich summary table of aggregate Dice/HD95/NSD metrics."""
    overall = summary.get("overall", {})
    table = Table(title=f"External validation — {dataset} / {structure}", show_header=True)
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
    console.print(
        Panel(
            f"n_subjects={summary.get('n_subjects', 0)}",
            border_style="blue",
            expand=False,
        )
    )


@app.command()
def evaluate_external(
    dataset: str = typer.Option(
        ..., "--dataset", help="Registry key of the external validation dataset, e.g. 'spider_canal'."
    ),
    structure: str = typer.Option(
        ..., "--structure", help="Structure name to score, e.g. 'canal'. Must be one of the "
        "checkpoint's trained classes AND present in --dataset's label_keys."
    ),
    train_config: Path = typer.Option(
        ..., "--train-config", help="Training config the checkpoint came from — used to recover "
        "which output class id corresponds to --structure."
    ),
    inference_config: Path = typer.Option(
        ..., "--inference-config", "-c", help="Inference config (model architecture, sliding "
        "window params) matching the checkpoint."
    ),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-m", help="Path to the trained checkpoint."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Defaults to experiments/external_eval_<dataset>/."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Only process the first N subjects (smoke test).", min=1
    ),
) -> None:
    """Run sliding-window inference on every subject of an external validation dataset.

    Scores only `--structure`'s channel of the checkpoint's multi-class
    output against that dataset's ground truth for the same structure —
    the external dataset is not expected to ship labels for the
    checkpoint's other trained classes.
    """
    console.print(Panel("[bold]CNS External Validation[/bold]", border_style="blue", expand=False))

    cfg = _load_yaml(_resolve(inference_config))
    train_cfg = _load_yaml(_resolve(train_config))
    class_id = _checkpoint_class_id(train_cfg, structure, train_config)

    spec = get_dataset(dataset)
    if structure not in spec.label_keys:
        raise typer.BadParameter(
            f"'{structure}' is not in {dataset}'s label_keys: {list(spec.label_keys)}"
        )

    checkpoint_path = _resolve(checkpoint)
    if not checkpoint_path.exists():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint_path}")

    out_dir = _resolve(output_dir) if output_dir is not None else _resolve(Path(f"experiments/external_eval_{dataset}"))
    predictions_dir = out_dir / "predictions"
    labels_dir = out_dir / "labels"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)

    datalist = create_datalist(
        root_dir=_resolve(spec.root),
        sites=None,
        min_file_size=1000,
        label_keys=spec.label_keys,
        require_all_labels=False,
    )
    datalist = flatten_structure_labels(datalist)
    datalist = [item for item in datalist if f"label_{structure}" in item]
    if not datalist:
        console.print(f"[bold red]No subjects with a '{structure}' label found in {dataset}[/bold red]")
        raise typer.Exit(code=1)
    if limit is not None:
        datalist = datalist[:limit]
    logger.info(
        "Evaluating %d subjects from %s (structure=%s, checkpoint class id=%d)",
        len(datalist), dataset, structure, class_id,
    )

    val_transforms = get_val_transforms({"spacing": cfg["preprocessing"]["spacing"]}, structures=[structure])

    model = create_segresnet(cfg["model"])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
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

    _print_summary(summary, dataset, structure)
    console.print(f"\n[bold green]Artifacts written to:[/bold green] {out_dir}")


if __name__ == "__main__":
    app()
