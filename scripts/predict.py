#!/usr/bin/env python
"""CLI entry point for batch inference and CPU-viewable artifact export.

Runs sliding-window inference with a trained checkpoint over a site split
(validation subjects by default) and writes prediction NIfTIs, mid-slice
overlay PNGs, and per-subject/aggregate Dice metrics to disk. Meant to run
once on a GPU node (see scripts/predict.slurm) so the results can be loaded
and visualized afterwards from a CPU-only notebook kernel.

The actual inference loop lives in `cns_segmentation.inference.run_predict()`
(shared with `scripts/evaluate.py`); this script is the Typer CLI wrapper.

Usage:
    python scripts/predict.py --config configs/inference.yaml
    python scripts/predict.py --checkpoint experiments/spine_segresnet_phase1_20260811_120000/checkpoints/best_model.pth
    python scripts/predict.py --split train --limit 5
    python scripts/predict.py --config configs/inference_canal.yaml --train-config configs/train_spine_canal.yaml
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from cns_segmentation.inference import load_yaml, run_predict

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


def _resolve(path: Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    return path if path.is_absolute() else _PROJECT_ROOT / path


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

    try:
        cfg = load_yaml(_resolve(config))
        train_cfg = load_yaml(_resolve(train_config))
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    checkpoint_path = _resolve(checkpoint) if checkpoint is not None else _resolve(Path(cfg["model"]["checkpoint"]))

    try:
        result = run_predict(
            config=cfg,
            train_config=train_cfg,
            checkpoint_path=checkpoint_path,
            project_root=_PROJECT_ROOT,
            split=split,
            output_dir=_resolve(output_dir) if output_dir is not None else None,
            limit=limit,
        )
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    _print_summary(result.summary)
    console.print(f"\n[bold green]Artifacts written to:[/bold green] {result.out_dir}")
    console.print(f"  Predictions: {result.predictions_dir}")
    console.print(f"  Labels:      {result.labels_dir}")
    console.print(f"  Overlays:    {result.overlays_dir}")
    console.print(f"  Per-subject: {result.out_dir / 'dice_per_subject.csv'}")
    console.print(f"  Summary:     {result.out_dir / 'metrics_summary.yaml'}")


if __name__ == "__main__":
    app()
