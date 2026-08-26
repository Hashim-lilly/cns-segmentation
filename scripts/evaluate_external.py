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

The actual inference loop lives in
`cns_segmentation.inference.run_evaluate_external()` (shared with
`scripts/evaluate.py`); this script is the Typer CLI wrapper.

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

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from cns_segmentation.inference import load_yaml, run_evaluate_external

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


def _resolve(path: Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    return path if path.is_absolute() else _PROJECT_ROOT / path


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

    try:
        cfg = load_yaml(_resolve(inference_config))
        train_cfg = load_yaml(_resolve(train_config))
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    checkpoint_path = _resolve(checkpoint)

    try:
        result = run_evaluate_external(
            dataset=dataset,
            structure=structure,
            train_config=train_cfg,
            inference_config=cfg,
            checkpoint_path=checkpoint_path,
            project_root=_PROJECT_ROOT,
            output_dir=_resolve(output_dir) if output_dir is not None else None,
            limit=limit,
        )
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    _print_summary(result.summary, dataset, structure)
    console.print(f"\n[bold green]Artifacts written to:[/bold green] {result.out_dir}")


if __name__ == "__main__":
    app()
