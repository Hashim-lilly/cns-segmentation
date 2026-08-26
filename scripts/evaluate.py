#!/usr/bin/env python
"""CLI entry point for a merged evaluation report across all trained structures.

Combines what `scripts/predict.py` (held-out spine-generic sites) and
`scripts/evaluate_external.py` (independent validation datasets) each check
one checkpoint/structure at a time into a single long-format table across
structures x sites x external datasets: `experiments/evaluate_report/merged_metrics.csv`.

Reuse-first by default: each structure's held-out and external metrics are
loaded directly from the existing `metrics_summary.yaml` artifacts already on
disk from real prior GPU runs (`experiments/uncertainty_<structure>/`,
`experiments/external_eval_<dataset>/`) rather than re-running inference.
Pass `--force-rerun` to call `run_predict`/`run_evaluate_external` fresh
instead (needed the first time, or after a checkpoint changes).

The actual inference loop (for force-rerun) lives in
`cns_segmentation.inference`, shared with `scripts/predict.py` and
`scripts/evaluate_external.py`; nothing here duplicates that logic.

IMPORTANT — provenance caveats (see `_PROVENANCE_NOTES` and the printed
provenance panel): the reused `uncertainty_<structure>/` artifacts are
MC-Dropout mean-probability outputs (from Phase 3's uncertainty pipeline),
not a deterministic `predict.py` argmax pass — they are close but not
guaranteed bit-identical to a fresh `predict.py` run. The csf and rootlets
held-out sets are tiny (n=2, n=4 respectively) — their aggregate stats are
not statistically meaningful and must not be read as a robust generalization
estimate.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --force-rerun
    python scripts/evaluate.py --force-rerun --structures cord,canal --limit 3
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from cns_segmentation.inference import load_yaml, run_evaluate_external, run_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="evaluate",
    help="Build one merged evaluation report across structures x sites x external datasets.",
    add_completion=False,
)

METRIC_KEYS = ["dice", "hausdorff95_mm", "volume_error_mm3", "surface_dice"]

# Manifest of every trained structure: checkpoint, configs, and where its
# held-out-site and external-validation metrics live/should be written.
# Checkpoint paths verified non-empty; none of configs/inference*.yaml's
# default model.checkpoint values are usable, so every path is spelled out
# here explicitly rather than relying on config defaults.
STRUCTURES = {
    "cord": {
        "checkpoint": "experiments/spine_segresnet_phase1_20260811_065325/checkpoints/best_model.pth",
        "train_config": "configs/train_spine.yaml",
        "inference_config": "configs/inference.yaml",
        "heldout_reuse_dir": "experiments/uncertainty_cord",
        "external": [],
    },
    "canal": {
        "checkpoint": "experiments/spine_segresnet_canal_20260819_053543/checkpoints/best_model.pth",
        "train_config": "configs/train_spine_canal.yaml",
        "inference_config": "configs/inference_canal.yaml",
        "heldout_reuse_dir": "experiments/uncertainty_canal",
        "external": [{"dataset": "spider_canal", "reuse_dir": "experiments/external_eval_spider_canal"}],
    },
    "csf": {
        "checkpoint": "experiments/spine_segresnet_csf_20260819_052857/checkpoints/best_model.pth",
        "train_config": "configs/train_spine_csf.yaml",
        "inference_config": "configs/inference_csf.yaml",
        "heldout_reuse_dir": "experiments/uncertainty_csf",
        "external": [],
    },
    "rootlets": {
        "checkpoint": "experiments/spine_segresnet_rootlets_20260819_053526/checkpoints/best_model.pth",
        "train_config": "configs/train_spine_rootlets.yaml",
        "inference_config": "configs/inference_rootlets.yaml",
        "heldout_reuse_dir": "experiments/uncertainty_rootlets",
        "external": [{"dataset": "openneuro_ds004507", "reuse_dir": "experiments/external_eval_openneuro_ds004507"}],
    },
}

_PROVENANCE_NOTES = [
    "Held-out numbers reused from experiments/uncertainty_<structure>/ are "
    "MC-Dropout mean-probability outputs (Phase 3), not a deterministic "
    "predict.py argmax pass — close but not guaranteed bit-identical to a "
    "fresh --force-rerun.",
    "csf held-out set is n=2 subjects; rootlets held-out set is n=4 subjects "
    "— both too small for a statistically meaningful generalization estimate.",
    "rootlets shows dice=0.0 on true held-out sites (cardiff/stanford) despite "
    "a 0.6121 training-time validation Dice cited elsewhere — the two numbers "
    "measure different things and must not be conflated (see ROADMAP.md).",
]


def _extract_structure_summary(summary: dict, structure: str) -> tuple[dict, dict]:
    """Pull one structure's overall/per_site stats out of a metrics_summary dict.

    Handles both shapes `aggregate_metrics()` can produce: flat (single-
    structure checkpoint, e.g. cord) where `overall["dice"]` exists directly,
    and nested (joint multi-class checkpoint) where `overall[structure]["dice"]`
    exists. External-validation summaries are always flat.
    """
    overall = summary.get("overall", {})
    per_site = summary.get("per_site", {})
    if "dice" in overall:
        return overall, per_site
    structure_overall = overall.get(structure, {})
    structure_per_site = {site: stats.get(structure, {}) for site, stats in per_site.items()}
    return structure_overall, structure_per_site


def _summary_to_rows(
    structure: str, eval_type: str, dataset: Optional[str], overall_stats: dict, per_site_stats: dict
) -> list[dict]:
    """Flatten one structure's overall + per-site stats into merged-table rows."""
    rows = []
    for metric in METRIC_KEYS:
        stats = overall_stats.get(metric, {})
        rows.append(
            {
                "structure": structure,
                "eval_type": eval_type,
                "dataset": dataset or "",
                "site": "overall",
                "metric": metric,
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "std": stats.get("std"),
                "n": stats.get("n"),
            }
        )
    for site, site_stats in per_site_stats.items():
        for metric in METRIC_KEYS:
            stats = site_stats.get(metric, {})
            rows.append(
                {
                    "structure": structure,
                    "eval_type": eval_type,
                    "dataset": dataset or "",
                    "site": site,
                    "metric": metric,
                    "mean": stats.get("mean"),
                    "median": stats.get("median"),
                    "std": stats.get("std"),
                    "n": stats.get("n"),
                }
            )
    return rows


def _load_or_run_heldout(structure: str, spec: dict, project_root: Path, force_rerun: bool, limit: Optional[int]) -> dict:
    """Load an existing held-out-site metrics_summary.yaml, or run predict fresh."""
    reuse_dir = project_root / spec["heldout_reuse_dir"]
    summary_path = reuse_dir / "metrics_summary.yaml"

    if not force_rerun and summary_path.exists():
        logger.info("[%s] reusing existing held-out metrics: %s", structure, summary_path)
        with open(summary_path, "r") as f:
            return yaml.safe_load(f)

    logger.info("[%s] running held-out inference fresh (force_rerun=%s, exists=%s)", structure, force_rerun, summary_path.exists())
    cfg = load_yaml(project_root / spec["inference_config"])
    train_cfg = load_yaml(project_root / spec["train_config"])
    checkpoint_path = project_root / spec["checkpoint"]
    result = run_predict(
        config=cfg,
        train_config=train_cfg,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        split="val",
        output_dir=reuse_dir,
        limit=limit,
        save_overlays=False,
    )
    return result.summary


def _load_or_run_external(
    structure: str, spec: dict, ext: dict, project_root: Path, force_rerun: bool, limit: Optional[int]
) -> dict:
    """Load an existing external-validation metrics_summary.yaml, or run evaluate_external fresh."""
    reuse_dir = project_root / ext["reuse_dir"]
    summary_path = reuse_dir / "metrics_summary.yaml"

    if not force_rerun and summary_path.exists():
        logger.info("[%s] reusing existing external metrics (%s): %s", structure, ext["dataset"], summary_path)
        with open(summary_path, "r") as f:
            return yaml.safe_load(f)

    logger.info(
        "[%s] running external evaluation fresh against %s (force_rerun=%s, exists=%s)",
        structure, ext["dataset"], force_rerun, summary_path.exists(),
    )
    cfg = load_yaml(project_root / spec["inference_config"])
    train_cfg = load_yaml(project_root / spec["train_config"])
    checkpoint_path = project_root / spec["checkpoint"]
    result = run_evaluate_external(
        dataset=ext["dataset"],
        structure=structure,
        train_config=train_cfg,
        inference_config=cfg,
        checkpoint_path=checkpoint_path,
        project_root=project_root,
        output_dir=reuse_dir,
        limit=limit,
    )
    return result.summary


def build_report(
    project_root: Path,
    structures: Optional[list[str]] = None,
    force_rerun: bool = False,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the merged long-format metrics table across all requested structures.

    Args:
        project_root: Repo root.
        structures: Subset of STRUCTURES keys to evaluate. Defaults to all four.
        force_rerun: Re-run inference instead of reusing on-disk metrics_summary.yaml.
        limit: Only process the first N subjects per run (force-rerun only; ignored on reuse).

    Returns:
        (merged_df, provenance) — provenance maps each structure/eval to
        "reused:<path>" or "force_rerun:<path>" plus the fixed caveats list.
    """
    names = structures if structures is not None else list(STRUCTURES.keys())
    unknown = [n for n in names if n not in STRUCTURES]
    if unknown:
        raise ValueError(f"Unknown structure(s): {unknown}. Known: {list(STRUCTURES.keys())}")

    rows: list[dict] = []
    provenance: dict = {"notes": _PROVENANCE_NOTES, "sources": {}}

    for structure in names:
        spec = STRUCTURES[structure]

        heldout_summary = _load_or_run_heldout(structure, spec, project_root, force_rerun, limit)
        overall_stats, per_site_stats = _extract_structure_summary(heldout_summary, structure)
        rows.extend(_summary_to_rows(structure, "heldout_site", None, overall_stats, per_site_stats))
        provenance["sources"][f"{structure}/heldout_site"] = {
            "path": str(project_root / spec["heldout_reuse_dir"] / "metrics_summary.yaml"),
            "mode": "force_rerun" if force_rerun else "reused",
            "n_subjects": heldout_summary.get("n_subjects"),
        }

        for ext in spec["external"]:
            ext_summary = _load_or_run_external(structure, spec, ext, project_root, force_rerun, limit)
            ext_overall, ext_per_site = _extract_structure_summary(ext_summary, structure)
            rows.extend(_summary_to_rows(structure, "external", ext["dataset"], ext_overall, ext_per_site))
            provenance["sources"][f"{structure}/external/{ext['dataset']}"] = {
                "path": str(project_root / ext["reuse_dir"] / "metrics_summary.yaml"),
                "mode": "force_rerun" if force_rerun else "reused",
                "n_subjects": ext_summary.get("n_subjects"),
            }

    return pd.DataFrame(rows), provenance


def _print_report(df: pd.DataFrame, provenance: dict) -> None:
    """Print a rich Dice-only overview table plus the provenance/caveats panel."""
    overview = df[(df["site"] == "overall") & (df["metric"] == "dice")]
    table = Table(title="Merged Dice overview (overall, per structure/eval)", show_header=True)
    table.add_column("Structure", style="bold cyan")
    table.add_column("Eval type")
    table.add_column("Dataset")
    table.add_column("Mean Dice")
    table.add_column("N")
    for _, row in overview.iterrows():
        table.add_row(
            row["structure"],
            row["eval_type"],
            row["dataset"] or "-",
            f"{row['mean']:.4f}" if pd.notna(row["mean"]) else "nan",
            str(int(row["n"])) if pd.notna(row["n"]) else "0",
        )
    console.print()
    console.print(table)

    notes_text = "\n".join(f"- {note}" for note in provenance["notes"])
    console.print(Panel(notes_text, title="Provenance caveats", border_style="yellow", expand=False))


@app.command()
def evaluate(
    structures: Optional[str] = typer.Option(
        None, "--structures", help="Comma-separated subset of structures to evaluate, e.g. 'cord,canal'. Defaults to all."
    ),
    force_rerun: bool = typer.Option(
        False, "--force-rerun", help="Re-run inference instead of reusing existing metrics_summary.yaml artifacts."
    ),
    output_dir: Path = typer.Option(
        Path("experiments/evaluate_report"), "--output-dir", "-o", help="Where to write merged_metrics.csv and provenance.yaml."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Only process the first N subjects per run (force-rerun only).", min=1
    ),
) -> None:
    """Build and print one merged evaluation report across structures x sites x external datasets."""
    console.print(Panel("[bold]CNS Merged Evaluation Report[/bold]", border_style="blue", expand=False))

    structure_list = [s.strip() for s in structures.split(",")] if structures else None

    try:
        df, provenance = build_report(_PROJECT_ROOT, structures=structure_list, force_rerun=force_rerun, limit=limit)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)

    out_dir = output_dir if output_dir.is_absolute() else _PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "merged_metrics.csv"
    provenance_path = out_dir / "provenance.yaml"
    df.to_csv(csv_path, index=False)
    with open(provenance_path, "w") as f:
        yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)

    _print_report(df, provenance)
    console.print(f"\n[bold green]Merged metrics written to:[/bold green] {csv_path}")
    console.print(f"[bold green]Provenance written to:[/bold green] {provenance_path}")


if __name__ == "__main__":
    app()
