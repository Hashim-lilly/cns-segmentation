#!/usr/bin/env python
"""CLI entry point for scoring pretrained comparison baselines against held-out sites.

Runs SCT's cord and rootlet models (`sct_deepseg spinalcord`/`sct_deepseg
rootlets`) and, if available, TotalSegmentator-MRI's cord class over the same
held-out spine-generic sites `scripts/evaluate.py` scores this repo's own
checkpoints on. Every baseline mask is scored through the existing
`evaluate_subject`/`aggregate_metrics` in `evaluation/metrics.py` — no new
metrics code. Output schema matches `scripts/evaluate.py`'s
`merged_metrics.csv` (structure, eval_type, dataset, site, metric,
mean/median/std/n) with `eval_type="baseline"` and `dataset=<baseline name>`,
so the two tables concatenate cleanly for the report/dashboard.

Ground-truth grid compatibility for SCT was verified directly (Step 0 of the
Phase 5 plan): SCT's native output grid matches the raw ground-truth label
grid exactly for both cord and rootlets, so no resampling step is needed —
baselines are scored directly against the raw (unresampled) dataset files,
not against `predict.py`'s canonical resampled grid.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --baselines sct_cord --limit 3
    python scripts/run_baselines.py --totalsegmentator-venv /tmp/ts_venv/bin/python
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

from cns_segmentation.baselines.sct_runner import score_sct_baseline
from cns_segmentation.baselines.totalsegmentator_runner import (
    check_totalsegmentator_available,
    score_totalsegmentator_cord,
)
from cns_segmentation.evaluation.metrics import aggregate_metrics
from cns_segmentation.inference import list_heldout_subjects, load_yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="run-baselines",
    help="Score pretrained comparison baselines (SCT, TotalSegmentator) against held-out sites.",
    add_completion=False,
)

METRIC_KEYS = ["dice", "hausdorff95_mm", "volume_error_mm3", "surface_dice"]

# Which structure's held-out subjects each baseline is scored against, and
# the train config used only to recover that structure's site split + which
# label_<structure> key to use as ground truth (no model/checkpoint loading).
BASELINE_STRUCTURES = {
    "sct_cord": {"structure": "cord", "train_config": "configs/train_spine.yaml"},
    "sct_rootlets": {"structure": "rootlets", "train_config": "configs/train_spine_rootlets.yaml"},
    "totalsegmentator_cord": {"structure": "cord", "train_config": "configs/train_spine.yaml"},
}


def _rows_from_results(baseline_name: str, structure: str, results: list[dict]) -> list[dict]:
    """Flatten aggregate_metrics() output into merged_metrics.csv rows for one baseline."""
    scored = [r for r in results if "error" not in r]
    if not scored:
        return []
    summary = aggregate_metrics(scored)
    rows = []
    for metric in METRIC_KEYS:
        stats = summary["overall"].get(metric, {})
        rows.append(
            {
                "structure": structure, "eval_type": "baseline", "dataset": baseline_name,
                "site": "overall", "metric": metric,
                "mean": stats.get("mean"), "median": stats.get("median"),
                "std": stats.get("std"), "n": stats.get("n"),
            }
        )
    for site, site_stats in summary["per_site"].items():
        for metric in METRIC_KEYS:
            stats = site_stats.get(metric, {})
            rows.append(
                {
                    "structure": structure, "eval_type": "baseline", "dataset": baseline_name,
                    "site": site, "metric": metric,
                    "mean": stats.get("mean"), "median": stats.get("median"),
                    "std": stats.get("std"), "n": stats.get("n"),
                }
            )
    return rows


def run_baseline(
    baseline_name: str, project_root: Path, output_dir: Path,
    totalsegmentator_venv: Optional[Path], limit: Optional[int], timeout: int,
) -> tuple[list[dict], list[dict]]:
    """Run one baseline over its held-out subjects.

    Returns:
        (results, rows) — raw per-subject results (including failures, for
        the caller to log) and the flattened merged-table rows (failures
        excluded, since aggregate_metrics() only accepts scored results).
    """
    spec = BASELINE_STRUCTURES[baseline_name]
    structure = spec["structure"]
    train_config = load_yaml(project_root / spec["train_config"])
    datalist = list_heldout_subjects(train_config, structure, project_root)
    if limit is not None:
        datalist = datalist[:limit]
    if not datalist:
        raise ValueError(f"No held-out subjects found for baseline '{baseline_name}' (structure={structure})")

    baseline_out_dir = output_dir / baseline_name
    results = []
    for item in datalist:
        subject_id, site = item["subject"], item["site"]
        logger.info("[%s] scoring %s (site=%s)", baseline_name, subject_id, site)
        if baseline_name == "sct_cord":
            result = score_sct_baseline("sct_cord", subject_id, site, item["image"], item["label"], baseline_out_dir, timeout=timeout)
        elif baseline_name == "sct_rootlets":
            result = score_sct_baseline("sct_rootlets", subject_id, site, item["image"], item["label"], baseline_out_dir, timeout=timeout)
        elif baseline_name == "totalsegmentator_cord":
            if totalsegmentator_venv is None or not check_totalsegmentator_available(totalsegmentator_venv):
                result = {"subject": subject_id, "site": site, "error": "totalsegmentator_not_available"}
            else:
                result = score_totalsegmentator_cord(
                    totalsegmentator_venv, subject_id, site, item["image"], item["label"], baseline_out_dir, timeout=timeout
                )
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        if "error" in result:
            logger.warning("[%s] %s failed: %s", baseline_name, subject_id, result["error"])
        results.append(result)

    return results, _rows_from_results(baseline_name, structure, results)


@app.command()
def run_baselines(
    baselines: Optional[str] = typer.Option(
        None, "--baselines", help="Comma-separated subset, e.g. 'sct_cord,sct_rootlets'. Defaults to all."
    ),
    output_dir: Path = typer.Option(
        Path("experiments/baselines_report"), "--output-dir", "-o", help="Where to write merged_metrics.csv and predictions."
    ),
    totalsegmentator_venv: Optional[Path] = typer.Option(
        None, "--totalsegmentator-venv", help="Path to a venv's python with totalsegmentator installed. Skipped if omitted or unavailable."
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only process the first N subjects per baseline.", min=1),
    timeout: int = typer.Option(600, "--timeout", help="Max seconds per-subject subprocess timeout."),
) -> None:
    """Score every requested baseline against its held-out subjects and write a merged report."""
    console.print(Panel("[bold]CNS Comparison Baselines[/bold]", border_style="blue", expand=False))

    names = [b.strip() for b in baselines.split(",")] if baselines else list(BASELINE_STRUCTURES.keys())
    unknown = [n for n in names if n not in BASELINE_STRUCTURES]
    if unknown:
        console.print(f"[bold red]Unknown baseline(s): {unknown}. Known: {list(BASELINE_STRUCTURES)}[/bold red]")
        raise typer.Exit(code=1)

    out_dir = output_dir if output_dir.is_absolute() else _PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    blockers: dict[str, str] = {}
    for name in names:
        try:
            results, rows = run_baseline(name, _PROJECT_ROOT, out_dir, totalsegmentator_venv, limit, timeout)
        except ValueError as exc:
            console.print(f"[bold red]{name}: {exc}[/bold red]")
            blockers[name] = str(exc)
            continue
        failed = [r for r in results if "error" in r]
        if failed and len(failed) == len(results):
            blockers[name] = failed[0]["error"]
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "merged_metrics.csv"
    df.to_csv(csv_path, index=False)
    with open(out_dir / "blockers.yaml", "w") as f:
        yaml.dump(blockers, f, default_flow_style=False, sort_keys=False)

    if not df.empty:
        table = Table(title="Baseline Dice overview (overall)", show_header=True)
        table.add_column("Baseline", style="bold cyan")
        table.add_column("Structure")
        table.add_column("Mean Dice")
        table.add_column("N")
        overview = df[(df["site"] == "overall") & (df["metric"] == "dice")]
        for _, row in overview.iterrows():
            table.add_row(row["dataset"], row["structure"], f"{row['mean']:.4f}" if pd.notna(row["mean"]) else "nan", str(int(row["n"])) if pd.notna(row["n"]) else "0")
        console.print()
        console.print(table)

    if blockers:
        console.print(Panel("\n".join(f"- {k}: {v}" for k, v in blockers.items()), title="Blockers", border_style="red", expand=False))

    console.print(f"\n[bold green]Merged baseline metrics written to:[/bold green] {csv_path}")
    console.print(f"[bold green]Blockers written to:[/bold green] {out_dir / 'blockers.yaml'}")


if __name__ == "__main__":
    app()
