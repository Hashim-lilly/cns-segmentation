#!/usr/bin/env python
"""Run the automated CFD segmentation pipeline on spine-generic subjects.

Usage:
    python scripts/run_pipeline.py --input data/spine-generic/sub-amu01/anat/sub-amu01_T2w.nii.gz
    python scripts/run_pipeline.py --batch data/spine-generic/ --output outputs/

This script orchestrates the pre-trained model pipeline:
  1. TotalSpineSeg / SCT → cord + canal
  2. model-canal-seg → dural sac
  3. RootletSeg → nerve rootlets
  4. Boolean extraction → CSF domain
  5. Mesh export → watertight STL
"""

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.pipeline import PipelineResult, SegmentationPipeline
from src.mesh.export import MeshExportConfig, export_cfd_mesh

app = typer.Typer(help="CFD Segmentation Pipeline — Spinal SAS")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def run_single(
    input: Path = typer.Option(..., "--input", "-i", help="Path to T2w NIfTI file"),
    output_dir: Path = typer.Option(
        Path("outputs/pipeline"), "--output", "-o", help="Output directory"
    ),
    skip_rootlets: bool = typer.Option(
        False, "--skip-rootlets", help="Skip rootlet segmentation"
    ),
    export_mesh: bool = typer.Option(
        True, "--mesh/--no-mesh", help="Export STL mesh from CSF domain"
    ),
    smooth_iterations: int = typer.Option(
        10, "--smooth", help="Taubin smoothing iterations for mesh"
    ),
):
    """Run pipeline on a single T2w volume."""
    console.print(f"\n[bold green]CNS CFD Segmentation Pipeline[/bold green]")
    console.print(f"Input: {input}")
    console.print(f"Output: {output_dir}\n")

    # Initialize pipeline
    pipeline = SegmentationPipeline(
        output_dir=output_dir,
        use_canal_seg=True,
        use_rootlets=not skip_rootlets,
    )

    # Run segmentation
    result = pipeline.run(input_t2w=input)

    if not result.success:
        console.print(f"[bold red]Pipeline failed![/bold red]")
        for err in result.errors:
            console.print(f"  ✗ {err}")
        raise typer.Exit(code=1)

    # Display results
    _print_result(result)

    # Export mesh
    if export_mesh and result.csf_domain is not None:
        mesh_config = MeshExportConfig(smooth_iterations=smooth_iterations)
        stl_path = output_dir / result.csf_domain.stem.replace(".nii", "") / "csf_domain.stl"

        console.print("\n[bold]Exporting CFD mesh...[/bold]")
        quality = export_cfd_mesh(
            mask_path=result.csf_domain,
            output_path=stl_path,
            config=mesh_config,
        )

        if quality is not None:
            _print_mesh_quality(quality, stl_path)

    console.print("\n[bold green]✓ Pipeline complete![/bold green]")


@app.command()
def run_batch(
    data_dir: Path = typer.Option(
        ..., "--data", "-d", help="Path to spine-generic data directory"
    ),
    output_dir: Path = typer.Option(
        Path("outputs/batch"), "--output", "-o", help="Output directory"
    ),
    max_subjects: int = typer.Option(
        0, "--max", "-n", help="Max subjects to process (0=all)"
    ),
    skip_rootlets: bool = typer.Option(
        False, "--skip-rootlets", help="Skip rootlet segmentation"
    ),
):
    """Run pipeline on multiple subjects from a BIDS dataset."""
    console.print(f"\n[bold green]CNS CFD Pipeline — Batch Mode[/bold green]")

    # Find all T2w files
    t2w_files = sorted(data_dir.glob("sub-*/anat/*_T2w.nii.gz"))

    if not t2w_files:
        console.print(f"[red]No T2w files found in {data_dir}[/red]")
        raise typer.Exit(code=1)

    if max_subjects > 0:
        t2w_files = t2w_files[:max_subjects]

    console.print(f"Found {len(t2w_files)} T2w volumes to process.\n")

    # Filter git-annex stubs
    valid_files = [f for f in t2w_files if f.stat().st_size > 1000]
    console.print(
        f"Valid files (not git-annex stubs): {len(valid_files)} / {len(t2w_files)}"
    )

    # Process each subject
    pipeline = SegmentationPipeline(
        output_dir=output_dir,
        use_canal_seg=True,
        use_rootlets=not skip_rootlets,
    )

    results: list[PipelineResult] = []
    for i, t2w_path in enumerate(valid_files, 1):
        console.print(f"\n[bold]─── [{i}/{len(valid_files)}] {t2w_path.name} ───[/bold]")
        result = pipeline.run(input_t2w=t2w_path)
        results.append(result)

    # Summary
    _print_batch_summary(results)


def _print_result(result: PipelineResult):
    """Print pipeline result in a formatted table."""
    table = Table(title="Segmentation Results")
    table.add_column("Output", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Status")

    outputs = [
        ("Cord mask", result.cord_mask),
        ("Canal mask", result.canal_mask),
        ("Rootlet mask", result.rootlet_mask),
        ("CSF domain", result.csf_domain),
        ("Combined labels", result.combined_labels),
    ]

    for name, path in outputs:
        if path is not None and path.exists():
            table.add_row(name, str(path), "✓")
        elif path is not None:
            table.add_row(name, str(path), "✗ missing")
        else:
            table.add_row(name, "—", "skipped")

    console.print(table)


def _print_mesh_quality(quality, stl_path: Path):
    """Print mesh quality metrics."""
    table = Table(title="Mesh Quality Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_column("Pass", style="green")

    table.add_row("Watertight", str(quality.is_watertight), "✓" if quality.is_watertight else "✗")
    table.add_row("Manifold", str(quality.is_manifold), "✓" if quality.is_manifold else "✗")
    table.add_row("Euler number", str(quality.euler_number), "✓" if quality.euler_number == 2 else "✗")
    table.add_row("Vertices", str(quality.vertex_count), "—")
    table.add_row("Faces", str(quality.face_count), "—")
    table.add_row("Volume", f"{quality.volume_cm3:.1f} cm³", "—")
    table.add_row("CFD-ready", str(quality.passes_cfd_check), "✓" if quality.passes_cfd_check else "✗")
    table.add_row("Output", str(stl_path), "—")

    console.print(table)


def _print_batch_summary(results: list[PipelineResult]):
    """Print batch processing summary."""
    n_total = len(results)
    n_success = sum(1 for r in results if r.success)
    n_failed = n_total - n_success

    console.print(f"\n[bold]═══ Batch Summary ═══[/bold]")
    console.print(f"  Total: {n_total}")
    console.print(f"  [green]Success: {n_success}[/green]")
    if n_failed > 0:
        console.print(f"  [red]Failed: {n_failed}[/red]")
        for r in results:
            if not r.success:
                console.print(f"    ✗ {r.input_path.name}: {', '.join(r.errors)}")


if __name__ == "__main__":
    app()
