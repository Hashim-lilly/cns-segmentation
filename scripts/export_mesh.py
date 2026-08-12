"""Export a watertight STL mesh from a NIfTI segmentation mask."""

import logging
from pathlib import Path

import typer
import yaml
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(
    input: Path = typer.Option(..., "--input", help="Input NIfTI segmentation (.nii.gz)"),
    output: Path = typer.Option(..., "--output", help="Output STL path"),
    label: int = typer.Option(1, "--label", help="Integer label value to extract"),
    config: Path = typer.Option(None, "--config", help="Optional inference YAML config"),
) -> None:
    """Convert a NIfTI segmentation mask to a watertight CFD-ready STL mesh."""
    from cns_segmentation.mesh import MeshExporter, MeshExportConfig

    mesh_cfg = {}
    if config is not None:
        cfg = yaml.safe_load(config.read_text())
        mesh_cfg = cfg.get("mesh", {})

    export_config = MeshExportConfig(
        smooth_iterations=mesh_cfg.get("smooth_iterations", 10),
        smooth_lambda=mesh_cfg.get("smooth_lambda", 0.5),
        smooth_mu=mesh_cfg.get("smooth_mu", -0.53),
        decimate_target=mesh_cfg.get("decimate_target"),
        min_component_volume_mm3=mesh_cfg.get("min_component_volume_mm3", 100.0),
    )

    exporter = MeshExporter(export_config)
    quality = exporter.export(input, output, label_value=label)

    if quality.passes_cfd_check:
        logger.info("Mesh passes CFD quality check: %s", output)
    else:
        logger.warning(
            "Mesh FAILED CFD quality check (watertight=%s, manifold=%s, degenerate=%s)",
            quality.is_watertight,
            quality.is_manifold,
            quality.has_degenerate_faces,
        )


if __name__ == "__main__":
    app()
