"""CFD-ready mesh generation from multi-label segmentation masks.

Produces watertight, manifold surface meshes of the spinal subarachnoid space
suitable for OpenFOAM or SimVascular CFD simulation.

Pipeline:
  1. Marching cubes → surface extraction per structure
  2. Boolean subtraction → CSF domain surface
  3. Fill holes → fix normals → remove degenerate faces
  4. Taubin smoothing (preserve volume, remove staircase artifacts)
  5. Validate: watertight + manifold + Euler χ=2
  6. Export STL

Critical rule: fill holes → fix normals → remove degenerate → smooth.
Never smooth first (per CLAUDE.md).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from skimage import measure

logger = logging.getLogger(__name__)

try:
    import trimesh
except ImportError:
    trimesh = None
    logger.warning("trimesh not installed; mesh operations unavailable.")


@dataclass
class MeshQuality:
    """Quality metrics for a generated mesh."""

    is_watertight: bool = False
    is_manifold: bool = False
    euler_number: int = 0
    vertex_count: int = 0
    face_count: int = 0
    volume_cm3: float = 0.0
    has_degenerate_faces: bool = False
    min_face_area: float = 0.0
    passes_cfd_check: bool = False


@dataclass
class MeshExportConfig:
    """Configuration for CFD mesh export."""

    smooth_iterations: int = 10
    smooth_lambda: float = 0.5  # Taubin smoothing lambda
    smooth_mu: float = -0.53  # Taubin smoothing mu (negative for volume preservation)
    decimate_target: Optional[int] = None  # Target face count; None = no decimation
    min_component_volume_mm3: float = 100.0  # Remove components smaller than this
    fill_holes_max_size: int = 100  # Max hole size (edges) to auto-fill
    output_format: str = "stl"


def extract_surface(
    mask_path: Path,
    config: Optional[MeshExportConfig] = None,
    label_value: int = 1,
) -> Optional["trimesh.Trimesh"]:
    """Extract a surface mesh from a binary NIfTI mask using marching cubes.

    Args:
        mask_path: Path to binary NIfTI segmentation mask.
        config: Mesh export configuration.
        label_value: Label value to extract (default 1 for binary masks).

    Returns:
        trimesh.Trimesh mesh object, or None on failure.
    """
    if trimesh is None:
        raise ImportError("trimesh is required for mesh operations: pip install trimesh")

    if config is None:
        config = MeshExportConfig()

    # Load mask
    nii = nib.load(mask_path)
    data = np.asarray(nii.dataobj)
    spacing = nii.header.get_zooms()[:3]

    # Extract the specified label
    binary_mask = (data == label_value).astype(np.float32)

    if binary_mask.sum() == 0:
        logger.warning("Empty mask for label %d in %s", label_value, mask_path)
        return None

    # Marching cubes
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            binary_mask, level=0.5, spacing=spacing
        )
    except (ValueError, RuntimeError) as e:
        logger.error("Marching cubes failed: %s", e)
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    logger.info(
        "Marching cubes: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces)
    )

    return mesh


def repair_mesh(
    mesh: "trimesh.Trimesh", config: Optional[MeshExportConfig] = None
) -> "trimesh.Trimesh":
    """Repair a mesh for CFD readiness.

    Order: fill holes → fix normals → remove degenerate → smooth.
    NEVER smooth first.

    Args:
        mesh: Input trimesh object.
        config: Mesh export configuration.

    Returns:
        Repaired mesh.
    """
    if config is None:
        config = MeshExportConfig()

    # Step 1: Fill holes
    trimesh.repair.fill_holes(mesh)
    logger.info("After hole filling: watertight=%s", mesh.is_watertight)

    # Step 2: Fix normals (consistent winding)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    # Step 3: Remove degenerate faces (zero-area triangles)
    # Identify degenerate faces by area
    face_areas = mesh.area_faces
    degenerate_mask = face_areas < 1e-10
    if degenerate_mask.any():
        n_degenerate = degenerate_mask.sum()
        # Remove degenerate faces
        mesh.update_faces(~degenerate_mask)
        logger.info("Removed %d degenerate faces.", n_degenerate)

    # Step 4: Remove small disconnected components
    if config.min_component_volume_mm3 > 0:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Keep only components above the volume threshold
            kept = []
            for comp in components:
                if comp.is_watertight and comp.volume > config.min_component_volume_mm3:
                    kept.append(comp)
                elif not comp.is_watertight:
                    # Keep non-watertight components (might be the main body)
                    kept.append(comp)
            if kept:
                mesh = trimesh.util.concatenate(kept)
                logger.info(
                    "Kept %d of %d components (volume threshold: %.1f mm³).",
                    len(kept),
                    len(components),
                    config.min_component_volume_mm3,
                )

    # Step 5: Taubin smoothing (volume-preserving)
    if config.smooth_iterations > 0:
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=config.smooth_lambda,
            nu=config.smooth_mu,
            iterations=config.smooth_iterations,
        )
        logger.info(
            "Taubin smoothing: %d iterations (λ=%.2f, μ=%.2f)",
            config.smooth_iterations,
            config.smooth_lambda,
            config.smooth_mu,
        )

    # Step 6: Optional decimation
    if config.decimate_target is not None and len(mesh.faces) > config.decimate_target:
        mesh = mesh.simplify_quadric_decimation(config.decimate_target)
        logger.info("Decimated to %d faces.", len(mesh.faces))

    return mesh


def validate_mesh(mesh: "trimesh.Trimesh") -> MeshQuality:
    """Validate mesh quality for CFD readiness.

    A CFD-ready mesh must be:
      - Watertight (no holes)
      - Manifold (consistent winding, every edge shared by exactly 2 faces)
      - Euler characteristic χ = 2 (topological sphere)
      - No degenerate faces

    Args:
        mesh: Trimesh object to validate.

    Returns:
        MeshQuality dataclass with all quality metrics.
    """
    quality = MeshQuality()

    quality.is_watertight = mesh.is_watertight
    quality.is_manifold = mesh.is_winding_consistent
    quality.euler_number = mesh.euler_number
    quality.vertex_count = len(mesh.vertices)
    quality.face_count = len(mesh.faces)

    # Volume (only meaningful if watertight)
    if mesh.is_watertight:
        quality.volume_cm3 = mesh.volume / 1000.0  # mm³ → cm³

    # Check for degenerate faces
    face_areas = mesh.area_faces
    quality.has_degenerate_faces = bool((face_areas < 1e-10).any())
    quality.min_face_area = float(face_areas.min()) if len(face_areas) > 0 else 0.0

    # CFD pass: watertight + manifold + Euler 2 + no degenerate
    quality.passes_cfd_check = (
        quality.is_watertight
        and quality.is_manifold
        and quality.euler_number == 2
        and not quality.has_degenerate_faces
    )

    logger.info(
        "Mesh QC: watertight=%s, manifold=%s, euler=%d, "
        "vertices=%d, faces=%d, volume=%.1f cm³, CFD-ready=%s",
        quality.is_watertight,
        quality.is_manifold,
        quality.euler_number,
        quality.vertex_count,
        quality.face_count,
        quality.volume_cm3,
        quality.passes_cfd_check,
    )

    return quality


def export_cfd_mesh(
    mask_path: Path,
    output_path: Path,
    config: Optional[MeshExportConfig] = None,
    label_value: int = 1,
) -> Optional[MeshQuality]:
    """Full pipeline: NIfTI mask → CFD-ready STL with quality report.

    Args:
        mask_path: Path to segmentation mask NIfTI.
        output_path: Path for output mesh file (STL/OBJ/PLY).
        config: Mesh export configuration.
        label_value: Label value to extract from multi-label mask.

    Returns:
        MeshQuality report, or None if pipeline failed.
    """
    if config is None:
        config = MeshExportConfig()

    # Extract surface
    mesh = extract_surface(mask_path, config, label_value)
    if mesh is None:
        return None

    # Repair
    mesh = repair_mesh(mesh, config)

    # Validate
    quality = validate_mesh(mesh)

    # Export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type=config.output_format)
    logger.info("Mesh exported to: %s", output_path)

    if not quality.passes_cfd_check:
        logger.warning(
            "⚠️  Mesh does NOT pass CFD readiness check. "
            "Review quality metrics before using in simulation."
        )

    return quality
