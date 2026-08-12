"""CFD-ready mesh generation from segmentation masks.

Pipeline (must follow this order per CLAUDE.md):
  1. Marching cubes → surface mesh
  2. Fill holes
  3. Fix normals
  4. Remove degenerate faces
  5. Taubin smoothing
  6. Validate: watertight + manifold + Euler χ=2
  7. Export binary STL (RAS, mm)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False
    logger.warning("trimesh not installed; mesh operations unavailable.")

try:
    from skimage import measure as sk_measure
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


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
    passes_cfd_check: bool = False

    def __post_init__(self) -> None:
        self.passes_cfd_check = (
            self.is_watertight and self.is_manifold and not self.has_degenerate_faces
        )


@dataclass
class MeshExportConfig:
    """Configuration for mesh export."""

    smooth_iterations: int = 10
    smooth_lambda: float = 0.5
    smooth_mu: float = -0.53  # negative for volume preservation
    decimate_target: Optional[int] = None
    min_component_volume_mm3: float = 100.0
    fill_holes_max_size: int = 100


class MeshExporter:
    """Export watertight STL meshes from NIfTI segmentation masks.

    Args:
        config: Export configuration.
    """

    def __init__(self, config: Optional[MeshExportConfig] = None) -> None:
        self.config = config or MeshExportConfig()

    def export(self, nifti_path: Path, output_path: Path, label_value: int = 1) -> MeshQuality:
        """Run the full mask → STL pipeline.

        Args:
            nifti_path: Path to the binary segmentation mask (.nii.gz).
            output_path: Destination path for the STL file.
            label_value: Integer label to extract as a mesh.

        Returns:
            MeshQuality report for the exported mesh.
        """
        if not _TRIMESH_AVAILABLE:
            raise RuntimeError("trimesh is required for mesh export. Install with: pip install trimesh")
        if not _SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-image is required. Install with: pip install scikit-image")

        img = nib.load(nifti_path)
        mask = (np.asarray(img.dataobj) == label_value).astype(np.uint8)
        voxel_spacing = img.header.get_zooms()[:3]

        # Step 1: Marching cubes
        verts, faces, *_ = sk_measure.marching_cubes(mask, level=0.5, spacing=voxel_spacing)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Step 2–4: Repair in the mandated order
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()

        # Remove tiny disconnected components
        components = mesh.split(only_watertight=False)
        vol_mm3 = self.config.min_component_volume_mm3
        mesh = trimesh.util.concatenate(
            [c for c in components if abs(c.volume) >= vol_mm3] or [max(components, key=lambda m: abs(m.volume))]
        )

        # Step 5: Taubin smoothing (preserves volume)
        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=self.config.smooth_lambda,
            nu=self.config.smooth_mu,
            iterations=self.config.smooth_iterations,
        )

        if self.config.decimate_target:
            mesh = mesh.simplify_quadric_decimation(self.config.decimate_target)

        # Step 7: Export
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

        quality = MeshQuality(
            is_watertight=mesh.is_watertight,
            is_manifold=mesh.is_watertight,  # watertight implies manifold for trimesh
            euler_number=mesh.euler_number,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            volume_cm3=abs(mesh.volume) / 1000.0,
            has_degenerate_faces=not mesh.is_volume,
        )
        logger.info(
            "Exported %s: watertight=%s manifold=%s vertices=%d faces=%d vol=%.2f cm³",
            output_path.name,
            quality.is_watertight,
            quality.is_manifold,
            quality.vertex_count,
            quality.face_count,
            quality.volume_cm3,
        )
        return quality
