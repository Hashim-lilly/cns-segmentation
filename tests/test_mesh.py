"""Tests for cns_segmentation.mesh.export and cns_segmentation.mesh.quality."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from cns_segmentation.mesh.export import (
    MeshExportConfig,
    _pymeshlab_repair,
    export_cfd_mesh,
    repair_mesh,
    validate_mesh,
)
from cns_segmentation.mesh.quality import is_manifold


class TestComponentFilter:
    def test_rejects_tiny_nonwatertight_sliver(self, icosphere_with_tiny_sliver: trimesh.Trimesh) -> None:
        config = MeshExportConfig(smooth_iterations=0)
        repaired = repair_mesh(icosphere_with_tiny_sliver, config)
        quality = validate_mesh(repaired)

        assert quality.is_watertight is True
        assert quality.euler_number == 2

    def test_keeps_fragment_above_area_threshold(self, icosphere_mesh: trimesh.Trimesh) -> None:
        # A large (>20mm^2), non-watertight open patch should survive the filter.
        open_patch = trimesh.Trimesh(vertices=icosphere_mesh.vertices.copy(), faces=icosphere_mesh.faces[:100].copy())
        assert open_patch.area > 20.0
        assert open_patch.is_watertight is False

        combined = trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=icosphere_mesh.vertices.copy(), faces=icosphere_mesh.faces[100:].copy()), open_patch]
        )
        config = MeshExportConfig(smooth_iterations=0, min_component_volume_mm3=1e9)
        components_before = combined.split(only_watertight=False)
        repaired = repair_mesh(combined, config)

        # Both fragments are non-watertight individually (split from a single
        # watertight sphere), so both clear the area threshold and get
        # concatenated back — total face count is preserved, not reduced.
        assert len(components_before) == 2
        assert len(repaired.faces) == len(combined.faces)


class TestPyMeshLabFallback:
    def test_unfixable_by_trimesh_alone(self, unfixable_by_trimesh_mesh: trimesh.Trimesh) -> None:
        config = MeshExportConfig(smooth_iterations=0)
        repaired = repair_mesh(unfixable_by_trimesh_mesh, config)
        quality = validate_mesh(repaired)

        assert quality.is_watertight is False

    def test_pymeshlab_fixes_what_trimesh_cannot(self, unfixable_by_trimesh_mesh: trimesh.Trimesh) -> None:
        config = MeshExportConfig(smooth_iterations=0)
        repaired = repair_mesh(unfixable_by_trimesh_mesh, config)
        assert validate_mesh(repaired).is_watertight is False

        fixed = _pymeshlab_repair(repaired, config)
        quality = validate_mesh(fixed)

        assert quality.is_watertight is True
        assert quality.euler_number == 2


class TestManifoldReconciliation:
    def test_closed_icosphere_is_manifold(self, icosphere_mesh: trimesh.Trimesh) -> None:
        assert is_manifold(icosphere_mesh) is True

    def test_open_mesh_with_consistent_winding_is_not_manifold(
        self, open_mesh_one_face_removed: trimesh.Trimesh
    ) -> None:
        # Empirically confirmed quirk: an open mesh can have consistent
        # winding among its surviving (exactly-2-face) edges while still
        # having a boundary edge — is_winding_consistent alone says nothing
        # about closedness, so it must not be used as the manifold check.
        assert open_mesh_one_face_removed.is_watertight is False
        assert open_mesh_one_face_removed.is_winding_consistent is True
        assert is_manifold(open_mesh_one_face_removed) is False


class TestExportCfdMeshIntegration:
    def _sphere_mask_path(self, tmp_path: Path) -> Path:
        shape = (40, 40, 40)
        grid = np.indices(shape) - np.array(shape)[:, None, None, None] / 2
        radius = np.sqrt((grid**2).sum(axis=0))
        binary = (radius < 12).astype(np.uint8)
        mask_path = tmp_path / "sphere_mask.nii.gz"
        nib.save(nib.Nifti1Image(binary, np.eye(4)), str(mask_path))
        return mask_path

    def test_clean_sphere_mask_passes_without_fallback(self, tmp_path: Path) -> None:
        mask_path = self._sphere_mask_path(tmp_path)
        output_path = tmp_path / "out.stl"
        config = MeshExportConfig(smooth_iterations=3)

        quality = export_cfd_mesh(mask_path=mask_path, output_path=output_path, config=config)

        assert quality is not None
        assert quality.passes_cfd_check is True
        assert quality.repair_forced is False
        assert output_path.exists()
