"""Shared pytest fixtures for the CNS test suite."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh


@pytest.fixture()
def icosphere_mesh() -> trimesh.Trimesh:
    """A closed, watertight, manifold reference mesh (χ=2).

    Radius 10 (not trimesh's unit-radius default) so its volume (~4047 mm³)
    and surface area (~1233 mm²) clear the module's default component
    thresholds (`min_component_volume_mm3=100.0`, `min_component_area_mm2=20.0`)
    — a unit-radius sphere (volume ~4.2 mm³) would itself look like a
    rejectable small fragment.
    """
    return trimesh.creation.icosphere(subdivisions=2, radius=10.0)


@pytest.fixture()
def icosphere_with_tiny_sliver(icosphere_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Watertight icosphere plus a disconnected, non-watertight, tiny-area triangle.

    Mirrors a spurious marching-cubes fragment: never watertight (single
    triangle = open boundary) and far below the 20 mm² default area
    threshold (area=0.125 mm²), so it should be rejected by the new
    component-area filter in `repair_mesh()`.
    """
    sliver = trimesh.Trimesh(
        vertices=[[100.0, 100.0, 100.0], [100.5, 100.0, 100.0], [100.0, 100.5, 100.0]],
        faces=[[0, 1, 2]],
    )
    return trimesh.util.concatenate([icosphere_mesh, sliver])


@pytest.fixture()
def unfixable_by_trimesh_mesh(icosphere_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Icosphere with 3 faces removed — empirically confirmed NOT closeable by
    trimesh's own fill_holes/fix_normals/fix_winding alone (stays non-watertight,
    euler=1), but closeable by the PyMeshLab fallback (euler->2). Removing only
    1 face is not a valid fixture here: trimesh alone already closes that case.
    """
    return trimesh.Trimesh(vertices=icosphere_mesh.vertices.copy(), faces=icosphere_mesh.faces[:-3].copy())


@pytest.fixture()
def open_mesh_one_face_removed(icosphere_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Icosphere with 1 face removed: is_watertight=False but is_winding_consistent=True.

    This is the case that made the old `is_manifold = mesh.is_winding_consistent`
    (cns-segmentation) / `is_manifold = mesh.is_watertight` (cns-cfd-simulation)
    definitions disagree — neither field alone is a correct manifold check.
    """
    return trimesh.Trimesh(vertices=icosphere_mesh.vertices.copy(), faces=icosphere_mesh.faces[:-1].copy())


@pytest.fixture()
def sample_config() -> dict:
    """Return a minimal configuration dictionary matching train_spine.yaml structure.

    Provides just enough keys for the transform factory functions to operate.
    """
    return {
        "spacing": [1.0, 0.5, 0.5],
        "patch_size": [48, 160, 160],
        "num_samples": 2,
    }


@pytest.fixture()
def synthetic_nifti_pair(tmp_path: Path) -> dict[str, Path]:
    """Create a synthetic image + label NIfTI pair on disk.

    The image is a random float volume and the label is a binary mask with a
    small foreground blob to ensure RandCropByPosNegLabel can find positive
    voxels.

    Returns:
        Dictionary with ``"image"`` and ``"label"`` keys pointing to NIfTI paths.
    """
    shape = (64, 192, 192)
    affine = np.diag([1.0, 0.5, 0.5, 1.0])

    # Synthetic T2w-like image (random intensities)
    rng = np.random.default_rng(seed=42)
    image_data = rng.standard_normal(shape).astype(np.float32)

    # Binary label with a central blob ensuring positive voxels exist
    label_data = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    label_data[cx - 5 : cx + 5, cy - 10 : cy + 10, cz - 10 : cz + 10] = 1

    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"

    nib.save(nib.Nifti1Image(image_data, affine), str(image_path))
    nib.save(nib.Nifti1Image(label_data, affine), str(label_path))

    return {"image": image_path, "label": label_path}


@pytest.fixture()
def synthetic_multistructure_nifti(tmp_path: Path) -> dict[str, Path]:
    """Create a synthetic image + two per-structure label NIfTIs with overlap.

    "cord" and "canal" masks deliberately overlap in a central region
    (mirroring real anatomy — the canal is a superset of the cord), so
    CompositeLabeld's overlap-resolution logic is actually exercised rather
    than only tested against disjoint regions.

    Returns:
        Dictionary with "image", "label_cord", "label_canal" keys pointing
        to NIfTI paths.
    """
    shape = (64, 192, 192)
    affine = np.diag([1.0, 0.5, 0.5, 1.0])

    rng = np.random.default_rng(seed=42)
    image_data = rng.standard_normal(shape).astype(np.float32)

    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2

    cord_data = np.zeros(shape, dtype=np.uint8)
    cord_data[cx - 5 : cx + 5, cy - 10 : cy + 10, cz - 10 : cz + 10] = 1

    # Canal fully contains the cord region (deliberate overlap) plus a wider
    # surrounding ring, mirroring real spine-generic anatomy.
    canal_data = np.zeros(shape, dtype=np.uint8)
    canal_data[cx - 8 : cx + 8, cy - 15 : cy + 15, cz - 15 : cz + 15] = 1

    image_path = tmp_path / "image.nii.gz"
    cord_path = tmp_path / "label_cord.nii.gz"
    canal_path = tmp_path / "label_canal.nii.gz"

    nib.save(nib.Nifti1Image(image_data, affine), str(image_path))
    nib.save(nib.Nifti1Image(cord_data, affine), str(cord_path))
    nib.save(nib.Nifti1Image(canal_data, affine), str(canal_path))

    return {"image": image_path, "label_cord": cord_path, "label_canal": canal_path}
