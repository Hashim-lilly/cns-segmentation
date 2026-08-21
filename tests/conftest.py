"""Shared pytest fixtures for the CNS test suite."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


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
