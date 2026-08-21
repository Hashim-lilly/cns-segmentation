"""Tests for MONAI transform pipelines defined in cns_segmentation.data.transforms.

Validates that training, validation, and post-processing pipelines produce
outputs with correct keys, shapes, dtypes, and deterministic behaviour.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from cns_segmentation.data.transforms import get_post_transforms, get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_nifti(
    path: Path,
    shape: tuple[int, ...] = (64, 192, 192),
    is_label: bool = False,
    seed: int = 0,
) -> Path:
    """Write a synthetic NIfTI volume to *path* and return it.

    Args:
        path: Destination file path (should end in .nii.gz).
        shape: Spatial dimensions of the volume.
        is_label: If True, create a binary mask with a central blob;
            otherwise create random float data.
        seed: RNG seed for reproducibility.

    Returns:
        The same *path* for convenience.
    """
    affine = np.diag([1.0, 0.5, 0.5, 1.0])
    rng = np.random.default_rng(seed=seed)

    if is_label:
        data = np.zeros(shape, dtype=np.uint8)
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        data[cx - 5 : cx + 5, cy - 10 : cy + 10, cz - 10 : cz + 10] = 1
    else:
        data = rng.standard_normal(shape).astype(np.float32)

    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


# ---------------------------------------------------------------------------
# Tests: get_train_transforms
# ---------------------------------------------------------------------------


class TestTrainTransforms:
    """Tests for the training transform pipeline."""

    def test_output_keys(
        self, sample_config: dict, synthetic_nifti_pair: dict[str, Path]
    ) -> None:
        """Output dictionaries must contain 'image' and 'label' keys."""
        transforms = get_train_transforms(sample_config)
        data = {
            "image": str(synthetic_nifti_pair["image"]),
            "label": str(synthetic_nifti_pair["label"]),
        }
        result = transforms(data)

        # RandCropByPosNegLabel produces a list of samples
        assert isinstance(result, list)
        for sample in result:
            assert "image" in sample
            assert "label" in sample

    def test_output_shapes_match_patch_size(
        self, sample_config: dict, synthetic_nifti_pair: dict[str, Path]
    ) -> None:
        """Cropped patches must have spatial shape equal to patch_size."""
        patch_size = sample_config["patch_size"]
        transforms = get_train_transforms(sample_config)
        data = {
            "image": str(synthetic_nifti_pair["image"]),
            "label": str(synthetic_nifti_pair["label"]),
        }
        result = transforms(data)

        for sample in result:
            # Shape is (C, D, H, W); spatial dims are [1:]
            img_spatial = list(sample["image"].shape[1:])
            lbl_spatial = list(sample["label"].shape[1:])
            assert img_spatial == patch_size, (
                f"Image spatial shape {img_spatial} != patch_size {patch_size}"
            )
            assert lbl_spatial == patch_size, (
                f"Label spatial shape {lbl_spatial} != patch_size {patch_size}"
            )

    def test_num_samples_produced(
        self, sample_config: dict, synthetic_nifti_pair: dict[str, Path]
    ) -> None:
        """Number of output samples must equal num_samples in config."""
        num_samples = sample_config["num_samples"]
        transforms = get_train_transforms(sample_config)
        data = {
            "image": str(synthetic_nifti_pair["image"]),
            "label": str(synthetic_nifti_pair["label"]),
        }
        result = transforms(data)

        assert len(result) == num_samples, (
            f"Expected {num_samples} samples, got {len(result)}"
        )

    def test_output_is_tensor(
        self, sample_config: dict, synthetic_nifti_pair: dict[str, Path]
    ) -> None:
        """Image and label in each sample should be tensor-like (MetaTensor)."""
        transforms = get_train_transforms(sample_config)
        data = {
            "image": str(synthetic_nifti_pair["image"]),
            "label": str(synthetic_nifti_pair["label"]),
        }
        result = transforms(data)

        for sample in result:
            assert isinstance(sample["image"], torch.Tensor)
            assert isinstance(sample["label"], torch.Tensor)


# ---------------------------------------------------------------------------
# Tests: get_val_transforms
# ---------------------------------------------------------------------------


class TestValTransforms:
    """Tests for the validation transform pipeline."""

    def test_deterministic(
        self, sample_config: dict, synthetic_nifti_pair: dict[str, Path]
    ) -> None:
        """Same input must produce identical output across two calls."""
        transforms = get_val_transforms(sample_config)
        data = {
            "image": str(synthetic_nifti_pair["image"]),
            "label": str(synthetic_nifti_pair["label"]),
        }

        result_a = transforms(data.copy())
        result_b = transforms(data.copy())

        assert torch.allclose(result_a["image"], result_b["image"]), (
            "Val transform is not deterministic for image"
        )
        assert torch.allclose(result_a["label"], result_b["label"]), (
            "Val transform is not deterministic for label"
        )

    def test_orientation_is_ras(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        """Output metadata must indicate RAS orientation."""
        # Create a volume with a non-RAS affine (LPS)
        lps_affine = np.diag([-1.0, -0.5, 0.5, 1.0])
        shape = (64, 192, 192)
        rng = np.random.default_rng(seed=7)
        image_data = rng.standard_normal(shape).astype(np.float32)
        label_data = np.zeros(shape, dtype=np.uint8)
        label_data[30:34, 90:100, 90:100] = 1

        img_path = tmp_path / "lps_image.nii.gz"
        lbl_path = tmp_path / "lps_label.nii.gz"
        nib.save(nib.Nifti1Image(image_data, lps_affine), str(img_path))
        nib.save(nib.Nifti1Image(label_data, lps_affine), str(lbl_path))

        transforms = get_val_transforms(sample_config)
        result = transforms({"image": str(img_path), "label": str(lbl_path)})

        # MONAI MetaTensor stores affine; after Orientationd(axcodes="RAS")
        # the diagonal of the affine should have positive signs for R, A, S
        meta_affine = result["image"].meta["affine"]
        # RAS means x+ = Right, y+ = Anterior, z+ = Superior
        # Diagonal entries should be positive (or absolute spacing values)
        assert meta_affine[0, 0] > 0, "X-axis not pointing Right (not RAS)"
        assert meta_affine[1, 1] > 0, "Y-axis not pointing Anterior (not RAS)"
        assert meta_affine[2, 2] > 0, "Z-axis not pointing Superior (not RAS)"

    def test_output_keys(
        self, sample_config: dict, synthetic_nifti_pair: dict[str, Path]
    ) -> None:
        """Output must contain 'image' and 'label' keys."""
        transforms = get_val_transforms(sample_config)
        data = {
            "image": str(synthetic_nifti_pair["image"]),
            "label": str(synthetic_nifti_pair["label"]),
        }
        result = transforms(data)

        assert "image" in result
        assert "label" in result


# ---------------------------------------------------------------------------
# Tests: structures / CompositeLabeld integration
# ---------------------------------------------------------------------------


class TestMultiStructureTransforms:
    """Tests for the CompositeLabeld-driven multi-structure pipeline path."""

    def test_train_transforms_composite_labels_with_structures(
        self, sample_config: dict, synthetic_multistructure_nifti: dict[str, Path]
    ) -> None:
        transforms = get_train_transforms(sample_config, structures=["canal", "cord"])
        data = {
            "image": str(synthetic_multistructure_nifti["image"]),
            "label_canal": str(synthetic_multistructure_nifti["label_canal"]),
            "label_cord": str(synthetic_multistructure_nifti["label_cord"]),
        }
        result = transforms(data)

        assert isinstance(result, list)
        for sample in result:
            assert "label" in sample
            assert "label_canal" not in sample
            assert "label_cord" not in sample
            unique_values = set(torch.unique(sample["label"]).tolist())
            assert unique_values.issubset({0.0, 1.0, 2.0})

    def test_val_transforms_composite_labels_with_structures(
        self, sample_config: dict, synthetic_multistructure_nifti: dict[str, Path]
    ) -> None:
        transforms = get_val_transforms(sample_config, structures=["canal", "cord"])
        data = {
            "image": str(synthetic_multistructure_nifti["image"]),
            "label_canal": str(synthetic_multistructure_nifti["label_canal"]),
            "label_cord": str(synthetic_multistructure_nifti["label_cord"]),
        }
        result = transforms(data)

        assert "label" in result
        assert "label_canal" not in result
        assert "label_cord" not in result
        unique_values = set(torch.unique(result["label"]).tolist())
        assert unique_values.issubset({0.0, 1.0, 2.0})

    def test_overlap_region_resolves_to_higher_priority_structure(
        self, sample_config: dict, synthetic_multistructure_nifti: dict[str, Path]
    ) -> None:
        """The fixture's cord/canal masks deliberately overlap at the volume
        center. DEFAULT_LABEL_PRIORITY ranks cord above canal, so the merged
        label at that overlap must carry cord's class id, while canal's own
        (non-overlapping) ring must keep canal's id."""
        transforms = get_val_transforms(sample_config, structures=["canal", "cord"])
        data = {
            "image": str(synthetic_multistructure_nifti["image"]),
            "label_canal": str(synthetic_multistructure_nifti["label_canal"]),
            "label_cord": str(synthetic_multistructure_nifti["label_cord"]),
        }
        result = transforms(data)
        label = result["label"]

        # canal=1, cord=2 per DEFAULT_LABEL_PRIORITY filtered order.
        assert label[0, 32, 96, 96] == 2  # inside both footprints -- cord wins
        assert label[0, 24, 96, 96] == 1  # canal-only ring


# ---------------------------------------------------------------------------
# Tests: get_post_transforms
# ---------------------------------------------------------------------------


class TestPostTransforms:
    """Tests for the post-processing transform pipeline."""

    def test_argmax_produces_integer_labels(self) -> None:
        """Applying pred post-transform (argmax) must yield integer-valued labels."""
        post = get_post_transforms(num_classes=2)
        pred_transform = post["pred"]

        # Simulate a softmax output: shape (num_classes, D, H, W)
        logits = torch.randn(2, 8, 16, 16)
        softmax_pred = torch.softmax(logits, dim=0)

        result = pred_transform(softmax_pred)

        # After argmax the channel dim collapses to 1 and values are class indices.
        # MONAI AsDiscrete returns float tensor with integer values (0.0, 1.0, ...).
        unique_values = torch.unique(result)
        for v in unique_values:
            assert v.item() == int(v.item()), (
                f"Expected integer value, got {v.item()}"
            )
            assert int(v.item()) in range(2), (
                f"Unexpected label value {v.item()}"
            )

    def test_post_transform_keys(self) -> None:
        """get_post_transforms must return dict with 'pred' and 'label' keys."""
        post = get_post_transforms(num_classes=2)
        assert "pred" in post
        assert "label" in post

    def test_label_to_onehot(self) -> None:
        """Label post-transform must convert integer labels to one-hot."""
        num_classes = 2
        post = get_post_transforms(num_classes=num_classes)
        label_transform = post["label"]

        # Simulate a single-channel integer label: shape (1, D, H, W)
        label = torch.zeros(1, 8, 16, 16, dtype=torch.long)
        label[0, 3:5, 7:9, 7:9] = 1

        result = label_transform(label)

        # One-hot should have shape (num_classes, D, H, W)
        assert result.shape[0] == num_classes, (
            f"Expected {num_classes} channels, got {result.shape[0]}"
        )


# ---------------------------------------------------------------------------
# Tests: helper and edge cases
# ---------------------------------------------------------------------------


class TestHelperAndEdgeCases:
    """Test the helper function and edge-case configurations."""

    def test_write_synthetic_nifti_creates_file(self, tmp_path: Path) -> None:
        """_write_synthetic_nifti must create a valid NIfTI file on disk."""
        path = tmp_path / "test.nii.gz"
        result_path = _write_synthetic_nifti(path, shape=(32, 64, 64))

        assert result_path.exists()
        img = nib.load(str(result_path))
        assert img.shape == (32, 64, 64)

    def test_write_synthetic_label_has_foreground(self, tmp_path: Path) -> None:
        """Label NIfTI must contain both foreground (1) and background (0)."""
        path = tmp_path / "label.nii.gz"
        _write_synthetic_nifti(path, shape=(32, 64, 64), is_label=True)

        data = np.asarray(nib.load(str(path)).dataobj)
        assert data.max() == 1
        assert data.min() == 0
        assert data.sum() > 0

    def test_minimal_config_uses_defaults(
        self, tmp_path: Path
    ) -> None:
        """An empty config dict should fall back to defaults without error."""
        img_path = _write_synthetic_nifti(
            tmp_path / "img.nii.gz", shape=(64, 192, 192)
        )
        lbl_path = _write_synthetic_nifti(
            tmp_path / "lbl.nii.gz", shape=(64, 192, 192), is_label=True
        )

        # Empty config -- all defaults should kick in
        transforms = get_train_transforms({})
        data = {"image": str(img_path), "label": str(lbl_path)}
        result = transforms(data)

        # Default patch_size is [48, 160, 160], default num_samples is 4
        assert isinstance(result, list)
        assert len(result) == 4
        for sample in result:
            assert list(sample["image"].shape[1:]) == [48, 160, 160]
