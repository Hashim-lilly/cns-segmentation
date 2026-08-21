"""Tests for the SPIDER lumbar-spine adapter (`data/adapters/spider.py`).

Network-free by design: `materialize_series` (and the pure filename-filter
`select_t2_series`) are fed small synthetic `.mha` bytes built locally with
SimpleITK, mirroring `tests/test_adapters.py`'s synthetic-fixture style for
`write_bids_subject`. Nothing here downloads from Zenodo or reads the real
materialized dataset.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from cns_segmentation.data.adapters.spider import (
    LICENSE,
    SITES_PER_PAPER,
    SPINAL_REGION,
    _is_t2_series,
    _mha_bytes_to_nifti_bytes,
    build_spec,
    materialize_series,
    select_t2_series,
)


def _mha_bytes(array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> bytes:
    """Build raw `.mha` bytes for a small synthetic volume (no network)."""
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vol.mha"
        sitk.WriteImage(image, str(path))
        return path.read_bytes()


def _read_nifti_array(nifti_bytes: bytes) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vol.nii.gz"
        path.write_bytes(nifti_bytes)
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


class TestIsT2Series:
    def test_true_t2_stem_is_selected(self):
        assert _is_t2_series("1_t2") is True

    def test_t1_stem_is_excluded(self):
        assert _is_t2_series("1_t1") is False

    def test_t2_space_stem_is_excluded(self):
        assert _is_t2_series("107_t2_SPACE") is False

    def test_case_insensitive_suffix_match(self):
        assert _is_t2_series("42_T2") is True


class TestSelectT2Series:
    def test_filters_mixed_listing_to_true_t2_only(self):
        names = [
            "masks/1_t1.mha",
            "masks/1_t2.mha",
            "masks/107_t2_SPACE.mha",
            "masks/108_t2.mha",
            "masks/109_t1.mha",
        ]
        assert select_t2_series(names) == ["108_t2", "1_t2"]

    def test_ignores_non_mha_members(self):
        names = ["masks/1_t2.mha", "overview.csv", "masks/README.txt"]
        assert select_t2_series(names) == ["1_t2"]

    def test_deduplicates_across_directories(self):
        names = ["masks/1_t2.mha", "images/1_t2.mha"]
        assert select_t2_series(names) == ["1_t2"]

    def test_empty_listing_returns_empty(self):
        assert select_t2_series([]) == []

    def test_result_is_sorted(self):
        names = ["masks/9_t2.mha", "masks/10_t2.mha", "masks/2_t2.mha"]
        # Lexicographic, not numeric — "10_t2" sorts before "2_t2".
        assert select_t2_series(names) == ["10_t2", "2_t2", "9_t2"]


class TestMhaBytesToNiftiBytes:
    def test_plain_conversion_preserves_values(self):
        array = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
        nifti_bytes = _mha_bytes_to_nifti_bytes(_mha_bytes(array), canal_only=False)
        assert np.array_equal(_read_nifti_array(nifti_bytes), array)

    def test_canal_only_thresholds_to_binary_mask(self):
        array = np.zeros((2, 3, 4), dtype=np.int16)
        array[0, 0, 0] = 100  # canal
        array[0, 1, 1] = 3  # vertebra, must be dropped
        array[1, 2, 2] = 205  # disc, must be dropped
        array[1, 0, 3] = 100  # canal
        nifti_bytes = _mha_bytes_to_nifti_bytes(_mha_bytes(array), canal_only=True)
        result = _read_nifti_array(nifti_bytes)
        expected = (array == 100).astype(np.uint8)
        assert result.dtype == np.uint8
        assert np.array_equal(result, expected)

    def test_canal_only_all_background_yields_all_zero_mask(self):
        array = np.full((2, 2, 2), 7, dtype=np.int16)  # vertebra label only
        nifti_bytes = _mha_bytes_to_nifti_bytes(_mha_bytes(array), canal_only=True)
        result = _read_nifti_array(nifti_bytes)
        assert result.max() == 0

    def test_geometry_is_preserved(self):
        array = np.ones((2, 2, 2), dtype=np.int16)
        nifti_bytes = _mha_bytes_to_nifti_bytes(_mha_bytes(array, spacing=(0.5, 0.6, 3.0)))
        with_tmp = _write_and_read_image(nifti_bytes)
        assert with_tmp.GetSpacing() == pytest.approx((0.5, 0.6, 3.0))


def _write_and_read_image(nifti_bytes: bytes) -> sitk.Image:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vol.nii.gz"
        path.write_bytes(nifti_bytes)
        return sitk.ReadImage(str(path))


class TestMaterializeSeries:
    def test_writes_image_and_canal_label_in_bids_shape(self, tmp_path: Path):
        image_array = (np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)) * 10
        mask_array = np.zeros((2, 3, 4), dtype=np.int16)
        mask_array[0, 0, 0] = 100
        mask_array[1, 1, 1] = 5  # vertebra, dropped

        materialize_series(
            _mha_bytes(image_array),
            _mha_bytes(mask_array),
            tmp_path,
            "sub-spider0001",
        )

        image_path = tmp_path / "sub-spider0001" / "anat" / "sub-spider0001_T2w.nii.gz"
        label_path = (
            tmp_path
            / "derivatives"
            / "labels"
            / "sub-spider0001"
            / "anat"
            / "sub-spider0001_T2w_label-canal_seg.nii.gz"
        )
        assert image_path.is_file()
        assert label_path.is_file()
        assert np.array_equal(_read_nifti_array(image_path.read_bytes()), image_array)
        assert np.array_equal(
            _read_nifti_array(label_path.read_bytes()), (mask_array == 100).astype(np.uint8)
        )

    def test_multiple_subjects_do_not_collide(self, tmp_path: Path):
        array = np.zeros((2, 2, 2), dtype=np.int16)
        for i in (1, 2):
            materialize_series(_mha_bytes(array), _mha_bytes(array), tmp_path, f"sub-spider000{i}")
        labels_dir = tmp_path / "derivatives" / "labels"
        assert {p.name for p in labels_dir.iterdir()} == {"sub-spider0001", "sub-spider0002"}


class TestBuildSpec:
    def test_reports_expected_static_fields(self):
        spec = build_spec(subject_count=210)
        assert spec.name == "spider_canal"
        assert spec.format == "bids-derivatives"
        assert spec.label_keys == {"canal": "canal_seg"}
        assert spec.spinal_region == SPINAL_REGION == "lumbar"
        assert spec.license == LICENSE == "CC-BY-4.0"
        assert spec.role == "validation"
        assert spec.sites == SITES_PER_PAPER == 4

    def test_subject_count_is_parameterized(self):
        assert build_spec(subject_count=210).subject_count == 210
        assert build_spec(subject_count=0).subject_count == 0

    def test_root_points_at_data_spider(self):
        spec = build_spec(subject_count=1)
        assert spec.root.parts[-2:] == ("data", "spider")
