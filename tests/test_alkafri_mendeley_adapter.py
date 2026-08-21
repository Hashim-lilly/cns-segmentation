"""Tests for the Al-Kafri/Mendeley thecal-sac adapter (`data/adapters/alkafri_mendeley.py`).

Network-free by design: `materialize_patient_disk` (and the pure filename-
correspondence check `find_verified_pairs`) are fed small synthetic PNG
bytes built locally with SimpleITK, mirroring `tests/test_spider_adapter.py`
and `tests/test_adapters.py`'s synthetic-fixture style. Nothing here
downloads from Mendeley or reads the real materialized dataset.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from cns_segmentation.data.adapters.alkafri_mendeley import (
    LICENSE,
    SITES,
    SPINAL_REGION,
    THECAL_SAC_LABEL_VALUE,
    _label_member,
    _slice_png_bytes_to_nifti_bytes,
    _t2_member,
    build_spec,
    find_verified_pairs,
    materialize_patient_disk,
)


def _png_bytes(array: np.ndarray) -> bytes:
    """Build raw 8-bit grayscale PNG bytes for a small synthetic 2D array (no network)."""
    image = sitk.GetImageFromArray(array.astype("uint8"))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slice.png"
        sitk.WriteImage(image, str(path))
        return path.read_bytes()


def _read_nifti_array(nifti_bytes: bytes) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vol.nii.gz"
        path.write_bytes(nifti_bytes)
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


class TestLabelAndT2MemberPaths:
    def test_label_member_path_shape(self):
        assert _label_member("0001", "3") == (
            "05_Final_Ground_Truth_Data/Label_Images/L1_0001_D3.png"
        )

    def test_t2_member_path_shape(self):
        assert _t2_member("0001", "3") == (
            "04_Intermediary_Ground_Truth_Data/T2_Output/T2_0001_D3.png"
        )


class TestFindVerifiedPairs:
    def test_keeps_pair_with_matching_t2_slice(self):
        names = [
            "05_Final_Ground_Truth_Data/Label_Images/L1_0001_D3.png",
            "04_Intermediary_Ground_Truth_Data/T2_Output/T2_0001_D3.png",
        ]
        assert find_verified_pairs(names) == [("0001", "3")]

    def test_skips_label_with_no_matching_t2_slice(self):
        names = [
            "05_Final_Ground_Truth_Data/Label_Images/L1_0001_D3.png",
            "05_Final_Ground_Truth_Data/Label_Images/L1_0002_D4.png",
            "04_Intermediary_Ground_Truth_Data/T2_Output/T2_0001_D3.png",
            # 0002/D4 has no matching T2_Output entry - must be skipped, not guessed.
        ]
        assert find_verified_pairs(names) == [("0001", "3")]

    def test_ignores_unrelated_zip_members(self):
        names = [
            "05_Final_Ground_Truth_Data/Label_Images/L1_0001_D3.png",
            "04_Intermediary_Ground_Truth_Data/T2_Output/T2_0001_D3.png",
            "05_Final_Ground_Truth_Data/Composite_Images/C1_0001_D3.png",
            "04_Intermediary_Ground_Truth_Data/T1_Output/T1_0001_D3.png",
            "09_Source_Code/03_Extract_Best_Slices/Slices Numbers.csv",
        ]
        assert find_verified_pairs(names) == [("0001", "3")]

    def test_sorted_by_integer_patient_then_disk(self):
        names = []
        for patient, disk in [("0010", "5"), ("0002", "3"), ("0002", "4"), ("0100", "3")]:
            names.append(f"05_Final_Ground_Truth_Data/Label_Images/L1_{patient}_D{disk}.png")
            names.append(f"04_Intermediary_Ground_Truth_Data/T2_Output/T2_{patient}_D{disk}.png")
        assert find_verified_pairs(names) == [
            ("0002", "3"),
            ("0002", "4"),
            ("0010", "5"),
            ("0100", "3"),
        ]

    def test_no_duplicate_pairs(self):
        names = [
            "05_Final_Ground_Truth_Data/Label_Images/L1_0001_D3.png",
            "04_Intermediary_Ground_Truth_Data/T2_Output/T2_0001_D3.png",
        ]
        assert len(find_verified_pairs(names)) == 1

    def test_empty_listing_returns_empty(self):
        assert find_verified_pairs([]) == []


class TestSlicePngBytesToNiftiBytes:
    def test_round_trips_grayscale_values(self):
        array = np.array([[50, 100, 150], [200, 250, 0]], dtype="uint8")
        nifti_bytes = _slice_png_bytes_to_nifti_bytes(_png_bytes(array))
        result = _read_nifti_array(nifti_bytes)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result[0], array)

    def test_keep_value_binarizes_to_thecal_sac_mask(self):
        array = np.array([[50, 100, 150], [150, 250, 150]], dtype="uint8")
        nifti_bytes = _slice_png_bytes_to_nifti_bytes(
            _png_bytes(array), keep_value=THECAL_SAC_LABEL_VALUE
        )
        result = _read_nifti_array(nifti_bytes)
        expected = (array == 150).astype("uint8")
        np.testing.assert_array_equal(result[0], expected)
        assert set(np.unique(result)) <= {0, 1}

    def test_non_2d_array_raises(self):
        # A 3-channel PNG would decode as (H, W, 3) — not a supported input
        # for this dataset's confirmed single-channel 8-bit grayscale PNGs.
        array = np.zeros((2, 2, 3), dtype="uint8")
        image = sitk.GetImageFromArray(array, isVector=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            sitk.WriteImage(image, str(path))
            rgb_png_bytes = path.read_bytes()
        with pytest.raises(ValueError):
            _slice_png_bytes_to_nifti_bytes(rgb_png_bytes)


class TestMaterializePatientDisk:
    def test_writes_image_and_thecal_sac_label(self, tmp_path: Path):
        t2_array = np.full((4, 4), 128, dtype="uint8")
        label_array = np.array(
            [
                [50, 50, 100, 100],
                [150, 150, 150, 100],
                [150, 150, 150, 200],
                [250, 250, 250, 250],
            ],
            dtype="uint8",
        )

        materialize_patient_disk(
            _png_bytes(t2_array), _png_bytes(label_array), tmp_path, "sub-alkafri0001"
        )

        image_path = tmp_path / "sub-alkafri0001" / "anat" / "sub-alkafri0001_T2w.nii.gz"
        label_path = (
            tmp_path
            / "derivatives"
            / "labels"
            / "sub-alkafri0001"
            / "anat"
            / "sub-alkafri0001_T2w_label-thecal_sac_seg.nii.gz"
        )
        assert image_path.is_file()
        assert label_path.is_file()

        image_result = _read_nifti_array(image_path.read_bytes())
        np.testing.assert_array_equal(image_result[0], t2_array)

        label_result = _read_nifti_array(label_path.read_bytes())
        expected_mask = (label_array == THECAL_SAC_LABEL_VALUE).astype("uint8")
        np.testing.assert_array_equal(label_result[0], expected_mask)
        assert label_result.sum() == 6  # six pixels equal 150 above


class TestBuildSpec:
    def test_spec_fields(self):
        spec = build_spec(subject_count=1545)
        assert spec.name == "alkafri_mendeley_thecal_sac"
        assert spec.format == "bids-derivatives"
        assert spec.label_keys == {"thecal_sac": "thecal_sac_seg"}
        assert spec.spinal_region == SPINAL_REGION == "lumbar"
        assert spec.subject_count == 1545
        assert spec.sites == SITES == 1
        assert spec.license == LICENSE == "CC BY 4.0"
        assert spec.role == "validation"
        assert spec.derivatives_subdir == "labels"

    def test_subject_count_reflects_argument(self):
        assert build_spec(subject_count=0).subject_count == 0
        assert build_spec(subject_count=1545).subject_count == 1545
