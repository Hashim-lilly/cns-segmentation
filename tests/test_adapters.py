"""Tests for the shared dataset-adapter contract in `data/adapters/base.py`."""

from pathlib import Path

from cns_segmentation.data.adapters.base import (
    is_prepared,
    subject_dirname,
    write_bids_subject,
)


class TestSubjectDirname:
    def test_pads_index_to_four_digits(self):
        assert subject_dirname("spider", 1) == "sub-spider0001"
        assert subject_dirname("spider", 42) == "sub-spider0042"

    def test_different_tags_cannot_collide(self):
        assert subject_dirname("spider", 1) != subject_dirname("alkafri", 1)


class TestWriteBidsSubject:
    def test_writes_image_and_labels_in_spine_generic_shape(self, tmp_path: Path):
        write_bids_subject(
            tmp_path,
            "sub-spider0001",
            image_bytes=b"\0" * 2000,
            labels={"canal_seg": b"\1" * 2000},
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
        assert image_path.read_bytes() == b"\0" * 2000
        assert label_path.read_bytes() == b"\1" * 2000

    def test_no_labels_skips_derivatives_dir(self, tmp_path: Path):
        write_bids_subject(
            tmp_path, "sub-alkafri0001", image_bytes=b"\0" * 2000, labels={}
        )
        assert (tmp_path / "sub-alkafri0001" / "anat").is_dir()
        assert not (tmp_path / "derivatives").exists()

    def test_multiple_labels_all_written(self, tmp_path: Path):
        write_bids_subject(
            tmp_path,
            "sub-spider0001",
            image_bytes=b"\0" * 2000,
            labels={"canal_seg": b"\1" * 2000, "SC_seg": b"\2" * 2000},
        )
        label_dir = tmp_path / "derivatives" / "labels" / "sub-spider0001" / "anat"
        assert {p.name for p in label_dir.iterdir()} == {
            "sub-spider0001_T2w_label-canal_seg.nii.gz",
            "sub-spider0001_T2w_label-SC_seg.nii.gz",
        }


class TestIsPrepared:
    def test_missing_derivatives_dir_is_not_prepared(self, tmp_path: Path):
        assert is_prepared(tmp_path, expected_subject_count=1) is False

    def test_below_expected_count_is_not_prepared(self, tmp_path: Path):
        write_bids_subject(
            tmp_path, "sub-spider0001", image_bytes=b"\0" * 2000, labels={"canal_seg": b"\1" * 2000}
        )
        assert is_prepared(tmp_path, expected_subject_count=2) is False

    def test_meeting_expected_count_is_prepared(self, tmp_path: Path):
        for i in (1, 2):
            write_bids_subject(
                tmp_path,
                subject_dirname("spider", i),
                image_bytes=b"\0" * 2000,
                labels={"canal_seg": b"\1" * 2000},
            )
        assert is_prepared(tmp_path, expected_subject_count=2) is True

    def test_stub_sized_labels_do_not_count(self, tmp_path: Path):
        write_bids_subject(
            tmp_path,
            "sub-spider0001",
            image_bytes=b"\0" * 2000,
            labels={"canal_seg": b"\1" * 50},  # below min_file_size
        )
        assert is_prepared(tmp_path, expected_subject_count=1, min_file_size=1000) is False
