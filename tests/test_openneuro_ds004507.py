"""Tests for the OpenNeuro ds004507 adapter's pure conversion/materialization logic.

No network access and no real downloaded data required — mirrors
`write_bids_subject`'s own synthetic-fixture style in `tests/test_adapters.py`.
`prepare()`'s actual network calls (`_download_bytes`/`_remote_exists`) are
intentionally not exercised here.
"""

from pathlib import Path

from cns_segmentation.data.adapters.openneuro_ds004507 import (
    SESSION_PREFERENCE,
    SOURCE_SUBJECTS,
    build_subject_mapping,
    materialize_subject,
    select_session,
)


class TestSelectSession:
    def test_prefers_head_normal_when_available(self):
        assert select_session({"ses-headNormal", "ses-headUp"}) == "ses-headNormal"

    def test_falls_back_to_head_up_when_normal_missing(self):
        assert select_session({"ses-headUp", "ses-headDown"}) == "ses-headUp"

    def test_falls_back_to_head_down_as_last_resort(self):
        assert select_session({"ses-headDown"}) == "ses-headDown"

    def test_returns_none_when_no_known_session_available(self):
        assert select_session(set()) is None
        assert select_session({"ses-unrelated"}) is None

    def test_preference_order_matches_documented_constant(self):
        assert SESSION_PREFERENCE == ["ses-headNormal", "ses-headUp", "ses-headDown"]


class TestBuildSubjectMapping:
    def test_maps_in_given_order_to_sequential_headpos_ids(self):
        mapping = build_subject_mapping(["sub-002", "sub-003", "sub-011"])
        assert mapping == {
            "sub-002": "sub-headpos0001",
            "sub-003": "sub-headpos0002",
            "sub-011": "sub-headpos0003",
        }

    def test_defaults_to_known_source_subjects(self):
        mapping = build_subject_mapping()
        assert set(mapping.keys()) == set(SOURCE_SUBJECTS)
        assert mapping["sub-002"] == "sub-headpos0001"
        assert mapping["sub-011"] == "sub-headpos0007"

    def test_new_ids_are_digits_after_letters_only(self):
        # Confirms the whole point of the remap: original digits-only IDs
        # (e.g. "sub-002") would fail spine_generic's letters-then-digits
        # site-tag pattern; the remapped ID must not.
        mapping = build_subject_mapping(["sub-002"])
        new_id = mapping["sub-002"]
        tag = new_id.removeprefix("sub-")
        letters = "".join(c for c in tag if c.isalpha())
        digits = "".join(c for c in tag if c.isdigit())
        assert letters + digits == tag
        assert letters == "headpos"


class TestMaterializeSubject:
    def test_writes_image_and_rootlets_label(self, tmp_path: Path):
        materialize_subject(
            tmp_path,
            "sub-headpos0001",
            image_bytes=b"\0" * 2000,
            label_bytes=b"\1" * 2000,
        )

        image_path = tmp_path / "sub-headpos0001" / "anat" / "sub-headpos0001_T2w.nii.gz"
        label_path = (
            tmp_path
            / "derivatives"
            / "labels"
            / "sub-headpos0001"
            / "anat"
            / "sub-headpos0001_T2w_label-rootlets_dseg.nii.gz"
        )
        assert image_path.read_bytes() == b"\0" * 2000
        assert label_path.read_bytes() == b"\1" * 2000

    def test_multiple_subjects_do_not_collide(self, tmp_path: Path):
        materialize_subject(tmp_path, "sub-headpos0001", b"\0" * 10, b"\1" * 10)
        materialize_subject(tmp_path, "sub-headpos0002", b"\2" * 10, b"\3" * 10)

        labels_dir = tmp_path / "derivatives" / "labels"
        assert {p.name for p in labels_dir.iterdir()} == {"sub-headpos0001", "sub-headpos0002"}
