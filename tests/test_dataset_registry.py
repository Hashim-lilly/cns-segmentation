"""Tests for the dataset registry and its `create_datalist` integration."""

from pathlib import Path

import pytest

from cns_segmentation.data.dataset_registry import (
    DATASETS,
    DatasetSpec,
    get_dataset,
    label_path,
    list_datasets,
    merge_label_keys,
)
from cns_segmentation.data.adapters.alkafri_mendeley import MIN_LABEL_FILE_SIZE as ALKAFRI_MIN_FILE_SIZE
from cns_segmentation.data.spine_generic import create_datalist


def _make_subject(root: Path, subject_id: str, contrast: str, suffixes: list[str], size: int = 2000):
    anat_dir = root / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / f"{subject_id}_{contrast}.nii.gz").write_bytes(b"\0" * size)

    label_dir = root / "derivatives" / "labels" / subject_id / "anat"
    label_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (label_dir / f"{subject_id}_{contrast}_label-{suffix}.nii.gz").write_bytes(
            b"\0" * size
        )


class TestRegistry:
    def test_get_dataset_returns_spec(self):
        spec = get_dataset("spine_generic_cord")
        assert spec.name == "spine_generic_cord"
        assert spec.label_keys == {"cord": "SC_seg"}

    def test_get_dataset_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_dataset("does_not_exist")

    def test_list_datasets_matches_registry_keys(self):
        assert sorted(list_datasets()) == sorted(DATASETS.keys())

    def test_label_path_builds_expected_path(self):
        spec = get_dataset("spine_generic_canal")
        path = label_path(spec, "sub-amu01", "canal", contrast="T2w")
        assert path.name == "sub-amu01_T2w_label-canal_seg.nii.gz"

    def test_label_path_unknown_structure_raises(self):
        spec = get_dataset("spine_generic_cord")
        with pytest.raises(KeyError, match="not in dataset"):
            label_path(spec, "sub-amu01", "canal")


class TestCreateDatalistLegacyShape:
    def test_default_call_has_no_labels_key(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["SC_seg"])
        result = create_datalist(root_dir=tmp_path)
        assert len(result) == 1
        assert set(result[0].keys()) == {"image", "label", "subject", "site"}

    def test_label_keys_none_unaffected_by_other_labels_present(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["SC_seg", "canal_seg", "CSF_seg"])
        result = create_datalist(root_dir=tmp_path, label_keys=None)
        assert len(result) == 1
        assert set(result[0].keys()) == {"image", "label", "subject", "site"}


class TestCreateDatalistMultiStructure:
    def test_merges_multiple_labels(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["SC_seg", "canal_seg"])
        result = create_datalist(
            root_dir=tmp_path,
            label_keys={"cord": "SC_seg", "canal": "canal_seg"},
        )
        assert len(result) == 1
        entry = result[0]
        assert set(entry["labels"].keys()) == {"cord", "canal"}
        assert entry["label"] == entry["labels"]["cord"]

    def test_partial_labels_included_when_not_required(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["SC_seg"])
        result = create_datalist(
            root_dir=tmp_path,
            label_keys={"cord": "SC_seg", "canal": "canal_seg"},
            require_all_labels=False,
        )
        assert len(result) == 1
        assert set(result[0]["labels"].keys()) == {"cord"}

    def test_require_all_labels_drops_incomplete_subject(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["SC_seg", "canal_seg"])
        _make_subject(tmp_path, "sub-barcelona02", "T2w", ["SC_seg"])
        result = create_datalist(
            root_dir=tmp_path,
            label_keys={"cord": "SC_seg", "canal": "canal_seg"},
            require_all_labels=True,
        )
        subjects = {entry["subject"] for entry in result}
        assert subjects == {"sub-amu01"}

    def test_no_labels_key_when_no_structure_found(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", [])
        result = create_datalist(
            root_dir=tmp_path,
            label_keys={"cord": "SC_seg"},
        )
        assert result == []

    def test_missing_cord_omits_legacy_label_key(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["canal_seg"])
        result = create_datalist(
            root_dir=tmp_path,
            label_keys={"canal": "canal_seg"},
        )
        assert len(result) == 1
        assert "label" not in result[0]

    def test_git_annex_stub_excluded_from_found(self, tmp_path: Path):
        _make_subject(tmp_path, "sub-amu01", "T2w", ["SC_seg"], size=2000)
        # Overwrite canal_seg with a git-annex-stub-sized file
        label_dir = tmp_path / "derivatives" / "labels" / "sub-amu01" / "anat"
        (label_dir / "sub-amu01_T2w_label-canal_seg.nii.gz").write_bytes(b"\0" * 50)
        result = create_datalist(
            root_dir=tmp_path,
            label_keys={"cord": "SC_seg", "canal": "canal_seg"},
        )
        assert len(result) == 1
        assert set(result[0]["labels"].keys()) == {"cord"}


class TestSass2017ReferenceEntry:
    def test_registered_as_comparison_only_with_no_label_keys(self):
        spec = get_dataset("sass_2017_reference")
        assert spec.role == "comparison_only"
        assert spec.label_keys == {}
        assert spec.subject_count == 1
        assert spec.sites == 1

    def test_root_points_at_geometry_validation_source(self):
        spec = get_dataset("sass_2017_reference")
        assert spec.root.name == "geometry_validation.py"
        assert spec.root.is_file()

    def test_excluded_from_live_create_datalist_cross_check(self):
        comparison_only = {
            name for name, spec in DATASETS.items() if spec.role == "comparison_only"
        }
        assert "sass_2017_reference" in comparison_only


class TestMergeLabelKeys:
    def test_merges_two_specs(self):
        merged = merge_label_keys(get_dataset("spine_generic_cord"), get_dataset("spine_generic_canal"))
        assert merged == {"cord": "SC_seg", "canal": "canal_seg"}

    def test_merges_three_specs(self):
        merged = merge_label_keys(
            get_dataset("spine_generic_cord"),
            get_dataset("spine_generic_canal"),
            get_dataset("spine_generic_rootlets"),
        )
        assert merged == {"cord": "SC_seg", "canal": "canal_seg", "rootlets": "rootlets_dseg"}

    def test_no_specs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            merge_label_keys()

    def test_later_spec_wins_on_collision(self):
        first = DatasetSpec(
            name="a", root=Path("/tmp/a"), format="bids-derivatives",
            label_keys={"cord": "SC_seg_v1"}, spinal_region="cervical",
            subject_count=1, sites=1, license="CC0",
        )
        second = DatasetSpec(
            name="b", root=Path("/tmp/b"), format="bids-derivatives",
            label_keys={"cord": "SC_seg_v2"}, spinal_region="cervical",
            subject_count=1, sites=1, license="CC0",
        )
        assert merge_label_keys(first, second) == {"cord": "SC_seg_v2"}


class TestRealDataCounts:
    """Cross-checks registry metadata against the live spine-generic dataset.

    Skips cleanly when the dataset isn't fetched (git-annex stubs only),
    since this repo may run in environments without the data pulled.
    """

    @pytest.mark.parametrize(
        "dataset_name",
        [name for name, spec in DATASETS.items() if spec.role != "comparison_only"],
    )
    def test_registry_counts_match_create_datalist(self, dataset_name: str):
        # comparison_only entries (e.g. labels_softseg) live under a non-"labels"
        # derivatives_subdir that create_datalist() doesn't read from — registered
        # as a pointer per Decision 4, not yet wired into the create_datalist path.
        spec = get_dataset(dataset_name)
        probe = spec.root / "derivatives" / "labels"
        if not probe.is_dir():
            pytest.skip(f"spine-generic data not present at {spec.root}")

        # alkafri_mendeley_thecal_sac's real label files (249-334 bytes, single-slice
        # 2D masks) fall under create_datalist()'s default min_file_size=1000, which
        # is tuned for full-3D git-annex-stub detection — without this override the
        # test silently skips via the "no result" branch below instead of asserting.
        min_file_size = ALKAFRI_MIN_FILE_SIZE if dataset_name == "alkafri_mendeley_thecal_sac" else 1000
        result = create_datalist(
            root_dir=spec.root, label_keys=spec.label_keys, min_file_size=min_file_size
        )
        if not result:
            pytest.skip(f"spine-generic data at {spec.root} appears to be git-annex stubs only")

        subjects = {entry["subject"] for entry in result}
        sites = {entry["site"] for entry in result}

        assert len(subjects) == spec.subject_count, (
            f"{dataset_name}: create_datalist found {len(subjects)} subjects, "
            f"registry expects {spec.subject_count}"
        )
        assert len(sites) == spec.sites, (
            f"{dataset_name}: create_datalist found {len(sites)} sites, "
            f"registry expects {spec.sites}"
        )

    def test_softseg_cord_counts_match_on_disk(self):
        """Live-checks the comparison_only labels_softseg entry directly.

        create_datalist() doesn't read derivatives_subdir (see the skip note
        above), so this walks derivatives/labels_softseg/ itself instead.
        """
        from cns_segmentation.data.spine_generic import _SITE_PATTERN

        spec = get_dataset("spine_generic_softseg_cord")
        probe = spec.root / "derivatives" / spec.derivatives_subdir
        if not probe.is_dir():
            pytest.skip(f"spine-generic data not present at {spec.root}")

        suffix = spec.label_keys["cord_soft"]
        subjects, sites = set(), set()
        for entry in probe.iterdir():
            if not entry.is_dir() or not entry.name.startswith("sub-"):
                continue
            path = entry / "anat" / f"{entry.name}_T2w_label-{suffix}.nii.gz"
            if not path.is_file() or path.stat().st_size < 1000:
                continue
            match = _SITE_PATTERN.match(entry.name)
            if not match:
                continue
            subjects.add(entry.name)
            sites.add(match.group(1))

        if not subjects:
            pytest.skip(f"spine-generic data at {probe} appears to be git-annex stubs only")

        assert len(subjects) == spec.subject_count
        assert len(sites) == spec.sites
