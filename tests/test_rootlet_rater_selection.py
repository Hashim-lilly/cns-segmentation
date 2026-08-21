"""Tests for spine-generic rootlets rater-selection materialization."""

from pathlib import Path

from cns_segmentation.data.rootlet_rater_selection import resolve_rootlet_rater_labels


def _write_rater_variant(root: Path, subject_id: str, rater: str, content: bytes) -> Path:
    anat_dir = root / "derivatives" / "labels" / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    path = anat_dir / f"{subject_id}_T2w_desc-{rater}_label-rootlets_dseg.nii.gz"
    path.write_bytes(content)
    return path


def _canonical_path(root: Path, subject_id: str) -> Path:
    return root / "derivatives" / "labels" / subject_id / "anat" / f"{subject_id}_T2w_label-rootlets_dseg.nii.gz"


class TestResolveRootletRaterLabels:
    def test_prefers_staple_over_rater1(self, tmp_path: Path):
        _write_rater_variant(tmp_path, "sub-amu02", "rater1", b"rater1-content")
        _write_rater_variant(tmp_path, "sub-amu02", "staple", b"staple-content")

        resolved = resolve_rootlet_rater_labels(tmp_path, subjects=["sub-amu02"])

        assert resolved == ["sub-amu02"]
        assert _canonical_path(tmp_path, "sub-amu02").read_bytes() == b"staple-content"

    def test_falls_back_to_rater1_when_no_staple(self, tmp_path: Path):
        _write_rater_variant(tmp_path, "sub-brnoUhb03", "rater1", b"rater1-content")

        resolved = resolve_rootlet_rater_labels(tmp_path, subjects=["sub-brnoUhb03"])

        assert resolved == ["sub-brnoUhb03"]
        assert _canonical_path(tmp_path, "sub-brnoUhb03").read_bytes() == b"rater1-content"

    def test_skips_subject_with_no_known_rater_variant(self, tmp_path: Path):
        _write_rater_variant(tmp_path, "sub-barcelona01", "rater2", b"rater2-content")

        resolved = resolve_rootlet_rater_labels(tmp_path, subjects=["sub-barcelona01"])

        assert resolved == []
        assert not _canonical_path(tmp_path, "sub-barcelona01").exists()

    def test_idempotent_skip_when_canonical_already_exists(self, tmp_path: Path):
        _write_rater_variant(tmp_path, "sub-amu02", "staple", b"staple-content")
        canonical = _canonical_path(tmp_path, "sub-amu02")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"pre-existing-content")

        resolved = resolve_rootlet_rater_labels(tmp_path, subjects=["sub-amu02"])

        assert resolved == ["sub-amu02"]
        assert canonical.read_bytes() == b"pre-existing-content"

    def test_force_overwrites_existing_canonical(self, tmp_path: Path):
        _write_rater_variant(tmp_path, "sub-amu02", "staple", b"staple-content")
        canonical = _canonical_path(tmp_path, "sub-amu02")
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"stale-content")

        resolved = resolve_rootlet_rater_labels(tmp_path, subjects=["sub-amu02"], force=True)

        assert resolved == ["sub-amu02"]
        assert canonical.read_bytes() == b"staple-content"

    def test_defaults_to_known_deferred_subjects(self, tmp_path: Path):
        _write_rater_variant(tmp_path, "sub-amu02", "staple", b"a")
        _write_rater_variant(tmp_path, "sub-barcelona01", "staple", b"b")
        _write_rater_variant(tmp_path, "sub-brnoUhb03", "staple", b"c")

        resolved = resolve_rootlet_rater_labels(tmp_path)

        assert set(resolved) == {"sub-amu02", "sub-barcelona01", "sub-brnoUhb03"}

    def test_copies_json_sidecar_when_present(self, tmp_path: Path):
        source = _write_rater_variant(tmp_path, "sub-amu02", "staple", b"staple-content")
        source.with_suffix("").with_suffix(".json").write_bytes(b'{"rater": "staple"}')

        resolve_rootlet_rater_labels(tmp_path, subjects=["sub-amu02"])

        canonical_json = _canonical_path(tmp_path, "sub-amu02").with_suffix("").with_suffix(".json")
        assert canonical_json.read_bytes() == b'{"rater": "staple"}'
