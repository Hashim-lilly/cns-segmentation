"""Tests for the shared cns_segmentation.inference module.

Argument-validation branches (bad split, missing checkpoint, empty datalist,
unknown structure) are tested directly with no model involved — they all
raise before any heavy model call happens. The "happy path" tests for
`run_predict`/`run_evaluate_external` use real tiny NIfTI files and a real
(but tiny, `init_filters=4`) SegResNet checkpoint, mirroring
`tests/test_trainer.py`'s `_make_real_subject` convention, rather than
mocking the model — this exercises the real transform/inference/metrics
chain end to end while staying fast. `run_evaluate_external`'s dataset comes
from a hardcoded registry entry whose `root` is a fixed absolute path, so its
happy-path and empty-datalist tests monkeypatch `get_dataset` to a synthetic
spec pointed at a tmp_path root — this keeps the test hermetic (no real
downloaded data required), matching `test_openneuro_ds004507.py`'s stated
convention.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn

import cns_segmentation.inference as inference_module
from cns_segmentation.data.dataset_registry import DatasetSpec
from cns_segmentation.data.dataset_registry import get_dataset as real_get_dataset
from cns_segmentation.inference import (
    InferenceResult,
    build_inferer,
    checkpoint_class_id,
    flatten_result,
    list_heldout_subjects,
    load_checkpoint_state_dict,
    load_model,
    load_yaml,
    predict_volume,
    resolve_structures,
    run_evaluate_external,
    run_predict,
)
from cns_segmentation.models.segresnet import create_segresnet

# Large enough that a gzip-compressed random-noise label mask lands safely
# above the 1000-byte git-annex-stub threshold `create_datalist()` filters on
# (a small contiguous blob compresses to ~150 bytes and gets mistaken for a
# stub) — see CLAUDE.md rule 2.
_SHAPE = (16, 32, 32)


def _write_real_subject(root: Path, subject_id: str, label_suffixes: dict[str, str]) -> None:
    """Write a real (non-stub) tiny T2w image + per-structure labels for `subject_id`.

    `label_suffixes` maps structure name -> BIDS derivative suffix, e.g.
    {"cord": "SC_seg"}.
    """
    anat_dir = root / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)

    rng = np.random.default_rng(0)
    image = rng.standard_normal(_SHAPE).astype(np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(anat_dir / f"{subject_id}_T2w.nii.gz"))

    label_dir = root / "derivatives" / "labels" / subject_id / "anat"
    label_dir.mkdir(parents=True, exist_ok=True)
    for suffix in label_suffixes.values():
        mask = (rng.random(_SHAPE) > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, affine), str(label_dir / f"{subject_id}_T2w_label-{suffix}.nii.gz"))


def _tiny_model_config(out_channels: int) -> dict:
    return {
        "spatial_dims": 3, "in_channels": 1, "out_channels": out_channels,
        "init_filters": 8, "blocks_down": [1, 1, 1, 1], "blocks_up": [1, 1, 1],
        "dropout_prob": 0.0,
    }


def _save_tiny_checkpoint(path: Path, model_cfg: dict) -> None:
    model = create_segresnet(model_cfg)
    torch.save({"model_state_dict": model.state_dict()}, path)


def _fake_dataset_spec(root: Path, label_keys: dict[str, str]) -> DatasetSpec:
    return DatasetSpec(
        name="fake_dataset", root=root, format="bids-derivatives", label_keys=label_keys,
        spinal_region="cervical", subject_count=1, sites=1, license="CC0", role="validation",
    )


class TestLoadYaml:
    def test_loads_valid_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("a: 1\nb:\n  c: 2\n")
        assert load_yaml(path) == {"a": 1, "b": {"c": 2}}

    def test_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_yaml(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("a: [unterminated\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_yaml(path)


class TestLoadCheckpointStateDict:
    def test_unwraps_model_state_dict_key(self, tmp_path):
        model = create_segresnet(_tiny_model_config(out_channels=2))
        path = tmp_path / "ckpt.pth"
        torch.save({"model_state_dict": model.state_dict(), "epoch": 5}, path)
        state_dict = load_checkpoint_state_dict(path, torch.device("cpu"))
        assert set(state_dict.keys()) == set(model.state_dict().keys())

    def test_returns_raw_state_dict_when_unwrapped(self, tmp_path):
        model = create_segresnet(_tiny_model_config(out_channels=2))
        path = tmp_path / "ckpt.pth"
        torch.save(model.state_dict(), path)
        state_dict = load_checkpoint_state_dict(path, torch.device("cpu"))
        assert set(state_dict.keys()) == set(model.state_dict().keys())


class TestLoadModel:
    def test_round_trips_weights_and_sets_eval_mode(self, tmp_path):
        cfg = _tiny_model_config(out_channels=3)
        path = tmp_path / "ckpt.pth"
        _save_tiny_checkpoint(path, cfg)

        model = load_model(cfg, path, torch.device("cpu"))
        assert model.training is False

        reference = create_segresnet(cfg)
        reference.load_state_dict(load_checkpoint_state_dict(path, torch.device("cpu")))
        for p1, p2 in zip(model.parameters(), reference.parameters()):
            assert torch.equal(p1, p2)


class TestBuildInferer:
    def test_builds_sliding_window_inferer_with_config_values(self):
        inferer = build_inferer({"roi_size": [8, 16, 16], "overlap": 0.5, "mode": "gaussian"})
        assert inferer.roi_size == [8, 16, 16]
        assert inferer.overlap == 0.5
        assert inferer.mode.value == "gaussian"

    def test_defaults_mode_to_gaussian_when_omitted(self):
        inferer = build_inferer({"roi_size": [8, 16, 16], "overlap": 0.25})
        assert inferer.mode.value == "gaussian"


class TestResolveStructures:
    def test_legacy_cord_only_returns_none(self):
        assert resolve_structures("spine_generic_cord") == (None, None)
        assert resolve_structures(["spine_generic_cord"]) == (None, None)

    def test_multi_structure_orders_by_label_priority(self):
        structures, class_map = resolve_structures(["spine_generic_cord", "spine_generic_rootlets"])
        assert structures == ["cord", "rootlets"]
        assert class_map == {"cord": 1, "rootlets": 2}


class TestCheckpointClassId:
    def test_returns_one_indexed_class_id(self):
        train_config = {"data": {"dataset": ["spine_generic_cord", "spine_generic_canal"]}}
        assert checkpoint_class_id(train_config, "canal") == 1
        assert checkpoint_class_id(train_config, "cord") == 2

    def test_unknown_structure_raises_value_error(self):
        train_config = {"data": {"dataset": ["spine_generic_cord", "spine_generic_canal"]}}
        with pytest.raises(ValueError, match="not a class this checkpoint was trained on"):
            checkpoint_class_id(train_config, "rootlets")

    def test_legacy_cord_only_checkpoint_raises_for_any_structure(self):
        train_config = {"data": {"dataset": "spine_generic_cord"}}
        with pytest.raises(ValueError, match="not a class this checkpoint was trained on"):
            checkpoint_class_id(train_config, "cord")


class TestKeepLargestComponent:
    def test_single_component_unchanged(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        out = inference_module._keep_largest_component(mask)
        np.testing.assert_array_equal(out, mask)

    def test_keeps_only_largest_of_multiple_components(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:4, 0:4, 0:4] = 1  # large component (64 voxels)
        mask[8:9, 8:9, 8:9] = 1  # tiny disconnected component (1 voxel)
        out = inference_module._keep_largest_component(mask)
        assert out[0, 0, 0] == 1
        assert out[8, 8, 8] == 0
        assert out.sum() == 64

    def test_empty_mask_returned_unchanged(self):
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        out = inference_module._keep_largest_component(mask)
        np.testing.assert_array_equal(out, mask)


class TestFlattenResult:
    def test_binary_result_returned_as_is(self):
        result = {"subject": "sub-amu01", "site": "amu", "dice": 0.9}
        assert flatten_result(result) == result

    def test_multi_structure_result_flattened_with_prefixed_keys(self):
        result = {
            "subject": "sub-amu01", "site": "amu",
            "canal": {"dice": 0.9, "hausdorff95_mm": 1.2},
            "cord": {"dice": 0.95, "hausdorff95_mm": 0.5},
        }
        flat = flatten_result(result)
        assert flat == {
            "subject": "sub-amu01", "site": "amu",
            "canal_dice": 0.9, "canal_hausdorff95_mm": 1.2,
            "cord_dice": 0.95, "cord_hausdorff95_mm": 0.5,
        }


class TestListHeldoutSubjects:
    """Builds one synthetic BIDS root shared by all three tests below.

    sub-amu01 (site=amu): has both cord and canal labels.
    sub-amu02 (site=amu): has only a cord label.
    sub-barcelona01 (site=barcelona): has both cord and canal labels, but is
    excluded from every test's val_sites=["amu"] filter.
    """

    @pytest.fixture()
    def bids_root(self, tmp_path):
        root = tmp_path / "data"
        _write_real_subject(root, "sub-amu01", {"cord": "SC_seg", "canal": "canal_seg"})
        _write_real_subject(root, "sub-amu02", {"cord": "SC_seg"})
        _write_real_subject(root, "sub-barcelona01", {"cord": "SC_seg", "canal": "canal_seg"})
        return tmp_path

    def test_legacy_cord_only_ignores_missing_canal_label(self, bids_root):
        train_config = {"data": {"root_dir": "data", "val_sites": ["amu"], "dataset": "spine_generic_cord"}}
        result = list_heldout_subjects(train_config, "cord", bids_root)
        assert {item["subject"] for item in result} == {"sub-amu01", "sub-amu02"}

    def test_multi_structure_requires_all_labels_present(self, bids_root):
        train_config = {
            "data": {"root_dir": "data", "val_sites": ["amu"], "dataset": ["spine_generic_cord", "spine_generic_canal"]}
        }
        result = list_heldout_subjects(train_config, "canal", bids_root)
        assert {item["subject"] for item in result} == {"sub-amu01"}
        assert Path(result[0]["label"]).name == "sub-amu01_T2w_label-canal_seg.nii.gz"

    def test_val_sites_filters_out_other_sites(self, bids_root):
        train_config = {
            "data": {"root_dir": "data", "val_sites": ["amu"], "dataset": ["spine_generic_cord", "spine_generic_canal"]}
        }
        result = list_heldout_subjects(train_config, "canal", bids_root)
        assert "sub-barcelona01" not in {item["subject"] for item in result}


class TestPredictVolume:
    class _ToyModel(nn.Module):
        def __init__(self, out_channels: int):
            super().__init__()
            self.conv = nn.Conv3d(1, out_channels, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    @staticmethod
    def _fake_inferer(inputs, net):
        return net(inputs)

    def test_basic_prediction_shape(self, synthetic_nifti_pair):
        model = self._ToyModel(out_channels=3)
        model.eval()
        result = predict_volume(
            image_path=synthetic_nifti_pair["image"],
            model=model, inferer=self._fake_inferer,
            spacing=[1.0, 0.5, 0.5], device=torch.device("cpu"),
            structures=["canal", "cord"],
        )
        assert result["pred"].shape == result["image"].shape
        assert result["pred"].max() < 3
        assert result["structures"] == ["canal", "cord"]
        assert result["affine"].shape == (4, 4)
        assert "mean_probs" not in result

    def test_uncertainty_maps_have_matching_spatial_shape(self, synthetic_nifti_pair):
        model = self._ToyModel(out_channels=2)
        model.eval()
        result = predict_volume(
            image_path=synthetic_nifti_pair["image"],
            model=model, inferer=self._fake_inferer,
            spacing=[1.0, 0.5, 0.5], device=torch.device("cpu"),
            uncertainty=True, n_mc_samples=3,
        )
        assert result["mean_probs"].shape[0] == 2
        assert result["entropy"].shape == result["pred"].shape
        assert result["variance"].shape == result["pred"].shape
        assert result["mutual_information"].shape == result["pred"].shape


class TestRunPredictArgumentHandling:
    def test_invalid_split_raises(self, tmp_path):
        with pytest.raises(ValueError, match="split must be"):
            run_predict({}, {"data": {}}, tmp_path / "ckpt.pth", tmp_path, split="test")

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Checkpoint not found"):
            run_predict({}, {"data": {}}, tmp_path / "missing.pth", tmp_path, split="val")

    def test_empty_datalist_raises(self, tmp_path):
        checkpoint_path = tmp_path / "ckpt.pth"
        checkpoint_path.write_bytes(b"\0")
        train_config = {"data": {"root_dir": "data", "val_sites": ["nowhere"], "dataset": "spine_generic_cord"}}
        with pytest.raises(ValueError, match="No subjects found"):
            run_predict({}, train_config, checkpoint_path, tmp_path, split="val", output_dir=tmp_path / "out")


class TestRunPredictHappyPath:
    def test_legacy_cord_only_real_tiny_run(self, tmp_path):
        root = tmp_path / "data"
        _write_real_subject(root, "sub-amu01", {"cord": "SC_seg"})

        model_cfg = _tiny_model_config(out_channels=2)
        checkpoint_path = tmp_path / "ckpt.pth"
        _save_tiny_checkpoint(checkpoint_path, model_cfg)

        train_config = {"data": {"root_dir": "data", "val_sites": ["amu"], "dataset": "spine_generic_cord"}}
        config = {
            "model": model_cfg,
            "inference": {"sliding_window": {"roi_size": _SHAPE, "overlap": 0.5, "mode": "gaussian"}},
            "preprocessing": {"spacing": [1.0, 1.0, 1.0]},
        }
        out_dir = tmp_path / "out"

        result = run_predict(
            config, train_config, checkpoint_path, tmp_path,
            split="val", output_dir=out_dir, save_overlays=False,
        )

        assert isinstance(result, InferenceResult)
        assert len(result.results) == 1
        assert result.results[0]["subject"] == "sub-amu01"
        assert (out_dir / "dice_per_subject.csv").exists()
        assert (out_dir / "metrics_summary.yaml").exists()
        assert (out_dir / "predictions" / "sub-amu01_pred.nii.gz").exists()


class TestRunEvaluateExternalArgumentHandling:
    def test_unknown_structure_for_checkpoint_raises(self, tmp_path):
        train_config = {"data": {"dataset": "spine_generic_cord"}}
        with pytest.raises(ValueError, match="not a class this checkpoint was trained on"):
            run_evaluate_external(
                "spine_generic_canal", "canal", train_config, {}, tmp_path / "ckpt.pth", tmp_path,
            )

    def test_structure_not_in_external_dataset_label_keys_raises(self, tmp_path):
        train_config = {"data": {"dataset": ["spine_generic_cord", "spine_generic_rootlets"]}}
        with pytest.raises(ValueError, match="not in .*label_keys"):
            run_evaluate_external(
                "spine_generic_canal", "rootlets", train_config, {}, tmp_path / "ckpt.pth", tmp_path,
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        train_config = {"data": {"dataset": ["spine_generic_cord", "spine_generic_canal"]}}
        with pytest.raises(ValueError, match="Checkpoint not found"):
            run_evaluate_external(
                "spine_generic_canal", "canal", train_config, {}, tmp_path / "missing.pth", tmp_path,
            )

    def test_empty_datalist_raises(self, tmp_path, monkeypatch):
        checkpoint_path = tmp_path / "ckpt.pth"
        checkpoint_path.write_bytes(b"\0")
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        spec = _fake_dataset_spec(empty_root, {"canal": "canal_seg"})
        monkeypatch.setattr(
            inference_module, "get_dataset",
            lambda name: spec if name == "fake_empty_external" else real_get_dataset(name),
        )

        train_config = {"data": {"dataset": ["spine_generic_cord", "spine_generic_canal"]}}
        with pytest.raises(ValueError, match="No subjects with"):
            run_evaluate_external(
                "fake_empty_external", "canal", train_config, {}, checkpoint_path, tmp_path,
            )


class TestRunEvaluateExternalHappyPath:
    def test_multi_structure_real_tiny_run(self, tmp_path, monkeypatch):
        root = tmp_path / "data"
        _write_real_subject(root, "sub-amu01", {"canal": "canal_seg"})
        spec = _fake_dataset_spec(root, {"canal": "canal_seg"})
        monkeypatch.setattr(
            inference_module, "get_dataset",
            lambda name: spec if name == "fake_canal_external" else real_get_dataset(name),
        )

        train_config = {"data": {"dataset": ["spine_generic_cord", "spine_generic_canal"]}}
        model_cfg = _tiny_model_config(out_channels=3)  # background + canal(1) + cord(2)
        checkpoint_path = tmp_path / "ckpt.pth"
        _save_tiny_checkpoint(checkpoint_path, model_cfg)

        inference_config = {
            "model": model_cfg,
            "inference": {"sliding_window": {"roi_size": _SHAPE, "overlap": 0.5, "mode": "gaussian"}},
            "preprocessing": {"spacing": [1.0, 1.0, 1.0]},
        }
        out_dir = tmp_path / "ext_out"

        result = run_evaluate_external(
            "fake_canal_external", "canal", train_config, inference_config,
            checkpoint_path, tmp_path, output_dir=out_dir,
        )

        assert isinstance(result, InferenceResult)
        assert len(result.results) == 1
        assert result.extra == {"dataset": "fake_canal_external", "structure": "canal", "class_id": 1}
        assert (out_dir / "dice_per_subject.csv").exists()
        assert (out_dir / "metrics_summary.yaml").exists()
