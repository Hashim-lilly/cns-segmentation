"""Tests for SegmentationTrainer's multi-dataset resolution and loss dispatch."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.losses import DiceCELoss

from cns_segmentation.losses import CombinedLoss
from cns_segmentation.training.trainer import SegmentationTrainer

_LABEL_SUFFIXES = {"cord": "SC_seg", "canal": "canal_seg"}
_SHAPE = (8, 16, 16)


def _make_real_subject(root: Path, subject_id: str, structures: dict[str, str]) -> None:
    """Write a valid, tiny NIfTI image + per-structure label(s) for `subject_id`.

    `structures` maps structure name -> label suffix, e.g. {"cord": "SC_seg"}.
    """
    anat_dir = root / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)

    rng = np.random.default_rng(0)
    image = rng.standard_normal(_SHAPE).astype(np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(anat_dir / f"{subject_id}_T2w.nii.gz"))

    label_dir = root / "derivatives" / "labels" / subject_id / "anat"
    label_dir.mkdir(parents=True, exist_ok=True)
    for suffix in structures.values():
        mask = np.zeros(_SHAPE, dtype=np.uint8)
        mask[2:6, 4:12, 4:12] = 1
        nib.save(nib.Nifti1Image(mask, affine), str(label_dir / f"{subject_id}_T2w_label-{suffix}.nii.gz"))


def _make_stub_subject(root: Path, subject_id: str, suffixes: list[str]) -> None:
    """Write git-annex-stub-sized (but present) image + label files.

    Used only for tests that raise before any file content is actually
    read (e.g. the out_channels validation in setup_data()).
    """
    anat_dir = root / subject_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / f"{subject_id}_T2w.nii.gz").write_bytes(b"\0" * 2000)

    label_dir = root / "derivatives" / "labels" / subject_id / "anat"
    label_dir.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (label_dir / f"{subject_id}_T2w_label-{suffix}.nii.gz").write_bytes(b"\0" * 2000)


def _base_config(
    tmp_path: Path,
    root_dir: Path,
    dataset,
    out_channels: int,
    train_sites: list[str],
    val_sites: list[str],
    patch_size=(4, 8, 8),
) -> dict:
    return {
        "data": {
            "root_dir": str(root_dir),
            "train_sites": train_sites,
            "val_sites": val_sites,
            "dataset": dataset,
            "min_file_size": 100,
        },
        "model": {
            "architecture": "SegResNet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": out_channels,
            "init_filters": 8,
            "blocks_down": [1, 1],
            "blocks_up": [1],
            "dropout_prob": 0.0,
        },
        "training": {
            "patch_size": list(patch_size),
            "batch_size": 1,
            "num_samples": 1,
            "epochs": 1,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "loss": {
                "name": "DiceCELoss",
                "params": {"include_background": False, "to_onehot_y": True, "softmax": True},
            },
        },
        "preprocessing": {"spacing": [1.0, 1.0, 1.0]},
        "output": {
            "experiment_dir": str(tmp_path / "experiments"),
            "experiment_name": "test_run",
        },
        "num_workers": 0,
        "seed": 42,
    }


class TestSetupDataLegacyPath:
    def test_single_string_dataset_defaults_to_cord_only(self, tmp_path):
        root = tmp_path / "spine-generic"
        _make_real_subject(root, "sub-siteA01", {"cord": "SC_seg"})
        _make_real_subject(root, "sub-siteB01", {"cord": "SC_seg"})

        config = _base_config(
            tmp_path, root, dataset="spine_generic_cord", out_channels=2,
            train_sites=["siteA"], val_sites=["siteB"],
        )
        trainer = SegmentationTrainer(config)
        trainer.setup_data()

        assert trainer.structures is None
        batch = next(iter(trainer.train_loader))
        assert batch["label"].shape[1] == 1


class TestSetupDataMultiStructure:
    def test_list_dataset_composites_labels_and_resolves_structures(self, tmp_path):
        root = tmp_path / "spine-generic"
        _make_real_subject(root, "sub-siteA01", _LABEL_SUFFIXES)
        _make_real_subject(root, "sub-siteA02", _LABEL_SUFFIXES)
        _make_real_subject(root, "sub-siteB01", _LABEL_SUFFIXES)

        config = _base_config(
            tmp_path, root,
            dataset=["spine_generic_cord", "spine_generic_canal"],
            out_channels=3,
            train_sites=["siteA"], val_sites=["siteB"],
        )
        trainer = SegmentationTrainer(config)
        trainer.setup_data()

        # DEFAULT_LABEL_PRIORITY = ["canal", "thecal_sac", "csf", "cord",
        # "rootlets"]; only canal/cord are requested here.
        assert trainer.structures == ["canal", "cord"]

        batch = next(iter(trainer.train_loader))
        label_vals = set(torch.unique(batch["label"]).tolist())
        assert label_vals.issubset({0.0, 1.0, 2.0})

    def test_out_channels_mismatch_raises_before_loading_data(self, tmp_path):
        root = tmp_path / "spine-generic"
        _make_stub_subject(root, "sub-siteA01", suffixes=list(_LABEL_SUFFIXES.values()))

        config = _base_config(
            tmp_path, root,
            dataset=["spine_generic_cord", "spine_generic_canal"],
            out_channels=99,
            train_sites=["siteA"], val_sites=["siteA"],
        )
        trainer = SegmentationTrainer(config)
        with pytest.raises(ValueError, match="out_channels"):
            trainer.setup_data()


class TestSetupDataRoleGuard:
    def test_non_train_dataset_raises_before_touching_disk(self, tmp_path):
        root = tmp_path / "spine-generic"  # deliberately empty — must not be read
        config = _base_config(
            tmp_path, root,
            dataset=["spine_generic_cord", "spine_generic_softseg_cord"],
            out_channels=3,
            train_sites=["siteA"], val_sites=["siteA"],
        )
        trainer = SegmentationTrainer(config)
        with pytest.raises(ValueError, match="role='train'"):
            trainer.setup_data()

    def test_all_train_datasets_pass_guard(self, tmp_path):
        root = tmp_path / "spine-generic"
        _make_real_subject(root, "sub-siteA01", _LABEL_SUFFIXES)
        _make_real_subject(root, "sub-siteA02", _LABEL_SUFFIXES)

        config = _base_config(
            tmp_path, root,
            dataset=["spine_generic_cord", "spine_generic_canal"],
            out_channels=3,
            train_sites=["siteA"], val_sites=["siteA"],
        )
        trainer = SegmentationTrainer(config)
        trainer.setup_data()  # must not raise
        assert trainer.structures == ["canal", "cord"]


class TestSetupDataRegionGuard:
    def test_mixed_regions_raises_by_default(self, tmp_path):
        root = tmp_path / "spine-generic"  # deliberately empty — must not be read
        config = _base_config(
            tmp_path, root,
            # cord/canal are "cervical-thoracic"; rootlets is "cervical".
            dataset=["spine_generic_cord", "spine_generic_rootlets"],
            out_channels=3,
            train_sites=["siteA"], val_sites=["siteA"],
        )
        trainer = SegmentationTrainer(config)
        with pytest.raises(ValueError, match="spinal regions"):
            trainer.setup_data()

    def test_mixed_regions_allowed_with_explicit_opt_in(self, tmp_path):
        root = tmp_path / "spine-generic"
        _make_real_subject(
            root, "sub-siteA01", {"cord": "SC_seg", "rootlets": "rootlets_dseg"}
        )
        _make_real_subject(
            root, "sub-siteA02", {"cord": "SC_seg", "rootlets": "rootlets_dseg"}
        )

        config = _base_config(
            tmp_path, root,
            dataset=["spine_generic_cord", "spine_generic_rootlets"],
            out_channels=3,
            train_sites=["siteA"], val_sites=["siteA"],
        )
        config["data"]["allow_mixed_regions"] = True
        trainer = SegmentationTrainer(config)
        trainer.setup_data()  # must not raise
        assert trainer.structures == ["cord", "rootlets"]

    def test_single_region_does_not_require_opt_in(self, tmp_path):
        root = tmp_path / "spine-generic"
        _make_real_subject(root, "sub-siteA01", _LABEL_SUFFIXES)
        _make_real_subject(root, "sub-siteA02", _LABEL_SUFFIXES)

        config = _base_config(
            tmp_path, root,
            dataset=["spine_generic_cord", "spine_generic_canal"],
            out_channels=3,
            train_sites=["siteA"], val_sites=["siteA"],
        )
        trainer = SegmentationTrainer(config)
        trainer.setup_data()  # must not raise


class TestSetupModelLossDispatch:
    def _config(self, tmp_path, loss_cfg):
        return {
            "model": {
                "architecture": "SegResNet",
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "init_filters": 8,
                "blocks_down": [1, 1],
                "blocks_up": [1],
                "dropout_prob": 0.0,
            },
            "training": {
                "epochs": 1,
                "lr": 1e-4,
                "weight_decay": 0.0,
                "batch_size": 1,
                "loss": loss_cfg,
            },
            "output": {
                "experiment_dir": str(tmp_path / "experiments"),
                "experiment_name": "loss_test",
            },
            "seed": 42,
        }

    def test_dicece_default(self, tmp_path):
        trainer = SegmentationTrainer(self._config(tmp_path, {"name": "DiceCELoss", "params": {}}))
        trainer.setup_model()
        assert isinstance(trainer.loss_function, DiceCELoss)

    def test_combined_loss_dispatch(self, tmp_path):
        loss_cfg = {"name": "CombinedLoss", "params": {"dice_ce_weight": 1.0, "cldice_weight": 0.5}}
        trainer = SegmentationTrainer(self._config(tmp_path, loss_cfg))
        trainer.setup_model()
        assert isinstance(trainer.loss_function, CombinedLoss)

    def test_unsupported_loss_raises(self, tmp_path):
        trainer = SegmentationTrainer(self._config(tmp_path, {"name": "NotARealLoss", "params": {}}))
        with pytest.raises(ValueError, match="Unsupported loss"):
            trainer.setup_model()
