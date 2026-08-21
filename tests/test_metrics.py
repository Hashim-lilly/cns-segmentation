"""Tests for cns_segmentation.evaluation.metrics class_map support."""

import nibabel as nib
import numpy as np
import pytest

from cns_segmentation.evaluation.metrics import aggregate_metrics, evaluate_subject


def _save(path, data, spacing=(1.0, 1.0, 1.0)):
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), str(path))


class TestEvaluateSubjectFlat:
    def test_none_class_map_returns_flat_dict(self, tmp_path):
        shape = (10, 10, 10)
        pred = np.zeros(shape, dtype=np.uint8)
        pred[2:6, 2:6, 2:6] = 1
        label = np.zeros(shape, dtype=np.uint8)
        label[2:6, 2:6, 2:6] = 1
        pred_path = tmp_path / "sub-amu01_pred.nii.gz"
        label_path = tmp_path / "sub-amu01_label.nii.gz"
        _save(pred_path, pred)
        _save(label_path, label)

        result = evaluate_subject(pred_path, label_path)
        assert set(result.keys()) == {
            "subject",
            "site",
            "dice",
            "hausdorff95_mm",
            "volume_error_mm3",
            "surface_dice",
        }
        assert result["dice"] == pytest.approx(1.0)

    def test_multiclass_arrays_binarized_when_no_class_map(self, tmp_path):
        shape = (10, 10, 10)
        pred = np.zeros(shape, dtype=np.uint8)
        pred[2:6, 2:6, 2:6] = 2
        label = np.zeros(shape, dtype=np.uint8)
        label[2:6, 2:6, 2:6] = 1
        pred_path = tmp_path / "a_pred.nii.gz"
        label_path = tmp_path / "a_label.nii.gz"
        _save(pred_path, pred)
        _save(label_path, label)

        result = evaluate_subject(pred_path, label_path)
        assert result["dice"] == pytest.approx(1.0)

    def test_missing_file_raises(self, tmp_path):
        label_path = tmp_path / "label.nii.gz"
        _save(label_path, np.zeros((4, 4, 4), dtype=np.uint8))
        with pytest.raises(FileNotFoundError):
            evaluate_subject(tmp_path / "missing_pred.nii.gz", label_path)


class TestEvaluateSubjectClassMap:
    def test_nested_shape_with_class_map(self, tmp_path):
        shape = (10, 10, 10)
        pred = np.zeros(shape, dtype=np.uint8)
        pred[2:6, 2:6, 2:6] = 1  # cord: matches label exactly
        pred[6:8, 2:6, 2:6] = 2  # canal: present only in prediction
        label = np.zeros(shape, dtype=np.uint8)
        label[2:6, 2:6, 2:6] = 1  # cord

        pred_path = tmp_path / "sub_pred.nii.gz"
        label_path = tmp_path / "sub_label.nii.gz"
        _save(pred_path, pred)
        _save(label_path, label)

        class_map = {"cord": 1, "canal": 2}
        result = evaluate_subject(pred_path, label_path, class_map=class_map)

        assert set(result.keys()) == {"subject", "site", "cord", "canal", "overall"}
        assert result["cord"]["dice"] == pytest.approx(1.0)
        assert result["canal"]["dice"] == pytest.approx(0.0)
        assert result["overall"]["dice"] < 1.0

    def test_class_map_overall_matches_flat_binarized_result(self, tmp_path):
        shape = (8, 8, 8)
        pred = np.zeros(shape, dtype=np.uint8)
        pred[1:4, 1:4, 1:4] = 1
        label = np.zeros(shape, dtype=np.uint8)
        label[1:4, 1:4, 1:4] = 1
        pred_path = tmp_path / "x_pred.nii.gz"
        label_path = tmp_path / "x_label.nii.gz"
        _save(pred_path, pred)
        _save(label_path, label)

        flat = evaluate_subject(pred_path, label_path)
        nested = evaluate_subject(pred_path, label_path, class_map={"cord": 1})

        assert nested["overall"]["dice"] == pytest.approx(flat["dice"])
        assert nested["cord"]["dice"] == pytest.approx(flat["dice"])


class TestAggregateMetrics:
    def test_empty_results(self):
        summary = aggregate_metrics([])
        assert summary == {"overall": {}, "per_site": {}, "n_subjects": 0}

    def test_flat_aggregation_unchanged(self):
        results = [
            {
                "subject": "s1",
                "site": "siteA",
                "dice": 0.9,
                "hausdorff95_mm": 1.0,
                "volume_error_mm3": 10.0,
                "surface_dice": 0.8,
            },
            {
                "subject": "s2",
                "site": "siteA",
                "dice": 0.8,
                "hausdorff95_mm": 2.0,
                "volume_error_mm3": 20.0,
                "surface_dice": 0.7,
            },
        ]
        summary = aggregate_metrics(results)
        assert summary["n_subjects"] == 2
        assert summary["overall"]["dice"]["mean"] == pytest.approx(0.85)
        assert summary["overall"]["dice"]["n"] == 2
        assert "siteA" in summary["per_site"]
        assert summary["per_site"]["siteA"]["dice"]["mean"] == pytest.approx(0.85)

    def test_nested_aggregation_per_structure(self):
        results = [
            {
                "subject": "s1",
                "site": "siteA",
                "cord": {"dice": 0.9, "hausdorff95_mm": 1.0, "volume_error_mm3": 10.0, "surface_dice": 0.8},
                "canal": {"dice": 0.6, "hausdorff95_mm": 3.0, "volume_error_mm3": 30.0, "surface_dice": 0.5},
                "overall": {"dice": 0.75, "hausdorff95_mm": 2.0, "volume_error_mm3": 20.0, "surface_dice": 0.65},
            },
            {
                "subject": "s2",
                "site": "siteB",
                "cord": {"dice": 0.8, "hausdorff95_mm": 1.5, "volume_error_mm3": 12.0, "surface_dice": 0.75},
                "canal": {"dice": 0.5, "hausdorff95_mm": 4.0, "volume_error_mm3": 35.0, "surface_dice": 0.45},
                "overall": {"dice": 0.65, "hausdorff95_mm": 2.5, "volume_error_mm3": 22.0, "surface_dice": 0.6},
            },
        ]
        summary = aggregate_metrics(results)
        assert set(summary["overall"].keys()) == {"cord", "canal", "overall"}
        assert summary["overall"]["cord"]["dice"]["mean"] == pytest.approx(0.85)
        assert summary["overall"]["canal"]["dice"]["mean"] == pytest.approx(0.55)
        assert summary["n_subjects"] == 2
        assert set(summary["per_site"].keys()) == {"siteA", "siteB"}
        assert summary["per_site"]["siteA"]["cord"]["dice"]["mean"] == pytest.approx(0.9)
