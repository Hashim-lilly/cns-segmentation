"""Tests for cns_segmentation.models.uncertainty and evaluation.calibration."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from cns_segmentation.evaluation.calibration import (
    ECEAccumulator,
    expected_calibration_error,
    plot_reliability_diagram,
)
from cns_segmentation.models.uncertainty import (
    MCDropoutWrapper,
    expected_entropy,
    mutual_information,
    predictive_entropy,
    predictive_variance,
)


class _ToyDropoutModel(nn.Module):
    """Minimal model with a dropout layer, for exercising MC-Dropout mechanics."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.dropout = nn.Dropout3d(p)
        self.conv = nn.Conv3d(2, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(self.dropout(x))


class TestPredictiveEntropy:
    def test_uniform_distribution_gives_max_entropy(self):
        num_classes = 4
        mean_probs = torch.full((1, num_classes), 1.0 / num_classes)
        entropy = predictive_entropy(mean_probs)
        assert entropy.item() == pytest.approx(np.log(num_classes), abs=1e-4)

    def test_one_hot_distribution_gives_near_zero_entropy(self):
        mean_probs = torch.tensor([[1.0, 0.0, 0.0]])
        entropy = predictive_entropy(mean_probs)
        assert entropy.item() < 1e-3

    def test_output_shape_drops_channel_dim(self):
        mean_probs = torch.softmax(torch.randn(2, 3, 4, 4), dim=1)
        entropy = predictive_entropy(mean_probs)
        assert entropy.shape == (2, 4, 4)


class TestMutualInformation:
    def test_zero_when_all_passes_identical(self):
        pass_probs = torch.tensor([[0.7, 0.3]])
        probs_stack = torch.stack([pass_probs] * 5, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        mi = mutual_information(mean_probs, probs_stack)
        assert mi.item() == pytest.approx(0.0, abs=1e-4)

    def test_positive_when_passes_disagree(self):
        probs_stack = torch.stack(
            [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])], dim=0
        )
        mean_probs = probs_stack.mean(dim=0)
        mi = mutual_information(mean_probs, probs_stack)
        assert mi.item() == pytest.approx(np.log(2), abs=1e-3)

    def test_expected_entropy_zero_for_confident_disagreeing_passes(self):
        probs_stack = torch.stack(
            [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])], dim=0
        )
        assert expected_entropy(probs_stack).item() == pytest.approx(0.0, abs=1e-3)


class TestPredictiveVariance:
    def test_zero_when_passes_identical(self):
        pass_probs = torch.tensor([[0.7, 0.3]])
        probs_stack = torch.stack([pass_probs] * 5, dim=0)
        variance = predictive_variance(probs_stack)
        assert variance.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_passes_differ(self):
        probs_stack = torch.stack(
            [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])], dim=0
        )
        variance = predictive_variance(probs_stack)
        assert variance.item() > 0.0


class TestMCDropoutWrapper:
    def test_predict_output_shape(self):
        model = _ToyDropoutModel(p=0.5)
        model.eval()
        wrapper = MCDropoutWrapper(model, n_samples=5)
        x = torch.randn(1, 2, 4, 4, 4)
        probs_stack = wrapper.predict(x)
        assert probs_stack.shape == (5, 1, 3, 4, 4, 4)

    def test_passes_vary_with_nonzero_dropout(self):
        torch.manual_seed(0)
        model = _ToyDropoutModel(p=0.5)
        model.eval()
        wrapper = MCDropoutWrapper(model, n_samples=8)
        x = torch.randn(1, 2, 4, 4, 4)
        probs_stack = wrapper.predict(x)
        assert probs_stack.std(dim=0).sum().item() > 0.0

    def test_passes_identical_with_zero_dropout(self):
        model = _ToyDropoutModel(p=0.0)
        model.eval()
        wrapper = MCDropoutWrapper(model, n_samples=4)
        x = torch.randn(1, 2, 4, 4, 4)
        probs_stack = wrapper.predict(x)
        assert probs_stack.std(dim=0).sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_model_training_mode_restored_after_predict(self):
        model = _ToyDropoutModel(p=0.5)
        model.eval()
        wrapper = MCDropoutWrapper(model, n_samples=3)
        wrapper.predict(torch.randn(1, 2, 4, 4, 4))
        assert model.training is False

        model.train()
        wrapper.predict(torch.randn(1, 2, 4, 4, 4))
        assert model.training is True

    def test_predict_with_inferer_callable(self):
        model = _ToyDropoutModel(p=0.0)
        model.eval()
        wrapper = MCDropoutWrapper(model, n_samples=2)
        x = torch.randn(1, 2, 4, 4, 4)
        calls = []

        def fake_inferer(inputs, net):
            calls.append(1)
            return net(inputs)

        probs_stack = wrapper.predict(x, inferer=fake_inferer)
        assert len(calls) == 2
        assert probs_stack.shape == (2, 1, 3, 4, 4, 4)

    def test_predict_with_uncertainty_keys_and_shapes(self):
        model = _ToyDropoutModel(p=0.5)
        model.eval()
        wrapper = MCDropoutWrapper(model, n_samples=6)
        x = torch.randn(1, 2, 4, 4, 4)
        result = wrapper.predict_with_uncertainty(x)

        assert set(result.keys()) == {"mean_probs", "entropy", "mutual_information", "variance"}
        assert result["mean_probs"].shape == (1, 3, 4, 4, 4)
        assert result["entropy"].shape == (1, 4, 4, 4)
        assert result["mutual_information"].shape == (1, 4, 4, 4)
        assert result["variance"].shape == (1, 4, 4, 4)
        assert torch.all(result["entropy"] >= 0.0)
        assert torch.all(result["mutual_information"] >= -1e-4)
        assert torch.all(result["variance"] >= 0.0)


class TestECEAccumulator:
    def test_perfectly_calibrated_gives_near_zero_ece(self):
        confidences = np.array([0.1] * 10 + [0.9] * 10)
        correct = np.array([0] * 9 + [1] * 1 + [1] * 9 + [0] * 1)
        result = expected_calibration_error(confidences, correct, n_bins=15)
        assert result["ece"] == pytest.approx(0.0, abs=1e-6)
        assert result["n_total"] == 20

    def test_fixed_miscalibration_matches_formula(self):
        confidences = np.full(100, 0.9)
        correct = np.array([1] * 50 + [0] * 50)
        result = expected_calibration_error(confidences, correct, n_bins=15)
        assert result["ece"] == pytest.approx(0.4, abs=1e-6)

    def test_streaming_updates_match_single_update(self):
        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, size=200)
        correct = rng.integers(0, 2, size=200).astype(bool)

        combined = ECEAccumulator(n_bins=15)
        combined.update(confidences, correct)
        combined_result = combined.compute()

        streamed = ECEAccumulator(n_bins=15)
        streamed.update(confidences[:80], correct[:80])
        streamed.update(confidences[80:], correct[80:])
        streamed_result = streamed.compute()

        assert streamed_result["ece"] == pytest.approx(combined_result["ece"], abs=1e-9)
        assert streamed_result["n_total"] == combined_result["n_total"]
        for a, b in zip(streamed_result["bins"], combined_result["bins"]):
            assert a["count"] == b["count"]
            if a["count"] > 0:
                assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-9)
                assert a["accuracy"] == pytest.approx(b["accuracy"], abs=1e-9)

    def test_empty_bins_report_nan_not_error(self):
        confidences = np.array([0.05, 0.06])
        correct = np.array([1, 1])
        result = expected_calibration_error(confidences, correct, n_bins=15)
        empty_bins = [b for b in result["bins"] if b["count"] == 0]
        assert len(empty_bins) > 0
        assert all(np.isnan(b["confidence"]) for b in empty_bins)
        assert np.isfinite(result["ece"])


class TestPlotReliabilityDiagram:
    def test_writes_png_file(self, tmp_path):
        confidences = np.array([0.1] * 10 + [0.9] * 10)
        correct = np.array([0] * 5 + [1] * 5 + [1] * 8 + [0] * 2)
        result = expected_calibration_error(confidences, correct, n_bins=15)

        out_path = tmp_path / "reliability.png"
        plot_reliability_diagram(result, out_path, title="Test")
        assert out_path.exists()
        assert out_path.stat().st_size > 0
