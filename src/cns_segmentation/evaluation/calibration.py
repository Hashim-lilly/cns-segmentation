"""Calibration evaluation for segmentation uncertainty estimates.

Provides 15-bin Expected Calibration Error (ECE) with reliability diagrams,
computed via a streaming accumulator so per-subject voxel arrays can be
folded in one subject at a time instead of concatenating full-resolution
arrays for an entire validation split in memory.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class ECEAccumulator:
    """Streaming accumulator for Expected Calibration Error.

    Bins voxels by predicted confidence into `n_bins` equal-width bins,
    accumulating per-bin confidence/correctness sums so `.update()` can be
    called repeatedly (e.g. once per subject) without holding every voxel
    in memory at once.
    """

    def __init__(self, n_bins: int = 15) -> None:
        """Create an accumulator with `n_bins` equal-width confidence bins.

        Args:
            n_bins: Number of reliability-diagram bins spanning [0, 1].
        """
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_conf_sum = np.zeros(n_bins)
        self.bin_correct_sum = np.zeros(n_bins)
        self.bin_count = np.zeros(n_bins, dtype=np.int64)

    def update(self, confidences: np.ndarray, correct: np.ndarray) -> None:
        """Fold a batch of per-voxel confidences/correctness into the bins.

        Args:
            confidences: Predicted confidence per voxel, values in [0, 1].
            correct: Whether the prediction was correct at each voxel
                (boolean or 0/1), same shape as `confidences`.
        """
        confidences = np.asarray(confidences).ravel()
        correct = np.asarray(correct).ravel().astype(bool)

        bin_idx = np.clip(
            np.digitize(confidences, self.bin_edges[1:-1]), 0, self.n_bins - 1
        )
        for b in range(self.n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            self.bin_conf_sum[b] += confidences[mask].sum()
            self.bin_correct_sum[b] += correct[mask].sum()
            self.bin_count[b] += int(mask.sum())

    def compute(self) -> dict:
        """Compute the final ECE and per-bin reliability-diagram stats.

        Returns:
            Dictionary with:
                - "ece": Expected Calibration Error, `Σ (n_bin/n_total)*|accuracy-confidence|`.
                - "n_total": total number of voxels folded in.
                - "bins": list of per-bin dicts with "bin_lower", "bin_upper",
                  "confidence" (mean confidence in the bin), "accuracy" (mean
                  correctness in the bin), and "count". Empty bins report
                  NaN confidence/accuracy and count 0.
        """
        n_total = int(self.bin_count.sum())
        ece = 0.0
        bins = []
        for b in range(self.n_bins):
            n_bin = int(self.bin_count[b])
            bin_lower = float(self.bin_edges[b])
            bin_upper = float(self.bin_edges[b + 1])
            if n_bin == 0:
                bins.append(
                    {
                        "bin_lower": bin_lower,
                        "bin_upper": bin_upper,
                        "confidence": float("nan"),
                        "accuracy": float("nan"),
                        "count": 0,
                    }
                )
                continue
            confidence = self.bin_conf_sum[b] / n_bin
            accuracy = self.bin_correct_sum[b] / n_bin
            if n_total > 0:
                ece += (n_bin / n_total) * abs(accuracy - confidence)
            bins.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "confidence": float(confidence),
                    "accuracy": float(accuracy),
                    "count": n_bin,
                }
            )

        return {"ece": float(ece), "n_total": n_total, "bins": bins}


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> dict:
    """Compute ECE in one shot for a single batch of confidences/correctness.

    Convenience wrapper around `ECEAccumulator` for callers that already
    have every voxel in memory at once (e.g. tests, small volumes).

    Args:
        confidences: Predicted confidence per voxel, values in [0, 1].
        correct: Whether the prediction was correct at each voxel.
        n_bins: Number of reliability-diagram bins.

    Returns:
        Same shape as `ECEAccumulator.compute()`.
    """
    accumulator = ECEAccumulator(n_bins=n_bins)
    accumulator.update(confidences, correct)
    return accumulator.compute()


def plot_reliability_diagram(
    bin_result: dict, out_path: Path, title: str = "Reliability Diagram"
) -> None:
    """Save a reliability diagram (accuracy vs. confidence per bin) as a PNG.

    Args:
        bin_result: Output of `ECEAccumulator.compute()` / `expected_calibration_error()`.
        out_path: Destination PNG path.
        title: Plot title.
    """
    bins = bin_result["bins"]
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
    accuracies = [b["accuracy"] for b in bins]
    bin_width = bins[0]["bin_upper"] - bins[0]["bin_lower"] if bins else 1.0 / 15

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.bar(
        bin_centers,
        accuracies,
        width=bin_width * 0.9,
        edgecolor="black",
        color="steelblue",
        label="Observed accuracy",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} (ECE={bin_result['ece']:.4f})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
