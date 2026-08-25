"""MC-Dropout uncertainty estimation for spinal cord segmentation.

Runs multiple stochastic forward passes with dropout kept active at
inference time, then derives predictive entropy, mutual information
(epistemic uncertainty), and per-voxel variance from the resulting
distribution of softmax predictions.
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def predictive_entropy(mean_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the predictive entropy H[ȳ] of a mean softmax distribution.

    Total uncertainty: H[ȳ] = -Σ_c ȳ_c * log(ȳ_c), reduced over the class
    (channel) dimension.

    Args:
        mean_probs: Mean softmax probabilities, shape [B, C, ...].
        eps: Small constant added before the log to avoid log(0).

    Returns:
        Tensor of shape [B, ...] (channel dimension reduced away).
    """
    return -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)


def expected_entropy(probs_stack: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute the expected per-pass entropy E_θ[H[y|θ]] across MC samples.

    Args:
        probs_stack: Stacked softmax probabilities from each stochastic
            pass, shape [N, B, C, ...].
        eps: Small constant added before the log to avoid log(0).

    Returns:
        Tensor of shape [B, ...]: per-pass entropy averaged over N.
    """
    per_pass_entropy = -(probs_stack * torch.log(probs_stack + eps)).sum(dim=2)
    return per_pass_entropy.mean(dim=0)


def mutual_information(
    mean_probs: torch.Tensor, probs_stack: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute the mutual information I[y, θ|x] — epistemic (model) uncertainty.

    I[y, θ|x] = H[ȳ] - E_θ[H[y|θ]]: the gap between total uncertainty and
    the average uncertainty of a single stochastic pass, i.e. how much of
    the uncertainty comes from disagreement between passes rather than
    from ambiguity within a single pass.

    Args:
        mean_probs: Mean softmax probabilities, shape [B, C, ...].
        probs_stack: Stacked per-pass softmax probabilities, shape [N, B, C, ...].
        eps: Small constant added before the log to avoid log(0).

    Returns:
        Tensor of shape [B, ...].
    """
    return predictive_entropy(mean_probs, eps) - expected_entropy(probs_stack, eps)


def predictive_variance(probs_stack: torch.Tensor) -> torch.Tensor:
    """Compute a simple per-voxel spread measure across MC samples.

    Variance across the N stochastic passes is computed per class, then
    averaged over the class (channel) dimension to give a single scalar
    map.

    Args:
        probs_stack: Stacked per-pass softmax probabilities, shape [N, B, C, ...].

    Returns:
        Tensor of shape [B, ...].
    """
    return probs_stack.var(dim=0, unbiased=False).mean(dim=1)


class MCDropoutWrapper:
    """Wraps a model to run MC-Dropout inference and derive uncertainty maps.

    Keeps dropout active at inference time (`model.train()`) while running
    multiple stochastic forward passes, then aggregates them into the mean
    prediction plus predictive entropy, mutual information, and variance.
    """

    def __init__(self, model: nn.Module, n_samples: int = 8) -> None:
        """Create an MC-Dropout wrapper around a trained model.

        Args:
            model: Trained model with at least one dropout layer.
            n_samples: Number of stochastic forward passes (N).
        """
        self.model = model
        self.n_samples = n_samples

    def predict(
        self,
        inputs: torch.Tensor,
        inferer: Optional[Callable[[torch.Tensor, nn.Module], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run N stochastic forward passes with dropout active.

        Args:
            inputs: Input tensor, shape [B, C_in, ...].
            inferer: Optional callable `(inputs, model) -> logits`, e.g. a
                MONAI `SlidingWindowInferer`, used when the input doesn't
                fit through the model directly. Defaults to calling the
                model directly on `inputs`.

        Returns:
            Stacked softmax probabilities from each pass, shape [N, B, C, ...].
        """
        was_training = self.model.training
        forward = inferer if inferer is not None else (lambda x, m: m(x))

        self.model.train()  # keep dropout active
        probs = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = forward(inputs, self.model)
                probs.append(torch.softmax(logits, dim=1))
        self.model.train(was_training)

        return torch.stack(probs, dim=0)

    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor,
        inferer: Optional[Callable[[torch.Tensor, nn.Module], torch.Tensor]] = None,
        eps: float = 1e-8,
    ) -> dict:
        """Run MC-Dropout inference and compute all uncertainty maps.

        Args:
            inputs: Input tensor, shape [B, C_in, ...].
            inferer: Optional callable `(inputs, model) -> logits`, as in `predict()`.
            eps: Small constant added before logs to avoid log(0).

        Returns:
            Dictionary with:
                - "mean_probs": mean softmax probabilities, shape [B, C, ...].
                - "entropy": predictive entropy, shape [B, ...].
                - "mutual_information": epistemic uncertainty, shape [B, ...].
                - "variance": per-voxel spread measure, shape [B, ...].
        """
        probs_stack = self.predict(inputs, inferer=inferer)
        mean_probs = probs_stack.mean(dim=0)

        return {
            "mean_probs": mean_probs,
            "entropy": predictive_entropy(mean_probs, eps),
            "mutual_information": mutual_information(mean_probs, probs_stack, eps),
            "variance": predictive_variance(probs_stack),
        }
