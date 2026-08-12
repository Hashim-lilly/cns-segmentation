"""MC-Dropout wrapper for uncertainty quantification at inference time."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MCDropoutWrapper(nn.Module):
    """Wraps any model for MC-Dropout inference.

    Keeps dropout layers active during inference by calling model.train()
    on modules that are Dropout instances while keeping BN in eval mode.

    Args:
        model: Trained segmentation model with dropout layers.
        n_samples: Number of stochastic forward passes.
    """

    def __init__(self, model: nn.Module, n_samples: int = 8) -> None:
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run MC-Dropout forward passes and return statistics.

        Args:
            x: Input tensor of shape [B, C, D, H, W].

        Returns:
            Dict with keys:
                - "mean": Mean softmax prediction [B, C, D, H, W]
                - "variance": Predictive variance [B, C, D, H, W]
                - "entropy": Predictive entropy [B, D, H, W]
        """
        self._enable_dropout()
        preds = torch.stack([torch.softmax(self.model(x), dim=1) for _ in range(self.n_samples)])
        self.model.eval()

        mean = preds.mean(dim=0)
        variance = preds.var(dim=0)
        # Entropy of the mean prediction
        eps = 1e-8
        entropy = -(mean * (mean + eps).log()).sum(dim=1)
        return {"mean": mean, "variance": variance, "entropy": entropy}

    def _enable_dropout(self) -> None:
        """Put model in train mode only for Dropout layers."""
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout | nn.Dropout2d | nn.Dropout3d):
                module.train()
