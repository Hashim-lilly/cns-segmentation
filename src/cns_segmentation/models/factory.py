"""SegResNet factory for spinal cord segmentation."""

import logging

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

logger = logging.getLogger(__name__)


def build_model(cfg: dict) -> nn.Module:
    """Instantiate a SegResNet from a config dict.

    Args:
        cfg: Model sub-config with keys: spatial_dims, in_channels,
             out_channels, init_filters, blocks_down, blocks_up, dropout_prob.

    Returns:
        SegResNet model on the appropriate device.
    """
    model = SegResNet(
        spatial_dims=cfg["spatial_dims"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        init_filters=cfg.get("init_filters", 32),
        blocks_down=cfg.get("blocks_down", [1, 2, 2, 4]),
        blocks_up=cfg.get("blocks_up", [1, 1, 1]),
        dropout_prob=cfg.get("dropout_prob", 0.2),
    )

    device = _get_device()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("SegResNet: %d trainable parameters, device=%s", n_params, device)
    return model


def _get_device() -> torch.device:
    """Select MPS, CUDA, or CPU in that order of preference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
