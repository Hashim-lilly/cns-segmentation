"""Model factory for SegResNet spinal cord segmentation.

Wraps MONAI's SegResNet for config-driven creation, checkpoint loading,
and device management.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Determine the best available compute device.

    Returns:
        torch.device: MPS if available, then CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Selected device: %s", device)
    return device


def empty_cache(device: torch.device) -> None:
    """Release cached memory for the given accelerator device, if applicable.

    No-op on CPU. Safe to call unconditionally after moving tensors off-device.

    Args:
        device: The torch device currently in use.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def create_segresnet(config: dict) -> nn.Module:
    """Create a SegResNet model from a configuration dictionary.

    Args:
        config: Model configuration dictionary. Supported keys:
            - spatial_dims: Number of spatial dimensions (default: 3).
            - in_channels: Number of input channels (default: 1).
            - out_channels: Number of output channels (default: 2).
            - init_filters: Number of initial filters (default: 32).
            - blocks_down: Residual blocks per encoder stage (default: [1,2,2,4]).
            - blocks_up: Residual blocks per decoder stage (default: [1,1,1]).
            - dropout_prob: Dropout probability (default: 0.2).

    Returns:
        nn.Module: Configured SegResNet model.
    """
    model = SegResNet(
        spatial_dims=config.get("spatial_dims", 3),
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 2),
        init_filters=config.get("init_filters", 32),
        blocks_down=config.get("blocks_down", [1, 2, 2, 4]),
        blocks_up=config.get("blocks_up", [1, 1, 1]),
        dropout_prob=config.get("dropout_prob", 0.2),
    )

    logger.info(
        "Created SegResNet: in_channels=%d, out_channels=%d, init_filters=%d",
        config.get("in_channels", 1),
        config.get("out_channels", 2),
        config.get("init_filters", 32),
    )
    return model


def load_model(config: dict, checkpoint_path: Path) -> nn.Module:
    """Create a model and load weights from a checkpoint.

    Args:
        config: Model configuration dictionary (passed to create_segresnet).
        checkpoint_path: Path to the saved state_dict checkpoint file.

    Returns:
        nn.Module: Model with loaded weights, on the appropriate device, in eval mode.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    model = create_segresnet(config)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Loaded checkpoint from %s onto %s", checkpoint_path, device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", total)
    return total
