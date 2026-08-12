"""Shared test fixtures for cns-segmentation."""

import numpy as np
import pytest


@pytest.fixture
def binary_mask_3d() -> np.ndarray:
    """Simple 3-D binary mask with a cylindrical spine-like structure."""
    mask = np.zeros((32, 64, 64), dtype=np.uint8)
    mask[4:28, 28:36, 28:36] = 1  # cylinder along axis-0
    return mask


@pytest.fixture
def pred_and_target():
    """Matching prediction/target pair and a shifted pair for Dice tests."""
    target = np.zeros((32, 64, 64), dtype=bool)
    target[4:28, 28:36, 28:36] = True

    pred_perfect = target.copy()
    pred_shifted = np.zeros_like(target)
    pred_shifted[4:28, 30:38, 28:36] = True  # 2-voxel shift
    return pred_perfect, pred_shifted, target
