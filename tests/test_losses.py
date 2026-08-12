"""Tests for loss functions."""

import pytest
import torch


def test_soft_cldice_perfect():
    from cns_segmentation.losses import SoftClDiceLoss

    loss_fn = SoftClDiceLoss()
    B, C, D, H, W = 1, 2, 16, 32, 32
    pred = torch.zeros(B, C, D, H, W)
    pred[:, 1, 4:12, 12:20, 12:20] = 10.0  # logit → foreground
    pred = torch.softmax(pred, dim=1)

    target = torch.zeros(B, C, D, H, W)
    target[:, 1, 4:12, 12:20, 12:20] = 1.0

    loss = loss_fn(pred, target)
    assert loss.item() < 0.05


def test_combined_loss_runs():
    from cns_segmentation.losses import CombinedLoss

    loss_fn = CombinedLoss()
    B, C, D, H, W = 1, 2, 8, 16, 16
    logits = torch.randn(B, C, D, H, W)
    target = torch.randint(0, C, (B, 1, D, H, W))
    loss = loss_fn(logits, target)
    assert loss.isfinite()
