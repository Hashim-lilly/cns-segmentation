"""Tests for evaluation metrics."""

import numpy as np
import pytest

from cns_segmentation.evaluation import compute_dice, compute_ece


def test_dice_perfect(pred_and_target):
    pred_perfect, _, target = pred_and_target
    assert compute_dice(pred_perfect, target) == pytest.approx(1.0)


def test_dice_zero_overlap():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[0:5, :, :] = True
    b[5:10, :, :] = True
    assert compute_dice(a, b) == pytest.approx(0.0)


def test_dice_both_empty():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    assert compute_dice(a, b) == pytest.approx(1.0)


def test_ece_perfect_calibration():
    n = 1000
    conf = np.linspace(0, 1, n)
    correct = (np.random.default_rng(42).random(n) < conf).astype(float)
    ece = compute_ece(conf, correct)
    assert ece < 0.1  # loose bound; perfect calibration gives ~0
