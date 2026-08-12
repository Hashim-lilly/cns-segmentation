"""Evaluation metrics: Dice, HD95, and calibration (ECE)."""

from cns_segmentation.evaluation.metrics import compute_dice, compute_hd95, compute_ece

__all__ = ["compute_dice", "compute_hd95", "compute_ece"]
