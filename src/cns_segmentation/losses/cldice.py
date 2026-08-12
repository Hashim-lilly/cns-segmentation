"""Soft clDice loss for topology-preserving segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """3-D soft erosion via min-pooling."""
    return -F.max_pool3d(-img, kernel_size=3, stride=1, padding=1)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """3-D soft dilation via max-pooling."""
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def _soft_skeletonize(img: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    """Iterative soft skeletonization.

    Args:
        img: Binary-ish tensor [B, 1, D, H, W].
        iterations: Number of thinning iterations (3 is sufficient for spine).

    Returns:
        Soft skeleton tensor.
    """
    skel = torch.zeros_like(img)
    for _ in range(iterations):
        eroded = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
        img = eroded
    return skel


class SoftClDiceLoss(nn.Module):
    """Soft clDice loss for preserving tubular topology.

    Reference: Shit et al. (2021) clDice — A Novel Topology-Preserving Loss
    Function for Tubular Structure Segmentation. CVPR.

    Args:
        iterations: Skeletonization iterations (3 is sufficient for spine).
        alpha: Weight blending clDice into Dice: loss = (1-alpha)*Dice + alpha*clDice.
        smooth: Laplace smoothing constant.
    """

    def __init__(self, iterations: int = 3, alpha: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + clDice loss.

        Args:
            pred: Softmax predictions [B, C, D, H, W], C=2.
            target: One-hot target [B, C, D, H, W], C=2.

        Returns:
            Scalar loss.
        """
        # Foreground channel only
        pred_fg = pred[:, 1:2]
        target_fg = target[:, 1:2]

        skel_pred = _soft_skeletonize(pred_fg, self.iterations)
        skel_target = _soft_skeletonize(target_fg, self.iterations)

        tprec = (skel_pred * target_fg).sum() + self.smooth
        tprec = tprec / (skel_pred.sum() + self.smooth)

        tsens = (skel_target * pred_fg).sum() + self.smooth
        tsens = tsens / (skel_target.sum() + self.smooth)

        cl_dice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens)

        # Standard Dice
        inter = (pred_fg * target_fg).sum()
        dice = 1.0 - (2.0 * inter + self.smooth) / (pred_fg.sum() + target_fg.sum() + self.smooth)

        return (1.0 - self.alpha) * dice + self.alpha * cl_dice
