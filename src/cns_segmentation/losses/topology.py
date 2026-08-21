"""Topology-aware loss functions for tubular/thin-structure segmentation.

Implements Soft clDice (Shit et al., 2021) via iterative 3D soft
morphological skeletonization, and a CombinedLoss that blends it with
MONAI's DiceCELoss. Both accept the same raw-logits / raw-integer-label
calling convention as DiceCELoss so either drops into
`SegmentationTrainer.train_epoch()`'s `loss_function(outputs, labels)` call
unchanged.
"""

import logging

import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.networks import one_hot
from torch import nn

logger = logging.getLogger(__name__)


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """3D soft erosion via min-pooling along each axis independently."""
    p1 = -F.max_pool3d(-img, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-img, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-img, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """3D soft dilation via max-pooling with a 3x3x3 window."""
    return F.max_pool3d(img, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    """3D soft morphological opening (erode then dilate)."""
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Approximate the morphological skeleton of a soft (probabilistic) mask.

    Iteratively erodes the input and keeps the residual (img - open(img)) at
    each step, which isolates thin ridge/centerline structures. Fully
    differentiable, operating on continuous [0, 1] inputs rather than binary
    masks — see Shit et al., "clDice — A Novel Topology-Preserving Loss
    Function for Tubular Structure Segmentation" (CVPR 2021).

    Args:
        img: Tensor of shape (B, 1, D, H, W) with values in [0, 1].
        iter_: Number of erosion iterations. `iter_=3` is sufficient for the
            spinal cord (per CLAUDE.md rule 7); thinner structures (rootlets)
            may need more.

    Returns:
        Soft skeleton tensor of the same shape as `img`.
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDice3D(nn.Module):
    """Differentiable centerline-Dice (clDice) loss for 3D volumes.

    Penalizes topological errors (breaks, gaps) in thin tubular structures
    that voxel-overlap losses like Dice are insensitive to, by comparing soft
    skeletons of the prediction and ground truth rather than the full masks.

    Args:
        iter_: Soft-skeletonization erosion iterations. Default 3.
        smooth: Additive smoothing to avoid division by zero. Default 1.0.
        include_background: Whether to include the background channel
            (class 0) in the loss. Default False, matching DiceCELoss's
            default convention in this codebase.
    """

    def __init__(
        self,
        iter_: int = 3,
        smooth: float = 1.0,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the soft clDice loss.

        Args:
            inputs: Raw logits, shape (B, C, D, H, W).
            targets: Integer class labels, shape (B, 1, D, H, W) or
                (B, D, H, W) — same convention DiceCELoss(to_onehot_y=True)
                expects.

        Returns:
            Scalar loss tensor: `1 - clDice`, averaged over foreground
            classes.
        """
        num_classes = inputs.shape[1]
        probs = torch.softmax(inputs, dim=1)

        if targets.dim() == inputs.dim() - 1:
            targets = targets.unsqueeze(1)
        targets_onehot = one_hot(targets, num_classes=num_classes)

        start = 0 if self.include_background else 1
        cldice_per_class = []
        for c in range(start, num_classes):
            pred_c = probs[:, c : c + 1]
            true_c = targets_onehot[:, c : c + 1]

            skel_pred = soft_skeletonize(pred_c, self.iter_)
            skel_true = soft_skeletonize(true_c, self.iter_)

            tprec = (skel_pred * true_c).sum() / (skel_pred.sum() + self.smooth)
            tsens = (skel_true * pred_c).sum() / (skel_true.sum() + self.smooth)
            cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
            cldice_per_class.append(cldice)

        mean_cldice = torch.stack(cldice_per_class).mean()
        return 1.0 - mean_cldice


class CombinedLoss(nn.Module):
    """Weighted sum of DiceCELoss and SoftClDice3D.

    `CombinedLoss(dice_ce_weight=1, cldice_weight=0)` is DiceCE-only;
    `(0, 1)` is clDice-only — the 3-way ablation in
    `docs/phases/phase2.md` ((1,0)/(0,1)/(1,0.5)) is expressed via these
    weights rather than a separate loss_name branch, so both terms share one
    mechanism and trainer.py's loss dispatch stays a single `elif`.

    Args:
        dice_ce_weight: Weight on the DiceCE term. Default 1.0.
        cldice_weight: Weight on the SoftClDice3D term. Default 0.5.
        cldice_iter: Soft-skeletonization erosion iterations. Default 3.
        include_background: Passed through to both component losses.
        to_onehot_y: Passed through to DiceCELoss.
        softmax: Passed through to DiceCELoss.
    """

    def __init__(
        self,
        dice_ce_weight: float = 1.0,
        cldice_weight: float = 0.5,
        cldice_iter: int = 3,
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.dice_ce_weight = dice_ce_weight
        self.cldice_weight = cldice_weight

        self.dice_ce = (
            DiceCELoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                softmax=softmax,
            )
            if dice_ce_weight > 0
            else None
        )
        self.cldice = (
            SoftClDice3D(iter_=cldice_iter, include_background=include_background)
            if cldice_weight > 0
            else None
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted combined loss.

        Args:
            inputs: Raw logits, shape (B, C, D, H, W).
            targets: Integer class labels, shape (B, 1, D, H, W).

        Returns:
            Scalar weighted loss tensor.
        """
        loss = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
        if self.dice_ce is not None:
            loss = loss + self.dice_ce_weight * self.dice_ce(inputs, targets)
        if self.cldice is not None:
            loss = loss + self.cldice_weight * self.cldice(inputs, targets)
        return loss
