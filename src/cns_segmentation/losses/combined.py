"""Combined DiceCE + SoftClDice loss with configurable weighting."""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from cns_segmentation.losses.cldice import SoftClDiceLoss


class CombinedLoss(nn.Module):
    """DiceCE + Soft clDice combined loss.

    Args:
        dice_ce_weight: Weight for the DiceCE component.
        cldice_weight: Weight for the clDice component.
        cldice_iterations: Soft skeletonization iterations.
        cldice_alpha: Internal clDice blend parameter.
    """

    def __init__(
        self,
        dice_ce_weight: float = 1.0,
        cldice_weight: float = 0.5,
        cldice_iterations: int = 3,
        cldice_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice_ce_weight = dice_ce_weight
        self.cldice_weight = cldice_weight

        self.dice_ce = DiceCELoss(
            include_background=False, to_onehot_y=True, softmax=True
        )
        self.cldice = SoftClDiceLoss(iterations=cldice_iterations, alpha=cldice_alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            pred: Raw logits [B, C, D, H, W].
            target: Integer label map [B, 1, D, H, W].

        Returns:
            Scalar combined loss.
        """
        loss_dice_ce = self.dice_ce(pred, target)

        # clDice needs softmax probabilities and one-hot target
        pred_soft = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred_soft).scatter_(
            1, target.long(), 1.0
        )
        loss_cldice = self.cldice(pred_soft, target_onehot)

        return self.dice_ce_weight * loss_dice_ce + self.cldice_weight * loss_cldice
