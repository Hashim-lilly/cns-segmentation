"""Tests for topology-aware loss functions in cns_segmentation.losses.topology."""

import torch
from monai.losses import DiceCELoss

from cns_segmentation.losses import CombinedLoss, SoftClDice3D, soft_skeletonize


class TestSoftSkeletonize:
    def test_output_shape_matches_input(self):
        img = torch.rand(1, 1, 8, 8, 8)
        skel = soft_skeletonize(img, iter_=3)
        assert skel.shape == img.shape

    def test_output_bounded_in_zero_one_ish_range(self):
        # relu-based accumulation of [0,1] deltas stays non-negative; ceiling
        # isn't strictly 1 but should stay small for a [0, 1] input.
        img = torch.rand(1, 1, 8, 8, 8)
        skel = soft_skeletonize(img, iter_=3)
        assert torch.all(skel >= 0)


class TestSoftClDice3D:
    def test_perfect_prediction_approaches_achievable_floor(self):
        # A thin tubular target: a 1-voxel-wide line through the volume.
        # `smooth=1.0` is additive in the *outer* tprec/tsens combination
        # (not just as zero-division guards), so cldice = 2*tprec*tsens /
        # (tprec+tsens+smooth) structurally caps below 1 even for a perfect
        # prediction: as tprec, tsens -> 1, cldice -> 2/3, so this loss
        # (1 - cldice) floors at 1/3, never at 0. A longer tube pushes
        # tprec/tsens closer to 1 and the loss closer to that 1/3 floor.
        shape = (1, 48, 8, 8)
        targets = torch.zeros(shape, dtype=torch.long)
        targets[0, 4:44, 4, 4] = 1

        logits = torch.full((1, 2, 48, 8, 8), -10.0)
        logits[:, 0] = 10.0
        logits[:, 1][targets == 1] = 10.0
        logits[:, 0][targets == 1] = -10.0

        loss_fn = SoftClDice3D(iter_=3)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() < 0.36

    def test_gradient_is_finite(self):
        torch.manual_seed(0)
        inputs = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        targets = torch.randint(0, 2, (1, 1, 8, 8, 8))

        loss_fn = SoftClDice3D(iter_=2)
        loss = loss_fn(inputs, targets)
        loss.backward()

        assert torch.isfinite(loss)
        assert inputs.grad is not None
        assert torch.all(torch.isfinite(inputs.grad))

    def test_disjoint_prediction_has_high_loss(self):
        shape = (1, 16, 8, 8)
        targets = torch.zeros(shape, dtype=torch.long)
        targets[0, 3:13, 4, 4] = 1

        # Prediction confidently marks a completely different line.
        logits = torch.full((1, 2, 16, 8, 8), 10.0)
        logits[:, 1] = -10.0
        logits[:, 1, 3:13, 6, 6] = 10.0
        logits[:, 0, 3:13, 6, 6] = -10.0

        loss_fn = SoftClDice3D(iter_=3)
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.7


class TestCombinedLoss:
    def _inputs_targets(self, num_classes: int = 3):
        torch.manual_seed(0)
        inputs = torch.randn(2, num_classes, 8, 8, 8)
        targets = torch.randint(0, num_classes, (2, 1, 8, 8, 8))
        return inputs, targets

    def test_dice_ce_only_matches_standalone_dicece(self):
        inputs, targets = self._inputs_targets()
        combined = CombinedLoss(dice_ce_weight=1.0, cldice_weight=0.0)
        standalone = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

        assert torch.allclose(combined(inputs, targets), standalone(inputs, targets))

    def test_cldice_only_matches_standalone_cldice(self):
        inputs, targets = self._inputs_targets()
        combined = CombinedLoss(dice_ce_weight=0.0, cldice_weight=1.0, cldice_iter=3)
        standalone = SoftClDice3D(iter_=3, include_background=False)

        assert torch.allclose(combined(inputs, targets), standalone(inputs, targets))

    def test_weighted_sum_matches_manual_combination(self):
        inputs, targets = self._inputs_targets()
        combined = CombinedLoss(dice_ce_weight=0.5, cldice_weight=0.5, cldice_iter=3)
        dice_ce = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        cldice = SoftClDice3D(iter_=3, include_background=False)

        expected = 0.5 * dice_ce(inputs, targets) + 0.5 * cldice(inputs, targets)
        assert torch.allclose(combined(inputs, targets), expected)

    def test_zero_weight_component_not_instantiated(self):
        combined = CombinedLoss(dice_ce_weight=1.0, cldice_weight=0.0)
        assert combined.cldice is None
        assert combined.dice_ce is not None

        combined2 = CombinedLoss(dice_ce_weight=0.0, cldice_weight=1.0)
        assert combined2.dice_ce is None
        assert combined2.cldice is not None
