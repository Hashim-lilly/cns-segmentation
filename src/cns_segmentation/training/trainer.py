"""Training engine with MLflow experiment tracking."""

import logging
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for 3-D segmentation models.

    Args:
        model: Segmentation model.
        loss_fn: Loss function.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Full training config dict.
        experiment_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        experiment_dir: Path,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.experiment_dir = experiment_dir
        experiment_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["training"]
        self.device = next(model.parameters()).device
        self.epochs = train_cfg["epochs"]
        self.roi_size = train_cfg["patch_size"]

        self.optimizer = AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg["scheduler_params"]["T_max"],
            eta_min=train_cfg["scheduler_params"]["eta_min"],
        )
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.best_dice = 0.0

    def train(self) -> None:
        """Run the full training loop, logging metrics to MLflow."""
        with mlflow.start_run():
            mlflow.log_params(self.cfg["training"])
            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch()
                val_dice = self._validate()
                self.scheduler.step()

                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_dice": val_dice}, step=epoch
                )
                logger.info("Epoch %d/%d — loss=%.4f  val_dice=%.4f", epoch, self.epochs, train_loss, val_dice)

                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    torch.save(self.model.state_dict(), self.experiment_dir / "best_model.pth")

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    def _validate(self) -> float:
        self.model.eval()
        self.dice_metric.reset()
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                preds = sliding_window_inference(
                    images, self.roi_size, sw_batch_size=1, predictor=self.model, overlap=0.5
                )
                preds_bin = (torch.softmax(preds, dim=1)[:, 1:2] > 0.5).long()
                self.dice_metric(y_pred=preds_bin, y=labels)

        return self.dice_metric.aggregate().item()
