"""Training loop for spinal cord segmentation.

Implements the SegmentationTrainer class which manages the full training
lifecycle: data loading, model setup, training epochs, validation with
sliding window inference, checkpointing, early stopping, and MLflow logging.
"""

import logging
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.spine_generic import create_datalist
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.segresnet import create_segresnet, empty_cache, get_device

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Manages training and validation for spinal cord segmentation.

    Orchestrates the full training pipeline including data loading,
    model initialization, training loop with validation, checkpointing,
    early stopping, and experiment tracking via MLflow.

    Args:
        config: Configuration dictionary parsed from train_spine.yaml.
    """

    def __init__(self, config: dict) -> None:
        """Initialize trainer from configuration dictionary.

        Args:
            config: Full configuration dict with keys: data, model,
                training, preprocessing, output, device, seed.
        """
        self.config = config
        self.device = get_device()
        self.seed = config.get("seed", 42)

        # Training hyperparameters
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["lr"]
        self.weight_decay = train_cfg["weight_decay"]
        self.batch_size = train_cfg["batch_size"]
        self.val_interval = train_cfg.get("val_interval", 5)
        self.num_workers = config.get("num_workers", 4)
        # On MPS, multiprocessing can cause issues with fork
        if self.device.type == "mps" and self.num_workers > 0:
            logger.info("Reducing num_workers to 0 for MPS device compatibility")
            self.num_workers = 0

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 20)
        self.min_delta = es_cfg.get("min_delta", 0.001)

        # Output paths
        output_cfg = config["output"]
        self.experiment_dir = Path(output_cfg["experiment_dir"])
        self.experiment_name = output_cfg["experiment_name"]
        self.checkpoint_dir = self.experiment_dir / self.experiment_name / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.loss_function: torch.nn.Module | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.best_metric: float = -1.0
        self.best_metric_epoch: int = -1
        self.epochs_no_improve: int = 0
        self.current_epoch: int = 0

        # Post-processing for validation
        num_classes = config["model"].get("out_channels", 2)
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
        self.post_label = Compose([AsDiscrete(to_onehot=num_classes)])

        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        logger.info("SegmentationTrainer initialized on device: %s", self.device)

    def setup_data(self) -> None:
        """Create train and validation dataloaders.

        Uses SpineGenericDataset with site-based splits and MONAI
        CacheDataset for efficient data loading with caching.
        """
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]

        root_dir = Path(data_cfg["root_dir"])
        # Resolve relative paths against the project working directory
        if not root_dir.is_absolute():
            root_dir = Path.cwd() / root_dir
        train_sites = data_cfg["train_sites"]
        val_sites = data_cfg["val_sites"]

        # Create data lists (list of dicts with "image" and "label" keys)
        train_files = create_datalist(
            root_dir=root_dir,
            sites=train_sites,
            min_file_size=data_cfg.get("min_file_size", 1000),
        )
        val_files = create_datalist(
            root_dir=root_dir,
            sites=val_sites,
            min_file_size=data_cfg.get("min_file_size", 1000),
        )

        if not train_files:
            raise RuntimeError(
                f"No training data found! Check that:\n"
                f"  1. root_dir exists: {root_dir}\n"
                f"  2. Labels are fetched (not git-annex stubs): "
                f"run 'bash scripts/setup_data.sh'\n"
                f"  3. Train sites match subject directories: {train_sites[:5]}..."
            )
        if not val_files:
            logger.warning("No validation data found — training will proceed without validation.")

        logger.info(
            "Data split: %d training volumes, %d validation volumes",
            len(train_files),
            len(val_files),
        )

        # Build transforms — pass config dict with expected keys
        preproc_cfg = self.config.get("preprocessing", {})
        transform_config = {
            "spacing": preproc_cfg.get("spacing", [1.0, 0.5, 0.5]),
            "patch_size": train_cfg["patch_size"],
            "num_samples": train_cfg.get("num_samples", 4),
        }
        train_transforms = get_train_transforms(transform_config)
        val_transforms = get_val_transforms(transform_config)

        # Create cached datasets
        train_dataset = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=self.num_workers,
        )
        val_dataset = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=self.num_workers,
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )

        logger.info("Dataloaders created (batch_size=%d)", self.batch_size)

    def setup_model(self) -> None:
        """Create SegResNet model, optimizer, scheduler, and loss function.

        Model architecture and training hyperparameters are driven by
        the configuration dictionary.
        """
        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        # Create model
        self.model = create_segresnet(model_cfg)
        self.model = self.model.to(self.device)
        logger.info(
            "Model created: %s (%.2fM parameters)",
            model_cfg["architecture"],
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

        # Optimizer
        optimizer_name = train_cfg.get("optimizer", "AdamW")
        if optimizer_name == "AdamW":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        scheduler_name = train_cfg.get("scheduler", "CosineAnnealingLR")
        scheduler_params = train_cfg.get("scheduler_params", {})
        if scheduler_name == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get("T_max", self.epochs),
                eta_min=scheduler_params.get("eta_min", 1e-6),
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        # Loss function
        loss_cfg = train_cfg.get("loss", {})
        loss_name = loss_cfg.get("name", "DiceCELoss")
        loss_params = loss_cfg.get("params", {})
        if loss_name == "DiceCELoss":
            self.loss_function = DiceCELoss(**loss_params)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        logger.info(
            "Optimizer: %s (lr=%.2e), Scheduler: %s, Loss: %s",
            optimizer_name,
            self.lr,
            scheduler_name,
            loss_name,
        )

    def train_epoch(self) -> float:
        """Run one training epoch.

        Iterates over all batches in the training dataloader, performs
        forward pass, computes loss, backpropagates, and steps the optimizer.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        step_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Train"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[yellow]{task.fields[loss]:.4f}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "Training",
                total=len(self.train_loader),
                loss=0.0,
            )

            for batch_data in self.train_loader:
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += loss.item()
                step_count += 1

                progress.update(task, advance=1, loss=loss.item())

                # Release accelerator memory between steps (no-op on CPU)
                del inputs, labels, outputs, loss
                empty_cache(self.device)

        mean_loss = epoch_loss / max(step_count, 1)
        return mean_loss

    def validate(self) -> float:
        """Run validation with sliding window inference.

        Performs sliding window inference on the validation set and
        computes the mean Dice score across all volumes.

        Returns:
            Mean Dice score on the validation set.
        """
        self.model.eval()
        sw_cfg = self.config["training"].get("sliding_window", {})

        inferer = SlidingWindowInferer(
            roi_size=sw_cfg.get("roi_size", self.config["training"]["patch_size"]),
            sw_batch_size=4,
            overlap=sw_cfg.get("overlap", 0.5),
            mode=sw_cfg.get("mode", "gaussian"),
        )

        dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Validate"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Validating", total=len(self.val_loader))

            with torch.no_grad():
                for val_data in self.val_loader:
                    val_inputs = val_data["image"].to(self.device)
                    val_labels = val_data["label"].to(self.device)

                    val_outputs = inferer(val_inputs, self.model)

                    # Post-process predictions and labels
                    val_outputs_list = decollate_batch(val_outputs)
                    val_labels_list = decollate_batch(val_labels)

                    val_outputs_post = [self.post_pred(i) for i in val_outputs_list]
                    val_labels_post = [self.post_label(i) for i in val_labels_list]

                    dice_metric(y_pred=val_outputs_post, y=val_labels_post)

                    progress.update(task, advance=1)

                    del val_inputs, val_labels, val_outputs
                    empty_cache(self.device)

        # Aggregate metric
        metric = dice_metric.aggregate().item()
        dice_metric.reset()

        return metric

    def train(self) -> dict[str, Any]:
        """Run the full training loop with validation and early stopping.

        Manages the complete training lifecycle: epoch iteration,
        periodic validation, checkpointing of best models, early stopping,
        and MLflow experiment tracking.

        Returns:
            Dictionary with final training results including best_metric,
            best_metric_epoch, and total training time.
        """
        if self.model is None:
            raise RuntimeError("Call setup_model() before train()")
        if self.train_loader is None:
            raise RuntimeError("Call setup_data() before train()")

        # Setup MLflow
        mlflow_uri = self.config["output"].get("mlflow_tracking_uri", "experiments/mlruns")
        mlflow.set_tracking_uri(str(Path(mlflow_uri).resolve()))
        mlflow.set_experiment(self.experiment_name)

        start_time = time.time()

        with mlflow.start_run(run_name=self.experiment_name):
            # Log parameters
            self._log_params()

            logger.info(
                "Starting training: %d epochs, val_interval=%d, patience=%d",
                self.epochs,
                self.val_interval,
                self.patience,
            )

            for epoch in range(self.current_epoch, self.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Training
                train_loss = self.train_epoch()
                self.scheduler.step()

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                logger.info(
                    "Epoch %d/%d — loss: %.4f, lr: %.2e, time: %.1fs",
                    epoch + 1,
                    self.epochs,
                    train_loss,
                    current_lr,
                    epoch_time,
                )

                # Log training metrics
                mlflow.log_metrics(
                    {"train_loss": train_loss, "learning_rate": current_lr},
                    step=epoch,
                )

                # Validation
                if (epoch + 1) % self.val_interval == 0:
                    val_dice = self.validate()

                    logger.info(
                        "Epoch %d/%d — val_dice: %.4f (best: %.4f @ epoch %d)",
                        epoch + 1,
                        self.epochs,
                        val_dice,
                        self.best_metric,
                        self.best_metric_epoch + 1,
                    )

                    mlflow.log_metric("val_dice", val_dice, step=epoch)

                    # Check for improvement
                    if val_dice > self.best_metric + self.min_delta:
                        self.best_metric = val_dice
                        self.best_metric_epoch = epoch
                        self.epochs_no_improve = 0

                        # Save best checkpoint
                        best_path = self.checkpoint_dir / "best_model.pth"
                        self.save_checkpoint(best_path, epoch, val_dice)
                        logger.info(
                            "New best model saved (dice=%.4f)", val_dice
                        )

                        # Log best model to MLflow
                        mlflow.log_artifact(str(best_path))
                        mlflow.log_metric("best_val_dice", val_dice, step=epoch)
                    else:
                        self.epochs_no_improve += self.val_interval

                    # Early stopping check
                    if self.epochs_no_improve >= self.patience:
                        logger.info(
                            "Early stopping triggered after %d epochs without "
                            "improvement (patience=%d)",
                            self.epochs_no_improve,
                            self.patience,
                        )
                        break

            # Save final checkpoint
            final_path = self.checkpoint_dir / "final_model.pth"
            self.save_checkpoint(final_path, self.current_epoch, self.best_metric)

            total_time = time.time() - start_time
            mlflow.log_metric("total_training_time_s", total_time)

        results = {
            "best_metric": self.best_metric,
            "best_metric_epoch": self.best_metric_epoch + 1,
            "total_epochs": self.current_epoch + 1,
            "total_time_s": total_time,
        }

        logger.info(
            "Training complete — best dice: %.4f at epoch %d (%.1f min total)",
            results["best_metric"],
            results["best_metric_epoch"],
            results["total_time_s"] / 60,
        )

        return results

    def save_checkpoint(self, path: Path, epoch: int, metric: float) -> None:
        """Save a training checkpoint to disk.

        Saves model state, optimizer state, scheduler state, and
        training metadata for resumption.

        Args:
            path: File path to save the checkpoint.
            epoch: Current epoch number.
            metric: Current best validation metric.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "best_metric_epoch": self.best_metric_epoch,
            "epochs_no_improve": self.epochs_no_improve,
            "config": self.config,
        }

        # Save on CPU so checkpoints are portable across MPS/CUDA/CPU devices
        if self.device.type != "cpu":
            checkpoint["model_state_dict"] = {
                k: v.cpu() for k, v in checkpoint["model_state_dict"].items()
            }

        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load a training checkpoint to resume training.

        Restores model state, optimizer state, scheduler state, and
        training metadata from a previously saved checkpoint.

        Args:
            path: File path to load the checkpoint from.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_metric = checkpoint["best_metric"]
        self.best_metric_epoch = checkpoint["best_metric_epoch"]
        self.epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        self.current_epoch = checkpoint["epoch"] + 1

        logger.info(
            "Checkpoint loaded from %s (epoch %d, best_dice=%.4f)",
            path,
            checkpoint["epoch"] + 1,
            self.best_metric,
        )

    def _log_params(self) -> None:
        """Log training configuration parameters to MLflow."""
        train_cfg = self.config["training"]
        model_cfg = self.config["model"]

        params = {
            "model_architecture": model_cfg["architecture"],
            "init_filters": model_cfg.get("init_filters", 32),
            "out_channels": model_cfg.get("out_channels", 2),
            "dropout_prob": model_cfg.get("dropout_prob", 0.2),
            "patch_size": str(train_cfg["patch_size"]),
            "batch_size": self.batch_size,
            "num_samples": train_cfg.get("num_samples", 4),
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "optimizer": train_cfg.get("optimizer", "AdamW"),
            "scheduler": train_cfg.get("scheduler", "CosineAnnealingLR"),
            "loss": train_cfg.get("loss", {}).get("name", "DiceCELoss"),
            "val_interval": self.val_interval,
            "early_stopping_patience": self.patience,
            "device": str(self.device),
            "seed": self.seed,
            "num_train_sites": len(self.config["data"]["train_sites"]),
            "num_val_sites": len(self.config["data"]["val_sites"]),
        }

        mlflow.log_params(params)
