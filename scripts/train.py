"""Train the segmentation model."""

import logging
from pathlib import Path

import typer
import yaml
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., "--config", help="Path to train YAML config")) -> None:
    """Train a SegResNet segmentation model from a YAML config."""
    cfg = yaml.safe_load(config.read_text())

    from cns_segmentation.data import SpineGenericDataset, build_dataloaders
    from cns_segmentation.data import build_train_transforms, build_val_transforms
    from cns_segmentation.models import build_model
    from cns_segmentation.training import Trainer

    # Build loss
    loss_cfg = cfg["training"]["loss"]
    if loss_cfg["name"] == "DiceCELoss":
        from monai.losses import DiceCELoss
        loss_fn = DiceCELoss(**loss_cfg.get("params", {}))
    elif loss_cfg["name"] == "CombinedLoss":
        from cns_segmentation.losses import CombinedLoss
        loss_fn = CombinedLoss(**loss_cfg.get("params", {}))
    else:
        raise ValueError(f"Unknown loss: {loss_cfg['name']}")

    data_cfg = cfg["data"]
    pre_cfg = cfg.get("preprocessing", {})
    spacing = pre_cfg.get("spacing", [1.0, 0.5, 0.5])
    train_cfg = cfg["training"]

    dataset = SpineGenericDataset(
        root_dir=data_cfg["root_dir"],
        train_sites=data_cfg["train_sites"],
        val_sites=data_cfg["val_sites"],
        contrast=data_cfg.get("contrast", "T2w"),
        label_key=data_cfg.get("label_key", "label-SC_seg"),
    )

    train_tf = build_train_transforms(
        patch_size=train_cfg["patch_size"],
        spacing=spacing,
        num_samples=train_cfg.get("num_samples", 4),
        pos_neg_ratio=train_cfg.get("pos_neg_ratio", 2.0),
    )
    val_tf = build_val_transforms(spacing=spacing)
    train_loader, val_loader = build_dataloaders(
        dataset, train_tf, val_tf, batch_size=train_cfg["batch_size"]
    )

    model = build_model(cfg["model"])

    experiment_dir = Path(cfg["output"]["experiment_dir"])
    trainer = Trainer(model, loss_fn, train_loader, val_loader, cfg, experiment_dir)
    trainer.train()
    logger.info("Training complete. Best model saved to %s", experiment_dir / "best_model.pth")


if __name__ == "__main__":
    app()
