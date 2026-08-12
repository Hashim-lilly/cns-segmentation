"""Evaluate a trained model on the validation set."""

import logging
from pathlib import Path

import typer
import yaml
from rich.logging import RichHandler
from rich.table import Table
from rich.console import Console

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)
app = typer.Typer()
console = Console()


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to inference YAML config"),
) -> None:
    """Run evaluation metrics on the validation set using a trained checkpoint."""
    cfg = yaml.safe_load(config.read_text())

    import torch
    from monai.inferers import sliding_window_inference

    from cns_segmentation.data import SpineGenericDataset, build_dataloaders, build_val_transforms
    from cns_segmentation.evaluation import compute_dice, compute_hd95
    from cns_segmentation.models import build_model

    model_cfg = cfg["model"]
    model = build_model(model_cfg)
    checkpoint = Path(model_cfg["checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location=next(model.parameters()).device))
    model.eval()
    device = next(model.parameters()).device

    # Minimal data config expected in inference yaml
    data_cfg = cfg.get("data", {})
    pre_cfg = cfg.get("preprocessing", {})
    spacing = pre_cfg.get("spacing", [1.0, 0.5, 0.5])

    val_tf = build_val_transforms(spacing=spacing)
    dataset = SpineGenericDataset(
        root_dir=data_cfg.get("root_dir", "data/spine-generic"),
        train_sites=[],
        val_sites=data_cfg.get("val_sites", []),
    )
    _, val_loader = build_dataloaders(dataset, val_tf, val_tf, batch_size=1)

    roi = cfg["inference"]["sliding_window"]["roi_size"]
    dices, hd95s = [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy().squeeze()
            preds = sliding_window_inference(images, roi, sw_batch_size=1, predictor=model, overlap=0.5)
            pred_bin = (torch.softmax(preds, dim=1)[:, 1] > cfg["inference"]["threshold"]).cpu().numpy().squeeze()
            dices.append(compute_dice(pred_bin, labels.astype(bool)))
            hd95s.append(compute_hd95(pred_bin, labels.astype(bool), tuple(spacing)))

    import numpy as np
    table = Table(title="Evaluation Results")
    table.add_column("Metric")
    table.add_column("Mean")
    table.add_column("Std")
    table.add_column("Target")
    table.add_row("Dice", f"{np.mean(dices):.4f}", f"{np.std(dices):.4f}", "≥ 0.93")
    table.add_row("HD95 (mm)", f"{np.mean(hd95s):.2f}", f"{np.std(hd95s):.2f}", "< 5.0")
    console.print(table)


if __name__ == "__main__":
    app()
