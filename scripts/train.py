#!/usr/bin/env python
"""CLI entry point for spinal cord segmentation training.

Usage:
    python scripts/train.py --config configs/train_spine.yaml
    python scripts/train.py --epochs 50 --lr 3e-4 --batch-size 4
"""

import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.segresnet import get_device  # noqa: E402
from src.training.trainer import SegmentationTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Logging & console setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="train",
    help="Train a spinal cord segmentation model.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        typer.BadParameter: If the file does not exist or cannot be parsed.
    """
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(f"Invalid YAML in {config_path}: {exc}")
    return cfg


def _apply_overrides(
    cfg: dict,
    epochs: Optional[int],
    lr: Optional[float],
    batch_size: Optional[int],
    seed: Optional[int],
    experiment_name: Optional[str],
) -> dict:
    """Apply CLI overrides to the loaded config dictionary.

    Args:
        cfg: Base configuration dictionary.
        epochs: Override for training.epochs.
        lr: Override for training.lr.
        batch_size: Override for training.batch_size.
        seed: Override for global seed.
        experiment_name: Override for output.experiment_name.

    Returns:
        Updated configuration dictionary.
    """
    if epochs is not None:
        cfg["training"]["epochs"] = epochs
        # Also update scheduler T_max if using CosineAnnealingLR
        scheduler_params = cfg["training"].get("scheduler_params", {})
        if cfg["training"].get("scheduler") == "CosineAnnealingLR":
            scheduler_params["T_max"] = epochs
            cfg["training"]["scheduler_params"] = scheduler_params
        logger.info("Override: epochs = %d", epochs)

    if lr is not None:
        cfg["training"]["lr"] = lr
        logger.info("Override: lr = %e", lr)

    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
        logger.info("Override: batch_size = %d", batch_size)

    if seed is not None:
        cfg["seed"] = seed
        logger.info("Override: seed = %d", seed)

    if experiment_name is not None:
        cfg["output"]["experiment_name"] = experiment_name
        logger.info("Override: experiment_name = %s", experiment_name)

    return cfg


def _resolve_device(cfg: dict) -> str:
    """Resolve the device string from config.

    Args:
        cfg: Configuration dictionary with a 'device' key.

    Returns:
        Device string suitable for torch (e.g. 'mps', 'cuda', 'cpu').
    """
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        return str(get_device())
    return device_cfg


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def _create_experiment_dir(cfg: dict) -> Path:
    """Create the experiment output directory.

    Args:
        cfg: Configuration dictionary with output settings.

    Returns:
        Path to the created experiment directory.
    """
    base_dir = Path(cfg["output"]["experiment_dir"])
    exp_name = cfg["output"]["experiment_name"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = _PROJECT_ROOT / base_dir / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config alongside experiment artifacts
    config_save_path = exp_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Experiment directory: %s", exp_dir)

    return exp_dir


def _print_summary(
    best_dice: float,
    total_time: float,
    exp_dir: Path,
    device: str,
    cfg: dict,
) -> None:
    """Print a rich summary panel after training completes.

    Args:
        best_dice: Best validation Dice score achieved.
        total_time: Total training wall-clock time in seconds.
        exp_dir: Experiment output directory.
        device: Device used for training.
        cfg: Resolved configuration dictionary.
    """
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()

    table.add_row("Best Val Dice", f"{best_dice:.4f}")
    table.add_row("Total Time", f"{hours:02d}h {minutes:02d}m {seconds:02d}s")
    table.add_row("Epochs", str(cfg["training"]["epochs"]))
    table.add_row("Learning Rate", f"{cfg['training']['lr']:.2e}")
    table.add_row("Batch Size", str(cfg["training"]["batch_size"]))
    table.add_row("Device", device)
    table.add_row("Output", str(exp_dir))

    panel = Panel(
        table,
        title="[bold green]Training Complete[/bold green]",
        border_style="green",
        expand=False,
    )
    console.print()
    console.print(panel)


# ---------------------------------------------------------------------------
# Main CLI command
# ---------------------------------------------------------------------------


@app.command()
def train(
    config: Path = typer.Option(
        Path("configs/train_spine.yaml"),
        "--config",
        "-c",
        help="Path to YAML training configuration file.",
        exists=False,  # We handle existence check ourselves for better error msg
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Override number of training epochs.",
        min=1,
    ),
    lr: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Override learning rate.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size.",
        min=1,
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Override random seed.",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Override experiment name.",
    ),
) -> None:
    """Train a spinal cord segmentation model.

    Loads configuration from a YAML file, applies any CLI overrides,
    sets up reproducibility, and runs the training loop. Handles
    interruption gracefully by saving a checkpoint before exiting.
    """
    console.print(
        Panel(
            "[bold]CNS Spinal Cord Segmentation Training[/bold]",
            border_style="blue",
            expand=False,
        )
    )

    # Resolve config path relative to project root if not absolute
    config_path = config if config.is_absolute() else _PROJECT_ROOT / config
    logger.info("Loading config from: %s", config_path)
    cfg = _load_config(config_path)

    # Apply CLI overrides
    cfg = _apply_overrides(cfg, epochs, lr, batch_size, seed, experiment_name)

    # Reproducibility
    _set_seed(cfg.get("seed", 42))

    # Device resolution
    device = _resolve_device(cfg)
    cfg["device"] = device
    logger.info("Using device: %s", device)

    # Experiment output directory
    exp_dir = _create_experiment_dir(cfg)

    # Inject experiment dir into config so trainer uses it
    cfg["output"]["experiment_dir"] = str(exp_dir.parent)
    cfg["output"]["experiment_name"] = exp_dir.name

    # Instantiate trainer
    trainer = SegmentationTrainer(config=cfg)

    # Setup data and model before training
    console.print("\n[bold cyan]Setting up data...[/bold cyan]")
    trainer.setup_data()

    console.print("[bold cyan]Setting up model...[/bold cyan]")
    trainer.setup_model()

    # Training loop with interrupt handling
    start_time = time.time()
    best_dice = 0.0

    try:
        console.print("\n[bold cyan]Starting training...[/bold cyan]\n")
        results = trainer.train()
        best_dice = results.get("best_metric", 0.0)

    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Training interrupted by user.[/bold yellow]"
        )
        logger.warning("KeyboardInterrupt received — saving checkpoint.")
        try:
            checkpoint_path = exp_dir / "checkpoint_interrupted.pt"
            trainer.save_checkpoint(
                checkpoint_path,
                epoch=trainer.current_epoch,
                metric=trainer.best_metric,
            )
            console.print(
                f"[yellow]Checkpoint saved to:[/yellow] {checkpoint_path}"
            )
        except Exception as save_exc:
            logger.error("Failed to save interrupt checkpoint: %s", save_exc)
        best_dice = getattr(trainer, "best_metric", 0.0)

    except Exception as exc:
        logger.exception("Training failed with error: %s", exc)
        console.print(f"\n[bold red]Training failed:[/bold red] {exc}")
        raise typer.Exit(code=1)

    total_time = time.time() - start_time
    _print_summary(best_dice, total_time, exp_dir, device, cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
