"""Unified metrics logging for RingRift AI training.

This module provides a backend-agnostic metrics logger that supports:
- TensorBoard (default, always available)
- Weights & Biases (optional, requires `pip install wandb`)
- Console logging (fallback)

Usage:
    from app.training.metrics_logger import MetricsLogger

    # Initialize with desired backends
    logger = MetricsLogger(
        experiment_name="square8_2p_training",
        backends=["tensorboard", "wandb"],  # or just ["tensorboard"]
        log_dir="logs/training",
        config={"board_type": "square8", "batch_size": 256},
    )

    # Log metrics
    logger.log_scalar("train/loss", 0.5, step=100)
    logger.log_scalars("accuracy", {"train": 0.8, "val": 0.75}, step=100)

    # Log at end
    logger.finish()
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsBackend(ABC):
    """Abstract base class for metrics logging backends."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""

    @abstractmethod
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        """Log multiple related scalar values."""

    @abstractmethod
    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram of values."""

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text content."""

    @abstractmethod
    def finish(self) -> None:
        """Finalize logging and clean up resources."""


class TensorBoardBackend(MetricsBackend):
    """TensorBoard logging backend using tensorboardX."""

    def __init__(self, log_dir: str, experiment_name: str):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                "tensorboardX is required for TensorBoard logging. "
                "Install with: pip install tensorboardX"
            )

        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_path))
        logger.info(f"TensorBoard logging to: {self.log_path}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)

    def finish(self) -> None:
        self.writer.close()
        logger.info(f"TensorBoard logs saved to: {self.log_path}")


class WandBBackend(MetricsBackend):
    """Weights & Biases logging backend."""

    def __init__(
        self,
        experiment_name: str,
        project: str = "ringrift-ai",
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb is required for W&B logging. "
                "Install with: pip install wandb"
            )

        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config or {},
            tags=tags or [],
            reinit=True,
        )
        logger.info(f"W&B run initialized: {self.run.url}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        prefixed = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
        self.wandb.log(prefixed, step=step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        self.wandb.log({tag: self.wandb.Histogram(values)}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        # W&B doesn't have native text logging, use a table
        self.wandb.log({tag: text}, step=step)

    def finish(self) -> None:
        self.wandb.finish()
        logger.info("W&B run finished")


class ConsoleBackend(MetricsBackend):
    """Simple console logging backend (fallback)."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        logger.info(f"Console logging for experiment: {experiment_name}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        print(f"[{self.experiment_name}] Step {step}: {tag} = {value:.6f}")

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        values_str = ", ".join(f"{k}={v:.6f}" for k, v in tag_scalar_dict.items())
        print(f"[{self.experiment_name}] Step {step}: {main_tag} {{ {values_str} }}")

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        import numpy as np
        arr = np.asarray(values)
        print(f"[{self.experiment_name}] Step {step}: {tag} histogram "
              f"(min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f})")

    def log_text(self, tag: str, text: str, step: int) -> None:
        print(f"[{self.experiment_name}] Step {step}: {tag} = {text[:100]}...")

    def finish(self) -> None:
        logger.info(f"Console logging complete for: {self.experiment_name}")


class JSONFileBackend(MetricsBackend):
    """JSON file logging backend for offline analysis."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_path = Path(log_dir) / experiment_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_path / "metrics.jsonl"
        self.metrics: list[dict[str, Any]] = []
        logger.info(f"JSON logging to: {self.metrics_file}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        entry = {"step": step, "tag": tag, "value": value, "type": "scalar"}
        self._write_entry(entry)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        entry = {
            "step": step,
            "tag": main_tag,
            "values": tag_scalar_dict,
            "type": "scalars"
        }
        self._write_entry(entry)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        import numpy as np
        arr = np.asarray(values)
        entry = {
            "step": step,
            "tag": tag,
            "stats": {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            },
            "type": "histogram"
        }
        self._write_entry(entry)

    def log_text(self, tag: str, text: str, step: int) -> None:
        entry = {"step": step, "tag": tag, "text": text, "type": "text"}
        self._write_entry(entry)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def finish(self) -> None:
        logger.info(f"JSON metrics saved to: {self.metrics_file}")


class MetricsLogger:
    """Unified metrics logger supporting multiple backends.

    This class provides a single interface for logging metrics to multiple
    backends simultaneously (TensorBoard, W&B, console, JSON).

    Example:
        logger = MetricsLogger(
            experiment_name="nn_training_v1",
            backends=["tensorboard", "wandb"],
            log_dir="logs/training",
            config={"learning_rate": 0.001, "batch_size": 256},
        )

        for epoch in range(100):
            logger.log_scalar("train/loss", train_loss, step=epoch)
            logger.log_scalar("val/loss", val_loss, step=epoch)
            logger.log_scalars("accuracy", {"train": 0.8, "val": 0.75}, step=epoch)

        logger.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        backends: list[str] | None = None,
        log_dir: str = "logs/training",
        config: dict[str, Any] | None = None,
        wandb_project: str = "ringrift-ai",
        wandb_tags: list[str] | None = None,
    ):
        """Initialize the metrics logger.

        Args:
            experiment_name: Name for this experiment/run
            backends: List of backends to use. Options: "tensorboard", "wandb",
                     "console", "json". Default: ["tensorboard", "json"]
            log_dir: Directory for log files
            config: Configuration dict to log (hyperparameters, etc.)
            wandb_project: W&B project name (only used if "wandb" in backends)
            wandb_tags: Tags for W&B run
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        self.backends: list[MetricsBackend] = []

        if backends is None:
            backends = ["tensorboard", "json"]

        # Initialize requested backends
        for backend_name in backends:
            try:
                if backend_name == "tensorboard":
                    self.backends.append(TensorBoardBackend(log_dir, experiment_name))
                elif backend_name == "wandb":
                    self.backends.append(WandBBackend(
                        experiment_name=experiment_name,
                        project=wandb_project,
                        config=config,
                        tags=wandb_tags,
                    ))
                elif backend_name == "console":
                    self.backends.append(ConsoleBackend(experiment_name))
                elif backend_name == "json":
                    self.backends.append(JSONFileBackend(log_dir, experiment_name))
                else:
                    logger.warning(f"Unknown backend: {backend_name}")
            except ImportError as e:
                logger.warning(f"Could not initialize {backend_name} backend: {e}")
                # Fall back to console if primary backend fails
                if backend_name in ["tensorboard", "wandb"] and not self.backends:
                    logger.info("Falling back to console logging")
                    self.backends.append(ConsoleBackend(experiment_name))

        if not self.backends:
            logger.warning("No backends initialized, using console fallback")
            self.backends.append(ConsoleBackend(experiment_name))

        logger.info(f"MetricsLogger initialized with {len(self.backends)} backend(s)")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value to all backends."""
        for backend in self.backends:
            try:
                backend.log_scalar(tag, value, step)
            except Exception as e:
                logger.warning(f"Error logging to {type(backend).__name__}: {e}")

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        """Log multiple related scalar values to all backends."""
        for backend in self.backends:
            try:
                backend.log_scalars(main_tag, tag_scalar_dict, step)
            except Exception as e:
                logger.warning(f"Error logging to {type(backend).__name__}: {e}")

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """Log a histogram of values to all backends."""
        for backend in self.backends:
            try:
                backend.log_histogram(tag, values, step)
            except Exception as e:
                logger.warning(f"Error logging to {type(backend).__name__}: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text content to all backends."""
        for backend in self.backends:
            try:
                backend.log_text(tag, text, step)
            except Exception as e:
                logger.warning(f"Error logging to {type(backend).__name__}: {e}")

    def log_config(self, config: dict[str, Any]) -> None:
        """Log configuration/hyperparameters."""
        self.config.update(config)
        config_str = json.dumps(config, indent=2)
        self.log_text("config", config_str, step=0)

    def log_model_summary(self, model: Any, step: int = 0) -> None:
        """Log model architecture summary."""
        try:
            import torch.nn as nn
            if isinstance(model, nn.Module):
                param_count = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                summary = f"Parameters: {param_count:,} (trainable: {trainable:,})"
                self.log_text("model/summary", summary, step=step)
                self.log_scalar("model/parameters", param_count, step=step)
                self.log_scalar("model/trainable_parameters", trainable, step=step)
        except Exception as e:
            logger.warning(f"Could not log model summary: {e}")

    def finish(self) -> None:
        """Finalize all backends and clean up resources."""
        for backend in self.backends:
            try:
                backend.finish()
            except Exception as e:
                logger.warning(f"Error finishing {type(backend).__name__}: {e}")


def create_training_logger(
    board_type: str,
    num_players: int,
    model_version: str = "v1",
    use_wandb: bool = False,
    log_dir: str = "logs/training",
    **extra_config: Any,
) -> MetricsLogger:
    """Convenience function to create a training logger with standard config.

    Args:
        board_type: Board type (square8, square19, hex)
        num_players: Number of players
        model_version: Model version string
        use_wandb: Whether to enable W&B logging
        log_dir: Directory for log files
        **extra_config: Additional config parameters to log

    Returns:
        Configured MetricsLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{board_type}_{num_players}p_{model_version}_{timestamp}"

    backends = ["tensorboard", "json"]
    if use_wandb:
        backends.append("wandb")

    config = {
        "board_type": board_type,
        "num_players": num_players,
        "model_version": model_version,
        **extra_config,
    }

    return MetricsLogger(
        experiment_name=experiment_name,
        backends=backends,
        log_dir=log_dir,
        config=config,
        wandb_tags=[board_type, f"{num_players}p", model_version],
    )
