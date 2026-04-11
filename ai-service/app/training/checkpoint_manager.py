"""Checkpoint lifecycle helpers for the training CLI."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from app.training.checkpoint_unified import AsyncCheckpointer, save_checkpoint
from app.training.model_versioning import save_model_checkpoint
from app.training.value_calibration import CalibrationTracker

logger = logging.getLogger(__name__)


@dataclass
class CheckpointServices:
    """Optional checkpoint-adjacent services used by training."""

    async_checkpointer: AsyncCheckpointer | None
    calibration_tracker: CalibrationTracker | None


def initialize_checkpoint_services(
    *,
    config: Any,
    track_calibration: bool,
    is_main: bool,
) -> CheckpointServices:
    """Initialize async checkpointing and value calibration helpers."""
    use_async_checkpoint = bool(getattr(config, "use_async_checkpoint", True))
    async_checkpointer: AsyncCheckpointer | None = None
    if use_async_checkpoint:
        async_checkpointer = AsyncCheckpointer(max_pending=2)
        if is_main:
            logger.info("Async checkpointing enabled (non-blocking I/O)")

    calibration_tracker: CalibrationTracker | None = None
    if track_calibration:
        calibration_tracker = CalibrationTracker(window_size=5000)
        if is_main:
            logger.info("Value calibration tracking enabled")

    return CheckpointServices(
        async_checkpointer=async_checkpointer,
        calibration_tracker=calibration_tracker,
    )


def save_early_stop_artifacts(
    *,
    model_to_save: nn.Module,
    optimizer: Any,
    epoch: int,
    checkpoint_dir: str,
    save_path: str,
    config: Any,
    num_players: int,
    early_stopper: Any,
    async_checkpointer: AsyncCheckpointer | None,
    epoch_scheduler: Any,
) -> str:
    """Persist the best model when early stopping fires."""
    final_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_early_stop_epoch_{epoch + 1}.pth",
    )
    if async_checkpointer is not None:
        async_checkpointer.save_async(
            model_to_save,
            optimizer,
            epoch,
            early_stopper.best_loss,
            final_checkpoint_path,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
        )
    else:
        save_checkpoint(
            model_to_save,
            optimizer,
            epoch,
            early_stopper.best_loss,
            final_checkpoint_path,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
        )

    save_model_checkpoint(
        model_to_save,
        save_path,
        training_info={
            "epoch": epoch,
            "loss": float(early_stopper.best_loss),
            "early_stopped": True,
        },
        board_type=config.board_type,
        num_players=num_players,
    )
    logger.info("Best model saved to %s", save_path)
    return final_checkpoint_path


def save_periodic_checkpoint(
    *,
    model_to_save: nn.Module,
    optimizer: Any,
    epoch: int,
    avg_val_loss: float,
    checkpoint_dir: str,
    async_checkpointer: AsyncCheckpointer | None,
    epoch_scheduler: Any,
    early_stopper: Any,
) -> str:
    """Save the regular interval checkpoint for a training epoch."""
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_epoch_{epoch + 1}.pth",
    )
    if async_checkpointer is not None:
        async_checkpointer.save_async(
            model_to_save,
            optimizer,
            epoch,
            avg_val_loss,
            checkpoint_path,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
        )
    else:
        save_checkpoint(
            model_to_save,
            optimizer,
            epoch,
            avg_val_loss,
            checkpoint_path,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
        )
    return checkpoint_path


def save_best_model_artifacts(
    *,
    model_to_save: nn.Module,
    save_path: str,
    config: Any,
    num_players: int,
    epoch: int,
    train_size: int,
    avg_val_loss: float,
    avg_train_loss: float,
    checkpoint_averager: Any,
) -> None:
    """Save the best model and a timestamped versioned checkpoint."""
    save_model_checkpoint(
        model_to_save,
        save_path,
        training_info={
            "epoch": epoch + 1,
            "samples_seen": train_size * (epoch + 1),
            "val_loss": float(avg_val_loss),
            "train_loss": float(avg_train_loss),
        },
        board_type=config.board_type,
        num_players=num_players,
    )
    logger.info("  New best model saved (Val Loss: %.4f)", avg_val_loss)

    if checkpoint_averager is not None:
        checkpoint_averager.add_checkpoint(
            model_to_save.state_dict(),
            epoch=epoch,
        )

    from datetime import datetime as dt

    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    version_path = save_path.replace(".pth", f"_{timestamp}.pth")
    save_model_checkpoint(
        model_to_save,
        version_path,
        training_info={
            "epoch": epoch + 1,
            "samples_seen": train_size * (epoch + 1),
            "val_loss": float(avg_val_loss),
            "train_loss": float(avg_train_loss),
            "timestamp": timestamp,
        },
        board_type=config.board_type,
        num_players=num_players,
    )
    logger.info("  Versioned checkpoint saved: %s", version_path)


def finalize_training_checkpoints(
    *,
    model_to_save_final: nn.Module,
    optimizer: Any,
    config: Any,
    checkpoint_dir: str,
    save_path: str,
    num_players: int,
    avg_val_loss: float,
    best_val_loss: float,
    best_train_loss_at_best_val: float,
    overfit_divergence_threshold: float,
    prefer_best_on_overfit: bool,
    early_stopper: Any,
    checkpoint_averager: Any,
    async_checkpointer: AsyncCheckpointer | None,
    epoch_scheduler: Any,
) -> str:
    """Persist the final checkpoint and handle checkpoint averaging."""
    final_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_final_epoch_{config.epochs_per_iter}.pth",
    )
    if async_checkpointer is not None:
        async_checkpointer.save_async(
            model_to_save_final,
            optimizer,
            config.epochs_per_iter - 1,
            avg_val_loss,
            final_checkpoint_path,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
        )
    else:
        save_checkpoint(
            model_to_save_final,
            optimizer,
            config.epochs_per_iter - 1,
            avg_val_loss,
            final_checkpoint_path,
            scheduler=epoch_scheduler,
            early_stopping=early_stopper,
        )
    logger.info("Training completed. Final checkpoint saved.")

    overfitting_detected = False
    if best_train_loss_at_best_val > 0 and best_val_loss != float("inf"):
        divergence = (best_val_loss - best_train_loss_at_best_val) / best_train_loss_at_best_val
        if divergence > overfit_divergence_threshold:
            overfitting_detected = True
            logger.warning(
                "[Overfitting Detected] Val/Train divergence: %.1f%% > %.0f%% threshold "
                "(train=%.4f, val=%.4f)",
                divergence * 100,
                overfit_divergence_threshold * 100,
                best_train_loss_at_best_val,
                best_val_loss,
            )

    skip_averaging = prefer_best_on_overfit and overfitting_detected
    if skip_averaging:
        logger.info(
            "[Best Checkpoint Selection] Keeping best validation loss checkpoint (skipping averaging due to overfitting)"
        )
        if early_stopper is not None and hasattr(early_stopper, "restore_best_weights"):
            logger.info(
                "[Auto-Restore Best] Restoring weights from best epoch (val_loss=%.4f)",
                early_stopper.best_loss,
            )
            early_stopper.restore_best_weights(model_to_save_final)
        if checkpoint_averager is not None:
            checkpoint_averager.cleanup()
    elif checkpoint_averager is not None and checkpoint_averager.num_stored >= 2:
        logger.info(
            "[Checkpoint Averaging] Averaging %d checkpoints...",
            checkpoint_averager.num_stored,
        )
        try:
            averaged_state_dict = checkpoint_averager.get_averaged_state_dict()
            averaged_path = save_path.replace(".pth", "_averaged.pth")
            model_to_save_final.load_state_dict(averaged_state_dict)
            checkpoint_info = {
                "epoch": config.epochs_per_iter,
                "averaged_checkpoints": checkpoint_averager.num_stored,
                "checkpoint_averaging": True,
            }
            save_model_checkpoint(
                model_to_save_final,
                averaged_path,
                training_info=checkpoint_info,
                board_type=config.board_type,
                num_players=num_players,
            )
            save_model_checkpoint(
                model_to_save_final,
                save_path,
                training_info=checkpoint_info,
                board_type=config.board_type,
                num_players=num_players,
            )
            logger.info(
                "[Checkpoint Averaging] Saved averaged model (%d checkpoints) to %s",
                checkpoint_averager.num_stored,
                save_path,
            )
        except (OSError, RuntimeError, ValueError, TypeError, MemoryError) as exc:
            logger.warning("[Checkpoint Averaging] Failed to average checkpoints: %s", exc)
        finally:
            checkpoint_averager.cleanup()
    elif checkpoint_averager is not None:
        logger.info(
            "[Checkpoint Averaging] Skipped: only %d checkpoint(s) available (need >= 2)",
            checkpoint_averager.num_stored,
        )
        checkpoint_averager.cleanup()

    if early_stopper is not None and not skip_averaging:
        if avg_val_loss > early_stopper.best_loss * 1.05:
            logger.info(
                "[Auto-Restore Best] Final loss (%.4f) > best loss (%.4f). Restoring best weights before final save.",
                avg_val_loss,
                early_stopper.best_loss,
            )
            early_stopper.restore_best_weights(model_to_save_final)
            save_model_checkpoint(
                model_to_save_final,
                save_path,
                training_info={
                    "epoch": (
                        early_stopper.best_epoch
                        if hasattr(early_stopper, "best_epoch")
                        else config.epochs_per_iter
                    ),
                    "best_val_loss": early_stopper.best_loss,
                    "auto_restored": True,
                },
                board_type=config.board_type,
                num_players=num_players,
            )

    return final_checkpoint_path


__all__ = [
    "CheckpointServices",
    "finalize_training_checkpoints",
    "initialize_checkpoint_services",
    "save_best_model_artifacts",
    "save_early_stop_artifacts",
    "save_periodic_checkpoint",
]
