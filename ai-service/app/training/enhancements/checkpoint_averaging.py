"""
Checkpoint Averaging for RingRift AI Training.

This module provides utilities for averaging model weights from multiple checkpoints
to improve final model performance.

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path

import torch

from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)


class CheckpointAverager:
    """
    Averages model weights from multiple checkpoints for improved performance.

    Checkpoint averaging typically provides +10-20 Elo improvement by reducing
    variance in the final model weights.

    Usage:
        averager = CheckpointAverager(num_checkpoints=5)

        # During training, save checkpoints
        for epoch in range(epochs):
            train_epoch()
            averager.add_checkpoint(model.state_dict())

        # Get averaged weights
        averaged_state = averager.get_averaged_state_dict()
        model.load_state_dict(averaged_state)
    """

    def __init__(
        self,
        num_checkpoints: int = 5,
        checkpoint_dir: Path | None = None,
        keep_on_disk: bool = False,
    ):
        """
        Args:
            num_checkpoints: Number of recent checkpoints to average
            checkpoint_dir: Directory to save checkpoints (if keep_on_disk=True)
            keep_on_disk: Save checkpoints to disk to save memory
        """
        self.num_checkpoints = num_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.keep_on_disk = keep_on_disk
        self._checkpoints: deque = deque(maxlen=num_checkpoints)
        self._checkpoint_paths: deque = deque(maxlen=num_checkpoints)

        if keep_on_disk and checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
        epoch: int | None = None,
    ) -> None:
        """Add a checkpoint to the averaging queue."""
        if self.keep_on_disk and self.checkpoint_dir:
            # Save to disk
            path = self.checkpoint_dir / f"avg_ckpt_{epoch or len(self._checkpoint_paths)}.pt"
            torch.save(state_dict, path)

            # Remove old file if queue is full
            if len(self._checkpoint_paths) == self.num_checkpoints:
                old_path = self._checkpoint_paths[0]
                if old_path.exists():
                    old_path.unlink()

            self._checkpoint_paths.append(path)
        else:
            # Keep in memory (deep copy to avoid reference issues)
            self._checkpoints.append(copy.deepcopy(state_dict))

    def get_averaged_state_dict(self) -> dict[str, torch.Tensor]:
        """
        Compute the average of all stored checkpoints.

        Returns:
            Averaged state dict
        """
        if self.keep_on_disk:
            checkpoints = [
                safe_load_checkpoint(p, warn_on_unsafe=False)
                for p in self._checkpoint_paths
                if p.exists()
            ]
        else:
            checkpoints = list(self._checkpoints)

        if not checkpoints:
            raise ValueError("No checkpoints available for averaging")

        # Initialize with first checkpoint
        averaged = {}
        for key in checkpoints[0]:
            averaged[key] = checkpoints[0][key].clone().float()

        # Add remaining checkpoints
        for ckpt in checkpoints[1:]:
            for key in averaged:
                averaged[key] += ckpt[key].float()

        # Divide by number of checkpoints
        num_ckpts = len(checkpoints)
        for key in averaged:
            averaged[key] /= num_ckpts
            # Restore original dtype
            averaged[key] = averaged[key].to(checkpoints[0][key].dtype)

        logger.info(f"Averaged {num_ckpts} checkpoints")
        return averaged

    def cleanup(self) -> None:
        """Remove checkpoint files from disk."""
        if self.keep_on_disk:
            for path in self._checkpoint_paths:
                if path.exists():
                    path.unlink()
            self._checkpoint_paths.clear()
        self._checkpoints.clear()

    @property
    def num_stored(self) -> int:
        """Number of checkpoints currently stored."""
        if self.keep_on_disk:
            return len(self._checkpoint_paths)
        return len(self._checkpoints)


def average_checkpoints(
    checkpoint_paths: list[str | Path],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """
    Average model weights from multiple checkpoint files.

    Args:
        checkpoint_paths: List of paths to checkpoint files
        device: Device to load checkpoints to

    Returns:
        Averaged state dict
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    averaged = None
    num_ckpts = 0

    for path in checkpoint_paths:
        ckpt = safe_load_checkpoint(path, map_location=device, warn_on_unsafe=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        if averaged is None:
            averaged = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in averaged:
                averaged[k] += state_dict[k].float()
        num_ckpts += 1

    for k in averaged:
        averaged[k] /= num_ckpts

    return averaged
