"""Learning rate scheduler utilities for training.

This module provides learning rate scheduler creation utilities with:
- Linear warmup support
- Cosine annealing (with and without restarts)
- Step decay (legacy)
- Composable warmup + main scheduler via SequentialLR

Usage:
    from app.training.schedulers import create_lr_scheduler, get_warmup_scheduler

    # Modern API with warmup + cosine annealing
    scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type="cosine",
        total_epochs=100,
        warmup_epochs=5,
        lr_min=1e-6,
    )

    # Legacy API (simple LambdaLR)
    scheduler = get_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=100)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


def get_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    scheduler_type: str = 'none',
) -> Any | None:
    """
    Create a learning rate scheduler with optional warmup.

    This is the legacy warmup scheduler that uses LambdaLR for simple
    scheduling. For advanced cosine annealing, use create_lr_scheduler()
    instead.

    Args:
        optimizer: The optimizer to schedule
        warmup_epochs: Number of epochs for linear warmup (0 to disable)
        total_epochs: Total number of training epochs
        scheduler_type: Type of scheduler after warmup
            ('none', 'step', 'cosine')

    Returns:
        LR scheduler or None if no scheduling requested
    """
    if warmup_epochs == 0 and scheduler_type == 'none':
        return None

    def lr_lambda(epoch: int) -> float:
        # Linear warmup phase
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        # Post-warmup phase
        if scheduler_type == 'none':
            return 1.0
        elif scheduler_type == 'step':
            # Step decay: reduce by 0.5 every 10 epochs after warmup
            steps = (epoch - warmup_epochs) // 10
            return 0.5 ** steps
        elif scheduler_type == 'cosine':
            # Cosine annealing after warmup
            remaining = max(1, total_epochs - warmup_epochs)
            progress = (epoch - warmup_epochs) / remaining
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    lr_min: float = 1e-6,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Create a learning rate scheduler with PyTorch's native implementations.

    Supports cosine annealing with optional warmup using SequentialLR to chain
    a linear warmup scheduler with the main scheduler.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler:
            - 'none': No scheduling (returns None)
            - 'step': Step decay (legacy, uses LambdaLR)
            - 'cosine': CosineAnnealingLR to lr_min over total_epochs
            - 'cosine-warm-restarts': CosineAnnealingWarmRestarts with
              T_0, T_mult
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for linear warmup (0 to disable)
        lr_min: Minimum learning rate for cosine annealing (eta_min)
        lr_t0: T_0 parameter for CosineAnnealingWarmRestarts
            (initial restart period)
        lr_t_mult: T_mult parameter for CosineAnnealingWarmRestarts
            (period multiplier)

    Returns:
        LR scheduler or None if scheduler_type is 'none' and warmup_epochs is 0
    """
    # For legacy 'step' scheduler or 'none' with warmup, use the old function
    if scheduler_type in ('none', 'step'):
        return get_warmup_scheduler(
            optimizer, warmup_epochs, total_epochs, scheduler_type
        )

    # Create the main scheduler based on type
    if scheduler_type == 'cosine':
        # Calculate T_max: epochs for cosine annealing (after warmup)
        t_max = max(1, total_epochs - warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=lr_min
        )
    elif scheduler_type == 'cosine-warm-restarts':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=lr_t0, T_mult=lr_t_mult, eta_min=lr_min
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
        return None

    # If no warmup, return the main scheduler directly
    if warmup_epochs == 0:
        return main_scheduler

    # Create warmup scheduler using LinearLR
    # LinearLR scales the learning rate from start_factor to end_factor
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(1, warmup_epochs),  # Start at lr/warmup_epochs
        end_factor=1.0,  # End at full lr
        total_iters=warmup_epochs,
    )

    # Chain warmup and main scheduler using SequentialLR
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    return combined_scheduler


__all__ = [
    "create_lr_scheduler",
    "get_warmup_scheduler",
]
