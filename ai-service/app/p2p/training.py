"""P2P Training Coordination Utilities.

Provides utilities for coordinating training across the P2P cluster.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import GPU_POWER_RANKINGS, get_p2p_config


@dataclass
class TrainingThresholds:
    """Thresholds for triggering training.

    These values determine when training should be triggered based on
    the amount of new data available.
    """

    # Minimum games required to trigger training
    min_games: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_MIN_TRAINING_GAMES", "1000") or 1000)
    )

    # Target batch size for training
    batch_size: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_TRAINING_BATCH_SIZE", "256") or 256)
    )

    # Minimum epochs per training run
    min_epochs: int = 5

    # Maximum epochs per training run
    max_epochs: int = 50

    # Hours between training runs (rate limiting)
    min_hours_between_training: float = field(
        default_factory=lambda: float(os.environ.get("RINGRIFT_MIN_HOURS_BETWEEN_TRAINING", "1.0") or 1.0)
    )

    # Elo improvement threshold to continue training
    elo_improvement_threshold: float = 10.0

    # Maximum consecutive stagnant runs before pause
    max_stagnant_runs: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_games": self.min_games,
            "batch_size": self.batch_size,
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "min_hours_between_training": self.min_hours_between_training,
            "elo_improvement_threshold": self.elo_improvement_threshold,
            "max_stagnant_runs": self.max_stagnant_runs,
        }


def calculate_training_priority(
    gpu_type: str,
    memory_gb: float,
    current_load: float,
    has_training_data: bool = True,
) -> float:
    """Calculate training priority score for a node.

    Higher score = better candidate for training.

    Args:
        gpu_type: Type of GPU (e.g., "H100", "4090")
        memory_gb: Available memory in GB
        current_load: Current load score (0-100)
        has_training_data: Whether node has local training data

    Returns:
        Priority score (higher = better)
    """
    config = get_p2p_config()

    # Base score from GPU power
    gpu_score = config.get_gpu_priority(gpu_type)

    # Memory bonus (more memory = can handle larger batches)
    memory_score = min(memory_gb / 64, 2.0) * 100  # Max 2x bonus for 128GB+

    # Load penalty (lower load = more capacity)
    load_penalty = current_load * 2  # 0-200 penalty

    # Data locality bonus
    data_bonus = 200 if has_training_data else 0

    # Calculate final score
    priority = gpu_score + memory_score - load_penalty + data_bonus

    return max(0, priority)


def should_trigger_training(
    games_available: int,
    hours_since_last_training: float,
    last_elo_change: float = 0.0,
    consecutive_stagnant: int = 0,
    thresholds: Optional[TrainingThresholds] = None,
) -> Tuple[bool, str]:
    """Determine if training should be triggered.

    Args:
        games_available: Number of games available for training
        hours_since_last_training: Hours since last training run
        last_elo_change: Elo change from last training
        consecutive_stagnant: Number of consecutive stagnant runs
        thresholds: Training thresholds (uses defaults if None)

    Returns:
        Tuple of (should_train, reason)
    """
    if thresholds is None:
        thresholds = TrainingThresholds()

    # Check rate limit
    if hours_since_last_training < thresholds.min_hours_between_training:
        return False, f"Rate limited ({hours_since_last_training:.1f}h < {thresholds.min_hours_between_training}h)"

    # Check stagnation
    if consecutive_stagnant >= thresholds.max_stagnant_runs:
        return False, f"Training paused due to {consecutive_stagnant} stagnant runs"

    # Check game count
    if games_available < thresholds.min_games:
        return False, f"Insufficient games ({games_available} < {thresholds.min_games})"

    # All checks passed
    return True, f"Ready to train ({games_available} games available)"


def rank_training_nodes(
    nodes: List[Dict[str, Any]],
    board_type: str = "square8",
    num_players: int = 2,
) -> List[Dict[str, Any]]:
    """Rank nodes by training suitability.

    Args:
        nodes: List of node info dicts
        board_type: Board type for training
        num_players: Number of players

    Returns:
        Sorted list of nodes (best candidates first)
    """
    ranked = []

    for node in nodes:
        # Skip offline nodes
        if node.get("health", "offline") == "offline":
            continue

        # Skip nodes without GPU
        if node.get("gpu_count", 0) == 0:
            continue

        # Calculate priority
        priority = calculate_training_priority(
            gpu_type=node.get("gpu_type", "Unknown"),
            memory_gb=node.get("memory_gb_available", 0),
            current_load=node.get("load_score", 100),
            has_training_data=node.get("has_training_data", False),
        )

        ranked.append({
            **node,
            "training_priority": priority,
        })

    # Sort by priority (highest first)
    ranked.sort(key=lambda n: n["training_priority"], reverse=True)

    return ranked


def estimate_training_time(
    games: int,
    gpu_type: str,
    epochs: int = 10,
) -> float:
    """Estimate training time in hours.

    Args:
        games: Number of games to train on
        gpu_type: Type of GPU
        epochs: Number of epochs

    Returns:
        Estimated hours
    """
    # Base rate: games per hour for H100
    h100_rate = 50000  # ~50k games/hour for H100

    # Get GPU priority (proxy for speed)
    config = get_p2p_config()
    gpu_priority = config.get_gpu_priority(gpu_type)
    h100_priority = GPU_POWER_RANKINGS["H100"]

    # Scale rate by GPU power
    rate = h100_rate * (gpu_priority / h100_priority)

    # Calculate time
    hours = (games * epochs) / max(rate, 1000)

    return hours


def get_optimal_batch_size(
    memory_gb: float,
    gpu_type: str,
) -> int:
    """Get optimal batch size for given hardware.

    Args:
        memory_gb: Available GPU memory in GB
        gpu_type: Type of GPU

    Returns:
        Recommended batch size
    """
    # Base batch sizes by memory
    if memory_gb >= 80:
        base_batch = 512
    elif memory_gb >= 48:
        base_batch = 384
    elif memory_gb >= 24:
        base_batch = 256
    elif memory_gb >= 16:
        base_batch = 128
    elif memory_gb >= 8:
        base_batch = 64
    else:
        base_batch = 32

    # Adjust for GPU architecture (some GPUs prefer certain batch sizes)
    if "H100" in gpu_type or "A100" in gpu_type:
        # These GPUs prefer larger batches
        base_batch = min(base_batch * 2, 1024)
    elif "3060" in gpu_type or "4060" in gpu_type:
        # Entry-level GPUs may have memory constraints
        base_batch = min(base_batch, 128)

    return base_batch
