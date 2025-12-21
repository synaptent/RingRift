"""
Shared seeding utilities for training and evaluation code.

This module centralises seeding of Python's ``random`` module, NumPy and
PyTorch so that training/evaluation jobs can enable reproducible runs
from a single integer seed.

Lane 3 Consolidation (2025-12):
    - Thread-safe seeding via threading.local()
    - Per-AI-instance seeding helpers
    - Distributed training worker seed derivation
    - Deterministic replay seed computation
"""

from __future__ import annotations

import hashlib
import os
import random
import threading

import numpy as np
import torch

# Thread-local storage for per-thread RNG state
_thread_local = threading.local()

# Global seed tracking for debugging
_global_seed: int | None = None
_seed_lock = threading.Lock()


def seed_all(seed: int, *, enable_cudnn_determinism: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible experiments.

    Args:
        seed: Integer seed to use for all RNGs.
        enable_cudnn_determinism: When True (default), configure cuDNN
            for deterministic behaviour at the cost of some performance.
    """
    global _global_seed
    with _seed_lock:
        _global_seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if enable_cudnn_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_thread_rng(seed: int | None = None) -> random.Random:
    """Get a thread-local Random instance for thread-safe randomness.

    Args:
        seed: Optional seed for the thread-local RNG. If None, uses
            a seed derived from the global seed and thread ID.

    Returns:
        A random.Random instance local to the current thread.
    """
    if not hasattr(_thread_local, "rng"):
        if seed is None:
            # Derive seed from global seed and thread ID
            thread_id = threading.current_thread().ident or 0
            base_seed = _global_seed if _global_seed is not None else 42
            seed = hash((base_seed, thread_id)) & 0xFFFFFFFF
        _thread_local.rng = random.Random(seed)
    return _thread_local.rng


def derive_worker_seed(base_seed: int, worker_id: int, rank: int = 0) -> int:
    """Derive a unique seed for a distributed training worker.

    Args:
        base_seed: The base experiment seed.
        worker_id: Data loader worker ID (0 to num_workers-1).
        rank: Distributed training rank (0 for single-GPU).

    Returns:
        A unique seed for this worker.
    """
    # Combine base_seed, worker_id, and rank to get unique seed
    combined = f"{base_seed}:{rank}:{worker_id}"
    h = hashlib.md5(combined.encode()).hexdigest()
    return int(h[:8], 16)


def derive_ai_seed(base_seed: int, ai_type: str, player_id: int, game_id: str = "") -> int:
    """Derive a deterministic seed for an AI instance.

    Use this to ensure AI instances get reproducible randomness while
    still being unique per game and player.

    Args:
        base_seed: The base experiment seed.
        ai_type: AI type string (e.g., "mcts", "gumbel_mcts", "random").
        player_id: Player number (1-4).
        game_id: Optional game identifier for per-game variation.

    Returns:
        A deterministic seed for this AI instance.
    """
    combined = f"{base_seed}:{ai_type}:{player_id}:{game_id}"
    h = hashlib.md5(combined.encode()).hexdigest()
    return int(h[:8], 16)


def derive_replay_seed(game_id: str, move_index: int = 0) -> int:
    """Derive a deterministic seed for replaying a specific game state.

    Args:
        game_id: The game identifier (UUID or similar).
        move_index: The move index for state-specific seeding.

    Returns:
        A deterministic seed for replaying this game state.
    """
    combined = f"replay:{game_id}:{move_index}"
    h = hashlib.md5(combined.encode()).hexdigest()
    return int(h[:8], 16)


def get_env_seed(default: int = 42) -> int:
    """Get seed from RINGRIFT_SEED environment variable.

    Args:
        default: Default seed if env var not set.

    Returns:
        Seed value from environment or default.
    """
    env_seed = os.environ.get("RINGRIFT_SEED")
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError:
            pass
    return default


def reset_thread_rng() -> None:
    """Reset the thread-local RNG.

    Call this between independent experiments to ensure clean state.
    """
    if hasattr(_thread_local, "rng"):
        delattr(_thread_local, "rng")


def get_global_seed() -> int | None:
    """Get the most recently set global seed.

    Returns:
        The seed passed to the last seed_all() call, or None if never called.
    """
    return _global_seed
