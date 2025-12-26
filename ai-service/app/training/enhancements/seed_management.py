"""
Seed Management for Reproducible Training.

Provides utilities for managing random seeds across:
- Python's random module
- NumPy's random generator
- PyTorch (CPU and CUDA)
- Optional: CuDNN determinism

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SeedManager:
    """
    Manages random seeds for reproducible training.

    Handles seeding for:
    - Python's random module
    - NumPy's random generator
    - PyTorch (CPU and CUDA)
    - Optional: CuDNN determinism

    Usage:
        # Set global seed for reproducibility
        seed_manager = SeedManager(seed=42)
        seed_manager.set_global_seed()

        # Get worker init function for DataLoader
        dataloader = DataLoader(
            dataset,
            worker_init_fn=seed_manager.get_worker_init_fn(),
        )

        # Log seed info for experiment tracking
        print(seed_manager.get_seed_info())
    """

    def __init__(
        self,
        seed: int | None = None,
        deterministic: bool = False,
        benchmark: bool = True,
    ):
        """
        Args:
            seed: Random seed (None for random seed)
            deterministic: Enable CuDNN deterministic mode (slower but reproducible)
            benchmark: Enable CuDNN benchmark mode (faster but non-deterministic)
        """
        self.seed = seed if seed is not None else self._generate_seed()
        self.deterministic = deterministic
        self.benchmark = benchmark

        self._initial_seed = self.seed
        self._seed_history: list[dict[str, Any]] = []

    def _generate_seed(self) -> int:
        """Generate a random seed."""
        import random
        return random.randint(0, 2**32 - 1)

    def set_global_seed(self) -> None:
        """Set seed for all random number generators."""
        import random

        # Python random
        random.seed(self.seed)

        # NumPy
        np.random.seed(self.seed)

        # PyTorch CPU
        torch.manual_seed(self.seed)

        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            # CuDNN settings
            torch.backends.cudnn.deterministic = self.deterministic
            torch.backends.cudnn.benchmark = self.benchmark and not self.deterministic

        # Record
        self._seed_history.append({
            'seed': self.seed,
            'timestamp': time.time(),
            'action': 'set_global_seed',
        })

        logger.info(
            f"Set global seed: {self.seed} "
            f"(deterministic={self.deterministic}, benchmark={self.benchmark})"
        )

    def get_worker_init_fn(self) -> Callable[[int], None]:
        """
        Get worker initialization function for DataLoader.

        Each worker gets a unique but reproducible seed based on the
        global seed and worker ID.

        Returns:
            Worker init function for DataLoader
        """
        base_seed = self.seed

        def worker_init_fn(worker_id: int) -> None:
            import random
            worker_seed = base_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return worker_init_fn

    def get_generator(self, offset: int = 0) -> torch.Generator:
        """
        Get a PyTorch Generator with reproducible seed.

        Args:
            offset: Offset to add to base seed (for different generators)

        Returns:
            Seeded torch.Generator
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + offset)
        return generator

    def fork(self, offset: int = 1) -> SeedManager:
        """
        Create a new SeedManager with an offset seed.

        Useful for creating reproducible but different seeds for
        different components (e.g., data augmentation vs dropout).

        Args:
            offset: Offset to add to base seed

        Returns:
            New SeedManager with offset seed
        """
        return SeedManager(
            seed=self.seed + offset,
            deterministic=self.deterministic,
            benchmark=self.benchmark,
        )

    def advance(self, steps: int = 1) -> None:
        """
        Advance the seed by a number of steps.

        Useful for resuming training with a different seed progression.

        Args:
            steps: Number of steps to advance
        """
        self.seed = (self.seed + steps) % (2**32)
        self._seed_history.append({
            'seed': self.seed,
            'timestamp': time.time(),
            'action': f'advance({steps})',
        })

    def get_seed_info(self) -> dict[str, Any]:
        """Get seed information for experiment tracking."""
        return {
            'initial_seed': self._initial_seed,
            'current_seed': self.seed,
            'deterministic': self.deterministic,
            'benchmark': self.benchmark,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__,
        }

    def save_state(self) -> dict[str, Any]:
        """Save RNG states for checkpointing."""
        import random

        state = {
            'seed': self.seed,
            'initial_seed': self._initial_seed,
            'python_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'seed_history': self._seed_history,
        }

        if torch.cuda.is_available():
            state['cuda_state'] = torch.cuda.get_rng_state_all()

        return state

    def load_state(self, state: dict[str, Any]) -> None:
        """Load RNG states from checkpoint."""
        import random

        self.seed = state['seed']
        self._initial_seed = state['initial_seed']
        self._seed_history = state.get('seed_history', [])

        random.setstate(state['python_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])

        if torch.cuda.is_available() and 'cuda_state' in state:
            torch.cuda.set_rng_state_all(state['cuda_state'])

        logger.info(f"Loaded RNG state from checkpoint (seed={self.seed})")

    def __repr__(self) -> str:
        return (
            f"SeedManager(seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"benchmark={self.benchmark})"
        )


def set_reproducible_seed(seed: int, deterministic: bool = True) -> SeedManager:
    """
    Convenience function to set a reproducible seed.

    Args:
        seed: Random seed to use
        deterministic: Enable full determinism (slower)

    Returns:
        SeedManager instance
    """
    manager = SeedManager(seed=seed, deterministic=deterministic, benchmark=False)
    manager.set_global_seed()
    return manager
