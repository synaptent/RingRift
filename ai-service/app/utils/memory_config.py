"""Memory configuration for training and inference limits."""

import os
from dataclasses import dataclass

import psutil

__all__ = [
    "MemoryConfig",
]


@dataclass
class MemoryConfig:
    """Configuration for memory limits during training and inference."""

    max_memory_gb: float = 16.0
    training_allocation: float = 0.60  # 60% for training buffers
    inference_allocation: float = 0.30  # 30% for inference (search)
    system_reserve: float = 0.10  # 10% reserved for system

    def get_training_limit_bytes(self) -> int:
        """Get the maximum bytes available for training buffers."""
        return int(self.max_memory_gb * self.training_allocation * 1024**3)

    def get_inference_limit_bytes(self) -> int:
        """Get the maximum bytes available for inference (search)."""
        return int(self.max_memory_gb * self.inference_allocation * 1024**3)

    def get_transposition_table_limit_bytes(self) -> int:
        """Get the maximum bytes for transposition tables.

        Half of inference allocation for transposition tables.
        """
        return self.get_inference_limit_bytes() // 2

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables."""
        max_gb = float(os.environ.get("RINGRIFT_MAX_MEMORY_GB", "16.0"))
        return cls(max_memory_gb=max_gb)

    def check_available_memory(self) -> tuple[bool, float]:
        """Check if enough memory is available.

        Returns:
            tuple[bool, float]: (ok, available_gb) where ok is True if
                enough memory is available for the system reserve.
        """
        available = psutil.virtual_memory().available / (1024**3)
        min_required = self.max_memory_gb * self.system_reserve
        return (available >= min_required, available)
