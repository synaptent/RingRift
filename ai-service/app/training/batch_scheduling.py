"""Batch Size Scheduling for RingRift AI Training.

Dynamically adjusts batch size during training:
- Small batches early (high gradient noise for exploration)
- Large batches later (stability for convergence)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchScheduleConfig:
    """Configuration for batch size scheduling."""
    initial_batch_size: int = 64
    final_batch_size: int = 512
    warmup_steps: int = 1000
    rampup_steps: int = 10000
    schedule_type: str = "linear"  # "linear", "exponential", "step"
    # For step schedule
    step_milestones: tuple = (5000, 10000, 20000)
    step_batch_sizes: tuple = (64, 128, 256, 512)


class BatchSizeScheduler:
    """Schedules batch size throughout training."""

    def __init__(self, config: Optional[BatchScheduleConfig] = None):
        """Initialize batch size scheduler.

        Args:
            config: Schedule configuration
        """
        self.config = config or BatchScheduleConfig()
        self._step = 0

    def get_batch_size(self, step: Optional[int] = None) -> int:
        """Get batch size for given step.

        Args:
            step: Training step (uses internal counter if None)

        Returns:
            Batch size for this step
        """
        if step is None:
            step = self._step

        config = self.config

        # Warmup period - use initial batch size
        if step < config.warmup_steps:
            return config.initial_batch_size

        # Calculate progress through rampup
        rampup_step = step - config.warmup_steps
        progress = min(1.0, rampup_step / max(1, config.rampup_steps))

        if config.schedule_type == "linear":
            batch_size = config.initial_batch_size + progress * (
                config.final_batch_size - config.initial_batch_size
            )

        elif config.schedule_type == "exponential":
            ratio = config.final_batch_size / config.initial_batch_size
            batch_size = config.initial_batch_size * (ratio ** progress)

        elif config.schedule_type == "step":
            batch_size = config.step_batch_sizes[0]
            for i, milestone in enumerate(config.step_milestones):
                if step >= milestone and i + 1 < len(config.step_batch_sizes):
                    batch_size = config.step_batch_sizes[i + 1]

        else:
            batch_size = config.initial_batch_size

        # Round to nearest power of 2 for efficiency
        return self._round_to_power_of_2(int(batch_size))

    def _round_to_power_of_2(self, n: int) -> int:
        """Round to nearest power of 2."""
        if n < 1:
            return 1
        # Find nearest power of 2
        lower = 2 ** int(math.log2(n))
        upper = lower * 2
        return lower if (n - lower) < (upper - n) else upper

    def step(self) -> int:
        """Advance step and return current batch size."""
        batch_size = self.get_batch_size()
        self._step += 1
        return batch_size

    def set_step(self, step: int):
        """Set current step."""
        self._step = step

    @property
    def current_step(self) -> int:
        return self._step

    def get_schedule_info(self) -> Dict[str, Any]:
        """Get current schedule information."""
        return {
            "current_step": self._step,
            "current_batch_size": self.get_batch_size(),
            "initial_batch_size": self.config.initial_batch_size,
            "final_batch_size": self.config.final_batch_size,
            "schedule_type": self.config.schedule_type,
        }


def create_batch_scheduler(
    initial: int = 64,
    final: int = 512,
    rampup_steps: int = 10000,
    schedule_type: str = "linear",
) -> BatchSizeScheduler:
    """Create a batch size scheduler."""
    config = BatchScheduleConfig(
        initial_batch_size=initial,
        final_batch_size=final,
        rampup_steps=rampup_steps,
        schedule_type=schedule_type,
    )
    return BatchSizeScheduler(config)
