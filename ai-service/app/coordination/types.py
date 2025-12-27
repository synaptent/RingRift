"""Canonical types for coordination modules.

This module consolidates commonly used types across the coordination package:

- BackpressureLevel: For queue and resource backpressure
- TaskType: Types of compute tasks
- Re-exports from app.core.node: NodeRole, NodeState, NodeHealth, Provider

Usage:
    from app.coordination.types import (
        BackpressureLevel,
        TaskType,
        NodeRole,
        NodeState,
    )

Migration:
    # Old imports (deprecated)
    from app.coordination.queue_monitor import BackpressureLevel
    from app.coordination.resource_monitoring_coordinator import BackpressureLevel
    from app.coordination.unified_resource_coordinator import BackpressureLevel, TaskType

    # New canonical imports
    from app.coordination.types import BackpressureLevel, TaskType

December 2025: Created to consolidate duplicate enum definitions.
"""

from __future__ import annotations

from enum import Enum

# Re-export core node types for convenience
from app.core.node import (
    GPUInfo,
    NodeHealth,
    NodeRole,
    NodeState,
    Provider,
)

__all__ = [
    # Core node types (re-exported)
    "GPUInfo",
    "NodeHealth",
    "NodeRole",
    "NodeState",
    "Provider",
    # Coordination types
    "BackpressureLevel",
    "TaskType",
    "BoardType",
    "WorkStatus",
    "HealthLevel",
]


class BackpressureLevel(str, Enum):
    """Backpressure severity level.

    Used by both queue monitors and resource coordinators to indicate
    the level of throttling that should be applied to production.

    Levels (in order of severity):
        NONE: No backpressure, operate normally
        LOW: Minor backpressure, reduce production slightly (~25%)
        SOFT: Soft throttle, reduce production moderately (~50%)
        MEDIUM: Medium backpressure, reduce production significantly (~75%)
        HARD: Hard throttle, reduce production heavily (~90%)
        HIGH: High backpressure, minimize production
        CRITICAL: Critical backpressure, consider stopping
        STOP: Stop production entirely

    Note: This unified enum supports both:
        - Queue-based: NONE, SOFT, HARD, STOP (from queue_monitor.py)
        - Resource-based: NONE, LOW, MEDIUM, HIGH, CRITICAL (from resource coordinators)

    For backward compatibility, both sets of values are supported.
    New code should use the severity levels that best match the use case.
    """

    NONE = "none"  # No backpressure
    LOW = "low"  # Minor (~25% reduction)
    SOFT = "soft"  # Soft throttle (~50% reduction)
    MEDIUM = "medium"  # Medium (~75% reduction)
    HARD = "hard"  # Hard throttle (~90% reduction)
    HIGH = "high"  # High (minimize production)
    CRITICAL = "critical"  # Critical (near-stop)
    STOP = "stop"  # Stop production entirely

    @classmethod
    def from_legacy_queue(cls, value: str) -> "BackpressureLevel":
        """Convert legacy queue_monitor BackpressureLevel value."""
        mapping = {
            "none": cls.NONE,
            "soft": cls.SOFT,
            "hard": cls.HARD,
            "stop": cls.STOP,
        }
        return mapping.get(value.lower(), cls.NONE)

    @classmethod
    def from_legacy_resource(cls, value: str) -> "BackpressureLevel":
        """Convert legacy resource coordinator BackpressureLevel value."""
        mapping = {
            "none": cls.NONE,
            "low": cls.LOW,
            "medium": cls.MEDIUM,
            "high": cls.HIGH,
            "critical": cls.CRITICAL,
        }
        return mapping.get(value.lower(), cls.NONE)

    def is_throttling(self) -> bool:
        """Return True if this level indicates some throttling is needed."""
        return self != BackpressureLevel.NONE

    def should_stop(self) -> bool:
        """Return True if this level indicates production should stop."""
        return self in (BackpressureLevel.CRITICAL, BackpressureLevel.STOP)

    def reduction_factor(self) -> float:
        """Return the production reduction factor (0.0 = stopped, 1.0 = full)."""
        factors = {
            BackpressureLevel.NONE: 1.0,
            BackpressureLevel.LOW: 0.75,
            BackpressureLevel.SOFT: 0.50,
            BackpressureLevel.MEDIUM: 0.25,
            BackpressureLevel.HARD: 0.10,
            BackpressureLevel.HIGH: 0.05,
            BackpressureLevel.CRITICAL: 0.01,
            BackpressureLevel.STOP: 0.0,
        }
        return factors.get(self, 1.0)


class TaskType(str, Enum):
    """Types of compute tasks in the cluster.

    Used by resource coordinators and job schedulers to categorize work.
    Consolidated from types.py and task_coordinator.py (December 2025).
    """

    # Core task types
    SELFPLAY = "selfplay"  # Self-play game generation
    GPU_SELFPLAY = "gpu_selfplay"  # GPU-accelerated self-play
    HYBRID_SELFPLAY = "hybrid_selfplay"  # Mixed CPU/GPU self-play
    TRAINING = "training"  # Neural network training
    EVALUATION = "evaluation"  # Model evaluation (gauntlet, etc.)
    EXPORT = "export"  # Data export (NPZ generation)
    SYNC = "sync"  # Data synchronization
    TOURNAMENT = "tournament"  # Tournament games
    PARITY = "parity"  # Parity testing

    # Optimization task types
    CMAES = "cmaes"  # CMA-ES hyperparameter optimization

    # Pipeline/loop task types
    PIPELINE = "pipeline"  # Full pipeline execution
    IMPROVEMENT_LOOP = "improvement_loop"  # Continuous improvement loop
    BACKGROUND_LOOP = "background_loop"  # Background maintenance loop

    # Fallback
    UNKNOWN = "unknown"  # Unknown/other task type


class BoardType(str, Enum):
    """Canonical board types for all configurations.

    December 2025: Consolidated from utilization_optimizer.py, unified_queue_populator.py
    """

    HEX8 = "hex8"           # Small hexagonal, 61 cells
    SQUARE8 = "square8"     # Standard 8x8, 64 cells
    SQUARE19 = "square19"   # Large square (Go-sized), 361 cells
    HEXAGONAL = "hexagonal" # Large hexagonal, 469 cells

    @property
    def cell_count(self) -> int:
        """Approximate cell count for this board type."""
        return {
            BoardType.HEX8: 61,
            BoardType.SQUARE8: 64,
            BoardType.SQUARE19: 361,
            BoardType.HEXAGONAL: 469,
        }[self]

    @property
    def vram_requirement_gb(self) -> float:
        """Approximate VRAM requirement for training/inference."""
        return {
            BoardType.HEX8: 4.0,
            BoardType.SQUARE8: 4.0,
            BoardType.SQUARE19: 8.0,
            BoardType.HEXAGONAL: 12.0,
        }[self]

    @classmethod
    def from_string(cls, s: str) -> "BoardType":
        """Parse board type from string, handling common variations."""
        s = s.lower().replace("-", "").replace("_", "")
        mapping = {
            "hex8": cls.HEX8,
            "hexsmall": cls.HEX8,
            "square8": cls.SQUARE8,
            "sq8": cls.SQUARE8,
            "square19": cls.SQUARE19,
            "sq19": cls.SQUARE19,
            "hexagonal": cls.HEXAGONAL,
            "hexlarge": cls.HEXAGONAL,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown board type: {s}")


class WorkStatus(str, Enum):
    """Status of work items in the work queue."""

    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class HealthLevel(str, Enum):
    """System health levels.

    Used by health monitors and daemon managers.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
