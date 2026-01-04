"""Training trigger type definitions.

Jan 4, 2026 - Sprint 17.9: Extracted from training_trigger_daemon.py as part of
daemon decomposition following selfplay_scheduler → selfplay_priority_types.py pattern.

This module contains pure data classes with no I/O dependencies:
- TrainingTriggerConfig - Configuration for training trigger decisions
- ConfigTrainingState - Tracks training state for a single configuration
- TrainingDecision - Result of training trigger decision check
- ArchitectureSpec - Specification for training a single architecture
- MultiArchitectureConfig - Configuration for multi-architecture training

Usage:
    from app.coordination.training_trigger_types import (
        TrainingTriggerConfig,
        ConfigTrainingState,
        TrainingDecision,
        ArchitectureSpec,
        MultiArchitectureConfig,
        TRIGGER_DEDUP_WINDOW_SECONDS,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.config.coordination_defaults import DataFreshnessDefaults

logger = logging.getLogger(__name__)

# Dec 29, 2025: Event deduplication to prevent multiple training triggers
# for the same config within a short window
TRIGGER_DEDUP_WINDOW_SECONDS = 300  # 5 minutes


@dataclass
class TrainingTriggerConfig:
    """Configuration for training trigger decisions."""

    enabled: bool = True
    # Data freshness - uses DataFreshnessDefaults for unified config (December 2025)
    # Default 4h from RINGRIFT_MAX_DATA_AGE_HOURS env var
    max_data_age_hours: float = field(
        default_factory=lambda: DataFreshnessDefaults().MAX_DATA_AGE_HOURS
    )
    # December 2025: Use training_freshness to trigger sync when data is stale
    enforce_freshness_with_sync: bool = field(
        default_factory=lambda: DataFreshnessDefaults().ENFORCE_FRESHNESS_WITH_SYNC
    )
    freshness_sync_timeout_seconds: float = field(
        default_factory=lambda: DataFreshnessDefaults().FRESHNESS_SYNC_TIMEOUT
    )
    # December 29, 2025: Strict mode - fail immediately if data is stale (no sync attempt)
    # Useful for high-quality training where only fresh data should be used
    strict_freshness_mode: bool = field(
        default_factory=lambda: DataFreshnessDefaults().STRICT_FRESHNESS
    )
    # Minimum samples to trigger training
    # December 29, 2025: Reduced from 10000 to 5000 for faster iteration cycles
    min_samples_threshold: int = 5000
    # December 29, 2025: Confidence-based early triggering
    # Allows training to start earlier if statistical confidence is high
    confidence_early_trigger_enabled: bool = True
    # Minimum samples to even consider confidence-based early trigger (safety floor)
    confidence_min_samples: int = 1000
    # Target confidence interval width (95% CI width, e.g., 0.05 = ±2.5%)
    confidence_target_ci_width: float = 0.05
    # Cooldown between training runs for same config
    # December 29, 2025: Reduced from 1.0 to 0.083 (5 min) for faster iteration cycles
    training_cooldown_hours: float = 0.083
    # Maximum concurrent training jobs
    # December 29, 2025: Increased from 10 to 20 for better multi-GPU cluster utilization
    # Cluster has ~36 nodes with GPUs; allowing more concurrent training maximizes throughput
    max_concurrent_training: int = 20
    # GPU utilization threshold for "idle"
    gpu_idle_threshold_percent: float = 20.0
    # Timeout for training subprocess (24 hours)
    training_timeout_seconds: int = 86400
    # December 29, 2025: Training timeout watchdog (Phase 2 - 48h autonomous operation)
    # Independent watchdog kills training jobs that exceed this limit
    # This catches hung processes even if the daemon restarts
    training_timeout_hours: float = 4.0
    # January 2, 2026: Graceful shutdown timeout - send SIGTERM first, wait this long,
    # then SIGKILL if process still running. Prevents checkpoint corruption.
    graceful_kill_timeout_seconds: float = 30.0
    # January 2, 2026: Maximum duration for evaluation backpressure before auto-recovery.
    # If EVALUATION_BACKPRESSURE_RELEASED event is lost/never received, training resumes
    # automatically after this timeout to prevent indefinite training pauses.
    # January 3, 2026: Reduced from 1800s (30 min) to 300s (5 min) based on pipeline
    # analysis showing shorter backpressure periods are optimal for throughput.
    backpressure_max_duration_seconds: float = 300.0  # 5 minutes
    # Check interval for periodic scans
    # December 29, 2025: Reduced from 120s to 30s for faster detection
    scan_interval_seconds: int = 30  # 30 seconds
    # Training epochs
    default_epochs: int = 50
    default_batch_size: int = 512
    # Model version
    model_version: str = "v2"
    # December 29, 2025: State persistence for daemon restarts (Phase 3)
    state_db_path: str = "data/coordination/training_trigger_state.db"
    state_save_interval_seconds: float = 300.0  # Save every 5 minutes
    # January 2026: Log aggregated game counts from all sources (local, cluster, S3, OWC)
    # Provides visibility into cluster-wide game availability for training decisions
    log_aggregated_game_counts: bool = True
    # Jan 2, 2026: Local-only mode - skip cluster checks when cluster is unavailable
    local_only_mode: bool = False
    # Jan 2, 2026: Timeout for cluster availability check before falling back to local mode
    cluster_availability_timeout_seconds: float = 10.0
    # Jan 2, 2026: Auto-detect local-only mode based on cluster availability
    auto_detect_local_mode: bool = True
    # January 3, 2026: Quality score confidence decay
    # Quality scores decay over time to prevent stale assessments from blocking training.
    # After half_life hours, quality confidence drops by 50% toward the decay floor.
    quality_decay_half_life_hours: float = 1.0  # Quality confidence halves every hour
    quality_decay_floor: float = 0.5  # Never decay below this quality level
    quality_decay_enabled: bool = True  # Enable/disable decay mechanism


@dataclass
class ConfigTrainingState:
    """Tracks training state for a single configuration."""

    config_key: str
    board_type: str
    num_players: int
    # Training status
    last_training_time: float = 0.0
    training_in_progress: bool = False
    training_pid: int | None = None
    # December 29, 2025: Training timeout watchdog (Phase 2)
    training_start_time: float = 0.0  # When current training started
    # Data status
    last_npz_update: float = 0.0
    npz_sample_count: int = 0
    npz_path: str = ""
    # Quality tracking
    last_elo: float = 1500.0
    elo_trend: float = 0.0  # positive = improving
    # December 29, 2025: Elo velocity tracking for training decisions
    elo_velocity: float = 0.0  # Elo/hour rate of change
    elo_velocity_trend: str = "stable"  # accelerating, stable, decelerating, plateauing
    last_elo_velocity_update: float = 0.0
    # Training intensity (set by master_loop or FeedbackLoopController)
    training_intensity: str = "normal"  # hot_path, accelerated, normal, reduced, paused
    consecutive_failures: int = 0
    # December 29, 2025: Track model path for event emission
    _pending_model_path: str = ""  # Path where current training will save model
    # January 3, 2026: Quality score tracking with confidence decay
    # Quality scores decay over time if not refreshed, preventing stale assessments
    # from blocking training when conditions have changed
    last_quality_score: float = 0.7  # Default to minimum threshold
    last_quality_update: float = 0.0  # Timestamp when quality was last updated
    # Sprint 12 Session 8: Quality score confidence based on game count
    # Configs with few games get lower confidence (more uncertainty about true quality)
    games_assessed: int = 0  # Number of games used in quality assessment


@dataclass
class TrainingDecision:
    """Result of training trigger decision check.

    December 30, 2025: Added for RPC API to expose training decision logic.
    Provides full condition details for debugging and monitoring.
    """

    config_key: str
    can_trigger: bool
    reason: str
    # Detailed condition states
    training_in_progress: bool = False
    intensity_paused: bool = False
    evaluation_backpressure: bool = False
    circuit_breaker_open: bool = False
    cooldown_remaining_hours: float = 0.0
    data_age_hours: float = 0.0
    max_data_age_hours: float = 4.0
    sample_count: int = 0
    sample_threshold: int = 5000
    gpu_available: bool = True
    concurrent_training_count: int = 0
    max_concurrent_training: int = 20
    # Data info
    npz_path: str = ""
    # Elo tracking
    current_elo: float = 1500.0
    elo_velocity: float = 0.0
    elo_velocity_trend: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_key": self.config_key,
            "can_trigger": self.can_trigger,
            "reason": self.reason,
            "conditions": {
                "training_in_progress": self.training_in_progress,
                "intensity_paused": self.intensity_paused,
                "evaluation_backpressure": self.evaluation_backpressure,
                "circuit_breaker_open": self.circuit_breaker_open,
                "cooldown_remaining_hours": round(self.cooldown_remaining_hours, 2),
                "data_age_hours": round(self.data_age_hours, 2),
                "max_data_age_hours": self.max_data_age_hours,
                "sample_count": self.sample_count,
                "sample_threshold": self.sample_threshold,
                "gpu_available": self.gpu_available,
                "concurrent_training_count": self.concurrent_training_count,
                "max_concurrent_training": self.max_concurrent_training,
            },
            "data_info": {
                "npz_path": self.npz_path,
                "samples": self.sample_count,
                "age_hours": round(self.data_age_hours, 2),
            },
            "elo_info": {
                "current_elo": round(self.current_elo, 1),
                "elo_velocity": round(self.elo_velocity, 3),
                "elo_velocity_trend": self.elo_velocity_trend,
            },
        }


# December 30, 2025: Multi-Architecture Training Support
# These dataclasses parse config/architecture_training.yaml to control which
# architectures are trained on which board configurations.


@dataclass
class ArchitectureSpec:
    """Specification for training a single architecture."""

    name: str
    enabled: bool
    configs: list[str]  # List of config_keys or ["*"] for all
    priority: float  # Fraction of training compute (0.0-1.0)
    description: str = ""
    # Training overrides
    epochs: int | None = None
    batch_size: int | None = None

    def matches_config(self, config_key: str) -> bool:
        """Check if this architecture should be trained on a config."""
        if not self.enabled:
            return False
        if "*" in self.configs:
            return True
        return config_key in self.configs


@dataclass
class MultiArchitectureConfig:
    """Configuration for multi-architecture training."""

    architectures: dict[str, ArchitectureSpec]
    min_samples_per_architecture: int = 3000
    max_concurrent_per_architecture: int = 2
    min_hours_between_runs: float = 4.0

    @classmethod
    def load(cls, config_path: Path | None = None) -> "MultiArchitectureConfig":
        """Load architecture config from YAML file.

        Args:
            config_path: Path to YAML config. If None, uses default path.

        Returns:
            MultiArchitectureConfig instance.
        """
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "architecture_training.yaml"

        architectures: dict[str, ArchitectureSpec] = {}

        if not config_path.exists():
            logger.warning(
                f"[MultiArchitectureConfig] Config not found: {config_path}, "
                "using default (v5 only)"
            )
            # Default: only train v5 on all configs
            architectures["v5"] = ArchitectureSpec(
                name="v5",
                enabled=True,
                configs=["*"],
                priority=1.0,
                description="Default v5 architecture",
            )
            return cls(architectures=architectures)

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[MultiArchitectureConfig] Failed to load {config_path}: {e}")
            # Fallback to v5 only
            architectures["v5"] = ArchitectureSpec(
                name="v5",
                enabled=True,
                configs=["*"],
                priority=1.0,
            )
            return cls(architectures=architectures)

        # Parse architectures
        arch_data = data.get("architectures", {})
        for arch_name, arch_spec in arch_data.items():
            if not isinstance(arch_spec, dict):
                continue

            # Get training overrides if present
            training_overrides = data.get("training_overrides", {}).get(arch_name, {})

            architectures[arch_name] = ArchitectureSpec(
                name=arch_name,
                enabled=arch_spec.get("enabled", True),
                configs=arch_spec.get("configs", []),
                priority=arch_spec.get("priority", 0.1),
                description=arch_spec.get("description", ""),
                epochs=training_overrides.get("epochs"),
                batch_size=training_overrides.get("batch_size"),
            )

        # Parse thresholds
        thresholds = data.get("thresholds", {})
        min_samples = thresholds.get("min_samples_per_architecture", 3000)
        max_concurrent = thresholds.get("max_concurrent_per_architecture", 2)
        min_hours = thresholds.get("min_hours_between_runs", 4.0)

        logger.info(
            f"[MultiArchitectureConfig] Loaded {len(architectures)} architectures: "
            f"{list(architectures.keys())}"
        )

        return cls(
            architectures=architectures,
            min_samples_per_architecture=min_samples,
            max_concurrent_per_architecture=max_concurrent,
            min_hours_between_runs=min_hours,
        )

    def get_architectures_for_config(self, config_key: str) -> list[ArchitectureSpec]:
        """Get list of architectures that should be trained on a config."""
        return [
            arch for arch in self.architectures.values()
            if arch.matches_config(config_key)
        ]


__all__ = [
    "TRIGGER_DEDUP_WINDOW_SECONDS",
    "TrainingTriggerConfig",
    "ConfigTrainingState",
    "TrainingDecision",
    "ArchitectureSpec",
    "MultiArchitectureConfig",
]
