"""Centralized Coordinator Configuration (December 2025).

This module provides a unified configuration system for all coordinators,
enabling consistent configuration across the coordination system.

Features:
- Typed configuration dataclasses for each coordinator
- Global configuration singleton
- Environment variable overrides
- Runtime configuration updates
- Configuration validation

Usage:
    from app.coordination.coordinator_config import (
        CoordinatorConfig,
        get_config,
        update_config,
    )

    # Get configuration
    config = get_config()
    print(f"Heartbeat threshold: {config.task_lifecycle.heartbeat_threshold_seconds}")

    # Update configuration at runtime
    update_config(task_lifecycle=TaskLifecycleConfig(heartbeat_threshold_seconds=120.0))

    # Use environment variables
    # COORDINATOR_HEARTBEAT_THRESHOLD=120.0
    # COORDINATOR_HANDLER_TIMEOUT=60.0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Individual Coordinator Configurations
# =============================================================================


@dataclass
class TaskLifecycleConfig:
    """Configuration for TaskLifecycleCoordinator."""

    heartbeat_threshold_seconds: float = 60.0
    orphan_check_interval_seconds: float = 30.0
    max_history: int = 1000


@dataclass
class SelfplayConfig:
    """Configuration for SelfplayOrchestrator."""

    max_history: int = 500
    stats_window_seconds: float = 3600.0  # 1 hour
    backpressure_threshold: float = 0.8


@dataclass
class PipelineConfig:
    """Configuration for DataPipelineOrchestrator."""

    max_history: int = 100
    auto_trigger: bool = False
    pause_on_critical_constraints: bool = True
    constraint_stale_seconds: float = 60.0


@dataclass
class OptimizationConfig:
    """Configuration for OptimizationCoordinator."""

    cmaes_cooldown_seconds: float = 3600.0  # 1 hour
    nas_cooldown_seconds: float = 7200.0  # 2 hours
    auto_trigger_on_plateau: bool = True
    min_plateau_epochs_for_trigger: int = 15


@dataclass
class MetricsConfig:
    """Configuration for MetricsAnalysisOrchestrator."""

    window_size: int = 100
    plateau_threshold: float = 0.001
    plateau_window: int = 10
    regression_threshold: float = 0.05
    anomaly_threshold: float = 3.0  # Standard deviations


@dataclass
class ResourceConfig:
    """Configuration for ResourceMonitoringCoordinator."""

    monitoring_interval_seconds: float = 30.0
    memory_warning_threshold: float = 0.8
    memory_critical_threshold: float = 0.95
    gpu_memory_warning_threshold: float = 0.85
    gpu_memory_critical_threshold: float = 0.95


@dataclass
class CacheConfig:
    """Configuration for CacheCoordinationOrchestrator."""

    invalidation_batch_size: int = 100
    max_cache_age_seconds: float = 3600.0  # 1 hour
    auto_refresh: bool = True


@dataclass
class HandlerResilienceConfig:
    """Configuration for handler resilience (timeouts, retries)."""

    timeout_seconds: float = 30.0
    emit_failure_events: bool = True
    emit_timeout_events: bool = True
    max_consecutive_failures: int = 5
    log_exceptions: bool = True


@dataclass
class HeartbeatConfig:
    """Configuration for coordinator heartbeats."""

    interval_seconds: float = 30.0
    enabled: bool = True


@dataclass
class EventBusConfig:
    """Configuration for the event bus."""

    max_history: int = 1000
    warn_unsubscribed: bool = True
    max_latency_samples: int = 1000


# =============================================================================
# Unified Configuration
# =============================================================================


@dataclass
class CoordinatorConfig:
    """Unified configuration for all coordinators.

    All coordinator configurations are accessible through this single
    class, providing a centralized configuration point.
    """

    task_lifecycle: TaskLifecycleConfig = field(default_factory=TaskLifecycleConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    handler_resilience: HandlerResilienceConfig = field(default_factory=HandlerResilienceConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoordinatorConfig:
        """Create from dictionary."""
        return cls(
            task_lifecycle=TaskLifecycleConfig(**data.get("task_lifecycle", {})),
            selfplay=SelfplayConfig(**data.get("selfplay", {})),
            pipeline=PipelineConfig(**data.get("pipeline", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
            metrics=MetricsConfig(**data.get("metrics", {})),
            resources=ResourceConfig(**data.get("resources", {})),
            cache=CacheConfig(**data.get("cache", {})),
            handler_resilience=HandlerResilienceConfig(**data.get("handler_resilience", {})),
            heartbeat=HeartbeatConfig(**data.get("heartbeat", {})),
            event_bus=EventBusConfig(**data.get("event_bus", {})),
        )

    @classmethod
    def from_environment(cls) -> CoordinatorConfig:
        """Create configuration from environment variables.

        Environment variables are prefixed with COORDINATOR_ and use
        uppercase snake_case naming.

        Examples:
            COORDINATOR_HEARTBEAT_THRESHOLD=120.0
            COORDINATOR_HANDLER_TIMEOUT=60.0
            COORDINATOR_AUTO_TRIGGER_PIPELINE=true
        """
        config = cls()

        # Task lifecycle
        if val := os.environ.get("COORDINATOR_HEARTBEAT_THRESHOLD"):
            config.task_lifecycle.heartbeat_threshold_seconds = float(val)
        if val := os.environ.get("COORDINATOR_ORPHAN_CHECK_INTERVAL"):
            config.task_lifecycle.orphan_check_interval_seconds = float(val)

        # Pipeline
        if val := os.environ.get("COORDINATOR_AUTO_TRIGGER_PIPELINE"):
            config.pipeline.auto_trigger = val.lower() in ("true", "1", "yes")

        # Optimization
        if val := os.environ.get("COORDINATOR_CMAES_COOLDOWN"):
            config.optimization.cmaes_cooldown_seconds = float(val)
        if val := os.environ.get("COORDINATOR_AUTO_TRIGGER_CMAES"):
            config.optimization.auto_trigger_on_plateau = val.lower() in ("true", "1", "yes")

        # Handler resilience
        if val := os.environ.get("COORDINATOR_HANDLER_TIMEOUT"):
            config.handler_resilience.timeout_seconds = float(val)
        if val := os.environ.get("COORDINATOR_MAX_CONSECUTIVE_FAILURES"):
            config.handler_resilience.max_consecutive_failures = int(val)

        # Heartbeat
        if val := os.environ.get("COORDINATOR_HEARTBEAT_INTERVAL"):
            config.heartbeat.interval_seconds = float(val)
        if val := os.environ.get("COORDINATOR_HEARTBEAT_ENABLED"):
            config.heartbeat.enabled = val.lower() in ("true", "1", "yes")

        # Metrics
        if val := os.environ.get("COORDINATOR_METRICS_WINDOW_SIZE"):
            config.metrics.window_size = int(val)
        if val := os.environ.get("COORDINATOR_PLATEAU_THRESHOLD"):
            config.metrics.plateau_threshold = float(val)

        # Resources
        if val := os.environ.get("COORDINATOR_MEMORY_WARNING_THRESHOLD"):
            config.resources.memory_warning_threshold = float(val)
        if val := os.environ.get("COORDINATOR_MEMORY_CRITICAL_THRESHOLD"):
            config.resources.memory_critical_threshold = float(val)

        return config


# =============================================================================
# Global Configuration Singleton
# =============================================================================

_config: CoordinatorConfig | None = None


def get_config() -> CoordinatorConfig:
    """Get the global coordinator configuration.

    Initializes from environment variables on first call.

    Returns:
        The global CoordinatorConfig instance
    """
    global _config
    if _config is None:
        _config = CoordinatorConfig.from_environment()
        logger.info("[CoordinatorConfig] Initialized from environment")
    return _config


def set_config(config: CoordinatorConfig) -> None:
    """Set the global coordinator configuration.

    Args:
        config: The configuration to use
    """
    global _config
    _config = config
    logger.info("[CoordinatorConfig] Configuration updated")


def reset_config() -> None:
    """Reset to default configuration (for testing)."""
    global _config
    _config = None


def update_config(**kwargs) -> CoordinatorConfig:
    """Update specific parts of the configuration.

    Args:
        **kwargs: Configuration sections to update
            (e.g., task_lifecycle=TaskLifecycleConfig(...))

    Returns:
        Updated CoordinatorConfig
    """
    config = get_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"[CoordinatorConfig] Updated {key}")
        else:
            logger.warning(f"[CoordinatorConfig] Unknown config section: {key}")

    return config


def validate_config(config: CoordinatorConfig | None = None) -> tuple:
    """Validate configuration values.

    Args:
        config: Configuration to validate (uses global if None)

    Returns:
        (valid, issues) tuple
    """
    config = config or get_config()
    issues = []

    # Validate task lifecycle
    if config.task_lifecycle.heartbeat_threshold_seconds < 10:
        issues.append("heartbeat_threshold_seconds too low (min 10)")
    if config.task_lifecycle.orphan_check_interval_seconds < 5:
        issues.append("orphan_check_interval_seconds too low (min 5)")

    # Validate handler resilience
    if config.handler_resilience.timeout_seconds < 1:
        issues.append("handler timeout_seconds too low (min 1)")
    if config.handler_resilience.max_consecutive_failures < 1:
        issues.append("max_consecutive_failures must be >= 1")

    # Validate resources
    if not (0 < config.resources.memory_warning_threshold < 1):
        issues.append("memory_warning_threshold must be between 0 and 1")
    if not (0 < config.resources.memory_critical_threshold < 1):
        issues.append("memory_critical_threshold must be between 0 and 1")
    if config.resources.memory_warning_threshold >= config.resources.memory_critical_threshold:
        issues.append("memory_warning_threshold must be less than critical")

    # Validate metrics
    if config.metrics.window_size < 10:
        issues.append("metrics window_size too small (min 10)")
    if config.metrics.plateau_threshold <= 0:
        issues.append("plateau_threshold must be positive")

    return (len(issues) == 0, issues)


__all__ = [
    "CacheConfig",
    # Unified config
    "CoordinatorConfig",
    "EventBusConfig",
    "HandlerResilienceConfig",
    "HeartbeatConfig",
    "MetricsConfig",
    "OptimizationConfig",
    "PipelineConfig",
    "ResourceConfig",
    "SelfplayConfig",
    # Individual configs
    "TaskLifecycleConfig",
    # Functions
    "get_config",
    "reset_config",
    "set_config",
    "update_config",
    "validate_config",
]
