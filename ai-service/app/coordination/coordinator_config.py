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
    """Configuration for DataPipelineOrchestrator.

    Note: auto_trigger defaults to True as of Dec 2025 to enable the full
    training pipeline automation. Set COORDINATOR_AUTO_TRIGGER_PIPELINE=false
    to disable if needed.
    """

    max_history: int = 100
    auto_trigger: bool = True  # Changed from False (Dec 2025)
    pause_on_critical_constraints: bool = True
    constraint_stale_seconds: float = 60.0

    # Per-stage auto-trigger controls (December 2025)
    auto_trigger_sync: bool = True
    auto_trigger_export: bool = True
    auto_trigger_training: bool = True
    auto_trigger_evaluation: bool = True
    auto_trigger_promotion: bool = True

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_reset_timeout_seconds: float = 300.0  # 5 minutes
    circuit_breaker_half_open_max_requests: int = 1

    # Training configuration (December 2025 - wired from CLI)
    training_epochs: int = 50
    training_batch_size: int = 512
    training_model_version: str = "v2"


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


# =============================================================================
# Daemon Exclusion Policy (December 2025)
# =============================================================================

# Default excluded nodes - dev machines and low storage nodes
_DEFAULT_EXCLUDED_NODES: frozenset[str] = frozenset({
    "mbp-16gb",        # Low storage laptop
    "mbp-64gb",        # Dev machine laptop
    "mbp-128gb",       # M4 Max MacBook Pro - don't fill disk
    "macbook-pro-2",   # Same machine, alternate node-id
    "aws-proxy",       # Relay-only node
})


@dataclass
class DaemonExclusionConfig:
    """Configuration for daemon node exclusions.

    Provides a unified exclusion policy for all daemons, replacing
    hardcoded frozensets in individual daemon files.
    """

    # Nodes that should NEVER receive synced data
    excluded_nodes: set[str] = field(default_factory=lambda: set(_DEFAULT_EXCLUDED_NODES))

    # NFS-connected nodes (skip sync between them)
    nfs_nodes: set[str] = field(default_factory=set)

    # Nodes marked as retired (skip entirely)
    retired_nodes: set[str] = field(default_factory=set)

    # Minimum free disk space (GB) for sync eligibility
    min_disk_free_gb: float = 50.0

    def should_exclude(self, node_id: str) -> bool:
        """Check if a node should be excluded from sync operations."""
        return (
            node_id in self.excluded_nodes or
            node_id in self.retired_nodes
        )

    def is_nfs_node(self, node_id: str) -> bool:
        """Check if a node is NFS-connected (skip inter-NFS sync)."""
        return node_id in self.nfs_nodes

    def add_excluded_node(self, node_id: str) -> None:
        """Add a node to the exclusion list."""
        self.excluded_nodes.add(node_id)

    def remove_excluded_node(self, node_id: str) -> None:
        """Remove a node from the exclusion list."""
        self.excluded_nodes.discard(node_id)


# Global exclusion policy singleton
_exclusion_policy: DaemonExclusionConfig | None = None


def get_exclusion_policy() -> DaemonExclusionConfig:
    """Get the global daemon exclusion policy.

    Loads from config files on first call:
    - unified_loop.yaml (auto_sync.exclude_hosts, data_aggregation.excluded_nodes)
    - distributed_hosts.yaml (nfs_nodes, retired)

    Returns:
        The global DaemonExclusionConfig instance
    """
    global _exclusion_policy
    if _exclusion_policy is None:
        _exclusion_policy = _load_exclusion_config()
        logger.info("[DaemonExclusionPolicy] Loaded exclusion config")
    return _exclusion_policy


def _load_exclusion_config() -> DaemonExclusionConfig:
    """Load exclusion configuration from config files."""
    from pathlib import Path

    config = DaemonExclusionConfig()

    # Try to find config directory
    base_dir = Path(__file__).resolve().parents[2]  # ai-service root
    config_dir = base_dir / "config"

    # Load from unified_loop.yaml
    unified_config_path = config_dir / "unified_loop.yaml"
    if unified_config_path.exists():
        try:
            import yaml
            with open(unified_config_path) as f:
                data = yaml.safe_load(f) or {}

            # auto_sync.exclude_hosts
            auto_sync = data.get("auto_sync", {})
            for node in auto_sync.get("exclude_hosts", []):
                config.excluded_nodes.add(node)

            # data_aggregation.excluded_nodes
            data_agg = data.get("data_aggregation", {})
            for node in data_agg.get("excluded_nodes", []):
                config.excluded_nodes.add(node)

        except Exception as e:
            logger.debug(f"Could not load unified_loop.yaml: {e}")

    # Load from distributed_hosts.yaml
    hosts_config_path = config_dir / "distributed_hosts.yaml"
    if hosts_config_path.exists():
        try:
            import yaml
            with open(hosts_config_path) as f:
                data = yaml.safe_load(f) or {}

            # Find NFS nodes
            for host in data.get("hosts", []):
                if host.get("is_nfs", False):
                    node_id = host.get("node_id") or host.get("name", "")
                    if node_id:
                        config.nfs_nodes.add(node_id)

                # Find retired nodes
                if host.get("retired", False):
                    node_id = host.get("node_id") or host.get("name", "")
                    if node_id:
                        config.retired_nodes.add(node_id)

        except Exception as e:
            logger.debug(f"Could not load distributed_hosts.yaml: {e}")

    return config


def reset_exclusion_policy() -> None:
    """Reset the exclusion policy (for testing)."""
    global _exclusion_policy
    _exclusion_policy = None


__all__ = [
    "CacheConfig",
    # Unified config
    "CoordinatorConfig",
    # Daemon exclusion policy
    "DaemonExclusionConfig",
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
    "get_exclusion_policy",
    "reset_config",
    "reset_exclusion_policy",
    "set_config",
    "update_config",
    "validate_config",
]
