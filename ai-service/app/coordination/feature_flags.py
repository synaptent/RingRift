"""Feature Flags for Coordination Module Optional Dependencies.

This module provides a centralized registry for checking availability of optional
coordination dependencies. It consolidates scattered try/except ImportError blocks
with HAS_* flags into a unified interface.

December 2025: Created to consolidate 45+ HAS_* flags across coordination modules.

Usage:
    from app.coordination.feature_flags import has_feature, get_feature

    # Check if feature is available
    if has_feature('circuit_breaker'):
        breaker = get_feature('circuit_breaker', 'get_training_breaker')
        ...

    # Or use backward-compatible HAS_* constants
    from app.coordination.feature_flags import HAS_CIRCUIT_BREAKER
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    """Specification for an optional coordination feature.

    Attributes:
        name: Feature name (e.g., 'circuit_breaker')
        module_path: Import path (e.g., 'app.distributed.circuit_breaker')
        exports: List of names to import from the module
        description: Human-readable description
    """

    name: str
    module_path: str
    exports: list[str]
    description: str = ""


@dataclass
class FeatureStatus:
    """Runtime status of an optional feature.

    Attributes:
        available: Whether the feature is importable
        exports: Dict mapping export names to their values (if available)
        error: Error message if import failed
    """

    available: bool
    exports: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# Registry of all optional features used by coordination modules
COORDINATION_FEATURE_SPECS: dict[str, FeatureSpec] = {
    # ==========================================================================
    # Event System Features
    # ==========================================================================
    "data_events": FeatureSpec(
        name="data_events",
        module_path="app.distributed.data_events",
        exports=["DataEventType", "emit_data_event"],
        description="Data event types and emission",
    ),
    "stage_events": FeatureSpec(
        name="stage_events",
        module_path="app.coordination.stage_events",
        exports=["StageEvent", "emit_stage_event"],
        description="Pipeline stage events",
    ),
    "cross_process_events": FeatureSpec(
        name="cross_process_events",
        module_path="app.distributed.cross_process_queue",
        exports=["CrossProcessQueue", "get_cross_process_queue"],
        description="Cross-process event queue",
    ),
    "dlq": FeatureSpec(
        name="dlq",
        module_path="app.coordination.dead_letter_queue",
        exports=["DeadLetterQueue", "get_dlq"],
        description="Dead letter queue for failed events",
    ),
    "event_emitters": FeatureSpec(
        name="event_emitters",
        module_path="app.coordination.event_router",
        exports=["emit_training_complete", "emit_evaluation_complete"],
        description="Centralized event emitters (via event_router)",
    ),
    "selfplay_events": FeatureSpec(
        name="selfplay_events",
        module_path="app.coordination.event_router",
        exports=["emit_selfplay_complete", "emit_selfplay_started"],
        description="Selfplay event emission (via event_router)",
    ),
    "exploration_events": FeatureSpec(
        name="exploration_events",
        module_path="app.coordination.event_router",
        exports=["emit_exploration_boost"],
        description="Exploration boost events (via event_router)",
    ),
    "node_events": FeatureSpec(
        name="node_events",
        module_path="app.coordination.event_router",
        exports=["emit_node_health_changed", "emit_node_offline"],
        description="Node health events (via event_router)",
    ),
    "cluster_events": FeatureSpec(
        name="cluster_events",
        module_path="app.coordination.event_router",
        exports=["emit_cluster_health_changed"],
        description="Cluster health events (via event_router)",
    ),
    "incompatibility_events": FeatureSpec(
        name="incompatibility_events",
        module_path="app.coordination.event_router",
        exports=["emit_incompatibility_detected"],
        description="Model incompatibility events (via event_router)",
    ),
    # ==========================================================================
    # Circuit Breaker & Resilience Features
    # ==========================================================================
    "circuit_breaker": FeatureSpec(
        name="circuit_breaker",
        module_path="app.distributed.circuit_breaker",
        exports=["CircuitState", "get_training_breaker", "CircuitBreaker"],
        description="Circuit breaker for fault tolerance",
    ),
    "resilient_handler": FeatureSpec(
        name="resilient_handler",
        module_path="app.coordination.resilient_event_handler",
        exports=["ResilientEventHandler"],
        description="Resilient event handling with retry",
    ),
    "backpressure": FeatureSpec(
        name="backpressure",
        module_path="app.coordination.backpressure",
        exports=["BackpressureController", "get_backpressure_controller"],
        description="Backpressure management",
    ),
    "stall_detection": FeatureSpec(
        name="stall_detection",
        module_path="app.coordination.progress_watchdog_daemon",
        exports=["ProgressWatchdogDaemon", "get_progress_watchdog"],
        description="Training stall detection",
    ),
    # ==========================================================================
    # Configuration & Defaults Features
    # ==========================================================================
    "dynamic_thresholds": FeatureSpec(
        name="dynamic_thresholds",
        module_path="app.coordination.dynamic_thresholds",
        exports=["ThresholdManager", "get_threshold_manager"],
        description="Dynamic threshold management",
    ),
    "centralized_defaults": FeatureSpec(
        name="centralized_defaults",
        module_path="app.coordination.coordination_defaults",
        exports=[
            "CircuitBreakerDefaults",
            "RetryDefaults",
            "TimeoutDefaults",
        ],
        description="Centralized coordination defaults",
    ),
    "bandwidth_config": FeatureSpec(
        name="bandwidth_config",
        module_path="app.coordination.sync_bandwidth",
        exports=["BandwidthConfig", "get_bandwidth_limit"],
        description="Bandwidth configuration for transfers",
    ),
    # ==========================================================================
    # Scheduler & Job Features
    # ==========================================================================
    "job_scheduler": FeatureSpec(
        name="job_scheduler",
        module_path="app.coordination.unified_scheduler",
        exports=["UnifiedScheduler", "get_unified_scheduler"],
        description="Unified job scheduler",
    ),
    "coordinator_registry": FeatureSpec(
        name="coordinator_registry",
        module_path="app.coordination.coordinator_registry",
        exports=["COORDINATOR_REGISTRY", "get_coordinator"],
        description="Coordinator registry",
    ),
    "orchestrator_registry": FeatureSpec(
        name="orchestrator_registry",
        module_path="app.coordination.orchestrator_registry",
        exports=["OrchestratorRegistry", "get_orchestrator_registry"],
        description="Orchestrator registry",
    ),
    # ==========================================================================
    # Data & Sync Features
    # ==========================================================================
    "data_catalog": FeatureSpec(
        name="data_catalog",
        module_path="app.distributed.data_catalog",
        exports=["DataCatalog", "get_data_catalog"],
        description="Cluster-wide data catalog",
    ),
    "quality_extraction": FeatureSpec(
        name="quality_extraction",
        module_path="app.coordination.quality_analysis",
        exports=["extract_quality_metrics", "QualityMetrics"],
        description="Quality metrics extraction",
    ),
    "queue_monitor": FeatureSpec(
        name="queue_monitor",
        module_path="app.coordination.queue_monitor",
        exports=["QueueMonitor", "get_queue_monitor"],
        description="Work queue monitoring",
    ),
    # ==========================================================================
    # Infrastructure Features
    # ==========================================================================
    "ssh_fallback": FeatureSpec(
        name="ssh_fallback",
        module_path="app.core.ssh",
        exports=["run_ssh_command", "SSHConnection"],
        description="SSH command execution",
    ),
    "aiohttp": FeatureSpec(
        name="aiohttp",
        module_path="aiohttp",
        exports=["ClientSession", "web"],
        description="Async HTTP client/server",
    ),
    "yaml": FeatureSpec(
        name="yaml",
        module_path="yaml",
        exports=["safe_load", "safe_dump"],
        description="YAML parsing",
    ),
    "redis": FeatureSpec(
        name="redis",
        module_path="redis",
        exports=["Redis", "ConnectionPool"],
        description="Redis client for distributed locks",
    ),
    # ==========================================================================
    # Protocol & Type Features
    # ==========================================================================
    "protocols": FeatureSpec(
        name="protocols",
        module_path="app.coordination.protocols",
        exports=["HealthCheckResult", "DaemonProtocol"],
        description="Coordination protocols",
    ),
    "centralized_emitters": FeatureSpec(
        name="centralized_emitters",
        module_path="app.coordination.event_router",
        exports=["emit_task_completed", "emit_task_failed"],
        description="Centralized task event emitters (via event_router)",
    ),
}


class CoordinationFeatureRegistry:
    """Registry for checking and accessing optional coordination features.

    Thread-safe singleton pattern.
    """

    _instance: "CoordinationFeatureRegistry | None" = None

    def __new__(cls) -> "CoordinationFeatureRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._status_cache: dict[str, FeatureStatus] = {}
        self._specs = COORDINATION_FEATURE_SPECS.copy()

    def _check_feature(self, feature_name: str) -> FeatureStatus:
        """Check if a feature is available and cache the result."""
        if feature_name in self._status_cache:
            return self._status_cache[feature_name]

        spec = self._specs.get(feature_name)
        if spec is None:
            status = FeatureStatus(
                available=False,
                error=f"Unknown feature: {feature_name}",
            )
            self._status_cache[feature_name] = status
            return status

        try:
            import importlib

            module = importlib.import_module(spec.module_path)

            exports: dict[str, Any] = {}
            for export_name in spec.exports:
                if hasattr(module, export_name):
                    exports[export_name] = getattr(module, export_name)
                else:
                    logger.debug(
                        f"Feature {feature_name}: export {export_name} not found"
                    )

            status = FeatureStatus(available=True, exports=exports)
        except ImportError as e:
            status = FeatureStatus(available=False, error=str(e))
        except Exception as e:
            status = FeatureStatus(available=False, error=f"Unexpected: {e}")

        self._status_cache[feature_name] = status
        return status

    def has_feature(self, feature_name: str) -> bool:
        """Check if a feature is available."""
        return self._check_feature(feature_name).available

    def get_feature(self, feature_name: str, export_name: str) -> Any | None:
        """Get an export from a feature."""
        status = self._check_feature(feature_name)
        if not status.available:
            return None
        return status.exports.get(export_name)

    def require_feature(self, feature_name: str, export_name: str) -> Any:
        """Get an export from a feature, raising if not available."""
        status = self._check_feature(feature_name)
        if not status.available:
            raise ImportError(
                f"Feature {feature_name} not available: {status.error}"
            )
        if export_name not in status.exports:
            raise ImportError(
                f"Export {export_name} not found in feature {feature_name}"
            )
        return status.exports[export_name]

    def get_all_exports(self, feature_name: str) -> dict[str, Any]:
        """Get all exports from a feature."""
        status = self._check_feature(feature_name)
        return status.exports.copy() if status.available else {}

    def get_status_report(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered features."""
        report: dict[str, dict[str, Any]] = {}
        for feature_name in self._specs:
            status = self._check_feature(feature_name)
            report[feature_name] = {
                "available": status.available,
                "exports": list(status.exports.keys()) if status.available else [],
                "error": status.error if not status.available else None,
                "description": self._specs[feature_name].description,
            }
        return report

    def clear_cache(self) -> None:
        """Clear the status cache (useful for testing)."""
        self._status_cache.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


# Module-level convenience functions
_registry: CoordinationFeatureRegistry | None = None


def _get_registry() -> CoordinationFeatureRegistry:
    """Get or create the feature registry singleton."""
    global _registry
    if _registry is None:
        _registry = CoordinationFeatureRegistry()
    return _registry


def has_feature(feature_name: str) -> bool:
    """Check if a coordination feature is available.

    Args:
        feature_name: Name of the feature to check.

    Returns:
        True if the feature is importable, False otherwise.
    """
    return _get_registry().has_feature(feature_name)


def get_feature(feature_name: str, export_name: str) -> Any | None:
    """Get an export from a coordination feature.

    Args:
        feature_name: Name of the feature.
        export_name: Name of the export to retrieve.

    Returns:
        The exported value, or None if not available.
    """
    return _get_registry().get_feature(feature_name, export_name)


def require_feature(feature_name: str, export_name: str) -> Any:
    """Get an export from a feature, raising if not available.

    Args:
        feature_name: Name of the feature.
        export_name: Name of the export to retrieve.

    Returns:
        The exported value.

    Raises:
        ImportError: If the feature is not available.
    """
    return _get_registry().require_feature(feature_name, export_name)


def get_feature_status() -> dict[str, dict[str, Any]]:
    """Get status of all registered coordination features."""
    return _get_registry().get_status_report()


def reset_feature_registry() -> None:
    """Reset the feature registry (for testing)."""
    global _registry
    _registry = None
    CoordinationFeatureRegistry.reset_instance()


# =============================================================================
# Backward-compatible HAS_* constants
# =============================================================================
# These allow gradual migration from scattered HAS_* flags to the registry


def __getattr__(name: str) -> Any:
    """Provide backward-compatible HAS_* constants.

    This allows code like:
        from app.coordination.feature_flags import HAS_CIRCUIT_BREAKER

    To work while migrating to:
        from app.coordination.feature_flags import has_feature
        if has_feature('circuit_breaker'):
            ...
    """
    if name.startswith("HAS_"):
        # Convert HAS_CIRCUIT_BREAKER to circuit_breaker
        feature_name = name[4:].lower()
        return has_feature(feature_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Pre-defined constants for commonly used features
# These are computed at import time for backward compatibility
# New code should use has_feature() instead
HAS_CIRCUIT_BREAKER = has_feature("circuit_breaker")
HAS_DATA_EVENTS = has_feature("data_events")
HAS_AIOHTTP = has_feature("aiohttp")
HAS_YAML = has_feature("yaml")
HAS_REDIS = has_feature("redis")
