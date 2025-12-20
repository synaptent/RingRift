"""Unified Registry Facade for RingRift AI (December 2025).

Provides a single entry point for accessing all registry types in the system.
Instead of importing from multiple registry modules, use UnifiedRegistry for:
- Model registration and lookup
- Orchestrator registration and health tracking
- Host/node discovery and health
- Task registry access

Benefits:
- Single import for all registry access
- Consistent interface across registries
- Centralized health monitoring
- Event-driven updates across registries

Usage:
    from app.coordination.unified_registry import (
        UnifiedRegistry,
        get_unified_registry,
    )

    registry = get_unified_registry()

    # Model operations
    models = registry.get_models(board_type="square8", stage="production")
    registry.register_model(model_id, board_type, num_players, metrics)

    # Orchestrator operations
    orchestrators = registry.get_active_orchestrators()
    registry.register_orchestrator(node_id, node_name, capabilities)

    # Health operations
    health = registry.get_cluster_health()

    # Unified status
    status = registry.get_status()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegistryHealth:
    """Health status for a registry component."""

    name: str
    available: bool
    healthy: bool
    item_count: int = 0
    last_access: float = 0.0
    errors_count: int = 0
    error_message: str | None = None


@dataclass
class ClusterHealth:
    """Aggregate health status across all registries."""

    healthy: bool
    registries: list[RegistryHealth] = field(default_factory=list)
    total_items: int = 0
    unhealthy_count: int = 0
    timestamp: float = field(default_factory=time.time)


class UnifiedRegistry:
    """Unified facade for all registry operations.

    Provides centralized access to:
    - ModelRegistry: Model versioning and lifecycle
    - OrchestratorRegistry: Orchestrator coordination
    - HealthRegistry: Node health tracking
    - DynamicHostRegistry: Dynamic host discovery

    Example:
        registry = get_unified_registry()

        # Get production models
        models = registry.get_models(stage="production")

        # Get healthy orchestrators
        orchestrators = registry.get_healthy_orchestrators()

        # Check cluster health
        if registry.is_cluster_healthy():
            print("Cluster is operational")
    """

    def __init__(self):
        """Initialize unified registry with lazy loading."""
        self._model_registry = None
        self._orchestrator_registry = None
        self._health_registry = None
        self._dynamic_registry = None
        self._training_registry = None

        # Track initialization attempts
        self._init_errors: dict[str, str] = {}

    # =========================================================================
    # Model Registry Operations
    # =========================================================================

    def _get_model_registry(self):
        """Lazy load ModelRegistry."""
        if self._model_registry is None:
            try:
                from app.training.model_registry import ModelRegistry
                self._model_registry = ModelRegistry()
            except ImportError as e:
                self._init_errors["model_registry"] = str(e)
                logger.debug(f"[UnifiedRegistry] ModelRegistry not available: {e}")
        return self._model_registry

    def get_models(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        stage: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get models matching criteria.

        Args:
            board_type: Filter by board type
            num_players: Filter by number of players
            stage: Filter by stage (development, staging, production)
            limit: Maximum results

        Returns:
            List of model info dicts
        """
        registry = self._get_model_registry()
        if not registry:
            return []

        try:
            models = registry.list_models(
                board_type=board_type,
                num_players=num_players,
                stage=stage,
                limit=limit,
            )
            return [m.to_dict() if hasattr(m, 'to_dict') else m for m in models]
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get models: {e}")
            return []

    def get_best_model(
        self,
        board_type: str,
        num_players: int,
        stage: str = "production",
    ) -> dict[str, Any] | None:
        """Get the best model for a configuration.

        Args:
            board_type: Board type
            num_players: Number of players
            stage: Stage to search

        Returns:
            Best model info or None
        """
        registry = self._get_model_registry()
        if not registry:
            return None

        try:
            model = registry.get_best_model(
                board_type=board_type,
                num_players=num_players,
                stage=stage,
            )
            if model:
                return model.to_dict() if hasattr(model, 'to_dict') else model
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get best model: {e}")
        return None

    def register_model(
        self,
        board_type: str,
        num_players: int,
        model_path: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Register a new model.

        Args:
            board_type: Board type
            num_players: Number of players
            model_path: Path to model checkpoint
            metrics: Performance metrics
            metadata: Additional metadata

        Returns:
            Model ID if successful
        """
        registry = self._get_model_registry()
        if not registry:
            return None

        try:
            return registry.register_model(
                board_type=board_type,
                num_players=num_players,
                model_path=model_path,
                metrics=metrics,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to register model: {e}")
            return None

    # =========================================================================
    # Orchestrator Registry Operations
    # =========================================================================

    def _get_orchestrator_registry(self):
        """Lazy load OrchestratorRegistry."""
        if self._orchestrator_registry is None:
            try:
                from app.coordination.orchestrator_registry import get_orchestrator_registry
                self._orchestrator_registry = get_orchestrator_registry()
            except ImportError as e:
                self._init_errors["orchestrator_registry"] = str(e)
                logger.debug(f"[UnifiedRegistry] OrchestratorRegistry not available: {e}")
        return self._orchestrator_registry

    def get_active_orchestrators(self) -> list[dict[str, Any]]:
        """Get all active orchestrators.

        Returns:
            List of orchestrator info dicts
        """
        registry = self._get_orchestrator_registry()
        if not registry:
            return []

        try:
            return registry.get_active_orchestrators()
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get orchestrators: {e}")
            return []

    def get_healthy_orchestrators(self) -> list[dict[str, Any]]:
        """Get orchestrators that are healthy and responsive.

        Returns:
            List of healthy orchestrator info
        """
        registry = self._get_orchestrator_registry()
        if not registry:
            return []

        try:
            all_orch = registry.get_active_orchestrators()
            # Filter to those with recent heartbeat
            cutoff = time.time() - 120  # 2 minute timeout
            return [o for o in all_orch if o.get("last_heartbeat", 0) > cutoff]
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get healthy orchestrators: {e}")
            return []

    def register_orchestrator(
        self,
        node_id: str,
        node_name: str,
        orchestrator_type: str = "coordinator",
        capabilities: dict[str, Any] | None = None,
    ) -> bool:
        """Register an orchestrator.

        Args:
            node_id: Unique node identifier
            node_name: Human-readable name
            orchestrator_type: Type (coordinator, scheduler, etc.)
            capabilities: Node capabilities

        Returns:
            True if successful
        """
        registry = self._get_orchestrator_registry()
        if not registry:
            return False

        try:
            registry.register(
                node_id=node_id,
                node_name=node_name,
                orchestrator_type=orchestrator_type,
                capabilities=capabilities or {},
            )
            return True
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to register orchestrator: {e}")
            return False

    # =========================================================================
    # Health Registry Operations
    # =========================================================================

    def _get_health_registry(self):
        """Lazy load HealthRegistry."""
        if self._health_registry is None:
            try:
                from app.distributed.health_registry import get_health_registry
                self._health_registry = get_health_registry()
            except ImportError as e:
                self._init_errors["health_registry"] = str(e)
                logger.debug(f"[UnifiedRegistry] HealthRegistry not available: {e}")
        return self._health_registry

    def get_node_health(self, node_id: str) -> dict[str, Any] | None:
        """Get health status for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            Health info or None
        """
        registry = self._get_health_registry()
        if not registry:
            return None

        try:
            return registry.get_node_health(node_id)
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get node health: {e}")
            return None

    def get_all_node_health(self) -> list[dict[str, Any]]:
        """Get health status for all nodes.

        Returns:
            List of node health info
        """
        registry = self._get_health_registry()
        if not registry:
            return []

        try:
            return registry.get_all_health()
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get all node health: {e}")
            return []

    def update_node_health(
        self,
        node_id: str,
        status: str = "healthy",
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        """Update health status for a node.

        Args:
            node_id: Node identifier
            status: Health status (healthy, degraded, unhealthy)
            metrics: Optional health metrics

        Returns:
            True if successful
        """
        registry = self._get_health_registry()
        if not registry:
            return False

        try:
            registry.update_health(node_id, status, metrics or {})
            return True
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to update node health: {e}")
            return False

    # =========================================================================
    # Dynamic Host Registry Operations
    # =========================================================================

    def _get_dynamic_registry(self):
        """Lazy load DynamicHostRegistry."""
        if self._dynamic_registry is None:
            try:
                from app.distributed.dynamic_registry import get_dynamic_registry
                self._dynamic_registry = get_dynamic_registry()
            except ImportError as e:
                self._init_errors["dynamic_registry"] = str(e)
                logger.debug(f"[UnifiedRegistry] DynamicRegistry not available: {e}")
        return self._dynamic_registry

    def get_available_hosts(self) -> list[dict[str, Any]]:
        """Get all available hosts in the cluster.

        Returns:
            List of host info dicts
        """
        registry = self._get_dynamic_registry()
        if not registry:
            return []

        try:
            return registry.get_available_hosts()
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to get hosts: {e}")
            return []

    def register_host(
        self,
        host: str,
        port: int,
        capabilities: dict[str, Any] | None = None,
    ) -> bool:
        """Register a host.

        Args:
            host: Hostname or IP
            port: Port number
            capabilities: Host capabilities

        Returns:
            True if successful
        """
        registry = self._get_dynamic_registry()
        if not registry:
            return False

        try:
            registry.register_host(host, port, capabilities or {})
            return True
        except Exception as e:
            logger.warning(f"[UnifiedRegistry] Failed to register host: {e}")
            return False

    # =========================================================================
    # Unified Operations
    # =========================================================================

    def get_cluster_health(self) -> ClusterHealth:
        """Get aggregate health status across all registries.

        Returns:
            ClusterHealth with status for each registry
        """
        registry_health = []
        total_items = 0
        unhealthy_count = 0

        # Check model registry
        model_reg = self._get_model_registry()
        if model_reg:
            try:
                stats = model_reg.get_stats() if hasattr(model_reg, 'get_stats') else {}
                health = RegistryHealth(
                    name="model_registry",
                    available=True,
                    healthy=True,
                    item_count=stats.get("total_models", 0),
                )
                total_items += health.item_count
            except Exception as e:
                health = RegistryHealth(
                    name="model_registry",
                    available=True,
                    healthy=False,
                    error_message=str(e),
                )
                unhealthy_count += 1
            registry_health.append(health)
        else:
            registry_health.append(RegistryHealth(
                name="model_registry",
                available=False,
                healthy=False,
                error_message=self._init_errors.get("model_registry"),
            ))

        # Check orchestrator registry
        orch_reg = self._get_orchestrator_registry()
        if orch_reg:
            try:
                orchestrators = orch_reg.get_active_orchestrators()
                health = RegistryHealth(
                    name="orchestrator_registry",
                    available=True,
                    healthy=True,
                    item_count=len(orchestrators),
                )
                total_items += health.item_count
            except Exception as e:
                health = RegistryHealth(
                    name="orchestrator_registry",
                    available=True,
                    healthy=False,
                    error_message=str(e),
                )
                unhealthy_count += 1
            registry_health.append(health)
        else:
            registry_health.append(RegistryHealth(
                name="orchestrator_registry",
                available=False,
                healthy=False,
                error_message=self._init_errors.get("orchestrator_registry"),
            ))

        # Check health registry
        health_reg = self._get_health_registry()
        if health_reg:
            try:
                nodes = health_reg.get_all_health() if hasattr(health_reg, 'get_all_health') else []
                health = RegistryHealth(
                    name="health_registry",
                    available=True,
                    healthy=True,
                    item_count=len(nodes),
                )
                total_items += health.item_count
            except Exception as e:
                health = RegistryHealth(
                    name="health_registry",
                    available=True,
                    healthy=False,
                    error_message=str(e),
                )
                unhealthy_count += 1
            registry_health.append(health)
        else:
            registry_health.append(RegistryHealth(
                name="health_registry",
                available=False,
                healthy=False,
                error_message=self._init_errors.get("health_registry"),
            ))

        # Check dynamic registry
        dyn_reg = self._get_dynamic_registry()
        if dyn_reg:
            try:
                hosts = dyn_reg.get_available_hosts() if hasattr(dyn_reg, 'get_available_hosts') else []
                health = RegistryHealth(
                    name="dynamic_registry",
                    available=True,
                    healthy=True,
                    item_count=len(hosts),
                )
                total_items += health.item_count
            except Exception as e:
                health = RegistryHealth(
                    name="dynamic_registry",
                    available=True,
                    healthy=False,
                    error_message=str(e),
                )
                unhealthy_count += 1
            registry_health.append(health)
        else:
            registry_health.append(RegistryHealth(
                name="dynamic_registry",
                available=False,
                healthy=False,
                error_message=self._init_errors.get("dynamic_registry"),
            ))

        return ClusterHealth(
            healthy=unhealthy_count == 0,
            registries=registry_health,
            total_items=total_items,
            unhealthy_count=unhealthy_count,
        )

    def is_cluster_healthy(self) -> bool:
        """Quick check if cluster is healthy.

        Returns:
            True if all available registries are healthy
        """
        health = self.get_cluster_health()
        return health.healthy

    def get_status(self) -> dict[str, Any]:
        """Get unified status across all registries.

        Returns:
            Dict with status for each registry
        """
        health = self.get_cluster_health()

        return {
            "healthy": health.healthy,
            "total_items": health.total_items,
            "unhealthy_count": health.unhealthy_count,
            "timestamp": health.timestamp,
            "registries": {
                r.name: {
                    "available": r.available,
                    "healthy": r.healthy,
                    "item_count": r.item_count,
                    "error": r.error_message,
                }
                for r in health.registries
            },
            "init_errors": self._init_errors,
        }

    def subscribe_to_changes(
        self,
        callback: Callable[[str, str, Any], None],
    ) -> None:
        """Subscribe to changes across all registries.

        Args:
            callback: Function(registry_name, event_type, data)
        """
        # Wire callback to each registry
        model_reg = self._get_model_registry()
        if model_reg and hasattr(model_reg, 'on_change'):
            model_reg.on_change(lambda et, d: callback("model_registry", et, d))

        orch_reg = self._get_orchestrator_registry()
        if orch_reg and hasattr(orch_reg, 'on_change'):
            orch_reg.on_change(lambda et, d: callback("orchestrator_registry", et, d))

        health_reg = self._get_health_registry()
        if health_reg and hasattr(health_reg, 'on_change'):
            health_reg.on_change(lambda et, d: callback("health_registry", et, d))


# =============================================================================
# Singleton Management
# =============================================================================

_unified_registry: UnifiedRegistry | None = None


def get_unified_registry() -> UnifiedRegistry:
    """Get the global UnifiedRegistry singleton.

    Returns:
        UnifiedRegistry instance
    """
    global _unified_registry
    if _unified_registry is None:
        _unified_registry = UnifiedRegistry()
    return _unified_registry


def reset_unified_registry() -> None:
    """Reset the unified registry singleton (for testing)."""
    global _unified_registry
    _unified_registry = None


__all__ = [
    "ClusterHealth",
    "RegistryHealth",
    # Main class
    "UnifiedRegistry",
    # Singleton access
    "get_unified_registry",
    "reset_unified_registry",
]
