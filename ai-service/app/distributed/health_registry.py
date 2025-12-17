"""Health Check Registry for RingRift AI service.

This module provides a centralized registry for health checks, allowing
different components to register their own health checks that are
automatically included in the overall health summary.

Usage:
    from app.distributed.health_registry import (
        register_health_check,
        get_health_summary,
        HealthStatus,
    )

    # Register a custom health check
    @register_health_check("my_component")
    def check_my_component() -> HealthStatus:
        # Check component health
        if is_healthy():
            return HealthStatus.ok("Component is running")
        return HealthStatus.error("Component is down")

    # Get overall health
    summary = get_health_summary()
    print(f"System healthy: {summary.healthy}")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Types
# =============================================================================


class HealthLevel(str, Enum):
    """Health status levels."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class HealthStatus:
    """Health status for a component."""
    level: HealthLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    @classmethod
    def ok(cls, message: str = "Healthy", **details) -> "HealthStatus":
        """Create an OK status."""
        return cls(level=HealthLevel.OK, message=message, details=details)

    @classmethod
    def warning(cls, message: str, **details) -> "HealthStatus":
        """Create a WARNING status."""
        return cls(level=HealthLevel.WARNING, message=message, details=details)

    @classmethod
    def error(cls, message: str, **details) -> "HealthStatus":
        """Create an ERROR status."""
        return cls(level=HealthLevel.ERROR, message=message, details=details)

    @classmethod
    def unknown(cls, message: str = "Status unknown", **details) -> "HealthStatus":
        """Create an UNKNOWN status."""
        return cls(level=HealthLevel.UNKNOWN, message=message, details=details)

    @property
    def healthy(self) -> bool:
        """Check if status is healthy (OK or WARNING)."""
        return self.level in (HealthLevel.OK, HealthLevel.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ComponentHealth:
    """Health information for a registered component."""
    name: str
    status: HealthStatus
    check_duration_ms: float = 0.0
    last_check: Optional[datetime] = None
    check_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.to_dict(),
            "check_duration_ms": self.check_duration_ms,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "check_count": self.check_count,
            "error_count": self.error_count,
        }


@dataclass
class HealthSummary:
    """Overall health summary."""
    healthy: bool
    timestamp: datetime
    components: List[ComponentHealth]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    check_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "healthy": self.healthy,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components],
            "issues": self.issues,
            "warnings": self.warnings,
            "check_duration_ms": self.check_duration_ms,
        }


# =============================================================================
# Health Check Registry
# =============================================================================


# Type for health check functions
HealthCheckFunc = Callable[[], HealthStatus]


class HealthRegistry:
    """Centralized registry for health checks.

    This singleton maintains a registry of health check functions that
    can be called to get the overall health status of the system.
    """

    _instance: Optional["HealthRegistry"] = None
    _lock = threading.RLock()

    def __new__(cls) -> "HealthRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._checks: Dict[str, HealthCheckFunc] = {}
        self._check_history: Dict[str, List[ComponentHealth]] = {}
        self._registry_lock = threading.RLock()

        # Register built-in checks
        self._register_builtin_checks()

    def _register_builtin_checks(self) -> None:
        """Register built-in health checks."""
        self.register("system_resources", self._check_system_resources)
        self.register("database_connectivity", self._check_database)

    def register(
        self,
        name: str,
        check_func: HealthCheckFunc,
        *,
        override: bool = False,
    ) -> None:
        """Register a health check.

        Args:
            name: Unique name for this check
            check_func: Function that returns HealthStatus
            override: Allow overriding existing check
        """
        with self._registry_lock:
            if name in self._checks and not override:
                logger.warning(f"Health check '{name}' already registered")
                return

            self._checks[name] = check_func
            self._check_history[name] = []
            logger.debug(f"Registered health check: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a health check.

        Args:
            name: Name of check to remove

        Returns:
            True if check was removed
        """
        with self._registry_lock:
            if name in self._checks:
                del self._checks[name]
                logger.debug(f"Unregistered health check: {name}")
                return True
            return False

    def list_checks(self) -> List[str]:
        """List all registered check names."""
        with self._registry_lock:
            return list(self._checks.keys())

    def run_check(self, name: str) -> ComponentHealth:
        """Run a specific health check.

        Args:
            name: Name of check to run

        Returns:
            ComponentHealth result
        """
        with self._registry_lock:
            if name not in self._checks:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.unknown(f"Check '{name}' not registered"),
                )

            check_func = self._checks[name]

        start = time.monotonic()
        try:
            status = check_func()
        except Exception as e:
            logger.warning(f"Health check '{name}' failed: {e}")
            status = HealthStatus.error(f"Check failed: {e}")

        duration_ms = (time.monotonic() - start) * 1000

        result = ComponentHealth(
            name=name,
            status=status,
            check_duration_ms=duration_ms,
            last_check=datetime.now(timezone.utc),
        )

        # Update history
        with self._registry_lock:
            if name in self._check_history:
                history = self._check_history[name]
                history.append(result)
                # Keep last 100 checks
                if len(history) > 100:
                    self._check_history[name] = history[-100:]

        return result

    def run_all_checks(self) -> HealthSummary:
        """Run all registered health checks.

        Returns:
            HealthSummary with all results
        """
        start = time.monotonic()
        components: List[ComponentHealth] = []
        issues: List[str] = []
        warnings: List[str] = []

        with self._registry_lock:
            check_names = list(self._checks.keys())

        for name in check_names:
            result = self.run_check(name)
            components.append(result)

            if result.status.level == HealthLevel.ERROR:
                issues.append(f"[{name}] {result.status.message}")
            elif result.status.level == HealthLevel.WARNING:
                warnings.append(f"[{name}] {result.status.message}")

        duration_ms = (time.monotonic() - start) * 1000

        return HealthSummary(
            healthy=len(issues) == 0,
            timestamp=datetime.now(timezone.utc),
            components=components,
            issues=issues,
            warnings=warnings,
            check_duration_ms=duration_ms,
        )

    def get_check_history(self, name: str, limit: int = 10) -> List[ComponentHealth]:
        """Get recent history for a check.

        Args:
            name: Check name
            limit: Max results to return

        Returns:
            List of recent check results
        """
        with self._registry_lock:
            history = self._check_history.get(name, [])
            return history[-limit:]

    # -------------------------------------------------------------------------
    # Built-in checks
    # -------------------------------------------------------------------------

    def _check_system_resources(self) -> HealthStatus:
        """Check system resource usage."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }

            # Check thresholds - 80% max utilization (enforced 2025-12-16)
            issues = []
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 80:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 70:  # 70% limit enforced 2025-12-15
                issues.append(f"High disk usage: {disk.percent}%")

            if issues:
                if any("High" in i for i in issues):
                    return HealthStatus.warning("; ".join(issues), **details)

            return HealthStatus.ok("Resources OK", **details)

        except ImportError:
            return HealthStatus.unknown("psutil not available")
        except Exception as e:
            return HealthStatus.error(f"Resource check failed: {e}")

    def _check_database(self) -> HealthStatus:
        """Check database connectivity."""
        try:
            from app.distributed.db_utils import get_db_path
            import sqlite3

            # Check Elo database
            elo_path = get_db_path("elo", create_dirs=False)
            if elo_path.exists():
                conn = sqlite3.connect(str(elo_path), timeout=5)
                conn.execute("SELECT 1")
                conn.close()
                return HealthStatus.ok("Database connectivity OK", elo_db=str(elo_path))
            else:
                return HealthStatus.warning("Elo database not found", elo_db=str(elo_path))

        except Exception as e:
            return HealthStatus.error(f"Database check failed: {e}")


# =============================================================================
# Module-level functions
# =============================================================================

_registry: Optional[HealthRegistry] = None


def get_registry() -> HealthRegistry:
    """Get the singleton health registry."""
    global _registry
    if _registry is None:
        _registry = HealthRegistry()
    return _registry


def register_health_check(name: str):
    """Decorator to register a health check function.

    Usage:
        @register_health_check("my_component")
        def check_my_component() -> HealthStatus:
            return HealthStatus.ok("All good")
    """
    def decorator(func: HealthCheckFunc) -> HealthCheckFunc:
        get_registry().register(name, func)
        return func
    return decorator


def get_health_summary() -> HealthSummary:
    """Get overall health summary."""
    return get_registry().run_all_checks()


def get_component_health(name: str) -> ComponentHealth:
    """Get health for a specific component."""
    return get_registry().run_check(name)


def list_health_checks() -> List[str]:
    """List all registered health checks."""
    return get_registry().list_checks()


# =============================================================================
# HTTP Health Endpoint Support
# =============================================================================


def health_endpoint_handler() -> Dict[str, Any]:
    """Handler for HTTP health endpoints.

    Returns a dict suitable for JSON serialization.

    Example FastAPI integration:
        from fastapi import FastAPI
        from app.distributed.health_registry import health_endpoint_handler

        app = FastAPI()

        @app.get("/health")
        def health():
            return health_endpoint_handler()
    """
    summary = get_health_summary()

    return {
        "status": "healthy" if summary.healthy else "unhealthy",
        "timestamp": summary.timestamp.isoformat(),
        "components": {c.name: c.status.level.value for c in summary.components},
        "issues": summary.issues,
        "warnings": summary.warnings,
    }


def liveness_check() -> bool:
    """Simple liveness check (is the process running).

    Returns True if the process is alive. Used for Kubernetes
    liveness probes.
    """
    return True


def readiness_check() -> bool:
    """Readiness check (is the service ready to handle requests).

    Returns True if all critical components are healthy.
    Used for Kubernetes readiness probes.
    """
    summary = get_health_summary()
    return summary.healthy
