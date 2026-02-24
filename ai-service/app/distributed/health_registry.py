"""Health Check Registry for RingRift AI service.

This module provides a centralized registry for health checks, allowing
different components to register their own health checks that are
automatically included in the overall health summary.

Usage:
    from app.distributed.health_registry import (
        register_health_check,
        get_health_summary,
        ComponentHealthStatus,
    )

    # Register a custom health check
    @register_health_check("my_component")
    def check_my_component() -> ComponentHealthStatus:
        # Check component health
        if is_healthy():
            return ComponentHealthStatus.ok("Component is running")
        return ComponentHealthStatus.error("Component is down")

    # Get overall health
    summary = get_health_summary()
    print(f"System healthy: {summary.healthy}")
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.config.thresholds import DISK_SYNC_TARGET_PERCENT

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Types
# =============================================================================


class HealthLevel(str, Enum):
    """Health status levels.

    DEPRECATED (January 2026): Use app.coordination.health.HealthStatus instead.
    This enum will be removed in Q2 2026.

    Migration:
        from app.coordination.health import HealthStatus, from_legacy_health_level

        # Convert existing HealthLevel to HealthStatus
        status = from_legacy_health_level(HealthLevel.OK)  # Returns HealthStatus.HEALTHY

        # Or use HealthStatus directly
        status = HealthStatus.HEALTHY

    Mapping:
        OK -> HealthStatus.HEALTHY
        WARNING -> HealthStatus.DEGRADED
        ERROR -> HealthStatus.UNHEALTHY
        UNKNOWN -> HealthStatus.UNKNOWN
    """
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"

    def to_health_status(self):
        """Convert to canonical HealthStatus.

        Returns:
            HealthStatus equivalent of this HealthLevel.
        """
        from app.coordination.health import from_legacy_health_level
        return from_legacy_health_level(self)


@dataclass
class ComponentHealthStatus:
    """Health status for a component.

    Note: This is component-specific health tracking with HealthLevel.
    For generic health states (HEALTHY, DEGRADED, etc.), use app.core.health.HealthState.
    """
    level: HealthLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    @classmethod
    def ok(cls, message: str = "Healthy", **details) -> ComponentHealthStatus:
        """Create an OK status."""
        return cls(level=HealthLevel.OK, message=message, details=details)

    @classmethod
    def warning(cls, message: str, **details) -> ComponentHealthStatus:
        """Create a WARNING status."""
        return cls(level=HealthLevel.WARNING, message=message, details=details)

    @classmethod
    def error(cls, message: str, **details) -> ComponentHealthStatus:
        """Create an ERROR status."""
        return cls(level=HealthLevel.ERROR, message=message, details=details)

    @classmethod
    def unknown(cls, message: str = "Status unknown", **details) -> ComponentHealthStatus:
        """Create an UNKNOWN status."""
        return cls(level=HealthLevel.UNKNOWN, message=message, details=details)

    @property
    def healthy(self) -> bool:
        """Check if status is healthy (OK or WARNING)."""
        return self.level in (HealthLevel.OK, HealthLevel.WARNING)

    def to_dict(self) -> dict[str, Any]:
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
    status: ComponentHealthStatus
    check_duration_ms: float = 0.0
    last_check: datetime | None = None
    check_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
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
    components: list[ComponentHealth]
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    check_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
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
HealthCheckFunc = Callable[[], ComponentHealthStatus]


class HealthRegistry:
    """Centralized registry for health checks.

    This singleton maintains a registry of health check functions that
    can be called to get the overall health status of the system.
    """

    _instance: HealthRegistry | None = None
    _lock = threading.RLock()

    def __new__(cls) -> HealthRegistry:
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
        self._checks: dict[str, HealthCheckFunc] = {}
        self._check_history: dict[str, list[ComponentHealth]] = {}
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
            check_func: Function that returns ComponentHealthStatus
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

    def list_checks(self) -> list[str]:
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
                    status=ComponentHealthStatus.unknown(f"Check '{name}' not registered"),
                )

            check_func = self._checks[name]

        start = time.monotonic()
        try:
            status = check_func()
        except Exception as e:
            logger.warning(f"Health check '{name}' failed: {e}")
            status = ComponentHealthStatus.error(f"Check failed: {e}")

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
        components: list[ComponentHealth] = []
        issues: list[str] = []
        warnings: list[str] = []

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

    def get_check_history(self, name: str, limit: int = 10) -> list[ComponentHealth]:
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

    def _check_system_resources(self) -> ComponentHealthStatus:
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
            if disk.percent > DISK_SYNC_TARGET_PERCENT:
                issues.append(f"High disk usage: {disk.percent}%")

            if issues and any("High" in i for i in issues):
                return ComponentHealthStatus.warning("; ".join(issues), **details)

            return ComponentHealthStatus.ok("Resources OK", **details)

        except ImportError:
            return ComponentHealthStatus.unknown("psutil not available")
        except Exception as e:
            return ComponentHealthStatus.error(f"Resource check failed: {e}")

    def _check_database(self) -> ComponentHealthStatus:
        """Check database connectivity."""
        try:
            import sqlite3

            from app.distributed.db_utils import get_db_path

            # Check Elo database
            elo_path = get_db_path("elo", create_dirs=False)
            if elo_path.exists():
                conn = sqlite3.connect(str(elo_path), timeout=5)
                conn.execute("SELECT 1")
                conn.close()
                return ComponentHealthStatus.ok("Database connectivity OK", elo_db=str(elo_path))
            else:
                return ComponentHealthStatus.warning("Elo database not found", elo_db=str(elo_path))

        except Exception as e:
            return ComponentHealthStatus.error(f"Database check failed: {e}")


# =============================================================================
# Module-level functions
# =============================================================================

_registry: HealthRegistry | None = None


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
        def check_my_component() -> ComponentHealthStatus:
            return ComponentHealthStatus.ok("All good")
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


def list_health_checks() -> list[str]:
    """List all registered health checks."""
    return get_registry().list_checks()


# =============================================================================
# HTTP Health Endpoint Support
# =============================================================================


def health_endpoint_handler() -> dict[str, Any]:
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


# =============================================================================
# Default Health Checks (registered at module load)
# =============================================================================

@register_health_check("memory")
def check_memory() -> ComponentHealthStatus:
    """Check system memory usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        pct = mem.percent

        if pct >= 95:
            return ComponentHealthStatus.error(
                f"Memory critically high: {pct:.1f}%",
                percent=pct,
                available_mb=mem.available / (1024 * 1024),
            )
        elif pct >= 85:
            return ComponentHealthStatus.warning(
                f"Memory high: {pct:.1f}%",
                percent=pct,
                available_mb=mem.available / (1024 * 1024),
            )
        return ComponentHealthStatus.ok(
            f"Memory OK: {pct:.1f}%",
            percent=pct,
            available_mb=mem.available / (1024 * 1024),
        )
    except ImportError:
        return ComponentHealthStatus.unknown("psutil not installed")
    except Exception as e:
        return ComponentHealthStatus.error(f"Memory check failed: {e}")


@register_health_check("disk")
def check_disk() -> ComponentHealthStatus:
    """Check disk usage for data directory."""
    try:
        import psutil
        from app.utils.paths import DATA_DIR

        disk = psutil.disk_usage(str(DATA_DIR))
        pct = disk.percent

        if pct >= 95:
            return ComponentHealthStatus.error(
                f"Disk critically full: {pct:.1f}%",
                percent=pct,
                free_gb=disk.free / (1024 ** 3),
                path=str(DATA_DIR),
            )
        elif pct >= 85:
            return ComponentHealthStatus.warning(
                f"Disk usage high: {pct:.1f}%",
                percent=pct,
                free_gb=disk.free / (1024 ** 3),
                path=str(DATA_DIR),
            )
        return ComponentHealthStatus.ok(
            f"Disk OK: {pct:.1f}%",
            percent=pct,
            free_gb=disk.free / (1024 ** 3),
            path=str(DATA_DIR),
        )
    except ImportError:
        return ComponentHealthStatus.unknown("psutil not installed")
    except Exception as e:
        return ComponentHealthStatus.error(f"Disk check failed: {e}")


@register_health_check("database")
def check_database() -> ComponentHealthStatus:
    """Check SQLite database connectivity."""
    try:
        import sqlite3
        from app.utils.paths import DATA_DIR

        # Try to connect to the main games database
        games_dir = DATA_DIR / "games"
        if not games_dir.exists():
            return ComponentHealthStatus.warning("No games directory found", path=str(games_dir))

        db_files = list(games_dir.glob("*.db"))
        if not db_files:
            return ComponentHealthStatus.ok("No databases to check", db_count=0)

        # Quick connectivity test - find first valid SQLite database
        valid_count = 0
        invalid_files = []
        test_db = None

        for db_file in db_files[:5]:  # Only check first 5 to keep it fast
            conn = None
            try:
                conn = sqlite3.connect(str(db_file), timeout=2)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                valid_count += 1
                if test_db is None:
                    test_db = db_file
            except sqlite3.DatabaseError:
                invalid_files.append(db_file.name)
            finally:
                if conn is not None:
                    conn.close()

        if valid_count == 0:
            return ComponentHealthStatus.warning(
                "No valid SQLite databases found",
                db_count=len(db_files),
                invalid_files=invalid_files[:3],
            )

        if invalid_files:
            return ComponentHealthStatus.warning(
                f"Database OK but {len(invalid_files)} invalid files",
                valid_count=valid_count,
                total_count=len(db_files),
                invalid_files=invalid_files[:3],
            )

        return ComponentHealthStatus.ok(
            f"Database OK ({len(db_files)} DBs found)",
            db_count=len(db_files),
            test_db=test_db.name if test_db else None,
        )
    except Exception as e:
        return ComponentHealthStatus.error(f"Database check failed: {e}")
