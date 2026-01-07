"""Hierarchical cascade circuit breaker for daemon restarts.

December 30, 2025: Replaces the global cascade circuit breaker with a
hierarchical, category-based system that provides granular control over
daemon restarts while maintaining system stability.

Design Principles:
- Per-category circuit breakers with independent thresholds and cooldowns
- Critical daemon exemptions (always allowed to restart)
- Startup grace period to prevent blocking during normal initialization
- Root cause analysis for debugging cascades

Usage:
    from app.coordination.cascade_breaker import (
        CascadeBreakerManager,
        get_cascade_breaker,
    )

    # Get singleton instance
    breaker = get_cascade_breaker()

    # Check if restart is allowed
    allowed, reason = breaker.can_restart(daemon_type)
    if not allowed:
        logger.info(f"Restart blocked: {reason}")
        return

    # Record the restart
    breaker.record_restart(daemon_type)
    await restart_daemon(daemon_type)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from app.coordination.daemon_types import (
    CRITICAL_DAEMONS,
    DaemonCategory,
    DaemonType,
    get_daemon_category,
)
from app.coordination.singleton_mixin import SingletonMixin

logger = logging.getLogger(__name__)

__all__ = [
    "CategoryBreakerConfig",
    "CategoryBreakerState",
    "CascadeBreakerConfig",
    "CascadeBreakerManager",
    "get_cascade_breaker",
    "reset_cascade_breaker",
]


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class CategoryBreakerConfig:
    """Configuration for a single category's circuit breaker.

    Attributes:
        threshold: Max restarts in window before breaker opens.
        window_seconds: Time window for counting restarts.
        cooldown_seconds: How long breaker stays open after tripping.
        exempt_from_global: If True, category is never blocked by global breaker.
    """

    threshold: int = 5
    window_seconds: int = 300  # 5 minutes
    cooldown_seconds: int = 60
    exempt_from_global: bool = False


# Default configurations per category
# Categories with exempt_from_global=True are critical for system operation
DEFAULT_CATEGORY_CONFIGS: dict[DaemonCategory, CategoryBreakerConfig] = {
    # Critical categories - high threshold, short cooldown, exempt from global
    DaemonCategory.EVENT: CategoryBreakerConfig(
        threshold=10,
        window_seconds=300,
        cooldown_seconds=30,
        exempt_from_global=True,
    ),
    DaemonCategory.PIPELINE: CategoryBreakerConfig(
        threshold=8,
        window_seconds=300,
        cooldown_seconds=45,
        exempt_from_global=True,
    ),
    DaemonCategory.FEEDBACK: CategoryBreakerConfig(
        threshold=8,
        window_seconds=300,
        cooldown_seconds=45,
        exempt_from_global=True,
    ),
    DaemonCategory.AUTONOMOUS: CategoryBreakerConfig(
        threshold=8,
        window_seconds=300,
        cooldown_seconds=30,
        exempt_from_global=True,
    ),
    # Standard categories - moderate thresholds
    DaemonCategory.SYNC: CategoryBreakerConfig(
        threshold=6,
        window_seconds=300,
        cooldown_seconds=60,
        exempt_from_global=False,
    ),
    DaemonCategory.HEALTH: CategoryBreakerConfig(
        threshold=6,
        window_seconds=300,
        cooldown_seconds=60,
        exempt_from_global=False,
    ),
    DaemonCategory.QUEUE: CategoryBreakerConfig(
        threshold=5,
        window_seconds=300,
        cooldown_seconds=60,
        exempt_from_global=False,
    ),
    DaemonCategory.RESOURCE: CategoryBreakerConfig(
        threshold=5,
        window_seconds=300,
        cooldown_seconds=90,
        exempt_from_global=False,
    ),
    # Less critical categories - lower thresholds, longer cooldowns
    DaemonCategory.EVALUATION: CategoryBreakerConfig(
        threshold=5,
        window_seconds=300,
        cooldown_seconds=90,
        exempt_from_global=False,
    ),
    DaemonCategory.DISTRIBUTION: CategoryBreakerConfig(
        threshold=4,
        window_seconds=300,
        cooldown_seconds=90,
        exempt_from_global=False,
    ),
    # Session 17.48: Increased thresholds and window to prevent restart-looping
    # Recovery daemons need more restarts to recover from persistent issues
    DaemonCategory.RECOVERY: CategoryBreakerConfig(
        threshold=8,  # Allow more restarts before tripping (was 4)
        window_seconds=600,  # 10-minute window (was 300s, more forgiving)
        cooldown_seconds=180,  # 3-minute cooldown (was 120s, more time to fix issues)
        exempt_from_global=False,
    ),
    DaemonCategory.PROVIDER: CategoryBreakerConfig(
        threshold=3,
        window_seconds=300,
        cooldown_seconds=120,
        exempt_from_global=False,
    ),
    DaemonCategory.MISC: CategoryBreakerConfig(
        threshold=4,
        window_seconds=300,
        cooldown_seconds=120,
        exempt_from_global=False,
    ),
}


@dataclass(frozen=True)
class CascadeBreakerConfig:
    """Configuration for the cascade breaker manager.

    Attributes:
        global_threshold: Max total restarts before global breaker trips.
        global_window_seconds: Time window for global restart counting.
        global_cooldown_seconds: Global breaker cooldown duration.
        startup_grace_period: Seconds after start with higher threshold.
        startup_threshold: Threshold to use during startup grace period.
        category_configs: Per-category breaker configurations.
        critical_exempt_daemons: Daemon names that always bypass all breakers.
    """

    # Session 17.48: Relaxed thresholds for stability
    global_threshold: int = 25  # 25 restarts in window (was 15)
    global_window_seconds: int = 300  # 5 minutes
    global_cooldown_seconds: int = 120  # 2 minutes
    startup_grace_period: int = 300  # 5 minutes (was 180s, more time for 112 daemon types)
    startup_threshold: int = 100  # Allow 100 restarts during init (was 50)

    category_configs: dict[DaemonCategory, CategoryBreakerConfig] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_CONFIGS)
    )

    critical_exempt_daemons: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "event_router",  # Core event bus
                "daemon_watchdog",  # Self-healing
                "data_pipeline",  # Data flow
                "feedback_loop",  # Training signals
                "memory_monitor",  # OOM prevention
            }
        )
    )


# =============================================================================
# State Tracking
# =============================================================================


@dataclass
class CategoryBreakerState:
    """Runtime state for a category's circuit breaker.

    Attributes:
        restart_timestamps: List of restart timestamps in the current window.
        breaker_open: Whether the breaker is currently open (blocking).
        opened_at: Timestamp when the breaker was opened.
        total_restarts: Total restarts ever recorded for this category.
        total_blocked: Total restart attempts blocked by this breaker.
    """

    restart_timestamps: list[float] = field(default_factory=list)
    breaker_open: bool = False
    opened_at: float = 0.0
    total_restarts: int = 0
    total_blocked: int = 0


# =============================================================================
# CascadeBreakerManager
# =============================================================================


class CascadeBreakerManager(SingletonMixin):
    """Hierarchical cascade circuit breaker manager.

    Provides per-category circuit breakers with independent thresholds and
    cooldowns, plus a global fallback breaker. Critical daemons can be
    exempted from all breakers.

    January 2026: Migrated to use SingletonMixin for consistency.

    Architecture:
        ┌─────────────────────────────────────┐
        │      CascadeBreakerManager          │
        │                                     │
        │  Global Breaker (fallback only)     │
        │         ↓ (if category not exempt)  │
        │                                     │
        ├───────────┬───────────┬─────────────┤
        │  EVENT    │   SYNC    │  PIPELINE   │
        │ (exempt)  │           │  (exempt)   │
        │ thresh=10 │ thresh=6  │ thresh=8    │
        │ cool=30s  │ cool=60s  │ cool=45s    │
        └───────────┴───────────┴─────────────┘

    Usage:
        >>> breaker = CascadeBreakerManager()
        >>> allowed, reason = breaker.can_restart(DaemonType.AUTO_SYNC)
        >>> if allowed:
        ...     breaker.record_restart(DaemonType.AUTO_SYNC)
        ...     await restart_daemon(DaemonType.AUTO_SYNC)
        ... else:
        ...     logger.warning(f"Restart blocked: {reason}")
    """

    def __init__(self, config: CascadeBreakerConfig | None = None) -> None:
        """Initialize the cascade breaker manager.

        Args:
            config: Optional configuration override. If None, uses defaults.
        """
        self._config = config or CascadeBreakerConfig()
        self._start_time = time.time()

        # Per-category state
        self._category_states: dict[DaemonCategory, CategoryBreakerState] = {
            category: CategoryBreakerState() for category in DaemonCategory
        }

        # Global state
        self._global_restart_timestamps: list[float] = []
        self._global_breaker_open: bool = False
        self._global_opened_at: float = 0.0

        # Stats
        self._total_allowed: int = 0
        self._total_blocked: int = 0
        self._blocked_by_category: dict[DaemonCategory, int] = {
            category: 0 for category in DaemonCategory
        }
        self._blocked_by_global: int = 0

    def can_restart(
        self,
        daemon_type: DaemonType,
    ) -> tuple[bool, str]:
        """Check if a daemon restart is allowed.

        Checks in order:
        1. Critical daemon exemption - always allowed
        2. Category-level breaker - blocked if category breaker open
        3. Global breaker (unless category exempt) - blocked if global open

        Args:
            daemon_type: The daemon type to check.

        Returns:
            Tuple of (allowed, reason). If allowed is True, reason is "allowed".
            If False, reason describes which breaker blocked it.

        Example:
            >>> allowed, reason = breaker.can_restart(DaemonType.AUTO_SYNC)
            >>> print(f"Allowed: {allowed}, Reason: {reason}")
            Allowed: False, Reason: category_sync_breaker_open
        """
        daemon_name = daemon_type.value
        category = get_daemon_category(daemon_type)
        current_time = time.time()

        # 1. Critical daemon exemption - always allowed
        if daemon_name in self._config.critical_exempt_daemons:
            self._total_allowed += 1
            return (True, "critical_daemon_exempt")

        # Check if daemon is in CRITICAL_DAEMONS set (from daemon_types.py)
        if daemon_type in CRITICAL_DAEMONS:
            self._total_allowed += 1
            return (True, "critical_daemon_set")

        # 2. Check startup grace period
        uptime = current_time - self._start_time
        if uptime < self._config.startup_grace_period:
            # During startup, use much higher threshold
            self._total_allowed += 1
            return (True, "startup_grace_period")

        # 3. Check category-level breaker
        cat_state = self._category_states[category]
        cat_config = self._config.category_configs.get(
            category, CategoryBreakerConfig()
        )

        if cat_state.breaker_open:
            elapsed = current_time - cat_state.opened_at
            if elapsed >= cat_config.cooldown_seconds:
                # Cooldown expired, close breaker
                cat_state.breaker_open = False
                cat_state.opened_at = 0.0
                logger.info(
                    f"[CascadeBreaker] Category {category.value} breaker closed "
                    f"after {cat_config.cooldown_seconds}s cooldown"
                )
            else:
                # Still in cooldown
                self._total_blocked += 1
                cat_state.total_blocked += 1
                self._blocked_by_category[category] += 1
                remaining = cat_config.cooldown_seconds - elapsed
                return (
                    False,
                    f"category_{category.value}_breaker_open (cooldown {remaining:.0f}s)",
                )

        # 4. Check global breaker (unless category exempt)
        if not cat_config.exempt_from_global:
            if self._global_breaker_open:
                elapsed = current_time - self._global_opened_at
                if elapsed >= self._config.global_cooldown_seconds:
                    # Cooldown expired, close breaker
                    self._global_breaker_open = False
                    self._global_opened_at = 0.0
                    logger.info(
                        f"[CascadeBreaker] Global breaker closed after "
                        f"{self._config.global_cooldown_seconds}s cooldown"
                    )
                else:
                    # Still in cooldown
                    self._total_blocked += 1
                    self._blocked_by_global += 1
                    remaining = self._config.global_cooldown_seconds - elapsed
                    return (
                        False,
                        f"global_breaker_open (cooldown {remaining:.0f}s)",
                    )

        self._total_allowed += 1
        return (True, "allowed")

    def record_restart(self, daemon_type: DaemonType) -> None:
        """Record a daemon restart and check thresholds.

        Should be called AFTER can_restart() returns True and BEFORE
        actually restarting the daemon.

        Args:
            daemon_type: The daemon type being restarted.

        Side effects:
            - Updates restart timestamps for category and global
            - May trip category or global breaker if threshold exceeded
            - Emits log messages when breakers trip
        """
        current_time = time.time()
        category = get_daemon_category(daemon_type)
        cat_state = self._category_states[category]
        cat_config = self._config.category_configs.get(
            category, CategoryBreakerConfig()
        )

        # Record in category
        cat_state.restart_timestamps.append(current_time)
        cat_state.total_restarts += 1

        # Record in global
        self._global_restart_timestamps.append(current_time)

        # Prune old timestamps from category
        cutoff = current_time - cat_config.window_seconds
        cat_state.restart_timestamps = [
            ts for ts in cat_state.restart_timestamps if ts > cutoff
        ]

        # Check category threshold
        recent_category = len(cat_state.restart_timestamps)
        if recent_category >= cat_config.threshold and not cat_state.breaker_open:
            cat_state.breaker_open = True
            cat_state.opened_at = current_time
            logger.warning(
                f"[CascadeBreaker] CATEGORY BREAKER TRIPPED: {category.value}! "
                f"{recent_category} restarts in {cat_config.window_seconds}s "
                f"(threshold: {cat_config.threshold}). "
                f"Blocking category for {cat_config.cooldown_seconds}s."
            )
            self._emit_category_breaker_event(category, recent_category)

        # Prune old timestamps from global
        global_cutoff = current_time - self._config.global_window_seconds
        self._global_restart_timestamps = [
            ts for ts in self._global_restart_timestamps if ts > global_cutoff
        ]

        # Check global threshold
        recent_global = len(self._global_restart_timestamps)
        if recent_global >= self._config.global_threshold and not self._global_breaker_open:
            self._global_breaker_open = True
            self._global_opened_at = current_time
            logger.warning(
                f"[CascadeBreaker] GLOBAL BREAKER TRIPPED! "
                f"{recent_global} restarts in {self._config.global_window_seconds}s "
                f"(threshold: {self._config.global_threshold}). "
                f"Blocking non-exempt categories for {self._config.global_cooldown_seconds}s."
            )
            self._emit_global_breaker_event(recent_global)

    def get_status(self) -> dict:
        """Get comprehensive status of all breakers.

        Returns:
            Dict with global and per-category breaker states.

        Example:
            >>> status = breaker.get_status()
            >>> print(status["global"]["breaker_open"])
            False
            >>> print(status["categories"]["sync"]["recent_restarts"])
            3
        """
        current_time = time.time()
        uptime = current_time - self._start_time

        # Global status
        global_cutoff = current_time - self._config.global_window_seconds
        recent_global = sum(
            1 for ts in self._global_restart_timestamps if ts > global_cutoff
        )

        result = {
            "uptime_seconds": uptime,
            "in_startup_grace": uptime < self._config.startup_grace_period,
            "total_allowed": self._total_allowed,
            "total_blocked": self._total_blocked,
            "global": {
                "breaker_open": self._global_breaker_open,
                "recent_restarts": recent_global,
                "threshold": self._config.global_threshold,
                "window_seconds": self._config.global_window_seconds,
                "blocked_count": self._blocked_by_global,
            },
            "categories": {},
        }

        if self._global_breaker_open:
            elapsed = current_time - self._global_opened_at
            result["global"]["cooldown_remaining"] = max(
                0, self._config.global_cooldown_seconds - elapsed
            )

        # Per-category status
        for category in DaemonCategory:
            cat_state = self._category_states[category]
            cat_config = self._config.category_configs.get(
                category, CategoryBreakerConfig()
            )

            cutoff = current_time - cat_config.window_seconds
            recent = sum(1 for ts in cat_state.restart_timestamps if ts > cutoff)

            cat_status = {
                "breaker_open": cat_state.breaker_open,
                "recent_restarts": recent,
                "threshold": cat_config.threshold,
                "cooldown_seconds": cat_config.cooldown_seconds,
                "exempt_from_global": cat_config.exempt_from_global,
                "total_restarts": cat_state.total_restarts,
                "total_blocked": cat_state.total_blocked,
            }

            if cat_state.breaker_open:
                elapsed = current_time - cat_state.opened_at
                cat_status["cooldown_remaining"] = max(
                    0, cat_config.cooldown_seconds - elapsed
                )

            result["categories"][category.value] = cat_status

        return result

    def reset(self) -> None:
        """Reset all breaker states.

        Useful for testing or manual recovery.
        """
        for category in DaemonCategory:
            self._category_states[category] = CategoryBreakerState()

        self._global_restart_timestamps = []
        self._global_breaker_open = False
        self._global_opened_at = 0.0

        logger.info("[CascadeBreaker] All breakers reset")

    def _emit_category_breaker_event(
        self, category: DaemonCategory, restart_count: int
    ) -> None:
        """Emit event when a category breaker trips."""
        try:
            from app.coordination.safe_event_emitter import safe_emit_event

            safe_emit_event(
                "CATEGORY_BREAKER_TRIPPED",
                {
                    "category": category.value,
                    "restart_count": restart_count,
                    "threshold": self._config.category_configs[category].threshold,
                    "cooldown_seconds": self._config.category_configs[
                        category
                    ].cooldown_seconds,
                },
                source="CascadeBreakerManager",
            )
        except ImportError:
            pass  # Event emission optional

    def _emit_global_breaker_event(self, restart_count: int) -> None:
        """Emit event when global breaker trips."""
        try:
            from app.coordination.safe_event_emitter import safe_emit_event

            safe_emit_event(
                "GLOBAL_BREAKER_TRIPPED",
                {
                    "restart_count": restart_count,
                    "threshold": self._config.global_threshold,
                    "cooldown_seconds": self._config.global_cooldown_seconds,
                },
                source="CascadeBreakerManager",
            )
        except ImportError:
            pass  # Event emission optional


# =============================================================================
# Singleton Access
# =============================================================================


def get_cascade_breaker() -> CascadeBreakerManager:
    """Get the singleton CascadeBreakerManager instance.

    January 2026: Now uses SingletonMixin.get_instance() (thread-safe, sync).

    Returns:
        The global CascadeBreakerManager instance.

    Example:
        >>> breaker = get_cascade_breaker()
        >>> allowed, reason = breaker.can_restart(DaemonType.AUTO_SYNC)
    """
    # Check for environment variable overrides on first instantiation
    config = _load_config_from_env()
    return CascadeBreakerManager.get_instance(config)


def reset_cascade_breaker() -> None:
    """Reset the singleton instance.

    Useful for testing.
    """
    if CascadeBreakerManager.has_instance():
        CascadeBreakerManager.get_instance().reset()
    CascadeBreakerManager.reset_instance()


def _load_config_from_env() -> CascadeBreakerConfig | None:
    """Load configuration from environment variables.

    Environment variables:
        RINGRIFT_CASCADE_GLOBAL_THRESHOLD: int
        RINGRIFT_CASCADE_GLOBAL_COOLDOWN: int (seconds)
        RINGRIFT_CASCADE_STARTUP_GRACE: int (seconds)
        RINGRIFT_CASCADE_STARTUP_THRESHOLD: int

    Returns:
        CascadeBreakerConfig if any overrides found, None otherwise.
    """
    overrides = {}

    if val := os.environ.get("RINGRIFT_CASCADE_GLOBAL_THRESHOLD"):
        try:
            overrides["global_threshold"] = int(val)
        except ValueError:
            logger.warning(f"Invalid RINGRIFT_CASCADE_GLOBAL_THRESHOLD: {val}")

    if val := os.environ.get("RINGRIFT_CASCADE_GLOBAL_COOLDOWN"):
        try:
            overrides["global_cooldown_seconds"] = int(val)
        except ValueError:
            logger.warning(f"Invalid RINGRIFT_CASCADE_GLOBAL_COOLDOWN: {val}")

    if val := os.environ.get("RINGRIFT_CASCADE_STARTUP_GRACE"):
        try:
            overrides["startup_grace_period"] = int(val)
        except ValueError:
            logger.warning(f"Invalid RINGRIFT_CASCADE_STARTUP_GRACE: {val}")

    if val := os.environ.get("RINGRIFT_CASCADE_STARTUP_THRESHOLD"):
        try:
            overrides["startup_threshold"] = int(val)
        except ValueError:
            logger.warning(f"Invalid RINGRIFT_CASCADE_STARTUP_THRESHOLD: {val}")

    if overrides:
        return CascadeBreakerConfig(**overrides)
    return None
