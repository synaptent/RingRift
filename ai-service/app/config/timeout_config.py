"""Centralized timeout and threshold configuration.

All coordination module timeouts should import from here to ensure
consistent configuration and easy tuning.

Usage:
    from app.config.timeout_config import TIMEOUTS

    timeout = TIMEOUTS.SYNC_INTERVAL
    # or
    from app.config.timeout_config import get_timeout
    timeout = get_timeout("sync_interval")

Environment Variable Overrides:
    Each timeout can be overridden via environment variable:
    RINGRIFT_TIMEOUT_{NAME} = value

    Example: RINGRIFT_TIMEOUT_SYNC_INTERVAL=600

Created: December 2025
Purpose: Consolidate 25+ hardcoded timeout values (Wave 3 Phase 1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar


def _env_float(name: str, default: float) -> float:
    """Get float from environment with fallback."""
    env_key = f"RINGRIFT_TIMEOUT_{name.upper()}"
    val = os.environ.get(env_key)
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_int(name: str, default: int) -> int:
    """Get int from environment with fallback."""
    env_key = f"RINGRIFT_TIMEOUT_{name.upper()}"
    val = os.environ.get(env_key)
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


@dataclass(frozen=True)
class TimeoutDefaults:
    """Centralized timeout and threshold defaults.

    All values can be overridden via environment variables with prefix
    RINGRIFT_TIMEOUT_ followed by the constant name.

    Categories:
        - Lock timeouts: Distributed lock acquisition
        - Health monitoring: Heartbeat and recovery
        - Sync intervals: Data synchronization
        - Daemon lifecycle: Restart and backoff
        - HTTP/Transport: Network operations
        - Scan intervals: Periodic checks
        - Cooldowns: Rate limiting
    """

    # =========================================================================
    # Lock Timeouts
    # =========================================================================

    # Maximum time a lock can be held (prevents deadlocks)
    LOCK_TIMEOUT_SECONDS: int = 3600  # 1 hour

    # Maximum time to wait for lock acquisition
    LOCK_ACQUIRE_TIMEOUT: int = 30  # 30 seconds

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    # Time without heartbeat before node considered unhealthy
    HEARTBEAT_THRESHOLD: float = 120.0  # 2 minutes

    # Minimum time between recovery attempts for same node
    RECOVERY_COOLDOWN: int = 300  # 5 minutes

    # Time before escalating to higher-severity recovery
    ESCALATION_COOLDOWN: int = 3600  # 1 hour

    # Time window for counting error bursts
    PAUSE_ERROR_BURST_WINDOW: int = 300  # 5 minutes

    # Delay before resuming after pause condition clears
    RESUME_DELAY_SECONDS: int = 120  # 2 minutes

    # =========================================================================
    # Sync Intervals
    # =========================================================================

    # Default interval between sync operations
    SYNC_INTERVAL: float = 300.0  # 5 minutes

    # Time after which data is considered stale
    STALE_DATA_THRESHOLD: int = 1800  # 30 minutes

    # Time after which stale data is critical
    CRITICAL_STALE_THRESHOLD: int = 3600  # 1 hour

    # =========================================================================
    # Daemon Lifecycle
    # =========================================================================

    # Time of stability before resetting restart counter
    RESTART_RESET_AFTER: int = 3600  # 1 hour

    # Maximum delay for exponential backoff on restart
    MAX_RESTART_DELAY: int = 300  # 5 minutes

    # =========================================================================
    # HTTP/Transport Timeouts
    # =========================================================================

    # Default HTTP request timeout
    HTTP_TIMEOUT: float = 120.0  # 2 minutes

    # SSH connection timeout
    SSH_TIMEOUT: float = 30.0  # 30 seconds

    # Transport layer recovery timeout
    TRANSPORT_RECOVERY_TIMEOUT: float = 300.0  # 5 minutes

    # =========================================================================
    # Scan Intervals
    # =========================================================================

    # Orphan game detection scan interval
    ORPHAN_SCAN_INTERVAL: float = 300.0  # 5 minutes

    # Curriculum check interval
    CURRICULUM_CHECK_INTERVAL: float = 120.0  # 2 minutes

    # Auto snapshot interval for coordinator persistence
    AUTO_SNAPSHOT_INTERVAL: float = 300.0  # 5 minutes

    # =========================================================================
    # Cooldowns (Rate Limiting)
    # =========================================================================

    # Minimum time between hyperparameter adjustments
    ADJUSTMENT_COOLDOWN: float = 300.0  # 5 minutes

    # Minimum time between curriculum advancements
    CURRICULUM_COOLDOWN: float = 600.0  # 10 minutes

    # Minimum time between repeated alerts for same issue
    ALERT_COOLDOWN: float = 300.0  # 5 minutes

    # =========================================================================
    # Sync Stall Handling
    # =========================================================================

    # Penalty time added when sync stalls
    STALL_PENALTY_SECONDS: int = 300  # 5 minutes

    # =========================================================================
    # Backoff Limits
    # =========================================================================

    # Minimum backoff for idle resource daemon
    BACKOFF_MIN_SECONDS: float = 60.0  # 1 minute

    # Maximum backoff for idle resource daemon
    BACKOFF_MAX_SECONDS: float = 1800.0  # 30 minutes


# Singleton instance with environment overrides applied
def _create_timeouts() -> TimeoutDefaults:
    """Create TimeoutDefaults with environment variable overrides."""
    return TimeoutDefaults(
        LOCK_TIMEOUT_SECONDS=_env_int("LOCK_TIMEOUT_SECONDS", 3600),
        LOCK_ACQUIRE_TIMEOUT=_env_int("LOCK_ACQUIRE_TIMEOUT", 30),
        HEARTBEAT_THRESHOLD=_env_float("HEARTBEAT_THRESHOLD", 120.0),
        RECOVERY_COOLDOWN=_env_int("RECOVERY_COOLDOWN", 300),
        ESCALATION_COOLDOWN=_env_int("ESCALATION_COOLDOWN", 3600),
        PAUSE_ERROR_BURST_WINDOW=_env_int("PAUSE_ERROR_BURST_WINDOW", 300),
        RESUME_DELAY_SECONDS=_env_int("RESUME_DELAY_SECONDS", 120),
        SYNC_INTERVAL=_env_float("SYNC_INTERVAL", 300.0),
        STALE_DATA_THRESHOLD=_env_int("STALE_DATA_THRESHOLD", 1800),
        CRITICAL_STALE_THRESHOLD=_env_int("CRITICAL_STALE_THRESHOLD", 3600),
        RESTART_RESET_AFTER=_env_int("RESTART_RESET_AFTER", 3600),
        MAX_RESTART_DELAY=_env_int("MAX_RESTART_DELAY", 300),
        HTTP_TIMEOUT=_env_float("HTTP_TIMEOUT", 120.0),
        SSH_TIMEOUT=_env_float("SSH_TIMEOUT", 30.0),
        TRANSPORT_RECOVERY_TIMEOUT=_env_float("TRANSPORT_RECOVERY_TIMEOUT", 300.0),
        ORPHAN_SCAN_INTERVAL=_env_float("ORPHAN_SCAN_INTERVAL", 300.0),
        CURRICULUM_CHECK_INTERVAL=_env_float("CURRICULUM_CHECK_INTERVAL", 120.0),
        AUTO_SNAPSHOT_INTERVAL=_env_float("AUTO_SNAPSHOT_INTERVAL", 300.0),
        ADJUSTMENT_COOLDOWN=_env_float("ADJUSTMENT_COOLDOWN", 300.0),
        CURRICULUM_COOLDOWN=_env_float("CURRICULUM_COOLDOWN", 600.0),
        ALERT_COOLDOWN=_env_float("ALERT_COOLDOWN", 300.0),
        STALL_PENALTY_SECONDS=_env_int("STALL_PENALTY_SECONDS", 300),
        BACKOFF_MIN_SECONDS=_env_float("BACKOFF_MIN_SECONDS", 60.0),
        BACKOFF_MAX_SECONDS=_env_float("BACKOFF_MAX_SECONDS", 1800.0),
    )


TIMEOUTS = _create_timeouts()


def get_timeout(name: str, default: float = 30.0) -> float:
    """Get timeout by name with sensible default.

    Args:
        name: Timeout constant name (case insensitive)
        default: Fallback value if name not found

    Returns:
        Timeout value in seconds

    Example:
        >>> get_timeout("sync_interval")
        300.0
        >>> get_timeout("nonexistent", 60.0)
        60.0
    """
    return getattr(TIMEOUTS, name.upper(), default)


def get_timeout_ms(name: str, default: float = 30000.0) -> float:
    """Get timeout by name in milliseconds.

    Args:
        name: Timeout constant name (case insensitive)
        default: Fallback value in ms if name not found

    Returns:
        Timeout value in milliseconds
    """
    seconds = getattr(TIMEOUTS, name.upper(), None)
    if seconds is not None:
        return seconds * 1000
    return default
