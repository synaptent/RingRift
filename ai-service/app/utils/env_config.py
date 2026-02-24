"""Centralized environment variable configuration.

This module provides typed access to all RINGRIFT_* environment variables
with documented defaults and type conversion.

Usage:
    from app.utils.env_config import env

    # Get typed values with defaults
    min_games = env.min_training_games  # int, default 1000
    is_debug = env.debug_mode  # bool, default False

    # Or use the getter functions directly
    from app.utils.env_config import get_int, get_bool, get_str

    min_games = get_int("RINGRIFT_MIN_TRAINING_GAMES", default=1000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


def get_str(key: str, default: str = "") -> str:
    """Get a string environment variable."""
    return os.environ.get(key, default)


def get_int(key: str, default: int = 0) -> int:
    """Get an integer environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_float(key: str, default: float = 0.0) -> float:
    """Get a float environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_bool(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable.

    Recognizes: true, 1, yes, on (case-insensitive) as True
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def get_list(key: str, default: list | None = None, sep: str = ",") -> list:
    """Get a list environment variable (comma-separated by default)."""
    value = os.environ.get(key)
    if value is None:
        return default if default is not None else []
    return [item.strip() for item in value.split(sep) if item.strip()]


@dataclass(frozen=True)
class EnvConfig:
    """Typed access to all RINGRIFT environment variables.

    This class provides a centralized, documented way to access
    environment configuration with proper defaults and types.
    """

    # ==========================================================================
    # Debug and Development
    # ==========================================================================

    @property
    def debug_mode(self) -> bool:
        """Enable debug mode with verbose logging."""
        return get_bool("RINGRIFT_DEBUG", False)

    @property
    def disable_neural_net(self) -> bool:
        """Disable neural network usage (fall back to heuristics)."""
        return get_bool("RINGRIFT_DISABLE_NEURAL_NET", False)

    # ==========================================================================
    # Training Configuration
    # ==========================================================================

    @property
    def min_training_games(self) -> int:
        """Minimum games required before training can start."""
        return get_int("RINGRIFT_MIN_TRAINING_GAMES", 1000)

    @property
    def training_batch_size(self) -> int:
        """Batch size for training."""
        return get_int("RINGRIFT_TRAINING_BATCH_SIZE", 256)

    @property
    def min_hours_between_training(self) -> float:
        """Minimum hours between training runs."""
        return get_float("RINGRIFT_MIN_HOURS_BETWEEN_TRAINING", 1.0)

    @property
    def min_games_training(self) -> int:
        """Minimum games for training trigger."""
        return get_int("RINGRIFT_MIN_GAMES_TRAINING", 300)

    @property
    def accel_min_games(self) -> int:
        """Accelerated minimum games threshold."""
        return get_int("RINGRIFT_ACCEL_MIN_GAMES", 150)

    # ==========================================================================
    # Selfplay Configuration
    # ==========================================================================

    @property
    def record_selfplay_games(self) -> bool:
        """Whether to record selfplay games to database."""
        return get_bool("RINGRIFT_RECORD_SELFPLAY_GAMES", True)

    @property
    def selfplay_db_path(self) -> str:
        """Path to selfplay database."""
        return get_str("RINGRIFT_SELFPLAY_DB_PATH", "")

    @property
    def snapshot_interval(self) -> int:
        """Interval for database snapshots."""
        return get_int("RINGRIFT_SNAPSHOT_INTERVAL", 20)

    # ==========================================================================
    # Cluster / P2P Configuration
    # ==========================================================================

    @property
    def coordinator_url(self) -> str:
        """URL of the P2P coordinator."""
        return get_str("RINGRIFT_COORDINATOR_URL", "")

    @property
    def coordinator_ip(self) -> str:
        """IP address of the coordinator."""
        return get_str("RINGRIFT_COORDINATOR_IP", "")

    @property
    def p2p_url(self) -> str:
        """P2P service URL."""
        from app.config.ports import get_local_p2p_url
        return get_str("RINGRIFT_P2P_URL", "") or get_local_p2p_url()

    @property
    def cluster_auth_token(self) -> str:
        """Authentication token for cluster communication."""
        return get_str("RINGRIFT_CLUSTER_AUTH_TOKEN", "")

    @property
    def cluster_auth_token_file(self) -> str:
        """Path to file containing cluster auth token."""
        return get_str("RINGRIFT_CLUSTER_AUTH_TOKEN_FILE", "")

    @property
    def p2p_agent_mode(self) -> bool:
        """Whether running in P2P agent mode."""
        return get_bool("RINGRIFT_P2P_AGENT_MODE", False)

    @property
    def build_version(self) -> str:
        """Build version string."""
        return get_str("RINGRIFT_BUILD_VERSION", "dev")

    # ==========================================================================
    # Resource Thresholds
    # ==========================================================================

    @property
    def max_disk_percent(self) -> float:
        """Maximum disk usage percentage (sync target threshold)."""
        try:
            from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
            return get_float("RINGRIFT_MAX_DISK_PERCENT", float(DISK_SYNC_TARGET_PERCENT))
        except ImportError:
            return get_float("RINGRIFT_MAX_DISK_PERCENT", 70.0)

    @property
    def disk_critical_threshold(self) -> int:
        """Critical disk usage threshold (%)."""
        try:
            from app.config.thresholds import DISK_CRITICAL_PERCENT
            return get_int("RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD", DISK_CRITICAL_PERCENT)
        except ImportError:
            return get_int("RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD", 90)

    @property
    def disk_warning_threshold(self) -> int:
        """Warning disk usage threshold (%)."""
        try:
            from app.config.thresholds import DISK_SYNC_TARGET_PERCENT
            return get_int("RINGRIFT_P2P_DISK_WARNING_THRESHOLD", DISK_SYNC_TARGET_PERCENT)
        except ImportError:
            return get_int("RINGRIFT_P2P_DISK_WARNING_THRESHOLD", 70)

    @property
    def memory_critical_threshold(self) -> int:
        """Critical memory usage threshold (%)."""
        return get_int("RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD", 95)

    @property
    def memory_warning_threshold(self) -> int:
        """Warning memory usage threshold (%)."""
        return get_int("RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD", 85)

    @property
    def min_memory_gb(self) -> int:
        """Minimum memory in GB."""
        return get_int("RINGRIFT_P2P_MIN_MEMORY_GB", 64)

    # ==========================================================================
    # GPU Configuration
    # ==========================================================================

    @property
    def target_gpu_util_min(self) -> int:
        """Target minimum GPU utilization (%)."""
        return get_int("RINGRIFT_P2P_TARGET_GPU_UTIL_MIN", 60)

    @property
    def target_gpu_util_max(self) -> int:
        """Target maximum GPU utilization (%)."""
        return get_int("RINGRIFT_P2P_TARGET_GPU_UTIL_MAX", 90)

    # ==========================================================================
    # Local Tasks Configuration
    # ==========================================================================

    @property
    def disable_local_tasks(self) -> bool:
        """Disable local CPU-intensive tasks (coordinator-only mode)."""
        return get_bool("RINGRIFT_DISABLE_LOCAL_TASKS", False)

    # ==========================================================================
    # Validation Configuration
    # ==========================================================================

    @property
    def parity_validation(self) -> str:
        """Parity validation mode: off, warn, strict."""
        return get_str("RINGRIFT_PARITY_VALIDATION", "off").lower()

    @property
    def parity_dump_dir(self) -> str:
        """Directory for parity failure dumps."""
        return get_str("RINGRIFT_PARITY_DUMP_DIR", "parity_failures")

    @property
    def fsm_validation_mode(self) -> str:
        """FSM validation mode: active, passive, off."""
        return get_str("RINGRIFT_FSM_VALIDATION_MODE", "active").lower()

    # ==========================================================================
    # Notifications
    # ==========================================================================

    @property
    def slack_webhook_url(self) -> str:
        """Slack webhook URL for notifications."""
        return get_str("RINGRIFT_SLACK_WEBHOOK_URL", "")

    @property
    def discord_webhook_url(self) -> str:
        """Discord webhook URL for notifications."""
        return get_str("RINGRIFT_DISCORD_WEBHOOK_URL", "")

    @property
    def webhook_url(self) -> str:
        """Generic webhook URL for notifications."""
        return get_str("RINGRIFT_WEBHOOK_URL", "")

    # ==========================================================================
    # Discovery and Inventory
    # ==========================================================================

    @property
    def discovery_interval(self) -> int:
        """Interval between node discovery runs (seconds)."""
        return get_int("RINGRIFT_DISCOVERY_INTERVAL", 60)

    @property
    def idle_check_interval(self) -> int:
        """Interval between idle checks (seconds)."""
        return get_int("RINGRIFT_IDLE_CHECK_INTERVAL", 60)

    @property
    def idle_gpu_threshold(self) -> int:
        """GPU utilization threshold for considering a node idle (%)."""
        if os.environ.get("RINGRIFT_IDLE_THRESHOLD") is not None:
            return get_int("RINGRIFT_IDLE_THRESHOLD", 10)
        return get_int("RINGRIFT_IDLE_GPU_THRESHOLD", 10)

    @property
    def auto_assign_enabled(self) -> bool:
        """Enable automatic work assignment to idle nodes."""
        return get_bool("RINGRIFT_AUTO_ASSIGN_ENABLED", True)


# Global singleton instance
env = EnvConfig()


# Convenience re-exports
__all__ = [
    "EnvConfig",
    "env",
    "get_bool",
    "get_float",
    "get_int",
    "get_list",
    "get_str",
]
