"""Centralized Environment Variable Configuration.

This module provides typed accessors for all RINGRIFT_* environment variables,
eliminating scattered os.environ.get() calls throughout the codebase.

Usage:
    from app.config.env import env

    # Get values with proper types and defaults
    node_id = env.node_id
    log_level = env.log_level
    is_coordinator = env.is_coordinator

    # Check feature flags
    if env.skip_shadow_contracts:
        # Skip validation

All values are cached on first access for performance.

Migration:
    Replace:
        os.environ.get("RINGRIFT_NODE_ID", "unknown")
    With:
        from app.config.env import env
        env.node_id
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

__all__ = ["env", "RingRiftEnv"]


@dataclass
class RingRiftEnv:
    """Centralized environment variable configuration.

    All RINGRIFT_* environment variables accessible via typed properties.
    Values are cached on first access.
    """

    # ==========================================================================
    # Node Identity
    # ==========================================================================

    @cached_property
    def node_id(self) -> str:
        """Node identifier for this machine."""
        return os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())

    @cached_property
    def orchestrator_id(self) -> str:
        """Orchestrator identifier."""
        return os.environ.get("RINGRIFT_ORCHESTRATOR", "unknown")

    @cached_property
    def hostname(self) -> str:
        """Machine hostname."""
        return socket.gethostname()

    # ==========================================================================
    # Paths
    # ==========================================================================

    @cached_property
    def ai_service_path(self) -> Path:
        """Path to ai-service directory."""
        path = os.environ.get("RINGRIFT_AI_SERVICE_PATH")
        if path:
            return Path(path)
        return Path(__file__).parent.parent.parent

    @cached_property
    def data_dir(self) -> Path:
        """Data directory path."""
        return Path(os.environ.get("RINGRIFT_DATA_DIR", "data"))

    @cached_property
    def config_path(self) -> Path | None:
        """Config file path override."""
        path = os.environ.get("RINGRIFT_CONFIG_PATH")
        return Path(path) if path else None

    @cached_property
    def elo_db_path(self) -> Path | None:
        """Elo database path override."""
        path = os.environ.get("RINGRIFT_ELO_DB")
        return Path(path) if path else None

    @cached_property
    def nfs_coordination_path(self) -> Path:
        """NFS coordination path."""
        return Path(os.environ.get(
            "RINGRIFT_NFS_COORDINATION_PATH",
            "/lambda/nfs/RingRift/coordination"
        ))

    # ==========================================================================
    # Logging
    # ==========================================================================

    @cached_property
    def log_level(self) -> str:
        """Log level (DEBUG, INFO, WARNING, ERROR)."""
        return os.environ.get("RINGRIFT_LOG_LEVEL", "INFO").upper()

    @cached_property
    def log_format(self) -> str:
        """Log format style (default, compact, verbose)."""
        return os.environ.get("RINGRIFT_LOG_FORMAT", "default").lower()

    @cached_property
    def log_json(self) -> bool:
        """Whether to use JSON logging."""
        return os.environ.get("RINGRIFT_LOG_JSON", "").lower() == "true"

    @cached_property
    def log_file(self) -> str | None:
        """Log file path if specified."""
        return os.environ.get("RINGRIFT_LOG_FILE")

    @cached_property
    def trace_debug(self) -> bool:
        """Whether trace debugging is enabled."""
        return os.environ.get("RINGRIFT_TRACE_DEBUG", "").lower() in ("1", "true", "yes")

    # ==========================================================================
    # P2P / Cluster
    # ==========================================================================

    @cached_property
    def coordinator_url(self) -> str:
        """Central coordinator URL if using agent mode."""
        return os.environ.get("RINGRIFT_COORDINATOR_URL", "")

    @cached_property
    def cluster_auth_token(self) -> str | None:
        """Cluster authentication token."""
        token = os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN")
        if token:
            return token
        token_file = os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN_FILE")
        if token_file and Path(token_file).exists():
            return Path(token_file).read_text().strip()
        return None

    @cached_property
    def build_version(self) -> str:
        """Build version label."""
        return os.environ.get("RINGRIFT_BUILD_VERSION", "dev")

    @cached_property
    def is_agent_mode(self) -> bool:
        """Whether running in agent mode (defers to coordinator)."""
        return os.environ.get("RINGRIFT_P2P_AGENT_MODE", "").lower() in ("1", "true", "yes", "on")

    @cached_property
    def health_port(self) -> int:
        """Health check endpoint port."""
        return int(os.environ.get("RINGRIFT_HEALTH_PORT", "8790"))

    # ==========================================================================
    # SSH
    # ==========================================================================

    @cached_property
    def ssh_user(self) -> str:
        """Default SSH user."""
        return os.environ.get("RINGRIFT_SSH_USER", "ubuntu")

    @cached_property
    def ssh_key(self) -> str | None:
        """Default SSH key path."""
        return os.environ.get("RINGRIFT_SSH_KEY")

    @cached_property
    def ssh_timeout(self) -> int:
        """SSH command timeout in seconds."""
        return int(os.environ.get("RINGRIFT_SSH_TIMEOUT", "60"))

    # ==========================================================================
    # Resource Management
    # ==========================================================================

    @cached_property
    def target_util_min(self) -> float:
        """Minimum target GPU utilization."""
        return float(os.environ.get("RINGRIFT_TARGET_UTIL_MIN", "60"))

    @cached_property
    def target_util_max(self) -> float:
        """Maximum target GPU utilization."""
        return float(os.environ.get("RINGRIFT_TARGET_UTIL_MAX", "80"))

    @cached_property
    def scale_up_threshold(self) -> float:
        """GPU utilization threshold to scale up."""
        return float(os.environ.get("RINGRIFT_SCALE_UP_THRESHOLD", "55"))

    @cached_property
    def scale_down_threshold(self) -> float:
        """GPU utilization threshold to scale down."""
        return float(os.environ.get("RINGRIFT_SCALE_DOWN_THRESHOLD", "85"))

    @cached_property
    def pid_kp(self) -> float:
        """PID controller proportional gain."""
        return float(os.environ.get("RINGRIFT_PID_KP", "0.3"))

    @cached_property
    def pid_ki(self) -> float:
        """PID controller integral gain."""
        return float(os.environ.get("RINGRIFT_PID_KI", "0.05"))

    @cached_property
    def pid_kd(self) -> float:
        """PID controller derivative gain."""
        return float(os.environ.get("RINGRIFT_PID_KD", "0.1"))

    @cached_property
    def idle_check_interval(self) -> int:
        """Idle resource check interval in seconds."""
        return int(os.environ.get("RINGRIFT_IDLE_CHECK_INTERVAL", "60"))

    @cached_property
    def idle_threshold(self) -> float:
        """GPU idle threshold percentage."""
        return float(os.environ.get("RINGRIFT_IDLE_THRESHOLD", "10.0"))

    @cached_property
    def idle_duration(self) -> int:
        """Time in seconds before a resource is considered idle."""
        return int(os.environ.get("RINGRIFT_IDLE_DURATION", "120"))

    # ==========================================================================
    # Process Management
    # ==========================================================================

    @cached_property
    def job_grace_period(self) -> int:
        """Seconds to wait before SIGKILL after SIGTERM."""
        return int(os.environ.get("RINGRIFT_JOB_GRACE_PERIOD", "60"))

    @cached_property
    def gpu_idle_threshold(self) -> int:
        """Seconds of GPU idle before killing stuck processes."""
        return int(os.environ.get("RINGRIFT_GPU_IDLE_THRESHOLD", "600"))

    @cached_property
    def runaway_selfplay_process_threshold(self) -> int:
        """Max selfplay processes per node."""
        return int(os.environ.get("RINGRIFT_RUNAWAY_SELFPLAY_PROCESS_THRESHOLD", "128"))

    # ==========================================================================
    # Feature Flags
    # ==========================================================================

    @cached_property
    def skip_shadow_contracts(self) -> bool:
        """Skip shadow contract validation."""
        return os.environ.get("RINGRIFT_SKIP_SHADOW_CONTRACTS", "").lower() in ("1", "true", "yes")

    @cached_property
    def parity_validation(self) -> str:
        """Parity validation mode (off, warn, strict)."""
        return os.environ.get("RINGRIFT_PARITY_VALIDATION", "warn")

    @cached_property
    def idle_resource_enabled(self) -> bool:
        """Whether idle resource daemon is enabled."""
        return os.environ.get("RINGRIFT_IDLE_RESOURCE_ENABLED", "1") == "1"

    @cached_property
    def lambda_idle_enabled(self) -> bool:
        """Whether Lambda idle daemon is enabled.

        DEPRECATED: Lambda account terminated December 2025.
        This setting will be removed in Q2 2026.
        """
        import warnings
        warnings.warn(
            "RINGRIFT_LAMBDA_IDLE_ENABLED is deprecated (Lambda account terminated Dec 2025). "
            "This setting will be removed in Q2 2026.",
            DeprecationWarning,
            stacklevel=2,
        )
        return os.environ.get("RINGRIFT_LAMBDA_IDLE_ENABLED", "1") == "1"

    @cached_property
    def auto_update_enabled(self) -> bool:
        """Whether auto-update is enabled."""
        return os.environ.get("RINGRIFT_P2P_AUTO_UPDATE", "false").strip().lower() in ("1", "true", "yes")

    # ==========================================================================
    # Training
    # ==========================================================================

    @cached_property
    def training_threshold(self) -> int:
        """Training trigger threshold in games."""
        return int(os.environ.get("RINGRIFT_TRAINING_THRESHOLD", "500"))

    # ==========================================================================
    # Lambda/Provider Specific
    # DEPRECATED: Lambda account terminated December 2025
    # These settings will be removed in Q2 2026
    # ==========================================================================

    @cached_property
    def lambda_idle_interval(self) -> int:
        """Lambda idle check interval.

        DEPRECATED: Lambda account terminated December 2025.
        """
        import warnings
        warnings.warn(
            "RINGRIFT_LAMBDA_IDLE_INTERVAL is deprecated (Lambda account terminated Dec 2025).",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(os.environ.get("RINGRIFT_LAMBDA_IDLE_INTERVAL", "300"))

    @cached_property
    def lambda_idle_threshold(self) -> float:
        """Lambda GPU idle threshold.

        DEPRECATED: Lambda account terminated December 2025.
        """
        import warnings
        warnings.warn(
            "RINGRIFT_LAMBDA_IDLE_THRESHOLD is deprecated (Lambda account terminated Dec 2025).",
            DeprecationWarning,
            stacklevel=2,
        )
        return float(os.environ.get("RINGRIFT_LAMBDA_IDLE_THRESHOLD", "5.0"))

    @cached_property
    def lambda_idle_duration(self) -> int:
        """Lambda idle duration threshold.

        DEPRECATED: Lambda account terminated December 2025.
        """
        import warnings
        warnings.warn(
            "RINGRIFT_LAMBDA_IDLE_DURATION is deprecated (Lambda account terminated Dec 2025).",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(os.environ.get("RINGRIFT_LAMBDA_IDLE_DURATION", "1800"))

    @cached_property
    def lambda_min_nodes(self) -> int:
        """Minimum Lambda nodes to keep.

        DEPRECATED: Lambda account terminated December 2025.
        """
        import warnings
        warnings.warn(
            "RINGRIFT_LAMBDA_MIN_NODES is deprecated (Lambda account terminated Dec 2025).",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(os.environ.get("RINGRIFT_LAMBDA_MIN_NODES", "1"))

    # ==========================================================================
    # Debug / Testing
    # ==========================================================================

    @cached_property
    def ts_replay_dump_dir(self) -> str | None:
        """Directory for TS replay dumps."""
        return os.environ.get("RINGRIFT_TS_REPLAY_DUMP_DIR")

    @cached_property
    def ts_replay_dump_state_at_k(self) -> str | None:
        """K values to dump state at during TS replay."""
        return os.environ.get("RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K")

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get raw environment variable (fallback for unmapped vars)."""
        full_key = key if key.startswith("RINGRIFT_") else f"RINGRIFT_{key}"
        return os.environ.get(full_key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get environment variable as int."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get environment variable as float."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get environment variable as bool."""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "on")

    def is_set(self, key: str) -> bool:
        """Check if environment variable is set."""
        full_key = key if key.startswith("RINGRIFT_") else f"RINGRIFT_{key}"
        return full_key in os.environ

    def __repr__(self) -> str:
        return f"RingRiftEnv(node_id={self.node_id!r})"


# Singleton instance
env = RingRiftEnv()
