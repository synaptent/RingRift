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

import logging
import os
import socket
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

__all__ = ["RingRiftEnv", "env"]

logger = logging.getLogger(__name__)


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
        """Node identifier for this machine.

        Uses the Unified Node Identity System for canonical resolution.
        Priority: RINGRIFT_NODE_ID env var > hostname match > Tailscale IP match

        Falls back to hostname if resolution fails (for backward compatibility).
        """
        try:
            from app.config.node_identity import get_node_id_safe

            return get_node_id_safe()
        except ImportError:
            # Fallback if node_identity module not available
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
        return os.environ.get("RINGRIFT_SKIP_SHADOW_CONTRACTS", "true").lower() in ("1", "true", "yes")

    @cached_property
    def parity_validation(self) -> str:
        """Parity validation mode (off, warn, strict)."""
        return os.environ.get("RINGRIFT_PARITY_VALIDATION", "off")

    @cached_property
    def idle_resource_enabled(self) -> bool:
        """Whether idle resource daemon is enabled."""
        return os.environ.get("RINGRIFT_IDLE_RESOURCE_ENABLED", "1") == "1"

    @cached_property
    def lambda_idle_enabled(self) -> bool:
        """Whether Lambda idle daemon is enabled.

        NOTE: Lambda account currently suspended pending support ticket resolution.
        """
        return os.environ.get("RINGRIFT_LAMBDA_IDLE_ENABLED", "1") == "1"

    @cached_property
    def auto_update_enabled(self) -> bool:
        """Whether auto-update is enabled."""
        return os.environ.get("RINGRIFT_P2P_AUTO_UPDATE", "false").strip().lower() in ("1", "true", "yes")

    @cached_property
    def autonomous_mode(self) -> bool:
        """Enable autonomous training mode.

        When enabled:
        - Stale data becomes warning, triggers auto-sync instead of error
        - Pending gate databases are allowed (TS parity not validated)
        - Non-canonical data sources are allowed
        - Training proceeds without manual intervention

        Use for 8+ hour unattended training runs on cluster.
        Set via RINGRIFT_AUTONOMOUS_MODE=1 or --autonomous CLI flag.
        """
        return os.environ.get("RINGRIFT_AUTONOMOUS_MODE", "").lower() in ("1", "true", "yes")

    @cached_property
    def allow_pending_gate(self) -> bool:
        """Allow training on databases with pending parity gate.

        Separate from autonomous_mode for fine-grained control.
        Set via RINGRIFT_ALLOW_PENDING_GATE=1.
        """
        return os.environ.get("RINGRIFT_ALLOW_PENDING_GATE", "").lower() in ("1", "true", "yes")

    # ==========================================================================
    # Coordinator-Only Mode (Dec 2025)
    # ==========================================================================

    @cached_property
    def is_coordinator(self) -> bool:
        """Check if this node is a coordinator-only node.

        Coordinators should NOT run CPU/GPU intensive processes like:
        - Training
        - Selfplay
        - Gauntlet/evaluation
        - Export

        Coordinators CAN run:
        - P2P daemon
        - Sync daemons
        - Monitoring
        - Health checks

        Detection order:
        1. RINGRIFT_IS_COORDINATOR environment variable
        2. Check distributed_hosts.yaml for role: coordinator
        """
        # Explicit environment variable takes precedence
        explicit = os.environ.get("RINGRIFT_IS_COORDINATOR", "").lower()
        if explicit in ("1", "true", "yes"):
            return True
        if explicit in ("0", "false", "no"):
            return False

        # Fall back to distributed_hosts.yaml
        return self._check_coordinator_from_config()

    @cached_property
    def selfplay_enabled(self) -> bool:
        """Whether selfplay is enabled on this node."""
        # Explicit override
        explicit = os.environ.get("RINGRIFT_SELFPLAY_ENABLED", "").lower()
        if explicit in ("0", "false", "no"):
            return False
        if explicit in ("1", "true", "yes"):
            return True
        # Coordinators have selfplay disabled by default
        if self.is_coordinator:
            return False
        # Check config
        return self._get_node_config_bool("selfplay_enabled", default=True)

    @cached_property
    def training_enabled(self) -> bool:
        """Whether training is enabled on this node."""
        explicit = os.environ.get("RINGRIFT_TRAINING_ENABLED", "").lower()
        if explicit in ("0", "false", "no"):
            return False
        if explicit in ("1", "true", "yes"):
            return True
        if self.is_coordinator:
            return False
        return self._get_node_config_bool("training_enabled", default=True)

    @cached_property
    def gauntlet_enabled(self) -> bool:
        """Whether gauntlet/evaluation is enabled on this node.

        Coordinators default to False: PyTorch MPS holds the Python GIL
        during forward passes, blocking the asyncio event loop for minutes
        on large boards (square19=361 cells). Evaluations dispatch to GPU
        cluster nodes instead via _dispatch_gauntlet_to_cluster().
        """
        explicit = os.environ.get("RINGRIFT_GAUNTLET_ENABLED", "").lower()
        if explicit in ("0", "false", "no"):
            return False
        if explicit in ("1", "true", "yes"):
            return True
        if self.is_coordinator:
            return False
        return self._get_node_config_bool("gauntlet_enabled", default=True)

    @cached_property
    def export_enabled(self) -> bool:
        """Whether data export is enabled on this node.

        Note: Coordinators NEED export enabled to convert consolidated game
        data into NPZ training files. Without this, the training pipeline
        stalls because no training data is ever generated.
        """
        explicit = os.environ.get("RINGRIFT_EXPORT_ENABLED", "").lower()
        if explicit in ("0", "false", "no"):
            return False
        if explicit in ("1", "true", "yes"):
            return True
        return self._get_node_config_bool("export_enabled", default=True)

    @cached_property
    def consolidation_enabled(self) -> bool:
        """Whether data consolidation is enabled on this node.

        Coordinators should NOT consolidate data locally as it creates
        large canonical_*.db files that fill up disk space.
        """
        explicit = os.environ.get("RINGRIFT_CONSOLIDATION_ENABLED", "").lower()
        if explicit in ("0", "false", "no"):
            return False
        if explicit in ("1", "true", "yes"):
            return True
        if self.is_coordinator:
            return False
        return self._get_node_config_bool("consolidation_enabled", default=True)

    def _check_coordinator_from_config(self) -> bool:
        """Check distributed_hosts.yaml for coordinator role."""
        try:
            config = self._get_node_config()
            if config:
                return config.get("role", "").lower() == "coordinator"
        except (FileNotFoundError, KeyError, AttributeError) as e:
            # Config not available or malformed - not a coordinator
            logger.debug(f"Could not determine coordinator role: {e}")
        return False

    def _get_node_config(self) -> dict | None:
        """Get this node's config from distributed_hosts.yaml."""
        try:
            import yaml

            # Build config paths from environment and standard locations
            config_paths = [
                self.ai_service_path / "config" / "distributed_hosts.yaml",
            ]

            # Add custom config path from environment if set
            custom_config = os.environ.get("RINGRIFT_CONFIG_PATH")
            if custom_config:
                config_paths.insert(0, Path(custom_config) / "distributed_hosts.yaml")

            # Add standard system locations (Unix/Linux)
            config_paths.extend([
                Path("/etc/ringrift/distributed_hosts.yaml"),
                Path.home() / ".config" / "ringrift" / "distributed_hosts.yaml",
            ])
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    hosts = config.get("hosts", {})
                    # Check by node_id first
                    if self.node_id in hosts:
                        return hosts[self.node_id]
                    # Check by hostname
                    if self.hostname in hosts:
                        return hosts[self.hostname]
                    # Check for Mac-specific hostname patterns
                    hostname_lower = self.hostname.lower()
                    is_mac = (
                        "macbook" in hostname_lower
                        or "mac-studio" in hostname_lower
                        or hostname_lower.startswith("armand")
                        or hostname_lower == "localhost"
                    )
                    if is_mac:
                        # Prefer local-mac for laptops, mac-studio for desktop
                        if "local-mac" in hosts and "macbook" in hostname_lower:
                            return hosts["local-mac"]
                        elif "mac-studio" in hosts and "studio" in hostname_lower:
                            return hosts["mac-studio"]
                        # Fallback to any coordinator entry for local development
                        for name in ["local-mac", "mac-studio"]:
                            if name in hosts:
                                return hosts[name]
            return None
        except (FileNotFoundError, OSError) as e:
            logger.debug(f"Could not read node config: {e}")
            return None
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"Malformed node config: {e}")
            return None

    def _get_node_config_bool(self, key: str, default: bool = True) -> bool:
        """Get a boolean from node config."""
        config = self._get_node_config()
        if config and key in config:
            val = config[key]
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("1", "true", "yes")
        return default

    def should_skip_intensive_process(self, process_type: str = "any") -> bool:
        """Check if this node should skip running intensive processes.

        Args:
            process_type: One of 'selfplay', 'training', 'gauntlet', 'export', 'any'

        Returns:
            True if the process should be SKIPPED on this node.
        """
        if process_type == "selfplay":
            return not self.selfplay_enabled
        elif process_type == "training":
            return not self.training_enabled
        elif process_type == "gauntlet":
            return not self.gauntlet_enabled
        elif process_type == "export":
            return not self.export_enabled
        else:  # "any"
            return self.is_coordinator

    # ==========================================================================
    # Training
    # ==========================================================================

    @cached_property
    def training_threshold(self) -> int:
        """Training trigger threshold in games."""
        return int(os.environ.get("RINGRIFT_TRAINING_THRESHOLD", "500"))

    @cached_property
    def min_games_for_training(self) -> int:
        """Minimum games required before training can begin."""
        return int(os.environ.get("RINGRIFT_MIN_GAMES_FOR_TRAINING", "100"))

    @cached_property
    def learning_rate(self) -> float:
        """Training learning rate."""
        return float(os.environ.get("RINGRIFT_LEARNING_RATE", "0.001"))

    @cached_property
    def batch_size(self) -> int:
        """Training batch size."""
        return int(os.environ.get("RINGRIFT_BATCH_SIZE", "512"))

    @cached_property
    def epochs(self) -> int:
        """Training epochs."""
        return int(os.environ.get("RINGRIFT_EPOCHS", "20"))

    @cached_property
    def checkpoint_dir(self) -> Path:
        """Training checkpoint directory."""
        return Path(os.environ.get("RINGRIFT_CHECKPOINT_DIR", "checkpoints"))

    # ==========================================================================
    # PyTorch Distributed Training
    # ==========================================================================

    @cached_property
    def master_addr(self) -> str:
        """Master node address for PyTorch distributed training.

        Used by torch.distributed for multi-node training coordination.
        Should point to the rank 0 node's IP address.
        """
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @cached_property
    def master_port(self) -> int:
        """Master node port for PyTorch distributed training.

        Used by torch.distributed for multi-node training coordination.
        Default 29500 is PyTorch's standard port.
        """
        return int(os.environ.get("MASTER_PORT", "29500"))

    # ==========================================================================
    # Lambda/Provider Specific
    # NOTE: Lambda account currently suspended pending support ticket resolution
    # ==========================================================================

    @cached_property
    def lambda_idle_interval(self) -> int:
        """Lambda idle check interval in seconds."""
        return int(os.environ.get("RINGRIFT_LAMBDA_IDLE_INTERVAL", "300"))

    @cached_property
    def lambda_idle_threshold(self) -> float:
        """Lambda GPU idle threshold percentage."""
        return float(os.environ.get("RINGRIFT_LAMBDA_IDLE_THRESHOLD", "5.0"))

    @cached_property
    def lambda_idle_duration(self) -> int:
        """Lambda idle duration threshold in seconds."""
        return int(os.environ.get("RINGRIFT_LAMBDA_IDLE_DURATION", "1800"))

    @cached_property
    def lambda_min_nodes(self) -> int:
        """Minimum Lambda nodes to keep running."""
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
    # P2P Protocol Settings (December 2025)
    # ==========================================================================

    @cached_property
    def p2p_startup_grace_period(self) -> int:
        """Grace period in seconds during P2P startup."""
        return int(os.environ.get("RINGRIFT_P2P_STARTUP_GRACE_PERIOD", "120"))

    @cached_property
    def consensus_mode(self) -> str:
        """Consensus protocol mode (bully, raft, hybrid)."""
        return os.environ.get("RINGRIFT_CONSENSUS_MODE", "bully")

    @cached_property
    def membership_mode(self) -> str:
        """Membership protocol mode (http, swim, hybrid)."""
        return os.environ.get("RINGRIFT_MEMBERSHIP_MODE", "http")

    @cached_property
    def raft_enabled(self) -> bool:
        """Whether Raft consensus is enabled."""
        return os.environ.get("RINGRIFT_RAFT_ENABLED", "").lower() in ("1", "true", "yes")

    @cached_property
    def swim_enabled(self) -> bool:
        """Whether SWIM membership is enabled."""
        return os.environ.get("RINGRIFT_SWIM_ENABLED", "").lower() in ("1", "true", "yes")

    @cached_property
    def extracted_loops(self) -> bool:
        """Whether to use extracted loop managers (December 2025)."""
        return os.environ.get("RINGRIFT_EXTRACTED_LOOPS", "true").lower() in ("1", "true", "yes")

    # ==========================================================================
    # Timeouts and Intervals (December 2025)
    # ==========================================================================

    @cached_property
    def http_timeout(self) -> int:
        """HTTP request timeout in seconds."""
        return int(os.environ.get("RINGRIFT_HTTP_TIMEOUT", "30"))

    @cached_property
    def rsync_timeout(self) -> int:
        """Rsync transfer timeout in seconds."""
        return int(os.environ.get("RINGRIFT_RSYNC_TIMEOUT", "300"))

    @cached_property
    def heartbeat_interval(self) -> int:
        """P2P heartbeat interval in seconds."""
        return int(os.environ.get("RINGRIFT_HEARTBEAT_INTERVAL", "15"))

    @cached_property
    def heartbeat_timeout(self) -> int:
        """P2P heartbeat timeout in seconds."""
        return int(os.environ.get("RINGRIFT_HEARTBEAT_TIMEOUT", "60"))

    @cached_property
    def elo_sync_interval(self) -> int:
        """Elo database sync interval in seconds."""
        return int(os.environ.get("RINGRIFT_ELO_SYNC_INTERVAL", "300"))

    @cached_property
    def event_handler_timeout(self) -> int:
        """Event handler timeout in seconds."""
        return int(os.environ.get("RINGRIFT_EVENT_HANDLER_TIMEOUT", "600"))

    @cached_property
    def lock_timeout(self) -> int:
        """Distributed lock timeout in seconds."""
        return int(os.environ.get("RINGRIFT_LOCK_TIMEOUT", "60"))

    @cached_property
    def work_queue_db(self) -> Path | None:
        """Work queue database path override."""
        path = os.environ.get("RINGRIFT_WORK_QUEUE_DB")
        return Path(path) if path else None

    # ==========================================================================
    # Circuit Breaker Settings (December 2025)
    # ==========================================================================

    @cached_property
    def cb_failure_threshold(self) -> int:
        """Circuit breaker failure threshold before opening."""
        return int(os.environ.get("RINGRIFT_CB_FAILURE_THRESHOLD", "5"))

    @cached_property
    def cb_recovery_timeout(self) -> int:
        """Circuit breaker recovery timeout in seconds."""
        return int(os.environ.get("RINGRIFT_CB_RECOVERY_TIMEOUT", "60"))

    @cached_property
    def cb_half_open_max_calls(self) -> int:
        """Circuit breaker half-open max test calls."""
        return int(os.environ.get("RINGRIFT_CB_HALF_OPEN_MAX_CALLS", "3"))

    # ==========================================================================
    # Backpressure Settings (December 2025)
    # ==========================================================================

    @cached_property
    def backpressure_gpu_threshold(self) -> float:
        """GPU utilization threshold for backpressure (Session 17.42: lowered from 90)."""
        return float(os.environ.get("RINGRIFT_BACKPRESSURE_GPU_THRESHOLD", "70"))

    @cached_property
    def backpressure_memory_threshold(self) -> float:
        """Memory utilization threshold for backpressure."""
        return float(os.environ.get("RINGRIFT_BACKPRESSURE_MEMORY_THRESHOLD", "85"))

    @cached_property
    def backpressure_disk_threshold(self) -> float:
        """Disk utilization threshold for backpressure."""
        return float(os.environ.get("RINGRIFT_BACKPRESSURE_DISK_THRESHOLD", "90"))

    # ==========================================================================
    # Parity and Validation (December 2025)
    # ==========================================================================

    @cached_property
    def allow_pending_gate(self) -> bool:
        """Allow databases with pending parity gate status."""
        return os.environ.get("RINGRIFT_ALLOW_PENDING_GATE", "").lower() in ("1", "true", "yes")

    @cached_property
    def require_critical_imports(self) -> bool:
        """Require all critical imports to succeed at startup."""
        return os.environ.get("RINGRIFT_REQUIRE_CRITICAL_IMPORTS", "").lower() in ("1", "true", "yes")

    # ==========================================================================
    # Alerting (December 2025)
    # ==========================================================================

    @cached_property
    def discord_webhook_url(self) -> str | None:
        """Discord webhook URL for alerts."""
        return os.environ.get("RINGRIFT_DISCORD_WEBHOOK_URL") or os.environ.get("RINGRIFT_DISCORD_WEBHOOK")

    @cached_property
    def slack_webhook_url(self) -> str | None:
        """Slack webhook URL for alerts."""
        return os.environ.get("RINGRIFT_SLACK_WEBHOOK_URL") or os.environ.get("RINGRIFT_SLACK_WEBHOOK")

    # ==========================================================================
    # P2P Network Settings (December 2025)
    # ==========================================================================

    @cached_property
    def p2p_port(self) -> int:
        """P2P orchestrator port (default 8770).

        Uses centralized P2P_DEFAULT_PORT from app/config/ports.py.
        """
        from app.config.ports import P2P_DEFAULT_PORT
        return P2P_DEFAULT_PORT

    @cached_property
    def p2p_seeds(self) -> list[str]:
        """P2P seed nodes (comma-separated host:port)."""
        seeds = os.environ.get("RINGRIFT_P2P_SEEDS", "")
        return [s.strip() for s in seeds.split(",") if s.strip()]

    @cached_property
    def p2p_url(self) -> str | None:
        """P2P orchestrator URL (for connecting to remote P2P)."""
        return os.environ.get("RINGRIFT_P2P_URL")

    @cached_property
    def node_role(self) -> str:
        """Node role (coordinator, training, selfplay, voter)."""
        return os.environ.get("RINGRIFT_NODE_ROLE", "selfplay")

    # ==========================================================================
    # Neural Network / GPU Settings (December 2025)
    # ==========================================================================

    @cached_property
    def disable_torch_compile(self) -> bool:
        """Disable torch.compile() for debugging."""
        return os.environ.get("RINGRIFT_DISABLE_TORCH_COMPILE", "").lower() in ("1", "true", "yes")

    @cached_property
    def disable_neural_net(self) -> bool:
        """Disable neural network loading entirely."""
        return os.environ.get("RINGRIFT_DISABLE_NEURAL_NET", "").lower() in ("1", "true", "yes")

    @cached_property
    def require_neural_net(self) -> bool:
        """Fail if neural network cannot be loaded."""
        return os.environ.get("RINGRIFT_REQUIRE_NEURAL_NET", "").lower() in ("1", "true", "yes")

    @cached_property
    def force_cpu(self) -> bool:
        """Force CPU-only mode even if GPU is available."""
        return os.environ.get("RINGRIFT_FORCE_CPU", "").lower() in ("1", "true", "yes")

    @cached_property
    def gpu_gumbel_disable(self) -> bool:
        """Disable GPU Gumbel MCTS (use CPU version)."""
        return os.environ.get("RINGRIFT_GPU_GUMBEL_DISABLE", "").lower() in ("1", "true", "yes")

    # ==========================================================================
    # Game Engine Optimization Flags (December 2025)
    # ==========================================================================

    @cached_property
    def use_fast_territory(self) -> bool:
        """Use fast territory scoring algorithm."""
        return os.environ.get("RINGRIFT_USE_FAST_TERRITORY", "true").lower() in ("1", "true", "yes")

    @cached_property
    def use_make_unmake(self) -> bool:
        """Use make/unmake move optimization."""
        return os.environ.get("RINGRIFT_USE_MAKE_UNMAKE", "true").lower() in ("1", "true", "yes")

    @cached_property
    def fsm_validation_mode(self) -> str:
        """FSM validation mode (strict, lenient, disabled)."""
        return os.environ.get("RINGRIFT_FSM_VALIDATION_MODE", "strict")

    @cached_property
    def strict_no_move_invariant(self) -> bool:
        """Enforce strict no-move invariant checking."""
        return os.environ.get("RINGRIFT_STRICT_NO_MOVE_INVARIANT", "").lower() in ("1", "true", "yes")

    # ==========================================================================
    # Task and Process Control (December 2025)
    # ==========================================================================

    @cached_property
    def disable_local_tasks(self) -> bool:
        """Disable running tasks on coordinator node."""
        return os.environ.get("RINGRIFT_DISABLE_LOCAL_TASKS", "").lower() in ("1", "true", "yes")

    @cached_property
    def parallel_workers(self) -> int:
        """Number of parallel workers for batch operations."""
        return int(os.environ.get("RINGRIFT_PARALLEL_WORKERS", "4"))

    @cached_property
    def record_selfplay_games(self) -> bool:
        """Record selfplay games to database."""
        return os.environ.get("RINGRIFT_RECORD_SELFPLAY_GAMES", "true").lower() in ("1", "true", "yes")

    @cached_property
    def job_origin(self) -> str | None:
        """Origin identifier for jobs (for tracking)."""
        return os.environ.get("RINGRIFT_JOB_ORIGIN")

    # ==========================================================================
    # Staging and Deployment (December 2025)
    # ==========================================================================

    @cached_property
    def staging_root(self) -> Path | None:
        """Staging directory root for deployments."""
        path = os.environ.get("RINGRIFT_STAGING_ROOT")
        return Path(path) if path else None

    @cached_property
    def staging_ssh_host(self) -> str | None:
        """Staging SSH host for deployments."""
        return os.environ.get("RINGRIFT_STAGING_SSH_HOST")

    @cached_property
    def s3_bucket(self) -> str | None:
        """S3 bucket for backups and model storage."""
        return os.environ.get("RINGRIFT_S3_BUCKET")

    @cached_property
    def cluster_api(self) -> str | None:
        """Cluster API endpoint URL."""
        return os.environ.get("RINGRIFT_CLUSTER_API")

    # ==========================================================================
    # Master Loop Settings (December 2025)
    # ==========================================================================

    @cached_property
    def master_loop_interval(self) -> float:
        """Master loop check interval in seconds."""
        return float(os.environ.get("RINGRIFT_MASTER_LOOP_INTERVAL", "30"))

    @cached_property
    def training_check_interval(self) -> float:
        """Training readiness check interval in seconds.

        Jan 2026: Reduced from 60s to 30s for faster signal-to-action latency.
        """
        return float(os.environ.get("RINGRIFT_TRAINING_CHECK_INTERVAL", "30"))

    @cached_property
    def allocation_check_interval(self) -> float:
        """Allocation rebalance interval in seconds."""
        return float(os.environ.get("RINGRIFT_ALLOCATION_CHECK_INTERVAL", "120"))

    @cached_property
    def min_games_for_export(self) -> int:
        """Minimum new games before triggering export.

        Dec 29, 2025: Increased default from 100 to 500 for better statistical
        significance in training data. Use get_min_games_for_export() for
        player-count aware thresholds.
        """
        return int(os.environ.get("RINGRIFT_MIN_GAMES_FOR_EXPORT", "500"))

    @cached_property
    def max_data_staleness_hours(self) -> float:
        """Deprecated staleness threshold (hours).

        Legacy alias retained for compatibility. Active freshness enforcement uses
        RINGRIFT_MAX_DATA_AGE_HOURS in DataFreshnessDefaults.
        """
        return float(os.environ.get("RINGRIFT_MAX_DATA_STALENESS_HOURS", "24.0"))

    @cached_property
    def state_save_interval(self) -> float:
        """State persistence save interval in seconds."""
        return float(os.environ.get("RINGRIFT_STATE_SAVE_INTERVAL", "300"))

    # ==========================================================================
    # Project Root Paths (December 2025)
    # ==========================================================================

    @cached_property
    def root_dir(self) -> Path | None:
        """RingRift root directory (monorepo root)."""
        path = os.environ.get("RINGRIFT_ROOT") or os.environ.get("RINGRIFT_DIR")
        return Path(path) if path else None

    @cached_property
    def npx_path(self) -> str | None:
        """Path to npx binary (for parity tests)."""
        return os.environ.get("RINGRIFT_NPX_PATH")

    # ==========================================================================
    # Container Environment (December 2025)
    # ==========================================================================

    @cached_property
    def is_container(self) -> bool:
        """Detect if running in container environment.

        Checks for Docker, Podman, LXC, Kubernetes, Vast.ai, and RunPod containers.
        Container nodes need userspace Tailscale with SOCKS5 proxy for P2P.
        """
        # Check for Docker
        if os.path.exists("/.dockerenv"):
            return True

        # Check cgroup for container indicators
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read().lower()
                if "docker" in cgroup_content or "lxc" in cgroup_content:
                    return True
        except (FileNotFoundError, PermissionError):
            pass

        # Check for Podman
        if os.path.exists("/run/.containerenv"):
            return True

        # Check environment variables set by container runtimes
        if os.environ.get("container") == "podman":
            return True
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return True

        # Check for Vast.ai specific indicators
        if os.environ.get("VAST_CONTAINERLABEL"):
            return True

        # Check for RunPod specific indicators
        if os.environ.get("RUNPOD_POD_ID"):
            return True

        return False

    @cached_property
    def container_type(self) -> str | None:
        """Get container type if running in container.

        Returns:
            'docker', 'podman', 'lxc', 'kubernetes', or None if not in container.
        """
        if os.path.exists("/.dockerenv"):
            return "docker"
        if os.path.exists("/run/.containerenv"):
            return "podman"
        if os.environ.get("container") == "podman":
            return "podman"
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            return "kubernetes"
        if os.environ.get("VAST_CONTAINERLABEL"):
            return "docker"  # Vast.ai uses Docker
        if os.environ.get("RUNPOD_POD_ID"):
            return "docker"  # RunPod uses Docker

        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read().lower()
                if "docker" in cgroup_content:
                    return "docker"
                if "lxc" in cgroup_content:
                    return "lxc"
        except (FileNotFoundError, PermissionError):
            pass

        return None

    @cached_property
    def tailscale_auth_key(self) -> str | None:
        """Tailscale auth key for container authentication.

        Used by container_tailscale_setup to authenticate userspace Tailscale.
        """
        return os.environ.get("TAILSCALE_AUTH_KEY")

    @cached_property
    def socks_proxy(self) -> str | None:
        """SOCKS5 proxy URL for P2P connections.

        Set automatically by container_tailscale_setup when userspace
        Tailscale is running (socks5://localhost:1055).
        """
        return os.environ.get("RINGRIFT_SOCKS_PROXY")

    @cached_property
    def tailscale_socks_port(self) -> int:
        """SOCKS5 proxy port for userspace Tailscale (default: 1055)."""
        return int(os.environ.get("RINGRIFT_TAILSCALE_SOCKS_PORT", "1055"))

    @cached_property
    def needs_userspace_tailscale(self) -> bool:
        """Check if this node needs userspace Tailscale for P2P.

        True for container environments that can't use kernel Tailscale.
        """
        # Explicit override
        explicit = os.environ.get("RINGRIFT_NEEDS_USERSPACE_TAILSCALE", "").lower()
        if explicit in ("1", "true", "yes"):
            return True
        if explicit in ("0", "false", "no"):
            return False
        # Default: containers need userspace Tailscale
        return self.is_container

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
