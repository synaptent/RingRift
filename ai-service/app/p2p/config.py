"""P2P Orchestrator Configuration.

Centralized configuration for the P2P distributed training cluster.
All configuration values are env-overridable for heterogeneous clusters.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.config.ports import GOSSIP_PORT, P2P_DEFAULT_PORT

# Jan 22, 2026: Import canonical timeout values from constants.py to ensure consistency.
# Previously hardcoded values (30s/90s) conflicted with constants.py (15s/120s),
# causing 30-second disagreement between modules and split-brain conditions.
from app.p2p.constants import (
    HEARTBEAT_INTERVAL,
    PEER_TIMEOUT,
    PEER_TIMEOUT_FAST,
    PEER_TIMEOUT_NAT_BLOCKED,
    ELECTION_TIMEOUT,
    LEADER_LEASE_DURATION,
    PEER_RETIRE_AFTER_SECONDS,
)

# Network Configuration
# NOTE: Using canonical port constants from app/config/ports.py
DEFAULT_PORT = P2P_DEFAULT_PORT
# HEARTBEAT_INTERVAL, PEER_TIMEOUT, ELECTION_TIMEOUT, LEADER_LEASE_DURATION
# are now imported from app.p2p.constants (canonical source)
LEADER_LEASE_RENEW_INTERVAL = 10  # How often leader renews lease
# Dec 29, 2025: Reduced from 60s to 15s for faster job status updates
JOB_CHECK_INTERVAL = int(os.environ.get("RINGRIFT_P2P_JOB_CHECK_INTERVAL", "15") or 15)
DISCOVERY_PORT = GOSSIP_PORT  # UDP port for peer discovery
DISCOVERY_INTERVAL = 120  # seconds between discovery broadcasts

# GPU Power Rankings for training node priority
# Higher score = more powerful GPU = higher priority for receiving training data
GPU_POWER_RANKINGS = {
    # Data center GPUs (highest priority)
    "H100": 2000,
    "H200": 2500,
    "GH200": 2000,
    "A100": 624,
    "A10G": 250,
    "A10": 250,
    "L40": 362,
    "V100": 125,
    # Consumer GPUs - RTX 50 series
    "5090": 419,
    "5080": 300,
    "5070": 200,
    # Consumer GPUs - RTX 40 series
    "4090": 330,
    "4080": 242,
    "4070": 184,
    "4060": 120,
    # Consumer GPUs - RTX 30 series
    "3090": 142,
    "3080": 119,
    "3070": 81,
    "3060": 51,
    # Apple Silicon
    "Apple M3": 30,
    "Apple M2": 25,
    "Apple M1": 20,
    "Apple MPS": 15,
    # Fallback
    "Unknown": 10,
}

# Tailscale CGNAT network
TAILSCALE_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


@dataclass
class P2PConfig:
    """Configuration for P2P orchestrator.

    All values can be overridden via environment variables with the
    RINGRIFT_P2P_ prefix.
    """

    # Resource thresholds - imported from app.config.thresholds (canonical source)
    DISK_CRITICAL_THRESHOLD: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD", "90") or 90)
    )
    DISK_WARNING_THRESHOLD: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_DISK_WARNING_THRESHOLD", "70") or 70)
    )
    DISK_CLEANUP_THRESHOLD: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD", "65") or 65)
    )
    MEMORY_CRITICAL_THRESHOLD: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD", "95") or 95)
    )
    MEMORY_WARNING_THRESHOLD: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD", "85") or 85)
    )
    MIN_MEMORY_GB_FOR_TASKS: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_MIN_MEMORY_GB", "64") or 64)
    )
    LOAD_MAX_FOR_NEW_JOBS: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS", "85") or 85)
    )

    # GPU utilization targeting
    TARGET_GPU_UTIL_MIN: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_TARGET_GPU_UTIL_MIN", "60") or 60)
    )
    TARGET_GPU_UTIL_MAX: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_TARGET_GPU_UTIL_MAX", "90") or 90)
    )
    GH200_MIN_SELFPLAY: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_GH200_MIN_SELFPLAY", "20") or 20)
    )
    GH200_MAX_SELFPLAY: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_GH200_MAX_SELFPLAY", "100") or 100)
    )

    # Connection robustness
    HTTP_CONNECT_TIMEOUT: int = 10
    HTTP_TOTAL_TIMEOUT: int = 30
    MAX_CONSECUTIVE_FAILURES: int = 3
    RETRY_DEAD_NODE_INTERVAL: int = 300

    # Peer management
    # Dec 2025: Increased from 1h to 24h - nodes were being retired too quickly
    # during temporary network issues, causing idle GPU waste
    PEER_RETIRE_AFTER_SECONDS: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS", "86400") or 86400)
    )
    # Dec 2025: Decreased from 1h to 5min - check retired nodes more frequently
    # to quickly un-retire nodes that come back online
    RETRY_RETIRED_NODE_INTERVAL: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL", "300") or 300)
    )

    # NAT/relay settings
    NAT_INBOUND_HEARTBEAT_STALE_SECONDS: int = 180
    RELAY_HEARTBEAT_INTERVAL: int = 15
    RELAY_COMMAND_TTL_SECONDS: int = 1800
    RELAY_COMMAND_MAX_BATCH: int = 16
    RELAY_COMMAND_MAX_ATTEMPTS: int = 3
    RELAY_MAX_PENDING_START_JOBS: int = 4

    # Peer bootstrap
    PEER_BOOTSTRAP_INTERVAL: int = 60
    PEER_BOOTSTRAP_MIN_PEERS: int = 3

    # Stuck job detection
    GPU_IDLE_RESTART_TIMEOUT: int = 300
    GPU_IDLE_THRESHOLD: int = 2

    # Load limiting
    LOAD_AVERAGE_MAX_MULTIPLIER: float = field(
        default_factory=lambda: float(os.environ.get("RINGRIFT_P2P_LOAD_AVG_MAX_MULT", "2.0") or 2.0)
    )
    SPAWN_RATE_LIMIT_PER_MINUTE: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_SPAWN_RATE_LIMIT", "5") or 5)
    )

    # Git auto-update
    GIT_UPDATE_CHECK_INTERVAL: int = field(
        default_factory=lambda: int(os.environ.get("RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL", "300") or 300)
    )
    AUTO_UPDATE_ENABLED: bool = field(
        default_factory=lambda: os.environ.get("RINGRIFT_P2P_AUTO_UPDATE", "false").strip().lower() in {"1", "true", "yes"}
    )

    # Data management
    DATA_MANAGEMENT_INTERVAL: int = 300
    DB_EXPORT_THRESHOLD_MB: int = 100
    TRAINING_DATA_SYNC_THRESHOLD_MB: int = 10
    MAX_CONCURRENT_EXPORTS: int = 2
    AUTO_TRAINING_THRESHOLD_MB: int = 50

    # Training node sync
    TRAINING_NODE_COUNT: int = 3
    TRAINING_SYNC_INTERVAL: float = 300.0
    MIN_GAMES_FOR_SYNC: int = 100

    # State directory
    STATE_DIR: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "logs" / "p2p_orchestrator"
    )

    def __post_init__(self):
        """Ensure state directory exists."""
        self.STATE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def coordinator_url(self) -> str:
        """URL of central coordinator if using agent mode."""
        return os.environ.get("RINGRIFT_COORDINATOR_URL", "")

    @property
    def agent_mode_enabled(self) -> bool:
        """Whether running in agent mode (defers to coordinator)."""
        return os.environ.get("RINGRIFT_P2P_AGENT_MODE", "").lower() in {"1", "true", "yes", "on"}

    @property
    def auth_token(self) -> str | None:
        """Cluster authentication token."""
        # First check env var directly
        token = os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN")
        if token:
            return token

        # Then check token file
        token_file = os.environ.get("RINGRIFT_CLUSTER_AUTH_TOKEN_FILE")
        if token_file and Path(token_file).exists():
            return Path(token_file).read_text().strip()

        return None

    @property
    def build_version(self) -> str:
        """Build version label."""
        return os.environ.get("RINGRIFT_BUILD_VERSION", "dev")

    def get_gpu_priority(self, gpu_name: str) -> int:
        """Get priority score for a GPU type."""
        # Try exact match
        if gpu_name in GPU_POWER_RANKINGS:
            return GPU_POWER_RANKINGS[gpu_name]

        # Try partial match
        for key, score in GPU_POWER_RANKINGS.items():
            if key.lower() in gpu_name.lower():
                return score

        return GPU_POWER_RANKINGS["Unknown"]


# Singleton instance
_config_instance: P2PConfig | None = None


def get_p2p_config() -> P2PConfig:
    """Get singleton P2P configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = P2PConfig()
    return _config_instance
