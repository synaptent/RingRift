"""Centralized constants for the RingRift AI service.

Dec 2025: Created to consolidate scattered port/timeout/threshold constants
that were duplicated across 20+ files.

Usage:
    from app.config.constants import P2P_DEFAULT_PORT, SSH_CONNECT_TIMEOUT

Migration:
    Replace hardcoded values like `8770` with `P2P_DEFAULT_PORT`.
"""

# ==============================================================================
# Network Ports
# ==============================================================================

# P2P orchestrator port - used for leader election, job distribution
P2P_DEFAULT_PORT = 8770

# Gossip protocol port for data sync
GOSSIP_PORT = 8771

# HTTP health check port
HEALTH_CHECK_PORT = 8772

# Redis default port (for distributed locking)
REDIS_DEFAULT_PORT = 6379

# ==============================================================================
# Timeouts (seconds)
# ==============================================================================

# SSH connection timeout
SSH_CONNECT_TIMEOUT = 10

# SSH command execution timeout
SSH_COMMAND_TIMEOUT = 30

# HTTP request timeout
HTTP_REQUEST_TIMEOUT = 30

# P2P heartbeat interval
HEARTBEAT_INTERVAL_SECONDS = 5.0

# P2P peer timeout (mark dead after this)
PEER_TIMEOUT_SECONDS = 30.0

# Leader election timeout
ELECTION_TIMEOUT_SECONDS = 10.0

# Job execution timeout (for selfplay jobs)
JOB_TIMEOUT_SECONDS = 3600  # 1 hour

# ==============================================================================
# Process Limits
# ==============================================================================

# Max selfplay processes per node (prevents runaway accumulation)
MAX_SELFPLAY_PROCESSES_PER_NODE = 50

# Runaway process threshold (triggers kill intervention)
RUNAWAY_PROCESS_THRESHOLD = 100

# Max job spawn rate per minute
MAX_SPAWN_RATE_PER_MINUTE = 40

# ==============================================================================
# Queue Thresholds
# ==============================================================================

# High queue depth - scale up
HIGH_QUEUE_DEPTH = 20

# Medium queue depth - normal operation
MEDIUM_QUEUE_DEPTH = 10

# Max queue depth - stop spawning
MAX_QUEUE_DEPTH = 100

# ==============================================================================
# Training Thresholds
# ==============================================================================

# Max pending training hours before pausing selfplay
MAX_PENDING_TRAINING_HOURS = 24.0

# Default batch size for training
DEFAULT_BATCH_SIZE = 512

# Default training epochs
DEFAULT_EPOCHS = 50

# ==============================================================================
# GPU Memory Thresholds (GB)
# ==============================================================================

# Minimum GPU memory for different board types
GPU_MEMORY_THRESHOLDS = {
    "hex8_2p": 4.0,
    "hex8_3p": 6.0,
    "hex8_4p": 8.0,
    "square8_2p": 4.0,
    "square8_3p": 6.0,
    "square8_4p": 8.0,
    "square19_2p": 16.0,
    "square19_3p": 24.0,
    "square19_4p": 32.0,
    "hexagonal_2p": 24.0,
    "hexagonal_3p": 32.0,
    "hexagonal_4p": 48.0,
}

# ==============================================================================
# Gumbel MCTS Budget Tiers
# ==============================================================================

GUMBEL_BUDGET_THROUGHPUT = 64   # Fast selfplay
GUMBEL_BUDGET_STANDARD = 150    # Standard quality
GUMBEL_BUDGET_QUALITY = 800     # High quality
GUMBEL_BUDGET_ULTIMATE = 1600   # Maximum quality
