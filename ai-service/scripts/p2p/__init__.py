"""P2P Orchestrator Module.

This package provides the distributed P2P orchestrator for RingRift AI training.
The orchestrator coordinates selfplay, training, and data sync across a cluster
of nodes.

Backward Compatibility:
    All types, constants, and utilities that were previously in p2p_orchestrator.py
    are re-exported from this package for backward compatibility.

Usage:
    from scripts.p2p import NodeRole, JobType, NodeInfo, ClusterJob
    from scripts.p2p.constants import PEER_TIMEOUT, DISK_WARNING_THRESHOLD
    from scripts.p2p.resource import check_all_resources
    from scripts.p2p.network import get_client_session, peer_request

Module Structure:
    - types.py: Enums (NodeRole, JobType)
    - constants.py: Configuration constants
    - models.py: Dataclasses (NodeInfo, ClusterJob, etc.)
    - resource.py: Resource checking utilities
    - network.py: HTTP client and circuit breaker utilities
    - utils.py: General utilities (systemd, etc.)
"""

# Re-export types for backward compatibility
from .types import (
    NodeRole,
    JobType,
)

# Re-export models for backward compatibility
from .models import (
    NodeInfo,
    ClusterJob,
    DistributedCMAESState,
    DistributedTournamentState,
    SSHTournamentRun,
    ImprovementLoopState,
    TrainingJob,
    TrainingThresholds,
    DataFileInfo,
    NodeDataManifest,
    ClusterDataManifest,
    DataSyncJob,
    ClusterSyncPlan,
)

# Re-export commonly used constants
from .constants import (
    DEFAULT_PORT,
    HEARTBEAT_INTERVAL,
    PEER_TIMEOUT,
    ELECTION_TIMEOUT,
    LEADER_LEASE_DURATION,
    DISK_CRITICAL_THRESHOLD,
    DISK_WARNING_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    LOAD_MAX_FOR_NEW_JOBS,
    GPU_POWER_RANKINGS,
    STATE_DIR,
)

# Re-export resource utilities
from .resource import (
    get_disk_usage_percent,
    check_disk_has_capacity,
    check_all_resources,
)

# Re-export network utilities
from .network import (
    AsyncLockWrapper,
    get_client_session,
    check_peer_circuit,
    record_peer_success,
    record_peer_failure,
    peer_request,
)

# Re-export general utilities
from .utils import (
    systemd_notify_watchdog,
    systemd_notify_ready,
)

# Re-export client utilities
from .client import (
    P2PClient,
    P2PClientError,
    JobRequest,
    JobResult,
    ClusterStatus,
    get_client,
    get_cluster_status,
    submit_selfplay_job,
    submit_training_job,
)

__all__ = [
    # Constants
    'DEFAULT_PORT',
    'DISK_CRITICAL_THRESHOLD',
    'DISK_WARNING_THRESHOLD',
    'ELECTION_TIMEOUT',
    'GPU_POWER_RANKINGS',
    'HEARTBEAT_INTERVAL',
    'LEADER_LEASE_DURATION',
    'LOAD_MAX_FOR_NEW_JOBS',
    'MEMORY_CRITICAL_THRESHOLD',
    'MEMORY_WARNING_THRESHOLD',
    'PEER_TIMEOUT',
    'STATE_DIR',
    # Network utilities
    'AsyncLockWrapper',
    'ClusterDataManifest',
    'ClusterJob',
    'ClusterStatus',
    'ClusterSyncPlan',
    'DataFileInfo',
    'DataSyncJob',
    'DistributedCMAESState',
    'DistributedTournamentState',
    'ImprovementLoopState',
    'JobRequest',
    'JobResult',
    'JobType',
    'NodeDataManifest',
    # Models
    'NodeInfo',
    # Types
    'NodeRole',
    # Client utilities
    'P2PClient',
    'P2PClientError',
    'SSHTournamentRun',
    'TrainingJob',
    'TrainingThresholds',
    'check_all_resources',
    'check_disk_has_capacity',
    'check_peer_circuit',
    'get_client',
    'get_client_session',
    'get_cluster_status',
    # Resource utilities
    'get_disk_usage_percent',
    'peer_request',
    'record_peer_failure',
    'record_peer_success',
    'submit_selfplay_job',
    'submit_training_job',
    'systemd_notify_ready',
    # General utilities
    'systemd_notify_watchdog',
]
