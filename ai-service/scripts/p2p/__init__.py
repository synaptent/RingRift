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

# Re-export metrics utilities (Dec 26, 2025)
from .metrics_manager import (
    MetricsManager,
    MetricsManagerMixin,
)

# Re-export resource detection utilities (Dec 26, 2025)
from .resource_detector import (
    ResourceDetector,
    ResourceDetectorMixin,
)

# Re-export network utilities (Dec 26, 2025)
from .network_utils import (
    NetworkUtils,
    NetworkUtilsMixin,
)

# Re-export peer manager (Dec 26, 2025 - Phase 2.1)
from .peer_manager import (
    PeerManagerMixin,
    get_peer_manager,
    set_peer_manager,
)

# Re-export SWIM + Raft integration (Dec 26, 2025 - Phase 5)
from .membership_mixin import MembershipMixin
from .consensus_mixin import ConsensusMixin
from .handlers.swim import SwimHandlersMixin
from .handlers.raft import RaftHandlersMixin

# Re-export gossip protocol (Dec 28, 2025 - GossipMetricsMixin merged into GossipProtocolMixin)
from .gossip_protocol import GossipProtocolMixin
# Backward-compat alias: GossipMetricsMixin is now part of GossipProtocolMixin
GossipMetricsMixin = GossipProtocolMixin  # Deprecated: use GossipProtocolMixin directly

# Re-export event emission mixin (Dec 28, 2025 - Phase 8)
from .event_emission_mixin import EventEmissionMixin

# Re-export failover integration (Dec 30, 2025 - Phase 9)
from .failover_integration import FailoverIntegrationMixin, is_failover_available
from .transport_cascade import TransportCascade, get_transport_cascade, TransportTier
from .protocol_union import ProtocolUnion, get_protocol_union, MembershipSource
from .union_discovery import UnionDiscovery, get_union_discovery, DiscoveredPeer

# Re-export NAT detection (Dec 30, 2025 - Phase 4)
from .nat_detection import (
    NATType,
    NATDetectionResult,
    NATDetector,
    detect_nat_type,
    get_cached_nat_type,
)

# Re-export ICE connection (Dec 30, 2025 - Phase 4)
from .ice_connection import (
    CandidateType,
    ICECandidate,
    ICEGatherer,
    ICEChecker,
    establish_connection as ice_establish_connection,
)

# Re-export transport metrics (Dec 30, 2025 - Phase 5)
from .transport_metrics import (
    TransportMetrics,
    TransportMetricsTracker,
    get_transport_metrics,
    record_transport_request,
    get_recommended_transport,
)

# Re-export connection pooling (Dec 30, 2025 - Phase 5)
from .connection_pool import (
    ConnectionConfig,
    PeerConnectionPool,
    get_connection_pool,
    get_pooled_session,
    start_connection_pool,
    stop_connection_pool,
)

# Re-export leader health probing (Dec 30, 2025 - Phase 6)
from .leader_health import (
    LeaderHealthProbe,
    LeaderHealthResult,
    LeaderHealthStatus,
    LeaderProbeConfig,
    get_leader_health_probe,
)

# Re-export graceful step-down (Dec 30, 2025 - Phase 6)
from .graceful_stepdown import (
    GracefulStepDown,
    StepDownConfig,
    StepDownReason,
    StepDownResult,
    step_down_leader,
    select_best_successor,
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
    # Metrics (Dec 26, 2025)
    'MetricsManager',
    'MetricsManagerMixin',
    # Resource detection (Dec 26, 2025)
    'ResourceDetector',
    'ResourceDetectorMixin',
    # Network utilities (Dec 26, 2025)
    'NetworkUtils',
    'NetworkUtilsMixin',
    # Peer manager (Dec 26, 2025 - Phase 2.1)
    'PeerManagerMixin',
    'get_peer_manager',
    'set_peer_manager',
    # SWIM + Raft integration (Dec 26, 2025 - Phase 5)
    'MembershipMixin',
    'ConsensusMixin',
    'SwimHandlersMixin',
    'RaftHandlersMixin',
    # Gossip protocol (Dec 26, 2025 - Phase 3)
    'GossipMetricsMixin',
    'GossipProtocolMixin',
    # Failover integration (Dec 30, 2025 - Phase 9)
    'FailoverIntegrationMixin',
    'is_failover_available',
    'TransportCascade',
    'get_transport_cascade',
    'TransportTier',
    'ProtocolUnion',
    'get_protocol_union',
    'MembershipSource',
    'UnionDiscovery',
    'get_union_discovery',
    'DiscoveredPeer',
    # NAT detection (Dec 30, 2025 - Phase 4)
    'NATType',
    'NATDetectionResult',
    'NATDetector',
    'detect_nat_type',
    'get_cached_nat_type',
    # ICE connection (Dec 30, 2025 - Phase 4)
    'CandidateType',
    'ICECandidate',
    'ICEGatherer',
    'ICEChecker',
    'ice_establish_connection',
    # Transport metrics (Dec 30, 2025 - Phase 5)
    'TransportMetrics',
    'TransportMetricsTracker',
    'get_transport_metrics',
    'record_transport_request',
    'get_recommended_transport',
    # Connection pooling (Dec 30, 2025 - Phase 5)
    'ConnectionConfig',
    'PeerConnectionPool',
    'get_connection_pool',
    'get_pooled_session',
    'start_connection_pool',
    'stop_connection_pool',
    # Leader health probing (Dec 30, 2025 - Phase 6)
    'LeaderHealthProbe',
    'LeaderHealthResult',
    'LeaderHealthStatus',
    'LeaderProbeConfig',
    'get_leader_health_probe',
    # Graceful step-down (Dec 30, 2025 - Phase 6)
    'GracefulStepDown',
    'StepDownConfig',
    'StepDownReason',
    'StepDownResult',
    'step_down_leader',
    'select_best_successor',
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
