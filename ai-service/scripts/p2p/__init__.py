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
    from scripts.p2p.resource_utils import check_all_resources
    from scripts.p2p.network import get_client_session, peer_request

Module Structure:
    - types.py: Enums (NodeRole, JobType)
    - constants.py: Configuration constants
    - models.py: Dataclasses (NodeInfo, ClusterJob, etc.)
    - resource.py: Resource checking utilities
    - network.py: HTTP client and circuit breaker utilities
    - utils.py: General utilities (systemd, etc.)
"""
from importlib import import_module

_EXPORTS = {
    "MetricsManager": (".metrics_manager", "MetricsManager"),
    "MetricsManagerMixin": (".metrics_manager", "MetricsManagerMixin"),
    "ResourceDetector": (".resource_detector", "ResourceDetector"),
    "ResourceDetectorMixin": (".resource_detector", "ResourceDetectorMixin"),
    "NetworkUtils": (".network_utils", "NetworkUtils"),
    "NetworkUtilsMixin": (".network_utils", "NetworkUtilsMixin"),
    "PeerManagerMixin": (".peer_manager", "PeerManagerMixin"),
    "get_peer_manager": (".peer_manager", "get_peer_manager"),
    "set_peer_manager": (".peer_manager", "set_peer_manager"),
    "MembershipMixin": (".membership_mixin", "MembershipMixin"),
    "ConsensusMixin": (".consensus_mixin", "ConsensusMixin"),
    "SwimHandlersMixin": (".handlers.swim", "SwimHandlersMixin"),
    "RaftHandlersMixin": (".handlers.raft", "RaftHandlersMixin"),
    "GossipMetricsMixin": (".gossip_protocol", "GossipProtocolMixin"),
    "GossipProtocolMixin": (".gossip_protocol", "GossipProtocolMixin"),
    "EventEmissionMixin": (".event_emission_mixin", "EventEmissionMixin"),
    "FailoverIntegrationMixin": (".failover_integration", "FailoverIntegrationMixin"),
    "is_failover_available": (".failover_integration", "is_failover_available"),
    "TransportCascade": (".transport_cascade", "TransportCascade"),
    "get_transport_cascade": (".transport_cascade", "get_transport_cascade"),
    "TransportTier": (".transport_cascade", "TransportTier"),
    "ProtocolUnion": (".protocol_union", "ProtocolUnion"),
    "get_protocol_union": (".protocol_union", "get_protocol_union"),
    "MembershipSource": (".protocol_union", "MembershipSource"),
    "UnionDiscovery": (".union_discovery", "UnionDiscovery"),
    "get_union_discovery": (".union_discovery", "get_union_discovery"),
    "DiscoveredPeer": (".union_discovery", "DiscoveredPeer"),
    "NATType": (".nat_detection", "NATType"),
    "NATDetectionResult": (".nat_detection", "NATDetectionResult"),
    "NATDetector": (".nat_detection", "NATDetector"),
    "detect_nat_type": (".nat_detection", "detect_nat_type"),
    "get_cached_nat_type": (".nat_detection", "get_cached_nat_type"),
    "CandidateType": (".ice_connection", "CandidateType"),
    "ICECandidate": (".ice_connection", "ICECandidate"),
    "ICEGatherer": (".ice_connection", "ICEGatherer"),
    "ICEChecker": (".ice_connection", "ICEChecker"),
    "ice_establish_connection": (".ice_connection", "establish_connection"),
    "TransportMetrics": (".transport_metrics", "TransportMetrics"),
    "TransportMetricsTracker": (".transport_metrics", "TransportMetricsTracker"),
    "get_transport_metrics": (".transport_metrics", "get_transport_metrics"),
    "record_transport_request": (".transport_metrics", "record_transport_request"),
    "get_recommended_transport": (".transport_metrics", "get_recommended_transport"),
    "ConnectionConfig": (".connection_pool", "ConnectionConfig"),
    "PeerConnectionPool": (".connection_pool", "PeerConnectionPool"),
    "get_connection_pool": (".connection_pool", "get_connection_pool"),
    "get_pooled_session": (".connection_pool", "get_pooled_session"),
    "start_connection_pool": (".connection_pool", "start_connection_pool"),
    "stop_connection_pool": (".connection_pool", "stop_connection_pool"),
    "LeaderHealthProbe": (".leader_health", "LeaderHealthProbe"),
    "LeaderHealthResult": (".leader_health", "LeaderHealthResult"),
    "LeaderHealthStatus": (".leader_health", "LeaderHealthStatus"),
    "LeaderProbeConfig": (".leader_health", "LeaderProbeConfig"),
    "get_leader_health_probe": (".leader_health", "get_leader_health_probe"),
    "GracefulStepDown": (".graceful_stepdown", "GracefulStepDown"),
    "StepDownConfig": (".graceful_stepdown", "StepDownConfig"),
    "StepDownReason": (".graceful_stepdown", "StepDownReason"),
    "StepDownResult": (".graceful_stepdown", "StepDownResult"),
    "step_down_leader": (".graceful_stepdown", "step_down_leader"),
    "select_best_successor": (".graceful_stepdown", "select_best_successor"),
    "DEFAULT_PORT": (".constants", "DEFAULT_PORT"),
    "DISK_CRITICAL_THRESHOLD": (".constants", "DISK_CRITICAL_THRESHOLD"),
    "DISK_WARNING_THRESHOLD": (".constants", "DISK_WARNING_THRESHOLD"),
    "ELECTION_TIMEOUT": (".constants", "ELECTION_TIMEOUT"),
    "GPU_POWER_RANKINGS": (".constants", "GPU_POWER_RANKINGS"),
    "HEARTBEAT_INTERVAL": (".constants", "HEARTBEAT_INTERVAL"),
    "LEADER_LEASE_DURATION": (".constants", "LEADER_LEASE_DURATION"),
    "LOAD_MAX_FOR_NEW_JOBS": (".constants", "LOAD_MAX_FOR_NEW_JOBS"),
    "MEMORY_CRITICAL_THRESHOLD": (".constants", "MEMORY_CRITICAL_THRESHOLD"),
    "MEMORY_WARNING_THRESHOLD": (".constants", "MEMORY_WARNING_THRESHOLD"),
    "PEER_TIMEOUT": (".constants", "PEER_TIMEOUT"),
    "STATE_DIR": (".constants", "STATE_DIR"),
    "AsyncLockWrapper": (".network", "AsyncLockWrapper"),
    "NonBlockingAsyncLockWrapper": (".network", "NonBlockingAsyncLockWrapper"),
    "LOCK_ORDER": (".network", "LOCK_ORDER"),
    "ClusterDataManifest": (".models", "ClusterDataManifest"),
    "ClusterJob": (".models", "ClusterJob"),
    "ClusterStatus": (".client", "ClusterStatus"),
    "ClusterSyncPlan": (".models", "ClusterSyncPlan"),
    "DataFileInfo": (".models", "DataFileInfo"),
    "DataSyncJob": (".models", "DataSyncJob"),
    "DistributedCMAESState": (".models", "DistributedCMAESState"),
    "DistributedTournamentState": (".models", "DistributedTournamentState"),
    "ImprovementLoopState": (".models", "ImprovementLoopState"),
    "JobRequest": (".client", "JobRequest"),
    "JobResult": (".client", "JobResult"),
    "JobType": (".types", "JobType"),
    "NodeDataManifest": (".models", "NodeDataManifest"),
    "NodeInfo": (".models", "NodeInfo"),
    "NodeRole": (".types", "NodeRole"),
    "P2PClient": (".client", "P2PClient"),
    "P2PClientError": (".client", "P2PClientError"),
    "SSHTournamentRun": (".models", "SSHTournamentRun"),
    "TrainingJob": (".models", "TrainingJob"),
    "TrainingThresholds": (".models", "TrainingThresholds"),
    "check_all_resources": (".resource_utils", "check_all_resources"),
    "check_disk_has_capacity": (".resource_utils", "check_disk_has_capacity"),
    "check_peer_circuit": (".network", "check_peer_circuit"),
    "get_client": (".client", "get_client"),
    "get_client_session": (".network", "get_client_session"),
    "get_cluster_status": (".client", "get_cluster_status"),
    "get_disk_usage_percent": (".resource_utils", "get_disk_usage_percent"),
    "peer_request": (".network", "peer_request"),
    "record_peer_failure": (".network", "record_peer_failure"),
    "record_peer_success": (".network", "record_peer_success"),
    "submit_selfplay_job": (".client", "submit_selfplay_job"),
    "submit_training_job": (".client", "submit_training_job"),
    "systemd_notify_ready": (".utils", "systemd_notify_ready"),
    "systemd_notify_watchdog": (".utils", "systemd_notify_watchdog"),
}

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


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
