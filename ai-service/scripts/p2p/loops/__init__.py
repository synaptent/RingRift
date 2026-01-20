"""P2P Orchestrator Background Loops.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This package contains background loop implementations extracted from p2p_orchestrator.py
for better modularity and testability.

Loop Categories:
- base.py: BaseLoop abstract class with error handling and backoff
- queue_populator_loop.py: Work queue population for selfplay jobs
- elo_sync_loop.py: Elo rating synchronization across cluster
- data_loops.py: Model sync, data aggregation
- network_loops.py: IP updates, Tailscale recovery
- discovery_loop.py: UDP broadcast and follower peer discovery
- coordination_loops.py: Auto-scaling, health aggregation
- job_loops.py: Job reaper, idle detection
- resilience_loops.py: Self-healing, predictive monitoring

Usage:
    from scripts.p2p.loops import BaseLoop, BackoffConfig, LoopManager, LoopStats
    from scripts.p2p.loops import QueuePopulatorLoop, JobReaperLoop

    class MyLoop(BaseLoop):
        async def _run_once(self) -> None:
            # Your loop logic here
            pass

    # Single loop usage
    loop = MyLoop(name="my_loop", interval=60.0)
    await loop.run_forever()

    # Multiple loops with manager
    manager = LoopManager()
    manager.register(MyLoop(name="loop1", interval=30.0))
    manager.register(QueuePopulatorLoop(
        get_role=lambda: NodeRole.LEADER,
        get_selfplay_scheduler=lambda: scheduler,
    ))
    await manager.start_all()
    # ... later ...
    await manager.stop_all()
"""

from .base import (
    BackoffConfig,
    BaseLoop,
    LoopManager,
    LoopStats,
)
from .loop_constants import (
    AUTO_UPDATE_ENABLED,
    DEFAULT_DISCOVERY_PORT,
    DEFAULT_P2P_PORT,
    JobDefaults,
    LoopIntervals,
    LoopLimits,
    LoopThresholds,
    LoopTimeouts,
)
from .coordination_loops import (
    AutoScalingConfig,
    AutoScalingLoop,
    HealthAggregationConfig,
    HealthAggregationLoop,
)
from .data_loops import (
    DataAggregationConfig,
    DataAggregationLoop,
    DataManagementConfig,
    DataManagementLoop,
    ModelSyncConfig,
    ModelSyncLoop,
)
from .elo_sync_loop import EloSyncLoop
from .job_loops import (
    IdleDetectionConfig,
    IdleDetectionLoop,
    JobReaperConfig,
    JobReaperLoop,
    JobReassignmentConfig,
    JobReassignmentLoop,
    PredictiveScalingConfig,
    PredictiveScalingLoop,
    SpawnVerificationConfig,
    SpawnVerificationLoop,
    WorkerPullConfig,
    WorkerPullLoop,
    WorkQueueMaintenanceConfig,
    WorkQueueMaintenanceLoop,
)
from .maintenance_loops import (
    CircuitBreakerDecayConfig,
    CircuitBreakerDecayLoop,
    GitUpdateConfig,
    GitUpdateLoop,
)
from .network_loops import (
    AwsIpUpdateLoop,
    HeartbeatConfig,
    HeartbeatLoop,
    IpDiscoveryConfig,
    IpDiscoveryLoop,
    NATManagementConfig,
    NATManagementLoop,
    ProviderIpUpdateConfig,
    TailscaleIpUpdateLoop,
    TailscaleDaemonHealthConfig,
    TailscaleDaemonHealthLoop,
    TailscaleKeepaliveConfig,
    TailscaleKeepaliveLoop,
    TailscalePeerDiscoveryConfig,
    TailscalePeerDiscoveryLoop,
    TailscaleRecoveryConfig,
    TailscaleRecoveryLoop,
    VastIpUpdateLoop,
    VoterHeartbeatConfig,
    VoterHeartbeatLoop,
)
from .discovery_loop import (
    DEFAULT_DISCOVERY_PORT,
    FollowerDiscoveryConfig,
    FollowerDiscoveryLoop,
    UdpDiscoveryConfig,
    UdpDiscoveryLoop,
)
from .manifest_collection_loop import ManifestCollectionLoop
from .queue_populator_loop import QueuePopulatorLoop
from .peer_cleanup_loop import (
    PeerCleanupConfig,
    PeerCleanupLoop,
)
from .peer_recovery_loop import (
    PeerRecoveryConfig,
    PeerRecoveryLoop,
)
from .cluster_healing_loop import (
    ClusterHealingConfig,
    ClusterHealingLoop,
    HostInfo as ClusterHostInfo,
)
from .remote_p2p_recovery_loop import (
    RemoteP2PRecoveryConfig,
    RemoteP2PRecoveryLoop,
)
from .resilience_loops import (
    PredictiveMonitoringConfig,
    PredictiveMonitoringLoop,
    SelfHealingConfig,
    SelfHealingLoop,
    SplitBrainDetectionConfig,
    SplitBrainDetectionLoop,
)
from .training_sync_loop import TrainingSyncLoop
from .validation_loop import (
    ValidationConfig,
    ValidationLoop,
)
from .autonomous_queue_loop import (
    AutonomousQueueConfig,
    AutonomousQueuePopulationLoop,
    AutonomousQueueState,
)
from .leader_probe_loop import LeaderProbeLoop
from .relay_health_loop import (
    RelayHealthConfig,
    RelayHealthLoop,
    RelayHealthStatus,
)
from .http_server_health_loop import (
    HttpServerHealthConfig,
    HttpServerHealthLoop,
)
from .swim_membership_loop import (
    SwimMembershipConfig,
    SwimMembershipLoop,
)
from .gossip_state_cleanup_loop import (
    GossipCleanupStats,
    GossipStateCleanupConfig,
    GossipStateCleanupLoop,
)
from .comprehensive_evaluation_loop import (
    ComprehensiveEvaluationConfig,
    ComprehensiveEvaluationLoop,
    CycleStats as ComprehensiveEvaluationCycleStats,
    EvaluationDispatchResult,
)
from .tournament_data_pipeline_loop import (
    TournamentDataPipelineConfig,
    TournamentDataPipelineLoop,
    PipelineCycleStats as TournamentPipelineCycleStats,
    DatabaseStats as TournamentDatabaseStats,
    ExportResult as TournamentExportResult,
)
from .evaluation_worker_loop import (
    EvaluationWorkerConfig,
    EvaluationWorkerLoop,
    create_evaluation_worker_loop,
)
from .orchestrator_context import (
    OrchestratorContext,
    IdleDetectionContext,
    JobReaperContext,
    PredictiveScalingContext,
)
from .quorum_crisis_discovery_loop import (
    QuorumCrisisDiscoveryLoop,
    QuorumCrisisConfig,
    CrisisStats,
)
from .voter_config_sync_loop import (
    VoterConfigSyncLoop,
    PeerConfigInfo,
    DriftResult,
)

__all__ = [
    # Base
    "BackoffConfig",
    "BaseLoop",
    "LoopManager",
    "LoopStats",
    # Constants (Sprint 10)
    "AUTO_UPDATE_ENABLED",
    "DEFAULT_DISCOVERY_PORT",
    "DEFAULT_P2P_PORT",
    "JobDefaults",
    "LoopIntervals",
    "LoopLimits",
    "LoopThresholds",
    # Queue
    "QueuePopulatorLoop",
    # Elo
    "EloSyncLoop",
    # Coordination
    "AutoScalingConfig",
    "AutoScalingLoop",
    "HealthAggregationConfig",
    "HealthAggregationLoop",
    # Data
    "DataAggregationConfig",
    "DataAggregationLoop",
    "DataManagementConfig",
    "DataManagementLoop",
    "ModelSyncConfig",
    "ModelSyncLoop",
    # Maintenance
    "GitUpdateConfig",
    "GitUpdateLoop",
    # Network
    "AwsIpUpdateLoop",
    "HeartbeatConfig",
    "HeartbeatLoop",
    "IpDiscoveryConfig",
    "IpDiscoveryLoop",
    "NATManagementConfig",
    "NATManagementLoop",
    "ProviderIpUpdateConfig",
    "TailscaleIpUpdateLoop",
    "TailscalePeerDiscoveryConfig",
    "TailscalePeerDiscoveryLoop",
    "TailscaleRecoveryConfig",
    "TailscaleRecoveryLoop",
    "TailscaleDaemonHealthConfig",
    "TailscaleDaemonHealthLoop",
    "TailscaleKeepaliveConfig",
    "TailscaleKeepaliveLoop",
    "VastIpUpdateLoop",
    "VoterHeartbeatConfig",
    "VoterHeartbeatLoop",
    # Discovery
    "DEFAULT_DISCOVERY_PORT",
    "FollowerDiscoveryConfig",
    "FollowerDiscoveryLoop",
    "UdpDiscoveryConfig",
    "UdpDiscoveryLoop",
    # Jobs
    "IdleDetectionConfig",
    "IdleDetectionLoop",
    "JobReaperConfig",
    "JobReaperLoop",
    "JobReassignmentConfig",
    "JobReassignmentLoop",
    "PredictiveScalingConfig",
    "PredictiveScalingLoop",
    "SpawnVerificationConfig",
    "SpawnVerificationLoop",
    "WorkerPullConfig",
    "WorkerPullLoop",
    "WorkQueueMaintenanceConfig",
    "WorkQueueMaintenanceLoop",
    # Training
    "TrainingSyncLoop",
    # Manifest
    "ManifestCollectionLoop",
    # Peer Cleanup (Dec 2025)
    "PeerCleanupConfig",
    "PeerCleanupLoop",
    # Peer Recovery (Dec 2025)
    "PeerRecoveryConfig",
    "PeerRecoveryLoop",
    # Cluster Healing (Jan 2026)
    "ClusterHealingConfig",
    "ClusterHealingLoop",
    "ClusterHostInfo",
    # Remote P2P Recovery (Dec 2025)
    "RemoteP2PRecoveryConfig",
    "RemoteP2PRecoveryLoop",
    # Resilience
    "PredictiveMonitoringConfig",
    "PredictiveMonitoringLoop",
    "SelfHealingConfig",
    "SelfHealingLoop",
    "SplitBrainDetectionConfig",
    "SplitBrainDetectionLoop",
    # Validation
    "ValidationConfig",
    "ValidationLoop",
    # Autonomous Queue (Jan 4, 2026 - Phase 2 P2P Resilience)
    "AutonomousQueueConfig",
    "AutonomousQueuePopulationLoop",
    "AutonomousQueueState",
    # Leader Probe (Jan 4, 2026 - Phase 5 P2P Resilience)
    "LeaderProbeLoop",
    # Relay Health (Jan 5, 2026 - Task 8.5)
    "RelayHealthConfig",
    "RelayHealthLoop",
    "RelayHealthStatus",
    # HTTP Server Health (Jan 7, 2026 - Zombie Detection)
    "HttpServerHealthConfig",
    "HttpServerHealthLoop",
    # SWIM Membership (Jan 9, 2026 - Fast Failure Detection)
    "SwimMembershipConfig",
    "SwimMembershipLoop",
    # Gossip State Cleanup (Jan 7, 2026 - Memory Leak Fix)
    "GossipCleanupStats",
    "GossipStateCleanupConfig",
    "GossipStateCleanupLoop",
    # Comprehensive Evaluation (Jan 2026 - Model Evaluation Pipeline)
    "ComprehensiveEvaluationConfig",
    "ComprehensiveEvaluationLoop",
    "ComprehensiveEvaluationCycleStats",
    "EvaluationDispatchResult",
    # Tournament Data Pipeline (Jan 2026 - Training Data Export)
    "TournamentDataPipelineConfig",
    "TournamentDataPipelineLoop",
    "TournamentPipelineCycleStats",
    "TournamentDatabaseStats",
    "TournamentExportResult",
    # Evaluation Worker (Jan 9, 2026 - Cluster-Wide Model Evaluation)
    "EvaluationWorkerConfig",
    "EvaluationWorkerLoop",
    "create_evaluation_worker_loop",
    # Orchestrator Context (Jan 9, 2026 - Phase 2 Decomposition)
    "OrchestratorContext",
    "IdleDetectionContext",
    "JobReaperContext",
    "PredictiveScalingContext",
    # Quorum Crisis Discovery (Jan 2026 - Fast recovery during quorum loss)
    "QuorumCrisisDiscoveryLoop",
    "QuorumCrisisConfig",
    "CrisisStats",
    # Voter Config Sync (Jan 20, 2026 - Consensus-safe config synchronization)
    "VoterConfigSyncLoop",
    "PeerConfigInfo",
    "DriftResult",
]
