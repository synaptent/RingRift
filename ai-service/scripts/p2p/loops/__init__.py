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
from .coordination_loops import (
    AutoScalingConfig,
    AutoScalingLoop,
    HealthAggregationConfig,
    HealthAggregationLoop,
)
from .data_loops import (
    DataAggregationConfig,
    DataAggregationLoop,
    ModelSyncConfig,
    ModelSyncLoop,
)
from .elo_sync_loop import EloSyncLoop
from .job_loops import (
    IdleDetectionConfig,
    IdleDetectionLoop,
    JobReaperConfig,
    JobReaperLoop,
    WorkerPullConfig,
    WorkerPullLoop,
    WorkQueueMaintenanceConfig,
    WorkQueueMaintenanceLoop,
)
from .network_loops import (
    IpDiscoveryConfig,
    IpDiscoveryLoop,
    NATManagementConfig,
    NATManagementLoop,
    TailscaleRecoveryConfig,
    TailscaleRecoveryLoop,
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
from .resilience_loops import (
    PredictiveMonitoringConfig,
    PredictiveMonitoringLoop,
    SelfHealingConfig,
    SelfHealingLoop,
)
from .training_sync_loop import TrainingSyncLoop
from .validation_loop import (
    ValidationConfig,
    ValidationLoop,
)

__all__ = [
    # Base
    "BackoffConfig",
    "BaseLoop",
    "LoopManager",
    "LoopStats",
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
    "ModelSyncConfig",
    "ModelSyncLoop",
    # Network
    "IpDiscoveryConfig",
    "IpDiscoveryLoop",
    "NATManagementConfig",
    "NATManagementLoop",
    "TailscaleRecoveryConfig",
    "TailscaleRecoveryLoop",
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
    "WorkerPullConfig",
    "WorkerPullLoop",
    "WorkQueueMaintenanceConfig",
    "WorkQueueMaintenanceLoop",
    # Training
    "TrainingSyncLoop",
    # Manifest
    "ManifestCollectionLoop",
    # Resilience
    "PredictiveMonitoringConfig",
    "PredictiveMonitoringLoop",
    "SelfHealingConfig",
    "SelfHealingLoop",
    # Validation
    "ValidationConfig",
    "ValidationLoop",
]
