"""P2P Orchestrator Managers.

This module contains domain-specific manager classes extracted from
p2p_orchestrator.py for better modularity and testability.

Managers:
- StateManager: SQLite persistence, cluster epoch tracking
- NodeSelector: Node ranking and selection for job dispatch
- SyncPlanner: Manifest collection and sync planning (December 2025)
- JobManager: Job spawning and lifecycle management (December 2025)
- TrainingCoordinator: Training dispatch and promotion (December 2025)
- SelfplayScheduler: Selfplay config selection and diversity (December 2025)
- WorkDiscoveryManager: Multi-channel work discovery (January 2026)

Factory:
- ManagerFactory: Dependency injection for all managers (January 2026)
- ManagerConfig: Shared configuration for factory
"""

from .job_manager import JobManager
from .manager_factory import (
    ManagerConfig,
    ManagerFactory,
    get_manager_factory,
    init_manager_factory,
    reset_manager_factory,
)
from .node_selector import NodeSelector
from .selfplay_scheduler import DiversityMetrics, SelfplayScheduler
from .state_manager import StateManager
from .sync_planner import SyncPlanner, SyncPlannerConfig, SyncStats
from .training_coordinator import TrainingCoordinator
from .work_discovery_manager import (
    DiscoveryChannel,
    DiscoveryResult,
    WorkDiscoveryConfig,
    WorkDiscoveryManager,
    WorkDiscoveryStats,
    get_work_discovery_manager,
    reset_work_discovery_manager,
    set_work_discovery_manager,
)
# January 2026: Phase 1 P2P Orchestrator Deep Decomposition
from .job_orchestration_manager import (
    JobOrchestrationConfig,
    JobOrchestrationManager,
    JobOrchestrationStats,
    create_job_orchestration_manager,
)
# January 2026: Phase 2 Quorum Manager extraction
from .quorum_manager import (
    QuorumConfig,
    QuorumManager,
    VoterHealthStatus,
    create_quorum_manager,
    get_quorum_manager,
    set_quorum_manager,
)
# January 2026: Phase 3 Network Config Manager extraction
from .network_config_manager import (
    NetworkConfig,
    NetworkConfigManager,
    NetworkState,
    create_network_config_manager,
    get_network_config_manager,
    set_network_config_manager,
)

__all__ = [
    "DiscoveryChannel",
    "DiscoveryResult",
    "DiversityMetrics",
    "JobManager",
    "ManagerConfig",
    "ManagerFactory",
    "NodeSelector",
    "SelfplayScheduler",
    "StateManager",
    "SyncPlanner",
    "SyncPlannerConfig",
    "SyncStats",
    "TrainingCoordinator",
    "WorkDiscoveryConfig",
    "WorkDiscoveryManager",
    "WorkDiscoveryStats",
    "get_manager_factory",
    "get_work_discovery_manager",
    "init_manager_factory",
    "reset_manager_factory",
    "reset_work_discovery_manager",
    "set_work_discovery_manager",
    # January 2026: Phase 1 P2P Orchestrator Deep Decomposition
    "JobOrchestrationConfig",
    "JobOrchestrationManager",
    "JobOrchestrationStats",
    "create_job_orchestration_manager",
    # Quorum Manager
    "QuorumConfig",
    "QuorumManager",
    "VoterHealthStatus",
    "create_quorum_manager",
    "get_quorum_manager",
    "set_quorum_manager",
    # Network Config Manager
    "NetworkConfig",
    "NetworkConfigManager",
    "NetworkState",
    "create_network_config_manager",
    "get_network_config_manager",
    "set_network_config_manager",
]
