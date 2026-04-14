"""P2P Orchestration module for distributed AI training cluster.

This module provides modular components for the P2P orchestrator:
- Configuration and constants
- Node and job management data structures
- Training coordination utilities
- Data synchronization helpers
- Webhook/notification services

The full P2P orchestrator (scripts/p2p_orchestrator.py) is being incrementally
modularized. New code should import from this module where possible.

Usage:
    from app.p2p import (
        P2PConfig,
        NodeRole,
        JobType,
        NodeInfo,
        TrainingThresholds,
    )

    # Check if a node can accept new jobs
    config = P2PConfig()
    if node.load_score < config.LOAD_MAX_FOR_NEW_JOBS:
        # Schedule job
        pass
"""

from .config import (
    # Constants are also available directly
    DEFAULT_PORT,
    GPU_POWER_RANKINGS,
    HEARTBEAT_INTERVAL,
    LEADER_LEASE_DURATION,
    PEER_TIMEOUT,
    P2PConfig,
    get_p2p_config,
)
from .models import (
    JobStatus,
    JobType,
    NodeHealth,
    NodeRole,
)
from .notifications import (
    WebhookConfig,
    send_webhook_notification,
)
from .training import (
    TrainingThresholds,
    calculate_training_priority,
    should_trigger_training,
)

# Hybrid coordinator for SWIM/Raft integration (Phase 2.4 - Dec 2025)
from .hybrid_coordinator import (
    CONSENSUS_MODE,
    MEMBERSHIP_MODE,
    HybridCoordinator,
    HybridStatus,
    create_hybrid_coordinator,
)

# SWIM adapter for gossip-based membership
from .swim_adapter import (
    SWIM_AVAILABLE,
    HybridMembershipManager,
    SwimConfig,
    SwimMembershipManager,
)

# Raft state machines for replicated work queue
from .raft_state import (
    PYSYNCOBJ_AVAILABLE,
    RAFT_ENABLED,
    ReplicatedJobAssignments,
    ReplicatedWorkQueue,
    WorkItem as RaftWorkItem,
    create_replicated_job_assignments,
    create_replicated_work_queue,
)

__all__ = [
    # Config constants
    "DEFAULT_PORT",
    "GPU_POWER_RANKINGS",
    "HEARTBEAT_INTERVAL",
    "LEADER_LEASE_DURATION",
    "PEER_TIMEOUT",
    # Models
    "JobStatus",
    "JobType",
    "NodeHealth",
    "NodeRole",
    # Config
    "P2PConfig",
    "get_p2p_config",
    # Training
    "TrainingThresholds",
    "calculate_training_priority",
    "should_trigger_training",
    # Notifications
    "WebhookConfig",
    "send_webhook_notification",
    # Hybrid coordinator (Phase 2.4)
    "CONSENSUS_MODE",
    "MEMBERSHIP_MODE",
    "HybridCoordinator",
    "HybridStatus",
    "create_hybrid_coordinator",
    # SWIM adapter
    "SWIM_AVAILABLE",
    "HybridMembershipManager",
    "SwimConfig",
    "SwimMembershipManager",
    # Raft state machines
    "PYSYNCOBJ_AVAILABLE",
    "RAFT_ENABLED",
    "RaftWorkItem",
    "ReplicatedJobAssignments",
    "ReplicatedWorkQueue",
    "create_replicated_job_assignments",
    "create_replicated_work_queue",
]


def __dir__() -> list[str]:
    """Expose the intended P2P surface for discoverability."""

    return sorted(set(globals()) | set(__all__))
