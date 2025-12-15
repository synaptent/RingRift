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
    P2PConfig,
    get_p2p_config,
    # Constants are also available directly
    DEFAULT_PORT,
    HEARTBEAT_INTERVAL,
    PEER_TIMEOUT,
    LEADER_LEASE_DURATION,
    GPU_POWER_RANKINGS,
)

from .models import (
    NodeRole,
    JobType,
    JobStatus,
    NodeHealth,
)

from .training import (
    TrainingThresholds,
    calculate_training_priority,
    should_trigger_training,
)

from .notifications import (
    WebhookConfig,
    send_webhook_notification,
)

__all__ = [
    # Config
    "P2PConfig",
    "get_p2p_config",
    "DEFAULT_PORT",
    "HEARTBEAT_INTERVAL",
    "PEER_TIMEOUT",
    "LEADER_LEASE_DURATION",
    "GPU_POWER_RANKINGS",
    # Models
    "NodeRole",
    "JobType",
    "JobStatus",
    "NodeHealth",
    # Training
    "TrainingThresholds",
    "calculate_training_priority",
    "should_trigger_training",
    # Notifications
    "WebhookConfig",
    "send_webhook_notification",
]
