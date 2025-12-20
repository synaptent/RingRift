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

__all__ = [
    "DEFAULT_PORT",
    "GPU_POWER_RANKINGS",
    "HEARTBEAT_INTERVAL",
    "LEADER_LEASE_DURATION",
    "PEER_TIMEOUT",
    "JobStatus",
    "JobType",
    "NodeHealth",
    # Models
    "NodeRole",
    # Config
    "P2PConfig",
    # Training
    "TrainingThresholds",
    # Notifications
    "WebhookConfig",
    "calculate_training_priority",
    "get_p2p_config",
    "send_webhook_notification",
    "should_trigger_training",
]
