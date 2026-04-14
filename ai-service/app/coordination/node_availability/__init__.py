"""Node Availability Management (December 2025).

Synchronizes cloud provider instance state with distributed_hosts.yaml configuration.
Solves the problem of stale configuration where nodes are marked 'ready' but actually
terminated in the cloud provider.

Key components:
- NodeAvailabilityDaemon: Main daemon that polls providers and updates config
- StateChecker: Base class for provider-specific state checkers
- ConfigUpdater: Atomic YAML updates with backup

Usage:
    from app.coordination.node_availability import (
        NodeAvailabilityDaemon,
        get_node_availability_daemon,
    )

    daemon = get_node_availability_daemon()
    await daemon.start()

Environment variables:
    RINGRIFT_NODE_AVAILABILITY_ENABLED: Enable/disable daemon (default: true)
    RINGRIFT_NODE_AVAILABILITY_DRY_RUN: Log only, no YAML writes (default: true)
    RINGRIFT_NODE_AVAILABILITY_INTERVAL: Check interval in seconds (default: 300)
"""

from app.coordination.node_availability.state_checker import (
    ProviderInstanceState,
    InstanceInfo,
    StateChecker,
    STATE_TO_YAML_STATUS,
)
from app.coordination.node_availability.config_updater import (
    ConfigUpdater,
    ConfigUpdateResult,
)
from app.coordination.node_availability.daemon import (
    NodeAvailabilityDaemon,
    NodeAvailabilityConfig,
    get_node_availability_daemon,
    reset_daemon_instance,
)

__all__ = [
    # State checker
    "ProviderInstanceState",
    "InstanceInfo",
    "StateChecker",
    "STATE_TO_YAML_STATUS",
    # Config updater
    "ConfigUpdater",
    "ConfigUpdateResult",
    # Daemon
    "NodeAvailabilityDaemon",
    "NodeAvailabilityConfig",
    "get_node_availability_daemon",
    "reset_daemon_instance",
]


def __dir__() -> list[str]:
    """Expose the intended package surface for discoverability and tests."""

    return sorted(set(globals()) | set(__all__))
