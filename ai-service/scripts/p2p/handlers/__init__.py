"""P2P Handler Mixins.

This package contains mixin classes for HTTP handlers extracted from p2p_orchestrator.py.
These mixins are designed to be inherited by P2POrchestrator, providing modular
endpoint implementations for the P2P cluster API.

Usage:
    from scripts.p2p.handlers import (
        ElectionHandlersMixin,
        GauntletHandlersMixin,
        RelayHandlersMixin,
        WorkQueueHandlersMixin,
    )

    class P2POrchestrator(
        WorkQueueHandlersMixin,
        ElectionHandlersMixin,
        RelayHandlersMixin,
        GauntletHandlersMixin,
    ):
        pass

Handler Categories:

    Core Cluster:
        - ElectionHandlersMixin: Leader election (Bully algorithm, voter quorum)
        - GossipHandlersMixin: Decentralized state sharing and manifests
        - RelayHandlersMixin: NAT-blocked node communication

    Protocol Integration (December 2025):
        - SwimHandlersMixin: SWIM membership status endpoints
        - RaftHandlersMixin: Raft consensus and work queue endpoints

    Work Distribution:
        - WorkQueueHandlersMixin: Distributed work queue management
        - GauntletHandlersMixin: Model evaluation distribution
        - TournamentHandlersMixin: Round-robin tournament execution
        - SSHTournamentHandlersMixin: SSH-based tournament execution

    Optimization:
        - CMAESHandlersMixin: Distributed hyperparameter optimization
        - EloSyncHandlersMixin: Elo rating synchronization

    Administration:
        - AdminHandlersMixin: Git status, code updates, health checks

    Network Discovery (December 2025):
        - NetworkDiscoveryMixin: Local IP/Tailscale detection, partition recovery

    Data Delivery (December 2025):
        - DeliveryHandlersMixin: Delivery verification and status endpoints

    Data Sync (December 2025):
        - SyncHandlersMixin: Cluster data sync (push/pull, receipts, file streaming)

See individual handler modules for endpoint documentation.
"""

from .base import BaseP2PHandler, P2PHandlerProtocol, make_json_response, make_error_response
# Dec 2025: Additional handler utilities
from .handlers_base import (
    safe_handler,
    leader_only,
    voter_only,
    EventBridgeManager,
    get_event_bridge,
    HandlerStatusMixin,
    success_response,
    error_response,
    parse_json_request,
    validate_node_id,
)
from .handlers_utils import (
    get_peer_info,
    get_alive_peers,
    format_timestamp,
    time_since,
    is_expired,
    RetryStrategy,
    RetryResult,
    MetricsCollector,
)
from .admin import AdminHandlersMixin
from .delivery import DeliveryHandlersMixin
from .network_discovery import NetworkDiscoveryMixin
from .cmaes import CMAESHandlersMixin
from .election import ElectionHandlersMixin
from .elo_sync import EloSyncHandlersMixin
from .gauntlet import GauntletHandlersMixin
from .gossip import GossipHandlersMixin
from .raft import RaftHandlersMixin
from .relay import RelayHandlersMixin
from .ssh_tournament import SSHTournamentHandlersMixin
from .swim import SwimHandlersMixin
from .tournament import TournamentHandlersMixin
from .registry import RegistryHandlersMixin
from .sync import SyncHandlersMixin
from .tables import TableHandlersMixin
from .work_queue import WorkQueueHandlersMixin

__all__ = [
    # Base class (Dec 2025 - handler consolidation)
    "BaseP2PHandler",
    "P2PHandlerProtocol",
    "make_json_response",
    "make_error_response",
    # Handler mixins
    "AdminHandlersMixin",
    "CMAESHandlersMixin",
    "DeliveryHandlersMixin",
    "ElectionHandlersMixin",
    "EloSyncHandlersMixin",
    "GauntletHandlersMixin",
    "GossipHandlersMixin",
    "NetworkDiscoveryMixin",
    "RaftHandlersMixin",
    "RegistryHandlersMixin",
    "RelayHandlersMixin",
    "SSHTournamentHandlersMixin",
    "SwimHandlersMixin",
    "SyncHandlersMixin",
    "TableHandlersMixin",
    "TournamentHandlersMixin",
    "WorkQueueHandlersMixin",
]
