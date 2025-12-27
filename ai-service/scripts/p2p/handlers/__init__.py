"""P2P Handler Mixins.

This package contains mixin classes for HTTP handlers extracted from p2p_orchestrator.py.
These mixins are designed to be inherited by P2POrchestrator.

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
"""

from .admin import AdminHandlersMixin
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
from .work_queue import WorkQueueHandlersMixin

__all__ = [
    "AdminHandlersMixin",
    "CMAESHandlersMixin",
    "ElectionHandlersMixin",
    "EloSyncHandlersMixin",
    "GauntletHandlersMixin",
    "GossipHandlersMixin",
    "RaftHandlersMixin",
    "RelayHandlersMixin",
    "SSHTournamentHandlersMixin",
    "SwimHandlersMixin",
    "TournamentHandlersMixin",
    "WorkQueueHandlersMixin",
]
