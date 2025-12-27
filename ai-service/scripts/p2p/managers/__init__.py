"""P2P Orchestrator Managers.

This module contains domain-specific manager classes extracted from
p2p_orchestrator.py for better modularity and testability.

Managers:
- StateManager: SQLite persistence, cluster epoch tracking
- NodeSelector: Node ranking and selection for job dispatch
- DataSyncManager: Manifest collection and sync planning (TODO)
- SelfplayScheduler: Selfplay config selection and diversity (TODO)
- TrainingCoordinator: Training dispatch and promotion (TODO)
- JobExecutor: Job spawning and monitoring (TODO)
"""

from .node_selector import NodeSelector
from .state_manager import StateManager

__all__ = [
    "NodeSelector",
    "StateManager",
]
