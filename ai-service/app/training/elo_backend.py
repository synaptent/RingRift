"""Backend discovery helpers for the Elo service."""

from __future__ import annotations

import logging
from typing import Any

from app.training.elo_types import EloBackendType

logger = logging.getLogger(__name__)

_raft_elo_store_available: bool | None = None
_raft_elo_store: Any = None
_raft_elo_node_id: str | None = None


def check_raft_elo_store_available() -> bool:
    """Check whether the replicated Elo store is available."""
    global _raft_elo_store_available, _raft_elo_store, _raft_elo_node_id

    if _raft_elo_store_available is not None:
        return _raft_elo_store_available

    try:
        from app.p2p.raft_state import PYSYNCOBJ_AVAILABLE, RAFT_ENABLED

        if not RAFT_ENABLED or not PYSYNCOBJ_AVAILABLE:
            logger.debug("Raft Elo store not available: Raft disabled or pysyncobj missing")
            _raft_elo_store_available = False
            return False

        try:
            import sys

            if "scripts.p2p_orchestrator" in sys.modules:
                p2p_module = sys.modules["scripts.p2p_orchestrator"]
                if hasattr(p2p_module, "P2POrchestrator"):
                    orchestrator_cls = p2p_module.P2POrchestrator
                    if hasattr(orchestrator_cls, "_instance") and orchestrator_cls._instance:
                        orchestrator = orchestrator_cls._instance
                        if hasattr(orchestrator, "replicated_elo_store"):
                            elo_store = orchestrator.replicated_elo_store
                            if elo_store and hasattr(elo_store, "is_ready") and elo_store.is_ready:
                                _raft_elo_store = elo_store
                                _raft_elo_node_id = getattr(orchestrator, "node_id", None)
                                _raft_elo_store_available = True
                                logger.info("Raft Elo store available via P2P orchestrator")
                                return True
        except Exception as exc:
            logger.debug("Could not get Raft Elo store from orchestrator: %s", exc)

        _raft_elo_store_available = False
        return False
    except ImportError as exc:
        logger.debug("Raft Elo store not available: %s", exc)
        _raft_elo_store_available = False
        return False


def reset_raft_elo_store_cache() -> None:
    """Reset cached Raft backend discovery state."""
    global _raft_elo_store_available, _raft_elo_store, _raft_elo_node_id
    _raft_elo_store_available = None
    _raft_elo_store = None
    _raft_elo_node_id = None


def get_raft_elo_store() -> Any:
    """Return the replicated Elo store if it is available."""
    if check_raft_elo_store_available():
        return _raft_elo_store
    return None


__all__ = [
    "EloBackendType",
    "check_raft_elo_store_available",
    "get_raft_elo_store",
    "reset_raft_elo_store_cache",
]
