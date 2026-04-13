"""Cluster coordination modules.

This package consolidates cluster-related coordination:
- health: Node and host health monitoring
- sync: Data synchronization
- transport: Cluster transport layer
- p2p: Peer-to-peer backend

December 2025: Consolidation from 75 → 15 modules.

Usage:
    from app.coordination.cluster.health import UnifiedHealthManager
    from app.coordination.cluster.sync import SyncScheduler
"""

from __future__ import annotations

import importlib

_SUBMODULES = {
    "health": ".health",
    "sync": ".sync",
    "transport": "app.coordination.cluster_transport",
    "p2p": "app.coordination.p2p_backend",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    """Resolve the documented cluster package submodules lazily."""
    if name in _SUBMODULES:
        module_path = _SUBMODULES[name]
        if module_path.startswith("."):
            return importlib.import_module(module_path, __name__)
        return importlib.import_module(module_path)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
