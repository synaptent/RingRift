"""Canonical cluster coordination package exports."""

from __future__ import annotations

import importlib

_SUBMODULES = {
    "health": ".health",
    "transport": "app.coordination.cluster_transport",
    "p2p": "app.coordination.p2p_backend",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    """Resolve documented package submodules lazily."""
    if name in _SUBMODULES:
        module_path = _SUBMODULES[name]
        if module_path.startswith("."):
            value = importlib.import_module(module_path, __name__)
        else:
            value = importlib.import_module(module_path)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
