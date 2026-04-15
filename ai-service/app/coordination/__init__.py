"""Thin compatibility facade for active coordination entrypoints.

Prefer direct submodule imports for specialized coordination code. This package
keeps historical top-level exports lazily available while routing the small
bootstrap/status surface through `lifecycle` and `status_reporting`.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

_EXPORT_MODULES = {
    "app.coordination._exports_core": "_exports_core.py",
    "app.coordination._exports_sync": "_exports_sync.py",
    "app.coordination._exports_daemon": "_exports_daemon.py",
    "app.coordination._exports_events": "_exports_events.py",
    "app.coordination._exports_orchestrators": "_exports_orchestrators.py",
    "app.coordination._exports_utils": "_exports_utils.py",
}
_LAZY_EXPORT_CACHE: dict[str, object] = {}


def _read_declared_exports(filename: str) -> list[str]:
    path = Path(__file__).with_name(filename)
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, list):
                        return [str(item) for item in value]
    raise RuntimeError(f"Unable to resolve __all__ from {path}")


_EXPORT_NAME_TO_MODULE = {
    export_name: module_name
    for module_name, filename in _EXPORT_MODULES.items()
    for export_name in _read_declared_exports(filename)
}
_LEGACY_ALIAS_EXPORTS: dict[str, tuple[str, ...]] = {
    "core_events": ("module", "app.coordination.core_events"),
    "core_utils": ("module", "app.coordination.core_utils"),
    "get_all_coordinator_status": ("attr", "app.coordination.status_reporting", "get_all_coordinator_status"),
    "get_system_health": ("attr", "app.coordination.status_reporting", "get_system_health"),
    "initialize_all_coordinators": ("attr", "app.coordination.lifecycle", "initialize_all_coordinators"),
    "is_heartbeat_running": ("attr", "app.coordination.lifecycle", "is_heartbeat_running"),
    "shutdown_all_coordinators": ("attr", "app.coordination.lifecycle", "shutdown_all_coordinators"),
    "start_coordinator_heartbeats": ("attr", "app.coordination.lifecycle", "start_coordinator_heartbeats"),
    "stop_coordinator_heartbeats": ("attr", "app.coordination.lifecycle", "stop_coordinator_heartbeats"),
}


def _resolve_alias(name: str) -> object:
    kind, *target = _LEGACY_ALIAS_EXPORTS[name]
    if kind == "module":
        return importlib.import_module(target[0])
    return getattr(importlib.import_module(target[0]), target[1])


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORT_CACHE:
        return _LAZY_EXPORT_CACHE[name]
    if name in _EXPORT_NAME_TO_MODULE:
        value = getattr(importlib.import_module(_EXPORT_NAME_TO_MODULE[name]), name)
    elif name in _LEGACY_ALIAS_EXPORTS:
        value = _resolve_alias(name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _LAZY_EXPORT_CACHE[name] = value
    globals()[name] = value
    return value


__all__ = [*_EXPORT_NAME_TO_MODULE.keys(), *_LEGACY_ALIAS_EXPORTS.keys()]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
