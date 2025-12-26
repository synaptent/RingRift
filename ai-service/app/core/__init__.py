"""Core shared infrastructure for the RingRift AI service.

This package provides standardized utilities used across all scripts:
- logging_config: Unified logging setup
- error_handler: Retry decorators, error recovery, emergency halt
- shutdown: Graceful shutdown coordination (December 2025)
- singleton_mixin: Thread-safe singleton patterns (December 2025)
- marshalling: Unified serialization patterns (December 2025)
- node: Unified NodeInfo dataclass (December 2025)
- ssh: Unified SSH helper module (December 2025)

Note (Dec 2025): Uses lazy imports to avoid loading heavy dependencies (torch via utils.paths).
Direct submodule imports like `from app.core.async_context import X` are fast.
"""

from __future__ import annotations

__all__ = [
    # Marshalling/Serialization (December 2025)
    "Codec",
    "Serializable",
    "SerializationError",
    "deserialize",
    "from_json",
    "register_codec",
    "serialize",
    "to_json",
    # Error handling (lazy - imports utils.paths)
    "FatalError",
    "RetryableError",
    "RingRiftError",
    "retry",
    "retry_async",
    "with_emergency_halt_check",
    # Shutdown (December 2025)
    "ShutdownManager",
    "get_shutdown_manager",
    "is_shutting_down",
    "on_shutdown",
    "request_shutdown",
    "shutdown_scope",
    # Singleton patterns (December 2025)
    "SingletonMeta",
    "SingletonMixin",
    "ThreadSafeSingletonMixin",
    "singleton",
    # Background tasks (December 2025)
    "TaskInfo",
    "TaskManager",
    "TaskState",
    "background_task",
    "get_task_manager",
    # Logging (fast)
    "get_logger",
    "setup_logging",
    # Node dataclass (December 2025)
    "ConnectionInfo",
    "GPUInfo",
    "HealthStatus",
    "JobStatus",
    "NodeHealth",
    "NodeInfo",
    "NodeRole",
    "NodeState",
    "Provider",
    "ProviderInfo",
    "ResourceMetrics",
    # SSH utilities (December 2025)
    "SSHClient",
    "SSHConfig",
    "SSHResult",
    "get_ssh_client",
    "run_ssh_command",
    "run_ssh_command_async",
    "run_ssh_command_sync",
]

# =============================================================================
# Fast imports (no heavy dependencies)
# =============================================================================

from app.core.logging_config import get_logger, setup_logging
from app.core.shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    is_shutting_down,
    on_shutdown,
    request_shutdown,
    shutdown_scope,
)
from app.core.singleton_mixin import (
    SingletonMeta,
    SingletonMixin,
    ThreadSafeSingletonMixin,
    singleton,
)
from app.core.tasks import (
    TaskInfo,
    TaskManager,
    TaskState,
    background_task,
    get_task_manager,
)

# =============================================================================
# Lazy imports for modules with heavy dependencies
# =============================================================================

_lazy_cache: dict = {}


def __getattr__(name: str):
    """Lazy import for modules with heavy dependencies."""

    # Error handling (imports app.utils.paths -> potential torch chain)
    if name in ("FatalError", "RetryableError", "RingRiftError",
                "retry", "retry_async", "with_emergency_halt_check"):
        if "error_handler" not in _lazy_cache:
            from app.core.error_handler import (
                FatalError as _FE,
                RetryableError as _RE,
                RingRiftError as _RRE,
                retry as _r,
                retry_async as _ra,
                with_emergency_halt_check as _wehc,
            )
            _lazy_cache["error_handler"] = {
                "FatalError": _FE,
                "RetryableError": _RE,
                "RingRiftError": _RRE,
                "retry": _r,
                "retry_async": _ra,
                "with_emergency_halt_check": _wehc,
            }
        return _lazy_cache["error_handler"][name]

    # Marshalling (lightweight but lazy for consistency)
    if name in ("Codec", "Serializable", "SerializationError",
                "deserialize", "from_json", "register_codec", "serialize", "to_json"):
        if "marshalling" not in _lazy_cache:
            from app.core.marshalling import (
                Codec as _C,
                Serializable as _S,
                SerializationError as _SE,
                deserialize as _d,
                from_json as _fj,
                register_codec as _rc,
                serialize as _s,
                to_json as _tj,
            )
            _lazy_cache["marshalling"] = {
                "Codec": _C,
                "Serializable": _S,
                "SerializationError": _SE,
                "deserialize": _d,
                "from_json": _fj,
                "register_codec": _rc,
                "serialize": _s,
                "to_json": _tj,
            }
        return _lazy_cache["marshalling"][name]

    # Node dataclass (lightweight but lazy for consistency)
    if name in ("ConnectionInfo", "GPUInfo", "HealthStatus", "JobStatus",
                "NodeHealth", "NodeInfo", "NodeRole", "NodeState",
                "Provider", "ProviderInfo", "ResourceMetrics"):
        if "node" not in _lazy_cache:
            from app.core.node import (
                ConnectionInfo as _CI,
                GPUInfo as _GI,
                HealthStatus as _HS,
                JobStatus as _JS,
                NodeHealth as _NH,
                NodeInfo as _NI,
                NodeRole as _NR,
                NodeState as _NS,
                Provider as _P,
                ProviderInfo as _PI,
                ResourceMetrics as _RM,
            )
            _lazy_cache["node"] = {
                "ConnectionInfo": _CI,
                "GPUInfo": _GI,
                "HealthStatus": _HS,
                "JobStatus": _JS,
                "NodeHealth": _NH,
                "NodeInfo": _NI,
                "NodeRole": _NR,
                "NodeState": _NS,
                "Provider": _P,
                "ProviderInfo": _PI,
                "ResourceMetrics": _RM,
            }
        return _lazy_cache["node"][name]

    # SSH utilities (lightweight but lazy for consistency)
    if name in ("SSHClient", "SSHConfig", "SSHResult",
                "get_ssh_client", "run_ssh_command",
                "run_ssh_command_async", "run_ssh_command_sync"):
        if "ssh" not in _lazy_cache:
            from app.core.ssh import (
                SSHClient as _SC,
                SSHConfig as _SCfg,
                SSHResult as _SR,
                get_ssh_client as _gsc,
                run_ssh_command as _rsc,
                run_ssh_command_async as _rsca,
                run_ssh_command_sync as _rscs,
            )
            _lazy_cache["ssh"] = {
                "SSHClient": _SC,
                "SSHConfig": _SCfg,
                "SSHResult": _SR,
                "get_ssh_client": _gsc,
                "run_ssh_command": _rsc,
                "run_ssh_command_async": _rsca,
                "run_ssh_command_sync": _rscs,
            }
        return _lazy_cache["ssh"][name]

    raise AttributeError(f"module 'app.core' has no attribute {name!r}")
