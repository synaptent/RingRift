"""Utility modules for RingRift AI.

This package provides reusable utilities harvested from archived debug scripts
and consolidated for maintainability.

Modules:
    debug_utils: State comparison and parity debugging utilities
    torch_utils: Safe PyTorch operations including device detection (canonical)
    numpy_utils: Safe NumPy operations including secure NPZ loading (Dec 2025)
    env_config: Typed environment variable access
    game_discovery: Unified game database discovery across all storage patterns

Device Management (Canonical Exports):
    get_device: Auto-detect best compute device (CUDA/MPS/CPU)
    get_device_info: Get detailed device information

Environment Configuration (Canonical Exports):
    env: Singleton EnvConfig instance for typed env var access
    get_str, get_int, get_float, get_bool, get_list: Direct env var getters

Game Discovery (Canonical Exports):
    GameDiscovery: Find all game databases across cluster storage patterns
    find_all_game_databases: Quick function to find all databases
    count_games_for_config: Count games for a board/player configuration
    get_game_counts_summary: Get summary of all game counts

Note (Dec 2025): This module uses lazy imports to avoid loading torch at package import time.
Use direct imports like `from app.utils.torch_utils import X` for heavy dependencies.
"""

from __future__ import annotations

__all__ = [
    "debug_utils",
    "env_config",
    "numpy_utils",
    "torch_utils",
    "game_discovery",
    "retry",
    # Safe NPZ loading (numpy_utils)
    "safe_load_npz",
    # Device management (lazy)
    "get_device",
    "get_device_info",
    # Environment configuration (always available)
    "EnvConfig",
    "env",
    "get_bool",
    "get_float",
    "get_int",
    "get_list",
    "get_str",
    # Game discovery (lazy)
    "GameDiscovery",
    "find_all_game_databases",
    "count_games_for_config",
    "get_game_counts_summary",
    # Retry utilities (Dec 2025)
    "retry",
    "retry_async",
    "RetryConfig",
    "RETRY_STANDARD",
    "RETRY_SSH",
    "RETRY_HTTP",
]

# =============================================================================
# Fast imports (no heavy dependencies)
# =============================================================================

# Environment configuration - no torch dependency
from app.utils.env_config import (
    EnvConfig,
    env,
    get_bool,
    get_float,
    get_int,
    get_list,
    get_str,
)

# =============================================================================
# Lazy imports for torch-dependent modules
# =============================================================================

_lazy_cache: dict = {}


def __getattr__(name: str):
    """Lazy import for torch-dependent utilities."""

    # Safe NPZ loading (numpy_utils - loads numpy)
    if name == "safe_load_npz":
        if "numpy_utils" not in _lazy_cache:
            from app.utils.numpy_utils import safe_load_npz as _slnpz
            _lazy_cache["numpy_utils"] = {"safe_load_npz": _slnpz}
        return _lazy_cache["numpy_utils"]["safe_load_npz"]

    # Device management (torch_utils - loads torch)
    if name in ("get_device", "get_device_info"):
        if "torch_utils" not in _lazy_cache:
            from app.utils.torch_utils import (
                get_device as _gd,
                get_device_info as _gdi,
            )
            _lazy_cache["torch_utils"] = {
                "get_device": _gd,
                "get_device_info": _gdi,
            }
        return _lazy_cache["torch_utils"][name]

    # Game discovery (may have heavy deps)
    if name in ("GameDiscovery", "find_all_game_databases",
                "count_games_for_config", "get_game_counts_summary"):
        if "game_discovery" not in _lazy_cache:
            from app.utils.game_discovery import (
                GameDiscovery as _GD,
                count_games_for_config as _cgfc,
                find_all_game_databases as _fagd,
                get_game_counts_summary as _ggcs,
            )
            _lazy_cache["game_discovery"] = {
                "GameDiscovery": _GD,
                "find_all_game_databases": _fagd,
                "count_games_for_config": _cgfc,
                "get_game_counts_summary": _ggcs,
            }
        return _lazy_cache["game_discovery"][name]

    # Retry utilities (Dec 2025)
    if name in ("retry", "retry_async", "RetryConfig", "RETRY_STANDARD",
                "RETRY_SSH", "RETRY_HTTP"):
        if "retry" not in _lazy_cache:
            from app.utils.retry import (
                retry as _retry,
                retry_async as _retry_async,
                RetryConfig as _RetryConfig,
                RETRY_STANDARD as _RETRY_STANDARD,
                RETRY_SSH as _RETRY_SSH,
                RETRY_HTTP as _RETRY_HTTP,
            )
            _lazy_cache["retry"] = {
                "retry": _retry,
                "retry_async": _retry_async,
                "RetryConfig": _RetryConfig,
                "RETRY_STANDARD": _RETRY_STANDARD,
                "RETRY_SSH": _RETRY_SSH,
                "RETRY_HTTP": _RETRY_HTTP,
            }
        return _lazy_cache["retry"][name]

    raise AttributeError(f"module 'app.utils' has no attribute {name!r}")
