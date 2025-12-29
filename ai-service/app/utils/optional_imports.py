"""Optional imports with availability flags - LAZY LOADING VERSION.

This module centralizes try/except import patterns for optional dependencies,
providing a clean API for checking availability and accessing imported modules.

IMPORTANT (December 2025): All imports are LAZY to minimize startup time.
Heavy libraries like torch, numpy, scipy, etc. are only loaded when first
accessed. This reduces the import time of this module from ~500ms to ~10ms.

Usage:
    from app.utils.optional_imports import (
        numpy, NUMPY_AVAILABLE,
        torch, TORCH_AVAILABLE,
        aiohttp, AIOHTTP_AVAILABLE,
    )

    if NUMPY_AVAILABLE:
        arr = numpy.array([1, 2, 3])

    # Or use the get_module helper
    from app.utils.optional_imports import get_module

    np = get_module("numpy")
    if np is not None:
        arr = np.array([1, 2, 3])

Performance Notes:
    - Module import time: ~10ms (was ~500ms before lazy loading)
    - First access to torch: ~300ms (deferred from import time)
    - First access to numpy: ~100ms (deferred from import time)
    - Set RINGRIFT_SKIP_OPTIONAL_IMPORTS=1 to skip all optional imports
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Cache for lazy-loaded modules and availability flags
_module_cache: dict[str, Any] = {}
_availability_cache: dict[str, bool] = {}

__all__ = [
    "AIOHTTP_AVAILABLE",
    "ASYNCIO_AVAILABLE",
    "CUDA_AVAILABLE",
    "FABRIC_AVAILABLE",
    "H5PY_AVAILABLE",
    "HTTPX_AVAILABLE",
    "MATPLOTLIB_AVAILABLE",
    "MPS_AVAILABLE",
    # Availability flags
    "NUMPY_AVAILABLE",
    "OPENTELEMETRY_AVAILABLE",
    "PANDAS_AVAILABLE",
    "PARAMIKO_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "RICH_AVAILABLE",
    "SCIPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "WEBSOCKETS_AVAILABLE",
    "YAML_AVAILABLE",
    "aiohttp",
    "fabric",
    "get_availability_summary",
    # Helper functions
    "get_module",
    "h5py",
    "httpx",
    "log_availability",
    "matplotlib",
    "np",
    # Module aliases
    "numpy",
    "pandas",
    "paramiko",
    "pd",
    "prometheus_client",
    "require_module",
    "rich",
    "scipy",
    "torch",
    "websockets",
    "yaml",
]


# =============================================================================
# Helper functions
# =============================================================================


def _should_skip_imports() -> bool:
    """Check if optional imports should be skipped."""
    skip_optional = os.getenv("RINGRIFT_SKIP_OPTIONAL_IMPORTS", "").strip().lower()
    return skip_optional in ("1", "true", "yes", "on")


def _try_import(module_name: str, package: str | None = None) -> tuple[Any, bool]:
    """Try to import a module, returning (module, available) tuple.

    Args:
        module_name: Name of the module to import.
        package: Optional package for relative imports.

    Returns:
        Tuple of (module or None, is_available).
    """
    if _should_skip_imports():
        logger.debug("Skipping optional import %s (RINGRIFT_SKIP_OPTIONAL_IMPORTS)", module_name)
        return None, False
    try:
        module = importlib.import_module(module_name, package)
        return module, True
    except ImportError:
        return None, False
    except Exception as exc:
        logger.debug("Optional import failed for %s: %s", module_name, exc)
        return None, False


def _lazy_import(module_name: str) -> Any | None:
    """Lazily import a module, caching the result.

    Args:
        module_name: Name of the module to import.

    Returns:
        The module if available, None otherwise.
    """
    if module_name in _module_cache:
        return _module_cache[module_name]

    module, available = _try_import(module_name)
    _module_cache[module_name] = module
    _availability_cache[module_name] = available
    return module


def _check_available(module_name: str) -> bool:
    """Check if a module is available (triggers lazy import if needed).

    Args:
        module_name: Name of the module to check.

    Returns:
        True if the module is available.
    """
    if module_name in _availability_cache:
        return _availability_cache[module_name]

    # Trigger lazy import to determine availability
    _lazy_import(module_name)
    return _availability_cache.get(module_name, False)


def get_module(module_name: str) -> Any | None:
    """Get a module if available, None otherwise.

    Args:
        module_name: Name of the module to import.

    Returns:
        The module if available, None otherwise.
    """
    return _lazy_import(module_name)


def require_module(module_name: str) -> Any:
    """Get a module, raising ImportError if not available.

    Args:
        module_name: Name of the module to import.

    Returns:
        The module.

    Raises:
        ImportError: If the module is not available.
    """
    module = _lazy_import(module_name)
    if module is None:
        raise ImportError(f"Required module '{module_name}' is not available")
    return module


# =============================================================================
# Module-level __getattr__ for lazy loading
# =============================================================================

# Mapping from attribute name to module name
_MODULE_MAPPING = {
    "numpy": "numpy",
    "np": "numpy",
    "torch": "torch",
    "aiohttp": "aiohttp",
    "asyncio": "asyncio",
    "websockets": "websockets",
    "httpx": "httpx",
    "prometheus_client": "prometheus_client",
    "pandas": "pandas",
    "pd": "pandas",
    "scipy": "scipy",
    "h5py": "h5py",
    "paramiko": "paramiko",
    "fabric": "fabric",
    "yaml": "yaml",
    "rich": "rich",
    "matplotlib": "matplotlib",
    "opentelemetry": "opentelemetry",
}

# Mapping from availability flag to module name
_AVAILABILITY_MAPPING = {
    "NUMPY_AVAILABLE": "numpy",
    "TORCH_AVAILABLE": "torch",
    "CUDA_AVAILABLE": "_cuda",  # Special handling
    "MPS_AVAILABLE": "_mps",    # Special handling
    "AIOHTTP_AVAILABLE": "aiohttp",
    "ASYNCIO_AVAILABLE": "asyncio",
    "WEBSOCKETS_AVAILABLE": "websockets",
    "HTTPX_AVAILABLE": "httpx",
    "PROMETHEUS_AVAILABLE": "prometheus_client",
    "PANDAS_AVAILABLE": "pandas",
    "SCIPY_AVAILABLE": "scipy",
    "H5PY_AVAILABLE": "h5py",
    "PARAMIKO_AVAILABLE": "paramiko",
    "FABRIC_AVAILABLE": "fabric",
    "YAML_AVAILABLE": "yaml",
    "RICH_AVAILABLE": "rich",
    "MATPLOTLIB_AVAILABLE": "matplotlib",
    "OPENTELEMETRY_AVAILABLE": "opentelemetry",
}


def _get_cuda_available() -> bool:
    """Check if CUDA is available (requires torch)."""
    torch_mod = _lazy_import("torch")
    if torch_mod is None:
        return False
    return torch_mod.cuda.is_available()


def _get_mps_available() -> bool:
    """Check if MPS is available (requires torch)."""
    torch_mod = _lazy_import("torch")
    if torch_mod is None:
        return False
    return hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available()


def _get_prometheus_component(name: str) -> Any | None:
    """Get a prometheus_client component lazily."""
    pc = _lazy_import("prometheus_client")
    if pc is None:
        return None
    return getattr(pc, name, None)


def _get_rich_component(name: str) -> Any | None:
    """Get a rich component lazily."""
    rich_mod = _lazy_import("rich")
    if rich_mod is None:
        return None
    if name == "Console":
        from rich.console import Console
        return Console
    elif name == "Table":
        from rich.table import Table
        return Table
    elif name == "Progress":
        from rich.progress import Progress
        return Progress
    elif name == "rich_console":
        from rich.console import Console
        return Console()
    return None


def _get_matplotlib_pyplot() -> Any | None:
    """Get matplotlib.pyplot lazily."""
    mpl = _lazy_import("matplotlib")
    if mpl is None:
        return None
    try:
        from matplotlib import pyplot
        return pyplot
    except ImportError:
        return None


def _get_opentelemetry_component(name: str) -> Any | None:
    """Get an opentelemetry component lazily."""
    otel = _lazy_import("opentelemetry")
    if otel is None:
        return None
    if name == "otel_trace":
        from opentelemetry import trace
        return trace
    elif name == "TracerProvider":
        from opentelemetry.sdk.trace import TracerProvider
        return TracerProvider
    return None


def __getattr__(name: str) -> Any:
    """Lazy loader for module-level attributes.

    This implements PEP 562 module __getattr__ to provide lazy loading
    of heavy optional dependencies like torch, numpy, etc.
    """
    # Handle module imports (numpy, torch, etc.)
    if name in _MODULE_MAPPING:
        return _lazy_import(_MODULE_MAPPING[name])

    # Handle availability flags (NUMPY_AVAILABLE, etc.)
    if name in _AVAILABILITY_MAPPING:
        mapped = _AVAILABILITY_MAPPING[name]
        if mapped == "_cuda":
            return _get_cuda_available()
        elif mapped == "_mps":
            return _get_mps_available()
        return _check_available(mapped)

    # Handle prometheus_client components
    if name in ("Counter", "Gauge", "Histogram", "generate_latest", "CONTENT_TYPE_LATEST"):
        return _get_prometheus_component(name)

    # Handle rich components
    if name in ("Console", "Table", "Progress", "rich_console"):
        return _get_rich_component(name)

    # Handle matplotlib pyplot
    if name in ("pyplot", "plt"):
        return _get_matplotlib_pyplot()

    # Handle opentelemetry components
    if name in ("otel_trace", "TracerProvider"):
        return _get_opentelemetry_component(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# Summary function
# =============================================================================

def get_availability_summary() -> dict[str, bool]:
    """Get a summary of all optional import availability.

    Note: This triggers lazy loading for each module to check availability.

    Returns:
        Dict mapping module names to availability status.
    """
    return {
        "numpy": _check_available("numpy"),
        "torch": _check_available("torch"),
        "cuda": _get_cuda_available(),
        "mps": _get_mps_available(),
        "aiohttp": _check_available("aiohttp"),
        "asyncio": _check_available("asyncio"),
        "websockets": _check_available("websockets"),
        "httpx": _check_available("httpx"),
        "prometheus_client": _check_available("prometheus_client"),
        "pandas": _check_available("pandas"),
        "scipy": _check_available("scipy"),
        "h5py": _check_available("h5py"),
        "paramiko": _check_available("paramiko"),
        "fabric": _check_available("fabric"),
        "yaml": _check_available("yaml"),
        "rich": _check_available("rich"),
        "matplotlib": _check_available("matplotlib"),
        "opentelemetry": _check_available("opentelemetry"),
    }


def log_availability() -> None:
    """Log the availability of all optional imports.

    Note: This triggers lazy loading for each module.
    """
    summary = get_availability_summary()
    available = [k for k, v in summary.items() if v]
    missing = [k for k, v in summary.items() if not v]

    logger.info(f"Optional imports available: {', '.join(available) or 'none'}")
    if missing:
        logger.debug(f"Optional imports not available: {', '.join(missing)}")
