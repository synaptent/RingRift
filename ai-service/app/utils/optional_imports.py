"""Optional imports with availability flags.

This module centralizes try/except import patterns for optional dependencies,
providing a clean API for checking availability and accessing imported modules.

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
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

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

def _try_import(module_name: str, package: str | None = None) -> tuple[Any, bool]:
    """Try to import a module, returning (module, available) tuple.

    Args:
        module_name: Name of the module to import.
        package: Optional package for relative imports.

    Returns:
        Tuple of (module or None, is_available).
    """
    try:
        module = importlib.import_module(module_name, package)
        return module, True
    except ImportError:
        return None, False


def get_module(module_name: str) -> Any | None:
    """Get a module if available, None otherwise.

    Args:
        module_name: Name of the module to import.

    Returns:
        The module if available, None otherwise.
    """
    module, _ = _try_import(module_name)
    return module


def require_module(module_name: str) -> Any:
    """Get a module, raising ImportError if not available.

    Args:
        module_name: Name of the module to import.

    Returns:
        The module.

    Raises:
        ImportError: If the module is not available.
    """
    module, available = _try_import(module_name)
    if not available:
        raise ImportError(f"Required module '{module_name}' is not available")
    return module


# =============================================================================
# NumPy
# =============================================================================

numpy, NUMPY_AVAILABLE = _try_import("numpy")
np = numpy  # Common alias


# =============================================================================
# PyTorch
# =============================================================================

torch, TORCH_AVAILABLE = _try_import("torch")

# Check for specific PyTorch features
CUDA_AVAILABLE = False
MPS_AVAILABLE = False

if TORCH_AVAILABLE:
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# =============================================================================
# Async libraries
# =============================================================================

aiohttp, AIOHTTP_AVAILABLE = _try_import("aiohttp")
asyncio, ASYNCIO_AVAILABLE = _try_import("asyncio")

# websockets
websockets, WEBSOCKETS_AVAILABLE = _try_import("websockets")

# httpx (async HTTP client)
httpx, HTTPX_AVAILABLE = _try_import("httpx")


# =============================================================================
# Prometheus / Metrics
# =============================================================================

prometheus_client, PROMETHEUS_AVAILABLE = _try_import("prometheus_client")

# Convenience access to common prometheus_client components
if PROMETHEUS_AVAILABLE:
    Counter = prometheus_client.Counter
    Gauge = prometheus_client.Gauge
    Histogram = prometheus_client.Histogram
    generate_latest = prometheus_client.generate_latest
    CONTENT_TYPE_LATEST = prometheus_client.CONTENT_TYPE_LATEST
else:
    Counter = None
    Gauge = None
    Histogram = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None


# =============================================================================
# Data processing
# =============================================================================

pandas, PANDAS_AVAILABLE = _try_import("pandas")
pd = pandas  # Common alias

scipy, SCIPY_AVAILABLE = _try_import("scipy")

h5py, H5PY_AVAILABLE = _try_import("h5py")


# =============================================================================
# Networking / SSH
# =============================================================================

paramiko, PARAMIKO_AVAILABLE = _try_import("paramiko")
fabric, FABRIC_AVAILABLE = _try_import("fabric")


# =============================================================================
# Configuration / YAML
# =============================================================================

yaml, YAML_AVAILABLE = _try_import("yaml")

# Note: yaml_utils provides higher-level YAML handling with error handling


# =============================================================================
# Rich console output
# =============================================================================

rich, RICH_AVAILABLE = _try_import("rich")

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.progress import Progress
    from rich.table import Table
    rich_console = Console()
else:
    Console = None
    Table = None
    Progress = None
    rich_console = None


# =============================================================================
# Plotting
# =============================================================================

matplotlib, MATPLOTLIB_AVAILABLE = _try_import("matplotlib")

if MATPLOTLIB_AVAILABLE:
    try:
        from matplotlib import pyplot as pyplot
        plt = pyplot  # Common alias
    except Exception:
        MATPLOTLIB_AVAILABLE = False
        pyplot = None
        plt = None
else:
    pyplot = None
    plt = None


# =============================================================================
# OpenTelemetry / Tracing
# =============================================================================

opentelemetry, OPENTELEMETRY_AVAILABLE = _try_import("opentelemetry")

if OPENTELEMETRY_AVAILABLE:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
else:
    otel_trace = None
    TracerProvider = None


# =============================================================================
# Summary function
# =============================================================================

def get_availability_summary() -> dict[str, bool]:
    """Get a summary of all optional import availability.

    Returns:
        Dict mapping module names to availability status.
    """
    return {
        "numpy": NUMPY_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "cuda": CUDA_AVAILABLE,
        "mps": MPS_AVAILABLE,
        "aiohttp": AIOHTTP_AVAILABLE,
        "asyncio": ASYNCIO_AVAILABLE,
        "websockets": WEBSOCKETS_AVAILABLE,
        "httpx": HTTPX_AVAILABLE,
        "prometheus_client": PROMETHEUS_AVAILABLE,
        "pandas": PANDAS_AVAILABLE,
        "scipy": SCIPY_AVAILABLE,
        "h5py": H5PY_AVAILABLE,
        "paramiko": PARAMIKO_AVAILABLE,
        "fabric": FABRIC_AVAILABLE,
        "yaml": YAML_AVAILABLE,
        "rich": RICH_AVAILABLE,
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "opentelemetry": OPENTELEMETRY_AVAILABLE,
    }


def log_availability() -> None:
    """Log the availability of all optional imports."""
    summary = get_availability_summary()
    available = [k for k, v in summary.items() if v]
    missing = [k for k, v in summary.items() if not v]

    logger.info(f"Optional imports available: {', '.join(available) or 'none'}")
    if missing:
        logger.debug(f"Optional imports not available: {', '.join(missing)}")
