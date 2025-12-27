"""Consolidated Coordination Utilities (December 2025).

This module consolidates utility functions used across the coordination layer:
- Distributed tracing (trace_id propagation, spans, collectors)
- Distributed locking (Redis + file-based, with fallback)
- Re-exports from app/utils for convenience (optional_imports, yaml_utils)

This is part of the 157→15 module consolidation (Phase 5).

Migration Guide:
    # Old imports (deprecated, still work):
    from app.coordination.tracing import TraceContext, new_trace, span
    from app.coordination.distributed_lock import DistributedLock, training_lock

    # New imports (preferred):
    from app.coordination.core_utils import (
        # Tracing
        TraceContext, TraceSpan, TraceCollector,
        new_trace, with_trace, span, traced,
        get_trace_id, set_trace_id, get_trace_context,

        # Locking
        DistributedLock, LockProtocol,
        training_lock, acquire_training_lock, cleanup_stale_locks,

        # Optional imports (from app/utils)
        get_module, require_module,
        TORCH_AVAILABLE, CUDA_AVAILABLE,

        # YAML utilities (from app/utils)
        load_yaml, safe_load_yaml, dump_yaml,
    )
"""

from __future__ import annotations

# =============================================================================
# Re-exports from tracing.py
# =============================================================================

from app.coordination.tracing import (
    TraceCollector,
    TraceContext,
    # Classes
    TraceSpan,
    collect_trace,
    extract_trace_from_event,
    extract_trace_from_headers,
    generate_span_id,
    # Functions
    generate_trace_id,
    get_trace_collector,
    get_trace_context,
    get_trace_id,
    inject_trace_into_event,
    inject_trace_into_headers,
    new_trace,
    set_trace_id,
    span,
    traced,
    with_trace,
)

# =============================================================================
# Re-exports from distributed_lock.py
# =============================================================================

from app.coordination.distributed_lock import (
    DEFAULT_ACQUIRE_TIMEOUT,
    # Constants
    DEFAULT_LOCK_TIMEOUT,
    # Main class
    DistributedLock,
    # Protocol
    LockProtocol,
    acquire_training_lock,
    # Functions
    cleanup_stale_locks,
    get_appropriate_lock,
    release_training_lock,
    training_lock,
)

# =============================================================================
# Re-exports from app/utils/optional_imports.py
# =============================================================================

from app.utils.optional_imports import (
    AIOHTTP_AVAILABLE,
    CUDA_AVAILABLE,
    MATPLOTLIB_AVAILABLE,
    MPS_AVAILABLE,
    # Availability flags
    NUMPY_AVAILABLE,
    OPENTELEMETRY_AVAILABLE,
    PANDAS_AVAILABLE,
    PROMETHEUS_AVAILABLE,
    RICH_AVAILABLE,
    TORCH_AVAILABLE,
    YAML_AVAILABLE,
    aiohttp,
    # Helper functions
    get_availability_summary,
    get_module,
    # Module aliases
    numpy,
    pandas,
    pd,
    require_module,
    rich,
    torch,
    yaml,
)

# =============================================================================
# Re-exports from app/utils/yaml_utils.py
# =============================================================================

from app.utils.yaml_utils import (
    ConfigDict,
    YAMLLoadError,
    dump_yaml,
    dumps_yaml,
    load_config_yaml,
    load_yaml,
    load_yaml_with_defaults,
    safe_load_yaml,
    validate_yaml_schema,
)

# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Tracing - Classes
    "TraceContext",
    "TraceSpan",
    "TraceCollector",
    # Tracing - Functions
    "generate_trace_id",
    "generate_span_id",
    "get_trace_id",
    "get_trace_context",
    "set_trace_id",
    "new_trace",
    "with_trace",
    "span",
    "traced",
    "inject_trace_into_event",
    "extract_trace_from_event",
    "inject_trace_into_headers",
    "extract_trace_from_headers",
    "get_trace_collector",
    "collect_trace",
    # Locking - Classes
    "DistributedLock",
    "LockProtocol",
    # Locking - Constants
    "DEFAULT_LOCK_TIMEOUT",
    "DEFAULT_ACQUIRE_TIMEOUT",
    # Locking - Functions
    "acquire_training_lock",
    "release_training_lock",
    "training_lock",
    "cleanup_stale_locks",
    "get_appropriate_lock",
    # Optional imports - Availability flags
    "NUMPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "AIOHTTP_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "PANDAS_AVAILABLE",
    "YAML_AVAILABLE",
    "RICH_AVAILABLE",
    "MATPLOTLIB_AVAILABLE",
    "OPENTELEMETRY_AVAILABLE",
    # Optional imports - Modules
    "numpy",
    "torch",
    "aiohttp",
    "pandas",
    "pd",
    "yaml",
    "rich",
    # Optional imports - Functions
    "get_module",
    "require_module",
    "get_availability_summary",
    # YAML utilities
    "load_yaml",
    "load_yaml_with_defaults",
    "safe_load_yaml",
    "dump_yaml",
    "dumps_yaml",
    "load_config_yaml",
    "validate_yaml_schema",
    "ConfigDict",
    "YAMLLoadError",
]
