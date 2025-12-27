"""Unit tests for core_utils.py - consolidated utilities module.

Tests the re-exports and functionality of:
- Distributed tracing (trace_id, spans, decorators)
- Distributed locking (DistributedLock, training_lock)
- Optional imports (availability flags, get_module, require_module)
- YAML utilities (load_yaml, dump_yaml)

December 2025: Created for Phase 5 consolidation testing.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test re-export availability
# =============================================================================


class TestCoreUtilsExports:
    """Test that all expected exports are accessible."""

    def test_tracing_classes_exported(self):
        """Verify tracing classes are exported."""
        from app.coordination.core_utils import (
            TraceCollector,
            TraceContext,
            TraceSpan,
        )

        assert TraceContext is not None
        assert TraceSpan is not None
        assert TraceCollector is not None

    def test_tracing_functions_exported(self):
        """Verify tracing functions are exported."""
        from app.coordination.core_utils import (
            generate_span_id,
            generate_trace_id,
            get_trace_context,
            get_trace_id,
            new_trace,
            set_trace_id,
            span,
            traced,
            with_trace,
        )

        assert callable(generate_trace_id)
        assert callable(generate_span_id)
        assert callable(get_trace_id)
        assert callable(set_trace_id)
        assert callable(new_trace)
        assert callable(with_trace)
        assert callable(span)
        assert callable(traced)
        assert callable(get_trace_context)

    def test_locking_classes_exported(self):
        """Verify locking classes are exported."""
        from app.coordination.core_utils import (
            DistributedLock,
            LockProtocol,
        )

        assert DistributedLock is not None
        assert LockProtocol is not None

    def test_locking_functions_exported(self):
        """Verify locking functions are exported."""
        from app.coordination.core_utils import (
            acquire_training_lock,
            cleanup_stale_locks,
            get_appropriate_lock,
            release_training_lock,
            training_lock,
        )

        assert callable(acquire_training_lock)
        assert callable(release_training_lock)
        assert callable(training_lock)
        assert callable(cleanup_stale_locks)
        assert callable(get_appropriate_lock)

    def test_locking_constants_exported(self):
        """Verify locking constants are exported."""
        from app.coordination.core_utils import (
            DEFAULT_ACQUIRE_TIMEOUT,
            DEFAULT_LOCK_TIMEOUT,
        )

        assert isinstance(DEFAULT_LOCK_TIMEOUT, (int, float))
        assert isinstance(DEFAULT_ACQUIRE_TIMEOUT, (int, float))
        assert DEFAULT_LOCK_TIMEOUT > 0
        assert DEFAULT_ACQUIRE_TIMEOUT > 0

    def test_optional_imports_flags_exported(self):
        """Verify optional import availability flags are exported."""
        from app.coordination.core_utils import (
            AIOHTTP_AVAILABLE,
            CUDA_AVAILABLE,
            MATPLOTLIB_AVAILABLE,
            MPS_AVAILABLE,
            NUMPY_AVAILABLE,
            OPENTELEMETRY_AVAILABLE,
            PANDAS_AVAILABLE,
            PROMETHEUS_AVAILABLE,
            RICH_AVAILABLE,
            TORCH_AVAILABLE,
            YAML_AVAILABLE,
        )

        # All should be boolean
        assert isinstance(NUMPY_AVAILABLE, bool)
        assert isinstance(TORCH_AVAILABLE, bool)
        assert isinstance(CUDA_AVAILABLE, bool)
        assert isinstance(MPS_AVAILABLE, bool)
        assert isinstance(AIOHTTP_AVAILABLE, bool)
        assert isinstance(PROMETHEUS_AVAILABLE, bool)
        assert isinstance(PANDAS_AVAILABLE, bool)
        assert isinstance(YAML_AVAILABLE, bool)
        assert isinstance(RICH_AVAILABLE, bool)
        assert isinstance(MATPLOTLIB_AVAILABLE, bool)
        assert isinstance(OPENTELEMETRY_AVAILABLE, bool)

    def test_optional_imports_functions_exported(self):
        """Verify optional import helper functions are exported."""
        from app.coordination.core_utils import (
            get_availability_summary,
            get_module,
            require_module,
        )

        assert callable(get_module)
        assert callable(require_module)
        assert callable(get_availability_summary)

    def test_yaml_utilities_exported(self):
        """Verify YAML utility functions are exported."""
        from app.coordination.core_utils import (
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

        assert callable(load_yaml)
        assert callable(safe_load_yaml)
        assert callable(dump_yaml)
        assert callable(dumps_yaml)
        assert callable(load_config_yaml)
        assert callable(load_yaml_with_defaults)
        assert callable(validate_yaml_schema)
        assert ConfigDict is not None
        assert YAMLLoadError is not None


# =============================================================================
# Test tracing functionality
# =============================================================================


class TestTracingFunctionality:
    """Test tracing functions work correctly."""

    def test_generate_trace_id_format(self):
        """Verify trace_id has expected format: trace-{16 hex chars}."""
        from app.coordination.core_utils import generate_trace_id

        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)
        assert trace_id.startswith("trace-")
        assert len(trace_id) == 22  # "trace-" (6) + 16 hex chars

    def test_generate_span_id_format(self):
        """Verify span_id has expected format: span-{8 hex chars}."""
        from app.coordination.core_utils import generate_span_id

        span_id = generate_span_id()
        assert isinstance(span_id, str)
        assert span_id.startswith("span-")
        assert len(span_id) == 13  # "span-" (5) + 8 hex chars

    def test_trace_context_creation(self):
        """Verify TraceContext can be created."""
        from app.coordination.core_utils import TraceContext, generate_trace_id

        trace_id = generate_trace_id()
        ctx = TraceContext(trace_id=trace_id, name="test")
        assert ctx.trace_id == trace_id
        assert ctx.name == "test"

    def test_new_trace_creates_context_manager(self):
        """Verify new_trace returns a context manager."""
        from app.coordination.core_utils import new_trace

        # new_trace returns a context manager, not a TraceContext directly
        with new_trace("test_trace") as ctx:
            assert ctx is not None
            assert ctx.trace_id is not None
            assert ctx.trace_id.startswith("trace-")

    def test_span_context_manager(self):
        """Verify span works as context manager."""
        from app.coordination.core_utils import new_trace, span

        # span() takes name and optional tags, no context argument
        with new_trace("test_trace"):
            with span("test_operation", tag1="value1") as s:
                # May return None if tracing is disabled
                pass  # Just verify no exception

    def test_traced_decorator(self):
        """Verify traced decorator works on functions."""
        from app.coordination.core_utils import traced

        @traced("test_func")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10


# =============================================================================
# Test locking functionality
# =============================================================================


class TestLockingFunctionality:
    """Test locking functions work correctly."""

    def test_distributed_lock_creation(self):
        """Verify DistributedLock can be instantiated."""
        from app.coordination.core_utils import DistributedLock

        # DistributedLock takes 'name' not 'lock_name'
        lock = DistributedLock(
            name="test_lock",
            lock_timeout=60,
            use_redis=False,  # Force file-based for testing
        )
        assert lock is not None
        assert lock.name == "test_lock"

    def test_distributed_lock_file_based(self):
        """Verify file-based locking works."""
        from app.coordination.core_utils import DistributedLock

        lock = DistributedLock(
            name="test_file_lock",
            lock_timeout=5,
            use_redis=False,  # Force file-based
        )

        # Acquire lock
        acquired = lock.acquire(timeout=5)
        assert acquired
        assert lock._acquired

        # Release lock
        lock.release()
        assert not lock._acquired

    def test_distributed_lock_context_manager(self):
        """Verify DistributedLock works as context manager."""
        from app.coordination.core_utils import DistributedLock

        lock = DistributedLock(
            name="test_ctx_lock",
            lock_timeout=5,
            use_redis=False,
        )

        with lock:
            # Lock should be acquired inside context
            assert lock._acquired

        # Lock should be released after context
        assert not lock._acquired

    def test_cleanup_stale_locks(self):
        """Verify cleanup_stale_locks runs without error."""
        from app.coordination.core_utils import cleanup_stale_locks

        # cleanup_stale_locks uses max_age_hours (not seconds)
        # Just verify it runs without raising
        result = cleanup_stale_locks(max_age_hours=24.0)  # Clean locks older than 1 day
        assert isinstance(result, dict)  # Returns stats dict


# =============================================================================
# Test optional imports functionality
# =============================================================================


class TestOptionalImportsFunctionality:
    """Test optional imports helper functions."""

    def test_get_module_returns_module(self):
        """Verify get_module returns a module when available."""
        from app.coordination.core_utils import get_module

        # os is always available
        os_module = get_module("os")
        assert os_module is not None
        assert hasattr(os_module, "path")

    def test_get_module_returns_none_for_missing(self):
        """Verify get_module returns None for missing module."""
        from app.coordination.core_utils import get_module

        fake = get_module("nonexistent_module_xyz123")
        assert fake is None

    def test_require_module_raises_for_missing(self):
        """Verify require_module raises ImportError for missing module."""
        from app.coordination.core_utils import require_module

        with pytest.raises(ImportError):
            require_module("nonexistent_module_xyz123")

    def test_require_module_returns_module(self):
        """Verify require_module returns module when available."""
        from app.coordination.core_utils import require_module

        os_module = require_module("os")
        assert os_module is not None
        assert hasattr(os_module, "path")

    def test_get_availability_summary(self):
        """Verify get_availability_summary returns dict."""
        from app.coordination.core_utils import get_availability_summary

        summary = get_availability_summary()
        assert isinstance(summary, dict)
        assert "numpy" in summary or "torch" in summary


# =============================================================================
# Test YAML utilities functionality
# =============================================================================


class TestYAMLUtilitiesFunctionality:
    """Test YAML utility functions."""

    def test_dump_yaml_produces_string(self):
        """Verify dump_yaml creates YAML string."""
        from app.coordination.core_utils import dumps_yaml

        data = {"key": "value", "number": 42}
        result = dumps_yaml(data)
        assert isinstance(result, str)
        assert "key: value" in result
        assert "42" in result

    def test_load_yaml_reads_file(self):
        """Verify load_yaml reads YAML file."""
        from app.coordination.core_utils import dump_yaml, load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            data = {"test": "data", "items": [1, 2, 3]}
            dump_yaml(data, Path(f.name))
            f.flush()

            result = load_yaml(Path(f.name))
            assert result == data

    def test_safe_load_yaml_returns_none_on_missing(self):
        """Verify safe_load_yaml returns None for missing file."""
        from app.coordination.core_utils import safe_load_yaml

        result = safe_load_yaml(Path("/nonexistent/path/file.yaml"))
        assert result is None

    def test_safe_load_yaml_returns_data(self):
        """Verify safe_load_yaml reads existing file."""
        from app.coordination.core_utils import dump_yaml, safe_load_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            data = {"safe": True}
            dump_yaml(data, Path(f.name))
            f.flush()

            result = safe_load_yaml(Path(f.name))
            assert result == data


# =============================================================================
# Test __all__ completeness
# =============================================================================


class TestAllExports:
    """Verify __all__ is complete and accurate."""

    def test_all_exports_are_accessible(self):
        """Verify every item in __all__ is actually exported."""
        import app.coordination.core_utils as module

        for name in module.__all__:
            assert hasattr(module, name), f"Missing export: {name}"
            obj = getattr(module, name)
            assert obj is not None, f"None value for: {name}"

    def test_expected_export_count(self):
        """Verify we have the expected number of exports."""
        from app.coordination.core_utils import __all__

        # Should have ~60+ exports based on module content
        assert len(__all__) >= 50, f"Only {len(__all__)} exports, expected 50+"
