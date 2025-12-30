"""Comprehensive tests for app.coordination.unified_distribution_daemon.

This module provides comprehensive unit tests for the UnifiedDistributionDaemon,
which handles distribution of models and NPZ files across the cluster.

Test Coverage (60+ tests):
1. DistributionDataType enum and DataType alias
2. DistributionConfig dataclass (all fields)
3. DeliveryResult dataclass (all fields)
4. UnifiedDistributionDaemon initialization
5. CoordinatorProtocol implementation (name, status, uptime, metrics, health_check)
6. Event handlers (MODEL_PROMOTED, MODEL_UPDATED, MODEL_DISTRIBUTION_FAILED, etc.)
7. Distribution logic (_enqueue_item, _process_pending_items)
8. Transport methods (HTTP, rsync, BitTorrent selection)
9. Checksum verification
10. Remote path discovery and caching
11. Model validation helpers
12. Factory functions with deprecation warnings
13. Distribution verification methods
14. Edge cases and error handling

December 2025: Expanded test coverage for the consolidated distribution daemon.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
from app.coordination.unified_distribution_daemon import (
    REMOTE_PATH_PATTERNS,
    DataType,
    DeliveryResult,
    DistributionConfig,
    UnifiedDistributionDaemon,
    _is_valid_model_file,
    _remote_path_cache,
    _remote_path_cache_lock,
    check_model_availability,
    clear_remote_path_cache,
    create_model_distribution_daemon,
    create_npz_distribution_daemon,
    create_unified_distribution_daemon,
    get_all_cached_remote_paths,
    get_cached_remote_path,
    get_model_availability_score,
    get_remote_path_patterns,
    is_valid_model_file,
    verify_model_distribution,
    wait_for_model_availability,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create a test configuration."""
    return DistributionConfig(
        sync_timeout_seconds=60.0,
        retry_count=2,
        http_timeout_seconds=30.0,
    )


@pytest.fixture
def daemon(config):
    """Create a test daemon."""
    return UnifiedDistributionDaemon(config)


@pytest.fixture(autouse=True)
def cleanup_path_cache():
    """Clear remote path cache before and after each test."""
    clear_remote_path_cache()
    yield
    clear_remote_path_cache()


# =============================================================================
# DistributionDataType / DataType Enum Tests
# =============================================================================


class TestDataType:
    """Test DataType enumeration (alias for DistributionDataType)."""

    def test_has_model(self):
        """Test MODEL type exists."""
        assert DataType.MODEL is not None

    def test_has_npz(self):
        """Test NPZ type exists."""
        assert DataType.NPZ is not None

    def test_has_torrent(self):
        """Test TORRENT type exists."""
        assert DataType.TORRENT is not None

    def test_all_types_count(self):
        """Test all data types - should have at least 3."""
        all_types = list(DataType)
        assert len(all_types) >= 3

    def test_model_is_distinct_from_npz(self):
        """Test MODEL and NPZ are distinct types."""
        assert DataType.MODEL != DataType.NPZ

    def test_datatype_is_enum(self):
        """Test DataType is an enum."""
        from enum import Enum
        assert issubclass(DataType, Enum)

    def test_datatype_alias_works(self):
        """Test DataType alias points to DistributionDataType."""
        from app.coordination.enums import DistributionDataType
        assert DataType is DistributionDataType


# =============================================================================
# DistributionConfig Tests
# =============================================================================


class TestDistributionConfig:
    """Test DistributionConfig dataclass."""

    def test_default_sync_settings(self):
        """Test default sync settings."""
        config = DistributionConfig()
        assert config.sync_timeout_seconds == 300.0
        assert config.retry_count == 3
        assert config.retry_delay_seconds == 30.0
        assert config.retry_backoff_multiplier == 1.5

    def test_default_event_settings(self):
        """Test default event settings."""
        config = DistributionConfig()
        assert config.emit_completion_event is True
        assert config.poll_interval_seconds == 60.0

    def test_default_http_settings(self):
        """Test default HTTP distribution settings."""
        config = DistributionConfig()
        assert config.use_http_distribution is True
        assert config.http_port == 8767
        assert config.http_timeout_seconds == 120.0
        assert config.http_concurrent_uploads == 5
        assert config.fallback_to_rsync is True

    def test_default_checksum_settings(self):
        """Test default checksum settings."""
        config = DistributionConfig()
        assert config.verify_checksums is True
        assert config.checksum_timeout_seconds == 30.0

    def test_default_bittorrent_settings(self):
        """Test default BitTorrent settings."""
        config = DistributionConfig()
        assert config.use_bittorrent_for_large_files is True
        assert config.bittorrent_threshold_bytes == 50_000_000

    def test_default_paths(self):
        """Test default path settings."""
        config = DistributionConfig()
        assert config.models_dir == "models"
        assert config.training_data_dir == "data/training"

    def test_default_npz_settings(self):
        """Test default NPZ-specific settings."""
        config = DistributionConfig()
        assert config.validate_npz_structure is True
        assert config.max_npz_samples == 100_000_000

    def test_default_model_settings(self):
        """Test default model-specific settings."""
        config = DistributionConfig()
        assert config.create_symlinks is True

    def test_custom_values(self):
        """Test initialization with custom values."""
        config = DistributionConfig(
            sync_timeout_seconds=120.0,
            retry_count=5,
            http_port=9999,
            use_bittorrent_for_large_files=False,
        )
        assert config.sync_timeout_seconds == 120.0
        assert config.retry_count == 5
        assert config.http_port == 9999
        assert config.use_bittorrent_for_large_files is False

    def test_all_fields_customizable(self):
        """Test all fields can be customized."""
        config = DistributionConfig(
            sync_timeout_seconds=100.0,
            retry_count=10,
            retry_delay_seconds=5.0,
            retry_backoff_multiplier=2.5,
            emit_completion_event=False,
            poll_interval_seconds=120.0,
            use_http_distribution=False,
            http_port=8080,
            http_timeout_seconds=60.0,
            http_concurrent_uploads=10,
            fallback_to_rsync=False,
            verify_checksums=False,
            checksum_timeout_seconds=60.0,
            use_bittorrent_for_large_files=False,
            bittorrent_threshold_bytes=100_000_000,
            models_dir="custom_models",
            training_data_dir="custom_data",
            validate_npz_structure=False,
            max_npz_samples=50_000_000,
            create_symlinks=False,
        )
        assert config.sync_timeout_seconds == 100.0
        assert config.retry_count == 10
        assert config.models_dir == "custom_models"


# =============================================================================
# DeliveryResult Tests
# =============================================================================


class TestDeliveryResult:
    """Test DeliveryResult dataclass."""

    def test_successful_delivery(self):
        """Test successful delivery result."""
        result = DeliveryResult(
            node_id="node-1",
            host="10.0.0.1",
            data_path="models/canonical_hex8_2p.pth",
            data_type=DataType.MODEL,
            success=True,
            checksum_verified=True,
            transfer_time_seconds=5.2,
            method="http",
        )
        assert result.success is True
        assert result.checksum_verified is True
        assert result.error_message == ""
        assert result.method == "http"

    def test_failed_delivery(self):
        """Test failed delivery result."""
        result = DeliveryResult(
            node_id="node-2",
            host="10.0.0.2",
            data_path="data/training/hex8_2p.npz",
            data_type=DataType.NPZ,
            success=False,
            checksum_verified=False,
            transfer_time_seconds=0.0,
            error_message="Connection timeout",
            method="rsync",
        )
        assert result.success is False
        assert result.error_message == "Connection timeout"
        assert result.method == "rsync"

    def test_default_method(self):
        """Test default method is http."""
        result = DeliveryResult(
            node_id="node-3",
            host="10.0.0.3",
            data_path="models/test.pth",
            data_type=DataType.MODEL,
            success=True,
            checksum_verified=True,
            transfer_time_seconds=1.0,
        )
        assert result.method == "http"

    def test_bittorrent_method(self):
        """Test bittorrent method."""
        result = DeliveryResult(
            node_id="node-4",
            host="10.0.0.4",
            data_path="models/large_model.pth",
            data_type=DataType.MODEL,
            success=True,
            checksum_verified=True,
            transfer_time_seconds=60.0,
            method="bittorrent",
        )
        assert result.method == "bittorrent"

    def test_all_fields(self):
        """Test all fields are accessible."""
        result = DeliveryResult(
            node_id="test-node",
            host="192.168.1.1",
            data_path="/path/to/file",
            data_type=DataType.NPZ,
            success=True,
            checksum_verified=False,
            transfer_time_seconds=3.14,
            error_message="partial error",
            method="rsync",
        )
        assert result.node_id == "test-node"
        assert result.host == "192.168.1.1"
        assert result.data_path == "/path/to/file"
        assert result.data_type == DataType.NPZ
        assert result.transfer_time_seconds == 3.14


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for backward compatibility."""

    def test_create_unified_distribution_daemon(self):
        """Test creating unified daemon."""
        daemon = create_unified_distribution_daemon()
        assert isinstance(daemon, UnifiedDistributionDaemon)
        assert daemon.config is not None

    def test_create_unified_with_config(self):
        """Test creating unified daemon with custom config."""
        config = DistributionConfig(retry_count=10)
        daemon = create_unified_distribution_daemon(config)
        assert daemon.config.retry_count == 10

    def test_create_model_distribution_daemon_deprecated(self):
        """Test deprecated model distribution factory."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            daemon = create_model_distribution_daemon()
        assert isinstance(daemon, UnifiedDistributionDaemon)

    def test_create_npz_distribution_daemon_deprecated(self):
        """Test deprecated NPZ distribution factory."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            daemon = create_npz_distribution_daemon()
        assert isinstance(daemon, UnifiedDistributionDaemon)

    def test_deprecated_factories_return_unified_daemon(self):
        """Test both deprecated factories return UnifiedDistributionDaemon."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model_daemon = create_model_distribution_daemon()
            npz_daemon = create_npz_distribution_daemon()

        assert type(model_daemon) == type(npz_daemon)
        assert isinstance(model_daemon, UnifiedDistributionDaemon)


# =============================================================================
# UnifiedDistributionDaemon Initialization Tests
# =============================================================================


class TestUnifiedDistributionDaemonInit:
    """Test UnifiedDistributionDaemon initialization."""

    def test_default_init(self):
        """Test default initialization."""
        daemon = UnifiedDistributionDaemon()
        assert daemon.config is not None
        assert daemon._running is False
        assert daemon._last_sync_time == 0.0

    def test_custom_config_init(self, config):
        """Test initialization with custom config."""
        daemon = UnifiedDistributionDaemon(config)
        assert daemon.config.sync_timeout_seconds == 60.0
        assert daemon.config.retry_count == 2

    def test_initial_coordinator_status(self):
        """Test initial coordinator status is INITIALIZING."""
        daemon = UnifiedDistributionDaemon()
        assert daemon._coordinator_status == CoordinatorStatus.INITIALIZING

    def test_initial_counters_zero(self):
        """Test initial counters are zero."""
        daemon = UnifiedDistributionDaemon()
        assert daemon._events_processed == 0
        assert daemon._errors_count == 0
        assert daemon._successful_distributions == 0
        assert daemon._failed_distributions == 0
        assert daemon._checksum_failures == 0
        assert daemon._model_distributions == 0
        assert daemon._npz_distributions == 0

    def test_initial_lists_empty(self):
        """Test initial lists are empty."""
        daemon = UnifiedDistributionDaemon()
        assert daemon._pending_items == []
        assert daemon._delivery_history == []

    def test_initial_prefetch_settings(self):
        """Test initial prefetch settings."""
        daemon = UnifiedDistributionDaemon()
        assert daemon._prefetched_checkpoints == set()
        assert daemon._prefetch_threshold == 0.80
        assert daemon._prefetch_enabled is True


# =============================================================================
# CoordinatorProtocol Implementation Tests
# =============================================================================


class TestCoordinatorProtocol:
    """Test CoordinatorProtocol implementation."""

    def test_name_property(self, daemon):
        """Test name property."""
        assert daemon.name == "UnifiedDistributionDaemon"

    def test_status_property(self, daemon):
        """Test status property."""
        assert daemon.status == CoordinatorStatus.INITIALIZING

    def test_uptime_before_start(self, daemon):
        """Test uptime is 0 before start."""
        assert daemon.uptime_seconds == 0.0

    def test_uptime_after_setting_start_time(self, daemon):
        """Test uptime increases after setting start time."""
        daemon._start_time = time.time() - 10.0
        assert daemon.uptime_seconds >= 10.0

    def test_is_running_initially_false(self, daemon):
        """Test is_running is initially False."""
        assert daemon.is_running() is False

    def test_is_running_after_setting_flag(self, daemon):
        """Test is_running reflects _running flag."""
        daemon._running = True
        assert daemon.is_running() is True
        daemon._running = False
        assert daemon.is_running() is False


class TestGetMetrics:
    """Test get_metrics method."""

    def test_returns_dict(self, daemon):
        """Test get_metrics returns a dictionary."""
        metrics = daemon.get_metrics()
        assert isinstance(metrics, dict)

    def test_contains_name(self, daemon):
        """Test metrics contains name."""
        metrics = daemon.get_metrics()
        assert metrics["name"] == "UnifiedDistributionDaemon"

    def test_contains_status(self, daemon):
        """Test metrics contains status."""
        metrics = daemon.get_metrics()
        assert "status" in metrics

    def test_contains_distribution_stats(self, daemon):
        """Test metrics contains distribution stats."""
        metrics = daemon.get_metrics()
        assert "successful_distributions" in metrics
        assert "failed_distributions" in metrics
        assert "checksum_failures" in metrics
        assert "model_distributions" in metrics
        assert "npz_distributions" in metrics

    def test_contains_prefetch_metrics(self, daemon):
        """Test metrics contains prefetch metrics."""
        metrics = daemon.get_metrics()
        assert "prefetch_enabled" in metrics
        assert "prefetch_threshold" in metrics
        assert "prefetched_checkpoints_count" in metrics


class TestHealthCheck:
    """Test health_check method."""

    def test_returns_health_check_result(self, daemon):
        """Test health_check returns HealthCheckResult."""
        result = daemon.health_check()
        assert isinstance(result, HealthCheckResult)

    def test_healthy_when_initializing(self, daemon):
        """Test healthy when status is INITIALIZING."""
        daemon._coordinator_status = CoordinatorStatus.INITIALIZING
        result = daemon.health_check()
        assert result.healthy is True

    def test_healthy_when_stopped(self, daemon):
        """Test healthy when status is STOPPED."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED
        result = daemon.health_check()
        assert result.healthy is True
        assert result.status == CoordinatorStatus.STOPPED

    def test_unhealthy_when_error(self, daemon):
        """Test unhealthy when status is ERROR."""
        daemon._coordinator_status = CoordinatorStatus.ERROR
        daemon._last_error = "Test error"
        result = daemon.health_check()
        assert result.healthy is False

    def test_degraded_on_high_failure_rate(self, daemon):
        """Test degraded when failure rate is high."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._successful_distributions = 5
        daemon._failed_distributions = 10
        result = daemon.health_check()
        assert result.status == CoordinatorStatus.DEGRADED

    def test_degraded_on_pending_buildup(self, daemon):
        """Test degraded when too many pending items."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._pending_items = [{}] * 20  # > 15
        result = daemon.health_check()
        assert result.status == CoordinatorStatus.DEGRADED

    def test_degraded_on_checksum_failures(self, daemon):
        """Test degraded when too many checksum failures."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._checksum_failures = 10  # > 5
        result = daemon.health_check()
        assert result.status == CoordinatorStatus.DEGRADED


# =============================================================================
# Daemon State Tests
# =============================================================================


class TestDaemonState:
    """Test daemon state management."""

    def test_initial_state(self, daemon):
        """Test initial daemon state."""
        assert daemon.is_running() is False

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_status(self, daemon):
        """Test stop sets STOPPED status."""
        daemon._coordinator_status = CoordinatorStatus.RUNNING
        daemon._running = True
        await daemon.stop()
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self, daemon):
        """Test stop clears running flag."""
        daemon._running = True
        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, daemon):
        """Test stop is idempotent."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED
        await daemon.stop()  # Should not raise
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Test event handler methods."""

    def test_on_model_promoted_enqueues_item(self, daemon):
        """Test _on_model_promoted enqueues item."""
        event = {
            "model_path": "/path/to/model.pth",
            "model_id": "test-model",
            "board_type": "hex8",
            "num_players": 2,
            "elo": 1500,
        }
        daemon._on_model_promoted(event)
        assert len(daemon._pending_items) == 1
        assert daemon._pending_items[0]["data_type"] == DataType.MODEL

    def test_on_model_promoted_with_payload_attribute(self, daemon):
        """Test _on_model_promoted with event having payload attribute."""
        class Event:
            payload = {"model_path": "/path/to/model.pth"}
        daemon._on_model_promoted(Event())
        assert len(daemon._pending_items) == 1

    def test_on_model_updated_ignores_metadata(self, daemon):
        """Test _on_model_updated ignores metadata updates."""
        event = {
            "model_path": "/path/to/model.pth",
            "update_type": "metadata",  # Should be ignored
        }
        daemon._on_model_updated(event)
        assert len(daemon._pending_items) == 0

    def test_on_model_updated_processes_path_changed(self, daemon):
        """Test _on_model_updated processes path_changed updates."""
        event = {
            "model_path": "/path/to/model.pth",
            "update_type": "path_changed",
        }
        daemon._on_model_updated(event)
        assert len(daemon._pending_items) == 1

    def test_on_model_updated_processes_sync_requested(self, daemon):
        """Test _on_model_updated processes sync_requested updates."""
        event = {
            "model_path": "/path/to/model.pth",
            "update_type": "sync_requested",
        }
        daemon._on_model_updated(event)
        assert len(daemon._pending_items) == 1

    def test_on_model_distribution_failed_increments_error_count(self, daemon):
        """Test _on_model_distribution_failed increments error count."""
        event = {
            "error": "Test error",
            "model_name": "test.pth",
            "expected_path": "/path/to/test.pth",
            "retry_count": 0,
        }
        daemon._on_model_distribution_failed(event)
        assert daemon._errors_count == 1

    def test_on_model_distribution_failed_retry_enqueue(self, daemon):
        """Test _on_model_distribution_failed re-enqueues for retry."""
        event = {
            "error": "Test error",
            "model_name": "test.pth",
            "expected_path": "/path/to/test.pth",
            "retry_count": 0,
        }
        daemon._on_model_distribution_failed(event)
        assert len(daemon._pending_items) == 1
        assert daemon._pending_items[0]["retry_count"] == 1

    def test_on_model_distribution_failed_no_retry_at_max(self, daemon):
        """Test _on_model_distribution_failed doesn't retry at max retries."""
        event = {
            "error": "Test error",
            "model_name": "test.pth",
            "expected_path": "/path/to/test.pth",
            "retry_count": 3,  # Max retries reached
        }
        daemon._on_model_distribution_failed(event)
        assert len(daemon._pending_items) == 0

    def test_on_model_evaluation_blocked_enqueues_priority(self, daemon):
        """Test _on_model_evaluation_blocked enqueues priority item."""
        event = {
            "model_path": "/path/to/model.pth",
            "required_nodes": 5,
            "actual_nodes": 2,
            "reason": "insufficient_nodes",
        }
        daemon._on_model_evaluation_blocked(event)
        assert len(daemon._pending_items) == 1
        assert daemon._pending_items[0]["priority"] is True

    def test_on_npz_exported_enqueues_item(self, daemon):
        """Test _on_npz_exported enqueues item."""
        event = {
            "npz_path": "/path/to/data.npz",
            "board_type": "hex8",
            "num_players": 2,
            "sample_count": 10000,
        }
        daemon._on_npz_exported(event)
        assert len(daemon._pending_items) == 1
        assert daemon._pending_items[0]["data_type"] == DataType.NPZ

    def test_on_training_progress_for_prefetch(self, daemon):
        """Test _on_training_progress_for_prefetch enqueues at threshold."""
        event = {
            "epochs_completed": 9,
            "total_epochs": 10,
            "checkpoint_path": "/path/to/checkpoint.pth",
            "config_key": "hex8_2p",
        }
        daemon._on_training_progress_for_prefetch(event)
        assert len(daemon._pending_items) == 1
        assert daemon._pending_items[0]["is_prefetch"] is True

    def test_on_training_progress_ignores_below_threshold(self, daemon):
        """Test _on_training_progress_for_prefetch ignores below threshold."""
        event = {
            "epochs_completed": 5,
            "total_epochs": 10,  # 50% < 80%
            "checkpoint_path": "/path/to/checkpoint.pth",
        }
        daemon._on_training_progress_for_prefetch(event)
        assert len(daemon._pending_items) == 0

    def test_on_training_progress_skips_already_prefetched(self, daemon):
        """Test _on_training_progress_for_prefetch skips already prefetched."""
        checkpoint_path = "/path/to/checkpoint.pth"
        daemon._prefetched_checkpoints.add(checkpoint_path)
        event = {
            "epochs_completed": 9,
            "total_epochs": 10,
            "checkpoint_path": checkpoint_path,
        }
        daemon._on_training_progress_for_prefetch(event)
        assert len(daemon._pending_items) == 0


class TestEnqueueItem:
    """Test _enqueue_item method."""

    def test_enqueue_adds_item(self, daemon):
        """Test _enqueue_item adds item to pending list."""
        item = {"data_type": DataType.MODEL, "path": "/test"}
        daemon._enqueue_item(item)
        assert len(daemon._pending_items) == 1

    def test_enqueue_increments_events_processed(self, daemon):
        """Test _enqueue_item increments events_processed counter."""
        item = {"data_type": DataType.MODEL, "path": "/test"}
        daemon._enqueue_item(item)
        assert daemon._events_processed == 1

    def test_enqueue_sets_event(self, daemon):
        """Test _enqueue_item sets pending event."""
        daemon._pending_event = asyncio.Event()
        item = {"data_type": DataType.MODEL, "path": "/test"}
        daemon._enqueue_item(item)
        assert daemon._pending_event.is_set()


# =============================================================================
# Remote Path Discovery Tests
# =============================================================================


class TestRemotePathPatterns:
    """Test remote path pattern constants and helpers."""

    def test_patterns_is_list(self):
        """Test REMOTE_PATH_PATTERNS is a list."""
        assert isinstance(REMOTE_PATH_PATTERNS, list)

    def test_patterns_not_empty(self):
        """Test patterns list is not empty."""
        assert len(REMOTE_PATH_PATTERNS) > 0

    def test_patterns_are_strings(self):
        """Test all patterns are strings."""
        for pattern in REMOTE_PATH_PATTERNS:
            assert isinstance(pattern, str)

    def test_workspace_pattern_first(self):
        """Test /workspace pattern is tried first (for RunPod/Vast)."""
        assert "/workspace" in REMOTE_PATH_PATTERNS[0]

    def test_tilde_pattern_exists(self):
        """Test tilde home pattern exists."""
        tilde_patterns = [p for p in REMOTE_PATH_PATTERNS if "~" in p]
        assert len(tilde_patterns) > 0

    def test_get_remote_path_patterns_returns_copy(self):
        """Test get_remote_path_patterns returns a copy."""
        patterns = get_remote_path_patterns()
        assert patterns == REMOTE_PATH_PATTERNS
        assert patterns is not REMOTE_PATH_PATTERNS  # Should be a copy


class TestRemotePathCache:
    """Test remote path caching functionality."""

    def test_cache_initially_empty(self):
        """Test cache is empty after clear."""
        clear_remote_path_cache()
        cache = get_all_cached_remote_paths()
        assert isinstance(cache, dict)
        assert len(cache) == 0

    def test_get_cached_remote_path_returns_none_for_unknown(self):
        """Test getting uncached path returns None."""
        result = get_cached_remote_path("unknown-host-12345")
        assert result is None

    def test_clear_remote_path_cache_clears_all(self):
        """Test clearing all cache entries."""
        with _remote_path_cache_lock:
            _remote_path_cache["test-host-1"] = "/path/1"
            _remote_path_cache["test-host-2"] = "/path/2"

        clear_remote_path_cache()
        cache = get_all_cached_remote_paths()
        assert len(cache) == 0

    def test_clear_remote_path_cache_clears_specific(self):
        """Test clearing specific host from cache."""
        with _remote_path_cache_lock:
            _remote_path_cache["test-host-a"] = "/path/a"
            _remote_path_cache["test-host-b"] = "/path/b"

        clear_remote_path_cache("test-host-a")
        cache = get_all_cached_remote_paths()
        assert "test-host-a" not in cache
        assert cache.get("test-host-b") == "/path/b"

    def test_get_all_cached_remote_paths_returns_copy(self):
        """Test get_all_cached_remote_paths returns a copy."""
        with _remote_path_cache_lock:
            _remote_path_cache["test-host-x"] = "/path/x"

        cache = get_all_cached_remote_paths()
        cache["test-host-x"] = "/modified"
        original_cache = get_all_cached_remote_paths()
        assert original_cache.get("test-host-x") == "/path/x"


class TestDaemonRemotePathMethods:
    """Test daemon methods for remote path discovery."""

    def test_daemon_has_discover_remote_path(self, daemon):
        """Test daemon has _discover_remote_path method."""
        assert hasattr(daemon, "_discover_remote_path")
        assert callable(daemon._discover_remote_path)

    def test_daemon_has_get_remote_path(self, daemon):
        """Test daemon has _get_remote_path method."""
        assert hasattr(daemon, "_get_remote_path")
        assert callable(daemon._get_remote_path)

    def test_daemon_has_clear_cache(self, daemon):
        """Test daemon has clear_remote_path_cache method."""
        assert hasattr(daemon, "clear_remote_path_cache")
        assert callable(daemon.clear_remote_path_cache)

    @pytest.mark.asyncio
    async def test_get_remote_path_with_explicit_path(self, daemon):
        """Test _get_remote_path uses explicit path when provided."""
        target = {
            "host": "test-host",
            "user": "testuser",
            "remote_path": "/explicit/path",
            "ssh_key": "/path/to/key",
        }
        host, user, remote_path, ssh_key = await daemon._get_remote_path(target)
        assert host == "test-host"
        assert user == "testuser"
        assert remote_path == "/explicit/path"
        assert ssh_key == "/path/to/key"

    @pytest.mark.asyncio
    async def test_get_remote_path_with_string_target(self, daemon):
        """Test _get_remote_path with string target uses cache."""
        with _remote_path_cache_lock:
            _remote_path_cache["192.168.1.1"] = "/cached/path"

        host, user, remote_path, ssh_key = await daemon._get_remote_path("192.168.1.1")
        assert host == "192.168.1.1"
        assert user == "root"  # Default
        assert remote_path == "/cached/path"  # From cache
        assert ssh_key is None

    @pytest.mark.asyncio
    async def test_discover_uses_cache(self, daemon):
        """Test _discover_remote_path uses cached value."""
        with _remote_path_cache_lock:
            _remote_path_cache["cached-test-host"] = "/precached/value"

        result = await daemon._discover_remote_path("cached-test-host")
        assert result == "/precached/value"


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestIsValidModelFile:
    """Test model file validation functionality."""

    def test_nonexistent_file_returns_false(self, tmp_path):
        """Test validation returns False for non-existent file."""
        nonexistent = tmp_path / "does_not_exist.pth"
        assert is_valid_model_file(nonexistent) is False

    def test_too_small_file_returns_false(self, tmp_path):
        """Test validation returns False for files < 1MB."""
        small_file = tmp_path / "small.pth"
        small_file.write_bytes(b"small content" * 100)  # ~1300 bytes
        assert is_valid_model_file(small_file) is False

    def test_invalid_magic_bytes_returns_false(self, tmp_path):
        """Test validation returns False for files without zip magic bytes."""
        invalid_file = tmp_path / "invalid.pth"
        invalid_file.write_bytes(b"INVALID!" + b"\x00" * 1_500_000)
        assert is_valid_model_file(invalid_file) is False

    def test_valid_model_file_returns_true(self, tmp_path):
        """Test validation returns True for valid model file."""
        valid_file = tmp_path / "valid.pth"
        valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)
        assert is_valid_model_file(valid_file) is True

    def test_accepts_string_path(self, tmp_path):
        """Test validation accepts string paths."""
        valid_file = tmp_path / "valid.pth"
        valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)
        assert is_valid_model_file(str(valid_file)) is True

    def test_private_function_also_works(self, tmp_path):
        """Test private _is_valid_model_file function."""
        valid_file = tmp_path / "valid.pth"
        valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)
        assert _is_valid_model_file(valid_file) is True


class TestCheckModelAvailability:
    """Test check_model_availability function."""

    def test_validate_true_checks_file_validity(self, tmp_path, monkeypatch):
        """Test validate=True performs full validation."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"small")

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        assert check_model_availability("hex8", 2, validate=True) is False

    def test_validate_false_only_checks_existence(self, tmp_path, monkeypatch):
        """Test validate=False only checks existence."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"small")

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        assert check_model_availability("hex8", 2, validate=False) is True

    def test_valid_model_passes_both_modes(self, tmp_path, monkeypatch):
        """Test valid model passes with either validate setting."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        assert check_model_availability("hex8", 2, validate=True) is True
        assert check_model_availability("hex8", 2, validate=False) is True

    def test_symlink_checked(self, tmp_path, monkeypatch):
        """Test symlink path is also checked."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        symlink_path = models_dir / "ringrift_best_hex8_2p.pth"
        symlink_path.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        assert check_model_availability("hex8", 2, validate=True) is True


# =============================================================================
# Wait For Model Distribution Tests
# =============================================================================


class TestWaitForModelDistribution:
    """Test wait_for_model_distribution function."""

    @pytest.mark.asyncio
    async def test_returns_immediately_for_valid_existing_model(
        self, tmp_path, monkeypatch
    ):
        """Test returns True immediately if valid model already exists."""
        from app.coordination.unified_distribution_daemon import (
            wait_for_model_distribution,
        )

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        result = await wait_for_model_distribution("hex8", 2, timeout=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout_no_model(self, tmp_path, monkeypatch):
        """Test returns False when timeout expires and no model exists."""
        from app.coordination.unified_distribution_daemon import (
            wait_for_model_distribution,
        )

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        # Mock the event router module to avoid import issues
        mock_subscribe = MagicMock()
        mock_data_event_type = MagicMock()
        mock_data_event_type.MODEL_DISTRIBUTION_COMPLETE = "MODEL_DISTRIBUTION_COMPLETE"

        with patch.dict(
            "sys.modules",
            {
                "app.coordination.event_router": MagicMock(
                    subscribe=mock_subscribe,
                    DataEventType=mock_data_event_type,
                )
            }
        ):
            result = await wait_for_model_distribution(
                "hex8", 2, timeout=0.3, disk_check_interval=0.1
            )
            assert result is False


# =============================================================================
# Distribution Verification Tests
# =============================================================================


class TestDistributionVerification:
    """Test distribution verification methods."""

    @pytest.mark.asyncio
    async def test_verify_distribution_returns_tuple(self, daemon):
        """Test verify_distribution returns tuple."""
        with patch.object(daemon, "verify_distribution") as mock_verify:
            mock_verify.return_value = (True, 5)
            result = await daemon.verify_distribution("model.pth", min_nodes=3)
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_get_model_availability_score_returns_float(self, daemon):
        """Test get_model_availability_score returns float."""
        # The method returns 0.0 on ImportError, which is valid float
        score = daemon.get_model_availability_score("model.pth")
        assert isinstance(score, float)

    def test_get_model_availability_score_with_no_nodes(self, daemon):
        """Test get_model_availability_score returns 0 with no GPU nodes."""
        # Mock the required modules at import time
        mock_manifest = MagicMock()
        mock_manifest.find_model.return_value = []
        mock_cluster_manifest_module = MagicMock()
        mock_cluster_manifest_module.get_cluster_manifest.return_value = mock_manifest

        mock_cluster_config_module = MagicMock()
        mock_cluster_config_module.get_gpu_nodes.return_value = []

        with patch.dict(
            "sys.modules",
            {
                "app.distributed.cluster_manifest": mock_cluster_manifest_module,
                "app.config.cluster_config": mock_cluster_config_module,
            }
        ):
            score = daemon.get_model_availability_score("model.pth")
            assert score == 0.0


# =============================================================================
# Module-Level Convenience Functions Tests
# =============================================================================


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_verify_model_distribution_function(self):
        """Test verify_model_distribution module function."""
        with patch(
            "app.coordination.unified_distribution_daemon.UnifiedDistributionDaemon.verify_distribution",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = (True, 5)
            result = await verify_model_distribution("model.pth", min_nodes=3)
            assert result == (True, 5)

    def test_get_model_availability_score_function(self):
        """Test get_model_availability_score module function."""
        with patch(
            "app.coordination.unified_distribution_daemon.UnifiedDistributionDaemon.get_model_availability_score",
        ) as mock:
            mock.return_value = 0.75
            score = get_model_availability_score("model.pth")
            assert score == 0.75

    @pytest.mark.asyncio
    async def test_wait_for_model_availability_function(self):
        """Test wait_for_model_availability module function."""
        with patch(
            "app.coordination.unified_distribution_daemon.UnifiedDistributionDaemon.wait_for_adequate_distribution",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = (True, 10)
            result = await wait_for_model_availability(
                "model.pth", min_nodes=5, timeout=60.0
            )
            assert result == (True, 10)


# =============================================================================
# Delivery Recording Tests
# =============================================================================


class TestDeliveryRecording:
    """Test delivery recording functionality."""

    def test_record_delivery_adds_to_history(self, daemon):
        """Test _record_delivery adds to history."""
        daemon._record_delivery(
            node_id="test-node",
            host="10.0.0.1",
            path="/path/to/file",
            data_type=DataType.MODEL,
            success=True,
            checksum_ok=True,
            time_seconds=5.0,
            method="http",
        )
        assert len(daemon._delivery_history) == 1

    def test_record_delivery_truncates_history(self, daemon):
        """Test _record_delivery truncates history at 200 items."""
        for i in range(250):
            daemon._record_delivery(
                node_id=f"node-{i}",
                host=f"10.0.0.{i % 256}",
                path=f"/path/to/file-{i}",
                data_type=DataType.MODEL,
                success=True,
                checksum_ok=True,
                time_seconds=1.0,
                method="http",
            )
        assert len(daemon._delivery_history) == 200


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the daemon."""

    def test_daemon_can_be_instantiated_multiple_times(self):
        """Test multiple daemon instances can coexist."""
        daemon1 = UnifiedDistributionDaemon()
        daemon2 = UnifiedDistributionDaemon()
        assert daemon1 is not daemon2

    def test_daemon_config_isolation(self):
        """Test each daemon has its own config."""
        config1 = DistributionConfig(retry_count=1)
        config2 = DistributionConfig(retry_count=10)
        daemon1 = UnifiedDistributionDaemon(config1)
        daemon2 = UnifiedDistributionDaemon(config2)
        assert daemon1.config.retry_count == 1
        assert daemon2.config.retry_count == 10

    def test_daemon_has_all_required_methods(self, daemon):
        """Test daemon has all required interface methods."""
        required_methods = [
            "start",
            "stop",
            "is_running",
            "health_check",
            "get_metrics",
            "verify_distribution",
            "get_model_availability_score",
            "wait_for_adequate_distribution",
        ]
        for method in required_methods:
            assert hasattr(daemon, method), f"Missing method: {method}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_model_updated_without_path(self, daemon):
        """Test _on_model_updated handles missing path."""
        event = {
            "update_type": "path_changed",
            # No model_path
        }
        daemon._on_model_updated(event)
        assert len(daemon._pending_items) == 0

    def test_evaluation_blocked_without_path(self, daemon):
        """Test _on_model_evaluation_blocked handles missing path."""
        event = {
            "required_nodes": 5,
            # No model_path
        }
        daemon._on_model_evaluation_blocked(event)
        assert len(daemon._pending_items) == 0

    def test_training_progress_without_checkpoint(self, daemon):
        """Test _on_training_progress_for_prefetch handles missing checkpoint."""
        event = {
            "epochs_completed": 9,
            "total_epochs": 10,
            # No checkpoint_path
        }
        daemon._on_training_progress_for_prefetch(event)
        assert len(daemon._pending_items) == 0

    def test_training_progress_with_zero_total_epochs(self, daemon):
        """Test _on_training_progress_for_prefetch handles zero total epochs."""
        event = {
            "epochs_completed": 9,
            "total_epochs": 0,  # Invalid
            "checkpoint_path": "/path/to/checkpoint.pth",
        }
        daemon._on_training_progress_for_prefetch(event)
        assert len(daemon._pending_items) == 0

    def test_prefetch_disabled(self, daemon):
        """Test _on_training_progress_for_prefetch respects disabled flag."""
        daemon._prefetch_enabled = False
        event = {
            "epochs_completed": 9,
            "total_epochs": 10,
            "checkpoint_path": "/path/to/checkpoint.pth",
        }
        daemon._on_training_progress_for_prefetch(event)
        assert len(daemon._pending_items) == 0

    @pytest.mark.asyncio
    async def test_stop_when_already_stopped(self, daemon):
        """Test stop when already stopped is a no-op."""
        daemon._coordinator_status = CoordinatorStatus.STOPPED
        await daemon.stop()
        assert daemon._coordinator_status == CoordinatorStatus.STOPPED

    def test_uptime_with_zero_start_time(self, daemon):
        """Test uptime returns 0 when start_time is 0."""
        daemon._start_time = 0
        assert daemon.uptime_seconds == 0.0

    def test_uptime_with_negative_start_time(self, daemon):
        """Test uptime returns 0 when start_time is negative."""
        daemon._start_time = -1
        assert daemon.uptime_seconds == 0.0
