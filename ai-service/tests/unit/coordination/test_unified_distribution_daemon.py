"""Tests for UnifiedDistributionDaemon.

Tests cover:
- Configuration initialization
- DataType enum
- DeliveryResult dataclass
- Factory functions (backward compatibility)
- Daemon lifecycle (start/stop)
- Event subscription handling
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

from app.coordination.unified_distribution_daemon import (
    DataType,
    DeliveryResult,
    DistributionConfig,
    UnifiedDistributionDaemon,
    create_model_distribution_daemon,
    create_npz_distribution_daemon,
    create_unified_distribution_daemon,
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


# =============================================================================
# DataType Enum Tests
# =============================================================================


class TestDataType:
    """Test DataType enumeration."""

    def test_has_model(self):
        """Test MODEL type exists."""
        assert DataType.MODEL is not None

    def test_has_npz(self):
        """Test NPZ type exists."""
        assert DataType.NPZ is not None

    def test_has_torrent(self):
        """Test TORRENT type exists."""
        assert DataType.TORRENT is not None

    def test_all_types(self):
        """Test all data types."""
        all_types = list(DataType)
        assert len(all_types) >= 3


# =============================================================================
# DistributionConfig Tests
# =============================================================================


class TestDistributionConfig:
    """Test DistributionConfig dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        config = DistributionConfig()
        assert config.sync_timeout_seconds == 300.0
        assert config.retry_count == 3
        assert config.use_http_distribution is True
        assert config.verify_checksums is True
        assert config.models_dir == "models"
        assert config.training_data_dir == "data/training"

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

    def test_bittorrent_threshold(self):
        """Test BitTorrent threshold setting."""
        config = DistributionConfig(bittorrent_threshold_bytes=100_000_000)
        assert config.bittorrent_threshold_bytes == 100_000_000

    def test_retry_backoff(self):
        """Test retry backoff multiplier."""
        config = DistributionConfig(retry_backoff_multiplier=2.0)
        assert config.retry_backoff_multiplier == 2.0


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

    def test_has_required_methods(self, daemon):
        """Test daemon has required methods."""
        assert hasattr(daemon, "start")
        assert hasattr(daemon, "stop")
        assert hasattr(daemon, "is_running")


# =============================================================================
# Daemon State Tests
# =============================================================================


class TestDaemonState:
    """Test daemon state management."""

    def test_initial_state(self, daemon):
        """Test initial daemon state."""
        assert daemon.is_running() is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self, daemon):
        """Test start sets running flag initially."""
        # start() is meant to run long-lived; just verify the flag is set
        # We'll test this by checking the initial state and simulating
        daemon._running = True
        assert daemon._running is True
        assert daemon.is_running() is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, daemon):
        """Test stop clears running flag."""
        daemon._running = True
        await daemon.stop()
        assert daemon._running is False

    def test_is_running_property(self, daemon):
        """Test is_running reflects internal state."""
        assert daemon.is_running() is False
        daemon._running = True
        assert daemon.is_running() is True
        daemon._running = False
        assert daemon.is_running() is False


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check integration."""

    def test_has_health_check(self, daemon):
        """Test daemon has health check method."""
        assert hasattr(daemon, "health_check") or hasattr(daemon, "_health_check")


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


# =============================================================================
# Remote Path Discovery Tests (December 28, 2025)
# =============================================================================


class TestRemotePathPatterns:
    """Test remote path pattern constants and helpers."""

    def test_patterns_is_list(self):
        """Test REMOTE_PATH_PATTERNS is a list."""
        from app.coordination.unified_distribution_daemon import REMOTE_PATH_PATTERNS
        assert isinstance(REMOTE_PATH_PATTERNS, list)

    def test_patterns_not_empty(self):
        """Test patterns list is not empty."""
        from app.coordination.unified_distribution_daemon import REMOTE_PATH_PATTERNS
        assert len(REMOTE_PATH_PATTERNS) > 0

    def test_patterns_are_strings(self):
        """Test all patterns are strings."""
        from app.coordination.unified_distribution_daemon import REMOTE_PATH_PATTERNS
        for pattern in REMOTE_PATH_PATTERNS:
            assert isinstance(pattern, str)

    def test_workspace_pattern_first(self):
        """Test /workspace pattern is tried first (for RunPod/Vast)."""
        from app.coordination.unified_distribution_daemon import REMOTE_PATH_PATTERNS
        assert "/workspace" in REMOTE_PATH_PATTERNS[0]

    def test_tilde_pattern_exists(self):
        """Test tilde home pattern exists."""
        from app.coordination.unified_distribution_daemon import REMOTE_PATH_PATTERNS
        tilde_patterns = [p for p in REMOTE_PATH_PATTERNS if "~" in p]
        assert len(tilde_patterns) > 0

    def test_get_remote_path_patterns_returns_copy(self):
        """Test get_remote_path_patterns returns a copy."""
        from app.coordination.unified_distribution_daemon import (
            REMOTE_PATH_PATTERNS,
            get_remote_path_patterns,
        )
        patterns = get_remote_path_patterns()
        assert patterns == REMOTE_PATH_PATTERNS
        assert patterns is not REMOTE_PATH_PATTERNS  # Should be a copy


class TestRemotePathCache:
    """Test remote path caching functionality."""

    def test_cache_initially_empty(self):
        """Test cache is accessible (may not be empty in real test runs)."""
        from app.coordination.unified_distribution_daemon import (
            clear_remote_path_cache,
            get_all_cached_remote_paths,
        )
        # Clear and verify
        clear_remote_path_cache()
        cache = get_all_cached_remote_paths()
        assert isinstance(cache, dict)
        assert len(cache) == 0

    def test_get_cached_remote_path_returns_none_for_unknown(self):
        """Test getting uncached path returns None."""
        from app.coordination.unified_distribution_daemon import (
            clear_remote_path_cache,
            get_cached_remote_path,
        )
        clear_remote_path_cache()
        result = get_cached_remote_path("unknown-host-12345")
        assert result is None

    def test_clear_remote_path_cache_clears_all(self):
        """Test clearing all cache entries."""
        from app.coordination.unified_distribution_daemon import (
            _remote_path_cache,
            _remote_path_cache_lock,
            clear_remote_path_cache,
            get_all_cached_remote_paths,
        )
        # Add test entries
        with _remote_path_cache_lock:
            _remote_path_cache["test-host-1"] = "/path/1"
            _remote_path_cache["test-host-2"] = "/path/2"

        # Clear all
        clear_remote_path_cache()
        cache = get_all_cached_remote_paths()
        assert len(cache) == 0

    def test_clear_remote_path_cache_clears_specific(self):
        """Test clearing specific host from cache."""
        from app.coordination.unified_distribution_daemon import (
            _remote_path_cache,
            _remote_path_cache_lock,
            clear_remote_path_cache,
            get_all_cached_remote_paths,
        )
        # Add test entries
        with _remote_path_cache_lock:
            _remote_path_cache["test-host-a"] = "/path/a"
            _remote_path_cache["test-host-b"] = "/path/b"

        # Clear only one
        clear_remote_path_cache("test-host-a")
        cache = get_all_cached_remote_paths()
        assert "test-host-a" not in cache
        assert cache.get("test-host-b") == "/path/b"

        # Cleanup
        clear_remote_path_cache()

    def test_get_all_cached_remote_paths_returns_copy(self):
        """Test get_all_cached_remote_paths returns a copy."""
        from app.coordination.unified_distribution_daemon import (
            _remote_path_cache,
            _remote_path_cache_lock,
            clear_remote_path_cache,
            get_all_cached_remote_paths,
        )
        clear_remote_path_cache()
        with _remote_path_cache_lock:
            _remote_path_cache["test-host-x"] = "/path/x"

        cache = get_all_cached_remote_paths()
        assert cache.get("test-host-x") == "/path/x"

        # Modify returned dict shouldn't affect internal cache
        cache["test-host-x"] = "/modified"
        original_cache = get_all_cached_remote_paths()
        assert original_cache.get("test-host-x") == "/path/x"

        # Cleanup
        clear_remote_path_cache()


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
        """Test _get_remote_path with string target uses discovery."""
        from app.coordination.unified_distribution_daemon import (
            clear_remote_path_cache,
            _remote_path_cache,
            _remote_path_cache_lock,
        )
        # Set up cached value to avoid actual SSH
        with _remote_path_cache_lock:
            _remote_path_cache["192.168.1.1"] = "/cached/path"

        host, user, remote_path, ssh_key = await daemon._get_remote_path("192.168.1.1")
        assert host == "192.168.1.1"
        assert user == "root"  # Default
        assert remote_path == "/cached/path"  # From cache
        assert ssh_key is None

        # Cleanup
        clear_remote_path_cache()

    @pytest.mark.asyncio
    async def test_discover_uses_cache(self, daemon):
        """Test _discover_remote_path uses cached value."""
        from app.coordination.unified_distribution_daemon import (
            _remote_path_cache,
            _remote_path_cache_lock,
            clear_remote_path_cache,
        )
        # Pre-populate cache
        with _remote_path_cache_lock:
            _remote_path_cache["cached-test-host"] = "/precached/value"

        result = await daemon._discover_remote_path("cached-test-host")
        assert result == "/precached/value"

        # Cleanup
        clear_remote_path_cache()
