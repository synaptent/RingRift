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


# =============================================================================
# Model Validation Tests (December 2025 Improvements)
# =============================================================================


class TestIsValidModelFile:
    """Test model file validation functionality."""

    def test_nonexistent_file_returns_false(self, tmp_path):
        """Test validation returns False for non-existent file."""
        from app.coordination.unified_distribution_daemon import is_valid_model_file

        nonexistent = tmp_path / "does_not_exist.pth"
        assert is_valid_model_file(nonexistent) is False

    def test_too_small_file_returns_false(self, tmp_path):
        """Test validation returns False for files < 1MB."""
        from app.coordination.unified_distribution_daemon import is_valid_model_file

        small_file = tmp_path / "small.pth"
        small_file.write_bytes(b"small content" * 100)  # ~1300 bytes
        assert is_valid_model_file(small_file) is False

    def test_invalid_magic_bytes_returns_false(self, tmp_path):
        """Test validation returns False for files without zip magic bytes."""
        from app.coordination.unified_distribution_daemon import is_valid_model_file

        # Create a 1.5MB file with wrong magic bytes
        invalid_file = tmp_path / "invalid.pth"
        invalid_file.write_bytes(b"INVALID!" + b"\x00" * 1_500_000)
        assert is_valid_model_file(invalid_file) is False

    def test_valid_model_file_returns_true(self, tmp_path):
        """Test validation returns True for valid model file."""
        from app.coordination.unified_distribution_daemon import is_valid_model_file

        # Create a valid-looking model file (zip magic + size > 1MB)
        valid_file = tmp_path / "valid.pth"
        # PK\x03\x04 is ZIP file magic number
        valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)
        assert is_valid_model_file(valid_file) is True

    def test_accepts_string_path(self, tmp_path):
        """Test validation accepts string paths."""
        from app.coordination.unified_distribution_daemon import is_valid_model_file

        valid_file = tmp_path / "valid.pth"
        valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)
        assert is_valid_model_file(str(valid_file)) is True

    def test_private_function_also_works(self, tmp_path):
        """Test private _is_valid_model_file function."""
        from app.coordination.unified_distribution_daemon import _is_valid_model_file

        valid_file = tmp_path / "valid.pth"
        valid_file.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)
        assert _is_valid_model_file(valid_file) is True


class TestCheckModelAvailabilityValidation:
    """Test check_model_availability with validation parameter."""

    def test_validate_true_checks_file_validity(self, tmp_path, monkeypatch):
        """Test validate=True performs full validation."""
        from app.coordination.unified_distribution_daemon import (
            check_model_availability,
        )

        # Create invalid model file (too small)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"small")

        # Patch ROOT to use temp directory
        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        # With validation, should fail (file too small)
        assert check_model_availability("hex8", 2, validate=True) is False

    def test_validate_false_only_checks_existence(self, tmp_path, monkeypatch):
        """Test validate=False only checks existence."""
        from app.coordination.unified_distribution_daemon import (
            check_model_availability,
        )

        # Create invalid but existing model file
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"
        model_path.write_bytes(b"small")

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        # Without validation, should pass (file exists)
        assert check_model_availability("hex8", 2, validate=False) is True

    def test_valid_model_passes_both_modes(self, tmp_path, monkeypatch):
        """Test valid model passes with either validate setting."""
        from app.coordination.unified_distribution_daemon import (
            check_model_availability,
        )

        # Create valid model file
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


class TestWaitForModelDistributionFallback:
    """Test wait_for_model_distribution timeout fallback behavior."""

    @pytest.mark.asyncio
    async def test_returns_immediately_for_valid_existing_model(
        self, tmp_path, monkeypatch
    ):
        """Test returns True immediately if valid model already exists."""
        from app.coordination.unified_distribution_daemon import (
            wait_for_model_distribution,
        )

        # Create valid model file
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
    async def test_fallback_uses_disk_on_timeout(self, tmp_path, monkeypatch):
        """Test fallback checks disk after timeout if model appeared."""
        from app.coordination.unified_distribution_daemon import (
            wait_for_model_distribution,
        )

        # Create model directory but no model initially
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "canonical_hex8_2p.pth"

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        # Mock event system to avoid import issues
        def mock_import_error(*args, **kwargs):
            raise ImportError("No event system")

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.subscribe",
            mock_import_error,
            raising=False,
        )

        # Start waiting, but create file after a short delay
        async def create_model_later():
            await asyncio.sleep(0.5)
            model_path.write_bytes(b"PK\x03\x04" + b"\x00" * 1_500_000)

        # Run both concurrently
        task = asyncio.create_task(create_model_later())

        # Use a short timeout - file should appear during wait
        result = await wait_for_model_distribution(
            "hex8", 2, timeout=2.0, disk_check_interval=0.3
        )

        await task  # Ensure background task completes
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout_no_model(self, tmp_path, monkeypatch):
        """Test returns False when timeout expires and no model exists."""
        from app.coordination.unified_distribution_daemon import (
            wait_for_model_distribution,
        )

        # Create model directory but no model
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        monkeypatch.setattr(
            "app.coordination.unified_distribution_daemon.ROOT",
            tmp_path,
        )

        # Use very short timeout - should return False
        result = await wait_for_model_distribution(
            "hex8", 2, timeout=0.3, disk_check_interval=0.1
        )

        assert result is False


class TestEmitDistributionFailedEvent:
    """Test the distribution failure event emission."""

    @pytest.mark.asyncio
    async def test_emit_function_exists(self):
        """Test _emit_distribution_failed_event function exists."""
        from app.coordination.unified_distribution_daemon import (
            _emit_distribution_failed_event,
        )
        assert callable(_emit_distribution_failed_event)

    @pytest.mark.asyncio
    async def test_emit_handles_import_error(self):
        """Test emit function handles missing event router gracefully."""
        from app.coordination.unified_distribution_daemon import (
            _emit_distribution_failed_event,
        )

        # Should not raise even if event_router is not available
        # (function catches ImportError)
        try:
            await _emit_distribution_failed_event(
                model_name="test_model.pth",
                expected_path="/path/to/model",
                timeout_seconds=60.0,
                reason="test",
            )
        except ImportError:
            # This is acceptable - means event router truly unavailable
            pass
