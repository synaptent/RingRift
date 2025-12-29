"""Tests for app.distributed.storage_provider module.

This module tests the unified storage provider abstraction for
Lambda NFS, Vast.ai ephemeral, and local storage.
"""

import os
import platform
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test StorageProviderType Enum
# =============================================================================


class TestStorageProviderType:
    """Tests for StorageProviderType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        from app.distributed.storage_provider import StorageProviderType

        assert StorageProviderType.LAMBDA_NFS.value == "lambda"
        assert StorageProviderType.VAST_EPHEMERAL.value == "vast"
        assert StorageProviderType.LOCAL.value == "local"
        assert StorageProviderType.AWS_EFS.value == "aws"
        assert StorageProviderType.UNKNOWN.value == "unknown"


# =============================================================================
# Test StorageCapabilities Dataclass
# =============================================================================


class TestStorageCapabilities:
    """Tests for StorageCapabilities dataclass."""

    def test_default_values(self):
        """Test default values."""
        from app.distributed.storage_provider import StorageCapabilities

        caps = StorageCapabilities()

        assert caps.has_shared_storage is False
        assert caps.skip_rsync_for_shared is False
        assert caps.supports_direct_nfs is False
        assert caps.ephemeral is False
        assert caps.has_ram_disk is False
        assert caps.max_sync_interval_seconds == 60
        assert caps.priority_in_fallback == 50

    def test_custom_values(self):
        """Test with custom values."""
        from app.distributed.storage_provider import StorageCapabilities

        caps = StorageCapabilities(
            has_shared_storage=True,
            skip_rsync_for_shared=True,
            supports_direct_nfs=True,
            ephemeral=False,
            has_ram_disk=True,
            max_sync_interval_seconds=300,
            priority_in_fallback=90,
        )

        assert caps.has_shared_storage is True
        assert caps.skip_rsync_for_shared is True
        assert caps.supports_direct_nfs is True
        assert caps.max_sync_interval_seconds == 300


# =============================================================================
# Test StoragePaths Dataclass
# =============================================================================


class TestStoragePaths:
    """Tests for StoragePaths dataclass."""

    def test_required_fields(self):
        """Test that required fields are set."""
        from app.distributed.storage_provider import StoragePaths

        paths = StoragePaths(
            selfplay_games=Path("/data/selfplay"),
            model_checkpoints=Path("/data/models"),
            training_data=Path("/data/training"),
            elo_database=Path("/data/elo.db"),
            sync_staging=Path("/data/staging"),
            local_scratch=Path("/tmp/scratch"),
        )

        assert paths.selfplay_games == Path("/data/selfplay")
        assert paths.model_checkpoints == Path("/data/models")
        assert paths.nfs_base is None

    def test_optional_nfs_base(self):
        """Test optional nfs_base field."""
        from app.distributed.storage_provider import StoragePaths

        paths = StoragePaths(
            selfplay_games=Path("/nfs/selfplay"),
            model_checkpoints=Path("/nfs/models"),
            training_data=Path("/nfs/training"),
            elo_database=Path("/nfs/elo.db"),
            sync_staging=Path("/nfs/staging"),
            local_scratch=Path("/tmp/scratch"),
            nfs_base=Path("/nfs"),
        )

        assert paths.nfs_base == Path("/nfs")


# =============================================================================
# Test LocalStorageProvider
# =============================================================================


class TestLocalStorageProvider:
    """Tests for LocalStorageProvider."""

    def test_provider_type(self):
        """Test provider type is LOCAL."""
        from app.distributed.storage_provider import (
            LocalStorageProvider,
            StorageProviderType,
        )

        provider = LocalStorageProvider()
        assert provider.provider_type == StorageProviderType.LOCAL

    def test_default_paths(self):
        """Test default paths are set correctly."""
        from app.distributed.storage_provider import LocalStorageProvider

        provider = LocalStorageProvider()

        # Should have valid paths
        assert provider.selfplay_dir is not None
        assert provider.models_dir is not None
        assert provider.training_dir is not None

    def test_custom_base_dir(self):
        """Test with custom base directory."""
        from app.distributed.storage_provider import LocalStorageProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            provider = LocalStorageProvider(base_dir=base)

            assert provider.selfplay_dir == base / "data" / "selfplay"
            assert provider.models_dir == base / "models"

    def test_capabilities(self):
        """Test capabilities are correct."""
        from app.distributed.storage_provider import LocalStorageProvider

        provider = LocalStorageProvider()
        caps = provider.capabilities

        assert caps.has_shared_storage is False
        assert caps.skip_rsync_for_shared is False
        assert caps.ephemeral is False
        # has_ram_disk depends on platform
        if platform.system() == "Linux":
            assert caps.has_ram_disk is True
        else:
            assert caps.has_ram_disk is False

    def test_has_shared_storage(self):
        """Test has_shared_storage property."""
        from app.distributed.storage_provider import LocalStorageProvider

        provider = LocalStorageProvider()
        assert provider.has_shared_storage is False

    def test_is_ephemeral(self):
        """Test is_ephemeral property."""
        from app.distributed.storage_provider import LocalStorageProvider

        provider = LocalStorageProvider()
        assert provider.is_ephemeral is False

    def test_should_skip_rsync_to(self):
        """Test should_skip_rsync_to for local provider."""
        from app.distributed.storage_provider import LocalStorageProvider

        provider = LocalStorageProvider()
        # Local provider should never skip rsync
        assert provider.should_skip_rsync_to("any-node") is False

    def test_ensure_directories(self):
        """Test ensure_directories creates directories."""
        from app.distributed.storage_provider import LocalStorageProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "subdir"
            provider = LocalStorageProvider(base_dir=base)

            # Initially directories don't exist
            assert not provider.selfplay_dir.exists()

            provider.ensure_directories()

            # Now they should exist
            assert provider.selfplay_dir.exists()
            assert provider.models_dir.exists()
            assert provider.training_dir.exists()
            assert provider.scratch_dir.exists() or not Path("/tmp/ringrift").exists()

    def test_get_sync_config(self):
        """Test get_sync_config returns valid config."""
        from app.distributed.storage_provider import LocalStorageProvider

        provider = LocalStorageProvider()
        config = provider.get_sync_config()

        assert isinstance(config, dict)
        assert "poll_interval_seconds" in config
        assert "skip_rsync_for_nfs" in config
        assert config["skip_rsync_for_nfs"] is False


# =============================================================================
# Test VastEphemeralProvider
# =============================================================================


class TestVastEphemeralProvider:
    """Tests for VastEphemeralProvider."""

    def test_provider_type(self):
        """Test provider type is VAST_EPHEMERAL."""
        from app.distributed.storage_provider import (
            VastEphemeralProvider,
            StorageProviderType,
        )

        provider = VastEphemeralProvider()
        assert provider.provider_type == StorageProviderType.VAST_EPHEMERAL

    def test_default_workspace_base(self):
        """Test default workspace base path."""
        from app.distributed.storage_provider import VastEphemeralProvider

        provider = VastEphemeralProvider()
        assert provider.selfplay_dir == Path("/workspace/data/selfplay")
        assert provider.models_dir == Path("/workspace/models")

    def test_custom_workspace_base(self):
        """Test with custom workspace base."""
        from app.distributed.storage_provider import VastEphemeralProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            provider = VastEphemeralProvider(workspace_base=base)

            assert provider.selfplay_dir == base / "data" / "selfplay"
            assert provider.models_dir == base / "models"

    def test_capabilities(self):
        """Test capabilities for Vast.ai."""
        from app.distributed.storage_provider import VastEphemeralProvider

        provider = VastEphemeralProvider()
        caps = provider.capabilities

        assert caps.has_shared_storage is False
        assert caps.ephemeral is True  # Data lost on shutdown
        assert caps.has_ram_disk is True  # /dev/shm available
        assert caps.max_sync_interval_seconds == 15  # Aggressive sync

    def test_is_ephemeral(self):
        """Test is_ephemeral property."""
        from app.distributed.storage_provider import VastEphemeralProvider

        provider = VastEphemeralProvider()
        assert provider.is_ephemeral is True

    def test_scratch_dir_uses_ram_disk(self):
        """Test that scratch uses RAM disk."""
        from app.distributed.storage_provider import VastEphemeralProvider

        provider = VastEphemeralProvider()
        assert "/dev/shm" in str(provider.scratch_dir)

    def test_is_available_with_workspace(self):
        """Test is_available when workspace exists."""
        from app.distributed.storage_provider import VastEphemeralProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()

            with patch.object(VastEphemeralProvider, "WORKSPACE_BASE", workspace):
                assert VastEphemeralProvider.is_available() is True

    def test_is_available_with_env_var(self):
        """Test is_available with VAST_CONTAINERLABEL env var."""
        from app.distributed.storage_provider import VastEphemeralProvider

        with patch.dict(os.environ, {"VAST_CONTAINERLABEL": "test-label"}):
            with patch.object(VastEphemeralProvider, "WORKSPACE_BASE", Path("/nonexistent")):
                assert VastEphemeralProvider.is_available() is True


# =============================================================================
# Test LambdaNFSProvider
# =============================================================================


class TestLambdaNFSProvider:
    """Tests for LambdaNFSProvider."""

    def test_provider_type(self):
        """Test provider type is LAMBDA_NFS."""
        from app.distributed.storage_provider import (
            LambdaNFSProvider,
            StorageProviderType,
        )

        provider = LambdaNFSProvider()
        assert provider.provider_type == StorageProviderType.LAMBDA_NFS

    def test_default_nfs_base(self):
        """Test default NFS base path."""
        from app.distributed.storage_provider import LambdaNFSProvider

        provider = LambdaNFSProvider()
        assert provider.paths.nfs_base == Path("/lambda/nfs/RingRift")

    def test_custom_nfs_base(self):
        """Test with custom NFS base."""
        from app.distributed.storage_provider import LambdaNFSProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            nfs_base = Path(tmpdir)
            provider = LambdaNFSProvider(nfs_base=nfs_base)

            assert provider.paths.nfs_base == nfs_base
            assert provider.selfplay_dir == nfs_base / "selfplay"

    def test_capabilities(self):
        """Test capabilities for Lambda NFS."""
        from app.distributed.storage_provider import LambdaNFSProvider

        provider = LambdaNFSProvider()
        caps = provider.capabilities

        assert caps.has_shared_storage is True
        assert caps.skip_rsync_for_shared is True
        assert caps.supports_direct_nfs is True
        assert caps.ephemeral is False
        assert caps.max_sync_interval_seconds == 300  # Less urgent with shared storage
        assert caps.priority_in_fallback == 90  # High priority

    def test_has_shared_storage(self):
        """Test has_shared_storage property."""
        from app.distributed.storage_provider import LambdaNFSProvider

        provider = LambdaNFSProvider()
        assert provider.has_shared_storage is True

    def test_is_available(self):
        """Test is_available check."""
        from app.distributed.storage_provider import LambdaNFSProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            nfs_path = Path(tmpdir)

            with patch.object(LambdaNFSProvider, "NFS_BASE", nfs_path):
                assert LambdaNFSProvider.is_available() is True

    def test_verify_nfs_mount_success(self):
        """Test NFS verification success."""
        from app.distributed.storage_provider import LambdaNFSProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            nfs_base = Path(tmpdir)
            provider = LambdaNFSProvider(nfs_base=nfs_base)

            # Verify should succeed on writable directory
            assert provider.verify_nfs_mount(force=True) is True
            assert provider.is_nfs_healthy is True

    def test_verify_nfs_mount_failure(self):
        """Test NFS verification failure."""
        from app.distributed.storage_provider import LambdaNFSProvider

        provider = LambdaNFSProvider(nfs_base=Path("/nonexistent/path"))

        # Verify should fail on non-existent path
        assert provider.verify_nfs_mount(force=True) is False

    def test_verify_nfs_mount_caching(self):
        """Test that NFS verification uses caching."""
        from app.distributed.storage_provider import LambdaNFSProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            nfs_base = Path(tmpdir)
            provider = LambdaNFSProvider(nfs_base=nfs_base)

            # First call should verify
            provider.verify_nfs_mount(force=True)

            # Second call should use cache (no force)
            first_check_time = provider._last_nfs_check

            result = provider.verify_nfs_mount(force=False)

            # Time should be the same (cached)
            assert provider._last_nfs_check == first_check_time
            assert result is True

    def test_should_skip_rsync_to_lambda_node(self):
        """Test should_skip_rsync_to for Lambda node target."""
        from app.distributed.storage_provider import LambdaNFSProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            nfs_base = Path(tmpdir)
            provider = LambdaNFSProvider(nfs_base=nfs_base)

            with patch("app.distributed.storage_provider.get_host_provider") as mock:
                mock.return_value = "lambda"

                # Should skip rsync between Lambda nodes
                assert provider.should_skip_rsync_to("lambda-node-1") is True

    def test_should_skip_rsync_to_non_lambda_node(self):
        """Test should_skip_rsync_to for non-Lambda node target."""
        from app.distributed.storage_provider import LambdaNFSProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            nfs_base = Path(tmpdir)
            provider = LambdaNFSProvider(nfs_base=nfs_base)

            with patch("app.distributed.storage_provider.get_host_provider") as mock:
                mock.return_value = "vast"

                # Should not skip rsync to non-Lambda nodes
                assert provider.should_skip_rsync_to("vast-node-1") is False


# =============================================================================
# Test Provider Detection
# =============================================================================


class TestProviderDetection:
    """Tests for provider auto-detection."""

    def test_detect_with_env_override_lambda(self):
        """Test detection with RINGRIFT_STORAGE_PROVIDER=lambda."""
        from app.distributed.storage_provider import (
            detect_storage_provider,
            StorageProviderType,
        )

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "lambda"}):
            result = detect_storage_provider()
            assert result == StorageProviderType.LAMBDA_NFS

    def test_detect_with_env_override_vast(self):
        """Test detection with RINGRIFT_STORAGE_PROVIDER=vast."""
        from app.distributed.storage_provider import (
            detect_storage_provider,
            StorageProviderType,
        )

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "vast"}):
            result = detect_storage_provider()
            assert result == StorageProviderType.VAST_EPHEMERAL

    def test_detect_with_env_override_local(self):
        """Test detection with RINGRIFT_STORAGE_PROVIDER=local."""
        from app.distributed.storage_provider import (
            detect_storage_provider,
            StorageProviderType,
        )

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            result = detect_storage_provider()
            assert result == StorageProviderType.LOCAL

    def test_detect_lambda_from_filesystem(self):
        """Test detection of Lambda from filesystem."""
        from app.distributed.storage_provider import (
            LambdaNFSProvider,
            VastEphemeralProvider,
            detect_storage_provider,
            StorageProviderType,
        )

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(LambdaNFSProvider, "is_available", return_value=True), \
             patch.object(VastEphemeralProvider, "is_available", return_value=False):

            result = detect_storage_provider()
            assert result == StorageProviderType.LAMBDA_NFS

    def test_detect_vast_from_filesystem(self):
        """Test detection of Vast.ai from filesystem."""
        from app.distributed.storage_provider import (
            LambdaNFSProvider,
            VastEphemeralProvider,
            detect_storage_provider,
            StorageProviderType,
        )

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(LambdaNFSProvider, "is_available", return_value=False), \
             patch.object(VastEphemeralProvider, "is_available", return_value=True):

            result = detect_storage_provider()
            assert result == StorageProviderType.VAST_EPHEMERAL

    def test_detect_local_fallback(self):
        """Test detection falls back to local."""
        from app.distributed.storage_provider import (
            LambdaNFSProvider,
            VastEphemeralProvider,
            detect_storage_provider,
            StorageProviderType,
        )

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(LambdaNFSProvider, "is_available", return_value=False), \
             patch.object(VastEphemeralProvider, "is_available", return_value=False), \
             patch("socket.gethostname", return_value="local-dev"):

            with patch("app.distributed.storage_provider.get_host_provider", return_value="unknown"):
                result = detect_storage_provider()
                assert result == StorageProviderType.LOCAL


# =============================================================================
# Test get_storage_provider Factory
# =============================================================================


class TestGetStorageProvider:
    """Tests for get_storage_provider factory function."""

    def test_returns_provider(self):
        """Test that get_storage_provider returns a provider."""
        from app.distributed.storage_provider import (
            get_storage_provider,
            StorageProvider,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            provider = get_storage_provider()
            assert isinstance(provider, StorageProvider)

    def test_caches_provider(self):
        """Test that provider is cached."""
        from app.distributed.storage_provider import (
            get_storage_provider,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            provider1 = get_storage_provider()
            provider2 = get_storage_provider()
            assert provider1 is provider2

    def test_force_refresh(self):
        """Test force_refresh bypasses cache."""
        from app.distributed.storage_provider import (
            get_storage_provider,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            provider1 = get_storage_provider()
            provider2 = get_storage_provider(force_refresh=True)
            # With force_refresh, should be different instance
            assert provider1 is not provider2

    def test_explicit_provider_type(self):
        """Test with explicit provider type."""
        from app.distributed.storage_provider import (
            get_storage_provider,
            StorageProviderType,
            LocalStorageProvider,
            clear_provider_cache,
        )

        clear_provider_cache()

        provider = get_storage_provider(provider_type=StorageProviderType.LOCAL)
        assert isinstance(provider, LocalStorageProvider)


# =============================================================================
# Test SyncProtocolConfig (TransportConfig)
# =============================================================================


class TestSyncProtocolConfig:
    """Tests for SyncProtocolConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        from app.distributed.storage_provider import SyncProtocolConfig

        config = SyncProtocolConfig()

        assert config.enable_aria2 is True
        assert config.enable_ssh is True
        assert config.enable_p2p is True
        assert config.enable_gossip is True
        assert config.enable_bittorrent is True
        assert config.aria2_connections_per_server == 16
        assert config.bittorrent_threshold_bytes == 50_000_000

    def test_fallback_chain_for_small_files(self):
        """Test fallback chain for small files."""
        from app.distributed.storage_provider import SyncProtocolConfig

        config = SyncProtocolConfig()

        chain = config.get_fallback_chain_for_size(1_000_000)  # 1MB
        assert chain == ["aria2", "ssh", "p2p"]

    def test_fallback_chain_for_large_files(self):
        """Test fallback chain for large files."""
        from app.distributed.storage_provider import SyncProtocolConfig

        config = SyncProtocolConfig()

        chain = config.get_fallback_chain_for_size(100_000_000)  # 100MB
        assert "bittorrent" in chain
        assert chain[0] == "bittorrent"

    def test_fallback_chain_bittorrent_disabled(self):
        """Test fallback chain when BitTorrent is disabled."""
        from app.distributed.storage_provider import SyncProtocolConfig

        config = SyncProtocolConfig(enable_bittorrent=False)

        chain = config.get_fallback_chain_for_size(100_000_000)  # 100MB
        assert "bittorrent" not in chain

    def test_backward_compat_alias(self):
        """Test TransportConfig backward-compat alias."""
        from app.distributed.storage_provider import TransportConfig, SyncProtocolConfig

        assert TransportConfig is SyncProtocolConfig


# =============================================================================
# Test get_optimal_sync_protocol_config
# =============================================================================


class TestGetOptimalSyncProtocolConfig:
    """Tests for get_optimal_sync_protocol_config function."""

    def test_returns_config(self):
        """Test that function returns a SyncProtocolConfig."""
        from app.distributed.storage_provider import (
            get_optimal_sync_protocol_config,
            SyncProtocolConfig,
            LocalStorageProvider,
        )

        provider = LocalStorageProvider()
        config = get_optimal_sync_protocol_config(provider)

        assert isinstance(config, SyncProtocolConfig)

    def test_shared_storage_config(self):
        """Test config for shared storage provider."""
        from app.distributed.storage_provider import (
            get_optimal_sync_protocol_config,
            LambdaNFSProvider,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LambdaNFSProvider(nfs_base=Path(tmpdir))
            config = get_optimal_sync_protocol_config(provider)

            # NFS nodes should disable SSH rsync
            assert config.enable_ssh is False
            assert config.enable_gossip is False

    def test_ephemeral_config(self):
        """Test config for ephemeral provider."""
        from app.distributed.storage_provider import (
            get_optimal_sync_protocol_config,
            VastEphemeralProvider,
        )

        provider = VastEphemeralProvider()
        config = get_optimal_sync_protocol_config(provider)

        # Ephemeral nodes should use aggressive sync
        assert config.enable_gossip is True
        assert "aria2" in config.fallback_chain

    def test_backward_compat_alias(self):
        """Test get_optimal_transport_config backward-compat alias."""
        from app.distributed.storage_provider import (
            get_optimal_transport_config,
            get_optimal_sync_protocol_config,
        )

        assert get_optimal_transport_config is get_optimal_sync_protocol_config


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_selfplay_dir(self):
        """Test get_selfplay_dir function."""
        from app.distributed.storage_provider import (
            get_selfplay_dir,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            result = get_selfplay_dir()
            assert isinstance(result, Path)
            assert "selfplay" in str(result)

    def test_get_models_dir(self):
        """Test get_models_dir function."""
        from app.distributed.storage_provider import (
            get_models_dir,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            result = get_models_dir()
            assert isinstance(result, Path)
            assert "models" in str(result)

    def test_get_training_dir(self):
        """Test get_training_dir function."""
        from app.distributed.storage_provider import (
            get_training_dir,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            result = get_training_dir()
            assert isinstance(result, Path)
            assert "training" in str(result)

    def test_get_scratch_dir(self):
        """Test get_scratch_dir function."""
        from app.distributed.storage_provider import (
            get_scratch_dir,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            result = get_scratch_dir()
            assert isinstance(result, Path)

    def test_should_sync_to_node(self):
        """Test should_sync_to_node function."""
        from app.distributed.storage_provider import (
            should_sync_to_node,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            # Local provider always needs sync
            assert should_sync_to_node("any-node") is True

    def test_is_nfs_available(self):
        """Test is_nfs_available function."""
        from app.distributed.storage_provider import (
            is_nfs_available,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            # Local provider doesn't support NFS
            assert is_nfs_available() is False

    def test_verify_nfs_health(self):
        """Test verify_nfs_health function."""
        from app.distributed.storage_provider import (
            verify_nfs_health,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            # Local provider returns False
            assert verify_nfs_health() is False

    def test_get_aria2_sources(self):
        """Test get_aria2_sources function."""
        from app.distributed.storage_provider import get_aria2_sources

        # Should return a list (may be empty)
        sources = get_aria2_sources()
        assert isinstance(sources, list)


# =============================================================================
# Test clear_provider_cache
# =============================================================================


class TestClearProviderCache:
    """Tests for clear_provider_cache function."""

    def test_clears_cache(self):
        """Test that clear_provider_cache clears the cache."""
        from app.distributed.storage_provider import (
            get_storage_provider,
            clear_provider_cache,
        )

        clear_provider_cache()

        with patch.dict(os.environ, {"RINGRIFT_STORAGE_PROVIDER": "local"}):
            provider1 = get_storage_provider()

            clear_provider_cache()

            provider2 = get_storage_provider()

            # Should be different instances after cache clear
            assert provider1 is not provider2
