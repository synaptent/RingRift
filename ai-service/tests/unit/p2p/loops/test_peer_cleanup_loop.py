"""Tests for PeerCleanupLoop.

December 2025: Tests for automatic stale peer cleanup functionality.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.loops.peer_cleanup_loop import (
    CleanupStats,
    PeerCleanupConfig,
    PeerCleanupLoop,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_peers() -> dict[str, dict[str, Any]]:
    """Create mock peers with various staleness levels."""
    now = time.time()
    return {
        "alive-1": {
            "node_id": "alive-1",
            "last_heartbeat": now - 30,  # 30 seconds ago - alive
            "retired": False,
            "is_alive": True,
        },
        "alive-2": {
            "node_id": "alive-2",
            "last_heartbeat": now - 60,  # 1 minute ago - alive
            "retired": False,
            "is_alive": True,
        },
        "stale-tier1": {
            "node_id": "stale-tier1",
            "last_heartbeat": now - 4000,  # ~1.1 hours - tier 1
            "retired": False,
            "is_alive": False,
        },
        "stale-tier2": {
            "node_id": "stale-tier2",
            "last_heartbeat": now - 25000,  # ~7 hours - tier 2
            "retired": False,
            "is_alive": False,
        },
        "retired-old": {
            "node_id": "retired-old",
            "last_heartbeat": now - 100000,  # ~28 hours - tier 3
            "retired": True,
            "is_alive": False,
        },
    }


@pytest.fixture
def config() -> PeerCleanupConfig:
    """Create test configuration with shorter thresholds."""
    return PeerCleanupConfig(
        cleanup_interval_seconds=60.0,
        tier1_stale_seconds=3600.0,  # 1 hour
        tier2_purge_seconds=21600.0,  # 6 hours
        tier3_cache_purge_seconds=86400.0,  # 24 hours
        max_purge_per_cycle=10,
        enabled=True,
        dry_run=False,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestPeerCleanupConfig:
    """Tests for PeerCleanupConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PeerCleanupConfig()
        assert config.cleanup_interval_seconds == 300.0
        assert config.tier1_stale_seconds == 3600.0
        assert config.tier2_purge_seconds == 21600.0
        assert config.tier3_cache_purge_seconds == 86400.0
        assert config.max_purge_per_cycle == 20
        assert config.enabled is True
        assert config.dry_run is False

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "RINGRIFT_PEER_CLEANUP_INTERVAL": "120",
                "RINGRIFT_PEER_STALE_THRESHOLD": "1800",
                "RINGRIFT_PEER_PURGE_THRESHOLD": "7200",
                "RINGRIFT_PEER_CLEANUP_ENABLED": "false",
                "RINGRIFT_PEER_CLEANUP_DRY_RUN": "true",
            },
        ):
            config = PeerCleanupConfig()
            assert config.cleanup_interval_seconds == 120.0
            assert config.tier1_stale_seconds == 1800.0
            assert config.tier2_purge_seconds == 7200.0
            assert config.enabled is False
            assert config.dry_run is True

    def test_config_validation_interval(self) -> None:
        """Test validation of cleanup_interval_seconds."""
        with pytest.raises(ValueError, match="cleanup_interval_seconds must be > 0"):
            PeerCleanupConfig(cleanup_interval_seconds=0)

    def test_config_validation_tier_order(self) -> None:
        """Test validation that tiers are in correct order."""
        with pytest.raises(
            ValueError, match="tier2_purge_seconds must be > tier1_stale_seconds"
        ):
            PeerCleanupConfig(
                tier1_stale_seconds=10000,
                tier2_purge_seconds=5000,
            )

    def test_config_validation_max_purge(self) -> None:
        """Test validation of max_purge_per_cycle."""
        with pytest.raises(ValueError, match="max_purge_per_cycle must be > 0"):
            PeerCleanupConfig(max_purge_per_cycle=0)


# =============================================================================
# Loop Tests
# =============================================================================


class TestPeerCleanupLoop:
    """Tests for PeerCleanupLoop."""

    def test_initialization(self, config: PeerCleanupConfig) -> None:
        """Test loop initialization."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(return_value=True),
            config=config,
        )
        assert loop.name == "peer_cleanup"
        assert loop.interval == 60.0
        assert loop.enabled is True

    def test_initialization_defaults(self) -> None:
        """Test loop initialization with default config."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(return_value=True),
        )
        assert loop.config.cleanup_interval_seconds == 300.0

    @pytest.mark.asyncio
    async def test_run_once_no_peers(self, config: PeerCleanupConfig) -> None:
        """Test run_once with no peers."""
        purge_mock = AsyncMock(return_value=True)
        emit_mock = MagicMock()

        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=purge_mock,
            emit_event=emit_mock,
            config=config,
        )

        await loop._run_once()

        purge_mock.assert_not_called()
        emit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_all_alive(self, config: PeerCleanupConfig) -> None:
        """Test run_once when all peers are alive."""
        now = time.time()
        peers = {
            "alive-1": {"last_heartbeat": now - 30, "retired": False, "is_alive": True},
            "alive-2": {"last_heartbeat": now - 60, "retired": False, "is_alive": True},
        }
        purge_mock = AsyncMock(return_value=True)
        emit_mock = MagicMock()

        loop = PeerCleanupLoop(
            get_all_peers=lambda: peers,
            purge_peer=purge_mock,
            emit_event=emit_mock,
            config=config,
        )

        await loop._run_once()

        purge_mock.assert_not_called()
        emit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_with_stale_peers(
        self, mock_peers: dict[str, dict[str, Any]], config: PeerCleanupConfig
    ) -> None:
        """Test run_once purges stale peers."""
        purged_peers: list[str] = []

        async def mock_purge(node_id: str) -> bool:
            purged_peers.append(node_id)
            return True

        emit_mock = MagicMock()

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(mock_peers),
            purge_peer=mock_purge,
            emit_event=emit_mock,
            config=config,
        )

        await loop._run_once()

        # Should purge tier2 and tier3 peers
        assert "stale-tier2" in purged_peers
        assert "retired-old" in purged_peers
        # Should NOT purge tier1 (candidate only) or alive peers
        assert "stale-tier1" not in purged_peers
        assert "alive-1" not in purged_peers
        assert "alive-2" not in purged_peers

        # Should emit event
        emit_mock.assert_called_once()
        call_args = emit_mock.call_args
        assert call_args[0][0] == "STALE_PEERS_PURGED"
        event_data = call_args[0][1]
        assert event_data["purged_count"] == 2

    @pytest.mark.asyncio
    async def test_run_once_dry_run(
        self, mock_peers: dict[str, dict[str, Any]], config: PeerCleanupConfig
    ) -> None:
        """Test dry run mode doesn't actually purge."""
        config.dry_run = True
        purge_mock = AsyncMock(return_value=True)
        emit_mock = MagicMock()

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(mock_peers),
            purge_peer=purge_mock,
            emit_event=emit_mock,
            config=config,
        )

        await loop._run_once()

        # Purge should not be called in dry run mode
        purge_mock.assert_not_called()
        emit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_disabled(
        self, mock_peers: dict[str, dict[str, Any]], config: PeerCleanupConfig
    ) -> None:
        """Test disabled loop doesn't run."""
        config.enabled = False
        purge_mock = AsyncMock(return_value=True)

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(mock_peers),
            purge_peer=purge_mock,
            config=config,
        )

        await loop._run_once()

        purge_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_max_purge_limit(
        self, config: PeerCleanupConfig
    ) -> None:
        """Test max_purge_per_cycle is respected."""
        now = time.time()
        # Create many stale peers
        peers = {
            f"stale-{i}": {
                "last_heartbeat": now - 100000,  # Very stale
                "retired": False,
                "is_alive": False,
            }
            for i in range(50)
        }

        config.max_purge_per_cycle = 5
        purged_peers: list[str] = []

        async def mock_purge(node_id: str) -> bool:
            purged_peers.append(node_id)
            return True

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(peers),
            purge_peer=mock_purge,
            config=config,
        )

        await loop._run_once()

        # Should only purge up to max_purge_per_cycle
        assert len(purged_peers) == 5

    @pytest.mark.asyncio
    async def test_run_once_purge_failure(
        self, mock_peers: dict[str, dict[str, Any]], config: PeerCleanupConfig
    ) -> None:
        """Test handling of purge failures."""
        async def mock_purge(node_id: str) -> bool:
            # Fail for retired-old
            return node_id != "retired-old"

        emit_mock = MagicMock()

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(mock_peers),
            purge_peer=mock_purge,
            emit_event=emit_mock,
            config=config,
        )

        await loop._run_once()

        # Event should still be emitted with partial success
        emit_mock.assert_called_once()
        event_data = emit_mock.call_args[0][1]
        assert event_data["purged_count"] == 1  # Only stale-tier2 succeeded

    @pytest.mark.asyncio
    async def test_run_once_purge_exception(
        self, config: PeerCleanupConfig
    ) -> None:
        """Test handling of purge exceptions."""
        now = time.time()
        peers = {
            "stale-1": {
                "last_heartbeat": now - 100000,
                "retired": False,
                "is_alive": False,
            },
        }

        async def mock_purge(node_id: str) -> bool:
            raise RuntimeError("Purge failed")

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(peers),
            purge_peer=mock_purge,
            config=config,
        )

        # Should not raise
        await loop._run_once()


# =============================================================================
# Classification Tests
# =============================================================================


class TestPeerClassification:
    """Tests for peer staleness classification."""

    def test_classify_alive_peer(self, config: PeerCleanupConfig) -> None:
        """Test classification of alive peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        now = time.time()
        peer = {"last_heartbeat": now - 30, "retired": False}

        tier = loop._classify_stale_peer(peer, now)
        assert tier is None

    def test_classify_tier1_peer(self, config: PeerCleanupConfig) -> None:
        """Test classification of tier 1 (stale candidate) peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        now = time.time()
        # Just over 1 hour stale
        peer = {"last_heartbeat": now - 4000, "retired": False}

        tier = loop._classify_stale_peer(peer, now)
        assert tier == 1

    def test_classify_tier2_peer(self, config: PeerCleanupConfig) -> None:
        """Test classification of tier 2 (purge from memory) peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        now = time.time()
        # Over 6 hours stale
        peer = {"last_heartbeat": now - 25000, "retired": False}

        tier = loop._classify_stale_peer(peer, now)
        assert tier == 2

    def test_classify_tier3_peer(self, config: PeerCleanupConfig) -> None:
        """Test classification of tier 3 (retired + very stale) peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        now = time.time()
        # Over 24 hours stale AND retired
        peer = {"last_heartbeat": now - 100000, "retired": True}

        tier = loop._classify_stale_peer(peer, now)
        assert tier == 3

    def test_classify_no_heartbeat(self, config: PeerCleanupConfig) -> None:
        """Test classification of peer with no heartbeat."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        peer = {"last_heartbeat": 0, "retired": False}

        tier = loop._classify_stale_peer(peer, time.time())
        assert tier == 2  # Treated as very stale


# =============================================================================
# Statistics Tests
# =============================================================================


class TestCleanupStats:
    """Tests for cleanup statistics."""

    def test_initial_stats(self) -> None:
        """Test initial statistics are zero."""
        stats = CleanupStats()
        assert stats.total_purged == 0
        assert stats.tier1_detected == 0
        assert stats.tier2_purged == 0
        assert stats.tier3_purged == 0
        assert stats.cycles_run == 0

    def test_get_cleanup_stats(self, config: PeerCleanupConfig) -> None:
        """Test get_cleanup_stats method."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        stats = loop.get_cleanup_stats()
        assert "total_purged" in stats
        assert "tier1_detected" in stats
        assert "tier2_purged" in stats
        assert "tier3_purged" in stats
        assert "config" in stats
        assert stats["config"]["interval_seconds"] == 60.0

    @pytest.mark.asyncio
    async def test_stats_updated_after_cleanup(
        self, mock_peers: dict[str, dict[str, Any]], config: PeerCleanupConfig
    ) -> None:
        """Test statistics are updated after cleanup cycle."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(mock_peers),
            purge_peer=AsyncMock(return_value=True),
            config=config,
        )

        await loop._run_once()

        stats = loop.get_cleanup_stats()
        assert stats["total_purged"] > 0
        assert stats["cycles_run"] == 1
        assert stats["last_cleanup_time"] > 0

    def test_reset_stats(self, config: PeerCleanupConfig) -> None:
        """Test reset_stats method."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        # Manually set some stats
        loop._stats.total_purged = 100
        loop._stats.cycles_run = 10

        loop.reset_stats()

        stats = loop.get_cleanup_stats()
        assert stats["total_purged"] == 0
        assert stats["cycles_run"] == 0


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_peer_heartbeat_dict(self, config: PeerCleanupConfig) -> None:
        """Test _get_peer_heartbeat with dict peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        peer = {"last_heartbeat": 12345.0}
        assert loop._get_peer_heartbeat(peer) == 12345.0

    def test_get_peer_heartbeat_object(self, config: PeerCleanupConfig) -> None:
        """Test _get_peer_heartbeat with object peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        class MockPeer:
            last_heartbeat = 67890.0

        assert loop._get_peer_heartbeat(MockPeer()) == 67890.0

    def test_is_peer_retired_dict(self, config: PeerCleanupConfig) -> None:
        """Test _is_peer_retired with dict peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        assert loop._is_peer_retired({"retired": True}) is True
        assert loop._is_peer_retired({"retired": False}) is False
        assert loop._is_peer_retired({}) is False

    def test_is_peer_alive_dict(self, config: PeerCleanupConfig) -> None:
        """Test _is_peer_alive with dict peer."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        assert loop._is_peer_alive({"is_alive": True}) is True
        assert loop._is_peer_alive({"is_alive": False}) is False

    def test_is_peer_alive_fallback(self, config: PeerCleanupConfig) -> None:
        """Test _is_peer_alive fallback to heartbeat check."""
        loop = PeerCleanupLoop(
            get_all_peers=lambda: {},
            purge_peer=AsyncMock(),
            config=config,
        )

        now = time.time()
        # Recent heartbeat = alive
        assert loop._is_peer_alive({"last_heartbeat": now - 30}) is True
        # Old heartbeat = not alive
        assert loop._is_peer_alive({"last_heartbeat": now - 120}) is False


# =============================================================================
# Sync Callback Tests
# =============================================================================


class TestSyncCallback:
    """Tests for sync purge_peer callback support."""

    @pytest.mark.asyncio
    async def test_sync_purge_callback(
        self, mock_peers: dict[str, dict[str, Any]], config: PeerCleanupConfig
    ) -> None:
        """Test that sync purge_peer callback is supported."""
        purged_peers: list[str] = []

        def sync_purge(node_id: str) -> bool:
            purged_peers.append(node_id)
            return True

        loop = PeerCleanupLoop(
            get_all_peers=lambda: dict(mock_peers),
            purge_peer=sync_purge,
            config=config,
        )

        await loop._run_once()

        # Should still work with sync callback
        assert len(purged_peers) > 0
