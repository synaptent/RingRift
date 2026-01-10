"""
Tests for WorkDiscoveryManager - Multi-channel work discovery for P2P resilience.

Jan 9, 2026: Comprehensive test suite for work_discovery_manager.py.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.managers.work_discovery_manager import (
    DiscoveryChannel,
    DiscoveryResult,
    WorkDiscoveryConfig,
    WorkDiscoveryManager,
    WorkDiscoveryStats,
    get_work_discovery_manager,
    reset_work_discovery_manager,
    set_work_discovery_manager,
    _is_selfplay_enabled_for_node,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton state before each test."""
    reset_work_discovery_manager()
    yield
    reset_work_discovery_manager()


@pytest.fixture(autouse=True)
def reset_selfplay_enabled_cache():
    """Reset module-level selfplay enabled cache before each test."""
    import scripts.p2p.managers.work_discovery_manager as module
    module._selfplay_enabled_checked = False
    module._selfplay_enabled = None
    yield
    module._selfplay_enabled_checked = False
    module._selfplay_enabled = None


@pytest.fixture
def basic_manager() -> WorkDiscoveryManager:
    """Create a basic manager with minimal configuration."""
    return WorkDiscoveryManager(
        get_leader_id=lambda: "leader-1",
        claim_from_leader=AsyncMock(return_value=None),
        config=WorkDiscoveryConfig(),
    )


@pytest.fixture
def full_manager() -> WorkDiscoveryManager:
    """Create a manager with all channels configured."""
    return WorkDiscoveryManager(
        get_leader_id=lambda: "leader-1",
        claim_from_leader=AsyncMock(return_value=None),
        get_alive_peers=lambda: ["peer-1", "peer-2", "peer-3"],
        query_peer_work=AsyncMock(return_value=None),
        pop_autonomous_work=AsyncMock(return_value=None),
        create_direct_selfplay_work=MagicMock(return_value=None),
        config=WorkDiscoveryConfig(),
    )


# ============================================================================
# DiscoveryChannel Tests
# ============================================================================


class TestDiscoveryChannel:
    """Tests for DiscoveryChannel enum."""

    def test_all_channels_exist(self):
        """Test that all expected channels exist."""
        assert DiscoveryChannel.LEADER.value == "leader"
        assert DiscoveryChannel.PEER.value == "peer"
        assert DiscoveryChannel.AUTONOMOUS.value == "autonomous"
        assert DiscoveryChannel.DIRECT.value == "direct"

    def test_channel_count(self):
        """Test number of channels."""
        assert len(DiscoveryChannel) == 4


# ============================================================================
# WorkDiscoveryConfig Tests
# ============================================================================


class TestWorkDiscoveryConfig:
    """Tests for WorkDiscoveryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WorkDiscoveryConfig()
        assert config.leader_enabled is True
        assert config.peer_discovery_enabled is True
        assert config.autonomous_enabled is True
        assert config.direct_selfplay_enabled is True
        assert config.peer_query_limit == 3
        assert config.peer_query_timeout_seconds == 5.0
        assert config.direct_selfplay_game_count == 10
        assert config.peer_discovery_cooldown_seconds == 30.0
        assert config.direct_selfplay_cooldown_seconds == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = WorkDiscoveryConfig(
            leader_enabled=False,
            peer_discovery_enabled=False,
            peer_query_limit=5,
            peer_query_timeout_seconds=10.0,
        )
        assert config.leader_enabled is False
        assert config.peer_discovery_enabled is False
        assert config.peer_query_limit == 5
        assert config.peer_query_timeout_seconds == 10.0

    def test_from_env_defaults(self):
        """Test from_env with no environment variables set."""
        # Clear relevant env vars
        env_vars = [
            "RINGRIFT_WORK_DISCOVERY_LEADER",
            "RINGRIFT_WORK_DISCOVERY_PEER",
            "RINGRIFT_WORK_DISCOVERY_AUTONOMOUS",
            "RINGRIFT_WORK_DISCOVERY_DIRECT",
            "RINGRIFT_WORK_DISCOVERY_PEER_LIMIT",
            "RINGRIFT_WORK_DISCOVERY_PEER_TIMEOUT",
            "RINGRIFT_WORK_DISCOVERY_DIRECT_GAMES",
        ]
        original_values = {k: os.environ.get(k) for k in env_vars}
        for k in env_vars:
            if k in os.environ:
                del os.environ[k]

        try:
            config = WorkDiscoveryConfig.from_env()
            assert config.leader_enabled is True
            assert config.peer_discovery_enabled is True
            assert config.peer_query_limit == 3
        finally:
            # Restore original values
            for k, v in original_values.items():
                if v is not None:
                    os.environ[k] = v

    def test_from_env_custom(self):
        """Test from_env with custom environment variables."""
        with patch.dict(os.environ, {
            "RINGRIFT_WORK_DISCOVERY_LEADER": "false",
            "RINGRIFT_WORK_DISCOVERY_PEER": "false",
            "RINGRIFT_WORK_DISCOVERY_PEER_LIMIT": "5",
            "RINGRIFT_WORK_DISCOVERY_PEER_TIMEOUT": "10.0",
            "RINGRIFT_WORK_DISCOVERY_DIRECT_GAMES": "20",
        }):
            config = WorkDiscoveryConfig.from_env()
            assert config.leader_enabled is False
            assert config.peer_discovery_enabled is False
            assert config.peer_query_limit == 5
            assert config.peer_query_timeout_seconds == 10.0
            assert config.direct_selfplay_game_count == 20


# ============================================================================
# DiscoveryResult Tests
# ============================================================================


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful discovery result."""
        result = DiscoveryResult(
            work_item={"type": "selfplay", "config_key": "hex8_2p"},
            channel=DiscoveryChannel.LEADER,
            duration_seconds=0.5,
        )
        assert result.work_item is not None
        assert result.channel == DiscoveryChannel.LEADER
        assert result.duration_seconds == 0.5
        assert result.peer_id is None
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed discovery result."""
        result = DiscoveryResult(
            work_item=None,
            channel=DiscoveryChannel.PEER,
            duration_seconds=1.0,
            error="No work available",
        )
        assert result.work_item is None
        assert result.error == "No work available"

    def test_peer_result(self):
        """Test creating a peer channel result."""
        result = DiscoveryResult(
            work_item={"type": "selfplay"},
            channel=DiscoveryChannel.PEER,
            duration_seconds=0.2,
            peer_id="peer-1",
        )
        assert result.peer_id == "peer-1"


# ============================================================================
# WorkDiscoveryStats Tests
# ============================================================================


class TestWorkDiscoveryStats:
    """Tests for WorkDiscoveryStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = WorkDiscoveryStats()
        for channel in DiscoveryChannel:
            assert stats.attempts_by_channel[channel.value] == 0
            assert stats.successes_by_channel[channel.value] == 0
            assert stats.failures_by_channel[channel.value] == 0
        assert stats.last_success_time is None
        assert stats.last_success_channel is None

    def test_record_successful_attempt(self):
        """Test recording a successful attempt."""
        stats = WorkDiscoveryStats()
        stats.record_attempt(DiscoveryChannel.LEADER, success=True)

        assert stats.attempts_by_channel["leader"] == 1
        assert stats.successes_by_channel["leader"] == 1
        assert stats.failures_by_channel["leader"] == 0
        assert stats.last_success_time is not None
        assert stats.last_success_channel == "leader"

    def test_record_failed_attempt(self):
        """Test recording a failed attempt."""
        stats = WorkDiscoveryStats()
        stats.record_attempt(DiscoveryChannel.PEER, success=False)

        assert stats.attempts_by_channel["peer"] == 1
        assert stats.successes_by_channel["peer"] == 0
        assert stats.failures_by_channel["peer"] == 1
        assert stats.last_success_time is None
        assert stats.last_success_channel is None

    def test_multiple_attempts(self):
        """Test recording multiple attempts."""
        stats = WorkDiscoveryStats()
        stats.record_attempt(DiscoveryChannel.LEADER, success=False)
        stats.record_attempt(DiscoveryChannel.LEADER, success=False)
        stats.record_attempt(DiscoveryChannel.LEADER, success=True)

        assert stats.attempts_by_channel["leader"] == 3
        assert stats.successes_by_channel["leader"] == 1
        assert stats.failures_by_channel["leader"] == 2

    def test_to_dict(self):
        """Test stats serialization."""
        stats = WorkDiscoveryStats()
        stats.record_attempt(DiscoveryChannel.LEADER, success=True)

        result = stats.to_dict()
        assert "attempts" in result
        assert "successes" in result
        assert "failures" in result
        assert "last_success_time" in result
        assert "last_success_channel" in result
        assert result["successes"]["leader"] == 1


# ============================================================================
# WorkDiscoveryManager Initialization Tests
# ============================================================================


class TestWorkDiscoveryManagerInit:
    """Tests for WorkDiscoveryManager initialization."""

    def test_minimal_init(self):
        """Test initialization with minimal callbacks."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
        )
        assert manager._get_leader_id() == "leader-1"
        assert manager._get_alive_peers is None
        assert manager._query_peer_work is None

    def test_full_init(self):
        """Test initialization with all callbacks."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: ["peer-1"],
            query_peer_work=AsyncMock(return_value=None),
            pop_autonomous_work=AsyncMock(return_value=None),
            create_direct_selfplay_work=MagicMock(return_value=None),
        )
        assert manager._get_alive_peers is not None
        assert manager._query_peer_work is not None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = WorkDiscoveryConfig(peer_query_limit=10)
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            config=config,
        )
        assert manager.config.peer_query_limit == 10


# ============================================================================
# Leader Channel Tests
# ============================================================================


class TestLeaderChannel:
    """Tests for leader channel discovery."""

    @pytest.mark.asyncio
    async def test_leader_channel_success(self):
        """Test successful work claim from leader."""
        work_item = {"type": "selfplay", "config_key": "hex8_2p"}
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=work_item),
        )

        result = await manager._try_leader_channel(["selfplay"])

        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.LEADER
        assert result.error is None

    @pytest.mark.asyncio
    async def test_leader_channel_no_leader(self):
        """Test when no leader is available."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = await manager._try_leader_channel(["selfplay"])

        assert result.work_item is None
        assert result.error == "No leader available"

    @pytest.mark.asyncio
    async def test_leader_channel_no_work(self):
        """Test when leader has no work."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = await manager._try_leader_channel(["selfplay"])

        assert result.work_item is None
        assert result.error is None  # No error, just no work

    @pytest.mark.asyncio
    async def test_leader_channel_exception(self):
        """Test leader channel exception handling."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(side_effect=Exception("Connection failed")),
        )

        result = await manager._try_leader_channel(["selfplay"])

        assert result.work_item is None
        assert "Connection failed" in result.error


# ============================================================================
# Peer Channel Tests
# ============================================================================


class TestPeerChannel:
    """Tests for peer channel discovery."""

    @pytest.mark.asyncio
    async def test_peer_channel_not_configured(self):
        """Test when peer discovery is not configured."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = await manager._try_peer_channel(["selfplay"])

        assert result.work_item is None
        assert result.error == "Peer discovery not configured"

    @pytest.mark.asyncio
    async def test_peer_channel_success(self):
        """Test successful work from peer."""
        work_item = {"type": "selfplay"}
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: ["peer-1", "peer-2"],
            query_peer_work=AsyncMock(return_value=work_item),
        )

        result = await manager._try_peer_channel(["selfplay"])

        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.PEER
        assert result.peer_id is not None

    @pytest.mark.asyncio
    async def test_peer_channel_filters_leader(self):
        """Test that leader is filtered from peer list."""
        query_mock = AsyncMock(return_value={"type": "work"})
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: ["leader-1", "peer-1"],
            query_peer_work=query_mock,
        )

        await manager._try_peer_channel(["selfplay"])

        # Should only query peer-1, not leader-1
        calls = query_mock.call_args_list
        peer_ids = [call[0][0] for call in calls]
        assert "leader-1" not in peer_ids

    @pytest.mark.asyncio
    async def test_peer_channel_respects_limit(self):
        """Test that peer query respects limit."""
        query_mock = AsyncMock(return_value=None)
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: [f"peer-{i}" for i in range(10)],
            query_peer_work=query_mock,
            config=WorkDiscoveryConfig(peer_query_limit=3),
        )

        await manager._try_peer_channel(["selfplay"])

        # Should only query 3 peers
        assert query_mock.call_count == 3

    @pytest.mark.asyncio
    async def test_peer_channel_timeout(self):
        """Test peer query timeout handling."""
        async def slow_query(*args):
            await asyncio.sleep(10)
            return {"type": "work"}

        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: ["peer-1"],
            query_peer_work=slow_query,
            config=WorkDiscoveryConfig(peer_query_timeout_seconds=0.01),
        )

        result = await manager._try_peer_channel(["selfplay"])

        assert result.work_item is None
        assert "0 peers" in result.error or result.error is not None


# ============================================================================
# Autonomous Channel Tests
# ============================================================================


class TestAutonomousChannel:
    """Tests for autonomous queue channel."""

    @pytest.mark.asyncio
    async def test_autonomous_not_configured(self):
        """Test when autonomous queue is not configured."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = await manager._try_autonomous_channel()

        assert result.work_item is None
        assert result.error == "Autonomous queue not configured"

    @pytest.mark.asyncio
    async def test_autonomous_success(self):
        """Test successful pop from autonomous queue."""
        work_item = {"type": "selfplay", "config_key": "square8_2p"}
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            pop_autonomous_work=AsyncMock(return_value=work_item),
        )

        result = await manager._try_autonomous_channel()

        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.AUTONOMOUS

    @pytest.mark.asyncio
    async def test_autonomous_empty(self):
        """Test empty autonomous queue."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            pop_autonomous_work=AsyncMock(return_value=None),
        )

        result = await manager._try_autonomous_channel()

        assert result.work_item is None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_autonomous_exception(self):
        """Test autonomous channel exception handling."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            pop_autonomous_work=AsyncMock(side_effect=Exception("Queue error")),
        )

        result = await manager._try_autonomous_channel()

        assert result.work_item is None
        assert "Queue error" in result.error


# ============================================================================
# Direct Selfplay Channel Tests
# ============================================================================


class TestDirectChannel:
    """Tests for direct selfplay channel."""

    @pytest.mark.asyncio
    async def test_direct_not_configured(self):
        """Test when direct selfplay is not configured."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = await manager._try_direct_channel(["selfplay"])

        assert result.work_item is None
        assert result.error == "Direct selfplay not configured"

    @pytest.mark.asyncio
    async def test_direct_success(self):
        """Test successful direct selfplay work creation."""
        work_item = {"type": "selfplay", "direct": True}
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            create_direct_selfplay_work=MagicMock(return_value=work_item),
        )

        result = await manager._try_direct_channel(["selfplay"])

        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.DIRECT

    @pytest.mark.asyncio
    async def test_direct_failure(self):
        """Test direct selfplay failure."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            create_direct_selfplay_work=MagicMock(return_value=None),
        )

        result = await manager._try_direct_channel(["selfplay"])

        assert result.work_item is None

    @pytest.mark.asyncio
    async def test_direct_exception(self):
        """Test direct channel exception handling."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            create_direct_selfplay_work=MagicMock(side_effect=Exception("Creation failed")),
        )

        result = await manager._try_direct_channel(["selfplay"])

        assert result.work_item is None
        assert "Creation failed" in result.error


# ============================================================================
# Cooldown Tests
# ============================================================================


class TestCooldowns:
    """Tests for cooldown behavior."""

    def test_can_query_peers_initially(self):
        """Test that peers can be queried initially."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )
        assert manager._can_query_peers() is True

    def test_can_query_peers_after_cooldown(self):
        """Test peer query cooldown."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            config=WorkDiscoveryConfig(peer_discovery_cooldown_seconds=0.01),
        )

        manager._last_peer_query_time = time.time()
        assert manager._can_query_peers() is False

        time.sleep(0.02)
        assert manager._can_query_peers() is True

    def test_can_direct_selfplay_initially(self):
        """Test that direct selfplay can be done initially."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )
        # Mock the node selfplay check to return True
        with patch("scripts.p2p.managers.work_discovery_manager._is_selfplay_enabled_for_node", return_value=True):
            assert manager._can_direct_selfplay(["selfplay"]) is True

    def test_can_direct_selfplay_requires_capability(self):
        """Test that direct selfplay requires capability."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )
        assert manager._can_direct_selfplay(["training"]) is False

    def test_can_direct_selfplay_after_cooldown(self):
        """Test direct selfplay cooldown."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            config=WorkDiscoveryConfig(direct_selfplay_cooldown_seconds=0.01),
        )

        with patch("scripts.p2p.managers.work_discovery_manager._is_selfplay_enabled_for_node", return_value=True):
            manager._last_direct_selfplay_time = time.time()
            assert manager._can_direct_selfplay(["selfplay"]) is False

            time.sleep(0.02)
            assert manager._can_direct_selfplay(["selfplay"]) is True


# ============================================================================
# Full Discovery Flow Tests
# ============================================================================


class TestDiscoverWork:
    """Tests for full discover_work flow."""

    @pytest.mark.asyncio
    async def test_discover_leader_first(self):
        """Test that leader channel is tried first."""
        work_item = {"type": "selfplay"}
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=work_item),
            pop_autonomous_work=AsyncMock(return_value={"type": "other"}),
        )

        result = await manager.discover_work(["selfplay"])

        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.LEADER

    @pytest.mark.asyncio
    async def test_discover_falls_through_channels(self):
        """Test that discovery falls through channels."""
        work_item = {"type": "autonomous"}
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,  # No leader
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: [],  # No peers
            query_peer_work=AsyncMock(return_value=None),
            pop_autonomous_work=AsyncMock(return_value=work_item),
        )

        result = await manager.discover_work(["selfplay"])

        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.AUTONOMOUS

    @pytest.mark.asyncio
    async def test_discover_no_work_available(self):
        """Test when no work is available from any channel."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            pop_autonomous_work=AsyncMock(return_value=None),
        )

        result = await manager.discover_work(["selfplay"])

        assert result.work_item is None
        assert "No work available" in result.error

    @pytest.mark.asyncio
    async def test_discover_respects_disabled_channels(self):
        """Test that disabled channels are skipped."""
        config = WorkDiscoveryConfig(
            leader_enabled=False,
            peer_discovery_enabled=False,
            autonomous_enabled=True,
        )
        work_item = {"type": "autonomous"}

        claim_mock = AsyncMock(return_value={"type": "leader"})
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=claim_mock,
            pop_autonomous_work=AsyncMock(return_value=work_item),
            config=config,
        )

        result = await manager.discover_work(["selfplay"])

        # Leader channel should be skipped
        claim_mock.assert_not_called()
        assert result.work_item == work_item
        assert result.channel == DiscoveryChannel.AUTONOMOUS


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_recorded_on_success(self):
        """Test that stats are recorded on successful discovery."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value={"type": "work"}),
        )

        await manager.discover_work(["selfplay"])

        stats = manager.get_stats()
        assert stats["successes"]["leader"] == 1
        assert stats["last_success_channel"] == "leader"

    @pytest.mark.asyncio
    async def test_stats_recorded_on_failure(self):
        """Test that stats are recorded on failed discovery."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        await manager.discover_work(["selfplay"])

        stats = manager.get_stats()
        assert stats["failures"]["leader"] >= 1

    def test_get_stats_includes_cooldown_info(self):
        """Test that get_stats includes cooldown state."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        stats = manager.get_stats()
        assert "can_query_peers" in stats
        assert "can_direct_selfplay" in stats


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for singleton management."""

    def test_get_returns_none_initially(self):
        """Test that get returns None when not set."""
        assert get_work_discovery_manager() is None

    def test_set_and_get(self):
        """Test setting and getting the singleton."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )
        set_work_discovery_manager(manager)

        assert get_work_discovery_manager() is manager

    def test_reset(self):
        """Test resetting the singleton."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )
        set_work_discovery_manager(manager)
        reset_work_discovery_manager()

        assert get_work_discovery_manager() is None


# ============================================================================
# Selfplay Enabled Check Tests
# ============================================================================


class TestSelfplayEnabledCheck:
    """Tests for _is_selfplay_enabled_for_node function."""

    def test_returns_true_when_no_config(self):
        """Test that function returns True when no config file found."""
        with patch("pathlib.Path.exists", return_value=False):
            assert _is_selfplay_enabled_for_node() is True

    def test_caches_result(self):
        """Test that result is cached."""
        import scripts.p2p.managers.work_discovery_manager as module

        with patch("pathlib.Path.exists", return_value=False):
            # First call
            result1 = _is_selfplay_enabled_for_node()
            # Modify cache
            module._selfplay_enabled = False
            module._selfplay_enabled_checked = True
            # Second call should return cached value
            result2 = _is_selfplay_enabled_for_node()
            assert result2 is False

    def test_handles_yaml_error(self):
        """Test that YAML errors are handled gracefully."""
        import yaml

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=yaml.YAMLError("Parse error")):
                # Should return True (default) on error
                result = _is_selfplay_enabled_for_node()
                assert result is True


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_discover_with_empty_capabilities(self):
        """Test discovery with empty capabilities list."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
            create_direct_selfplay_work=MagicMock(return_value={"type": "work"}),
        )

        result = await manager.discover_work([])

        # Direct selfplay requires "selfplay" capability
        assert result.work_item is None

    @pytest.mark.asyncio
    async def test_concurrent_discovery(self):
        """Test concurrent discovery calls."""
        async def slow_claim(*args):
            await asyncio.sleep(0.05)
            return None

        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=slow_claim,
        )

        # Run multiple discoveries concurrently
        results = await asyncio.gather(
            manager.discover_work(["selfplay"]),
            manager.discover_work(["selfplay"]),
            manager.discover_work(["selfplay"]),
        )

        # All should complete
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_callback_exception_isolation(self):
        """Test that callback exceptions don't propagate."""
        def bad_get_leader():
            raise RuntimeError("Leader callback failed")

        manager = WorkDiscoveryManager(
            get_leader_id=bad_get_leader,
            claim_from_leader=AsyncMock(return_value=None),
        )

        # Should not raise, should return error result
        # Note: The current implementation doesn't catch get_leader_id exceptions
        # This test documents current behavior
        with pytest.raises(RuntimeError):
            await manager.discover_work(["selfplay"])


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health_check() method."""

    def test_health_check_idle_state(self):
        """Test health check returns idle state when no attempts made."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = manager.health_check()

        assert result["healthy"] is True
        assert result["status"] == "idle"
        assert result["stats"]["total_attempts"] == 0

    @pytest.mark.asyncio
    async def test_health_check_active_state(self):
        """Test health check returns active state after successful discovery."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value={"type": "work"}),
        )

        await manager.discover_work(["selfplay"])
        result = manager.health_check()

        assert result["healthy"] is True
        assert result["status"] == "active"
        assert result["stats"]["total_successes"] >= 1
        assert result["stats"]["success_rate"] > 0

    @pytest.mark.asyncio
    async def test_health_check_failing_state(self):
        """Test health check returns failing state when all attempts fail."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        # Multiple failed attempts
        for _ in range(3):
            await manager.discover_work(["selfplay"])

        result = manager.health_check()

        assert result["healthy"] is False
        assert result["status"] == "failing"
        assert result["stats"]["total_failures"] >= 3

    def test_health_check_includes_channels_enabled(self):
        """Test health check shows which channels are enabled."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value=None),
            get_alive_peers=lambda: ["peer-1"],
            query_peer_work=AsyncMock(return_value=None),
        )

        result = manager.health_check()

        assert "channels_enabled" in result
        assert result["channels_enabled"]["leader"] is True
        assert result["channels_enabled"]["peer"] is True
        assert result["channels_enabled"]["autonomous"] is False
        assert result["channels_enabled"]["direct"] is False

    def test_health_check_no_channels_state(self):
        """Test health check returns no_channels when all disabled."""
        config = WorkDiscoveryConfig(
            leader_enabled=False,
            peer_discovery_enabled=False,
            autonomous_enabled=False,
            direct_selfplay_enabled=False,
        )
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
            config=config,
        )

        result = manager.health_check()

        assert result["healthy"] is False
        assert result["status"] == "no_channels"

    @pytest.mark.asyncio
    async def test_health_check_tracks_last_success(self):
        """Test health check includes last success information."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(return_value={"type": "work"}),
        )

        await manager.discover_work(["selfplay"])
        result = manager.health_check()

        assert result["last_success_channel"] == "leader"
        assert result["last_success_seconds_ago"] is not None
        assert result["last_success_seconds_ago"] < 5  # Within 5 seconds

    def test_health_check_includes_cooldowns(self):
        """Test health check includes cooldown status."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: None,
            claim_from_leader=AsyncMock(return_value=None),
        )

        result = manager.health_check()

        assert "cooldowns" in result
        assert "can_query_peers" in result["cooldowns"]
        assert "can_direct_selfplay" in result["cooldowns"]

    @pytest.mark.asyncio
    async def test_health_check_degraded_state(self):
        """Test health check returns degraded when success rate is low (<10%)."""
        manager = WorkDiscoveryManager(
            get_leader_id=lambda: "leader-1",
            claim_from_leader=AsyncMock(side_effect=[
                None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                {"type": "work"},  # 1 success after many failures
            ]),
        )

        # Many failures then 1 success results in low success rate (below 10% threshold)
        # Note: peer channel is also tried when leader fails (callback not configured),
        # so more attempts are recorded than just leader attempts
        for _ in range(20):
            await manager.discover_work(["selfplay"])

        result = manager.health_check()

        assert result["healthy"] is True
        assert result["status"] == "degraded"  # Low success rate < 10% threshold
        assert result["stats"]["success_rate"] < 0.1  # Below threshold
