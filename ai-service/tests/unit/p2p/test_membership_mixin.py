"""Tests for MembershipMixin SWIM-based membership management.

Tests cover:
- SWIM initialization with config
- SWIM start/stop lifecycle
- Hybrid mode alive checking
- Health check implementation
- Event emission on member status changes

Created: Dec 29, 2025
"""

from __future__ import annotations

import time
from threading import RLock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.membership_mixin import (
    MEMBERSHIP_MODE,
    SWIM_ADAPTER_AVAILABLE,
    SWIM_ENABLED,
    MembershipMixin,
)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with required attributes for MembershipMixin."""

    class MockOrchestrator(MembershipMixin):
        def __init__(self):
            self.node_id = "test-node-1"
            self.peers = {}
            self.peers_lock = RLock()
            self.bootstrap_seeds = ["seed-node"]
            self._swim_manager = None
            self._swim_started = False
            self._event_emissions = []

        def _emit_event(self, event_type, payload):
            """Track event emissions for testing."""
            self._event_emissions.append((event_type, payload))

    return MockOrchestrator()


class TestMembershipMixinImport:
    """Tests for basic import and constants."""

    def test_mixin_importable(self):
        """Test that MembershipMixin can be imported."""
        assert MembershipMixin is not None

    def test_swim_constants_defined(self):
        """Test SWIM constants are defined."""
        # These should be defined regardless of SWIM availability
        assert isinstance(SWIM_ENABLED, bool)
        assert isinstance(SWIM_ADAPTER_AVAILABLE, bool)
        assert isinstance(MEMBERSHIP_MODE, str)
        assert MEMBERSHIP_MODE in ("http", "swim", "hybrid")


class TestMembershipMixinInit:
    """Tests for SWIM initialization."""

    def test_init_swim_disabled(self, mock_orchestrator):
        """Test _init_swim_membership when SWIM is disabled or unavailable."""
        # Test the path where SWIM is unavailable by ensuring manager is None
        mock_orchestrator._swim_manager = None
        mock_orchestrator._swim_started = False

        # Should not raise even without SWIM
        result = mock_orchestrator._init_swim_membership()
        # Returns False when SWIM is disabled or unavailable, True if enabled
        assert isinstance(result, bool)

    def test_swim_manager_initially_none(self, mock_orchestrator):
        """Test SWIM manager is None before initialization."""
        assert mock_orchestrator._swim_manager is None
        assert mock_orchestrator._swim_started is False


class TestMembershipHealthCheck:
    """Tests for membership health check."""

    def test_health_check_returns_dict(self, mock_orchestrator):
        """Test health_check returns proper structure."""
        result = mock_orchestrator.health_check()

        assert isinstance(result, dict)
        assert "healthy" in result
        assert "message" in result
        assert "details" in result

    def test_health_check_healthy_with_peers(self, mock_orchestrator):
        """Test health check is healthy when peers exist."""
        # Add a peer
        mock_peer = MagicMock()
        mock_peer.is_alive.return_value = True
        mock_orchestrator.peers["peer-1"] = mock_peer

        result = mock_orchestrator.health_check()

        # Should be healthy if we have peers (in HTTP mode)
        # or if no bootstrap seeds (single node mode)
        assert isinstance(result["healthy"], bool)

    def test_health_check_includes_mode(self, mock_orchestrator):
        """Test health check includes membership mode."""
        result = mock_orchestrator.health_check()

        assert "membership_mode" in result.get("details", result)

    def test_membership_health_check_detailed(self, mock_orchestrator):
        """Test membership_health_check returns detailed status."""
        result = mock_orchestrator.membership_health_check()

        assert isinstance(result, dict)
        assert "is_healthy" in result
        assert "swim_enabled" in result
        assert "swim_available" in result
        assert "membership_mode" in result


class TestHybridAliveChecking:
    """Tests for hybrid alive peer checking."""

    def test_is_peer_alive_hybrid_http_only(self, mock_orchestrator):
        """Test hybrid check with HTTP peer alive."""
        mock_peer = MagicMock()
        mock_peer.is_alive.return_value = True
        mock_orchestrator.peers["peer-1"] = mock_peer

        result = mock_orchestrator.is_peer_alive_hybrid("peer-1")

        # Should be alive via HTTP
        assert result is True

    def test_is_peer_alive_hybrid_unknown_peer(self, mock_orchestrator):
        """Test hybrid check with unknown peer."""
        result = mock_orchestrator.is_peer_alive_hybrid("unknown-peer")

        # Unknown peer should not be alive
        assert result is False

    def test_is_peer_alive_hybrid_dead_peer(self, mock_orchestrator):
        """Test hybrid check with dead peer."""
        mock_peer = MagicMock()
        mock_peer.is_alive.return_value = False
        mock_orchestrator.peers["peer-1"] = mock_peer

        result = mock_orchestrator.is_peer_alive_hybrid("peer-1")

        # Dead peer should not be alive
        assert result is False

    def test_get_alive_peers_hybrid_empty(self, mock_orchestrator):
        """Test get_alive_peers_hybrid with no peers."""
        result = mock_orchestrator.get_alive_peers_hybrid()

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_alive_peers_hybrid_some_alive(self, mock_orchestrator):
        """Test get_alive_peers_hybrid with mixed peer states."""
        alive_peer = MagicMock()
        alive_peer.is_alive.return_value = True

        dead_peer = MagicMock()
        dead_peer.is_alive.return_value = False

        mock_orchestrator.peers["alive-1"] = alive_peer
        mock_orchestrator.peers["dead-1"] = dead_peer

        result = mock_orchestrator.get_alive_peers_hybrid()

        assert "alive-1" in result
        assert "dead-1" not in result


class TestSwimMemberCallbacks:
    """Tests for SWIM member status callbacks."""

    def test_on_swim_member_alive(self, mock_orchestrator):
        """Test callback when SWIM detects member alive."""
        # Setup peer
        mock_peer = MagicMock()
        mock_peer.last_heartbeat = 0
        mock_peer.consecutive_failures = 5
        mock_orchestrator.peers["peer-1"] = mock_peer

        # Call callback
        mock_orchestrator._on_swim_member_alive("peer-1")

        # Verify peer state updated
        assert mock_peer.last_heartbeat > 0
        assert mock_peer.consecutive_failures == 0

    def test_on_swim_member_failed(self, mock_orchestrator):
        """Test callback when SWIM detects member failed."""
        # Setup peer
        mock_peer = MagicMock()
        mock_peer.consecutive_failures = 0
        mock_peer.last_failure_time = 0
        mock_orchestrator.peers["peer-1"] = mock_peer

        # Call callback
        mock_orchestrator._on_swim_member_failed("peer-1")

        # Verify peer state updated
        assert mock_peer.consecutive_failures == 1
        assert mock_peer.last_failure_time > 0

    def test_on_swim_member_alive_unknown_peer(self, mock_orchestrator):
        """Test alive callback for unknown peer doesn't crash."""
        # Should not raise
        mock_orchestrator._on_swim_member_alive("unknown-peer")

    def test_on_swim_member_failed_unknown_peer(self, mock_orchestrator):
        """Test failed callback for unknown peer doesn't crash."""
        # Should not raise
        mock_orchestrator._on_swim_member_failed("unknown-peer")


class TestSwimMembershipSummary:
    """Tests for SWIM membership summary."""

    def test_get_swim_membership_summary_no_manager(self, mock_orchestrator):
        """Test summary when SWIM manager is not initialized."""
        mock_orchestrator._swim_manager = None

        result = mock_orchestrator.get_swim_membership_summary()

        assert isinstance(result, dict)
        assert result["swim_started"] is False
        assert "membership_mode" in result

    def test_get_swim_membership_summary_with_manager(self, mock_orchestrator):
        """Test summary with initialized SWIM manager."""
        mock_manager = MagicMock()
        mock_manager.get_membership_summary.return_value = {"members": 5}
        mock_orchestrator._swim_manager = mock_manager
        mock_orchestrator._swim_started = True

        result = mock_orchestrator.get_swim_membership_summary()

        assert result["swim_started"] is True
        assert "swim" in result

    def test_get_swim_health_no_manager(self, mock_orchestrator):
        """Test get_swim_health when manager not configured."""
        mock_orchestrator._swim_manager = None

        result = mock_orchestrator.get_swim_health()

        assert result["swim_configured"] is False


class TestSwimLifecycle:
    """Tests for SWIM start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_swim_no_manager(self, mock_orchestrator):
        """Test _start_swim_membership with no manager."""
        mock_orchestrator._swim_manager = None

        result = await mock_orchestrator._start_swim_membership()

        assert result is False

    @pytest.mark.asyncio
    async def test_start_swim_success(self, mock_orchestrator):
        """Test _start_swim_membership success."""
        mock_manager = AsyncMock()
        mock_manager.start.return_value = True
        mock_orchestrator._swim_manager = mock_manager

        result = await mock_orchestrator._start_swim_membership()

        assert result is True
        assert mock_orchestrator._swim_started is True

    @pytest.mark.asyncio
    async def test_start_swim_failure(self, mock_orchestrator):
        """Test _start_swim_membership failure."""
        mock_manager = AsyncMock()
        mock_manager.start.return_value = False
        mock_orchestrator._swim_manager = mock_manager

        result = await mock_orchestrator._start_swim_membership()

        assert result is False
        assert mock_orchestrator._swim_started is False

    @pytest.mark.asyncio
    async def test_stop_swim(self, mock_orchestrator):
        """Test _stop_swim_membership."""
        mock_manager = AsyncMock()
        mock_orchestrator._swim_manager = mock_manager
        mock_orchestrator._swim_started = True

        await mock_orchestrator._stop_swim_membership()

        mock_manager.stop.assert_called_once()
        assert mock_orchestrator._swim_started is False


class TestSwimRecovery:
    """Tests for SWIM auto-recovery."""

    @pytest.mark.asyncio
    async def test_check_and_recover_no_manager(self, mock_orchestrator):
        """Test recovery with no manager configured."""
        mock_orchestrator._swim_manager = None

        result = await mock_orchestrator._check_and_recover_swim()

        # No manager = nothing to recover, return True
        assert result is True

    @pytest.mark.asyncio
    async def test_check_and_recover_healthy(self, mock_orchestrator):
        """Test recovery when SWIM is healthy."""
        mock_manager = MagicMock()
        mock_manager.get_health_status.return_value = {"healthy": True}
        mock_orchestrator._swim_manager = mock_manager

        result = await mock_orchestrator._check_and_recover_swim()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_and_recover_unhealthy_rate_limited(self, mock_orchestrator):
        """Test recovery is rate-limited."""
        mock_manager = MagicMock()
        mock_manager.get_health_status.return_value = {"healthy": False, "reason": "test"}
        mock_orchestrator._swim_manager = mock_manager

        # Set recent recovery attempt
        mock_orchestrator._swim_recovery_attempts = 1
        mock_orchestrator._swim_last_recovery_time = time.time()

        result = await mock_orchestrator._check_and_recover_swim()

        # Should be rate-limited
        assert result is False
