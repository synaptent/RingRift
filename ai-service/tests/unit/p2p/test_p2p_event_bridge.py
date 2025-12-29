"""Tests for P2P Event Bridge.

December 2025: Tests for the P2P to EventRouter bridge module.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# =============================================================================
# Config Key Parsing Tests
# =============================================================================


class TestParseConfigKeySafe:
    """Tests for _parse_config_key_safe function."""

    def test_parse_valid_config_key(self):
        """Test parsing valid config key."""
        from scripts.p2p.p2p_event_bridge import _parse_config_key_safe

        board_type, num_players = _parse_config_key_safe("hex8_2p")
        assert board_type == "hex8"
        assert num_players == 2

    def test_parse_config_key_empty(self):
        """Test parsing empty config key."""
        from scripts.p2p.p2p_event_bridge import _parse_config_key_safe

        board_type, num_players = _parse_config_key_safe("")
        assert board_type == ""
        assert num_players == 0

    def test_parse_config_key_no_underscore(self):
        """Test parsing config key without underscore."""
        from scripts.p2p.p2p_event_bridge import _parse_config_key_safe

        board_type, num_players = _parse_config_key_safe("invalid")
        assert board_type == ""
        assert num_players == 0

    def test_parse_config_key_invalid_players(self):
        """Test parsing config key with invalid players."""
        from scripts.p2p.p2p_event_bridge import _parse_config_key_safe

        board_type, num_players = _parse_config_key_safe("hex8_abc")
        # Fallback parsing should fail
        assert board_type == ""
        assert num_players == 0


# =============================================================================
# Config Key Validation Tests
# =============================================================================


class TestValidateConfigKey:
    """Tests for _validate_config_key function."""

    def test_validate_valid_config(self):
        """Test validation of valid config key."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("hex8_2p") is True
        assert _validate_config_key("square8_3p") is True
        assert _validate_config_key("hexagonal_4p") is True
        assert _validate_config_key("square19_2p") is True

    def test_validate_empty_config(self):
        """Test validation of empty config key."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("") is False

    def test_validate_special_all(self):
        """Test validation of special 'all' config."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("all") is True

    def test_validate_node_specific(self):
        """Test validation of node-specific config."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("node:runpod-h100") is True

    def test_validate_invalid_board_type(self):
        """Test validation of invalid board type."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("invalid_2p") is False

    def test_validate_invalid_players_format(self):
        """Test validation of invalid players format."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("hex8_5p") is False  # Only 2-4 supported
        assert _validate_config_key("hex8_2") is False   # Missing 'p'

    def test_validate_no_underscore(self):
        """Test validation of config without underscore."""
        from scripts.p2p.p2p_event_bridge import _validate_config_key

        assert _validate_config_key("hex82p") is False


# =============================================================================
# Work Completed Event Tests
# =============================================================================


class TestEmitP2PWorkCompleted:
    """Tests for emit_p2p_work_completed function."""

    @pytest.mark.asyncio
    async def test_emit_selfplay_complete(self):
        """Test emitting SELFPLAY_COMPLETE event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_work_completed(
                    work_id="work-123",
                    work_type="selfplay",
                    config_key="hex8_2p",
                    result={"games_generated": 500},
                    node_id="runpod-h100",
                    duration_seconds=120.0,
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "SELFPLAY_COMPLETE"
        assert call_kwargs["payload"]["config_key"] == "hex8_2p"
        assert call_kwargs["payload"]["games_generated"] == 500
        assert call_kwargs["payload"]["node_id"] == "runpod-h100"

    @pytest.mark.asyncio
    async def test_emit_training_complete(self):
        """Test emitting TRAINING_COMPLETED event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_work_completed(
                    work_id="work-456",
                    work_type="training",
                    config_key="hex8_2p",
                    result={"model_path": "/models/test.pth", "val_loss": 0.5},
                    node_id="nebius-h100",
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "TRAINING_COMPLETED"
        assert call_kwargs["payload"]["model_path"] == "/models/test.pth"

    @pytest.mark.asyncio
    async def test_emit_tournament_complete(self):
        """Test emitting EVALUATION_COMPLETED event for tournament."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_work_completed(
                    work_id="work-789",
                    work_type="tournament",
                    config_key="hex8_2p",
                    result={"best_model": "model-1", "best_elo": 1500},
                    node_id="runpod-a100",
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "EVALUATION_COMPLETED"

    @pytest.mark.asyncio
    async def test_emit_work_completed_no_router(self):
        """Test emit when event router not available."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        # Should not raise, just log
        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", False):
            await emit_p2p_work_completed(
                work_id="work-123",
                work_type="selfplay",
                config_key="hex8_2p",
                result={},
                node_id="test-node",
            )

    @pytest.mark.asyncio
    async def test_emit_work_completed_unknown_type(self):
        """Test emit with unknown work type."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_work_completed(
                    work_id="work-123",
                    work_type="unknown_type",
                    config_key="hex8_2p",
                    result={},
                    node_id="test-node",
                )

        # Should not emit for unknown types
        mock_publish.assert_not_called()


# =============================================================================
# Work Failed Event Tests
# =============================================================================


class TestEmitP2PWorkFailed:
    """Tests for emit_p2p_work_failed function."""

    @pytest.mark.asyncio
    async def test_emit_training_failed(self):
        """Test emitting TRAINING_FAILED event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_failed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_work_failed(
                    work_id="work-123",
                    work_type="training",
                    config_key="hex8_2p",
                    error="Out of memory",
                    node_id="runpod-h100",
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "TRAINING_FAILED"
        assert call_kwargs["payload"]["error"] == "Out of memory"

    @pytest.mark.asyncio
    async def test_emit_non_training_failed(self):
        """Test emit for non-training failures just logs."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_failed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_work_failed(
                    work_id="work-123",
                    work_type="selfplay",
                    config_key="hex8_2p",
                    error="Test error",
                    node_id="test-node",
                )

        # Should not emit for non-training failures
        mock_publish.assert_not_called()


# =============================================================================
# Gauntlet Completed Event Tests
# =============================================================================


class TestEmitP2PGauntletCompleted:
    """Tests for emit_p2p_gauntlet_completed function."""

    @pytest.mark.asyncio
    async def test_emit_gauntlet_passed(self):
        """Test emitting gauntlet passed event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_gauntlet_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_gauntlet_completed(
                    model_id="model-new",
                    baseline_id="model-baseline",
                    config_key="hex8_2p",
                    wins=8,
                    total_games=10,
                    win_rate=0.8,
                    passed=True,
                    node_id="runpod-h100",
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "EVALUATION_COMPLETED"
        assert call_kwargs["payload"]["model_id"] == "model-new"
        assert call_kwargs["payload"]["win_rate"] == 0.8
        assert call_kwargs["payload"]["passed"] is True

    @pytest.mark.asyncio
    async def test_emit_gauntlet_failed(self):
        """Test emitting gauntlet failed event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_gauntlet_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_gauntlet_completed(
                    model_id="model-new",
                    baseline_id="model-baseline",
                    config_key="hex8_2p",
                    wins=2,
                    total_games=10,
                    win_rate=0.2,
                    passed=False,
                    node_id="runpod-h100",
                )

        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["payload"]["passed"] is False


# =============================================================================
# Node Lifecycle Event Tests
# =============================================================================


class TestNodeLifecycleEvents:
    """Tests for node online/offline events."""

    @pytest.mark.asyncio
    async def test_emit_node_online(self):
        """Test emitting HOST_ONLINE event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_node_online

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_node_online(
                    node_id="runpod-h100",
                    host_type="gpu",
                    capabilities={"gpu": "H100", "vram_gb": 80},
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "HOST_ONLINE"
        assert call_kwargs["payload"]["node_id"] == "runpod-h100"
        assert call_kwargs["payload"]["host_type"] == "gpu"

    @pytest.mark.asyncio
    async def test_emit_node_offline(self):
        """Test emitting HOST_OFFLINE event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_node_offline

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_node_offline(
                    node_id="runpod-h100",
                    reason="timeout",
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "HOST_OFFLINE"
        assert call_kwargs["payload"]["reason"] == "timeout"


# =============================================================================
# Cluster Health Event Tests
# =============================================================================


class TestClusterHealthEvents:
    """Tests for cluster health events."""

    @pytest.mark.asyncio
    async def test_emit_cluster_healthy(self):
        """Test emitting P2P_CLUSTER_HEALTHY event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_cluster_healthy

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_cluster_healthy(
                    alive_peers=30,
                    total_peers=36,
                    leader_id="nebius-h100",
                    quorum=True,
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "P2P_CLUSTER_HEALTHY"
        assert call_kwargs["payload"]["alive_peers"] == 30
        assert call_kwargs["payload"]["leader_id"] == "nebius-h100"

    @pytest.mark.asyncio
    async def test_emit_cluster_unhealthy(self):
        """Test emitting P2P_CLUSTER_UNHEALTHY event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_cluster_unhealthy

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_cluster_unhealthy(
                    alive_peers=2,
                    total_peers=36,
                    reason="quorum_lost",
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "P2P_CLUSTER_UNHEALTHY"
        assert call_kwargs["payload"]["reason"] == "quorum_lost"


# =============================================================================
# Elo Sync Event Tests
# =============================================================================


class TestEloSyncEvents:
    """Tests for Elo sync events."""

    @pytest.mark.asyncio
    async def test_emit_elo_updated(self):
        """Test emitting ELO_UPDATED event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_elo_updated

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_elo_updated(
                    model_id="model-v2",
                    config_key="hex8_2p",
                    old_elo=1400.0,
                    new_elo=1450.0,
                    games_played=50,
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "ELO_UPDATED"
        assert call_kwargs["payload"]["old_elo"] == 1400.0
        assert call_kwargs["payload"]["new_elo"] == 1450.0
        assert call_kwargs["payload"]["elo_delta"] == 50.0


# =============================================================================
# Leader Election Event Tests
# =============================================================================


class TestLeaderElectionEvents:
    """Tests for leader election events."""

    @pytest.mark.asyncio
    async def test_emit_leader_changed(self):
        """Test emitting LEADER_CHANGED event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_leader_changed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                await emit_p2p_leader_changed(
                    new_leader_id="nebius-h100-2",
                    old_leader_id="nebius-h100-1",
                    term=5,
                )

        mock_publish.assert_called_once()
        call_kwargs = mock_publish.call_args.kwargs
        assert call_kwargs["event_type"] == "LEADER_CHANGED"
        assert call_kwargs["payload"]["new_leader_id"] == "nebius-h100-2"
        assert call_kwargs["payload"]["old_leader_id"] == "nebius-h100-1"
        assert call_kwargs["payload"]["term"] == 5


# =============================================================================
# Sync Version Tests
# =============================================================================


class TestSyncVersions:
    """Tests for synchronous wrapper functions."""

    def test_emit_work_completed_sync_in_loop(self):
        """Test sync version schedules task when in async context."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed_sync

        # Create a running event loop
        async def run_test():
            with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", False):
                # Should not raise when called from async context
                emit_p2p_work_completed_sync(
                    work_id="work-123",
                    work_type="selfplay",
                    config_key="hex8_2p",
                    result={},
                    node_id="test-node",
                )

        asyncio.run(run_test())

    def test_emit_node_online_sync(self):
        """Test sync version of node online event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_node_online_sync

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", False):
            # Should not raise
            emit_p2p_node_online_sync(
                node_id="test-node",
                host_type="gpu",
            )

    def test_emit_node_offline_sync(self):
        """Test sync version of node offline event."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_node_offline_sync

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", False):
            # Should not raise
            emit_p2p_node_offline_sync(
                node_id="test-node",
                reason="test",
            )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in event emission."""

    @pytest.mark.asyncio
    async def test_emit_handles_publish_error(self):
        """Test that publish errors are caught and logged."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        mock_publish = AsyncMock(side_effect=RuntimeError("test error"))

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                # Should not raise
                await emit_p2p_work_completed(
                    work_id="work-123",
                    work_type="selfplay",
                    config_key="hex8_2p",
                    result={},
                    node_id="test-node",
                )

    @pytest.mark.asyncio
    async def test_emit_handles_invalid_payload(self):
        """Test handling of events with missing payload fields."""
        from scripts.p2p.p2p_event_bridge import emit_p2p_work_completed

        mock_publish = AsyncMock()

        with patch("scripts.p2p.p2p_event_bridge.HAS_EVENT_ROUTER", True):
            with patch("scripts.p2p.p2p_event_bridge.publish", mock_publish):
                # Should handle None result gracefully
                await emit_p2p_work_completed(
                    work_id="work-123",
                    work_type="selfplay",
                    config_key="hex8_2p",
                    result=None,  # type: ignore
                    node_id="test-node",
                )


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test that __all__ contains expected exports."""
        from scripts.p2p import p2p_event_bridge

        expected = {
            "emit_p2p_work_completed",
            "emit_p2p_work_completed_sync",
            "emit_p2p_work_failed",
            "emit_p2p_gauntlet_completed",
            "emit_p2p_node_online",
            "emit_p2p_node_online_sync",
            "emit_p2p_node_offline",
            "emit_p2p_node_offline_sync",
            "emit_p2p_cluster_healthy",
            "emit_p2p_cluster_unhealthy",
            "emit_p2p_elo_updated",
            "emit_p2p_leader_changed",
            "HAS_EVENT_ROUTER",
        }

        assert set(p2p_event_bridge.__all__) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
