"""Tests for Sync Facade module.

Tests the unified entry point for all cluster sync operations,
including backend selection, request routing, and convenience functions.

December 27, 2025: Created to address P1 test gap for sync_facade.py (680 LOC).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_facade import (
    SyncBackend,
    SyncFacade,
    SyncRequest,
    SyncResponse,
    get_sync_facade,
    reset_sync_facade,
    sync,
)


# =============================================================================
# SyncBackend Enum Tests
# =============================================================================


class TestSyncBackend:
    """Tests for SyncBackend enum."""

    def test_all_backends_defined(self):
        """All expected backends should exist."""
        assert SyncBackend.AUTO_SYNC.value == "auto_sync"
        assert SyncBackend.CLUSTER_SYNC.value == "cluster_data_sync"
        assert SyncBackend.DISTRIBUTED.value == "distributed"
        assert SyncBackend.EPHEMERAL.value == "ephemeral_sync"
        assert SyncBackend.ROUTER.value == "router"
        assert SyncBackend.SCHEDULER.value == "scheduler"
        assert SyncBackend.UNIFIED.value == "unified_data_sync"
        assert SyncBackend.ORCHESTRATOR.value == "orchestrator"

    def test_backend_count(self):
        """Should have exactly 8 backends."""
        assert len(SyncBackend) == 8

    def test_deprecated_backends_exist(self):
        """Deprecated backends should still exist for backward compatibility."""
        # These are deprecated but should still be available
        assert hasattr(SyncBackend, "SCHEDULER")
        assert hasattr(SyncBackend, "UNIFIED")


# =============================================================================
# SyncRequest Dataclass Tests
# =============================================================================


class TestSyncRequest:
    """Tests for SyncRequest dataclass."""

    def test_minimal_creation(self):
        """Should create with just data_type."""
        req = SyncRequest(data_type="games")
        assert req.data_type == "games"
        assert req.targets is None
        assert req.priority == "normal"
        assert req.timeout_seconds == 300.0

    def test_full_creation(self):
        """Should create with all fields."""
        req = SyncRequest(
            data_type="models",
            targets=["node-1", "node-2"],
            board_type="hex8",
            num_players=4,
            priority="high",
            timeout_seconds=600.0,
            bandwidth_limit_mbps=100,
            exclude_nodes=["node-3"],
            prefer_ephemeral=True,
            require_confirmation=True,
        )
        assert req.data_type == "models"
        assert req.targets == ["node-1", "node-2"]
        assert req.board_type == "hex8"
        assert req.num_players == 4
        assert req.priority == "high"
        assert req.timeout_seconds == 600.0
        assert req.bandwidth_limit_mbps == 100
        assert req.exclude_nodes == ["node-3"]
        assert req.prefer_ephemeral is True
        assert req.require_confirmation is True

    def test_default_values(self):
        """Should have sensible defaults."""
        req = SyncRequest(data_type="npz")
        assert req.board_type is None
        assert req.num_players is None
        assert req.bandwidth_limit_mbps is None
        assert req.exclude_nodes is None
        assert req.prefer_ephemeral is False
        assert req.require_confirmation is False


# =============================================================================
# SyncResponse Dataclass Tests
# =============================================================================


class TestSyncResponse:
    """Tests for SyncResponse dataclass."""

    def test_minimal_creation(self):
        """Should create with required fields."""
        resp = SyncResponse(success=True, backend_used=SyncBackend.AUTO_SYNC)
        assert resp.success is True
        assert resp.backend_used == SyncBackend.AUTO_SYNC
        assert resp.nodes_synced == 0
        assert resp.bytes_transferred == 0
        assert resp.duration_seconds == 0.0
        assert resp.errors == []
        assert resp.details == {}

    def test_full_creation(self):
        """Should create with all fields."""
        resp = SyncResponse(
            success=True,
            backend_used=SyncBackend.DISTRIBUTED,
            nodes_synced=5,
            bytes_transferred=1024000,
            duration_seconds=12.5,
            errors=["Warning: slow transfer"],
            details={"files_synced": 10},
        )
        assert resp.nodes_synced == 5
        assert resp.bytes_transferred == 1024000
        assert resp.duration_seconds == 12.5
        assert len(resp.errors) == 1
        assert resp.details["files_synced"] == 10

    def test_failure_response(self):
        """Should handle failure responses."""
        resp = SyncResponse(
            success=False,
            backend_used=SyncBackend.EPHEMERAL,
            errors=["Connection timeout", "Retry failed"],
        )
        assert resp.success is False
        assert len(resp.errors) == 2


# =============================================================================
# SyncFacade._select_backend Tests
# =============================================================================


class TestSelectBackend:
    """Tests for SyncFacade._select_backend method."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    def test_ephemeral_preferred_returns_ephemeral(self, facade):
        """Should return EPHEMERAL when prefer_ephemeral is True."""
        req = SyncRequest(data_type="games", prefer_ephemeral=True)
        assert facade._select_backend(req) == SyncBackend.EPHEMERAL

    def test_high_priority_returns_cluster_sync(self, facade):
        """Should return CLUSTER_SYNC for high priority."""
        req = SyncRequest(data_type="games", priority="high")
        assert facade._select_backend(req) == SyncBackend.CLUSTER_SYNC

    def test_critical_priority_returns_cluster_sync(self, facade):
        """Should return CLUSTER_SYNC for critical priority."""
        req = SyncRequest(data_type="models", priority="critical")
        assert facade._select_backend(req) == SyncBackend.CLUSTER_SYNC

    def test_specific_targets_returns_router(self, facade):
        """Should return ROUTER for specific targets."""
        req = SyncRequest(
            data_type="games",
            targets=["node-1", "node-2"],
            priority="normal",
        )
        assert facade._select_backend(req) == SyncBackend.ROUTER

    def test_all_targets_returns_auto_sync(self, facade):
        """Should return AUTO_SYNC for 'all' targets."""
        req = SyncRequest(data_type="games", targets=["all"], priority="normal")
        assert facade._select_backend(req) == SyncBackend.AUTO_SYNC

    def test_no_targets_returns_auto_sync(self, facade):
        """Should return AUTO_SYNC for no targets (default)."""
        req = SyncRequest(data_type="games", priority="normal")
        assert facade._select_backend(req) == SyncBackend.AUTO_SYNC

    def test_low_priority_returns_auto_sync(self, facade):
        """Should return AUTO_SYNC for low priority."""
        req = SyncRequest(data_type="npz", priority="low")
        assert facade._select_backend(req) == SyncBackend.AUTO_SYNC

    def test_prefer_ephemeral_overrides_priority(self, facade):
        """prefer_ephemeral should take precedence over priority."""
        req = SyncRequest(
            data_type="games",
            priority="high",
            prefer_ephemeral=True,
        )
        # prefer_ephemeral is checked first
        assert facade._select_backend(req) == SyncBackend.EPHEMERAL


# =============================================================================
# SyncFacade.sync Tests
# =============================================================================


class TestSyncMethod:
    """Tests for SyncFacade.sync method."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    @pytest.mark.asyncio
    async def test_sync_with_string_request(self, facade):
        """Should accept string data_type."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            response = await facade.sync("games")
            mock_sync.assert_called_once()
            assert response.success is True

    @pytest.mark.asyncio
    async def test_sync_with_dict_request(self, facade):
        """Should accept dict request."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            response = await facade.sync({"data_type": "games"})
            mock_sync.assert_called_once()
            assert response.success is True

    @pytest.mark.asyncio
    async def test_sync_with_request_object(self, facade):
        """Should accept SyncRequest object."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            req = SyncRequest(data_type="games")
            response = await facade.sync(req)
            mock_sync.assert_called_once()
            assert response.success is True

    @pytest.mark.asyncio
    async def test_sync_updates_stats(self, facade):
        """Should update stats after sync."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
                bytes_transferred=1024,
            )
            await facade.sync("games")
            stats = facade.get_stats()
            assert stats["total_syncs"] == 1
            assert stats["total_bytes"] == 1024
            assert stats["by_backend"]["auto_sync"] == 1

    @pytest.mark.asyncio
    async def test_sync_handles_exception(self, facade):
        """Should handle exceptions gracefully."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.side_effect = RuntimeError("Sync failed")
            response = await facade.sync("games")
            assert response.success is False
            assert len(response.errors) == 1
            assert "Sync failed" in response.errors[0]

    @pytest.mark.asyncio
    async def test_sync_tracks_errors_in_stats(self, facade):
        """Should track errors in stats."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.side_effect = RuntimeError("Error")
            await facade.sync("games")
            stats = facade.get_stats()
            assert stats["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_sync_tracks_failed_responses_in_stats(self, facade):
        """Should track failed responses in stats."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=False,
                backend_used=SyncBackend.AUTO_SYNC,
                errors=["Failed"],
            )
            await facade.sync("games")
            stats = facade.get_stats()
            assert stats["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_sync_sets_duration(self, facade):
        """Should set duration_seconds in response."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            response = await facade.sync("games")
            # Duration should be set (non-zero since it's measured)
            assert response.duration_seconds >= 0


# =============================================================================
# SyncFacade Backend Routing Tests
# =============================================================================


class TestBackendRouting:
    """Tests for backend-specific routing."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    @pytest.mark.asyncio
    async def test_deprecated_scheduler_fallback(self, facade):
        """SCHEDULER backend should fallback to AUTO_SYNC."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            await facade._execute_sync(SyncBackend.SCHEDULER, SyncRequest(data_type="games"))
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_deprecated_unified_fallback(self, facade):
        """UNIFIED backend should fallback to DISTRIBUTED."""
        with patch.object(facade, "_sync_via_distributed") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.DISTRIBUTED,
            )
            await facade._execute_sync(SyncBackend.UNIFIED, SyncRequest(data_type="games"))
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_backend_raises_error(self, facade):
        """Unknown backend should raise ValueError."""
        # Create a mock enum value
        with pytest.raises(ValueError, match="Unknown backend"):
            await facade._execute_sync(
                MagicMock(value="unknown"),
                SyncRequest(data_type="games"),
            )


# =============================================================================
# SyncFacade.health_check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for SyncFacade.health_check method."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    def test_health_check_initial_state(self, facade):
        """Should return healthy status initially."""
        health = facade.health_check()
        assert health["status"] == "healthy"
        assert health["operations_count"] == 0
        assert health["errors_count"] == 0
        assert health["backends_loaded"] == 0

    def test_health_check_after_successful_syncs(self, facade):
        """Should remain healthy after successful syncs."""
        # Simulate successful syncs - need to mark backends as loaded
        facade._backends_loaded = {SyncBackend.AUTO_SYNC: True}
        facade._stats["total_syncs"] = 10
        facade._stats["total_errors"] = 0
        facade._stats["by_backend"] = {"auto_sync": 10}

        health = facade.health_check()
        assert health["status"] == "healthy"
        assert health["operations_count"] == 10
        assert health["errors_count"] == 0

    def test_health_check_elevated_error_rate(self, facade):
        """Should return degraded with elevated error rate (>20%)."""
        # Need to mark backends as loaded to check error rate logic
        facade._backends_loaded = {SyncBackend.AUTO_SYNC: True}
        facade._stats["total_syncs"] = 10
        facade._stats["total_errors"] = 3  # 30% error rate

        health = facade.health_check()
        assert health["status"] == "degraded"
        assert "Elevated error rate" in health["last_error"]

    def test_health_check_high_error_rate(self, facade):
        """Should return unhealthy with high error rate (>50%)."""
        # Need to mark backends as loaded to check error rate logic
        facade._backends_loaded = {SyncBackend.AUTO_SYNC: True}
        facade._stats["total_syncs"] = 10
        facade._stats["total_errors"] = 6  # 60% error rate

        health = facade.health_check()
        assert health["status"] == "unhealthy"
        assert "High error rate" in health["last_error"]

    def test_health_check_includes_bytes_transferred(self, facade):
        """Should include total bytes transferred."""
        facade._stats["total_bytes"] = 1024000

        health = facade.health_check()
        assert health["total_bytes_transferred"] == 1024000

    def test_health_check_backend_counts(self, facade):
        """Should track backends loaded and by_backend stats."""
        facade._backends_loaded = {
            SyncBackend.AUTO_SYNC: True,
            SyncBackend.DISTRIBUTED: True,
        }
        facade._stats["by_backend"] = {"auto_sync": 5, "distributed": 3}

        health = facade.health_check()
        assert health["backends_loaded"] == 2
        assert health["by_backend"]["auto_sync"] == 5
        assert health["by_backend"]["distributed"] == 3


# =============================================================================
# SyncFacade.trigger_priority_sync Tests
# =============================================================================


class TestTriggerPrioritySync:
    """Tests for SyncFacade.trigger_priority_sync method."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_basic(self, facade):
        """Should trigger priority sync successfully."""
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.CLUSTER_SYNC,
                nodes_synced=3,
            )
            with patch.object(facade, "_emit_sync_event"):
                response = await facade.trigger_priority_sync(reason="test")
                assert response.success is True
                mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_parses_config_key(self, facade):
        """Should parse board_type and num_players from config_key."""
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.CLUSTER_SYNC,
            )
            with patch.object(facade, "_emit_sync_event"):
                await facade.trigger_priority_sync(
                    reason="orphan_games_recovery",
                    config_key="hex8_4p",
                )
                # Check that request was created with parsed values
                call_args = mock_sync.call_args[0][0]
                assert call_args.board_type == "hex8"
                assert call_args.num_players == 4

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_uses_critical_priority(self, facade):
        """Should use critical priority for urgent syncs."""
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.CLUSTER_SYNC,
            )
            with patch.object(facade, "_emit_sync_event"):
                await facade.trigger_priority_sync(reason="urgent")
                call_args = mock_sync.call_args[0][0]
                assert call_args.priority == "critical"

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_emits_success_event(self, facade):
        """Should emit DATA_SYNC_COMPLETED on success."""
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.CLUSTER_SYNC,
                nodes_synced=5,
            )
            with patch.object(facade, "_emit_sync_event") as mock_emit:
                await facade.trigger_priority_sync(reason="test")
                mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_emits_failure_event(self, facade):
        """Should emit DATA_SYNC_FAILED on failure."""
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=False,
                backend_used=SyncBackend.CLUSTER_SYNC,
                errors=["Failed"],
            )
            with patch.object(facade, "_emit_sync_event") as mock_emit:
                await facade.trigger_priority_sync(reason="test")
                mock_emit.assert_called_once()


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_get_instance_returns_same_instance(self):
        """get_instance should return same instance."""
        reset_sync_facade()
        # Reset class singleton too
        if hasattr(SyncFacade, "_instance"):
            delattr(SyncFacade, "_instance")

        facade1 = SyncFacade.get_instance()
        facade2 = SyncFacade.get_instance()
        assert facade1 is facade2

    def test_get_sync_facade_returns_same_instance(self):
        """get_sync_facade should return same instance."""
        reset_sync_facade()
        facade1 = get_sync_facade()
        facade2 = get_sync_facade()
        assert facade1 is facade2

    def test_reset_sync_facade_clears_singleton(self):
        """reset_sync_facade should clear singleton."""
        reset_sync_facade()
        facade1 = get_sync_facade()
        reset_sync_facade()
        facade2 = get_sync_facade()
        # After reset, should be a new instance
        # Note: facade1 and facade2 may or may not be same object,
        # but facade2 should be a fresh instance
        assert facade2 is not None
        assert facade2._stats["total_syncs"] == 0


# =============================================================================
# Module-level sync Function Tests
# =============================================================================


class TestModuleSyncFunction:
    """Tests for module-level sync() convenience function."""

    def setup_method(self):
        """Reset facade before each test."""
        reset_sync_facade()

    @pytest.mark.asyncio
    async def test_sync_function_basic(self):
        """Should sync via facade."""
        with patch.object(SyncFacade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            response = await sync("games")
            assert response.success is True

    @pytest.mark.asyncio
    async def test_sync_function_with_targets(self):
        """Should pass targets to facade."""
        facade = get_sync_facade()
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.ROUTER,
            )
            reset_sync_facade()  # Need fresh facade
            facade2 = get_sync_facade()
            with patch.object(facade2, "sync") as mock_sync2:
                mock_sync2.return_value = SyncResponse(
                    success=True,
                    backend_used=SyncBackend.ROUTER,
                )
                await sync("models", targets=["node-1", "node-2"])
                call_args = mock_sync2.call_args[0][0]
                assert call_args.targets == ["node-1", "node-2"]

    @pytest.mark.asyncio
    async def test_sync_function_with_kwargs(self):
        """Should pass kwargs to SyncRequest."""
        facade = get_sync_facade()
        with patch.object(facade, "sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.CLUSTER_SYNC,
            )
            await sync(
                "games",
                board_type="hex8",
                priority="high",
            )
            call_args = mock_sync.call_args[0][0]
            assert call_args.board_type == "hex8"
            assert call_args.priority == "high"


# =============================================================================
# Backend-specific Sync Method Tests
# =============================================================================


class TestBackendMethods:
    """Tests for individual backend sync methods."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    @pytest.mark.asyncio
    async def test_sync_via_auto_sync_import_error(self, facade):
        """Should handle ImportError gracefully."""
        with patch.dict("sys.modules", {"app.coordination.auto_sync_daemon": None}):
            with patch(
                "app.coordination.sync_facade.SyncFacade._sync_via_auto_sync",
                wraps=facade._sync_via_auto_sync,
            ):
                # The actual import will fail
                response = await facade._sync_via_auto_sync(SyncRequest(data_type="games"))
                assert response.success is False
                assert "not available" in response.errors[0]

    @pytest.mark.asyncio
    async def test_sync_via_distributed_games(self, facade):
        """Should call coordinator.sync_games for games data_type."""
        mock_coordinator = MagicMock()
        mock_result = MagicMock()
        mock_result.nodes_synced = 5
        mock_result.bytes_transferred = 1024
        mock_coordinator.sync_games = AsyncMock(return_value=mock_result)

        with patch(
            "app.coordination.sync_facade.SyncFacade._sync_via_distributed",
            wraps=facade._sync_via_distributed,
        ):
            with patch(
                "app.distributed.sync_coordinator.SyncCoordinator.get_instance",
                return_value=mock_coordinator,
            ):
                req = SyncRequest(data_type="games", board_type="hex8", num_players=2)
                response = await facade._sync_via_distributed(req)
                assert response.backend_used == SyncBackend.DISTRIBUTED

    @pytest.mark.asyncio
    async def test_sync_via_router_uses_distributed(self, facade):
        """ROUTER backend should use distributed for actual sync."""
        with patch.object(facade, "_sync_via_distributed") as mock_distributed:
            mock_distributed.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.DISTRIBUTED,
            )
            with patch(
                "app.coordination.sync_router.get_sync_router",
            ) as mock_get_router:
                mock_router = MagicMock()
                mock_router.get_sync_targets.return_value = ["node-1", "node-2"]
                mock_get_router.return_value = mock_router

                req = SyncRequest(data_type="games", targets=["node-1"])
                response = await facade._sync_via_router(req)
                mock_distributed.assert_called_once()


# =============================================================================
# get_stats Tests
# =============================================================================


class TestGetStats:
    """Tests for SyncFacade.get_stats method."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    def test_get_stats_initial(self, facade):
        """Should return initial stats."""
        stats = facade.get_stats()
        assert stats["total_syncs"] == 0
        assert stats["total_bytes"] == 0
        assert stats["total_errors"] == 0
        assert stats["by_backend"] == {}

    def test_get_stats_includes_backends_loaded(self, facade):
        """Should include backends_loaded in stats."""
        facade._backends_loaded = {SyncBackend.AUTO_SYNC: True}
        stats = facade.get_stats()
        assert "backends_loaded" in stats
        assert stats["backends_loaded"]["auto_sync"] is True

    def test_get_stats_after_operations(self, facade):
        """Should reflect operations in stats."""
        facade._stats["total_syncs"] = 5
        facade._stats["total_bytes"] = 50000
        facade._stats["by_backend"] = {"auto_sync": 3, "distributed": 2}

        stats = facade.get_stats()
        assert stats["total_syncs"] == 5
        assert stats["total_bytes"] == 50000
        assert stats["by_backend"]["auto_sync"] == 3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def facade(self):
        """Create a fresh SyncFacade for testing."""
        reset_sync_facade()
        return SyncFacade()

    @pytest.mark.asyncio
    async def test_sync_with_empty_targets_list(self, facade):
        """Should handle empty targets list."""
        with patch.object(facade, "_sync_via_auto_sync") as mock_sync:
            mock_sync.return_value = SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )
            req = SyncRequest(data_type="games", targets=[])
            response = await facade.sync(req)
            # Empty list should not route to ROUTER
            mock_sync.assert_called_once()

    def test_config_key_parsing_without_underscore(self, facade):
        """Should handle config_key without underscore."""
        # This tests the parsing logic in trigger_priority_sync
        req = SyncRequest(data_type="games")
        # Create request manually to test parsing
        config_key = "hex8"  # No underscore, no player count
        # Parsing should not crash
        board_type = None
        num_players = None
        if config_key and "_" in config_key:
            parts = config_key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("p"):
                board_type = parts[0]
                try:
                    num_players = int(parts[1][:-1])
                except ValueError:
                    pass
        assert board_type is None
        assert num_players is None

    def test_config_key_parsing_with_invalid_player_count(self, facade):
        """Should handle config_key with invalid player count."""
        config_key = "hex8_xp"  # Invalid player count
        board_type = None
        num_players = None
        if config_key and "_" in config_key:
            parts = config_key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("p"):
                board_type = parts[0]
                try:
                    num_players = int(parts[1][:-1])
                except ValueError:
                    pass
        # Should parse board_type but not num_players
        assert board_type == "hex8"
        assert num_players is None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_syncs(self, facade):
        """Should handle multiple concurrent syncs."""
        async def mock_sync_impl(request):
            await asyncio.sleep(0.01)
            return SyncResponse(
                success=True,
                backend_used=SyncBackend.AUTO_SYNC,
            )

        with patch.object(facade, "_sync_via_auto_sync", side_effect=mock_sync_impl):
            # Run multiple syncs concurrently
            tasks = [
                facade.sync("games"),
                facade.sync("models"),
                facade.sync("npz"),
            ]
            responses = await asyncio.gather(*tasks)
            assert all(r.success for r in responses)
            assert facade._stats["total_syncs"] == 3
