"""Unit tests for TrainingCoordinator data pre-positioning functionality.

Jan 2026: Tests for Phase 4 of Cluster Manifest Training Integration.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSelectBestDataSource:
    """Tests for _select_best_data_source method."""

    def test_no_catalog_returns_none(self):
        """Test graceful handling when data catalog not available."""
        from app.coordination.training_coordinator import TrainingCoordinator

        with patch(
            "app.coordination.training_coordinator.HAS_DATA_CATALOG", False
        ):
            coordinator = MagicMock(spec=TrainingCoordinator)
            coordinator._node_name = "local-node"
            coordinator._select_best_data_source = (
                TrainingCoordinator._select_best_data_source.__get__(
                    coordinator, TrainingCoordinator
                )
            )

            result = coordinator._select_best_data_source("hex8", 2)
            assert result is None

    def test_prefers_local_data(self):
        """Test that local data gets highest score."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._select_best_data_source = (
            TrainingCoordinator._select_best_data_source.__get__(
                coordinator, TrainingCoordinator
            )
        )

        mock_registry = MagicMock()
        mock_registry.get_data_sources.return_value = [
            {"node_id": "local-node", "game_count": 5000, "path": "data/games/hex8_2p.db"},
            {"node_id": "remote-node", "game_count": 10000, "path": "data/games/hex8_2p.db"},
        ]

        with patch(
            "app.coordination.training_coordinator.HAS_DATA_CATALOG", True
        ):
            with patch(
                "app.coordination.training_coordinator.get_data_registry",
                return_value=mock_registry,
            ):
                result = coordinator._select_best_data_source("hex8", 2)

        assert result is not None
        assert result["node_id"] == "local-node"
        assert result["is_local"] is True

    def test_prefers_higher_game_count(self):
        """Test that sources with more games get higher scores."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._select_best_data_source = (
            TrainingCoordinator._select_best_data_source.__get__(
                coordinator, TrainingCoordinator
            )
        )

        mock_registry = MagicMock()
        mock_registry.get_data_sources.return_value = [
            {"node_id": "remote-1", "game_count": 5000, "path": "data/games/hex8_2p.db"},
            {"node_id": "remote-2", "game_count": 15000, "path": "data/games/hex8_2p.db"},
        ]

        with patch(
            "app.coordination.training_coordinator.HAS_DATA_CATALOG", True
        ):
            with patch(
                "app.coordination.training_coordinator.get_data_registry",
                return_value=mock_registry,
            ):
                result = coordinator._select_best_data_source("hex8", 2)

        assert result is not None
        assert result["node_id"] == "remote-2"
        assert result["game_count"] == 15000

    def test_no_sources_returns_none(self):
        """Test handling when no data sources available."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._select_best_data_source = (
            TrainingCoordinator._select_best_data_source.__get__(
                coordinator, TrainingCoordinator
            )
        )

        mock_registry = MagicMock()
        mock_registry.get_data_sources.return_value = []

        with patch(
            "app.coordination.training_coordinator.HAS_DATA_CATALOG", True
        ):
            with patch(
                "app.coordination.training_coordinator.get_data_registry",
                return_value=mock_registry,
            ):
                result = coordinator._select_best_data_source("hex8", 2)

        assert result is None

    def test_target_node_override(self):
        """Test that target_node parameter affects locality scoring."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._select_best_data_source = (
            TrainingCoordinator._select_best_data_source.__get__(
                coordinator, TrainingCoordinator
            )
        )

        mock_registry = MagicMock()
        mock_registry.get_data_sources.return_value = [
            {"node_id": "local-node", "game_count": 5000, "path": "data/games/hex8_2p.db"},
            {"node_id": "target-node", "game_count": 5000, "path": "data/games/hex8_2p.db"},
        ]

        with patch(
            "app.coordination.training_coordinator.HAS_DATA_CATALOG", True
        ):
            with patch(
                "app.coordination.training_coordinator.get_data_registry",
                return_value=mock_registry,
            ):
                result = coordinator._select_best_data_source(
                    "hex8", 2, target_node="target-node"
                )

        assert result is not None
        assert result["node_id"] == "target-node"
        assert result["is_local"] is True


class TestEnsureDataAtNode:
    """Tests for _ensure_data_at_node method."""

    @pytest.mark.asyncio
    async def test_local_data_returns_true(self):
        """Test that local data needs no sync."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator._ensure_data_at_node = (
            TrainingCoordinator._ensure_data_at_node.__get__(
                coordinator, TrainingCoordinator
            )
        )

        # Mock _select_best_data_source to return local data
        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "local-node",
                "game_count": 5000,
                "is_local": True,
            }
        )

        result = await coordinator._ensure_data_at_node("hex8", 2)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_sources_returns_false(self):
        """Test handling when no sources available."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator._ensure_data_at_node = (
            TrainingCoordinator._ensure_data_at_node.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(return_value=None)

        result = await coordinator._ensure_data_at_node("hex8", 2)
        assert result is False

    @pytest.mark.asyncio
    async def test_triggers_sync_for_remote_data(self):
        """Test that sync is triggered for remote data."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator._ensure_data_at_node = (
            TrainingCoordinator._ensure_data_at_node.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "remote-node",
                "game_count": 10000,
                "is_local": False,
            }
        )

        mock_facade = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.nodes_synced = 1
        mock_facade.trigger_priority_sync = AsyncMock(return_value=mock_response)

        with patch(
            "app.coordination.training_coordinator.HAS_SYNC_FACADE", True
        ):
            with patch(
                "app.coordination.training_coordinator.get_sync_facade",
                return_value=mock_facade,
            ):
                result = await coordinator._ensure_data_at_node("hex8", 2)

        assert result is True
        mock_facade.trigger_priority_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_sync_failure(self):
        """Test graceful handling of sync failures."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator._ensure_data_at_node = (
            TrainingCoordinator._ensure_data_at_node.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "remote-node",
                "game_count": 10000,
                "is_local": False,
            }
        )

        mock_facade = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.errors = ["Connection failed"]
        mock_facade.trigger_priority_sync = AsyncMock(return_value=mock_response)

        with patch(
            "app.coordination.training_coordinator.HAS_SYNC_FACADE", True
        ):
            with patch(
                "app.coordination.training_coordinator.get_sync_facade",
                return_value=mock_facade,
            ):
                result = await coordinator._ensure_data_at_node("hex8", 2)

        assert result is False

    @pytest.mark.asyncio
    async def test_no_sync_facade_returns_false(self):
        """Test handling when sync facade not available."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator._ensure_data_at_node = (
            TrainingCoordinator._ensure_data_at_node.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "remote-node",
                "game_count": 10000,
                "is_local": False,
            }
        )

        with patch(
            "app.coordination.training_coordinator.HAS_SYNC_FACADE", False
        ):
            result = await coordinator._ensure_data_at_node("hex8", 2)

        assert result is False


class TestPrepareTrainingDataAsync:
    """Tests for prepare_training_data_async method."""

    @pytest.mark.asyncio
    async def test_returns_result_dict(self):
        """Test that method returns expected result dictionary."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator.prepare_training_data_async = (
            TrainingCoordinator.prepare_training_data_async.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "local-node",
                "game_count": 5000,
                "is_local": True,
            }
        )
        coordinator._ensure_data_at_node = AsyncMock(return_value=True)

        result = await coordinator.prepare_training_data_async("hex8", 2)

        assert "ready" in result
        assert "source" in result
        assert "synced" in result
        assert "games" in result
        assert "config_key" in result
        assert result["ready"] is True
        assert result["games"] == 5000

    @pytest.mark.asyncio
    async def test_marks_synced_if_recent(self):
        """Test that synced flag is set for recent syncs."""
        import time
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {"hex8_2p": time.time() - 30}  # 30 seconds ago
        coordinator.prepare_training_data_async = (
            TrainingCoordinator.prepare_training_data_async.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "local-node",
                "game_count": 5000,
                "is_local": True,
            }
        )
        coordinator._ensure_data_at_node = AsyncMock(return_value=True)

        result = await coordinator.prepare_training_data_async("hex8", 2)

        assert result["synced"] is True

    @pytest.mark.asyncio
    async def test_not_synced_if_old(self):
        """Test that synced flag is false for old syncs."""
        import time
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {"hex8_2p": time.time() - 120}  # 2 min ago
        coordinator.prepare_training_data_async = (
            TrainingCoordinator.prepare_training_data_async.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(
            return_value={
                "node_id": "local-node",
                "game_count": 5000,
                "is_local": True,
            }
        )
        coordinator._ensure_data_at_node = AsyncMock(return_value=True)

        result = await coordinator.prepare_training_data_async("hex8", 2)

        assert result["synced"] is False

    @pytest.mark.asyncio
    async def test_handles_no_data(self):
        """Test handling when no data is available."""
        from app.coordination.training_coordinator import TrainingCoordinator

        coordinator = MagicMock(spec=TrainingCoordinator)
        coordinator._node_name = "local-node"
        coordinator._config_sync_times = {}
        coordinator.prepare_training_data_async = (
            TrainingCoordinator.prepare_training_data_async.__get__(
                coordinator, TrainingCoordinator
            )
        )

        coordinator._select_best_data_source = MagicMock(return_value=None)
        coordinator._ensure_data_at_node = AsyncMock(return_value=False)

        result = await coordinator.prepare_training_data_async("hex8", 2)

        assert result["ready"] is False
        assert result["source"] is None
        assert result["games"] == 0
