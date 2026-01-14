"""Unit tests for predictive sync functionality.

Jan 2026: Tests for Phase 3 of Cluster Manifest Training Integration.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPredictTrainingNode:
    """Tests for _predict_training_node method."""

    @pytest.mark.asyncio
    async def test_predict_from_pending_jobs(self):
        """Test prediction from pending training jobs."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        # Create a mock mixin instance
        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._predict_training_node = PipelineEventHandlerMixin._predict_training_node.__get__(
            mixin, PipelineEventHandlerMixin
        )

        mock_queue = MagicMock()
        mock_queue.get_pending_jobs.return_value = [
            {"target_node": "training-node-1", "config_key": "hex8_2p"}
        ]

        with patch(
            "app.coordination.work_queue.get_work_queue",
            return_value=mock_queue,
        ):
            result = await mixin._predict_training_node("hex8_2p")

        assert result == "training-node-1"

    @pytest.mark.asyncio
    async def test_predict_no_pending_jobs_returns_none(self):
        """Test prediction when no pending jobs exist."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._predict_training_node = PipelineEventHandlerMixin._predict_training_node.__get__(
            mixin, PipelineEventHandlerMixin
        )

        mock_queue = MagicMock()
        mock_queue.get_pending_jobs.return_value = []
        # Also make the fallback fail
        mock_queue.get_likely_training_assignment.side_effect = AttributeError()

        with patch(
            "app.coordination.work_queue.get_work_queue",
            return_value=mock_queue,
        ):
            result = await mixin._predict_training_node("hex8_2p")

        assert result is None

    @pytest.mark.asyncio
    async def test_predict_handles_import_error(self):
        """Test graceful handling when work queue not available."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._predict_training_node = PipelineEventHandlerMixin._predict_training_node.__get__(
            mixin, PipelineEventHandlerMixin
        )

        with patch(
            "app.coordination.work_queue.get_work_queue",
            side_effect=ImportError("No work queue"),
        ):
            result = await mixin._predict_training_node("hex8_2p")

        assert result is None


class TestTriggerPredictiveSync:
    """Tests for _trigger_predictive_sync_if_needed method."""

    @pytest.mark.asyncio
    async def test_sync_triggered_for_remote_node(self):
        """Test that sync is triggered when training node is different from local."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._trigger_predictive_sync_if_needed = (
            PipelineEventHandlerMixin._trigger_predictive_sync_if_needed.__get__(
                mixin, PipelineEventHandlerMixin
            )
        )
        mixin._predict_training_node = AsyncMock(return_value="remote-node-1")

        mock_facade = MagicMock()
        mock_facade.trigger_priority_sync = AsyncMock()

        mock_env = MagicMock()
        mock_env.node_id = "local-node"

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            return_value=mock_facade,
        ):
            with patch("app.config.env.env", mock_env):
                await mixin._trigger_predictive_sync_if_needed("hex8_2p")

        # Sync should be triggered for remote node
        mock_facade.trigger_priority_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_sync_for_local_node(self):
        """Test that sync is skipped when training node is local."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._trigger_predictive_sync_if_needed = (
            PipelineEventHandlerMixin._trigger_predictive_sync_if_needed.__get__(
                mixin, PipelineEventHandlerMixin
            )
        )
        mixin._predict_training_node = AsyncMock(return_value="local-node")

        mock_facade = MagicMock()
        mock_facade.trigger_priority_sync = AsyncMock()

        mock_env = MagicMock()
        mock_env.node_id = "local-node"

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            return_value=mock_facade,
        ):
            with patch("app.config.env.env", mock_env):
                await mixin._trigger_predictive_sync_if_needed("hex8_2p")

        # No sync should be triggered (local node)
        mock_facade.trigger_priority_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_sync_when_no_prediction(self):
        """Test that sync is skipped when no training node predicted."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._trigger_predictive_sync_if_needed = (
            PipelineEventHandlerMixin._trigger_predictive_sync_if_needed.__get__(
                mixin, PipelineEventHandlerMixin
            )
        )
        mixin._predict_training_node = AsyncMock(return_value=None)

        mock_facade = MagicMock()
        mock_facade.trigger_priority_sync = AsyncMock()

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            return_value=mock_facade,
        ):
            await mixin._trigger_predictive_sync_if_needed("hex8_2p")

        # No sync should be triggered (no prediction)
        mock_facade.trigger_priority_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_handling_of_sync_failure(self):
        """Test graceful handling when sync fails."""
        from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin

        mixin = MagicMock(spec=PipelineEventHandlerMixin)
        mixin._trigger_predictive_sync_if_needed = (
            PipelineEventHandlerMixin._trigger_predictive_sync_if_needed.__get__(
                mixin, PipelineEventHandlerMixin
            )
        )
        mixin._predict_training_node = AsyncMock(return_value="remote-node-1")

        mock_facade = MagicMock()
        mock_facade.trigger_priority_sync = AsyncMock(
            side_effect=RuntimeError("Sync failed")
        )

        mock_env = MagicMock()
        mock_env.node_id = "local-node"

        with patch(
            "app.coordination.sync_facade.get_sync_facade",
            return_value=mock_facade,
        ):
            with patch("app.config.env.env", mock_env):
                # Should not raise an exception
                await mixin._trigger_predictive_sync_if_needed("hex8_2p")
