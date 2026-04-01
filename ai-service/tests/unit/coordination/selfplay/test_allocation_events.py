"""Tests for allocation_events.py - Allocation event emission helpers.

January 2026: Tests for the extracted allocation event functions.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestEmitAllocationUpdated:
    """Tests for emit_allocation_updated function."""

    def test_handles_publish_errors_gracefully(self):
        """Test router publish failures are swallowed."""
        from app.coordination.selfplay.allocation_events import emit_allocation_updated

        with patch(
            "app.coordination.event_router.publish_sync",
            side_effect=RuntimeError("test error"),
        ):
            emit_allocation_updated(
                allocation={"hex8_2p": {"node1": 100}},
                total_games=100,
                trigger="allocate_batch",
            )

    def test_emits_event_with_allocation_dict(self):
        """Test emits event with allocation dictionary."""
        from app.coordination.selfplay.allocation_events import emit_allocation_updated

        mock_publish = MagicMock()
        with patch(
            "app.coordination.event_router.publish_sync",
            mock_publish,
        ):
            emit_allocation_updated(
                allocation={"hex8_2p": {"node1": 50, "node2": 50}},
                total_games=100,
                trigger="allocate_batch",
            )

        assert mock_publish.called
        call_args = mock_publish.call_args
        payload = call_args[0][1]
        assert payload["trigger"] == "allocate_batch"
        assert payload["total_games"] == 100
        assert "hex8_2p" in payload["configs"]
        assert payload["node_count"] == 2
        assert call_args.kwargs["source"] == "allocation_events"

    def test_emits_event_with_single_config(self):
        """Test emits event with single config key."""
        from app.coordination.selfplay.allocation_events import emit_allocation_updated

        mock_publish = MagicMock()
        with patch(
            "app.coordination.event_router.publish_sync",
            mock_publish,
        ):
            emit_allocation_updated(
                allocation=None,
                total_games=50,
                trigger="exploration_boost",
                config_key="square8_4p",
            )

        assert mock_publish.called
        payload = mock_publish.call_args[0][1]
        assert "square8_4p" in payload["configs"]

    def test_includes_exploration_boost_from_priorities(self):
        """Test includes exploration boost when config_priorities provided."""
        from app.coordination.selfplay.allocation_events import emit_allocation_updated
        from app.coordination.selfplay_priority_types import ConfigPriority

        mock_publish = MagicMock()
        mock_priority = MagicMock(spec=ConfigPriority)
        mock_priority.exploration_boost = 0.15
        mock_priority.curriculum_weight = 1.2

        with patch(
            "app.coordination.event_router.publish_sync",
            mock_publish,
        ):
            emit_allocation_updated(
                allocation=None,
                total_games=50,
                trigger="exploration_boost",
                config_key="hex8_2p",
                config_priorities={"hex8_2p": mock_priority},
            )

        payload = mock_publish.call_args[0][1]
        assert payload["exploration_boost"] == 0.15
        assert payload["curriculum_weight"] == 1.2

    def test_handles_exception_gracefully(self):
        """Test handles exceptions without raising."""
        from app.coordination.selfplay.allocation_events import emit_allocation_updated

        with patch(
            "app.coordination.event_router.publish_sync",
            side_effect=RuntimeError("test error"),
        ):
            # Should not raise
            emit_allocation_updated(
                allocation={"hex8_2p": {"node1": 100}},
                total_games=100,
                trigger="allocate_batch",
            )


class TestEmitStarvationAlert:
    """Tests for emit_starvation_alert function."""

    def test_handles_publish_errors_gracefully(self):
        """Test router publish failures are swallowed."""
        from app.coordination.selfplay.allocation_events import emit_starvation_alert

        with patch(
            "app.coordination.event_router.publish_sync",
            side_effect=RuntimeError("test error"),
        ):
            emit_starvation_alert(
                config_key="hex8_4p",
                game_count=15,
                tier="ULTRA",
            )

    def test_emits_event_with_starvation_details(self):
        """Test emits event with starvation details."""
        from app.coordination.selfplay.allocation_events import emit_starvation_alert

        mock_publish = MagicMock()
        with patch(
            "app.coordination.event_router.publish_sync",
            mock_publish,
        ):
            emit_starvation_alert(
                config_key="square19_3p",
                game_count=10,
                tier="EMERGENCY",
            )

        assert mock_publish.called
        payload = mock_publish.call_args[0][1]
        assert payload["config_key"] == "square19_3p"
        assert payload["game_count"] == 10
        assert payload["tier"] == "EMERGENCY"
        assert "threshold" in payload
        assert "multiplier" in payload
        assert "timestamp" in payload
        assert mock_publish.call_args.kwargs["source"] == "allocation_events"

    def test_handles_exception_gracefully(self):
        """Test handles exceptions without raising."""
        from app.coordination.selfplay.allocation_events import emit_starvation_alert

        with patch(
            "app.coordination.event_router.publish_sync",
            side_effect=RuntimeError("test error"),
        ):
            emit_starvation_alert(
                config_key="hex8_4p",
                game_count=15,
                tier="ULTRA",
            )


class TestEmitIdleNodeWorkInjected:
    """Tests for emit_idle_node_work_injected function."""

    def test_handles_publish_errors_gracefully(self):
        """Test router publish failures are swallowed."""
        from app.coordination.selfplay.allocation_events import emit_idle_node_work_injected

        with patch(
            "app.coordination.event_router.publish_sync",
            side_effect=RuntimeError("test error"),
        ):
            emit_idle_node_work_injected(
                node_id="worker-1",
                config_key="hex8_2p",
                games=100,
            )

    def test_emits_event_with_injection_details(self):
        """Test emits event with work injection details."""
        from app.coordination.selfplay.allocation_events import emit_idle_node_work_injected

        mock_publish = MagicMock()
        with patch(
            "app.coordination.event_router.publish_sync",
            mock_publish,
        ):
            emit_idle_node_work_injected(
                node_id="vast-12345",
                config_key="square8_4p",
                games=200,
                reason="idle_threshold_exceeded",
            )

        assert mock_publish.called
        payload = mock_publish.call_args[0][1]
        assert payload["node_id"] == "vast-12345"
        assert payload["config_key"] == "square8_4p"
        assert payload["games"] == 200
        assert payload["reason"] == "idle_threshold_exceeded"
        assert "timestamp" in payload
        assert mock_publish.call_args.kwargs["source"] == "allocation_events"

    def test_default_reason(self):
        """Test default reason is applied."""
        from app.coordination.selfplay.allocation_events import emit_idle_node_work_injected

        mock_publish = MagicMock()
        with patch(
            "app.coordination.event_router.publish_sync",
            mock_publish,
        ):
            emit_idle_node_work_injected(
                node_id="worker-1",
                config_key="hex8_2p",
                games=100,
            )

        payload = mock_publish.call_args[0][1]
        assert payload["reason"] == "idle_threshold_exceeded"


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_exist(self):
        """Test that __all__ exports are valid."""
        from app.coordination.selfplay import allocation_events

        for name in allocation_events.__all__:
            assert hasattr(allocation_events, name), f"{name} not in module"

    def test_imports_from_package(self):
        """Test functions can be imported from package."""
        from app.coordination.selfplay import (
            emit_allocation_updated,
            emit_starvation_alert,
            emit_idle_node_work_injected,
        )

        # All imports should be callable
        assert callable(emit_allocation_updated)
        assert callable(emit_starvation_alert)
        assert callable(emit_idle_node_work_injected)

    def test_functions_have_docstrings(self):
        """Test all functions have docstrings."""
        from app.coordination.selfplay.allocation_events import (
            emit_allocation_updated,
            emit_starvation_alert,
            emit_idle_node_work_injected,
        )

        assert emit_allocation_updated.__doc__ is not None
        assert emit_starvation_alert.__doc__ is not None
        assert emit_idle_node_work_injected.__doc__ is not None
