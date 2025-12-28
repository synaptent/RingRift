"""Tests for PipelineEventHandlerMixin.

December 2025: Comprehensive tests for all event handlers in the mixin.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum, auto
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Create a mock PipelineStage enum for testing
class PipelineStage(Enum):
    """Mock pipeline stages for testing."""
    IDLE = auto()
    SELFPLAY = auto()
    SYNC = auto()
    DATA_SYNC = auto()
    NPZ_EXPORT = auto()
    TRAINING = auto()
    EVALUATION = auto()
    PROMOTION = auto()


class TestHelper:
    """Test helper class that includes the mixin for testing."""

    def __init__(self):
        # Initialize all required attributes from the mixin docstring
        self._current_stage = PipelineStage.IDLE
        self._current_iteration = 0
        self._current_board_type: str | None = None
        self._current_num_players: int | None = None
        self.auto_trigger = True
        self.auto_trigger_sync = True
        self.auto_trigger_export = True
        self._paused = False
        self._pause_reason: str | None = None
        self._backpressure_active = False
        self._resource_constraints: dict = {}
        self._stage_metadata: dict = {}
        self._quality_distribution: dict = {}
        self._last_quality_update = 0.0
        self._cache_invalidation_count = 0
        self._pending_cache_refresh = False
        self._active_optimization: str | None = None
        self._optimization_run_id: str | None = None
        self._optimization_start_time = 0.0
        self._stats = {"total_games": 0}

    # Mock methods expected by the mixin
    def _should_process_stage_data_event(self, event_name: str) -> bool:
        return True

    def _get_board_config(self, metadata: dict) -> tuple[str, int]:
        config_key = metadata.get("config_key", "hex8_2p")
        parts = config_key.split("_")
        board_type = parts[0] if parts else "hex8"
        num_players = int(parts[1].replace("p", "")) if len(parts) > 1 else 2
        return board_type, num_players

    def _next_data_event_iteration(self) -> int:
        self._current_iteration += 1
        return self._current_iteration

    def _current_iteration_for_data_event(self) -> int:
        return self._current_iteration

    async def _on_selfplay_complete(self, result: Any) -> None:
        self._last_selfplay_result = result

    async def _on_sync_complete(self, result: Any) -> None:
        self._last_sync_result = result

    async def _on_training_complete(self, result: Any) -> None:
        self._last_training_result = result

    async def _on_training_failed(self, result: Any) -> None:
        self._last_training_failed_result = result

    async def _on_evaluation_complete(self, result: Any) -> None:
        self._last_evaluation_result = result

    async def _on_promotion_complete(self, result: Any) -> None:
        self._last_promotion_result = result

    async def _trigger_orphan_recovery_sync(
        self, source_node: str, config_key: str, orphan_count: int
    ) -> None:
        self._orphan_recovery_triggered = {
            "source_node": source_node,
            "config_key": config_key,
            "orphan_count": orphan_count,
        }

    async def _pause_pipeline(self, reason: str) -> None:
        self._paused = True
        self._pause_reason = reason

    async def _resume_pipeline(self) -> None:
        self._paused = False
        self._pause_reason = None

    def _has_critical_constraints(self) -> bool:
        return any(
            c.get("severity") == "critical"
            for c in self._resource_constraints.values()
        )

    def _transition_to(
        self, stage: PipelineStage, iteration: int, success: bool = True, metadata: dict | None = None
    ) -> None:
        self._current_stage = stage
        self._current_iteration = iteration
        self._last_transition = {
            "stage": stage,
            "iteration": iteration,
            "success": success,
            "metadata": metadata or {},
        }

    async def _auto_trigger_export(self, iteration: int) -> None:
        self._export_triggered = iteration


# Import the mixin after defining TestHelper
from app.coordination.pipeline_event_handler_mixin import PipelineEventHandlerMixin


class MixedTestHelper(PipelineEventHandlerMixin, TestHelper):
    """Concrete class combining mixin with test helper."""

    def __init__(self):
        TestHelper.__init__(self)


@pytest.fixture
def handler():
    """Create a test handler with all required attributes."""
    return MixedTestHelper()


@pytest.fixture
def mock_event():
    """Create a mock event factory."""
    def _make_event(payload: dict | None = None):
        event = SimpleNamespace()
        event.payload = payload or {}
        return event
    return _make_event


# =============================================================================
# Core Pipeline Event Handler Tests
# =============================================================================

class TestCorePipelineEvents:
    """Tests for core pipeline event handlers."""

    @pytest.mark.asyncio
    async def test_on_data_selfplay_complete_basic(self, handler, mock_event):
        """Test basic selfplay complete handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "games_played": 100,
            "success": True,
        })

        await handler._on_data_selfplay_complete(event)

        assert hasattr(handler, "_last_selfplay_result")
        assert handler._last_selfplay_result.games_generated == 100
        assert handler._last_selfplay_result.board_type == "hex8"
        assert handler._last_selfplay_result.num_players == 2

    @pytest.mark.asyncio
    async def test_on_data_selfplay_complete_alternate_keys(self, handler, mock_event):
        """Test selfplay complete with alternate payload keys."""
        event = mock_event({
            "config": "square8_4p",
            "games_generated": 50,
        })

        await handler._on_data_selfplay_complete(event)

        assert handler._last_selfplay_result.games_generated == 50

    @pytest.mark.asyncio
    async def test_on_data_selfplay_complete_no_games(self, handler, mock_event):
        """Test selfplay complete with no games - should skip."""
        event = mock_event({
            "config_key": "hex8_2p",
            "games_played": 0,
        })

        await handler._on_data_selfplay_complete(event)

        # Should not process when no games
        assert not hasattr(handler, "_last_selfplay_result")

    @pytest.mark.asyncio
    async def test_on_data_sync_completed(self, handler, mock_event):
        """Test sync completed handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "games_synced": 75,
        })

        await handler._on_data_sync_completed(event)

        assert hasattr(handler, "_last_sync_result")
        assert handler._last_sync_result.success is True
        assert handler._last_sync_result.metadata.get("games_synced") == 75

    @pytest.mark.asyncio
    async def test_on_data_sync_failed(self, handler, mock_event):
        """Test sync failed handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "error": "Connection refused",
        })

        await handler._on_data_sync_failed(event)

        assert handler._last_sync_result.success is False
        assert handler._last_sync_result.error == "Connection refused"

    @pytest.mark.asyncio
    async def test_on_data_training_completed(self, handler, mock_event):
        """Test training completed handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "checkpoint_path": "/models/test.pth",
            "final_train_loss": 0.5,
            "final_val_loss": 0.6,
            "board_type": "hex8",
            "num_players": 2,
        })

        await handler._on_data_training_completed(event)

        assert handler._last_training_result.success is True
        assert handler._last_training_result.model_path == "/models/test.pth"
        assert handler._last_training_result.train_loss == 0.5
        assert handler._last_training_result.val_loss == 0.6

    @pytest.mark.asyncio
    async def test_on_data_training_completed_extracts_model_id(self, handler, mock_event):
        """Test that model_id is extracted from path if not provided."""
        event = mock_event({
            "config_key": "hex8_2p",
            "model_path": "/models/canonical_hex8_2p.pth",
        })

        await handler._on_data_training_completed(event)

        assert handler._last_training_result.model_id == "canonical_hex8_2p"

    @pytest.mark.asyncio
    async def test_on_data_training_failed(self, handler, mock_event):
        """Test training failed handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "error": "Out of memory",
        })

        await handler._on_data_training_failed(event)

        assert handler._last_training_failed_result.success is False
        assert handler._last_training_failed_result.error == "Out of memory"

    @pytest.mark.asyncio
    async def test_on_data_evaluation_completed(self, handler, mock_event):
        """Test evaluation completed handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "win_rate": 0.75,
            "elo_delta": 50,
            "model_path": "/models/test.pth",
        })

        await handler._on_data_evaluation_completed(event)

        assert handler._last_evaluation_result.success is True
        assert handler._last_evaluation_result.win_rate == 0.75
        assert handler._last_evaluation_result.elo_delta == 50

    @pytest.mark.asyncio
    async def test_on_data_evaluation_failed(self, handler, mock_event):
        """Test evaluation failed handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "error": "Gauntlet timeout",
        })

        await handler._on_data_evaluation_failed(event)

        assert handler._last_evaluation_result.success is False
        assert handler._last_evaluation_result.error == "Gauntlet timeout"

    @pytest.mark.asyncio
    async def test_on_data_model_promoted(self, handler, mock_event):
        """Test model promoted handling."""
        event = mock_event({
            "config_key": "hex8_2p",
            "promoted": True,
            "promotion_reason": "Win rate threshold exceeded",
            "board_type": "hex8",
            "num_players": 2,
        })

        await handler._on_data_model_promoted(event)

        assert handler._last_promotion_result.promoted is True
        assert handler._last_promotion_result.promotion_reason == "Win rate threshold exceeded"


# =============================================================================
# Orphan Games Event Handler Tests
# =============================================================================

class TestOrphanGamesEvents:
    """Tests for orphan games event handlers."""

    @pytest.mark.asyncio
    async def test_on_orphan_games_detected(self, handler, mock_event):
        """Test orphan games detection triggers sync."""
        event = mock_event({
            "orphan_count": 50,
            "source_node": "vast-12345",
            "config_key": "hex8_2p",
        })

        await handler._on_orphan_games_detected(event)

        assert handler._orphan_games_pending == 50
        assert handler._orphan_recovery_triggered["orphan_count"] == 50

    @pytest.mark.asyncio
    async def test_on_orphan_games_detected_zero_count(self, handler, mock_event):
        """Test orphan games detection with zero count skips processing."""
        event = mock_event({
            "orphan_count": 0,
            "source_node": "vast-12345",
        })

        await handler._on_orphan_games_detected(event)

        assert not hasattr(handler, "_orphan_recovery_triggered")

    @pytest.mark.asyncio
    async def test_on_orphan_games_detected_auto_trigger_disabled(self, handler, mock_event):
        """Test orphan games detection respects auto_trigger setting."""
        handler.auto_trigger = False
        event = mock_event({
            "orphan_count": 50,
            "source_node": "vast-12345",
            "config_key": "hex8_2p",
        })

        await handler._on_orphan_games_detected(event)

        assert handler._orphan_games_pending == 50
        assert not hasattr(handler, "_orphan_recovery_triggered")

    @pytest.mark.asyncio
    async def test_on_orphan_games_registered(self, handler, mock_event):
        """Test orphan games registration updates pending count."""
        handler._orphan_games_pending = 100
        event = mock_event({
            "registered_count": 50,
            "config_key": "hex8_2p",
            "board_type": "hex8",
            "num_players": 2,
        })

        with patch("app.distributed.data_events.emit_data_event", new_callable=AsyncMock):
            await handler._on_orphan_games_registered(event)

        assert handler._orphan_games_pending == 50

    @pytest.mark.asyncio
    async def test_on_orphan_games_registered_zero_count(self, handler, mock_event):
        """Test orphan games registration with zero count skips processing."""
        event = mock_event({
            "registered_count": 0,
        })

        await handler._on_orphan_games_registered(event)

        assert not hasattr(handler, "_orphan_games_pending") or handler._orphan_games_pending == 0


# =============================================================================
# Consolidation Event Handler Tests
# =============================================================================

class TestConsolidationEvents:
    """Tests for consolidation event handlers."""

    @pytest.mark.asyncio
    async def test_on_consolidation_started(self, handler, mock_event):
        """Test consolidation started tracking."""
        event = mock_event({
            "board_type": "hex8",
            "num_players": 2,
        })

        await handler._on_consolidation_started(event)

        assert "hex8_2p" in handler._consolidations_in_progress

    @pytest.mark.asyncio
    async def test_on_consolidation_complete(self, handler, mock_event):
        """Test consolidation complete handling."""
        handler._consolidations_in_progress = {"hex8_2p"}
        event = mock_event({
            "board_type": "hex8",
            "num_players": 2,
            "games_consolidated": 100,
            "canonical_db": "/data/games/canonical_hex8_2p.db",
        })

        with patch("app.distributed.data_events.emit_data_event", new_callable=AsyncMock):
            await handler._on_consolidation_complete(event)

        assert "hex8_2p" not in handler._consolidations_in_progress


# =============================================================================
# Repair Event Handler Tests
# =============================================================================

class TestRepairEvents:
    """Tests for repair event handlers."""

    @pytest.mark.asyncio
    async def test_on_repair_completed(self, handler, mock_event):
        """Test repair completed handling."""
        handler._current_stage = PipelineStage.SELFPLAY
        event = mock_event({
            "repair_type": "game_data",
            "files_repaired": 10,
            "source_node": "vast-12345",
        })

        with patch("app.distributed.data_events.emit_data_event", new_callable=AsyncMock):
            await handler._on_repair_completed(event)

        # Should not raise and should log

    @pytest.mark.asyncio
    async def test_on_repair_failed(self, handler, mock_event):
        """Test repair failed handling increments error count."""
        event = mock_event({
            "repair_type": "game_data",
            "error": "Node unreachable",
            "source_node": "vast-12345",
        })

        await handler._on_repair_failed(event)

        assert handler._repair_failure_count == 1

    @pytest.mark.asyncio
    async def test_on_repair_failed_cumulative(self, handler, mock_event):
        """Test repair failures are cumulative."""
        handler._repair_failure_count = 5
        event = mock_event({
            "repair_type": "game_data",
            "error": "Timeout",
        })

        await handler._on_repair_failed(event)

        assert handler._repair_failure_count == 6


# =============================================================================
# Quality and Curriculum Event Handler Tests
# =============================================================================

class TestQualityAndCurriculumEvents:
    """Tests for quality and curriculum event handlers."""

    @pytest.mark.asyncio
    async def test_on_quality_score_updated(self, handler, mock_event):
        """Test quality score tracking."""
        event = mock_event({
            "game_id": "game-123",
            "quality_score": 0.8,
            "config_key": "hex8_2p",
        })

        await handler._on_quality_score_updated(event)

        assert len(handler._recent_quality_scores) == 1
        assert handler._recent_quality_scores[0] == 0.8

    @pytest.mark.asyncio
    async def test_on_quality_score_updated_rolling_buffer(self, handler, mock_event):
        """Test quality score rolling buffer limits to 100 entries."""
        # Add 110 quality scores
        for i in range(110):
            event = mock_event({
                "quality_score": 0.5 + (i * 0.001),
            })
            await handler._on_quality_score_updated(event)

        assert len(handler._recent_quality_scores) == 100

    @pytest.mark.asyncio
    async def test_on_curriculum_rebalanced(self, handler, mock_event):
        """Test curriculum rebalancing updates weights."""
        event = mock_event({
            "config_key": "hex8_2p",
            "weights": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
        })

        await handler._on_curriculum_rebalanced(event)

        assert handler._curriculum_weights["hex8_2p"] == {"easy": 0.3, "medium": 0.5, "hard": 0.2}

    @pytest.mark.asyncio
    async def test_on_curriculum_advanced(self, handler, mock_event):
        """Test curriculum advancement tracking."""
        event = mock_event({
            "config_key": "hex8_2p",
            "tier": 3,
            "old_tier": 2,
        })

        await handler._on_curriculum_advanced(event)

        assert handler._curriculum_tiers["hex8_2p"] == 3


# =============================================================================
# Resource Constraint Event Handler Tests
# =============================================================================

class TestResourceConstraintEvents:
    """Tests for resource constraint event handlers."""

    @pytest.mark.asyncio
    async def test_on_resource_constraint_detected(self, handler, mock_event):
        """Test resource constraint tracking."""
        event = mock_event({
            "resource_type": "gpu_memory",
            "severity": "warning",
            "current_value": 85,
            "threshold": 90,
            "node_id": "gpu-node-1",
        })

        await handler._on_resource_constraint_detected(event)

        assert "gpu_memory" in handler._resource_constraints
        assert handler._resource_constraints["gpu_memory"]["severity"] == "warning"
        assert not handler._paused

    @pytest.mark.asyncio
    async def test_on_resource_constraint_critical_pauses(self, handler, mock_event):
        """Test critical resource constraint pauses pipeline in training stage."""
        handler._current_stage = PipelineStage.TRAINING
        event = mock_event({
            "resource_type": "disk_space",
            "severity": "critical",
            "current_value": 95,
            "threshold": 90,
            "node_id": "storage-node",
        })

        await handler._on_resource_constraint_detected(event)

        assert handler._paused is True
        assert "disk_space" in handler._pause_reason

    @pytest.mark.asyncio
    async def test_on_backpressure_activated(self, handler, mock_event):
        """Test backpressure activation."""
        event = mock_event({
            "source": "sync_coordinator",
            "level": "medium",
        })

        await handler._on_backpressure_activated(event)

        assert handler._backpressure_active is True
        assert not handler._paused  # Medium level doesn't pause

    @pytest.mark.asyncio
    async def test_on_backpressure_activated_high_pauses(self, handler, mock_event):
        """Test high backpressure pauses pipeline."""
        event = mock_event({
            "source": "sync_coordinator",
            "level": "high",
        })

        await handler._on_backpressure_activated(event)

        assert handler._paused is True
        assert "Backpressure" in handler._pause_reason

    @pytest.mark.asyncio
    async def test_on_backpressure_released(self, handler, mock_event):
        """Test backpressure release clears flag."""
        handler._backpressure_active = True
        event = mock_event({
            "source": "sync_coordinator",
        })

        await handler._on_backpressure_released(event)

        assert handler._backpressure_active is False

    @pytest.mark.asyncio
    async def test_on_backpressure_released_resumes_if_paused(self, handler, mock_event):
        """Test backpressure release resumes pipeline if paused due to backpressure."""
        handler._backpressure_active = True
        handler._paused = True
        handler._pause_reason = "Backpressure from sync: high"
        event = mock_event({
            "source": "sync_coordinator",
        })

        await handler._on_backpressure_released(event)

        assert handler._paused is False


# =============================================================================
# Cache and Optimization Event Handler Tests
# =============================================================================

class TestCacheAndOptimizationEvents:
    """Tests for cache and optimization event handlers."""

    @pytest.mark.asyncio
    async def test_on_quality_distribution_changed(self, handler, mock_event):
        """Test quality distribution tracking."""
        event = mock_event({
            "distribution": {"high": 0.3, "medium": 0.5, "low": 0.2},
        })

        await handler._on_quality_distribution_changed(event)

        assert handler._quality_distribution == {"high": 0.3, "medium": 0.5, "low": 0.2}
        assert handler._last_quality_update > 0

    @pytest.mark.asyncio
    async def test_on_cache_invalidated_model(self, handler, mock_event):
        """Test model cache invalidation sets pending refresh."""
        event = mock_event({
            "invalidation_type": "model",
            "count": 5,
            "target_id": "hex8_2p",
        })

        await handler._on_cache_invalidated(event)

        assert handler._cache_invalidation_count == 5
        assert handler._pending_cache_refresh is True

    @pytest.mark.asyncio
    async def test_on_cache_invalidated_other_type(self, handler, mock_event):
        """Test non-model cache invalidation doesn't set pending refresh."""
        event = mock_event({
            "invalidation_type": "game",
            "count": 10,
            "target_id": "game-123",
        })

        await handler._on_cache_invalidated(event)

        assert handler._cache_invalidation_count == 10
        assert handler._pending_cache_refresh is False

    @pytest.mark.asyncio
    async def test_on_optimization_triggered_cmaes(self, handler, mock_event):
        """Test CMA-ES optimization tracking."""
        event = mock_event({
            "run_id": "cmaes-run-123",
            "reason": "plateau_detected",
        })
        event.event_type = SimpleNamespace(value="cmaes_triggered")

        await handler._on_optimization_triggered(event)

        assert handler._active_optimization == "cmaes"
        assert handler._optimization_run_id == "cmaes-run-123"

    @pytest.mark.asyncio
    async def test_on_optimization_triggered_nas(self, handler, mock_event):
        """Test NAS optimization tracking."""
        event = mock_event({
            "run_id": "nas-run-456",
            "reason": "architecture_search",
        })
        event.event_type = SimpleNamespace(value="nas_triggered")

        await handler._on_optimization_triggered(event)

        assert handler._active_optimization == "nas"
        assert handler._optimization_run_id == "nas-run-456"


# =============================================================================
# Pipeline Status Event Handler Tests
# =============================================================================

class TestPipelineStatusEvents:
    """Tests for pipeline status event handlers."""

    @pytest.mark.asyncio
    async def test_on_new_games_available(self, handler, mock_event):
        """Test new games availability tracking."""
        event = mock_event({
            "config_key": "hex8_2p",
            "new_games": 50,
            "source": "selfplay",
        })

        await handler._on_new_games_available(event)

        assert handler._new_games_tracker.get("hex8_2p") == 50
        assert handler._stats["total_games"] == 50

    @pytest.mark.asyncio
    async def test_on_new_games_available_cumulative(self, handler, mock_event):
        """Test new games tracking is cumulative."""
        handler._new_games_tracker = {"hex8_2p": 100}
        handler._stats = {"total_games": 100}
        event = mock_event({
            "config_key": "hex8_2p",
            "new_games": 50,
        })

        await handler._on_new_games_available(event)

        assert handler._new_games_tracker["hex8_2p"] == 150
        assert handler._stats["total_games"] == 150

    @pytest.mark.asyncio
    async def test_on_regression_detected(self, handler, mock_event):
        """Test regression detection tracking."""
        event = mock_event({
            "config_key": "hex8_2p",
            "severity": "moderate",
            "elo_change": -50,
            "reason": "Win rate dropped",
        })

        await handler._on_regression_detected(event)

        assert handler._regression_count == 1
        assert handler._last_regression["severity"] == "moderate"

    @pytest.mark.asyncio
    async def test_on_regression_detected_severe_updates_metadata(self, handler, mock_event):
        """Test severe regression updates stage metadata."""
        event = mock_event({
            "config_key": "hex8_2p",
            "severity": "severe",
            "elo_change": -100,
        })

        await handler._on_regression_detected(event)

        assert handler._stage_metadata["regression_detected"] is True
        assert handler._stage_metadata["regression_severity"] == "severe"

    @pytest.mark.asyncio
    async def test_on_promotion_failed(self, handler, mock_event):
        """Test promotion failure tracking."""
        handler._current_stage = PipelineStage.PROMOTION
        event = mock_event({
            "config_key": "hex8_2p",
            "reason": "Below threshold",
            "model_path": "/models/test.pth",
        })

        await handler._on_promotion_failed(event)

        assert handler._promotion_failure_count == 1
        assert handler._current_stage == PipelineStage.EVALUATION  # Transition back

    @pytest.mark.asyncio
    async def test_on_promotion_candidate(self, handler, mock_event):
        """Test promotion candidate tracking."""
        handler._current_stage = PipelineStage.EVALUATION
        event = mock_event({
            "model_id": "model-123",
            "board_type": "hex8",
            "num_players": 2,
            "win_rate_vs_heuristic": 0.75,
        })

        await handler._on_promotion_candidate(event)

        assert len(handler._promotion_candidates) == 1
        assert handler._promotion_candidates[0]["model_id"] == "model-123"
        assert handler._stage_metadata["candidates"] == 1

    @pytest.mark.asyncio
    async def test_on_database_created(self, handler, mock_event):
        """Test database creation tracking."""
        event = mock_event({
            "db_path": "/data/games/selfplay.db",
            "board_type": "hex8",
            "num_players": 2,
        })

        await handler._on_database_created(event)

        assert len(handler._new_databases) == 1
        assert handler._new_databases[0]["db_path"] == "/data/games/selfplay.db"

    @pytest.mark.asyncio
    async def test_on_work_queued(self, handler, mock_event):
        """Test work queue tracking."""
        event = mock_event({
            "work_type": "selfplay",
            "config": "hex8_2p",
        })

        await handler._on_work_queued(event)

        assert handler._queued_work_count == 1


# =============================================================================
# Sync and Feedback Event Handler Tests
# =============================================================================

class TestSyncAndFeedbackEvents:
    """Tests for sync and feedback event handlers."""

    @pytest.mark.asyncio
    async def test_on_game_synced(self, handler, mock_event):
        """Test game sync tracking."""
        handler._current_stage = PipelineStage.IDLE
        event = mock_event({
            "node_id": "vast-12345",
            "games_pushed": 25,
            "target_nodes": ["node-1", "node-2"],
            "is_ephemeral": True,
        })

        await handler._on_game_synced(event)

        assert handler._games_synced_count == 25

    @pytest.mark.asyncio
    async def test_on_game_synced_triggers_export_in_sync_stage(self, handler, mock_event):
        """Test game sync triggers export when in SYNC stage."""
        handler._current_stage = PipelineStage.SYNC
        event = mock_event({
            "node_id": "vast-12345",
            "games_pushed": 25,
            "target_nodes": ["node-1"],
        })

        await handler._on_game_synced(event)

        assert handler._current_stage == PipelineStage.NPZ_EXPORT
        assert handler._export_triggered == handler._current_iteration

    @pytest.mark.asyncio
    async def test_on_exploration_boost(self, handler, mock_event):
        """Test exploration boost tracking."""
        event = mock_event({
            "config_key": "hex8_2p",
            "boost_factor": 1.5,
            "reason": "plateau_detected",
        })

        await handler._on_exploration_boost(event)

        assert handler._exploration_boost_count == 1

    @pytest.mark.asyncio
    async def test_on_sync_triggered(self, handler, mock_event):
        """Test sync trigger tracking."""
        handler._current_stage = PipelineStage.IDLE
        event = mock_event({
            "reason": "stale_data",
            "config_key": "hex8_2p",
            "data_age_hours": 2.5,
            "source": "training_freshness",
        })

        await handler._on_sync_triggered(event)

        assert handler._sync_trigger_count == 1
        assert handler._current_stage == PipelineStage.SYNC

    @pytest.mark.asyncio
    async def test_on_data_stale(self, handler, mock_event):
        """Test stale data tracking."""
        event = mock_event({
            "config_key": "hex8_2p",
            "data_age_hours": 3.0,
            "max_age_hours": 1.0,
            "source": "train_cli",
        })

        with patch("app.coordination.pipeline_event_handler_mixin.get_sync_facade", return_value=None):
            await handler._on_data_stale(event)

        assert handler._stale_data_count == 1


# =============================================================================
# S3 Backup Event Handler Tests
# =============================================================================

class TestS3BackupEvents:
    """Tests for S3 backup event handlers."""

    @pytest.mark.asyncio
    async def test_on_s3_backup_completed(self, handler, mock_event):
        """Test S3 backup completion tracking."""
        event = mock_event({
            "uploaded_count": 50,
            "bucket": "ringrift-models",
            "duration_seconds": 120.5,
            "promotions": ["model-1", "model-2"],
        })

        await handler._on_s3_backup_completed(event)

        assert handler._s3_backup_stats["backups_completed"] == 1
        assert handler._s3_backup_stats["files_backed_up"] == 50
        assert handler._s3_backup_stats["last_backup_time"] > 0

    @pytest.mark.asyncio
    async def test_on_s3_backup_completed_cumulative(self, handler, mock_event):
        """Test S3 backup stats are cumulative."""
        handler._s3_backup_stats = {
            "backups_completed": 5,
            "files_backed_up": 100,
            "last_backup_time": time.time() - 3600,
        }
        event = mock_event({
            "uploaded_count": 25,
            "bucket": "ringrift-models",
            "duration_seconds": 60,
            "promotions": [],
        })

        await handler._on_s3_backup_completed(event)

        assert handler._s3_backup_stats["backups_completed"] == 6
        assert handler._s3_backup_stats["files_backed_up"] == 125
