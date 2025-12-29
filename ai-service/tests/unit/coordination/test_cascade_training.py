"""Tests for cascade_training.py - Multiplayer bootstrapping orchestrator.

This module tests the cascade training system that progressively trains
2p → 3p → 4p models using transfer learning.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.cascade_training import (
    CascadeConfig,
    CascadeStage,
    CascadeState,
    CascadeTrainingOrchestrator,
    get_cascade_orchestrator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset singleton before each test."""
    CascadeTrainingOrchestrator.reset_instance()
    yield
    CascadeTrainingOrchestrator.reset_instance()


@pytest.fixture
def config() -> CascadeConfig:
    """Create a test configuration."""
    return CascadeConfig(
        min_elo_for_transfer=1200.0,
        min_games_for_training=500,
        bootstrap_selfplay_multiplier=1.5,
        board_types=["hex8", "square8"],
        models_dir="models",
        data_dir="data/games",
        check_interval=60.0,
    )


@pytest.fixture
def orchestrator(config: CascadeConfig) -> CascadeTrainingOrchestrator:
    """Create a test orchestrator."""
    return CascadeTrainingOrchestrator(config)


# =============================================================================
# CascadeStage Tests
# =============================================================================


class TestCascadeStage:
    """Tests for CascadeStage enum."""

    def test_all_stages_exist(self) -> None:
        """Test all expected stages exist."""
        assert CascadeStage.NOT_STARTED.value == "not_started"
        assert CascadeStage.TRAINING_2P.value == "training_2p"
        assert CascadeStage.READY_2P.value == "ready_2p"
        assert CascadeStage.TRAINING_3P.value == "training_3p"
        assert CascadeStage.READY_3P.value == "ready_3p"
        assert CascadeStage.TRAINING_4P.value == "training_4p"
        assert CascadeStage.COMPLETE.value == "complete"

    def test_stage_count(self) -> None:
        """Test correct number of stages."""
        assert len(CascadeStage) == 7


# =============================================================================
# CascadeState Tests
# =============================================================================


class TestCascadeState:
    """Tests for CascadeState dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        state = CascadeState(board_type="hex8")
        assert state.board_type == "hex8"
        assert state.stage == CascadeStage.NOT_STARTED
        assert state.model_2p is None
        assert state.model_3p is None
        assert state.model_4p is None
        assert state.elo_2p == 0.0
        assert state.elo_3p == 0.0
        assert state.elo_4p == 0.0
        assert state.min_elo_for_transfer == 1200.0
        assert state.min_games_for_transfer == 1000

    def test_can_transfer_to_3p_success(self) -> None:
        """Test successful 2p→3p transfer eligibility."""
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.READY_2P,
            model_2p="models/hex8_2p.pth",
            elo_2p=1300.0,
        )
        assert state.can_transfer_to_3p() is True

    def test_can_transfer_to_3p_no_model(self) -> None:
        """Test 2p→3p blocked when no model."""
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.READY_2P,
            elo_2p=1300.0,
        )
        assert state.can_transfer_to_3p() is False

    def test_can_transfer_to_3p_low_elo(self) -> None:
        """Test 2p→3p blocked when Elo too low."""
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.READY_2P,
            model_2p="models/hex8_2p.pth",
            elo_2p=1100.0,  # Below 1200 threshold
        )
        assert state.can_transfer_to_3p() is False

    def test_can_transfer_to_3p_wrong_stage(self) -> None:
        """Test 2p→3p blocked when wrong stage."""
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.NOT_STARTED,  # Not READY_2P
            model_2p="models/hex8_2p.pth",
            elo_2p=1300.0,
        )
        assert state.can_transfer_to_3p() is False

    def test_can_transfer_to_4p_success(self) -> None:
        """Test successful 3p→4p transfer eligibility."""
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.READY_3P,
            model_3p="models/hex8_3p.pth",
            elo_3p=1250.0,
        )
        assert state.can_transfer_to_4p() is True

    def test_can_transfer_to_4p_no_model(self) -> None:
        """Test 3p→4p blocked when no model."""
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.READY_3P,
            elo_3p=1250.0,
        )
        assert state.can_transfer_to_4p() is False

    def test_to_dict(self) -> None:
        """Test dictionary serialization."""
        now = datetime.now()
        state = CascadeState(
            board_type="hex8",
            stage=CascadeStage.READY_2P,
            model_2p="models/hex8_2p.pth",
            elo_2p=1350.0,
            training_started=now,
            last_updated=now,
        )
        result = state.to_dict()

        assert result["board_type"] == "hex8"
        assert result["stage"] == "ready_2p"
        assert result["model_2p"] == "models/hex8_2p.pth"
        assert result["elo_2p"] == 1350.0
        assert result["training_started"] == now.isoformat()
        assert result["last_updated"] == now.isoformat()

    def test_to_dict_no_training_started(self) -> None:
        """Test serialization when training_started is None."""
        state = CascadeState(board_type="hex8")
        result = state.to_dict()
        assert result["training_started"] is None


# =============================================================================
# CascadeConfig Tests
# =============================================================================


class TestCascadeConfig:
    """Tests for CascadeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CascadeConfig()
        assert config.min_elo_for_transfer == 1200.0
        assert config.min_games_for_training == 500
        assert config.bootstrap_selfplay_multiplier == 1.5
        assert "hex8" in config.board_types
        assert "square8" in config.board_types
        assert config.models_dir == "models"
        assert config.data_dir == "data/games"
        assert config.check_interval == 300.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CascadeConfig(
            min_elo_for_transfer=1500.0,
            min_games_for_training=1000,
            board_types=["hex8"],
        )
        assert config.min_elo_for_transfer == 1500.0
        assert config.min_games_for_training == 1000
        assert config.board_types == ["hex8"]


# =============================================================================
# CascadeTrainingOrchestrator Tests
# =============================================================================


class TestCascadeTrainingOrchestrator:
    """Tests for CascadeTrainingOrchestrator class."""

    def test_initialization(self, orchestrator: CascadeTrainingOrchestrator, config: CascadeConfig) -> None:
        """Test orchestrator initialization."""
        assert orchestrator.config == config
        assert "hex8" in orchestrator._states
        assert "square8" in orchestrator._states
        assert orchestrator._states["hex8"].stage == CascadeStage.NOT_STARTED

    def test_default_config(self) -> None:
        """Test initialization with default config."""
        orch = CascadeTrainingOrchestrator()
        assert orch.config.min_elo_for_transfer == 1200.0
        assert len(orch._states) == 4  # All canonical boards

    def test_singleton_pattern(self) -> None:
        """Test singleton get_instance."""
        orch1 = CascadeTrainingOrchestrator.get_instance()
        orch2 = CascadeTrainingOrchestrator.get_instance()
        assert orch1 is orch2

    def test_singleton_reset(self) -> None:
        """Test singleton reset_instance."""
        orch1 = CascadeTrainingOrchestrator.get_instance()
        CascadeTrainingOrchestrator.reset_instance()
        orch2 = CascadeTrainingOrchestrator.get_instance()
        assert orch1 is not orch2

    def test_event_subscriptions(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test event subscription setup."""
        subs = orchestrator._get_event_subscriptions()
        assert "TRAINING_COMPLETED" in subs
        assert "EVALUATION_COMPLETED" in subs
        assert "MODEL_PROMOTED" in subs
        assert "ELO_UPDATED" in subs
        assert "CASCADE_TRANSFER_TRIGGERED" in subs


class TestEventHandlers:
    """Tests for event handler methods."""

    @pytest.mark.asyncio
    async def test_on_training_completed_2p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling 2p training completion."""
        orchestrator._states["hex8"].stage = CascadeStage.TRAINING_2P

        event = {
            "config_key": "hex8_2p",
            "model_path": "models/hex8_2p.pth",
        }
        await orchestrator._on_training_completed(event)

        assert orchestrator._states["hex8"].model_2p == "models/hex8_2p.pth"
        assert orchestrator._states["hex8"].stage == CascadeStage.READY_2P

    @pytest.mark.asyncio
    async def test_on_training_completed_3p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling 3p training completion."""
        orchestrator._states["hex8"].stage = CascadeStage.TRAINING_3P

        event = {
            "config_key": "hex8_3p",
            "model_path": "models/hex8_3p.pth",
        }
        await orchestrator._on_training_completed(event)

        assert orchestrator._states["hex8"].model_3p == "models/hex8_3p.pth"
        assert orchestrator._states["hex8"].stage == CascadeStage.READY_3P

    @pytest.mark.asyncio
    async def test_on_training_completed_4p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling 4p training completion (cascade complete)."""
        orchestrator._states["hex8"].stage = CascadeStage.TRAINING_4P

        event = {
            "config_key": "hex8_4p",
            "model_path": "models/hex8_4p.pth",
        }
        await orchestrator._on_training_completed(event)

        assert orchestrator._states["hex8"].model_4p == "models/hex8_4p.pth"
        assert orchestrator._states["hex8"].stage == CascadeStage.COMPLETE

    @pytest.mark.asyncio
    async def test_on_training_completed_invalid_config(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling invalid config_key."""
        event = {"config_key": "", "model_path": "models/test.pth"}
        await orchestrator._on_training_completed(event)  # Should not raise

        event = {"config_key": "invalid", "model_path": "models/test.pth"}
        await orchestrator._on_training_completed(event)  # Should not raise

    @pytest.mark.asyncio
    async def test_on_training_completed_unknown_board(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling unknown board type."""
        event = {
            "config_key": "unknown_2p",
            "model_path": "models/unknown_2p.pth",
        }
        await orchestrator._on_training_completed(event)  # Should not raise

    @pytest.mark.asyncio
    async def test_on_elo_updated_2p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling Elo update for 2p."""
        event = {"config_key": "hex8_2p", "elo": 1350.0}
        await orchestrator._on_elo_updated(event)
        assert orchestrator._states["hex8"].elo_2p == 1350.0

    @pytest.mark.asyncio
    async def test_on_elo_updated_triggers_transfer(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test Elo update triggers cascade transfer."""
        state = orchestrator._states["hex8"]
        state.stage = CascadeStage.READY_2P
        state.model_2p = "models/hex8_2p.pth"
        state.min_elo_for_transfer = 1200.0

        with patch.object(orchestrator, "_trigger_transfer", new_callable=AsyncMock) as mock_trigger:
            event = {"config_key": "hex8_2p", "elo": 1250.0}
            await orchestrator._on_elo_updated(event)
            mock_trigger.assert_called_once_with("hex8", 2, 3)

    @pytest.mark.asyncio
    async def test_on_model_promoted(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test handling model promotion event."""
        event = {
            "config_key": "hex8_2p",
            "model_path": "models/canonical_hex8_2p.pth",
        }
        await orchestrator._on_model_promoted(event)
        assert orchestrator._states["hex8"].model_2p == "models/canonical_hex8_2p.pth"

    @pytest.mark.asyncio
    async def test_on_evaluation_completed(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test evaluation completed updates Elo."""
        event = {"config_key": "square8_3p", "elo": 1400.0}
        await orchestrator._on_evaluation_completed(event)
        assert orchestrator._states["square8"].elo_3p == 1400.0


class TestCascadeStatus:
    """Tests for cascade status methods."""

    def test_get_cascade_status(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test getting cascade status for a board type."""
        status = orchestrator.get_cascade_status("hex8")
        assert status is not None
        assert status.board_type == "hex8"

    def test_get_cascade_status_unknown(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test getting status for unknown board type."""
        status = orchestrator.get_cascade_status("unknown")
        assert status is None

    def test_get_all_cascade_status(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test getting all cascade statuses."""
        statuses = orchestrator.get_all_cascade_status()
        assert "hex8" in statuses
        assert "square8" in statuses
        assert statuses["hex8"]["stage"] == "not_started"


class TestBootstrapPriority:
    """Tests for bootstrap priority calculation."""

    def test_priority_2p_not_started(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test 2p config gets boost when not started."""
        orchestrator._states["hex8"].stage = CascadeStage.NOT_STARTED
        priority = orchestrator.get_bootstrap_priority("hex8_2p")
        assert priority == 1.5  # Boosted

    def test_priority_2p_training(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test 2p config gets boost when training."""
        orchestrator._states["hex8"].stage = CascadeStage.TRAINING_2P
        priority = orchestrator.get_bootstrap_priority("hex8_2p")
        assert priority == 1.5  # Boosted

    def test_priority_3p_ready_2p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test 3p config gets boost after 2p ready."""
        orchestrator._states["hex8"].stage = CascadeStage.READY_2P
        priority = orchestrator.get_bootstrap_priority("hex8_3p")
        assert priority == 1.5  # Boosted

    def test_priority_4p_ready_3p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test 4p config gets boost after 3p ready."""
        orchestrator._states["hex8"].stage = CascadeStage.READY_3P
        priority = orchestrator.get_bootstrap_priority("hex8_4p")
        assert priority == 1.5  # Boosted

    def test_priority_no_boost_complete(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test no boost when cascade complete."""
        orchestrator._states["hex8"].stage = CascadeStage.COMPLETE
        assert orchestrator.get_bootstrap_priority("hex8_2p") == 1.0
        assert orchestrator.get_bootstrap_priority("hex8_3p") == 1.0
        assert orchestrator.get_bootstrap_priority("hex8_4p") == 1.0

    def test_priority_invalid_config(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test invalid config returns default priority."""
        assert orchestrator.get_bootstrap_priority("invalid") == 1.0
        assert orchestrator.get_bootstrap_priority("hex8") == 1.0  # Missing player count
        assert orchestrator.get_bootstrap_priority("unknown_2p") == 1.0


class TestHealthCheck:
    """Tests for health check method."""

    def test_health_check_all_not_started(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test health check when all not started."""
        result = orchestrator.health_check()
        assert result.healthy is True
        assert "0/" in result.message  # 0 complete
        assert result.details["complete"] == 0

    def test_health_check_some_complete(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test health check with some cascades complete."""
        orchestrator._states["hex8"].stage = CascadeStage.COMPLETE
        result = orchestrator.health_check()
        assert result.healthy is True
        assert result.details["complete"] == 1

    def test_health_check_in_progress(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test health check with cascade in progress."""
        orchestrator._states["hex8"].stage = CascadeStage.TRAINING_3P
        result = orchestrator.health_check()
        assert result.details["in_progress"] == 1


class TestTriggerTransfer:
    """Tests for transfer triggering."""

    @pytest.mark.asyncio
    async def test_trigger_transfer_2p_to_3p(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test triggering 2p→3p transfer."""
        orchestrator._states["hex8"].model_2p = "models/hex8_2p.pth"

        with patch.object(orchestrator, "_emit_event", new_callable=AsyncMock) as mock_emit:
            await orchestrator._trigger_transfer("hex8", 2, 3)

            mock_emit.assert_called_once()
            args = mock_emit.call_args[0]
            assert args[0] == "CASCADE_TRANSFER_TRIGGERED"
            assert args[1]["board_type"] == "hex8"
            assert args[1]["source_players"] == 2
            assert args[1]["target_players"] == 3

    @pytest.mark.asyncio
    async def test_trigger_transfer_no_source_model(self, orchestrator: CascadeTrainingOrchestrator) -> None:
        """Test transfer not triggered without source model."""
        orchestrator._states["hex8"].model_2p = None

        with patch.object(orchestrator, "_emit_event", new_callable=AsyncMock) as mock_emit:
            await orchestrator._trigger_transfer("hex8", 2, 3)
            mock_emit.assert_not_called()


# =============================================================================
# Module-level Function Tests
# =============================================================================


class TestGetCascadeOrchestrator:
    """Tests for get_cascade_orchestrator function."""

    def test_returns_singleton(self) -> None:
        """Test get_cascade_orchestrator returns singleton."""
        orch1 = get_cascade_orchestrator()
        orch2 = get_cascade_orchestrator()
        assert orch1 is orch2

    def test_with_config(self) -> None:
        """Test get_cascade_orchestrator with custom config."""
        config = CascadeConfig(board_types=["hex8"])
        orch = get_cascade_orchestrator(config)
        assert "hex8" in orch._states
