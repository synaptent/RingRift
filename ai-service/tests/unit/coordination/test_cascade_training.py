"""Tests for cascade training orchestrator.

December 29, 2025
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.cascade_training import (
    CascadeConfig,
    CascadeStage,
    CascadeState,
    CascadeTrainingOrchestrator,
)


class TestCascadeStage:
    """Tests for CascadeStage enum."""

    def test_all_stages_exist(self):
        """Verify all expected stages are defined."""
        assert CascadeStage.NOT_STARTED.value == "not_started"
        assert CascadeStage.TRAINING_2P.value == "training_2p"
        assert CascadeStage.READY_2P.value == "ready_2p"
        assert CascadeStage.TRAINING_3P.value == "training_3p"
        assert CascadeStage.READY_3P.value == "ready_3p"
        assert CascadeStage.TRAINING_4P.value == "training_4p"
        assert CascadeStage.COMPLETE.value == "complete"

    def test_stage_count(self):
        """Ensure we have all 7 stages."""
        assert len(CascadeStage) == 7


class TestCascadeState:
    """Tests for CascadeState dataclass."""

    def test_default_state(self):
        """Test default state initialization."""
        state = CascadeState(board_type="hex8")
        assert state.board_type == "hex8"
        assert state.stage == CascadeStage.NOT_STARTED
        assert state.model_2p is None
        assert state.model_3p is None
        assert state.model_4p is None
        assert state.elo_2p == 0.0
        assert state.elo_3p == 0.0
        assert state.elo_4p == 0.0
        assert state.training_started is None
        assert state.min_elo_for_transfer == 1200.0
        assert state.min_games_for_transfer == 1000

    def test_can_transfer_to_3p_no_model(self):
        """Cannot transfer without a 2p model."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.READY_2P
        state.elo_2p = 1500.0
        assert not state.can_transfer_to_3p()

    def test_can_transfer_to_3p_low_elo(self):
        """Cannot transfer if Elo is too low."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.READY_2P
        state.model_2p = "models/canonical_hex8_2p.pth"
        state.elo_2p = 1100.0  # Below threshold
        assert not state.can_transfer_to_3p()

    def test_can_transfer_to_3p_wrong_stage(self):
        """Cannot transfer from wrong stage."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.NOT_STARTED
        state.model_2p = "models/canonical_hex8_2p.pth"
        state.elo_2p = 1500.0
        assert not state.can_transfer_to_3p()

    def test_can_transfer_to_3p_success(self):
        """Can transfer when all conditions met."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.READY_2P
        state.model_2p = "models/canonical_hex8_2p.pth"
        state.elo_2p = 1500.0
        assert state.can_transfer_to_3p()

    def test_can_transfer_to_3p_training_stage(self):
        """Can transfer from TRAINING_3P stage too."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.TRAINING_3P
        state.model_2p = "models/canonical_hex8_2p.pth"
        state.elo_2p = 1500.0
        assert state.can_transfer_to_3p()

    def test_can_transfer_to_4p_no_model(self):
        """Cannot transfer without a 3p model."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.READY_3P
        state.elo_3p = 1500.0
        assert not state.can_transfer_to_4p()

    def test_can_transfer_to_4p_low_elo(self):
        """Cannot transfer if 3p Elo is too low."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.READY_3P
        state.model_3p = "models/canonical_hex8_3p.pth"
        state.elo_3p = 1100.0  # Below threshold
        assert not state.can_transfer_to_4p()

    def test_can_transfer_to_4p_success(self):
        """Can transfer to 4p when conditions met."""
        state = CascadeState(board_type="hex8")
        state.stage = CascadeStage.READY_3P
        state.model_3p = "models/canonical_hex8_3p.pth"
        state.elo_3p = 1500.0
        assert state.can_transfer_to_4p()

    def test_custom_elo_threshold(self):
        """Custom Elo threshold is respected."""
        state = CascadeState(board_type="hex8", min_elo_for_transfer=1000.0)
        state.stage = CascadeStage.READY_2P
        state.model_2p = "models/canonical_hex8_2p.pth"
        state.elo_2p = 1050.0  # Above custom threshold
        assert state.can_transfer_to_3p()

    def test_to_dict_all_fields(self):
        """to_dict includes all expected fields."""
        state = CascadeState(board_type="square8")
        state.stage = CascadeStage.READY_2P
        state.model_2p = "models/test.pth"
        state.elo_2p = 1400.0

        d = state.to_dict()
        assert d["board_type"] == "square8"
        assert d["stage"] == "ready_2p"
        assert d["model_2p"] == "models/test.pth"
        assert d["model_3p"] is None
        assert d["model_4p"] is None
        assert d["elo_2p"] == 1400.0
        assert d["elo_3p"] == 0.0
        assert d["elo_4p"] == 0.0
        assert d["min_elo_for_transfer"] == 1200.0
        assert d["min_games_for_transfer"] == 1000
        assert d["training_started"] is None
        assert "last_updated" in d

    def test_to_dict_with_training_started(self):
        """to_dict includes training_started when set."""
        state = CascadeState(board_type="hex8")
        state.training_started = datetime(2025, 12, 29, 12, 0, 0)

        d = state.to_dict()
        assert d["training_started"] == "2025-12-29T12:00:00"


class TestCascadeConfig:
    """Tests for CascadeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CascadeConfig()
        assert config.min_elo_for_transfer == 1200.0
        assert config.min_games_for_training == 500
        assert config.bootstrap_selfplay_multiplier == 1.5
        assert config.board_types == ["hex8", "square8", "square19", "hexagonal"]
        assert config.models_dir == "models"
        assert config.data_dir == "data/games"
        assert config.check_interval == 300.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CascadeConfig(
            min_elo_for_transfer=1000.0,
            min_games_for_training=1000,
            bootstrap_selfplay_multiplier=2.0,
            board_types=["hex8"],
            check_interval=60.0,
        )
        assert config.min_elo_for_transfer == 1000.0
        assert config.min_games_for_training == 1000
        assert config.bootstrap_selfplay_multiplier == 2.0
        assert config.board_types == ["hex8"]
        assert config.check_interval == 60.0


class TestCascadeTrainingOrchestrator:
    """Tests for CascadeTrainingOrchestrator class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        CascadeTrainingOrchestrator.reset_instance()
        yield
        CascadeTrainingOrchestrator.reset_instance()

    def test_singleton_pattern(self):
        """Test singleton pattern works."""
        instance1 = CascadeTrainingOrchestrator.get_instance()
        instance2 = CascadeTrainingOrchestrator.get_instance()
        assert instance1 is instance2

    def test_singleton_reset(self):
        """Test singleton can be reset."""
        instance1 = CascadeTrainingOrchestrator.get_instance()
        CascadeTrainingOrchestrator.reset_instance()
        instance2 = CascadeTrainingOrchestrator.get_instance()
        assert instance1 is not instance2

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        orchestrator = CascadeTrainingOrchestrator()
        assert orchestrator.config is not None
        assert len(orchestrator._states) == 4  # All board types

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = CascadeConfig(board_types=["hex8", "square8"])
        orchestrator = CascadeTrainingOrchestrator(config)
        assert len(orchestrator._states) == 2
        assert "hex8" in orchestrator._states
        assert "square8" in orchestrator._states

    def test_states_initialized_correctly(self):
        """Test that states are initialized for each board type."""
        orchestrator = CascadeTrainingOrchestrator()
        for board_type in ["hex8", "square8", "square19", "hexagonal"]:
            assert board_type in orchestrator._states
            state = orchestrator._states[board_type]
            assert state.board_type == board_type
            assert state.stage == CascadeStage.NOT_STARTED

    def test_event_subscriptions(self):
        """Test event subscriptions are defined."""
        orchestrator = CascadeTrainingOrchestrator()
        subs = orchestrator._get_event_subscriptions()

        assert "TRAINING_COMPLETED" in subs
        assert "EVALUATION_COMPLETED" in subs
        assert "MODEL_PROMOTED" in subs
        assert "ELO_UPDATED" in subs
        assert "CASCADE_TRANSFER_TRIGGERED" in subs

    def test_handler_inheritance(self):
        """Test that orchestrator inherits from HandlerBase."""
        orchestrator = CascadeTrainingOrchestrator()
        assert hasattr(orchestrator, "health_check")
        assert hasattr(orchestrator, "start")
        assert hasattr(orchestrator, "stop")

    @pytest.mark.asyncio
    async def test_on_cascade_transfer_triggered_missing_fields(self):
        """Test transfer handler with missing fields."""
        orchestrator = CascadeTrainingOrchestrator()

        # Should log warning and return without error
        await orchestrator._on_cascade_transfer_triggered({
            "board_type": "hex8",
            # Missing other required fields
        })
        # No exception = success

    @pytest.mark.asyncio
    async def test_on_cascade_transfer_triggered_valid(self):
        """Test transfer handler with valid event."""
        orchestrator = CascadeTrainingOrchestrator()

        with patch.object(orchestrator, "_execute_transfer", new_callable=AsyncMock) as mock_transfer:
            mock_transfer.return_value = True
            await orchestrator._on_cascade_transfer_triggered({
                "board_type": "hex8",
                "source_players": 2,
                "target_players": 3,
                "source_model": "models/canonical_hex8_2p.pth",
            })
            mock_transfer.assert_called_once()

    def test_cycle_interval_from_config(self):
        """Test that cycle interval comes from config."""
        config = CascadeConfig(check_interval=120.0)
        orchestrator = CascadeTrainingOrchestrator(config)
        assert orchestrator._cycle_interval == 120.0

    def test_health_check_returns_result(self):
        """Test health_check returns proper result."""
        orchestrator = CascadeTrainingOrchestrator()
        result = orchestrator.health_check()

        # HandlerBase provides health_check
        assert hasattr(result, "healthy")


class TestCascadeTransferLogic:
    """Tests for transfer execution logic."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        CascadeTrainingOrchestrator.reset_instance()
        yield
        CascadeTrainingOrchestrator.reset_instance()

    @pytest.mark.asyncio
    async def test_execute_transfer_success(self):
        """Test successful transfer execution."""
        orchestrator = CascadeTrainingOrchestrator()

        with patch.object(orchestrator, "_run_transfer_sync") as mock_sync, \
             patch.object(orchestrator, "_emit_event", new_callable=AsyncMock) as mock_emit, \
             patch("pathlib.Path.exists", return_value=True):

            result = await orchestrator._execute_transfer(
                board_type="hex8",
                source_model="models/canonical_hex8_2p.pth",
                source_players=2,
                target_players=3,
            )

            assert result is True
            mock_sync.assert_called_once()
            mock_emit.assert_called_once()
            # Check emit was called with TRAINING_REQUESTED
            call_args = mock_emit.call_args[0]
            assert call_args[0] == "TRAINING_REQUESTED"

    @pytest.mark.asyncio
    async def test_execute_transfer_output_not_found(self):
        """Test transfer fails when output not created."""
        orchestrator = CascadeTrainingOrchestrator()

        with patch.object(orchestrator, "_run_transfer_sync"), \
             patch("pathlib.Path.exists", return_value=False):

            result = await orchestrator._execute_transfer(
                board_type="hex8",
                source_model="models/canonical_hex8_2p.pth",
                source_players=2,
                target_players=3,
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_execute_transfer_handles_exception(self):
        """Test transfer handles exceptions gracefully."""
        orchestrator = CascadeTrainingOrchestrator()

        with patch.object(orchestrator, "_run_transfer_sync", side_effect=RuntimeError("Test error")), \
             patch.object(orchestrator, "_emit_event", new_callable=AsyncMock) as mock_emit:

            result = await orchestrator._execute_transfer(
                board_type="hex8",
                source_model="models/canonical_hex8_2p.pth",
                source_players=2,
                target_players=3,
            )

            assert result is False
            # Should emit CASCADE_TRANSFER_FAILED
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args[0]
            assert call_args[0] == "CASCADE_TRANSFER_FAILED"


class TestCascadeStateTransitions:
    """Tests for state transition logic."""

    def test_state_progression(self):
        """Test expected state progression."""
        state = CascadeState(board_type="hex8")

        # Start
        assert state.stage == CascadeStage.NOT_STARTED

        # Training 2p
        state.stage = CascadeStage.TRAINING_2P
        assert not state.can_transfer_to_3p()

        # Ready 2p with model and Elo
        state.stage = CascadeStage.READY_2P
        state.model_2p = "models/canonical_hex8_2p.pth"
        state.elo_2p = 1500.0
        assert state.can_transfer_to_3p()
        assert not state.can_transfer_to_4p()

        # Training 3p
        state.stage = CascadeStage.TRAINING_3P
        # Still can transfer from 2p in this stage
        assert state.can_transfer_to_3p()

        # Ready 3p
        state.stage = CascadeStage.READY_3P
        state.model_3p = "models/canonical_hex8_3p.pth"
        state.elo_3p = 1400.0
        assert state.can_transfer_to_4p()

        # Training 4p
        state.stage = CascadeStage.TRAINING_4P
        assert state.can_transfer_to_4p()

        # Complete
        state.stage = CascadeStage.COMPLETE
        state.model_4p = "models/canonical_hex8_4p.pth"
        # Neither transfer valid from complete stage
        assert not state.can_transfer_to_3p()
        assert not state.can_transfer_to_4p()
