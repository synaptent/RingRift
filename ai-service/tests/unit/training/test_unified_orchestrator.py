"""Tests for UnifiedTrainingOrchestrator (training execution).

Tests cover:
- OrchestratorConfig dataclass
- Component wrappers (HotBufferWrapper, EnhancementsWrapper, etc.)
- UnifiedTrainingOrchestrator initialization
- Factory functions
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Test OrchestratorConfig Dataclass
# =============================================================================

class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        # Basic settings
        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.epochs == 50
        assert config.batch_size == 256
        assert config.learning_rate == 0.001

    def test_distributed_defaults(self):
        """Test distributed training defaults."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.enable_distributed is False
        assert config.world_size == 1
        assert config.compress_gradients is False
        assert config.use_amp is True

    def test_hot_buffer_defaults(self):
        """Test hot buffer defaults."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.enable_hot_buffer is True
        assert config.hot_buffer_size == 10000
        assert config.hot_buffer_priority_alpha == 0.6

    def test_enhancement_defaults(self):
        """Test enhancement defaults."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.enable_enhancements is True
        assert config.enable_auxiliary_tasks is False
        assert config.enable_gradient_surgery is False
        assert config.enable_elo_weighting is True
        assert config.enable_curriculum is True
        assert config.enable_augmentation is True

    def test_background_eval_defaults(self):
        """Test background eval defaults."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.enable_background_eval is True
        assert config.eval_interval_steps == 1000
        assert config.eval_games_per_check == 20

    def test_checkpoint_defaults(self):
        """Test checkpoint defaults."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.checkpoint_dir == "data/checkpoints"
        assert config.checkpoint_interval == 1000
        assert config.keep_top_k_checkpoints == 3
        assert config.auto_resume is True

    def test_adaptive_defaults(self):
        """Test adaptive settings defaults."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.enable_adaptive_lr is True
        assert config.enable_adaptive_batch is False

    def test_custom_values(self):
        """Test custom configuration."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig(
            board_type="hex6",
            num_players=3,
            epochs=100,
            batch_size=512,
            enable_distributed=True,
            world_size=4,
        )

        assert config.board_type == "hex6"
        assert config.num_players == 3
        assert config.epochs == 100
        assert config.batch_size == 512
        assert config.enable_distributed is True
        assert config.world_size == 4


# =============================================================================
# Test Component Wrappers
# =============================================================================

class TestHotBufferWrapper:
    """Tests for HotBufferWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        from app.training.unified_orchestrator import (
            HotBufferWrapper,
            OrchestratorConfig,
        )

        config = OrchestratorConfig()
        wrapper = HotBufferWrapper(config)

        assert wrapper.config == config
        assert wrapper._buffer is None
        assert wrapper.available is False

    def test_available_when_not_initialized(self):
        """Test available property when not initialized."""
        from app.training.unified_orchestrator import (
            HotBufferWrapper,
            OrchestratorConfig,
        )

        wrapper = HotBufferWrapper(OrchestratorConfig())
        assert wrapper.available is False

    def test_add_game_when_not_available(self):
        """Test add_game does nothing when buffer unavailable."""
        from app.training.unified_orchestrator import (
            HotBufferWrapper,
            OrchestratorConfig,
        )

        wrapper = HotBufferWrapper(OrchestratorConfig())
        # Should not raise
        wrapper.add_game({"test": "game"})

    def test_get_batch_returns_none_when_unavailable(self):
        """Test get_batch returns None when unavailable."""
        from app.training.unified_orchestrator import (
            HotBufferWrapper,
            OrchestratorConfig,
        )

        wrapper = HotBufferWrapper(OrchestratorConfig())
        result = wrapper.get_batch(32)
        assert result is None


class TestEnhancementsWrapper:
    """Tests for EnhancementsWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        from app.training.unified_orchestrator import (
            EnhancementsWrapper,
            OrchestratorConfig,
        )

        config = OrchestratorConfig()
        wrapper = EnhancementsWrapper(config)

        assert wrapper.config == config
        assert wrapper._manager is None
        assert wrapper.available is False

    def test_get_batch_size_default(self):
        """Test get_batch_size returns config default when unavailable."""
        from app.training.unified_orchestrator import (
            EnhancementsWrapper,
            OrchestratorConfig,
        )

        config = OrchestratorConfig(batch_size=128)
        wrapper = EnhancementsWrapper(config)

        assert wrapper.get_batch_size() == 128

    def test_compute_sample_weights_default(self):
        """Test compute_sample_weights returns ones when unavailable."""
        from app.training.unified_orchestrator import (
            EnhancementsWrapper,
            OrchestratorConfig,
        )
        import numpy as np

        wrapper = EnhancementsWrapper(OrchestratorConfig())
        elos = np.array([1500, 1600, 1700])
        weights = wrapper.compute_sample_weights(elos)

        assert np.allclose(weights, np.ones(3))

    def test_get_curriculum_params_default(self):
        """Test get_curriculum_params returns empty when unavailable."""
        from app.training.unified_orchestrator import (
            EnhancementsWrapper,
            OrchestratorConfig,
        )

        wrapper = EnhancementsWrapper(OrchestratorConfig())
        params = wrapper.get_curriculum_params()

        assert params == {}


class TestDistributedWrapper:
    """Tests for DistributedWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        from app.training.unified_orchestrator import (
            DistributedWrapper,
            OrchestratorConfig,
        )

        config = OrchestratorConfig()
        wrapper = DistributedWrapper(config)

        assert wrapper.config == config
        assert wrapper._trainer is None
        assert wrapper.available is False

    def test_is_main_process_default(self):
        """Test is_main_process returns True when unavailable."""
        from app.training.unified_orchestrator import (
            DistributedWrapper,
            OrchestratorConfig,
        )

        wrapper = DistributedWrapper(OrchestratorConfig())
        assert wrapper.is_main_process is True

    def test_wrap_model_passthrough(self):
        """Test wrap_model returns model unchanged when unavailable."""
        from app.training.unified_orchestrator import (
            DistributedWrapper,
            OrchestratorConfig,
        )

        wrapper = DistributedWrapper(OrchestratorConfig())
        mock_model = MagicMock()
        result = wrapper.wrap_model(mock_model)

        assert result is mock_model


class TestBackgroundEvalWrapper:
    """Tests for BackgroundEvalWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        from app.training.unified_orchestrator import (
            BackgroundEvalWrapper,
            OrchestratorConfig,
        )

        config = OrchestratorConfig()
        wrapper = BackgroundEvalWrapper(config)

        assert wrapper.config == config
        assert wrapper._evaluator is None
        assert wrapper.available is False

    def test_get_current_elo_default(self):
        """Test get_current_elo returns initial rating when unavailable."""
        from app.training.unified_orchestrator import (
            BackgroundEvalWrapper,
            OrchestratorConfig,
        )

        wrapper = BackgroundEvalWrapper(OrchestratorConfig())
        elo = wrapper.get_current_elo()

        # Should return INITIAL_ELO_RATING from thresholds
        assert isinstance(elo, (int, float))
        assert elo > 0

    def test_should_early_stop_default(self):
        """Test should_early_stop returns False when unavailable."""
        from app.training.unified_orchestrator import (
            BackgroundEvalWrapper,
            OrchestratorConfig,
        )

        wrapper = BackgroundEvalWrapper(OrchestratorConfig())
        assert wrapper.should_early_stop() is False


class TestCheckpointWrapper:
    """Tests for CheckpointWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        from app.training.unified_orchestrator import (
            CheckpointWrapper,
            OrchestratorConfig,
        )

        config = OrchestratorConfig()
        wrapper = CheckpointWrapper(config)

        assert wrapper.config == config
        assert wrapper._manager is None
        assert wrapper.available is False

    def test_should_save_returns_false_when_unavailable(self):
        """Test should_save returns False when unavailable."""
        from app.training.unified_orchestrator import (
            CheckpointWrapper,
            OrchestratorConfig,
        )

        wrapper = CheckpointWrapper(OrchestratorConfig())
        assert wrapper.should_save(epoch=1, loss=0.5, step=100) is False

    def test_load_latest_returns_none_when_unavailable(self):
        """Test load_latest returns None when unavailable."""
        from app.training.unified_orchestrator import (
            CheckpointWrapper,
            OrchestratorConfig,
        )

        wrapper = CheckpointWrapper(OrchestratorConfig())
        assert wrapper.load_latest() is None

    def test_get_stats_returns_empty_when_unavailable(self):
        """Test get_stats returns empty dict when unavailable."""
        from app.training.unified_orchestrator import (
            CheckpointWrapper,
            OrchestratorConfig,
        )

        wrapper = CheckpointWrapper(OrchestratorConfig())
        assert wrapper.get_stats() == {}


# =============================================================================
# Test UnifiedTrainingOrchestrator
# =============================================================================

class TestUnifiedTrainingOrchestrator:
    """Tests for UnifiedTrainingOrchestrator class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PyTorch model."""
        model = MagicMock()
        model.parameters.return_value = iter([MagicMock()])
        model.state_dict.return_value = {}
        return model

    def test_initialization_with_defaults(self, mock_model):
        """Test orchestrator initialization with defaults."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)

        assert orch._model is mock_model
        assert orch._step == 0
        assert orch._epoch == 0
        assert orch._initialized is False
        assert orch.config.board_type == "square8"

    def test_initialization_with_config(self, mock_model):
        """Test orchestrator initialization with custom config."""
        from app.training.unified_orchestrator import (
            UnifiedTrainingOrchestrator,
            OrchestratorConfig,
        )

        config = OrchestratorConfig(board_type="hex6", num_players=3)
        orch = UnifiedTrainingOrchestrator(mock_model, config)

        assert orch.config.board_type == "hex6"
        assert orch.config.num_players == 3

    def test_model_property(self, mock_model):
        """Test model property."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        assert orch.model is mock_model

    def test_step_property(self, mock_model):
        """Test step property."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        assert orch.step == 0

    def test_epoch_property(self, mock_model):
        """Test epoch property."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        assert orch.epoch == 0

    def test_set_epoch(self, mock_model):
        """Test set_epoch method."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        orch.set_epoch(5)
        assert orch.epoch == 5

    def test_should_stop_default(self, mock_model):
        """Test should_stop returns False by default."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        assert orch.should_stop() is False

    def test_get_health_status_before_init(self, mock_model):
        """Test get_health_status before initialization."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        status = orch.get_health_status()

        assert status["initialized"] is False
        assert status["init_time_ms"] == 0

    def test_get_training_quality_insufficient_data(self, mock_model):
        """Test get_training_quality with insufficient data."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        quality = orch.get_training_quality()

        assert quality["loss_plateau"] is False
        assert quality["overfit_detected"] is False
        assert quality["last_loss"] is None

    def test_get_training_quality_with_history(self, mock_model):
        """Test get_training_quality with loss history."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)

        # Add loss history (decreasing = improving)
        orch._loss_history = [1.0 - i * 0.05 for i in range(20)]

        quality = orch.get_training_quality()

        assert quality["last_loss"] == pytest.approx(0.05, abs=0.01)

    def test_get_metrics(self, mock_model):
        """Test get_metrics method."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(mock_model)
        orch._step = 100
        orch._epoch = 5

        metrics = orch.get_metrics()

        assert metrics["step"] == 100
        assert metrics["epoch"] == 5
        assert "elo" in metrics


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_orchestrator(self):
        """Test create_orchestrator factory function."""
        from app.training.unified_orchestrator import create_orchestrator

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])

        orch = create_orchestrator(
            mock_model,
            board_type="hex6",
            num_players=4,
            epochs=25,
        )

        assert orch.config.board_type == "hex6"
        assert orch.config.num_players == 4
        assert orch.config.epochs == 25


# =============================================================================
# Integration Tests
# =============================================================================

class TestOrchestratorIntegration:
    """Integration tests for unified orchestrator."""

    def test_config_all_attributes(self):
        """Test OrchestratorConfig has all expected attributes."""
        from app.training.unified_orchestrator import OrchestratorConfig

        config = OrchestratorConfig()

        # Check key attributes exist
        assert hasattr(config, "board_type")
        assert hasattr(config, "num_players")
        assert hasattr(config, "epochs")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "enable_distributed")
        assert hasattr(config, "enable_hot_buffer")
        assert hasattr(config, "enable_enhancements")
        assert hasattr(config, "enable_background_eval")
        assert hasattr(config, "checkpoint_dir")
        assert hasattr(config, "enable_adaptive_lr")

    def test_orchestrator_has_expected_methods(self):
        """Test UnifiedTrainingOrchestrator has expected methods."""
        from app.training.unified_orchestrator import UnifiedTrainingOrchestrator

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        orch = UnifiedTrainingOrchestrator(mock_model)

        assert hasattr(orch, "initialize")
        assert hasattr(orch, "train_step")
        assert hasattr(orch, "cleanup")
        assert hasattr(orch, "get_metrics")
        assert hasattr(orch, "get_health_status")
        assert hasattr(orch, "start_background_services")
        assert hasattr(orch, "stop_background_services")
