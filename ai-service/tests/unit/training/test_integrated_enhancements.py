"""Unit tests for integrated_enhancements module.

December 2025: Tests for the deprecated IntegratedTrainingManager class
and related functionality. This module is deprecated in favor of
UnifiedTrainingOrchestrator, but tests ensure backward compatibility
during the deprecation period.
"""

import warnings
from dataclasses import fields
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestIntegratedEnhancementsConfig:
    """Tests for IntegratedEnhancementsConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that all config fields have sensible defaults."""
        # Import inside test to capture deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
            )

        config = IntegratedEnhancementsConfig()

        # Auxiliary tasks
        assert config.auxiliary_tasks_enabled is True
        assert config.aux_game_length_weight == 0.1
        assert config.aux_piece_count_weight == 0.1
        assert config.aux_outcome_weight == 0.05

        # Gradient surgery
        assert config.gradient_surgery_enabled is True
        assert config.gradient_surgery_method == "pcgrad"

        # Batch scheduling
        assert config.batch_scheduling_enabled is True
        assert config.batch_initial_size == 64
        assert config.batch_final_size == 512
        assert config.batch_warmup_steps == 1000

        # Background eval
        assert config.background_eval_enabled is True
        assert config.eval_interval_steps == 1000

        # ELO weighting
        assert config.elo_weighting_enabled is True
        assert config.elo_base_rating == 1500.0

        # Curriculum
        assert config.curriculum_enabled is True
        assert config.curriculum_auto_advance is True

        # Augmentation
        assert config.augmentation_enabled is True
        assert config.augmentation_mode == "all"

        # Reanalysis
        assert config.reanalysis_enabled is True
        assert config.reanalysis_blend_ratio == 0.7

        # Distillation
        assert config.distillation_enabled is True
        assert config.distillation_temperature == 3.0

    def test_custom_values(self) -> None:
        """Test that custom values override defaults."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
            )

        config = IntegratedEnhancementsConfig(
            auxiliary_tasks_enabled=False,
            batch_initial_size=32,
            elo_base_rating=1800.0,
            reanalysis_blend_ratio=0.5,
        )

        assert config.auxiliary_tasks_enabled is False
        assert config.batch_initial_size == 32
        assert config.elo_base_rating == 1800.0
        assert config.reanalysis_blend_ratio == 0.5

    def test_is_dataclass(self) -> None:
        """Test that config is a proper dataclass."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
            )

        config_fields = fields(IntegratedEnhancementsConfig)
        assert len(config_fields) > 30  # Has many configuration options


class TestIntegratedTrainingManagerInit:
    """Tests for IntegratedTrainingManager initialization."""

    def test_default_initialization(self) -> None:
        """Test manager initializes with defaults."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        manager = IntegratedTrainingManager()

        assert manager.config is not None
        assert isinstance(manager.config, IntegratedEnhancementsConfig)
        assert manager.model is None
        assert manager.board_type == "square8"
        assert manager._step == 0

    def test_custom_config(self) -> None:
        """Test manager accepts custom config."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        config = IntegratedEnhancementsConfig(
            auxiliary_tasks_enabled=False,
            gradient_surgery_enabled=False,
        )
        manager = IntegratedTrainingManager(config=config, board_type="hex8")

        assert manager.config.auxiliary_tasks_enabled is False
        assert manager.config.gradient_surgery_enabled is False
        assert manager.board_type == "hex8"

    def test_with_model(self) -> None:
        """Test manager accepts a model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedTrainingManager,
            )

        mock_model = MagicMock()
        manager = IntegratedTrainingManager(model=mock_model)

        assert manager.model is mock_model


class TestIntegratedTrainingManagerMethods:
    """Tests for IntegratedTrainingManager methods."""

    @pytest.fixture
    def manager(self):
        """Create a manager instance for testing."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        config = IntegratedEnhancementsConfig(
            # Disable components that require external dependencies
            background_eval_enabled=False,
            curriculum_enabled=False,
            reanalysis_enabled=False,
            distillation_enabled=False,
        )
        return IntegratedTrainingManager(config=config)

    def test_update_step(self, manager) -> None:
        """Test step counter updates."""
        assert manager._step == 0

        manager.update_step(100)
        assert manager._step == 100

        manager.update_step(200)
        assert manager._step == 200

    def test_get_batch_size_without_scheduler(self, manager) -> None:
        """Test batch size returns default when scheduler not initialized."""
        batch_size = manager.get_batch_size()
        assert batch_size == manager.config.batch_initial_size

    def test_get_batch_size_with_scheduler(self, manager) -> None:
        """Test batch size from scheduler."""
        manager.initialize_all()
        manager.update_step(0)

        batch_size = manager.get_batch_size()
        assert batch_size == manager.config.batch_initial_size

    def test_get_current_elo_without_evaluator(self, manager) -> None:
        """Test Elo returns None when evaluator not initialized."""
        elo = manager.get_current_elo()
        assert elo is None

    def test_should_early_stop_without_evaluator(self, manager) -> None:
        """Test early stop returns False when evaluator not initialized."""
        should_stop = manager.should_early_stop()
        assert should_stop is False

    def test_get_baseline_gating_status_without_evaluator(self, manager) -> None:
        """Test baseline gating returns disabled status."""
        passed, failures, count = manager.get_baseline_gating_status()
        assert passed is True
        assert failures == []
        assert count == 0

    def test_should_warn_baseline_failures(self, manager) -> None:
        """Test baseline failure warning."""
        should_warn = manager.should_warn_baseline_failures(threshold=3)
        assert should_warn is False

    def test_get_curriculum_parameters_without_controller(self, manager) -> None:
        """Test curriculum params return defaults when not initialized."""
        params = manager.get_curriculum_parameters()
        assert params == {}

    def test_should_reanalyze_disabled(self, manager) -> None:
        """Test reanalysis check returns False when disabled."""
        should = manager.should_reanalyze()
        assert should is False

    def test_should_distill_disabled(self, manager) -> None:
        """Test distillation check returns False when disabled."""
        should = manager.should_distill(current_epoch=20)
        assert should is False

    def test_get_metrics(self, manager) -> None:
        """Test metrics retrieval."""
        metrics = manager.get_metrics()
        assert isinstance(metrics, dict)
        assert "step" in metrics
        assert metrics["step"] == 0

    def test_count_enabled(self, manager) -> None:
        """Test counting enabled modules."""
        count_before = manager._count_enabled()
        assert count_before == 0

        manager.initialize_all()
        count_after = manager._count_enabled()
        assert count_after >= 1


class TestAugmentBatch:
    """Tests for batch augmentation methods."""

    @pytest.fixture
    def manager_with_augmentation(self):
        """Create manager with augmentation enabled."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        config = IntegratedEnhancementsConfig(
            augmentation_enabled=True,
            augmentation_mode="all",
            background_eval_enabled=False,
            curriculum_enabled=False,
            reanalysis_enabled=False,
            distillation_enabled=False,
        )
        manager = IntegratedTrainingManager(config=config, board_type="square8")
        manager.initialize_all()
        return manager

    def test_augment_batch_dense_returns_dict(self, manager_with_augmentation) -> None:
        """Test augment_batch_dense returns a dictionary."""
        batch = {
            "features": np.random.randn(4, 16, 8, 8).astype(np.float32),
            "policy": np.random.randn(4, 64).astype(np.float32),
            "value": np.random.randn(4).astype(np.float32),
        }

        result = manager_with_augmentation.augment_batch_dense(batch)
        assert isinstance(result, dict)
        assert "features" in result
        assert "policy" in result
        assert "value" in result


class TestComputeSampleWeights:
    """Tests for ELO-weighted sample computation."""

    @pytest.fixture
    def manager_with_elo(self):
        """Create manager with ELO weighting enabled."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        config = IntegratedEnhancementsConfig(
            elo_weighting_enabled=True,
            elo_base_rating=1500.0,
            elo_weight_scale=400.0,
            elo_min_weight=0.5,
            elo_max_weight=2.0,
            background_eval_enabled=False,
            curriculum_enabled=False,
            reanalysis_enabled=False,
            distillation_enabled=False,
        )
        manager = IntegratedTrainingManager(config=config)
        manager.initialize_all()
        return manager

    def test_compute_sample_weights_uniform_elo(self, manager_with_elo) -> None:
        """Test sample weights with uniform Elo ratings."""
        opponent_elos = np.array([1500.0, 1500.0, 1500.0, 1500.0])
        weights = manager_with_elo.compute_sample_weights(opponent_elos)

        assert len(weights) == 4
        assert np.allclose(weights, weights[0])

    def test_compute_sample_weights_varied_elo(self, manager_with_elo) -> None:
        """Test sample weights with varied Elo ratings."""
        opponent_elos = np.array([1200.0, 1500.0, 1800.0])
        weights = manager_with_elo.compute_sample_weights(opponent_elos)

        assert len(weights) == 3
        assert weights[2] > weights[1]
        assert weights[1] > weights[0]


class TestDeprecationWarning:
    """Tests for deprecation warnings."""

    def test_import_emits_deprecation_warning(self) -> None:
        """Test that importing the module emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            import importlib
            import app.training.integrated_enhancements

            importlib.reload(app.training.integrated_enhancements)

            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "IntegratedTrainingManager" in str(deprecation_warnings[0].message)


class TestGetEnhancementDefaults:
    """Tests for get_enhancement_defaults function."""

    def test_returns_dict(self) -> None:
        """Test that defaults are returned as dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                get_enhancement_defaults,
            )

        defaults = get_enhancement_defaults()
        assert isinstance(defaults, dict)

    def test_contains_all_categories(self) -> None:
        """Test that defaults contain all enhancement categories."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                get_enhancement_defaults,
            )

        defaults = get_enhancement_defaults()

        assert "auxiliary_tasks" in defaults
        assert "gradient_surgery" in defaults
        assert "batch_scheduling" in defaults
        assert "background_eval" in defaults
        assert "elo_weighting" in defaults
        assert "curriculum" in defaults
        assert "augmentation" in defaults
        assert "reanalysis" in defaults
        assert "distillation" in defaults

    def test_defaults_have_enabled_flag(self) -> None:
        """Test that each category has enabled flag."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                get_enhancement_defaults,
            )

        defaults = get_enhancement_defaults()

        for category, values in defaults.items():
            assert "enabled" in values, f"Category {category} missing 'enabled' flag"
            assert isinstance(values["enabled"], bool)


class TestReanalysisStats:
    """Tests for reanalysis statistics."""

    def test_get_reanalysis_stats_disabled(self) -> None:
        """Test reanalysis stats when disabled."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        config = IntegratedEnhancementsConfig(reanalysis_enabled=False)
        manager = IntegratedTrainingManager(config=config)

        stats = manager.get_reanalysis_stats()

        assert isinstance(stats, dict)
        assert stats.get("enabled") is False
        assert stats.get("positions_reanalyzed", 0) == 0


class TestDistillationStats:
    """Tests for distillation statistics."""

    def test_get_distillation_stats_disabled(self) -> None:
        """Test distillation stats when disabled."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )

        config = IntegratedEnhancementsConfig(distillation_enabled=False)
        manager = IntegratedTrainingManager(config=config)

        stats = manager.get_distillation_stats()

        assert isinstance(stats, dict)
        assert stats.get("enabled") is False


class TestThreadSafety:
    """Tests for thread safety of the manager."""

    def test_metrics_access_is_thread_safe(self) -> None:
        """Test that metrics access uses lock."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from app.training.integrated_enhancements import (
                IntegratedTrainingManager,
            )

        manager = IntegratedTrainingManager()

        assert hasattr(manager, "_lock")
        assert manager._lock is not None

        metrics = manager.get_metrics()
        assert isinstance(metrics, dict)
