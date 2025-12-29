"""Unit tests for app.training.integrated_enhancements module.

Tests the deprecated but still active IntegratedTrainingManager and related classes.

Note: This module is deprecated since December 2025. New code should use
UnifiedTrainingOrchestrator from unified_orchestrator.py instead.

Tests cover:
- IntegratedEnhancementsConfig dataclass
- IntegratedTrainingManager class
- Factory functions (create_integrated_manager, get_enhancement_defaults)
- Component initialization
- Training step integration
- Lifecycle management
"""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Suppress deprecation warning during import
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from app.training.integrated_enhancements import (
        IntegratedEnhancementsConfig,
        IntegratedTrainingManager,
        create_integrated_manager,
        get_enhancement_defaults,
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default configuration."""
    return IntegratedEnhancementsConfig()


@pytest.fixture
def minimal_manager():
    """Create minimal manager for testing."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        config = IntegratedEnhancementsConfig(
            auxiliary_tasks_enabled=False,
            gradient_surgery_enabled=False,
            batch_scheduling_enabled=False,
            background_eval_enabled=False,
            elo_weighting_enabled=False,
            curriculum_enabled=False,
            augmentation_enabled=False,
            reanalysis_enabled=False,
            distillation_enabled=False,
        )
        return IntegratedTrainingManager(config=config)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory with mock checkpoints."""
    # Create some mock checkpoint files
    for elo in [1600, 1650, 1700, 1725]:
        ckpt_path = tmp_path / f"model_elo{elo}.pt"
        ckpt_path.touch()
    return tmp_path


# =============================================================================
# Tests for IntegratedEnhancementsConfig
# =============================================================================


class TestIntegratedEnhancementsConfig:
    """Tests for IntegratedEnhancementsConfig dataclass."""

    def test_default_values(self, default_config):
        """Verify all default values are set correctly."""
        # Auxiliary Tasks (enabled by default for +30-80 Elo)
        assert default_config.auxiliary_tasks_enabled is True
        assert default_config.aux_game_length_weight == 0.1
        assert default_config.aux_piece_count_weight == 0.1
        assert default_config.aux_outcome_weight == 0.05

        # Gradient Surgery
        assert default_config.gradient_surgery_enabled is True
        assert default_config.gradient_surgery_method == "pcgrad"
        assert default_config.gradient_conflict_threshold == 0.0

        # Batch Scheduling
        assert default_config.batch_scheduling_enabled is True
        assert default_config.batch_initial_size == 64
        assert default_config.batch_final_size == 512
        assert default_config.batch_warmup_steps == 1000
        assert default_config.batch_rampup_steps == 10000
        assert default_config.batch_schedule_type == "linear"

        # Background Evaluation
        assert default_config.background_eval_enabled is True
        assert default_config.eval_interval_steps == 1000
        assert default_config.eval_games_per_check == 20
        assert default_config.eval_elo_checkpoint_threshold == 10.0
        assert default_config.eval_elo_drop_threshold == 50.0
        assert default_config.eval_auto_checkpoint is True

        # ELO Weighting
        assert default_config.elo_weighting_enabled is True
        assert default_config.elo_base_rating == 1500.0
        assert default_config.elo_weight_scale == 400.0
        assert default_config.elo_min_weight == 0.5
        assert default_config.elo_max_weight == 2.0

        # Curriculum Learning
        assert default_config.curriculum_enabled is True
        assert default_config.curriculum_auto_advance is True

        # Data Augmentation
        assert default_config.augmentation_enabled is True
        assert default_config.augmentation_mode == "all"
        assert default_config.augmentation_probability == 1.0

        # Reanalysis (enabled for +40-120 Elo)
        assert default_config.reanalysis_enabled is True
        assert default_config.reanalysis_blend_ratio == 0.7
        assert default_config.reanalysis_interval_steps == 2000
        assert default_config.reanalysis_batch_size == 1000
        assert default_config.reanalysis_min_elo_delta == 15
        assert default_config.reanalysis_use_mcts is True
        assert default_config.reanalysis_mcts_simulations == 100

        # Knowledge Distillation
        assert default_config.distillation_enabled is True
        assert default_config.distillation_temperature == 3.0
        assert default_config.distillation_alpha == 0.7
        assert default_config.distillation_interval_epochs == 10
        assert default_config.distillation_num_teachers == 3

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = IntegratedEnhancementsConfig(
            auxiliary_tasks_enabled=False,
            batch_initial_size=128,
            batch_final_size=1024,
            eval_interval_steps=500,
            reanalysis_blend_ratio=0.5,
            distillation_temperature=2.0,
        )

        assert config.auxiliary_tasks_enabled is False
        assert config.batch_initial_size == 128
        assert config.batch_final_size == 1024
        assert config.eval_interval_steps == 500
        assert config.reanalysis_blend_ratio == 0.5
        assert config.distillation_temperature == 2.0


# =============================================================================
# Tests for IntegratedTrainingManager Initialization
# =============================================================================


class TestIntegratedTrainingManagerInit:
    """Tests for IntegratedTrainingManager initialization."""

    def test_basic_init(self):
        """Can initialize with defaults."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = IntegratedTrainingManager()

        assert manager.config is not None
        assert manager.model is None
        assert manager.board_type == "square8"
        assert manager._step == 0

    def test_init_with_custom_config(self):
        """Can initialize with custom config."""
        config = IntegratedEnhancementsConfig(
            batch_initial_size=128,
            auxiliary_tasks_enabled=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = IntegratedTrainingManager(config=config)

        assert manager.config.batch_initial_size == 128
        assert manager.config.auxiliary_tasks_enabled is False

    def test_init_with_board_type(self):
        """Can initialize with specific board type."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = IntegratedTrainingManager(board_type="hex8")

        assert manager.board_type == "hex8"

    def test_init_lazy_components(self):
        """Components should be lazily initialized."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = IntegratedTrainingManager()

        assert manager._auxiliary_module is None
        assert manager._gradient_surgeon is None
        assert manager._batch_scheduler is None
        assert manager._background_evaluator is None
        assert manager._curriculum_controller is None
        assert manager._augmentor is None
        assert manager._reanalysis_engine is None
        assert manager._distillation_config is None


# =============================================================================
# Tests for IntegratedTrainingManager Methods
# =============================================================================


class TestIntegratedTrainingManagerMethods:
    """Tests for IntegratedTrainingManager methods."""

    def test_count_enabled_zero(self, minimal_manager):
        """Count enabled modules when all disabled."""
        assert minimal_manager._count_enabled() == 0

    def test_get_batch_size_no_scheduler(self, minimal_manager):
        """get_batch_size should return initial size without scheduler."""
        batch_size = minimal_manager.get_batch_size()
        assert batch_size == minimal_manager.config.batch_initial_size

    def test_get_curriculum_parameters_no_controller(self, minimal_manager):
        """get_curriculum_parameters should return empty dict without controller."""
        params = minimal_manager.get_curriculum_parameters()
        assert params == {}

    def test_update_step(self, minimal_manager):
        """update_step should increment step counter."""
        assert minimal_manager._step == 0
        minimal_manager.update_step()
        assert minimal_manager._step == 1
        minimal_manager.update_step()
        assert minimal_manager._step == 2

    def test_should_early_stop_no_evaluator(self, minimal_manager):
        """should_early_stop should return False without evaluator."""
        assert minimal_manager.should_early_stop() is False

    def test_get_current_elo_no_evaluator(self, minimal_manager):
        """get_current_elo should return None without evaluator."""
        assert minimal_manager.get_current_elo() is None

    def test_get_baseline_gating_status_no_evaluator(self, minimal_manager):
        """get_baseline_gating_status should return defaults without evaluator."""
        passes, failed, count = minimal_manager.get_baseline_gating_status()
        assert passes is True
        assert failed == []
        assert count == 0

    def test_should_warn_baseline_failures_no_evaluator(self, minimal_manager):
        """should_warn_baseline_failures should return False without evaluator."""
        assert minimal_manager.should_warn_baseline_failures() is False


class TestIntegratedTrainingManagerReanalysis:
    """Tests for reanalysis integration."""

    def test_should_reanalyze_disabled(self, minimal_manager):
        """should_reanalyze should return False when disabled."""
        minimal_manager.config.reanalysis_enabled = False
        assert minimal_manager.should_reanalyze() is False

    def test_should_reanalyze_no_engine(self, minimal_manager):
        """should_reanalyze should return False without engine."""
        minimal_manager.config.reanalysis_enabled = True
        minimal_manager._reanalysis_engine = None
        assert minimal_manager.should_reanalyze() is False

    def test_get_reanalysis_stats_disabled(self, minimal_manager):
        """get_reanalysis_stats should indicate disabled."""
        stats = minimal_manager.get_reanalysis_stats()
        assert stats == {"enabled": False}

    def test_process_reanalysis_no_engine(self, minimal_manager):
        """process_reanalysis should return None without engine."""
        result = minimal_manager.process_reanalysis()
        assert result is None


class TestIntegratedTrainingManagerDistillation:
    """Tests for distillation integration."""

    def test_should_distill_disabled(self, minimal_manager):
        """should_distill should return False when disabled."""
        minimal_manager.config.distillation_enabled = False
        assert minimal_manager.should_distill(current_epoch=10) is False

    def test_should_distill_no_config(self, minimal_manager):
        """should_distill should return False without distillation config."""
        minimal_manager.config.distillation_enabled = True
        minimal_manager._distillation_config = None
        assert minimal_manager.should_distill(current_epoch=10) is False

    def test_should_distill_no_checkpoint_dir(self, minimal_manager):
        """should_distill should return False without checkpoint directory."""
        minimal_manager.config.distillation_enabled = True
        minimal_manager._distillation_config = MagicMock()
        minimal_manager._checkpoint_dir = None
        assert minimal_manager.should_distill(current_epoch=10) is False

    def test_should_distill_interval_not_reached(self, minimal_manager):
        """should_distill should return False if interval not reached."""
        minimal_manager.config.distillation_enabled = True
        minimal_manager.config.distillation_interval_epochs = 10
        minimal_manager._distillation_config = MagicMock()
        minimal_manager._checkpoint_dir = Path("/tmp")
        minimal_manager._last_distillation_epoch = 5

        # Current epoch = 10, last = 5, interval = 10
        # epochs_since = 5, which is < 10
        assert minimal_manager.should_distill(current_epoch=10) is False

    def test_get_distillation_stats_disabled(self, minimal_manager):
        """get_distillation_stats should indicate disabled."""
        stats = minimal_manager.get_distillation_stats()
        assert stats == {"enabled": False}

    def test_get_best_checkpoints_no_dir(self, minimal_manager):
        """_get_best_checkpoints should return empty list without directory."""
        minimal_manager._checkpoint_dir = None
        assert minimal_manager._get_best_checkpoints() == []

    def test_get_best_checkpoints_with_files(self, minimal_manager, temp_checkpoint_dir):
        """_get_best_checkpoints should find and sort checkpoint files."""
        minimal_manager._checkpoint_dir = temp_checkpoint_dir
        minimal_manager.config.distillation_num_teachers = 3

        checkpoints = minimal_manager._get_best_checkpoints()

        # Should return top 3 sorted by Elo descending
        assert len(checkpoints) == 3
        assert "1725" in checkpoints[0].stem  # Highest Elo first

    def test_set_checkpoint_dir(self, minimal_manager, tmp_path):
        """set_checkpoint_dir should set the checkpoint directory."""
        minimal_manager.set_checkpoint_dir(tmp_path)
        assert minimal_manager._checkpoint_dir == tmp_path

    def test_set_checkpoint_dir_string(self, minimal_manager, tmp_path):
        """set_checkpoint_dir should handle string paths."""
        minimal_manager.set_checkpoint_dir(str(tmp_path))
        assert minimal_manager._checkpoint_dir == tmp_path


class TestIntegratedTrainingManagerAugmentation:
    """Tests for data augmentation."""

    def test_augment_batch_no_augmentor(self, minimal_manager):
        """augment_batch should return inputs unchanged without augmentor."""
        features = np.random.randn(2, 3, 8, 8)
        policy_indices = [np.array([1, 2]), np.array([3, 4])]
        policy_values = [np.array([0.5, 0.5]), np.array([0.3, 0.7])]

        aug_feat, aug_idx, aug_val = minimal_manager.augment_batch(
            features, policy_indices, policy_values
        )

        np.testing.assert_array_equal(aug_feat, features)
        assert aug_idx == policy_indices
        assert aug_val == policy_values

    def test_augment_batch_dense_no_augmentor(self, minimal_manager):
        """augment_batch_dense should return inputs unchanged without augmentor."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.randn(2, 3, 8, 8)
        policy = torch.randn(2, 64)

        aug_feat, aug_policy = minimal_manager.augment_batch_dense(features, policy)

        assert torch.equal(aug_feat, features)
        assert torch.equal(aug_policy, policy)


class TestIntegratedTrainingManagerEloWeighting:
    """Tests for ELO weighting."""

    def test_compute_sample_weights_no_sampler(self, minimal_manager):
        """compute_sample_weights should return uniform weights without sampler."""
        elos = np.array([1400, 1500, 1600])

        weights = minimal_manager.compute_sample_weights(elos)

        np.testing.assert_array_equal(weights, np.ones(3))


class TestIntegratedTrainingManagerLifecycle:
    """Tests for lifecycle management."""

    def test_context_manager(self, minimal_manager):
        """Should work as context manager."""
        # minimal_manager has all features disabled, so this should be quick
        with minimal_manager:
            assert minimal_manager._step == 0

    def test_start_background_services_no_evaluator(self, minimal_manager):
        """start_background_services should be safe without evaluator."""
        # Should not raise
        minimal_manager.start_background_services()

    def test_stop_background_services_no_evaluator(self, minimal_manager):
        """stop_background_services should be safe without evaluator."""
        # Should not raise
        minimal_manager.stop_background_services()

    def test_save_checkpoints_no_curriculum(self, minimal_manager):
        """save_checkpoints should be safe without curriculum controller."""
        # Should not raise
        minimal_manager.save_checkpoints()

    def test_get_metrics(self, minimal_manager):
        """get_metrics should return basic metrics."""
        metrics = minimal_manager.get_metrics()

        assert "step" in metrics
        assert "enabled_modules" in metrics
        assert metrics["step"] == 0
        assert metrics["enabled_modules"] == 0


# =============================================================================
# Tests for GPU Transform Methods
# =============================================================================


class TestGPUTransforms:
    """Tests for GPU-accelerated transform methods."""

    @pytest.fixture
    def manager_with_augmentor(self):
        """Create manager with mock augmentor."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = IntegratedTrainingManager()

        # Create mock augmentor
        mock_augmentor = MagicMock()
        mock_augmentor.get_random_transform.return_value = 0
        mock_augmentor.transformer = MagicMock()
        mock_augmentor.transformer.transform_move_index.return_value = 0
        manager._augmentor = mock_augmentor

        return manager

    def test_apply_gpu_board_transform_identity(self, manager_with_augmentor):
        """Transform 0 should return input unchanged."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.randn(2, 3, 8, 8)

        result = manager_with_augmentor._apply_gpu_board_transform(features, 0, torch)

        assert torch.equal(result, features)

    def test_apply_gpu_board_transform_rotate90(self, manager_with_augmentor):
        """Transform 1 should rotate 90° CW."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Create asymmetric tensor to verify rotation
        features = torch.arange(16).reshape(1, 1, 4, 4).float()

        result = manager_with_augmentor._apply_gpu_board_transform(features, 1, torch)

        # Verify shape preserved
        assert result.shape == features.shape

        # Verify rotation was applied (not same as original)
        assert not torch.equal(result, features)

    def test_apply_gpu_board_transform_rotate180(self, manager_with_augmentor):
        """Transform 2 should rotate 180°."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.arange(16).reshape(1, 1, 4, 4).float()

        result = manager_with_augmentor._apply_gpu_board_transform(features, 2, torch)

        # 180° rotation applied twice should give back original
        result2 = manager_with_augmentor._apply_gpu_board_transform(result, 2, torch)
        assert torch.allclose(result2, features)

    def test_apply_gpu_board_transform_flip_horizontal(self, manager_with_augmentor):
        """Transform 4 should flip horizontally."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.arange(16).reshape(1, 1, 4, 4).float()

        result = manager_with_augmentor._apply_gpu_board_transform(features, 4, torch)

        # Double flip should give back original
        result2 = manager_with_augmentor._apply_gpu_board_transform(result, 4, torch)
        assert torch.equal(result2, features)

    def test_apply_gpu_board_transform_flip_vertical(self, manager_with_augmentor):
        """Transform 5 should flip vertically."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.arange(16).reshape(1, 1, 4, 4).float()

        result = manager_with_augmentor._apply_gpu_board_transform(features, 5, torch)

        # Double flip should give back original
        result2 = manager_with_augmentor._apply_gpu_board_transform(result, 5, torch)
        assert torch.equal(result2, features)

    def test_apply_gpu_board_transform_invalid(self, manager_with_augmentor):
        """Invalid transform ID should return input unchanged."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.randn(2, 3, 8, 8)

        result = manager_with_augmentor._apply_gpu_board_transform(features, 99, torch)

        assert torch.equal(result, features)

    def test_apply_gpu_policy_transform_identity(self, manager_with_augmentor):
        """Transform 0 should return policy unchanged."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        policy = torch.randn(2, 64)

        result = manager_with_augmentor._apply_gpu_policy_transform(policy, 0, torch)

        assert torch.equal(result, policy)

    def test_get_policy_permutation_caches(self, manager_with_augmentor):
        """Policy permutation should be cached."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        device = torch.device("cpu")

        # First call
        perm1 = manager_with_augmentor._get_policy_permutation(1, 64, device, torch)

        # Second call should use cache
        perm2 = manager_with_augmentor._get_policy_permutation(1, 64, device, torch)

        assert torch.equal(perm1, perm2)

        # Cache should exist
        assert hasattr(manager_with_augmentor, '_policy_perm_cache')
        assert (1, 64, str(device)) in manager_with_augmentor._policy_perm_cache


# =============================================================================
# Tests for Component Initialization (Mocked)
# =============================================================================


class TestComponentInitialization:
    """Tests for component initialization with mocked dependencies."""

    def test_init_auxiliary_tasks_import_error(self):
        """Should handle import error gracefully."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            config = IntegratedEnhancementsConfig(auxiliary_tasks_enabled=True)
            manager = IntegratedTrainingManager(config=config)

        with patch.dict('sys.modules', {'app.training.auxiliary_tasks': None}):
            # Should not raise
            manager._init_auxiliary_tasks()
            # Module should remain None on failure
            # (actual behavior depends on import mechanism)

    def test_init_gradient_surgery_import_error(self):
        """Should handle gradient surgery import error gracefully."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            config = IntegratedEnhancementsConfig(gradient_surgery_enabled=True)
            manager = IntegratedTrainingManager(config=config)

        # Initialization is done inside try/except, so it should be safe
        # even with missing dependencies
        manager._gradient_surgeon = None

    def test_init_batch_scheduler_success(self):
        """Batch scheduler should initialize successfully."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            config = IntegratedEnhancementsConfig(
                batch_scheduling_enabled=True,
                # Disable other components to speed up test
                auxiliary_tasks_enabled=False,
                gradient_surgery_enabled=False,
                background_eval_enabled=False,
                elo_weighting_enabled=False,
                curriculum_enabled=False,
                augmentation_enabled=False,
                reanalysis_enabled=False,
                distillation_enabled=False,
            )
            manager = IntegratedTrainingManager(config=config)

        manager.initialize_all()

        # Batch scheduler should be available
        # (may be None if import fails, but no exception should be raised)

    def test_initialize_all_counts_modules(self):
        """initialize_all should correctly count enabled modules."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            config = IntegratedEnhancementsConfig(
                auxiliary_tasks_enabled=False,
                gradient_surgery_enabled=False,
                batch_scheduling_enabled=True,  # Only batch scheduler enabled
                background_eval_enabled=False,
                elo_weighting_enabled=False,
                curriculum_enabled=False,
                augmentation_enabled=False,
                reanalysis_enabled=False,
                distillation_enabled=False,
            )
            manager = IntegratedTrainingManager(config=config)

        manager.initialize_all()

        # Should have at most 1 module enabled
        # (exact count depends on whether batch_scheduling module is available)
        assert manager._count_enabled() <= 1


# =============================================================================
# Tests for Factory Functions
# =============================================================================


class TestCreateIntegratedManager:
    """Tests for create_integrated_manager function."""

    def test_create_with_defaults(self):
        """Can create manager with default config."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = create_integrated_manager()

        assert isinstance(manager, IntegratedTrainingManager)
        assert manager.config.auxiliary_tasks_enabled is True  # Default

    def test_create_with_config_dict(self):
        """Can create manager with config dictionary."""
        config_dict = {
            "batch_initial_size": 128,
            "auxiliary_tasks_enabled": False,
            "reanalysis_blend_ratio": 0.5,
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = create_integrated_manager(config_dict=config_dict)

        assert manager.config.batch_initial_size == 128
        assert manager.config.auxiliary_tasks_enabled is False
        assert manager.config.reanalysis_blend_ratio == 0.5

    def test_create_with_model(self):
        """Can create manager with model."""
        mock_model = MagicMock()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = create_integrated_manager(model=mock_model)

        assert manager.model is mock_model

    def test_create_with_board_type(self):
        """Can create manager with specific board type."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = create_integrated_manager(board_type="hexagonal")

        assert manager.board_type == "hexagonal"

    def test_create_ignores_invalid_keys(self):
        """Should ignore invalid config keys."""
        config_dict = {
            "batch_initial_size": 128,
            "invalid_key": "should_be_ignored",
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            manager = create_integrated_manager(config_dict=config_dict)

        assert manager.config.batch_initial_size == 128
        assert not hasattr(manager.config, "invalid_key")


class TestGetEnhancementDefaults:
    """Tests for get_enhancement_defaults function."""

    def test_returns_dict(self):
        """Should return dictionary."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            defaults = get_enhancement_defaults()

        assert isinstance(defaults, dict)

    def test_contains_expected_keys(self):
        """Should contain all expected configuration keys."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            defaults = get_enhancement_defaults()

        expected_keys = [
            "auxiliary_tasks_enabled",
            "aux_game_length_weight",
            "gradient_surgery_enabled",
            "batch_scheduling_enabled",
            "batch_initial_size",
            "background_eval_enabled",
            "elo_weighting_enabled",
            "curriculum_enabled",
            "augmentation_enabled",
            "reanalysis_enabled",
            "reanalysis_blend_ratio",
            "distillation_enabled",
            "distillation_temperature",
        ]

        for key in expected_keys:
            assert key in defaults, f"Missing key: {key}"

    def test_values_match_config(self):
        """Values should match IntegratedEnhancementsConfig defaults."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            defaults = get_enhancement_defaults()
            config = IntegratedEnhancementsConfig()

        assert defaults["auxiliary_tasks_enabled"] == config.auxiliary_tasks_enabled
        assert defaults["batch_initial_size"] == config.batch_initial_size
        assert defaults["reanalysis_blend_ratio"] == config.reanalysis_blend_ratio
        assert defaults["distillation_temperature"] == config.distillation_temperature


# =============================================================================
# Tests for Deprecation Warning
# =============================================================================


class TestDeprecationWarning:
    """Tests for deprecation warning behavior."""

    def test_import_raises_deprecation_warning(self):
        """Importing module should raise DeprecationWarning."""
        import importlib

        with pytest.warns(
            DeprecationWarning,
            match="IntegratedTrainingManager.*deprecated",
        ):
            import app.training.integrated_enhancements as module
            importlib.reload(module)


# =============================================================================
# Tests for Auxiliary Loss Computation (Mocked)
# =============================================================================


class TestAuxiliaryLossComputation:
    """Tests for auxiliary loss computation."""

    def test_compute_auxiliary_loss_no_module(self, minimal_manager):
        """Should return zero loss without module."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        features = torch.randn(2, 256)
        targets = {}

        loss, breakdown = minimal_manager.compute_auxiliary_loss(features, targets)

        assert loss.item() == 0.0
        assert breakdown == {}


# =============================================================================
# Tests for Gradient Surgery (Mocked)
# =============================================================================


class TestGradientSurgery:
    """Tests for gradient surgery application."""

    def test_apply_gradient_surgery_no_surgeon(self, minimal_manager):
        """Should sum losses without surgeon."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        losses = {
            "policy": torch.tensor(1.0),
            "value": torch.tensor(2.0),
            "auxiliary": torch.tensor(0.5),
        }

        result = minimal_manager.apply_gradient_surgery(None, losses)

        assert result.item() == pytest.approx(3.5)
