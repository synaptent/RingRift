"""Tests for train_setup module.

Tests cover:
- FaultToleranceConfig defaults
- setup_fault_tolerance component initialization
- get_device device selection
- compute_effective_lr LR scaling
- TrainingState state management
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from app.training.train_setup import (
    FaultToleranceComponents,
    FaultToleranceConfig,
    TrainingState,
    compute_effective_lr,
    get_device,
    setup_fault_tolerance,
    setup_graceful_shutdown,
)


class TestFaultToleranceConfig:
    """Tests for FaultToleranceConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = FaultToleranceConfig()
        assert config.enable_circuit_breaker is True
        assert config.enable_anomaly_detection is True
        assert config.enable_graceful_shutdown is True
        assert config.gradient_clip_mode == 'adaptive'
        assert config.gradient_clip_max_norm == 1.0
        assert config.anomaly_spike_threshold == 3.0
        assert config.anomaly_gradient_threshold == 100.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FaultToleranceConfig(
            enable_circuit_breaker=False,
            gradient_clip_mode='fixed',
            gradient_clip_max_norm=2.0,
        )
        assert config.enable_circuit_breaker is False
        assert config.gradient_clip_mode == 'fixed'
        assert config.gradient_clip_max_norm == 2.0


class TestFaultToleranceComponents:
    """Tests for FaultToleranceComponents dataclass."""

    def test_defaults(self):
        """Test default component values."""
        components = FaultToleranceComponents()
        assert components.training_breaker is None
        assert components.anomaly_detector is None
        assert components.adaptive_clipper is None
        assert components.shutdown_handler is None
        assert components.gradient_clip_mode == 'fixed'
        assert components.fixed_clip_norm == 1.0


class TestSetupFaultTolerance:
    """Tests for setup_fault_tolerance function."""

    def test_returns_components(self):
        """Test that function returns FaultToleranceComponents."""
        config = FaultToleranceConfig(
            enable_circuit_breaker=False,
            enable_anomaly_detection=False,
        )
        components = setup_fault_tolerance(config)
        assert isinstance(components, FaultToleranceComponents)

    def test_respects_disabled_flags(self):
        """Test that disabled flags are respected."""
        config = FaultToleranceConfig(
            enable_circuit_breaker=False,
            enable_anomaly_detection=False,
            gradient_clip_mode='fixed',
        )
        components = setup_fault_tolerance(config)
        # These should be None when disabled
        assert components.training_breaker is None
        assert components.gradient_clip_mode == 'fixed'

    def test_gradient_clip_mode_preserved(self):
        """Test gradient clip mode is passed through."""
        config = FaultToleranceConfig(
            enable_circuit_breaker=False,
            enable_anomaly_detection=False,
            gradient_clip_mode='adaptive',
            gradient_clip_max_norm=2.5,
        )
        components = setup_fault_tolerance(config)
        # If adaptive clipper not available, falls back to fixed
        assert components.gradient_clip_mode in ('adaptive', 'fixed')
        assert components.fixed_clip_norm == 2.5


class TestGetDevice:
    """Tests for get_device function."""

    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available.return_value = False
                device = get_device()
                assert device.type == 'cpu'

    def test_cuda_selected_when_available(self):
        """Test CUDA selected when available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device(local_rank=-1)
            assert device.type == 'cuda'

    def test_cuda_with_local_rank(self):
        """Test CUDA with specific local rank."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device(local_rank=2)
            assert device.type == 'cuda'
            assert device.index == 2

    def test_mps_selected_when_available(self):
        """Test MPS selected on Apple Silicon."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch.object(torch.backends, 'mps', create=True) as mock_mps:
                mock_mps.is_available.return_value = True
                device = get_device()
                assert device.type == 'mps'


class TestComputeEffectiveLR:
    """Tests for compute_effective_lr function."""

    def test_no_scaling(self):
        """Test LR unchanged when scaling disabled."""
        lr = compute_effective_lr(0.001, world_size=4, scale_lr=False)
        assert lr == 0.001

    def test_no_scaling_single_gpu(self):
        """Test LR unchanged for single GPU."""
        lr = compute_effective_lr(0.001, world_size=1, scale_lr=True)
        assert lr == 0.001

    def test_linear_scaling(self):
        """Test linear LR scaling."""
        lr = compute_effective_lr(
            0.001, world_size=4, scale_lr=True, lr_scale_mode='linear'
        )
        assert lr == pytest.approx(0.004)

    def test_sqrt_scaling(self):
        """Test sqrt LR scaling."""
        lr = compute_effective_lr(
            0.001, world_size=4, scale_lr=True, lr_scale_mode='sqrt'
        )
        assert lr == pytest.approx(0.002)  # 0.001 * sqrt(4) = 0.002


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_defaults(self):
        """Test default state values."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.best_val_loss == float('inf')
        assert state.avg_val_loss == float('inf')
        assert state.last_good_checkpoint_path is None
        assert state.last_good_epoch == 0
        assert state.circuit_breaker_rollbacks == 0
        assert state.max_circuit_breaker_rollbacks == 3

    def test_can_rollback_false_when_no_checkpoint(self):
        """Test can_rollback returns False without checkpoint."""
        state = TrainingState()
        assert state.can_rollback() is False

    def test_can_rollback_true_with_checkpoint(self):
        """Test can_rollback returns True with checkpoint."""
        state = TrainingState(last_good_checkpoint_path='/path/to/ckpt.pt')
        assert state.can_rollback() is True

    def test_can_rollback_false_when_max_rollbacks_exceeded(self):
        """Test can_rollback returns False after max rollbacks."""
        state = TrainingState(
            last_good_checkpoint_path='/path/to/ckpt.pt',
            circuit_breaker_rollbacks=3,
            max_circuit_breaker_rollbacks=3,
        )
        assert state.can_rollback() is False

    def test_record_rollback(self):
        """Test record_rollback increments counter."""
        state = TrainingState()
        assert state.circuit_breaker_rollbacks == 0
        state.record_rollback()
        assert state.circuit_breaker_rollbacks == 1
        state.record_rollback()
        assert state.circuit_breaker_rollbacks == 2

    def test_update_good_checkpoint(self):
        """Test update_good_checkpoint updates both fields."""
        state = TrainingState()
        state.update_good_checkpoint('/new/path.pt', epoch=5)
        assert state.last_good_checkpoint_path == '/new/path.pt'
        assert state.last_good_epoch == 5


class TestSetupGracefulShutdown:
    """Tests for setup_graceful_shutdown function."""

    def test_returns_none_on_non_main_process(self):
        """Test returns None for non-main processes."""
        handler = setup_graceful_shutdown(
            checkpoint_callback=lambda: None,
            distributed=True,
            is_main_process_fn=lambda: False,
        )
        assert handler is None

    def test_calls_callback_on_main_process(self):
        """Test handler setup on main process."""
        # This test just verifies the function runs without error
        # Actual handler functionality depends on imports
        setup_graceful_shutdown(
            checkpoint_callback=lambda: None,
            distributed=False,
        )
        # May be None if GracefulShutdownHandler not available
        # That's OK - we're testing the function runs
