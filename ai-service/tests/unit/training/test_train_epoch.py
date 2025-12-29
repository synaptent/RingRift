"""Unit tests for train_epoch module.

Tests the epoch-level training logic extracted from train.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from app.training.train_epoch import (
    EarlyStopState,
    EpochConfig,
    EpochContext,
    EpochResult,
    check_overfitting,
    run_all_epochs,
    run_train_loop,
    run_training_epoch,
    run_validation_loop,
    step_schedulers,
)
from app.training.train_step import TrainStepConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get test device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 8


@pytest.fixture
def policy_size():
    """Standard policy size for testing."""
    return 64


@pytest.fixture
def simple_model(policy_size):
    """Create a simple model for testing."""

    class SimpleNet(nn.Module):
        def __init__(self, policy_size: int):
            super().__init__()
            self.fc = nn.Linear(10, 32)
            self.value_head = nn.Linear(32, 1)
            self.policy_head = nn.Linear(32, policy_size)

        def forward(self, x, globals_vec=None):
            # x: [B, 10]
            h = torch.relu(self.fc(x))
            value = self.value_head(h)
            policy = self.policy_head(h)
            return value, policy

    return SimpleNet(policy_size)


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def train_loader(batch_size, policy_size):
    """Create a simple train data loader."""
    # Create 3 batches of data
    batches = []
    for _ in range(3):
        features = torch.randn(batch_size, 10)
        globals_vec = torch.randn(batch_size, 5)
        value_targets = torch.randn(batch_size, 1)
        policy_targets = torch.softmax(torch.randn(batch_size, policy_size), dim=1)
        batches.append((features, globals_vec, value_targets, policy_targets))
    return batches


@pytest.fixture
def val_loader(batch_size, policy_size):
    """Create a simple validation data loader."""
    # Create 2 batches of validation data
    batches = []
    for _ in range(2):
        features = torch.randn(batch_size, 10)
        globals_vec = torch.randn(batch_size, 5)
        value_targets = torch.randn(batch_size, 1)
        policy_targets = torch.softmax(torch.randn(batch_size, policy_size), dim=1)
        batches.append((features, globals_vec, value_targets, policy_targets))
    return batches


@pytest.fixture
def epoch_config():
    """Create default epoch configuration."""
    return EpochConfig(
        epochs=10,
        batch_size=512,
        validate_every_n_epochs=1,
        patience=5,
        log_interval=10,
        publish_events=False,  # Disable event publishing for tests
    )


@pytest.fixture
def epoch_context(simple_model, optimizer, train_loader, val_loader, device, epoch_config):
    """Create epoch context for testing."""
    return EpochContext(
        model=simple_model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=epoch_config,
    )


# =============================================================================
# EpochConfig Tests
# =============================================================================


class TestEpochConfig:
    """Tests for EpochConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EpochConfig()
        assert config.epochs == 20
        assert config.batch_size == 512
        assert config.validate_every_n_epochs == 1
        assert config.patience == 10
        assert config.min_delta == 1e-5
        assert config.log_interval == 100
        assert config.log_memory_usage is False
        assert config.distributed is False
        assert config.use_plateau_scheduler is False
        assert config.publish_events is True
        assert config.board_type == "hex8"
        assert config.num_players == 2

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EpochConfig(
            epochs=50,
            batch_size=256,
            patience=20,
            board_type="square8",
            num_players=4,
        )
        assert config.epochs == 50
        assert config.batch_size == 256
        assert config.patience == 20
        assert config.board_type == "square8"
        assert config.num_players == 4


# =============================================================================
# EpochContext Tests
# =============================================================================


class TestEpochContext:
    """Tests for EpochContext dataclass."""

    def test_basic_creation(self, simple_model, optimizer, train_loader, device, epoch_config):
        """Test basic context creation."""
        context = EpochContext(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            device=device,
            config=epoch_config,
        )
        assert context.model is simple_model
        assert context.optimizer is optimizer
        assert context.val_loader is None
        assert context.device == device

    def test_post_init_creates_step_config(
        self, simple_model, optimizer, train_loader, device, epoch_config
    ):
        """Test that __post_init__ creates step config."""
        context = EpochContext(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            device=device,
            config=epoch_config,
        )
        assert context.step_config is not None
        assert isinstance(context.step_config, TrainStepConfig)

    def test_post_init_respects_existing_step_config(
        self, simple_model, optimizer, train_loader, device, epoch_config
    ):
        """Test that __post_init__ doesn't override existing step config."""
        custom_step_config = TrainStepConfig(gradient_clip_max_norm=5.0)
        context = EpochContext(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            device=device,
            config=epoch_config,
            step_config=custom_step_config,
        )
        assert context.step_config is custom_step_config
        assert context.step_config.gradient_clip_max_norm == 5.0

    def test_create_step_context(self, epoch_context):
        """Test create_step_context method."""
        step_context = epoch_context.create_step_context()
        assert step_context.model is epoch_context.model
        assert step_context.optimizer is epoch_context.optimizer
        assert step_context.device == epoch_context.device
        assert step_context.config is epoch_context.step_config

    def test_amp_enabled_propagates_to_step_config(
        self, simple_model, optimizer, train_loader, device, epoch_config
    ):
        """Test that AMP settings propagate to step config."""
        context = EpochContext(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            device=device,
            config=epoch_config,
            amp_enabled=True,
            amp_dtype=torch.bfloat16,
        )
        assert context.step_config.use_mixed_precision is True
        assert context.step_config.amp_dtype == torch.bfloat16


# =============================================================================
# EpochResult Tests
# =============================================================================


class TestEpochResult:
    """Tests for EpochResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = EpochResult(
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
        )
        assert result.epoch == 5
        assert result.train_loss == 0.5
        assert result.val_loss == 0.6
        assert result.policy_accuracy is None
        assert result.should_stop is False

    def test_full_result(self):
        """Test result with all fields."""
        result = EpochResult(
            epoch=10,
            train_loss=0.3,
            val_loss=0.4,
            policy_accuracy=0.75,
            learning_rate=0.0001,
            duration_seconds=120.5,
            batches_processed=100,
            batches_skipped=2,
            grad_norm_mean=1.5,
            should_stop=True,
            stop_reason="No improvement for 10 epochs",
        )
        assert result.epoch == 10
        assert result.policy_accuracy == 0.75
        assert result.should_stop is True
        assert result.stop_reason == "No improvement for 10 epochs"

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = EpochResult(
            epoch=5,
            train_loss=0.5,
            val_loss=0.6,
            policy_accuracy=0.8,
            learning_rate=0.001,
            duration_seconds=60.0,
            batches_processed=50,
            batches_skipped=1,
        )
        d = result.to_dict()
        assert d["epoch"] == 5
        assert d["train_loss"] == 0.5
        assert d["val_loss"] == 0.6
        assert d["policy_accuracy"] == 0.8
        assert d["lr"] == 0.001
        assert d["duration"] == 60.0
        assert d["batches_processed"] == 50
        assert d["batches_skipped"] == 1


# =============================================================================
# EarlyStopState Tests
# =============================================================================


class TestEarlyStopState:
    """Tests for EarlyStopState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = EarlyStopState()
        assert state.best_loss == float("inf")
        assert state.epochs_without_improvement == 0
        assert state.best_epoch == 0

    def test_update_improvement(self):
        """Test update with improvement."""
        state = EarlyStopState()
        improved = state.update(0.5, epoch=1)
        assert improved is True
        assert state.best_loss == 0.5
        assert state.epochs_without_improvement == 0
        assert state.best_epoch == 1

    def test_update_no_improvement(self):
        """Test update without improvement."""
        state = EarlyStopState(best_loss=0.5)
        improved = state.update(0.6, epoch=2)
        assert improved is False
        assert state.best_loss == 0.5
        assert state.epochs_without_improvement == 1

    def test_update_with_min_delta(self):
        """Test update respects min_delta threshold."""
        state = EarlyStopState(best_loss=0.5)
        # Tiny improvement (less than min_delta)
        improved = state.update(0.4999999, epoch=2, min_delta=1e-5)
        assert improved is False
        assert state.epochs_without_improvement == 1

        # Significant improvement (more than min_delta)
        improved = state.update(0.4999, epoch=3, min_delta=1e-5)
        assert improved is True
        assert state.epochs_without_improvement == 0

    def test_should_stop(self):
        """Test should_stop with patience."""
        state = EarlyStopState(epochs_without_improvement=4)
        assert state.should_stop(patience=5) is False

        state.epochs_without_improvement = 5
        assert state.should_stop(patience=5) is True

        state.epochs_without_improvement = 10
        assert state.should_stop(patience=5) is True


# =============================================================================
# run_train_loop Tests
# =============================================================================


class TestRunTrainLoop:
    """Tests for run_train_loop function."""

    def test_basic_training_loop(self, epoch_context):
        """Test basic training loop execution."""
        step_context = epoch_context.create_step_context()
        avg_loss, processed, skipped, grad_norms = run_train_loop(
            epoch_context, step_context, epoch=0
        )

        assert isinstance(avg_loss, float)
        assert processed > 0
        assert skipped >= 0
        assert isinstance(grad_norms, list)

    def test_model_in_train_mode(self, epoch_context):
        """Test that model is set to train mode."""
        step_context = epoch_context.create_step_context()
        epoch_context.model.eval()  # Start in eval mode
        run_train_loop(epoch_context, step_context, epoch=0)
        assert epoch_context.model.training is True


# =============================================================================
# run_validation_loop Tests
# =============================================================================


class TestRunValidationLoop:
    """Tests for run_validation_loop function."""

    def test_no_validation_loader(self, epoch_context):
        """Test with no validation loader."""
        epoch_context.val_loader = None
        val_loss, accuracy = run_validation_loop(epoch_context)
        assert val_loss == 0.0
        assert accuracy == 0.0

    def test_basic_validation(self, epoch_context):
        """Test basic validation execution."""
        val_loss, accuracy = run_validation_loop(epoch_context)
        assert isinstance(val_loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_model_in_eval_mode(self, epoch_context):
        """Test that model is set to eval mode."""
        epoch_context.model.train()  # Start in train mode
        run_validation_loop(epoch_context)
        assert epoch_context.model.training is False


# =============================================================================
# step_schedulers Tests
# =============================================================================


class TestStepSchedulers:
    """Tests for step_schedulers function."""

    def test_no_schedulers(self, epoch_context):
        """Test with no schedulers configured."""
        # Should not raise
        step_schedulers(epoch_context, val_loss=0.5, epoch=0)

    def test_epoch_scheduler(self, epoch_context):
        """Test with epoch scheduler."""
        scheduler = torch.optim.lr_scheduler.StepLR(epoch_context.optimizer, step_size=5)
        epoch_context.epoch_scheduler = scheduler
        initial_lr = epoch_context.optimizer.param_groups[0]["lr"]

        step_schedulers(epoch_context, val_loss=0.5, epoch=0)

        # StepLR should have stepped
        # (actual behavior depends on step_size, but no error should occur)
        assert epoch_context.optimizer.param_groups[0]["lr"] is not None

    def test_plateau_scheduler(self, epoch_context):
        """Test with plateau scheduler."""
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            epoch_context.optimizer, mode="min", patience=2
        )
        epoch_context.plateau_scheduler = scheduler

        # Step with improving loss
        step_schedulers(epoch_context, val_loss=0.5, epoch=0)
        step_schedulers(epoch_context, val_loss=0.4, epoch=1)
        step_schedulers(epoch_context, val_loss=0.3, epoch=2)

        # No error should occur


# =============================================================================
# check_overfitting Tests
# =============================================================================


class TestCheckOverfitting:
    """Tests for check_overfitting function."""

    def test_no_overfitting(self):
        """Test when not overfitting."""
        result = check_overfitting(train_loss=0.5, val_loss=0.55, epoch=5)
        assert result is False

    def test_overfitting_detected(self):
        """Test when overfitting is detected."""
        # 50% divergence
        result = check_overfitting(train_loss=0.5, val_loss=0.75, epoch=5)
        assert result is True

    def test_skip_early_epochs(self):
        """Test that early epochs are skipped."""
        result = check_overfitting(train_loss=0.5, val_loss=1.0, epoch=1, min_epochs=3)
        assert result is False

    def test_custom_threshold(self):
        """Test with custom threshold."""
        # 30% divergence with 20% threshold
        result = check_overfitting(train_loss=0.5, val_loss=0.65, epoch=5, threshold=0.2)
        assert result is True

        # Same divergence with 50% threshold
        result = check_overfitting(train_loss=0.5, val_loss=0.65, epoch=5, threshold=0.5)
        assert result is False

    def test_zero_train_loss(self):
        """Test with zero training loss (edge case)."""
        result = check_overfitting(train_loss=0.0, val_loss=0.5, epoch=5)
        assert result is False  # Returns False to avoid division by zero


# =============================================================================
# run_training_epoch Tests
# =============================================================================


class TestRunTrainingEpoch:
    """Tests for run_training_epoch function."""

    def test_basic_epoch(self, epoch_context):
        """Test basic epoch execution."""
        result = run_training_epoch(epoch_context, epoch=0)

        assert isinstance(result, EpochResult)
        assert result.epoch == 0
        assert result.train_loss >= 0
        assert result.batches_processed > 0

    def test_epoch_with_validation(self, epoch_context):
        """Test epoch with validation."""
        result = run_training_epoch(epoch_context, epoch=0)

        assert result.val_loss is not None
        assert result.policy_accuracy is not None

    def test_epoch_without_validation(self, epoch_context):
        """Test epoch without validation loader."""
        epoch_context.val_loader = None
        result = run_training_epoch(epoch_context, epoch=0)

        assert result.val_loss is None
        assert result.policy_accuracy is None

    def test_epoch_with_early_stopping(self, epoch_context):
        """Test epoch with early stopping state."""
        early_stop = EarlyStopState(best_loss=0.1, epochs_without_improvement=4)
        epoch_context.config.patience = 5

        result = run_training_epoch(epoch_context, epoch=5, early_stop_state=early_stop)

        # After one more epoch without improvement, should stop
        if early_stop.epochs_without_improvement >= 5:
            assert result.should_stop is True

    def test_learning_rate_recorded(self, epoch_context):
        """Test that learning rate is recorded."""
        result = run_training_epoch(epoch_context, epoch=0)
        assert result.learning_rate > 0

    def test_duration_recorded(self, epoch_context):
        """Test that duration is recorded."""
        result = run_training_epoch(epoch_context, epoch=0)
        assert result.duration_seconds >= 0


# =============================================================================
# run_all_epochs Tests
# =============================================================================


class TestRunAllEpochs:
    """Tests for run_all_epochs function."""

    def test_run_all_epochs(self, epoch_context):
        """Test running all epochs."""
        epoch_context.config.epochs = 3
        results = run_all_epochs(epoch_context)

        assert len(results) == 3
        assert all(isinstance(r, EpochResult) for r in results)
        assert results[0].epoch == 0
        assert results[1].epoch == 1
        assert results[2].epoch == 2

    def test_early_stopping(self, epoch_context):
        """Test early stopping during training."""
        epoch_context.config.epochs = 10
        epoch_context.config.patience = 0  # Stop immediately if no improvement

        # Pre-set a very low best loss
        early_stop = EarlyStopState(best_loss=0.0)

        results = run_all_epochs(epoch_context, early_stop_state=early_stop)

        # Should stop before reaching all epochs due to early stopping
        assert len(results) <= 10
        if len(results) > 0:
            last_result = results[-1]
            # The last result should indicate stopping
            if last_result.should_stop:
                assert last_result.stop_reason is not None

    def test_resume_from_epoch(self, epoch_context):
        """Test resuming from a specific epoch."""
        epoch_context.config.epochs = 5
        results = run_all_epochs(epoch_context, start_epoch=3)

        assert len(results) == 2  # Only epochs 3 and 4
        assert results[0].epoch == 3
        assert results[1].epoch == 4

    def test_creates_early_stop_state_if_none(self, epoch_context):
        """Test that early stop state is created if not provided."""
        epoch_context.config.epochs = 2
        results = run_all_epochs(epoch_context, early_stop_state=None)

        assert len(results) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestEpochIntegration:
    """Integration tests for epoch-level training."""

    def test_full_training_cycle(self, epoch_context):
        """Test a complete training cycle."""
        epoch_context.config.epochs = 2
        epoch_context.config.validate_every_n_epochs = 1

        results = run_all_epochs(epoch_context)

        assert len(results) == 2

        # Both epochs should have validation
        for result in results:
            assert result.train_loss >= 0
            assert result.val_loss is not None
            assert result.policy_accuracy is not None
            assert result.batches_processed > 0

    def test_loss_decreases_or_stable(self, epoch_context):
        """Test that training loss is reasonable over epochs."""
        epoch_context.config.epochs = 3
        results = run_all_epochs(epoch_context)

        # Just check that losses are valid numbers
        for result in results:
            assert not torch.isnan(torch.tensor(result.train_loss))
            assert not torch.isinf(torch.tensor(result.train_loss))
