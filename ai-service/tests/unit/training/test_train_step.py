"""Unit tests for train_step.py.

December 2025: Tests for the core batch-level training step module.
"""

import pytest
import torch
import torch.nn as nn

from app.training.train_step import (
    BatchData,
    LossComponents,
    TrainStepConfig,
    TrainStepContext,
    TrainStepResult,
    apply_label_smoothing,
    compute_entropy_bonus,
    compute_policy_loss,
    compute_value_loss,
    pad_policy_targets,
    parse_batch,
    run_backward,
    run_training_step,
    step_optimizer,
    transfer_batch_to_device,
    validate_policy_targets,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 8


@pytest.fixture
def policy_size():
    """Standard policy size for tests."""
    return 64


@pytest.fixture
def simple_batch(batch_size, policy_size):
    """Create a simple 4-tuple batch."""
    return (
        torch.randn(batch_size, 40, 9, 9),  # features
        torch.randn(batch_size, 20),  # globals
        torch.rand(batch_size),  # value targets
        torch.softmax(torch.randn(batch_size, policy_size), dim=1),  # policy targets
    )


@pytest.fixture
def streaming_batch(batch_size, policy_size):
    """Create a streaming format batch."""
    features = torch.randn(batch_size, 40, 9, 9)
    globals_vec = torch.randn(batch_size, 20)
    value_targets = torch.rand(batch_size)
    policy_targets = torch.softmax(torch.randn(batch_size, policy_size), dim=1)
    return ((features, globals_vec), (value_targets, policy_targets))


@pytest.fixture
def simple_model(policy_size):
    """Create a simple mock model for testing."""
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_size = policy_size
            self.value_head = nn.Linear(20, 1)
            self.policy_head = nn.Linear(20, policy_size)

        def forward(self, features, globals_vec):
            # Simple forward - just use globals
            batch_size = globals_vec.size(0)
            value = self.value_head(globals_vec)
            policy = self.policy_head(globals_vec)
            return value, policy

    return MockModel()


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for the model."""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


# =============================================================================
# BatchData Tests
# =============================================================================


class TestBatchData:
    """Tests for BatchData dataclass."""

    def test_create_basic(self, batch_size, policy_size):
        """Test creating BatchData with required fields only."""
        batch = BatchData(
            features=torch.randn(batch_size, 40, 9, 9),
            globals_vec=torch.randn(batch_size, 20),
            value_targets=torch.rand(batch_size),
            policy_targets=torch.rand(batch_size, policy_size),
        )
        assert batch.features.shape[0] == batch_size
        assert batch.num_players is None
        assert batch.heuristics is None

    def test_create_with_num_players(self, batch_size, policy_size):
        """Test creating BatchData with num_players."""
        batch = BatchData(
            features=torch.randn(batch_size, 40, 9, 9),
            globals_vec=torch.randn(batch_size, 20),
            value_targets=torch.rand(batch_size),
            policy_targets=torch.rand(batch_size, policy_size),
            num_players=torch.tensor([2] * batch_size, dtype=torch.long),
        )
        assert batch.num_players is not None
        assert batch.num_players.shape == (batch_size,)

    def test_create_with_heuristics(self, batch_size, policy_size):
        """Test creating BatchData with heuristics."""
        batch = BatchData(
            features=torch.randn(batch_size, 40, 9, 9),
            globals_vec=torch.randn(batch_size, 20),
            value_targets=torch.rand(batch_size),
            policy_targets=torch.rand(batch_size, policy_size),
            heuristics=torch.randn(batch_size, 21),  # 21 fast heuristics
        )
        assert batch.heuristics is not None
        assert batch.heuristics.shape == (batch_size, 21)


# =============================================================================
# Parse Batch Tests
# =============================================================================


class TestParseBatch:
    """Tests for parse_batch function."""

    def test_parse_4_tuple(self, simple_batch):
        """Test parsing a 4-tuple batch."""
        batch = parse_batch(simple_batch, is_streaming=False)
        assert isinstance(batch, BatchData)
        assert batch.features.shape[0] == 8
        assert batch.globals_vec.shape[0] == 8
        assert batch.num_players is None

    def test_parse_5_tuple_with_num_players(self, batch_size, policy_size):
        """Test parsing a 5-tuple batch with num_players."""
        raw = (
            torch.randn(batch_size, 40, 9, 9),
            torch.randn(batch_size, 20),
            torch.rand(batch_size),
            torch.softmax(torch.randn(batch_size, policy_size), dim=1),
            torch.tensor([2] * batch_size, dtype=torch.long),  # num_players
        )
        batch = parse_batch(raw, is_streaming=False)
        assert batch.num_players is not None
        assert batch.heuristics is None

    def test_parse_5_tuple_with_heuristics(self, batch_size, policy_size):
        """Test parsing a 5-tuple batch with heuristics."""
        raw = (
            torch.randn(batch_size, 40, 9, 9),
            torch.randn(batch_size, 20),
            torch.rand(batch_size),
            torch.softmax(torch.randn(batch_size, policy_size), dim=1),
            torch.randn(batch_size, 21),  # heuristics (float)
        )
        batch = parse_batch(raw, is_streaming=False)
        assert batch.num_players is None
        assert batch.heuristics is not None

    def test_parse_6_tuple(self, batch_size, policy_size):
        """Test parsing a 6-tuple batch."""
        raw = (
            torch.randn(batch_size, 40, 9, 9),
            torch.randn(batch_size, 20),
            torch.rand(batch_size),
            torch.softmax(torch.randn(batch_size, policy_size), dim=1),
            torch.tensor([4] * batch_size, dtype=torch.long),  # num_players
            torch.randn(batch_size, 49),  # full heuristics
        )
        batch = parse_batch(raw, is_streaming=False)
        assert batch.num_players is not None
        assert batch.heuristics is not None
        assert batch.heuristics.shape == (batch_size, 49)

    def test_parse_streaming(self, streaming_batch):
        """Test parsing a streaming format batch."""
        batch = parse_batch(streaming_batch, is_streaming=True)
        assert isinstance(batch, BatchData)
        assert batch.features.shape[0] == 8


# =============================================================================
# Transfer Batch Tests
# =============================================================================


class TestTransferBatchToDevice:
    """Tests for transfer_batch_to_device function."""

    def test_transfer_to_same_device(self, simple_batch, device):
        """Test transferring batch when already on target device."""
        batch = parse_batch(simple_batch, is_streaming=False)
        transferred = transfer_batch_to_device(batch, device)
        assert transferred.features.device == device
        assert transferred.globals_vec.device == device

    def test_transfer_with_optional_fields(self, batch_size, policy_size, device):
        """Test transferring batch with optional fields."""
        batch = BatchData(
            features=torch.randn(batch_size, 40, 9, 9),
            globals_vec=torch.randn(batch_size, 20),
            value_targets=torch.rand(batch_size),
            policy_targets=torch.rand(batch_size, policy_size),
            num_players=torch.tensor([2] * batch_size, dtype=torch.long),
            heuristics=torch.randn(batch_size, 21),
        )
        transferred = transfer_batch_to_device(batch, device)
        assert transferred.num_players.device == device
        assert transferred.heuristics.device == device


# =============================================================================
# Policy Processing Tests
# =============================================================================


class TestPadPolicyTargets:
    """Tests for pad_policy_targets function."""

    def test_pad_when_smaller(self, batch_size):
        """Test padding when targets are smaller than model policy size."""
        targets = torch.randn(batch_size, 64)
        padded = pad_policy_targets(targets, model_policy_size=128)
        assert padded.shape == (batch_size, 128)
        # Verify padding is zeros
        assert torch.all(padded[:, 64:] == 0)

    def test_no_pad_when_same_size(self, batch_size):
        """Test no padding when sizes match."""
        targets = torch.randn(batch_size, 64)
        padded = pad_policy_targets(targets, model_policy_size=64)
        assert padded.shape == (batch_size, 64)
        assert torch.equal(padded, targets)


class TestApplyLabelSmoothing:
    """Tests for apply_label_smoothing function."""

    def test_no_smoothing_when_epsilon_zero(self, batch_size, policy_size):
        """Test no change when epsilon is 0."""
        targets = torch.softmax(torch.randn(batch_size, policy_size), dim=1)
        valid_mask = torch.ones(batch_size, dtype=torch.bool)
        smoothed = apply_label_smoothing(targets, epsilon=0.0, valid_mask=valid_mask)
        assert torch.allclose(smoothed, targets)

    def test_smoothing_applies_correctly(self, batch_size, policy_size):
        """Test that label smoothing modifies targets."""
        targets = torch.zeros(batch_size, policy_size)
        targets[:, 0] = 1.0  # One-hot
        valid_mask = torch.ones(batch_size, dtype=torch.bool)
        smoothed = apply_label_smoothing(targets, epsilon=0.1, valid_mask=valid_mask)
        # Check that one-hot is no longer exactly 1.0
        assert smoothed[:, 0].max() < 1.0
        # Check sum is still 1.0
        assert torch.allclose(smoothed.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestValidatePolicyTargets:
    """Tests for validate_policy_targets function."""

    def test_valid_normalized_targets(self, batch_size, policy_size):
        """Test validation passes for normalized targets."""
        targets = torch.softmax(torch.randn(batch_size, policy_size), dim=1)
        valid_mask = torch.ones(batch_size, dtype=torch.bool)
        # Should not raise
        validate_policy_targets(targets, valid_mask, batch_idx=0)

    def test_severely_denormalized_raises(self, batch_size, policy_size):
        """Test validation raises for severely denormalized targets."""
        targets = torch.ones(batch_size, policy_size) * 0.1  # Sums to policy_size * 0.1
        valid_mask = torch.ones(batch_size, dtype=torch.bool)
        with pytest.raises(ValueError, match="severely denormalized"):
            validate_policy_targets(targets, valid_mask, batch_idx=0)


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestComputeValueLoss:
    """Tests for compute_value_loss function."""

    def test_scalar_value_loss(self, batch_size, device):
        """Test computing scalar value loss."""
        value_pred = torch.randn(batch_size, 1)
        value_targets = torch.rand(batch_size)
        criterion = nn.MSELoss()
        loss = compute_value_loss(
            value_pred, value_targets, num_players=None,
            use_multi_player=False, value_criterion=criterion, device=device
        )
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_multi_player_value_loss_fallback(self, batch_size, device):
        """Test multi-player loss falls back to MSE when module unavailable."""
        # Just ensure it doesn't crash
        value_pred = torch.randn(batch_size, 4)
        value_targets = torch.rand(batch_size, 4)
        criterion = nn.MSELoss()
        loss = compute_value_loss(
            value_pred, value_targets, num_players=4,
            use_multi_player=True, value_criterion=criterion, device=device
        )
        assert loss.ndim == 0


class TestComputePolicyLoss:
    """Tests for compute_policy_loss function."""

    def test_basic_policy_loss(self, batch_size, policy_size, device):
        """Test computing basic policy loss."""
        policy_pred = torch.randn(batch_size, policy_size)
        policy_targets = torch.softmax(torch.randn(batch_size, policy_size), dim=1)
        config = TrainStepConfig()
        value_targets = torch.rand(batch_size)

        loss, log_probs = compute_policy_loss(
            policy_pred, policy_targets, config, value_targets, device
        )
        assert loss.ndim == 0
        assert log_probs.shape == (batch_size, policy_size)


class TestComputeEntropyBonus:
    """Tests for compute_entropy_bonus function."""

    def test_no_bonus_when_weight_zero(self, batch_size, policy_size, device):
        """Test no entropy bonus when weight is 0."""
        log_probs = torch.log_softmax(torch.randn(batch_size, policy_size), dim=1)
        bonus = compute_entropy_bonus(log_probs, entropy_weight=0.0, device=device)
        assert bonus.item() == 0.0

    def test_bonus_applied_when_weight_positive(self, batch_size, policy_size, device):
        """Test entropy bonus applied when weight > 0."""
        log_probs = torch.log_softmax(torch.randn(batch_size, policy_size), dim=1)
        bonus = compute_entropy_bonus(log_probs, entropy_weight=0.01, device=device)
        assert bonus.item() != 0.0


# =============================================================================
# LossComponents Tests
# =============================================================================


class TestLossComponents:
    """Tests for LossComponents dataclass."""

    def test_total_with_basic_losses(self, device):
        """Test total computation with basic losses."""
        losses = LossComponents(
            value_loss=torch.tensor(0.5, device=device),
            policy_loss=torch.tensor(0.3, device=device),
        )
        assert torch.isclose(losses.total, torch.tensor(0.8, device=device))

    def test_total_with_all_losses(self, device):
        """Test total computation with all loss components."""
        losses = LossComponents(
            value_loss=torch.tensor(0.5, device=device),
            policy_loss=torch.tensor(0.3, device=device),
            rank_loss=torch.tensor(0.1, device=device),
            aux_loss=torch.tensor(0.05, device=device),
            entropy_bonus=torch.tensor(-0.02, device=device),
        )
        expected = 0.5 + 0.3 + 0.1 + 0.05 + (-0.02)
        assert torch.isclose(losses.total, torch.tensor(expected, device=device))

    def test_as_dict(self, device):
        """Test as_dict returns correct structure."""
        losses = LossComponents(
            value_loss=torch.tensor(0.5, device=device),
            policy_loss=torch.tensor(0.3, device=device),
            rank_loss=torch.tensor(0.1, device=device),
        )
        d = losses.as_dict()
        assert "value" in d
        assert "policy" in d
        assert "rank" in d


# =============================================================================
# Config and Context Tests
# =============================================================================


class TestTrainStepConfig:
    """Tests for TrainStepConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainStepConfig()
        assert config.policy_weight == 1.0
        assert config.gradient_accumulation_steps == 1
        assert config.use_mixed_precision is False
        assert config.model_version == "v2"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainStepConfig(
            policy_weight=0.5,
            use_mixed_precision=True,
            gradient_accumulation_steps=4,
        )
        assert config.policy_weight == 0.5
        assert config.use_mixed_precision is True
        assert config.gradient_accumulation_steps == 4


class TestTrainStepContext:
    """Tests for TrainStepContext dataclass."""

    def test_default_value_criterion(self, simple_model, optimizer, device):
        """Test that default value criterion is created."""
        config = TrainStepConfig()
        context = TrainStepContext(
            model=simple_model,
            optimizer=optimizer,
            device=device,
            config=config,
        )
        assert context.value_criterion is not None
        assert isinstance(context.value_criterion, nn.MSELoss)


class TestTrainStepResult:
    """Tests for TrainStepResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic result."""
        result = TrainStepResult(
            loss=0.5,
            value_loss=0.3,
            policy_loss=0.2,
        )
        assert result.loss == 0.5
        assert result.skipped is False
        assert result.error is None

    def test_skipped_result(self):
        """Test creating a skipped result."""
        result = TrainStepResult(
            loss=0.0,
            value_loss=0.0,
            policy_loss=0.0,
            skipped=True,
            error="Circuit breaker open",
        )
        assert result.skipped is True
        assert result.error is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestRunTrainingStep:
    """Integration tests for run_training_step function."""

    def test_basic_training_step(self, simple_batch, simple_model, optimizer, device):
        """Test a basic training step execution."""
        config = TrainStepConfig()
        context = TrainStepContext(
            model=simple_model,
            optimizer=optimizer,
            device=device,
            config=config,
        )

        result = run_training_step(
            context=context,
            raw_batch=simple_batch,
            batch_idx=0,
            is_streaming=False,
        )

        assert isinstance(result, TrainStepResult)
        assert not result.skipped
        assert result.loss > 0
        assert result.value_loss >= 0
        assert result.policy_loss >= 0

    def test_training_step_with_streaming(self, streaming_batch, simple_model, optimizer, device):
        """Test training step with streaming batch format."""
        config = TrainStepConfig()
        context = TrainStepContext(
            model=simple_model,
            optimizer=optimizer,
            device=device,
            config=config,
        )

        result = run_training_step(
            context=context,
            raw_batch=streaming_batch,
            batch_idx=0,
            is_streaming=True,
        )

        assert isinstance(result, TrainStepResult)
        assert not result.skipped
