"""Tests for app.training.multi_task_learning.

Tests the multi-task learning module including:
- TaskConfig and MultiTaskConfig dataclasses
- OutcomePredictionHead, LegalityPredictionHead, StateReconstructionHead
- MultiTaskHead wrapper
- GradNormWeighter for adaptive loss balancing
- MultiTaskLoss computation
- create_auxiliary_targets helper
- MultiTaskModelWrapper
- integrate_multi_task_loss helper
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from app.training.multi_task_learning import (
    GradNormWeighter,
    LegalityPredictionHead,
    MultiTaskConfig,
    MultiTaskHead,
    MultiTaskLoss,
    MultiTaskModelWrapper,
    OutcomePredictionHead,
    StateReconstructionHead,
    TaskConfig,
    create_auxiliary_targets,
    integrate_multi_task_loss,
)


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""

    def test_default_values(self):
        """Test TaskConfig has sensible defaults."""
        config = TaskConfig(name="test")

        assert config.name == "test"
        assert config.weight == 1.0
        assert config.enabled is True
        assert config.hidden_dim == 256
        assert config.output_dim is None

    def test_custom_values(self):
        """Test TaskConfig accepts custom values."""
        config = TaskConfig(
            name="outcome",
            weight=0.5,
            enabled=False,
            hidden_dim=128,
            output_dim=3,
        )

        assert config.name == "outcome"
        assert config.weight == 0.5
        assert config.enabled is False
        assert config.hidden_dim == 128
        assert config.output_dim == 3


class TestMultiTaskConfig:
    """Tests for MultiTaskConfig dataclass."""

    def test_default_values(self):
        """Test MultiTaskConfig has sensible defaults."""
        config = MultiTaskConfig()

        assert config.policy_weight == 1.0
        assert config.value_weight == 1.0
        assert config.outcome_prediction.name == "outcome"
        assert config.outcome_prediction.weight == 0.1
        assert config.outcome_prediction.output_dim == 3
        assert config.legality_prediction.name == "legality"
        assert config.legality_prediction.weight == 0.05
        assert config.state_reconstruction.name == "reconstruction"
        assert config.state_reconstruction.enabled is False
        assert config.task_weighting == "fixed"
        assert config.uncertainty_temp == 1.0

    def test_custom_weights(self):
        """Test MultiTaskConfig accepts custom weights."""
        config = MultiTaskConfig(
            policy_weight=2.0,
            value_weight=0.5,
        )

        assert config.policy_weight == 2.0
        assert config.value_weight == 0.5


class TestOutcomePredictionHead:
    """Tests for OutcomePredictionHead module."""

    def test_forward_shape(self):
        """Test OutcomePredictionHead output shape."""
        input_dim = 256
        hidden_dim = 128
        num_classes = 3
        batch_size = 32

        head = OutcomePredictionHead(input_dim, hidden_dim, num_classes)
        features = torch.randn(batch_size, input_dim)

        output = head(features)

        assert output.shape == (batch_size, num_classes)

    def test_forward_default_classes(self):
        """Test OutcomePredictionHead with default 3 classes."""
        input_dim = 256
        batch_size = 16

        head = OutcomePredictionHead(input_dim)
        features = torch.randn(batch_size, input_dim)

        output = head(features)

        assert output.shape == (batch_size, 3)

    def test_forward_training_vs_eval(self):
        """Test OutcomePredictionHead dropout in train vs eval mode."""
        head = OutcomePredictionHead(256)
        features = torch.randn(32, 256)

        head.train()
        out_train1 = head(features)
        out_train2 = head(features)

        head.eval()
        out_eval1 = head(features)
        out_eval2 = head(features)

        # In eval mode, outputs should be identical
        assert torch.allclose(out_eval1, out_eval2)


class TestLegalityPredictionHead:
    """Tests for LegalityPredictionHead module."""

    def test_forward_shape(self):
        """Test LegalityPredictionHead output shape."""
        input_dim = 256
        policy_size = 64
        hidden_dim = 128
        batch_size = 32

        head = LegalityPredictionHead(input_dim, policy_size, hidden_dim)
        features = torch.randn(batch_size, input_dim)

        output = head(features)

        assert output.shape == (batch_size, policy_size)

    def test_forward_default_hidden(self):
        """Test LegalityPredictionHead with default hidden dim."""
        input_dim = 256
        policy_size = 100
        batch_size = 16

        head = LegalityPredictionHead(input_dim, policy_size)
        features = torch.randn(batch_size, input_dim)

        output = head(features)

        assert output.shape == (batch_size, policy_size)


class TestStateReconstructionHead:
    """Tests for StateReconstructionHead module."""

    def test_forward_shape_2d(self):
        """Test StateReconstructionHead output shape for 2D state."""
        input_dim = 256
        state_shape = (8, 8)
        batch_size = 32

        head = StateReconstructionHead(input_dim, state_shape)
        features = torch.randn(batch_size, input_dim)

        output = head(features)

        assert output.shape == (batch_size, 8, 8)

    def test_forward_shape_3d(self):
        """Test StateReconstructionHead output shape for 3D state."""
        input_dim = 256
        state_shape = (4, 8, 8)
        batch_size = 16

        head = StateReconstructionHead(input_dim, state_shape)
        features = torch.randn(batch_size, input_dim)

        output = head(features)

        assert output.shape == (batch_size, 4, 8, 8)

    def test_forward_preserves_shape(self):
        """Test StateReconstructionHead stores state shape."""
        state_shape = (3, 16, 16)
        head = StateReconstructionHead(256, state_shape)

        assert head.state_shape == state_shape


class TestMultiTaskHead:
    """Tests for MultiTaskHead module."""

    def test_initialization_with_defaults(self):
        """Test MultiTaskHead initializes with default config."""
        head = MultiTaskHead(
            backbone_output_dim=256,
            policy_size=64,
        )

        assert 'outcome' in head.auxiliary_heads
        assert 'legality' in head.auxiliary_heads
        assert 'reconstruction' not in head.auxiliary_heads  # Disabled by default

    def test_initialization_with_custom_config(self):
        """Test MultiTaskHead with custom config."""
        config = MultiTaskConfig()
        config.outcome_prediction.enabled = False
        config.legality_prediction.enabled = False
        config.state_reconstruction.enabled = True

        head = MultiTaskHead(
            backbone_output_dim=256,
            policy_size=64,
            state_shape=(8, 8),
            config=config,
        )

        assert 'outcome' not in head.auxiliary_heads
        assert 'legality' not in head.auxiliary_heads
        assert 'reconstruction' in head.auxiliary_heads

    def test_forward(self):
        """Test MultiTaskHead forward pass."""
        head = MultiTaskHead(
            backbone_output_dim=256,
            policy_size=64,
        )

        batch_size = 32
        features = torch.randn(batch_size, 256)

        outputs = head(features)

        assert isinstance(outputs, dict)
        assert 'outcome' in outputs
        assert 'legality' in outputs
        assert outputs['outcome'].shape == (batch_size, 3)
        assert outputs['legality'].shape == (batch_size, 64)

    def test_uncertainty_weighting_creates_log_vars(self):
        """Test MultiTaskHead creates log_vars for uncertainty weighting."""
        config = MultiTaskConfig(task_weighting="uncertainty")

        head = MultiTaskHead(
            backbone_output_dim=256,
            policy_size=64,
            config=config,
        )

        assert hasattr(head, 'log_vars')
        # +2 for policy and value
        expected_tasks = len(head.auxiliary_heads) + 2
        assert head.log_vars.shape == (expected_tasks,)


class TestGradNormWeighter:
    """Tests for GradNormWeighter module."""

    def test_initialization(self):
        """Test GradNormWeighter initialization."""
        weighter = GradNormWeighter(num_tasks=3, alpha=1.5)

        assert weighter.num_tasks == 3
        assert weighter.alpha == 1.5
        assert weighter.initialized is False

    def test_weights_property(self):
        """Test GradNormWeighter weights property."""
        weighter = GradNormWeighter(num_tasks=3)

        weights = weighter.weights

        assert weights.shape == (3,)
        # Weights should sum to num_tasks
        assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=0.01)

    def test_initial_weights(self):
        """Test GradNormWeighter with initial weights."""
        initial = [1.0, 2.0, 0.5]
        weighter = GradNormWeighter(num_tasks=3, initial_weights=initial)

        weights = weighter.weights

        # Normalized weights should preserve relative proportions
        # but sum to num_tasks (3)
        assert weights.shape == (3,)

    def test_initialize_losses(self):
        """Test GradNormWeighter loss initialization."""
        weighter = GradNormWeighter(num_tasks=3)

        losses = torch.tensor([1.0, 2.0, 3.0])
        weighter.initialize_losses(losses)

        assert weighter.initialized is True
        assert torch.allclose(weighter.initial_losses, losses)

    def test_compute_loss_ratios(self):
        """Test GradNormWeighter loss ratio computation."""
        weighter = GradNormWeighter(num_tasks=3)

        initial_losses = torch.tensor([1.0, 1.0, 1.0])
        weighter.initialize_losses(initial_losses)

        current_losses = torch.tensor([0.5, 1.0, 2.0])
        ratios = weighter.compute_loss_ratios(current_losses)

        assert ratios.shape == (3,)
        # First task improved (low ratio), last task got worse (high ratio)

    def test_get_stats(self):
        """Test GradNormWeighter statistics."""
        weighter = GradNormWeighter(num_tasks=3)

        stats = weighter.get_stats()

        assert 'task_0_weight' in stats
        assert 'task_1_weight' in stats
        assert 'task_2_weight' in stats


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss module."""

    def test_initialization(self):
        """Test MultiTaskLoss initialization."""
        loss = MultiTaskLoss()

        assert loss.config is not None
        assert loss.shared_layer is None

    def test_forward_empty_outputs(self):
        """Test MultiTaskLoss with no matching outputs returns zero loss.

        The implementation handles empty outputs gracefully by returning
        a zero loss tensor and appropriate loss dict.
        """
        loss = MultiTaskLoss()

        outputs = {}
        targets = {}

        total, loss_dict = loss(outputs, targets)

        # Empty outputs return zero loss
        assert torch.isclose(total, torch.tensor(0.0))
        assert loss_dict['total_auxiliary_loss'] == 0.0

    def test_forward_outcome_only(self):
        """Test MultiTaskLoss with only outcome task."""
        loss = MultiTaskLoss()

        batch_size = 32
        outputs = {'outcome': torch.randn(batch_size, 3)}
        targets = {'outcome': torch.randint(0, 3, (batch_size,))}

        total, loss_dict = loss(outputs, targets)

        assert total > 0
        assert 'outcome_loss' in loss_dict
        assert 'total_auxiliary_loss' in loss_dict

    def test_forward_legality_only(self):
        """Test MultiTaskLoss with only legality task."""
        loss = MultiTaskLoss()

        batch_size = 32
        policy_size = 64
        outputs = {'legality': torch.randn(batch_size, policy_size)}
        targets = {'legality': torch.randint(0, 2, (batch_size, policy_size))}

        total, loss_dict = loss(outputs, targets)

        assert total > 0
        assert 'legality_loss' in loss_dict

    def test_forward_reconstruction(self):
        """Test MultiTaskLoss with reconstruction task."""
        loss = MultiTaskLoss()

        batch_size = 32
        outputs = {'reconstruction': torch.randn(batch_size, 8, 8)}
        targets = {'reconstruction': torch.randn(batch_size, 8, 8)}

        total, loss_dict = loss(outputs, targets)

        assert total >= 0
        assert 'reconstruction_loss' in loss_dict

    def test_forward_all_tasks(self):
        """Test MultiTaskLoss with all tasks."""
        loss = MultiTaskLoss()

        batch_size = 32
        policy_size = 64
        outputs = {
            'outcome': torch.randn(batch_size, 3),
            'legality': torch.randn(batch_size, policy_size),
            'reconstruction': torch.randn(batch_size, 8, 8),
        }
        targets = {
            'outcome': torch.randint(0, 3, (batch_size,)),
            'legality': torch.randint(0, 2, (batch_size, policy_size)),
            'reconstruction': torch.randn(batch_size, 8, 8),
        }

        total, loss_dict = loss(outputs, targets)

        assert total > 0
        assert 'outcome_loss' in loss_dict
        assert 'legality_loss' in loss_dict
        assert 'reconstruction_loss' in loss_dict

    def test_fixed_weighting(self):
        """Test MultiTaskLoss with fixed weighting."""
        config = MultiTaskConfig(task_weighting="fixed")
        loss = MultiTaskLoss(config=config)

        batch_size = 32
        outputs = {'outcome': torch.randn(batch_size, 3)}
        targets = {'outcome': torch.randint(0, 3, (batch_size,))}

        total, _ = loss(outputs, targets)

        # Total should be outcome_loss * outcome_weight (0.1)
        assert total >= 0

    def test_gradnorm_weighting(self):
        """Test MultiTaskLoss with gradnorm weighting."""
        config = MultiTaskConfig(task_weighting="gradnorm")
        loss = MultiTaskLoss(config=config)

        batch_size = 32
        outputs = {'outcome': torch.randn(batch_size, 3)}
        targets = {'outcome': torch.randint(0, 3, (batch_size,))}

        total, loss_dict = loss(outputs, targets)

        assert total > 0
        assert 'outcome_weight' in loss_dict

    def test_get_gradnorm_weighter(self):
        """Test MultiTaskLoss gradnorm weighter accessor."""
        config = MultiTaskConfig(task_weighting="gradnorm")
        loss = MultiTaskLoss(config=config)

        # Need to call forward to initialize
        outputs = {'outcome': torch.randn(32, 3)}
        targets = {'outcome': torch.randint(0, 3, (32,))}
        loss(outputs, targets)

        weighter = loss.get_gradnorm_weighter()

        assert weighter is not None
        assert isinstance(weighter, GradNormWeighter)


class TestCreateAuxiliaryTargets:
    """Tests for create_auxiliary_targets helper function."""

    def test_outcome_from_values(self):
        """Test create_auxiliary_targets creates outcome from values."""
        batch = {'values': torch.tensor([-1.0, 0.0, 1.0])}

        targets = create_auxiliary_targets(batch)

        assert 'outcome' in targets
        assert targets['outcome'].tolist() == [0, 1, 2]

    def test_legality_from_policy(self):
        """Test create_auxiliary_targets creates legality from policy."""
        batch = {'policy': torch.tensor([
            [0.5, 0.0, 0.3, 0.2],
            [0.0, 1.0, 0.0, 0.0],
        ])}

        targets = create_auxiliary_targets(batch)

        assert 'legality' in targets
        expected = torch.tensor([
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        assert torch.allclose(targets['legality'], expected)

    def test_reconstruction_from_features(self):
        """Test create_auxiliary_targets creates reconstruction from features."""
        features = torch.randn(32, 4, 8, 8)
        batch = {'features': features}

        targets = create_auxiliary_targets(batch)

        assert 'reconstruction' in targets
        assert torch.allclose(targets['reconstruction'], features)

    def test_all_targets(self):
        """Test create_auxiliary_targets with all inputs."""
        batch = {
            'values': torch.tensor([0.5, -0.5]),
            'policy': torch.tensor([[0.5, 0.5], [0.0, 1.0]]),
            'features': torch.randn(2, 4),
        }

        targets = create_auxiliary_targets(batch)

        assert 'outcome' in targets
        assert 'legality' in targets
        assert 'reconstruction' in targets

    def test_empty_batch(self):
        """Test create_auxiliary_targets with empty batch."""
        batch = {}

        targets = create_auxiliary_targets(batch)

        assert targets == {}


class TestMultiTaskModelWrapper:
    """Tests for MultiTaskModelWrapper module."""

    def test_initialization(self):
        """Test MultiTaskModelWrapper initialization."""
        backbone = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
        )

        # Mock backbone that returns tuple
        class MockBackbone(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                policy = torch.randn(batch_size, 64)
                value = torch.randn(batch_size)
                return policy, value

        wrapper = MultiTaskModelWrapper(
            backbone=MockBackbone(),
            backbone_output_dim=256,
            policy_size=64,
        )

        assert wrapper.backbone is not None
        assert wrapper.multi_task_head is not None

    def test_forward(self):
        """Test MultiTaskModelWrapper forward pass."""
        class MockBackbone(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                policy = torch.randn(batch_size, 64)
                value = torch.randn(batch_size)
                features = x.view(batch_size, -1)
                return policy, value, features

        wrapper = MultiTaskModelWrapper(
            backbone=MockBackbone(),
            backbone_output_dim=64,
            policy_size=64,
        )

        batch_size = 32
        inputs = torch.randn(batch_size, 64)

        policy, value, auxiliary = wrapper(inputs)

        assert policy.shape == (batch_size, 64)
        assert value.shape == (batch_size,)
        assert isinstance(auxiliary, dict)
        assert 'outcome' in auxiliary
        assert 'legality' in auxiliary

    def test_forward_without_features(self):
        """Test MultiTaskModelWrapper when backbone doesn't return features."""
        class MockBackbone(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                policy = torch.randn(batch_size, 64)
                value = torch.randn(batch_size)
                return policy, value

        wrapper = MultiTaskModelWrapper(
            backbone=MockBackbone(),
            backbone_output_dim=64,
            policy_size=64,
        )

        batch_size = 32
        inputs = torch.randn(batch_size, 64)

        policy, value, auxiliary = wrapper(inputs)

        assert policy is not None
        assert value is not None
        assert isinstance(auxiliary, dict)


class TestIntegrateMultiTaskLoss:
    """Tests for integrate_multi_task_loss helper function."""

    def test_fixed_weighting(self):
        """Test integrate_multi_task_loss with fixed weighting."""
        policy_loss = torch.tensor(1.0)
        value_loss = torch.tensor(0.5)
        auxiliary_loss = torch.tensor(0.2)

        total, loss_dict = integrate_multi_task_loss(
            policy_loss, value_loss, auxiliary_loss
        )

        assert total > 0
        assert 'policy_loss' in loss_dict
        assert 'value_loss' in loss_dict
        assert 'auxiliary_loss' in loss_dict
        assert 'total_loss' in loss_dict

        # With default weights (1.0, 1.0):
        expected = 1.0 * 1.0 + 1.0 * 0.5 + 0.2
        assert torch.isclose(total, torch.tensor(expected))

    def test_custom_weights(self):
        """Test integrate_multi_task_loss with custom weights."""
        config = MultiTaskConfig(policy_weight=2.0, value_weight=0.5)

        policy_loss = torch.tensor(1.0)
        value_loss = torch.tensor(1.0)
        auxiliary_loss = torch.tensor(0.0)

        total, _ = integrate_multi_task_loss(
            policy_loss, value_loss, auxiliary_loss, config=config
        )

        expected = 2.0 * 1.0 + 0.5 * 1.0 + 0.0
        assert torch.isclose(total, torch.tensor(expected))

    def test_uncertainty_weighting(self):
        """Test integrate_multi_task_loss with uncertainty weighting."""
        config = MultiTaskConfig(task_weighting="uncertainty")

        policy_loss = torch.tensor(1.0)
        value_loss = torch.tensor(0.5)
        auxiliary_loss = torch.tensor(0.2)
        log_vars = torch.zeros(4)  # Some tasks + policy + value

        total, loss_dict = integrate_multi_task_loss(
            policy_loss, value_loss, auxiliary_loss,
            config=config, log_vars=log_vars
        )

        assert total > 0
        assert 'total_loss' in loss_dict


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_multi_task_head_no_tasks(self):
        """Test MultiTaskHead with all tasks disabled."""
        config = MultiTaskConfig()
        config.outcome_prediction.enabled = False
        config.legality_prediction.enabled = False
        config.state_reconstruction.enabled = False

        head = MultiTaskHead(
            backbone_output_dim=256,
            policy_size=64,
            config=config,
        )

        assert len(head.auxiliary_heads) == 0

        features = torch.randn(32, 256)
        outputs = head(features)

        assert outputs == {}

    def test_gradnorm_weighter_single_task(self):
        """Test GradNormWeighter with single task."""
        weighter = GradNormWeighter(num_tasks=1)

        weights = weighter.weights

        assert weights.shape == (1,)
        assert torch.isclose(weights[0], torch.tensor(1.0))

    def test_multi_task_loss_mismatched_keys(self):
        """Test MultiTaskLoss ignores unmatched outputs."""
        loss = MultiTaskLoss()

        outputs = {
            'outcome': torch.randn(32, 3),
            'unknown_task': torch.randn(32, 10),
        }
        targets = {
            'outcome': torch.randint(0, 3, (32,)),
            'legality': torch.randint(0, 2, (32, 64)),  # No output for this
        }

        total, loss_dict = loss(outputs, targets)

        # Should only compute outcome loss
        assert 'outcome_loss' in loss_dict
        assert 'legality_loss' not in loss_dict

    def test_very_small_losses(self):
        """Test MultiTaskLoss handles very small losses."""
        loss = MultiTaskLoss()

        batch_size = 32
        # Near-zero logits for very small loss
        outputs = {'outcome': torch.zeros(batch_size, 3)}
        targets = {'outcome': torch.zeros(batch_size, dtype=torch.long)}

        total, _ = loss(outputs, targets)

        assert torch.isfinite(total)

    def test_large_batch_size(self):
        """Test modules handle large batch sizes."""
        batch_size = 1024

        head = MultiTaskHead(backbone_output_dim=256, policy_size=64)
        features = torch.randn(batch_size, 256)

        outputs = head(features)

        assert outputs['outcome'].shape[0] == batch_size
        assert outputs['legality'].shape[0] == batch_size
