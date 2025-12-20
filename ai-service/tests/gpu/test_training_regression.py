"""GPU Training Regression Tests.

Per Section 3.3.3 of the action plan, these tests verify:
- GPU training produces consistent results with fixed data
- Loss decreases during training (learning is occurring)
- Gradient norms are in expected ranges (no exploding/vanishing)
- GPU and CPU training produce similar final loss

These tests use @pytest.mark.gpu and are designed to catch regressions
in the training pipeline that could silently break GPU training.

Usage:
    # Run GPU tests only
    pytest tests/gpu/test_training_regression.py -v -m gpu

    # Run in CI (skips if no GPU)
    pytest tests/gpu/test_training_regression.py -v
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

# Optional imports for GPU
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check CUDA availability
CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

# Custom GPU marker
gpu = pytest.mark.gpu
skipif_no_gpu = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA not available"
)
skipif_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available"
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fixed_seed():
    """Set deterministic seed for reproducibility."""
    seed = 42
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if CUDA_AVAILABLE:
            torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def synthetic_training_data(fixed_seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate fixed synthetic training data for regression tests.

    Returns deterministic data with known properties:
    - Features: 8x8x10 board representation (100 samples)
    - Values: Game outcomes in [-1, 1]
    - Policies: Probability distributions over 64 moves
    """
    np.random.seed(fixed_seed)
    n_samples = 100
    board_size = 8
    n_channels = 10
    policy_size = 64

    # Generate consistent random data
    features = np.random.randn(n_samples, n_channels, board_size, board_size).astype(np.float32)
    values = np.tanh(np.random.randn(n_samples)).astype(np.float32)  # Values in [-1, 1]
    policies = np.random.dirichlet(np.ones(policy_size), n_samples).astype(np.float32)

    return features, values, policies


@pytest.fixture
def training_npz_file(synthetic_training_data, tmp_path) -> Path:
    """Create a fixed NPZ file for training regression tests."""
    features, values, policies = synthetic_training_data
    npz_path = tmp_path / "regression_test_data.npz"

    np.savez_compressed(
        npz_path,
        features=features,
        values=values,
        policy_values=policies,
    )

    return npz_path


@pytest.fixture
def simple_model():
    """Create a simple CNN model for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    class SimpleCNN(nn.Module):
        """Minimal CNN for regression testing."""

        def __init__(self, in_channels=10, board_size=8, policy_size=64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.bn = nn.BatchNorm2d(32)
            self.relu = nn.ReLU()

            # Value head
            self.value_conv = nn.Conv2d(32, 1, 1)
            self.value_fc = nn.Linear(board_size * board_size, 64)
            self.value_out = nn.Linear(64, 1)

            # Policy head
            self.policy_conv = nn.Conv2d(32, 2, 1)
            self.policy_fc = nn.Linear(2 * board_size * board_size, policy_size)

        def forward(self, x):
            # Shared layers
            x = self.relu(self.conv1(x))
            x = self.relu(self.bn(self.conv2(x)))

            # Value head
            v = self.value_conv(x)
            v = v.view(v.size(0), -1)
            v = self.relu(self.value_fc(v))
            v = torch.tanh(self.value_out(v))

            # Policy head
            p = self.policy_conv(x)
            p = p.view(p.size(0), -1)
            p = self.policy_fc(p)
            p = torch.softmax(p, dim=1)

            return p, v.squeeze(-1)

    return SimpleCNN()


# =============================================================================
# GPU Training Regression Tests
# =============================================================================


@skipif_no_torch
class TestTrainingDataConsistency:
    """Tests for training data consistency."""

    def test_synthetic_data_deterministic(self, fixed_seed):
        """Verify synthetic data is deterministic with fixed seed."""
        np.random.seed(fixed_seed)
        data1 = np.random.randn(10, 10).astype(np.float32)

        np.random.seed(fixed_seed)
        data2 = np.random.randn(10, 10).astype(np.float32)

        np.testing.assert_array_equal(data1, data2)

    def test_synthetic_data_valid_ranges(self, synthetic_training_data):
        """Verify synthetic data is within valid ranges."""
        features, values, policies = synthetic_training_data

        # Features should be reasonable (no NaN/Inf)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

        # Values in [-1, 1]
        assert np.all(values >= -1.0) and np.all(values <= 1.0)

        # Policies sum to 1
        policy_sums = np.sum(policies, axis=1)
        np.testing.assert_array_almost_equal(policy_sums, np.ones(len(policies)), decimal=5)

    def test_npz_file_loadable(self, training_npz_file):
        """Verify NPZ file can be loaded correctly."""
        data = np.load(training_npz_file)
        assert 'features' in data
        assert 'values' in data
        assert 'policy_values' in data
        assert len(data['features']) == len(data['values']) == len(data['policy_values'])


@skipif_no_gpu
@gpu
class TestGPUTrainingBasics:
    """Basic GPU training tests."""

    def test_model_moves_to_gpu(self, simple_model):
        """Model should successfully move to GPU."""
        model = simple_model.cuda()
        assert next(model.parameters()).is_cuda

    def test_forward_pass_on_gpu(self, simple_model, synthetic_training_data):
        """Forward pass should work on GPU."""
        features, _, _ = synthetic_training_data
        model = simple_model.cuda()

        x = torch.from_numpy(features[:10]).cuda()
        policy, value = model(x)

        assert policy.is_cuda
        assert value.is_cuda
        assert policy.shape == (10, 64)
        assert value.shape == (10,)

    def test_backward_pass_on_gpu(self, simple_model, synthetic_training_data):
        """Backward pass should compute gradients on GPU."""
        features, values, policies = synthetic_training_data
        model = simple_model.cuda()

        x = torch.from_numpy(features[:10]).cuda()
        y_value = torch.from_numpy(values[:10]).cuda()
        y_policy = torch.from_numpy(policies[:10]).cuda()

        policy_pred, value_pred = model(x)

        # Combined loss
        value_loss = nn.MSELoss()(value_pred, y_value)
        policy_loss = -torch.mean(torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1))
        loss = value_loss + policy_loss

        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert param.grad.is_cuda


@skipif_no_gpu
@gpu
class TestGPUTrainingRegression:
    """Core GPU training regression tests."""

    def test_loss_decreases_during_training(self, simple_model, synthetic_training_data):
        """Training should decrease loss over epochs (learning is occurring)."""
        features, values, policies = synthetic_training_data
        model = simple_model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(values),
            torch.from_numpy(policies),
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        losses = []
        for epoch in range(5):
            epoch_loss = 0.0
            for batch_x, batch_v, batch_p in loader:
                batch_x = batch_x.cuda()
                batch_v = batch_v.cuda()
                batch_p = batch_p.cuda()

                optimizer.zero_grad()
                policy_pred, value_pred = model(batch_x)

                value_loss = nn.MSELoss()(value_pred, batch_v)
                policy_loss = -torch.mean(torch.sum(batch_p * torch.log(policy_pred + 1e-8), dim=1))
                loss = value_loss + policy_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / len(loader))

        # Loss should decrease (with some tolerance for noise)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    def test_gradient_norms_in_range(self, simple_model, synthetic_training_data):
        """Gradient norms should be in expected range (no explosion/vanishing)."""
        features, values, policies = synthetic_training_data
        model = simple_model.cuda()

        x = torch.from_numpy(features[:32]).cuda()
        y_value = torch.from_numpy(values[:32]).cuda()
        y_policy = torch.from_numpy(policies[:32]).cuda()

        policy_pred, value_pred = model(x)
        value_loss = nn.MSELoss()(value_pred, y_value)
        policy_loss = -torch.mean(torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1))
        loss = value_loss + policy_loss
        loss.backward()

        # Compute gradient norms
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Gradient norm should be reasonable (not exploding or vanishing)
        assert 0.001 < total_norm < 100, (
            f"Gradient norm out of range: {total_norm:.6f}"
        )

    def test_training_deterministic_with_seed(self, simple_model, synthetic_training_data):
        """Training should be deterministic with same seed."""
        features, values, policies = synthetic_training_data

        def train_one_epoch(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = simple_model.__class__().cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            x = torch.from_numpy(features[:16]).cuda()
            y_value = torch.from_numpy(values[:16]).cuda()
            y_policy = torch.from_numpy(policies[:16]).cuda()

            optimizer.zero_grad()
            policy_pred, value_pred = model(x)
            value_loss = nn.MSELoss()(value_pred, y_value)
            policy_loss = -torch.mean(torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1))
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            return loss.item()

        loss1 = train_one_epoch(123)
        loss2 = train_one_epoch(123)

        assert abs(loss1 - loss2) < 1e-5, (
            f"Training not deterministic: {loss1:.6f} vs {loss2:.6f}"
        )

    def test_model_checkpoint_saves_and_loads(self, simple_model, synthetic_training_data, tmp_path):
        """Model checkpoint should save and load correctly."""
        features, _, _ = synthetic_training_data
        model = simple_model.cuda()

        # Get initial output
        x = torch.from_numpy(features[:4]).cuda()
        with torch.no_grad():
            policy1, value1 = model(x)

        # Save checkpoint
        checkpoint_path = tmp_path / "model_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Create new model and load
        model2 = simple_model.__class__().cuda()
        model2.load_state_dict(torch.load(checkpoint_path))

        # Outputs should match
        with torch.no_grad():
            policy2, value2 = model2(x)

        torch.testing.assert_close(policy1, policy2)
        torch.testing.assert_close(value1, value2)


@skipif_no_gpu
@gpu
class TestGPUvsCPUParity:
    """Tests that GPU and CPU training produce similar results."""

    def test_forward_pass_parity(self, simple_model, synthetic_training_data):
        """GPU and CPU forward passes should produce same results."""
        features, _, _ = synthetic_training_data
        x = torch.from_numpy(features[:4])

        # CPU forward
        model_cpu = simple_model
        model_cpu.eval()
        with torch.no_grad():
            policy_cpu, value_cpu = model_cpu(x)

        # GPU forward (same weights)
        model_gpu = simple_model.__class__()
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu = model_gpu.cuda().eval()
        with torch.no_grad():
            policy_gpu, value_gpu = model_gpu(x.cuda())

        # Compare (allow small numerical tolerance)
        torch.testing.assert_close(
            policy_cpu, policy_gpu.cpu(), rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            value_cpu, value_gpu.cpu(), rtol=1e-4, atol=1e-5
        )

    def test_training_loss_parity(self, simple_model, synthetic_training_data):
        """GPU and CPU should compute same loss for same input."""
        features, values, policies = synthetic_training_data

        x = torch.from_numpy(features[:16])
        y_value = torch.from_numpy(values[:16])
        y_policy = torch.from_numpy(policies[:16])

        def compute_loss(model, x, y_value, y_policy):
            policy_pred, value_pred = model(x)
            value_loss = nn.MSELoss()(value_pred, y_value)
            policy_loss = -torch.mean(torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1))
            return (value_loss + policy_loss).item()

        # CPU loss
        model_cpu = simple_model
        model_cpu.eval()
        loss_cpu = compute_loss(model_cpu, x, y_value, y_policy)

        # GPU loss (same weights)
        model_gpu = simple_model.__class__()
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu = model_gpu.cuda().eval()
        loss_gpu = compute_loss(model_gpu, x.cuda(), y_value.cuda(), y_policy.cuda())

        # Losses should be very close
        assert abs(loss_cpu - loss_gpu) < 1e-3, (
            f"Loss mismatch: CPU={loss_cpu:.6f}, GPU={loss_gpu:.6f}"
        )

    def test_training_convergence_parity(self, simple_model, synthetic_training_data, fixed_seed):
        """GPU and CPU training should converge to similar loss after multiple epochs.

        This is the A/B training comparison test that verifies model quality
        is equivalent between GPU and CPU training pipelines.
        """
        features, values, policies = synthetic_training_data
        n_epochs = 10
        batch_size = 16
        lr = 0.01

        def train_model(model, device, seed):
            """Train a model for n_epochs and return final loss."""
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed_all(seed)

            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            x = torch.from_numpy(features[:batch_size]).to(device)
            y_value = torch.from_numpy(values[:batch_size]).to(device)
            y_policy = torch.from_numpy(policies[:batch_size]).to(device)

            losses = []
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                policy_pred, value_pred = model(x)
                value_loss = nn.MSELoss()(value_pred, y_value)
                policy_loss = -torch.mean(
                    torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1)
                )
                loss = value_loss + policy_loss
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            return losses[-1], losses

        # Train on CPU
        model_cpu = simple_model.__class__()
        model_cpu.load_state_dict(simple_model.state_dict())
        final_loss_cpu, losses_cpu = train_model(model_cpu, 'cpu', fixed_seed)

        # Train on GPU with same initial weights and seed
        model_gpu = simple_model.__class__()
        model_gpu.load_state_dict(simple_model.state_dict())
        final_loss_gpu, losses_gpu = train_model(model_gpu, 'cuda', fixed_seed)

        # Final losses should be very close (within 5%)
        loss_diff_pct = abs(final_loss_cpu - final_loss_gpu) / max(final_loss_cpu, 1e-6) * 100
        assert loss_diff_pct < 5.0, (
            f"Training convergence mismatch: CPU={final_loss_cpu:.6f}, GPU={final_loss_gpu:.6f} "
            f"({loss_diff_pct:.1f}% difference)"
        )

        # Both should show learning (loss decreased)
        assert losses_cpu[-1] < losses_cpu[0], "CPU model did not learn"
        assert losses_gpu[-1] < losses_gpu[0], "GPU model did not learn"

        # Learning curves should be similar (correlation > 0.9)
        correlation = np.corrcoef(losses_cpu, losses_gpu)[0, 1]
        assert correlation > 0.9, (
            f"Training curves diverged: correlation={correlation:.3f}"
        )


@skipif_no_gpu
@gpu
class TestGPUMemoryBehavior:
    """Tests for GPU memory behavior during training."""

    def test_no_memory_leak_in_training_loop(self, simple_model, synthetic_training_data):
        """Training loop should not leak GPU memory."""
        features, values, policies = synthetic_training_data
        model = simple_model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.from_numpy(features[:32]).cuda()
        y_value = torch.from_numpy(values[:32]).cuda()
        y_policy = torch.from_numpy(policies[:32]).cuda()

        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            policy_pred, value_pred = model(x)
            loss = nn.MSELoss()(value_pred, y_value)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Training iterations
        for _ in range(20):
            optimizer.zero_grad()
            policy_pred, value_pred = model(x)
            value_loss = nn.MSELoss()(value_pred, y_value)
            policy_loss = -torch.mean(torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1))
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly (allow 10% tolerance)
        memory_growth = (final_memory - initial_memory) / max(initial_memory, 1)
        assert memory_growth < 0.1, (
            f"Memory grew by {memory_growth*100:.1f}%: "
            f"initial={initial_memory/1e6:.1f}MB, final={final_memory/1e6:.1f}MB"
        )

    def test_batch_size_scaling(self, simple_model, synthetic_training_data):
        """Larger batch sizes should use more memory proportionally."""
        features, values, policies = synthetic_training_data
        model = simple_model.cuda()

        def get_memory_for_batch(batch_size):
            torch.cuda.empty_cache()
            x = torch.from_numpy(features[:batch_size]).cuda()
            y_value = torch.from_numpy(values[:batch_size]).cuda()

            policy_pred, value_pred = model(x)
            loss = nn.MSELoss()(value_pred, y_value)
            loss.backward()

            torch.cuda.synchronize()
            return torch.cuda.memory_allocated()

        mem_small = get_memory_for_batch(8)
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)

        mem_large = get_memory_for_batch(32)

        # Larger batch should use more memory
        assert mem_large > mem_small, (
            f"Larger batch should use more memory: "
            f"small={mem_small/1e6:.1f}MB, large={mem_large/1e6:.1f}MB"
        )


# =============================================================================
# Integration with Real Training Components
# =============================================================================


@skipif_no_gpu
@gpu
class TestRealTrainingComponents:
    """Tests with actual RingRift training components."""

    def test_ringrift_model_trains_on_gpu(self, synthetic_training_data):
        """RingRift CNN should train on GPU."""
        try:
            from app.ai.neural_net import RingRiftCNN_v2
        except ImportError:
            pytest.skip("RingRiftCNN_v2 not available")

        features, values, policies = synthetic_training_data

        # Create model with correct dimensions
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=10,
            policy_size=64,
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.from_numpy(features[:16]).cuda()
        y_value = torch.from_numpy(values[:16]).cuda()
        y_policy = torch.from_numpy(policies[:16]).cuda()

        # Training step
        optimizer.zero_grad()
        policy_pred, value_pred = model(x)

        # Handle potential extra dimensions
        if value_pred.dim() > 1:
            value_pred = value_pred.squeeze(-1)

        value_loss = nn.MSELoss()(value_pred, y_value)
        policy_loss = -torch.mean(torch.sum(y_policy * torch.log(policy_pred + 1e-8), dim=1))
        loss = value_loss + policy_loss

        loss.backward()
        optimizer.step()

        # Should complete without error
        assert loss.item() > 0

    def test_data_validation_before_gpu_training(self, training_npz_file):
        """Data validation should work before GPU training."""
        try:
            from app.training.data_validation import validate_npz_file
        except ImportError:
            pytest.skip("data_validation not available")

        result = validate_npz_file(str(training_npz_file))
        assert result.valid, f"Validation failed: {result.summary()}"
