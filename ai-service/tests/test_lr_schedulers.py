"""
Tests for learning rate schedulers in the training module.

Tests cover:
- CosineAnnealingLR creates correct LR curve
- CosineAnnealingWarmRestarts with different T_0 and T_mult values
- Combination with warmup (warmup → cosine)
- Scheduler state save/restore
- Backward compatibility (no scheduler specified)
"""

import pytest
import torch.nn as nn
import torch.optim as optim
import os
import tempfile
import math
from typing import List

from app.training.schedulers import (
    create_lr_scheduler,
    get_warmup_scheduler,
)
from app.training.checkpointing import (
    save_checkpoint,
    load_checkpoint,
)
from app.training.train_cli import parse_args


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


def get_lr(optimizer: optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def simulate_epochs(
    optimizer: optim.Optimizer,
    scheduler,
    epochs: int
) -> list[float]:
    """Simulate training epochs and return list of learning rates."""
    lrs = []
    for _ in range(epochs):
        lrs.append(get_lr(optimizer))
        if scheduler is not None:
            scheduler.step()
    return lrs


class TestCosineAnnealingLR:
    """Tests for CosineAnnealingLR scheduler."""

    def test_cosine_lr_decays_to_min(self):
        """Test that cosine scheduler decays from initial to min LR."""
        model = DummyModel()
        initial_lr = 0.001
        lr_min = 1e-6
        total_epochs = 100

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine',
            total_epochs=total_epochs,
            warmup_epochs=0,
            lr_min=lr_min,
        )

        lrs = simulate_epochs(optimizer, scheduler, total_epochs)

        # Should start near initial LR
        assert lrs[0] == pytest.approx(initial_lr, rel=0.01)

        # Should end near minimum LR (use larger tolerance due to cosine curve)
        # The cosine curve approaches but doesn't exactly hit the minimum
        assert lrs[-1] == pytest.approx(lr_min, rel=0.5)
        assert lrs[-1] < initial_lr * 0.01  # Should be much smaller than start

        # LR should decrease monotonically (cosine curve)
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-10  # Allow small numerical error

    def test_cosine_lr_curve_shape(self):
        """Test that cosine scheduler follows half-cosine curve."""
        model = DummyModel()
        initial_lr = 1.0
        lr_min = 0.0
        total_epochs = 100

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine',
            total_epochs=total_epochs,
            warmup_epochs=0,
            lr_min=lr_min,
        )

        lrs = simulate_epochs(optimizer, scheduler, total_epochs)

        # Check midpoint - should be ~0.5 for standard cosine
        midpoint = total_epochs // 2
        cos_val = math.cos(math.pi * midpoint / total_epochs)
        expected_mid = 0.5 * (1.0 + cos_val)
        assert lrs[midpoint] == pytest.approx(expected_mid, rel=0.05)


class TestCosineAnnealingWarmRestarts:
    """Tests for CosineAnnealingWarmRestarts scheduler."""

    def test_warm_restarts_basic(self):
        """Test basic warm restarts behavior."""
        model = DummyModel()
        initial_lr = 0.001
        lr_min = 1e-6
        t0 = 10
        t_mult = 1  # Use 1 for simpler testing

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine-warm-restarts',
            total_epochs=50,
            warmup_epochs=0,
            lr_min=lr_min,
            lr_t0=t0,
            lr_t_mult=t_mult,
        )

        lrs = simulate_epochs(optimizer, scheduler, 25)

        # At epoch 0, should be at initial LR
        assert lrs[0] == pytest.approx(initial_lr, rel=0.01)

        # At epoch t0 (restart), should be back near initial LR
        # Note: lrs[t0] is the LR at the START of epoch t0
        assert lrs[t0] == pytest.approx(initial_lr, rel=0.01)

        # At epoch 2*t0 (second restart), should be back near initial LR
        assert lrs[2 * t0] == pytest.approx(initial_lr, rel=0.01)

    def test_warm_restarts_with_t_mult(self):
        """Test warm restarts with T_mult > 1."""
        model = DummyModel()
        initial_lr = 0.001
        lr_min = 1e-6
        t0 = 5
        t_mult = 2

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine-warm-restarts',
            total_epochs=100,
            warmup_epochs=0,
            lr_min=lr_min,
            lr_t0=t0,
            lr_t_mult=t_mult,
        )

        lrs = simulate_epochs(optimizer, scheduler, 20)

        # First restart at epoch 5 (T_0)
        assert lrs[t0] == pytest.approx(initial_lr, rel=0.01)

        # Second restart at epoch 15 (T_0 + T_0*T_mult = 5 + 10)
        assert lrs[15] == pytest.approx(initial_lr, rel=0.01)

    def test_warm_restarts_different_t0_values(self):
        """Test warm restarts with different T_0 values."""
        model = DummyModel()
        initial_lr = 0.001

        for t0 in [5, 10, 20]:
            optimizer = optim.Adam(model.parameters(), lr=initial_lr)
            scheduler = create_lr_scheduler(
                optimizer,
                scheduler_type='cosine-warm-restarts',
                total_epochs=100,
                warmup_epochs=0,
                lr_min=1e-6,
                lr_t0=t0,
                lr_t_mult=1,
            )

            lrs = simulate_epochs(optimizer, scheduler, t0 * 2 + 1)

            # Should restart at each t0 boundary
            assert lrs[t0] == pytest.approx(initial_lr, rel=0.01), \
                f"Failed for t0={t0} at epoch {t0}"


class TestWarmupWithCosine:
    """Tests for warmup combined with cosine scheduling."""

    def test_warmup_then_cosine(self):
        """Test warmup followed by cosine annealing."""
        model = DummyModel()
        initial_lr = 0.001
        warmup_epochs = 5
        total_epochs = 50

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine',
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            lr_min=1e-6,
        )

        lrs = simulate_epochs(optimizer, scheduler, total_epochs)

        # During warmup, LR should increase
        for i in range(1, warmup_epochs):
            assert lrs[i] >= lrs[i - 1], \
                f"LR should increase during warmup at epoch {i}"

        # At end of warmup, should be at/near initial LR
        assert lrs[warmup_epochs] == pytest.approx(initial_lr, rel=0.15)

        # After warmup, LR should decrease (cosine decay)
        for i in range(warmup_epochs + 1, total_epochs):
            # Allow small tolerance for numerical precision
            assert lrs[i] <= lrs[i - 1] + 1e-10, \
                f"LR should decrease after warmup at epoch {i}"

    def test_warmup_then_cosine_warm_restarts(self):
        """Test warmup followed by cosine with warm restarts."""
        model = DummyModel()
        initial_lr = 0.001
        warmup_epochs = 5
        t0 = 10

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine-warm-restarts',
            total_epochs=100,
            warmup_epochs=warmup_epochs,
            lr_min=1e-6,
            lr_t0=t0,
            lr_t_mult=1,
        )

        lrs = simulate_epochs(optimizer, scheduler, warmup_epochs + t0 + 5)

        # During warmup, LR should increase
        for i in range(1, warmup_epochs):
            assert lrs[i] >= lrs[i - 1]

        # At epoch warmup_epochs + t0, should restart (back to initial)
        # Note: After warmup, the cosine restarts scheduler takes over
        restart_epoch = warmup_epochs + t0
        assert lrs[restart_epoch] == pytest.approx(initial_lr, rel=0.15)


class TestSchedulerStateSaveRestore:
    """Tests for saving and restoring scheduler state."""

    def test_scheduler_state_save_restore(self):
        """Test that scheduler state is correctly saved and restored."""
        model = DummyModel()
        initial_lr = 0.001
        total_epochs = 50

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine',
            total_epochs=total_epochs,
            warmup_epochs=0,
            lr_min=1e-6,
        )

        assert scheduler is not None

        # Simulate 20 epochs
        for _ in range(20):
            scheduler.step()

        lr_at_epoch_20 = get_lr(optimizer)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
            save_checkpoint(
                model, optimizer, epoch=19, loss=0.5,
                path=checkpoint_path, scheduler=scheduler
            )

            # Create new model, optimizer, scheduler
            new_model = DummyModel()
            new_optimizer = optim.Adam(new_model.parameters(), lr=initial_lr)
            new_scheduler = create_lr_scheduler(
                new_optimizer,
                scheduler_type='cosine',
                total_epochs=total_epochs,
                warmup_epochs=0,
                lr_min=1e-6,
            )

            # Load checkpoint
            epoch, loss = load_checkpoint(
                checkpoint_path,
                new_model, new_optimizer,
                scheduler=new_scheduler
            )

            assert epoch == 19

            # LR should match what it was at epoch 20
            assert get_lr(new_optimizer) == pytest.approx(
                lr_at_epoch_20, rel=0.001
            )

    def test_warm_restarts_state_restore(self):
        """Test that warm restarts scheduler state is restored correctly."""
        model = DummyModel()
        initial_lr = 0.001
        t0 = 10

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='cosine-warm-restarts',
            total_epochs=100,
            warmup_epochs=0,
            lr_min=1e-6,
            lr_t0=t0,
            lr_t_mult=1,
        )

        assert scheduler is not None

        # Simulate 15 epochs (past first restart)
        for _ in range(15):
            scheduler.step()

        lr_at_epoch_15 = get_lr(optimizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
            save_checkpoint(
                model, optimizer, epoch=14, loss=0.5,
                path=checkpoint_path, scheduler=scheduler
            )

            # Create fresh scheduler and restore
            new_model = DummyModel()
            new_optimizer = optim.Adam(new_model.parameters(), lr=initial_lr)
            new_scheduler = create_lr_scheduler(
                new_optimizer,
                scheduler_type='cosine-warm-restarts',
                total_epochs=100,
                warmup_epochs=0,
                lr_min=1e-6,
                lr_t0=t0,
                lr_t_mult=1,
            )

            load_checkpoint(
                checkpoint_path,
                new_model, new_optimizer,
                scheduler=new_scheduler
            )

            # LR should match
            assert get_lr(new_optimizer) == pytest.approx(
                lr_at_epoch_15, rel=0.001
            )


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing behavior."""

    def test_no_scheduler_constant_lr(self):
        """Test that no scheduler means constant LR."""
        model = DummyModel()
        initial_lr = 0.001

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='none',
            total_epochs=100,
            warmup_epochs=0,
        )

        # Should return None for 'none' with no warmup
        assert scheduler is None

        # Verify LR stays constant (no scheduler to step)
        for _ in range(10):
            assert get_lr(optimizer) == initial_lr

    def test_none_with_warmup_then_constant(self):
        """Test warmup followed by constant LR when scheduler is 'none'."""
        model = DummyModel()
        initial_lr = 0.001
        warmup_epochs = 5

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='none',
            total_epochs=50,
            warmup_epochs=warmup_epochs,
        )

        assert scheduler is not None

        lrs = simulate_epochs(optimizer, scheduler, 20)

        # During warmup, LR should increase
        for i in range(1, warmup_epochs):
            assert lrs[i] >= lrs[i - 1]

        # After warmup, LR should stay constant at initial
        for i in range(warmup_epochs, 20):
            assert lrs[i] == pytest.approx(initial_lr, rel=0.01)

    def test_legacy_step_scheduler(self):
        """Test that legacy 'step' scheduler still works."""
        model = DummyModel()
        initial_lr = 0.001

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = create_lr_scheduler(
            optimizer,
            scheduler_type='step',
            total_epochs=50,
            warmup_epochs=0,
        )

        assert scheduler is not None

        lrs = simulate_epochs(optimizer, scheduler, 25)

        # Step decay: LR drops by 0.5 every 10 epochs
        assert lrs[0] == pytest.approx(initial_lr, rel=0.01)
        # After 10 epochs: initial_lr * 0.5
        assert lrs[10] == pytest.approx(initial_lr * 0.5, rel=0.01)
        # After 20 epochs: initial_lr * 0.25
        assert lrs[20] == pytest.approx(initial_lr * 0.25, rel=0.01)

    def test_legacy_warmup_scheduler_function(self):
        """Test that the legacy get_warmup_scheduler still works."""
        model = DummyModel()
        initial_lr = 0.001
        warmup_epochs = 5

        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        scheduler = get_warmup_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=50,
            scheduler_type='cosine',
        )

        assert scheduler is not None

        lrs = simulate_epochs(optimizer, scheduler, 20)

        # Verify warmup occurs
        for i in range(1, warmup_epochs):
            assert lrs[i] >= lrs[i - 1]


class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_default_scheduler_is_none(self):
        """Test that default scheduler is None (set by config or training logic)."""
        args = parse_args([])
        assert args.lr_scheduler is None

    def test_cosine_scheduler_args(self):
        """Test parsing cosine scheduler arguments."""
        args = parse_args([
            '--lr-scheduler', 'cosine',
            '--lr-min', '0.00001',
        ])
        assert args.lr_scheduler == 'cosine'
        assert args.lr_min == 0.00001

    def test_warm_restarts_scheduler_args(self):
        """Test parsing warm restarts scheduler arguments."""
        args = parse_args([
            '--lr-scheduler', 'warmrestart',
            '--lr-t0', '15',
            '--lr-t-mult', '3',
            '--lr-min', '0.0001',
        ])
        assert args.lr_scheduler == 'warmrestart'
        assert args.lr_t0 == 15
        assert args.lr_t_mult == 3
        assert args.lr_min == 0.0001

    def test_default_lr_min(self):
        """Test default value for lr_min is None (set by training logic)."""
        args = parse_args([])
        assert args.lr_min is None

    def test_default_t0_and_t_mult(self):
        """Test default values for T_0 and T_mult."""
        args = parse_args([])
        assert args.lr_t0 == 10
        assert args.lr_t_mult == 2