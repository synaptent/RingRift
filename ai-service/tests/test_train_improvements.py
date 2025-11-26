"""
Tests for training improvements in train.py:
- EarlyStopping class
- Checkpoint save/load functions
- Learning rate warmup scheduler
- CLI argument parsing
"""

import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.training.train import (  # noqa: E402
    EarlyStopping,
    get_warmup_scheduler,
    load_checkpoint,
    parse_args,
    save_checkpoint,
)


class SimpleModel(nn.Module):
    """Simple neural network for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestEarlyStopping(unittest.TestCase):
    """Tests for EarlyStopping class."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        es = EarlyStopping()
        self.assertEqual(es.patience, 10)
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_loss, float("inf"))
        self.assertIsNone(es.best_state)
        self.assertFalse(es.should_stop)

    def test_custom_patience(self) -> None:
        """Test custom patience parameter."""
        es = EarlyStopping(patience=5, min_delta=0.001)
        self.assertEqual(es.patience, 5)
        self.assertEqual(es.min_delta, 0.001)

    def test_improvement_resets_counter(self) -> None:
        """Test that improvement resets counter."""
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        # First call - should improve
        es(1.0, model)
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_loss, 1.0)

        # Worse loss - counter increments
        es(1.1, model)
        self.assertEqual(es.counter, 1)

        # Better loss - counter resets
        es(0.5, model)
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_loss, 0.5)

    def test_triggers_after_patience_exceeded(self) -> None:
        """Test that early stopping triggers after patience exceeded."""
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        es(1.0, model)  # Best so far
        self.assertFalse(es.should_stop)

        es(1.1, model)  # Worse - counter 1
        self.assertFalse(es.should_stop)

        es(1.2, model)  # Worse - counter 2
        self.assertFalse(es.should_stop)

        es(1.3, model)  # Worse - counter 3, triggers
        self.assertTrue(es.should_stop)

    def test_min_delta_threshold(self) -> None:
        """Test min_delta threshold for improvement detection."""
        es = EarlyStopping(patience=3, min_delta=0.1)
        model = SimpleModel()

        es(1.0, model)  # Best so far
        self.assertEqual(es.counter, 0)

        # Improvement less than min_delta - not considered improvement
        es(0.95, model)
        self.assertEqual(es.counter, 1)
        self.assertEqual(es.best_loss, 1.0)  # Not updated

        # Improvement greater than min_delta
        es(0.85, model)
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_loss, 0.85)  # Updated

    def test_saves_best_state(self) -> None:
        """Test that best model state is saved."""
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        # Modify model weights
        with torch.no_grad():
            model.fc.weight.fill_(1.0)

        es(1.0, model)
        self.assertIsNotNone(es.best_state)

        # Change model weights
        with torch.no_grad():
            model.fc.weight.fill_(2.0)

        # Worse loss - counter increments, best_state unchanged
        es(1.1, model)

        # Restore best state to model
        assert es.best_state is not None
        model.load_state_dict(es.best_state)
        self.assertTrue(torch.all(model.fc.weight == 1.0))

    def test_restore_best_weights(self) -> None:
        """Test restore_best_weights method."""
        es = EarlyStopping(patience=3)
        model = SimpleModel()

        with torch.no_grad():
            model.fc.weight.fill_(1.0)

        es(1.0, model)

        # Change weights
        with torch.no_grad():
            model.fc.weight.fill_(5.0)

        # Restore using the correct method name
        es.restore_best_weights(model)
        self.assertTrue(torch.all(model.fc.weight == 1.0))


class TestCheckpointFunctions(unittest.TestCase):
    """Tests for checkpoint save/load functions."""

    def test_save_and_load_checkpoint(self) -> None:
        """Test saving and loading a checkpoint."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Set specific values for testing
        with torch.no_grad():
            model.fc.weight.fill_(42.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(model, optimizer, epoch=5, loss=0.123, path=path)

            # Verify file was created
            self.assertTrue(os.path.exists(path))

            # Create new model and load checkpoint
            new_model = SimpleModel()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            epoch, loss = load_checkpoint(path, new_model, new_optimizer)

            self.assertEqual(epoch, 5)
            self.assertAlmostEqual(loss, 0.123, places=5)
            self.assertTrue(torch.all(new_model.fc.weight == 42.0))

    def test_load_checkpoint_model_only(self) -> None:
        """Test loading checkpoint without optimizer."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with torch.no_grad():
            model.fc.weight.fill_(99.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(model, optimizer, epoch=10, loss=0.5, path=path)

            new_model = SimpleModel()
            epoch, loss = load_checkpoint(path, new_model, optimizer=None)

            self.assertEqual(epoch, 10)
            self.assertAlmostEqual(loss, 0.5, places=5)
            self.assertTrue(torch.all(new_model.fc.weight == 99.0))

    def test_save_checkpoint_with_scheduler(self) -> None:
        """Test saving checkpoint with scheduler state."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        # Step scheduler a few times
        for _ in range(3):
            scheduler.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(
                model, optimizer, epoch=3, loss=0.1, path=path,
                scheduler=scheduler
            )

            checkpoint = torch.load(path)
            self.assertIn("scheduler_state_dict", checkpoint)
            self.assertIsNotNone(checkpoint["scheduler_state_dict"])

    def test_save_checkpoint_with_early_stopping(self) -> None:
        """Test saving checkpoint with early stopping state."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        es = EarlyStopping(patience=5)
        es(1.0, model)
        es(1.1, model)  # counter = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(
                model, optimizer, epoch=2, loss=0.2, path=path,
                early_stopping=es
            )

            checkpoint = torch.load(path)
            self.assertIn("early_stopping", checkpoint)
            es_state = checkpoint["early_stopping"]
            self.assertEqual(es_state["counter"], 1)
            self.assertEqual(es_state["best_loss"], 1.0)


class TestWarmupScheduler(unittest.TestCase):
    """Tests for learning rate warmup scheduler."""

    def test_no_warmup(self) -> None:
        """Test with zero warmup epochs."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scheduler = get_warmup_scheduler(
            optimizer, warmup_epochs=0, total_epochs=100
        )
        self.assertIsNone(scheduler)

    def test_linear_warmup(self) -> None:
        """Test linear warmup increases LR progressively."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        scheduler = get_warmup_scheduler(
            optimizer, warmup_epochs=5, total_epochs=100, scheduler_type="none"
        )
        self.assertIsNotNone(scheduler)
        assert scheduler is not None  # For type checker

        # Check that LR increases during warmup
        # PyTorch LRScheduler calls step() in __init__, start from epoch 1
        initial_lr = scheduler.get_last_lr()[0]
        # Should be less than base LR during warmup
        self.assertLess(initial_lr, 0.01)

        scheduler.step()
        lr_after_step1 = scheduler.get_last_lr()[0]
        self.assertGreater(lr_after_step1, initial_lr)

        scheduler.step()
        lr_after_step2 = scheduler.get_last_lr()[0]
        # LR continues to increase
        self.assertGreater(lr_after_step2, lr_after_step1)

    def test_warmup_then_constant(self) -> None:
        """Test warmup followed by constant LR."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        scheduler = get_warmup_scheduler(
            optimizer, warmup_epochs=3, total_epochs=10, scheduler_type="none"
        )
        assert scheduler is not None  # For type checker

        # After warmup
        for _ in range(3):
            scheduler.step()

        # Should be at base LR
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.01, places=6)

        # Continue stepping - should stay constant
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.01, places=6)

    def test_warmup_step_scheduler(self) -> None:
        """Test warmup followed by step decay scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Create warmup + step scheduler
        scheduler = get_warmup_scheduler(
            optimizer, warmup_epochs=3, total_epochs=20, scheduler_type="step"
        )
        assert scheduler is not None  # For type checker

        # During warmup - LR should increase
        self.assertLess(scheduler.get_last_lr()[0], 0.01)

        for _ in range(3):
            scheduler.step()

        # At base LR after warmup
        initial_lr = scheduler.get_last_lr()[0]
        self.assertAlmostEqual(initial_lr, 0.01, places=5)

    def test_warmup_cosine_scheduler(self) -> None:
        """Test warmup followed by cosine annealing scheduler."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        scheduler = get_warmup_scheduler(
            optimizer, warmup_epochs=2, total_epochs=10,
            scheduler_type="cosine"
        )
        assert scheduler is not None  # For type checker

        # Warmup phase
        self.assertLess(scheduler.get_last_lr()[0], 0.01)

        scheduler.step()
        scheduler.step()

        # After warmup, cosine annealing should start
        for _ in range(4):
            scheduler.step()

        # LR should decrease with cosine schedule
        lr = scheduler.get_last_lr()[0]
        self.assertLess(lr, 0.01)  # Should be less than initial


class TestCliArguments(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def test_default_arguments(self) -> None:
        """Test default argument values."""
        args = parse_args([])

        self.assertEqual(args.early_stopping_patience, 10)
        self.assertEqual(args.checkpoint_dir, "checkpoints")
        self.assertEqual(args.checkpoint_interval, 5)
        self.assertEqual(args.warmup_epochs, 0)
        self.assertEqual(args.lr_scheduler, "none")
        self.assertIsNone(args.resume)

    def test_early_stopping_disabled(self) -> None:
        """Test disabling early stopping."""
        args = parse_args(["--early-stopping-patience", "0"])
        self.assertEqual(args.early_stopping_patience, 0)

    def test_custom_checkpoint_settings(self) -> None:
        """Test custom checkpoint settings."""
        args = parse_args([
            "--checkpoint-dir", "/custom/path",
            "--checkpoint-interval", "10"
        ])
        self.assertEqual(args.checkpoint_dir, "/custom/path")
        self.assertEqual(args.checkpoint_interval, 10)

    def test_warmup_settings(self) -> None:
        """Test warmup settings."""
        args = parse_args([
            "--warmup-epochs", "5",
            "--lr-scheduler", "cosine"
        ])
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.lr_scheduler, "cosine")

    def test_resume_checkpoint(self) -> None:
        """Test resume from checkpoint."""
        args = parse_args(["--resume", "/path/to/checkpoint.pt"])
        self.assertEqual(args.resume, "/path/to/checkpoint.pt")

    def test_scheduler_choices(self) -> None:
        """Test valid scheduler choices."""
        for choice in ["none", "step", "cosine"]:
            args = parse_args(["--lr-scheduler", choice])
            self.assertEqual(args.lr_scheduler, choice)

    def test_board_type_choices(self) -> None:
        """Test valid board type choices."""
        for choice in ["square8", "square19", "hexagonal"]:
            args = parse_args(["--board-type", choice])
            self.assertEqual(args.board_type, choice)

    def test_training_parameters(self) -> None:
        """Test basic training parameters."""
        args = parse_args([
            "--epochs", "50",
            "--batch-size", "64",
            "--learning-rate", "0.0001",
            "--seed", "42"
        ])
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.learning_rate, 0.0001)
        self.assertEqual(args.seed, 42)


if __name__ == "__main__":
    unittest.main()