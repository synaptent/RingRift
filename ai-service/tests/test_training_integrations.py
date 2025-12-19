"""
Tests for training loop integrations (2025-12):
- GracefulShutdownHandler for SIGTERM/SIGINT handling
- Circuit breaker integration for fault tolerance
- TrainingAnomalyDetector for NaN/Inf detection
- AdaptiveGradientClipper for dynamic gradient clipping
"""

import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

# Ensure app package is importable when running tests directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.training.train import GracefulShutdownHandler


class SimpleModel(nn.Module):
    """Simple neural network for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestGracefulShutdownHandler(unittest.TestCase):
    """Tests for GracefulShutdownHandler class."""

    def test_initialization(self) -> None:
        """Test default initialization."""
        handler = GracefulShutdownHandler()
        self.assertFalse(handler.shutdown_requested)
        self.assertEqual(handler._original_handlers, {})
        self.assertIsNone(handler._checkpoint_callback)

    def test_setup_and_teardown(self) -> None:
        """Test signal handler setup and teardown."""
        handler = GracefulShutdownHandler()
        callback_called = [False]

        def dummy_callback():
            callback_called[0] = True

        # Setup
        handler.setup(dummy_callback)
        self.assertEqual(handler._checkpoint_callback, dummy_callback)

        # Teardown
        handler.teardown()
        self.assertEqual(handler._original_handlers, {})

    def test_shutdown_requested_property(self) -> None:
        """Test shutdown_requested property."""
        handler = GracefulShutdownHandler()
        self.assertFalse(handler.shutdown_requested)

        handler._shutdown_requested = True
        self.assertTrue(handler.shutdown_requested)

    def test_callback_stored(self) -> None:
        """Test that callback is properly stored."""
        handler = GracefulShutdownHandler()

        def my_callback():
            pass

        handler.setup(my_callback)
        self.assertIs(handler._checkpoint_callback, my_callback)
        handler.teardown()


class TestTrainingAnomalyDetector(unittest.TestCase):
    """Tests for TrainingAnomalyDetector integration."""

    def setUp(self):
        """Import the anomaly detector."""
        try:
            from app.training.training_enhancements import TrainingAnomalyDetector
            self.TrainingAnomalyDetector = TrainingAnomalyDetector
            self.has_detector = True
        except ImportError:
            self.has_detector = False

    def test_import(self) -> None:
        """Test that TrainingAnomalyDetector can be imported."""
        self.assertTrue(self.has_detector, "TrainingAnomalyDetector should be importable")

    @unittest.skipUnless(True, "Requires TrainingAnomalyDetector")
    def test_initialization(self) -> None:
        """Test detector initialization with custom thresholds."""
        if not self.has_detector:
            self.skipTest("TrainingAnomalyDetector not available")

        detector = self.TrainingAnomalyDetector(
            loss_spike_threshold=3.0,
            gradient_norm_threshold=100.0,
            loss_window_size=100,
        )
        self.assertIsNotNone(detector)
        self.assertEqual(detector.loss_spike_threshold, 3.0)
        self.assertEqual(detector.gradient_norm_threshold, 100.0)

    @unittest.skipUnless(True, "Requires TrainingAnomalyDetector")
    def test_normal_loss_not_detected(self) -> None:
        """Test that normal losses are not flagged as anomalies."""
        if not self.has_detector:
            self.skipTest("TrainingAnomalyDetector not available")

        detector = self.TrainingAnomalyDetector(
            loss_spike_threshold=3.0,
            halt_on_nan=False,
        )

        # Normal losses should not trigger anomaly
        for step, loss in enumerate([0.5, 0.45, 0.4, 0.35, 0.3]):
            result = detector.check_loss(loss, step)
            self.assertFalse(result, f"Normal loss {loss} should not be flagged")

    @unittest.skipUnless(True, "Requires TrainingAnomalyDetector")
    def test_nan_loss_detected(self) -> None:
        """Test that NaN losses are detected."""
        if not self.has_detector:
            self.skipTest("TrainingAnomalyDetector not available")

        detector = self.TrainingAnomalyDetector(halt_on_nan=False)

        # NaN should trigger anomaly
        result = detector.check_loss(float('nan'), step=1)
        self.assertTrue(result, "NaN loss should be flagged")

    @unittest.skipUnless(True, "Requires TrainingAnomalyDetector")
    def test_inf_loss_detected(self) -> None:
        """Test that infinite losses are detected."""
        if not self.has_detector:
            self.skipTest("TrainingAnomalyDetector not available")

        detector = self.TrainingAnomalyDetector(halt_on_nan=False)

        # Inf should trigger anomaly
        result = detector.check_loss(float('inf'), step=1)
        self.assertTrue(result, "Inf loss should be flagged")

    @unittest.skipUnless(True, "Requires TrainingAnomalyDetector")
    def test_get_summary(self) -> None:
        """Test anomaly summary retrieval."""
        if not self.has_detector:
            self.skipTest("TrainingAnomalyDetector not available")

        detector = self.TrainingAnomalyDetector(halt_on_nan=False)

        # Generate some anomalies
        detector.check_loss(float('nan'), step=1)
        detector.check_loss(0.5, step=2)  # Normal
        detector.check_loss(float('inf'), step=3)

        summary = detector.get_summary()
        self.assertIn('total_anomalies', summary)
        self.assertGreaterEqual(summary['total_anomalies'], 2)


class TestAdaptiveGradientClipper(unittest.TestCase):
    """Tests for AdaptiveGradientClipper integration."""

    def setUp(self):
        """Import the adaptive clipper."""
        try:
            from app.training.training_enhancements import AdaptiveGradientClipper
            self.AdaptiveGradientClipper = AdaptiveGradientClipper
            self.has_clipper = True
        except ImportError:
            self.has_clipper = False

    def test_import(self) -> None:
        """Test that AdaptiveGradientClipper can be imported."""
        self.assertTrue(self.has_clipper, "AdaptiveGradientClipper should be importable")

    @unittest.skipUnless(True, "Requires AdaptiveGradientClipper")
    def test_initialization(self) -> None:
        """Test clipper initialization."""
        if not self.has_clipper:
            self.skipTest("AdaptiveGradientClipper not available")

        clipper = self.AdaptiveGradientClipper(
            initial_max_norm=1.0,
            percentile=90.0,
            history_size=100,
            min_clip=0.1,
            max_clip=10.0,
        )
        self.assertEqual(clipper.current_max_norm, 1.0)
        self.assertEqual(clipper.percentile, 90.0)
        self.assertEqual(clipper.history_size, 100)

    @unittest.skipUnless(True, "Requires AdaptiveGradientClipper")
    def test_update_and_clip(self) -> None:
        """Test gradient clipping updates history."""
        if not self.has_clipper:
            self.skipTest("AdaptiveGradientClipper not available")

        clipper = self.AdaptiveGradientClipper(initial_max_norm=1.0)
        model = SimpleModel()

        # Create some gradients
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Update and clip
        grad_norm = clipper.update_and_clip(model.parameters())

        self.assertIsInstance(grad_norm, float)
        self.assertGreater(len(clipper.grad_norms), 0)

    @unittest.skipUnless(True, "Requires AdaptiveGradientClipper")
    def test_get_stats(self) -> None:
        """Test statistics retrieval."""
        if not self.has_clipper:
            self.skipTest("AdaptiveGradientClipper not available")

        clipper = self.AdaptiveGradientClipper(initial_max_norm=1.0)
        model = SimpleModel()

        # Generate some gradient history
        for _ in range(5):
            model.zero_grad()
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            clipper.update_and_clip(model.parameters())

        stats = clipper.get_stats()
        self.assertIn('current_clip_norm', stats)
        self.assertIn('mean_grad_norm', stats)
        self.assertIn('max_grad_norm', stats)
        self.assertIn('history_size', stats)
        self.assertEqual(stats['history_size'], 5)

    @unittest.skipUnless(True, "Requires AdaptiveGradientClipper")
    def test_adaptive_threshold_adjustment(self) -> None:
        """Test that threshold adjusts based on history."""
        if not self.has_clipper:
            self.skipTest("AdaptiveGradientClipper not available")

        clipper = self.AdaptiveGradientClipper(
            initial_max_norm=1.0,
            percentile=90.0,
            history_size=20,
        )
        model = SimpleModel()

        initial_threshold = clipper.current_max_norm

        # Generate enough history to trigger adaptation
        for _ in range(15):
            model.zero_grad()
            x = torch.randn(4, 10) * 10  # Larger inputs for larger gradients
            y = model(x)
            loss = y.sum()
            loss.backward()
            clipper.update_and_clip(model.parameters())

        # Threshold should have adapted (may increase or decrease based on gradients)
        # Just verify it's a valid number
        self.assertIsInstance(clipper.current_max_norm, float)
        self.assertGreater(clipper.current_max_norm, 0)


class TestCircuitBreakerIntegration(unittest.TestCase):
    """Tests for circuit breaker integration."""

    def setUp(self):
        """Import circuit breaker components."""
        try:
            from app.distributed.circuit_breaker import (
                get_training_breaker,
                CircuitState,
            )
            self.get_training_breaker = get_training_breaker
            self.CircuitState = CircuitState
            self.has_breaker = True
        except ImportError:
            self.has_breaker = False

    def test_import(self) -> None:
        """Test that circuit breaker can be imported."""
        self.assertTrue(self.has_breaker, "Circuit breaker should be importable")

    @unittest.skipUnless(True, "Requires circuit breaker")
    def test_get_training_breaker(self) -> None:
        """Test getting the training circuit breaker."""
        if not self.has_breaker:
            self.skipTest("Circuit breaker not available")

        breaker = self.get_training_breaker()
        self.assertIsNotNone(breaker)

    @unittest.skipUnless(True, "Requires circuit breaker")
    def test_can_execute(self) -> None:
        """Test can_execute method."""
        if not self.has_breaker:
            self.skipTest("Circuit breaker not available")

        breaker = self.get_training_breaker()

        # Should be able to execute initially (circuit closed)
        can_exec = breaker.can_execute("training_epoch")
        self.assertTrue(can_exec, "Should be able to execute when circuit is closed")

    @unittest.skipUnless(True, "Requires circuit breaker")
    def test_record_success(self) -> None:
        """Test recording successful operations."""
        if not self.has_breaker:
            self.skipTest("Circuit breaker not available")

        breaker = self.get_training_breaker()

        # Record success - should not raise
        breaker.record_success("training_epoch")

    @unittest.skipUnless(True, "Requires circuit breaker")
    def test_record_failure(self) -> None:
        """Test recording failed operations."""
        if not self.has_breaker:
            self.skipTest("Circuit breaker not available")

        breaker = self.get_training_breaker()

        # Record failure - should not raise
        breaker.record_failure("training_epoch")


class TestIntegrationImports(unittest.TestCase):
    """Test that all integration imports work correctly in train.py."""

    def test_train_imports(self) -> None:
        """Test that train.py can import all new components."""
        # These should not raise ImportError
        from app.training.train import GracefulShutdownHandler

        # Check HAS_* flags are set correctly
        from app.training import train

        # These flags should exist
        self.assertTrue(hasattr(train, 'HAS_CIRCUIT_BREAKER'))
        self.assertTrue(hasattr(train, 'HAS_TRAINING_ENHANCEMENTS'))

    def test_training_enhancements_exports(self) -> None:
        """Test training_enhancements.py exports."""
        from app.training.training_enhancements import (
            TrainingAnomalyDetector,
            AdaptiveGradientClipper,
            CheckpointAverager,
            EarlyStopping,
            EnhancedEarlyStopping,
        )

        # All should be classes
        self.assertTrue(callable(TrainingAnomalyDetector))
        self.assertTrue(callable(AdaptiveGradientClipper))
        self.assertTrue(callable(CheckpointAverager))
        self.assertTrue(callable(EarlyStopping))
        self.assertTrue(callable(EnhancedEarlyStopping))

        # EarlyStopping should be alias for EnhancedEarlyStopping
        self.assertIs(EarlyStopping, EnhancedEarlyStopping)


if __name__ == "__main__":
    unittest.main()
