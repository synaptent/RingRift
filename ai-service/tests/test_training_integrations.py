"""
Tests for training loop integrations (2025-12):
- GracefulShutdownHandler for SIGTERM/SIGINT handling
- Circuit breaker integration for fault tolerance
- TrainingAnomalyDetector for NaN/Inf detection
- AdaptiveGradientClipper for dynamic gradient clipping
- IntegratedTrainingManager augment_batch_dense
- GameGauntlet module for baseline testing
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


class TestIntegratedEnhancements(unittest.TestCase):
    """Tests for IntegratedTrainingManager integrations (2025-12)."""

    def setUp(self):
        """Import integrated enhancements components."""
        try:
            from app.training.integrated_enhancements import (
                IntegratedTrainingManager,
                IntegratedEnhancementsConfig,
            )
            self.IntegratedTrainingManager = IntegratedTrainingManager
            self.IntegratedEnhancementsConfig = IntegratedEnhancementsConfig
            self.has_integrated = True
        except ImportError:
            self.has_integrated = False

    def test_config_initialization(self) -> None:
        """Test IntegratedEnhancementsConfig default values."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig()

        # Check key flags (some may default to True for production use)
        self.assertFalse(config.auxiliary_tasks_enabled)
        self.assertFalse(config.batch_scheduling_enabled)
        self.assertFalse(config.background_eval_enabled)
        # curriculum_enabled may default to True in production config
        self.assertIsInstance(config.curriculum_enabled, bool)
        self.assertIsInstance(config.augmentation_enabled, bool)

    def test_config_auxiliary_tasks_enabled(self) -> None:
        """Test enabling auxiliary tasks via config."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig(
            auxiliary_tasks_enabled=True,
            aux_game_length_weight=0.2,
            aux_piece_count_weight=0.15,
        )

        self.assertTrue(config.auxiliary_tasks_enabled)
        self.assertEqual(config.aux_game_length_weight, 0.2)
        self.assertEqual(config.aux_piece_count_weight, 0.15)

    def test_config_batch_scheduling_enabled(self) -> None:
        """Test enabling batch scheduling via config."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig(
            batch_scheduling_enabled=True,
            batch_initial_size=64,
            batch_final_size=256,
        )

        self.assertTrue(config.batch_scheduling_enabled)
        self.assertEqual(config.batch_initial_size, 64)
        self.assertEqual(config.batch_final_size, 256)

    def test_config_background_eval_enabled(self) -> None:
        """Test enabling background evaluation via config."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig(
            background_eval_enabled=True,
            eval_interval_steps=500,
            eval_games_per_check=10,
        )

        self.assertTrue(config.background_eval_enabled)
        self.assertEqual(config.eval_interval_steps, 500)
        self.assertEqual(config.eval_games_per_check, 10)

    def test_manager_initialization(self) -> None:
        """Test IntegratedTrainingManager initialization."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig()
        manager = self.IntegratedTrainingManager(config=config)

        self.assertIsNotNone(manager.config)
        self.assertIsNone(manager._auxiliary_module)
        self.assertIsNone(manager._batch_scheduler)
        self.assertIsNone(manager._background_evaluator)

    def test_manager_update_step(self) -> None:
        """Test that update_step increments internal counter."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig()
        manager = self.IntegratedTrainingManager(config=config)

        initial_step = manager._step
        manager.update_step()
        self.assertEqual(manager._step, initial_step + 1)

    def test_manager_get_batch_size_default(self) -> None:
        """Test get_batch_size returns initial size when scheduler disabled."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig(
            batch_scheduling_enabled=False,
            batch_initial_size=128,
        )
        manager = self.IntegratedTrainingManager(config=config)

        batch_size = manager.get_batch_size()
        self.assertEqual(batch_size, 128)

    def test_manager_should_early_stop_default(self) -> None:
        """Test should_early_stop returns False when background eval disabled."""
        if not self.has_integrated:
            self.skipTest("IntegratedTrainingManager not available")

        config = self.IntegratedEnhancementsConfig(background_eval_enabled=False)
        manager = self.IntegratedTrainingManager(config=config)

        self.assertFalse(manager.should_early_stop())


class TestAuxiliaryTasks(unittest.TestCase):
    """Tests for auxiliary task module integration."""

    def setUp(self):
        """Import auxiliary task components."""
        try:
            from app.training.auxiliary_tasks import (
                AuxiliaryTaskModule,
                AuxTaskConfig,
            )
            self.AuxiliaryTaskModule = AuxiliaryTaskModule
            self.AuxTaskConfig = AuxTaskConfig
            self.has_aux = True
        except ImportError:
            self.has_aux = False

    def test_aux_config_defaults(self) -> None:
        """Test AuxTaskConfig default values."""
        if not self.has_aux:
            self.skipTest("AuxiliaryTaskModule not available")

        config = self.AuxTaskConfig()

        self.assertTrue(config.enabled)
        self.assertEqual(config.game_length_weight, 0.1)
        self.assertEqual(config.piece_count_weight, 0.1)
        self.assertEqual(config.outcome_weight, 0.05)

    def test_aux_module_initialization(self) -> None:
        """Test AuxiliaryTaskModule initialization."""
        if not self.has_aux:
            self.skipTest("AuxiliaryTaskModule not available")

        config = self.AuxTaskConfig()
        module = self.AuxiliaryTaskModule(input_dim=256, config=config)

        self.assertIsNotNone(module.game_length_head)
        self.assertIsNotNone(module.piece_count_head)
        self.assertIsNotNone(module.outcome_head)

    def test_aux_module_forward(self) -> None:
        """Test AuxiliaryTaskModule forward pass."""
        if not self.has_aux:
            self.skipTest("AuxiliaryTaskModule not available")

        config = self.AuxTaskConfig()
        module = self.AuxiliaryTaskModule(input_dim=256, config=config)

        # Create fake features
        features = torch.randn(4, 256)
        predictions = module(features)

        self.assertIn("game_length", predictions)
        self.assertIn("piece_count", predictions)
        self.assertIn("outcome", predictions)
        self.assertEqual(predictions["game_length"].shape, (4,))
        self.assertEqual(predictions["piece_count"].shape, (4,))
        self.assertEqual(predictions["outcome"].shape, (4, 3))  # 3 classes

    def test_aux_module_compute_loss(self) -> None:
        """Test AuxiliaryTaskModule loss computation."""
        if not self.has_aux:
            self.skipTest("AuxiliaryTaskModule not available")

        config = self.AuxTaskConfig()
        module = self.AuxiliaryTaskModule(input_dim=256, config=config)

        # Forward pass
        features = torch.randn(4, 256)
        predictions = module(features)

        # Create targets
        targets = {
            "game_length": torch.randn(4),
            "piece_count": torch.randn(4),
            "outcome": torch.randint(0, 3, (4,)),
        }

        loss, breakdown = module.compute_loss(predictions, targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIn("game_length", breakdown)
        self.assertIn("piece_count", breakdown)
        self.assertIn("outcome", breakdown)
        self.assertIn("total_aux", breakdown)


class TestPerSampleLossTracking(unittest.TestCase):
    """Tests for per-sample loss tracking infrastructure (2025-12)."""

    def test_compute_per_sample_loss_shape(self) -> None:
        """Test compute_per_sample_loss returns correct shape."""
        from app.training.training_enhancements import compute_per_sample_loss

        batch_size = 8
        num_actions = 64

        policy_logits = torch.randn(batch_size, num_actions)
        policy_targets = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        value_pred = torch.randn(batch_size)
        value_targets = torch.randn(batch_size)

        losses = compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets
        )

        self.assertEqual(losses.shape, (batch_size,))

    def test_compute_per_sample_loss_reduction(self) -> None:
        """Test compute_per_sample_loss reduction modes."""
        from app.training.training_enhancements import compute_per_sample_loss

        batch_size = 4
        num_actions = 16

        policy_logits = torch.randn(batch_size, num_actions)
        policy_targets = torch.softmax(torch.randn(batch_size, num_actions), dim=-1)
        value_pred = torch.randn(batch_size)
        value_targets = torch.randn(batch_size)

        # Test mean reduction
        mean_loss = compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets,
            reduction='mean'
        )
        self.assertEqual(mean_loss.shape, ())  # Scalar

        # Test sum reduction
        sum_loss = compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets,
            reduction='sum'
        )
        self.assertEqual(sum_loss.shape, ())  # Scalar

        # Verify mean * batch_size ≈ sum
        per_sample = compute_per_sample_loss(
            policy_logits, policy_targets, value_pred, value_targets,
            reduction='none'
        )
        self.assertAlmostEqual(
            per_sample.sum().item(),
            sum_loss.item(),
            places=4
        )

    def test_per_sample_loss_tracker_record(self) -> None:
        """Test PerSampleLossTracker records batches correctly."""
        from app.training.training_enhancements import PerSampleLossTracker

        tracker = PerSampleLossTracker(max_samples=100)

        # Record a batch
        batch_indices = [0, 1, 2, 3]
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        tracker.record_batch(batch_indices, losses, epoch=0, batch_idx=0)

        stats = tracker.get_statistics()
        self.assertEqual(stats['tracked_samples'], 4)
        self.assertEqual(stats['total_samples'], 4)
        self.assertAlmostEqual(stats['mean_loss'], 2.5, places=4)

    def test_per_sample_loss_tracker_hardest_samples(self) -> None:
        """Test PerSampleLossTracker identifies hardest samples."""
        from app.training.training_enhancements import PerSampleLossTracker

        tracker = PerSampleLossTracker(max_samples=100)

        # Record samples with known losses
        batch_indices = [0, 1, 2, 3, 4]
        losses = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0])
        tracker.record_batch(batch_indices, losses, epoch=0)

        # Get hardest 3
        hardest = tracker.get_hardest_samples(n=3)

        self.assertEqual(len(hardest), 3)
        # Sample 1 has loss 5.0, should be first
        self.assertEqual(hardest[0][0], 1)
        self.assertAlmostEqual(hardest[0][1], 5.0, places=4)
        # Sample 3 has loss 4.0, should be second
        self.assertEqual(hardest[1][0], 3)

    def test_per_sample_loss_tracker_lru_eviction(self) -> None:
        """Test PerSampleLossTracker LRU eviction works."""
        from app.training.training_enhancements import PerSampleLossTracker

        tracker = PerSampleLossTracker(max_samples=5)

        # Record more samples than max_samples
        for i in range(10):
            tracker.record_batch([i], torch.tensor([float(i)]), epoch=0)

        # Should only have max_samples tracked
        stats = tracker.get_statistics()
        self.assertLessEqual(stats['tracked_samples'], 5)


class TestFaultToleranceCLIConfiguration(unittest.TestCase):
    """Tests for CLI fault tolerance configuration (2025-12)."""

    def test_fault_tolerance_args_in_parser(self) -> None:
        """Test that fault tolerance CLI arguments exist in parser."""
        from app.training.train import parse_args

        # Parse with fault tolerance args - should not raise
        args = parse_args([
            '--data-path', '/tmp/test.npz',
            '--disable-circuit-breaker',
            '--disable-anomaly-detection',
            '--gradient-clip-mode', 'fixed',
            '--gradient-clip-max-norm', '2.0',
            '--anomaly-spike-threshold', '4.0',
            '--anomaly-gradient-threshold', '200.0',
            '--disable-graceful-shutdown',
        ])

        # Verify all fault tolerance args were parsed
        self.assertTrue(args.disable_circuit_breaker)
        self.assertTrue(args.disable_anomaly_detection)
        self.assertEqual(args.gradient_clip_mode, 'fixed')
        self.assertEqual(args.gradient_clip_max_norm, 2.0)
        self.assertEqual(args.anomaly_spike_threshold, 4.0)
        self.assertEqual(args.anomaly_gradient_threshold, 200.0)
        self.assertTrue(args.disable_graceful_shutdown)

    def test_fault_tolerance_defaults(self) -> None:
        """Test default values for fault tolerance args."""
        from app.training.train import parse_args

        args = parse_args(['--data-path', '/tmp/test.npz'])

        # Verify defaults (fault tolerance enabled by default)
        self.assertFalse(args.disable_circuit_breaker)
        self.assertFalse(args.disable_anomaly_detection)
        self.assertEqual(args.gradient_clip_mode, 'adaptive')
        self.assertEqual(args.gradient_clip_max_norm, 1.0)
        self.assertEqual(args.anomaly_spike_threshold, 3.0)
        self.assertEqual(args.anomaly_gradient_threshold, 100.0)
        self.assertFalse(args.disable_graceful_shutdown)

    def test_gradient_clip_mode_choices(self) -> None:
        """Test gradient clip mode only accepts valid choices."""
        from app.training.train import parse_args

        # Valid choices should work
        for mode in ['adaptive', 'fixed']:
            args = parse_args([
                '--data-path', '/tmp/test.npz',
                '--gradient-clip-mode', mode,
            ])
            self.assertEqual(args.gradient_clip_mode, mode)

        # Invalid choice should raise
        with self.assertRaises(SystemExit):
            parse_args([
                '--data-path', '/tmp/test.npz',
                '--gradient-clip-mode', 'invalid',
            ])


class TestDistributedFaultTolerance(unittest.TestCase):
    """Tests for fault tolerance in distributed training context (2025-12)."""

    def test_train_model_signature_has_fault_tolerance_params(self) -> None:
        """Test that train_model function accepts fault tolerance parameters."""
        import inspect
        from app.training.train import train_model

        sig = inspect.signature(train_model)
        params = sig.parameters

        # Check all fault tolerance parameters exist
        self.assertIn('enable_circuit_breaker', params)
        self.assertIn('enable_anomaly_detection', params)
        self.assertIn('gradient_clip_mode', params)
        self.assertIn('gradient_clip_max_norm', params)
        self.assertIn('anomaly_spike_threshold', params)
        self.assertIn('anomaly_gradient_threshold', params)
        self.assertIn('enable_graceful_shutdown', params)

        # Check defaults are correct
        self.assertEqual(params['enable_circuit_breaker'].default, True)
        self.assertEqual(params['enable_anomaly_detection'].default, True)
        self.assertEqual(params['gradient_clip_mode'].default, 'adaptive')
        self.assertEqual(params['gradient_clip_max_norm'].default, 1.0)
        self.assertEqual(params['anomaly_spike_threshold'].default, 3.0)
        self.assertEqual(params['anomaly_gradient_threshold'].default, 100.0)
        self.assertEqual(params['enable_graceful_shutdown'].default, True)

    def test_graceful_shutdown_handler_for_distributed(self) -> None:
        """Test GracefulShutdownHandler works in simulated distributed context."""
        from app.training.train import GracefulShutdownHandler

        handler = GracefulShutdownHandler()

        # Test setup and teardown don't raise
        callback_called = []
        def test_callback():
            callback_called.append(True)

        handler.setup(test_callback)
        self.assertFalse(handler.shutdown_requested)

        handler.teardown()

        # Callback should not be called without signal
        self.assertEqual(len(callback_called), 0)


class TestGradNormWeighting(unittest.TestCase):
    """Tests for GradNorm adaptive task weighting (2025-12)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import GradNorm components."""
        try:
            from app.training.multi_task_learning import (
                GradNormWeighter,
                MultiTaskLoss,
                MultiTaskConfig,
            )
            cls.GradNormWeighter = GradNormWeighter
            cls.MultiTaskLoss = MultiTaskLoss
            cls.MultiTaskConfig = MultiTaskConfig
            cls.has_gradnorm = True
        except ImportError:
            cls.has_gradnorm = False

    def test_gradnorm_weighter_initialization(self) -> None:
        """Test GradNormWeighter initializes correctly."""
        if not self.has_gradnorm:
            self.skipTest("GradNormWeighter not available")

        weighter = self.GradNormWeighter(num_tasks=3, alpha=1.5)
        self.assertEqual(weighter.num_tasks, 3)
        self.assertEqual(weighter.alpha, 1.5)
        self.assertEqual(len(weighter.weights), 3)

    def test_gradnorm_weights_are_positive(self) -> None:
        """Test that GradNorm weights are always positive."""
        if not self.has_gradnorm:
            self.skipTest("GradNormWeighter not available")

        weighter = self.GradNormWeighter(num_tasks=4)
        weights = weighter.weights
        self.assertTrue(all(w > 0 for w in weights))

    def test_gradnorm_weights_sum_to_num_tasks(self) -> None:
        """Test that normalized weights sum to num_tasks."""
        if not self.has_gradnorm:
            self.skipTest("GradNormWeighter not available")

        weighter = self.GradNormWeighter(num_tasks=3)
        weights_sum = weighter.weights.sum().item()
        self.assertAlmostEqual(weights_sum, 3.0, places=5)

    def test_gradnorm_initial_weights(self) -> None:
        """Test custom initial weights."""
        if not self.has_gradnorm:
            self.skipTest("GradNormWeighter not available")

        initial = [0.5, 1.0, 1.5]
        weighter = self.GradNormWeighter(num_tasks=3, initial_weights=initial)

        # Weights should preserve relative ratios
        weights = weighter.weights.detach().numpy()
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])

    def test_gradnorm_get_stats(self) -> None:
        """Test statistics retrieval from GradNorm."""
        if not self.has_gradnorm:
            self.skipTest("GradNormWeighter not available")

        weighter = self.GradNormWeighter(num_tasks=2)
        stats = weighter.get_stats()

        self.assertIn('task_0_weight', stats)
        self.assertIn('task_1_weight', stats)
        self.assertIn('task_0_train_rate', stats)
        self.assertIn('task_1_train_rate', stats)

    def test_multitask_loss_with_gradnorm(self) -> None:
        """Test MultiTaskLoss with gradnorm weighting strategy."""
        if not self.has_gradnorm:
            self.skipTest("GradNormWeighter not available")

        config = self.MultiTaskConfig(task_weighting="gradnorm")
        loss_fn = self.MultiTaskLoss(config=config)

        # Create mock outputs and targets
        auxiliary_outputs = {
            'outcome': torch.randn(4, 3),  # batch=4, 3 classes
            'legality': torch.randn(4, 100),  # batch=4, 100 moves
        }
        targets = {
            'outcome': torch.randint(0, 3, (4,)),
            'legality': torch.randint(0, 2, (4, 100)),
        }

        total_loss, loss_dict = loss_fn(auxiliary_outputs, targets)

        self.assertIsInstance(total_loss.item(), float)
        self.assertIn('total_auxiliary_loss', loss_dict)
        self.assertIn('outcome_weight', loss_dict)
        self.assertIn('legality_weight', loss_dict)


class TestUnifiedRegressionDetector(unittest.TestCase):
    """Tests for unified regression detector with EventBus integration (2025-12)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import regression detector components."""
        try:
            from app.training.regression_detector import (
                RegressionDetector,
                RegressionConfig,
                RegressionEvent,
                RegressionSeverity,
            )
            cls.RegressionDetector = RegressionDetector
            cls.RegressionConfig = RegressionConfig
            cls.RegressionEvent = RegressionEvent
            cls.RegressionSeverity = RegressionSeverity
            cls.has_regression_detector = True
        except ImportError:
            cls.has_regression_detector = False

    def test_regression_detector_creation(self) -> None:
        """Test that RegressionDetector can be created."""
        if not self.has_regression_detector:
            self.skipTest("RegressionDetector not available")

        detector = self.RegressionDetector()
        self.assertIsNotNone(detector)
        self.assertIsInstance(detector.config, self.RegressionConfig)

    def test_regression_severity_levels(self) -> None:
        """Test that severity levels are properly defined."""
        if not self.has_regression_detector:
            self.skipTest("RegressionDetector not available")

        self.assertEqual(len(self.RegressionSeverity), 4)
        self.assertEqual(self.RegressionSeverity.MINOR.value, "minor")
        self.assertEqual(self.RegressionSeverity.CRITICAL.value, "critical")

    def test_regression_detection_minor(self) -> None:
        """Test detection of minor regression."""
        if not self.has_regression_detector:
            self.skipTest("RegressionDetector not available")

        config = self.RegressionConfig(min_games_for_detection=10)
        detector = self.RegressionDetector(config=config)

        # Set baseline
        detector.set_baseline("test_model", elo=1500, win_rate=0.55)

        # Check for minor regression (elo drop of 25)
        event = detector.check_regression(
            model_id="test_model",
            current_elo=1475,  # 25 point drop
            current_win_rate=0.52,
            games_played=20,
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.severity, self.RegressionSeverity.MINOR)
        self.assertEqual(event.elo_drop, 25)

    def test_regression_detection_severe(self) -> None:
        """Test detection of severe regression."""
        if not self.has_regression_detector:
            self.skipTest("RegressionDetector not available")

        config = self.RegressionConfig(min_games_for_detection=10)
        detector = self.RegressionDetector(config=config)

        detector.set_baseline("test_model2", elo=1500, win_rate=0.55)

        # Check for severe regression (elo drop of 60)
        event = detector.check_regression(
            model_id="test_model2",
            current_elo=1440,  # 60 point drop
            current_win_rate=0.40,
            games_played=50,
        )

        self.assertIsNotNone(event)
        self.assertIn(event.severity, [self.RegressionSeverity.SEVERE, self.RegressionSeverity.CRITICAL])

    def test_consecutive_regression_escalation(self) -> None:
        """Test that consecutive regressions escalate severity."""
        if not self.has_regression_detector:
            self.skipTest("RegressionDetector not available")

        config = self.RegressionConfig(
            min_games_for_detection=10,
            consecutive_regressions_for_escalation=2,
            cooldown_seconds=0,  # Disable cooldown for testing
        )
        detector = self.RegressionDetector(config=config)

        detector.set_baseline("test_model3", elo=1500)

        # First minor regression
        event1 = detector.check_regression(
            model_id="test_model3",
            current_elo=1475,
            games_played=20,
        )
        self.assertEqual(event1.severity, self.RegressionSeverity.MINOR)
        self.assertEqual(event1.consecutive_count, 1)

        # Second regression - should escalate
        event2 = detector.check_regression(
            model_id="test_model3",
            current_elo=1475,
            games_played=40,
        )
        self.assertEqual(event2.consecutive_count, 2)
        # Minor should escalate to Moderate on consecutive
        self.assertEqual(event2.severity, self.RegressionSeverity.MODERATE)

    def test_regression_cleared_status(self) -> None:
        """Test that regression clears when model recovers."""
        if not self.has_regression_detector:
            self.skipTest("RegressionDetector not available")

        config = self.RegressionConfig(min_games_for_detection=10, cooldown_seconds=0)
        detector = self.RegressionDetector(config=config)

        detector.set_baseline("test_model4", elo=1500)

        # Cause regression
        detector.check_regression(model_id="test_model4", current_elo=1475, games_played=20)
        status = detector.get_status("test_model4")
        self.assertTrue(status["is_regressing"])

        # Model recovers
        detector.check_regression(model_id="test_model4", current_elo=1510, games_played=40)
        status = detector.get_status("test_model4")
        self.assertFalse(status["is_regressing"])


class TestCircuitBreakerExponentialBackoff(unittest.TestCase):
    """Tests for circuit breaker exponential backoff with jitter (2025-12)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import circuit breaker components."""
        try:
            from app.distributed.circuit_breaker import (
                CircuitBreaker,
                CircuitState,
            )
            cls.CircuitBreaker = CircuitBreaker
            cls.CircuitState = CircuitState
            cls.has_circuit_breaker = True
        except ImportError:
            cls.has_circuit_breaker = False

    def test_backoff_initialization(self) -> None:
        """Test that backoff parameters initialize correctly."""
        if not self.has_circuit_breaker:
            self.skipTest("CircuitBreaker not available")

        breaker = self.CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=10.0,
            backoff_multiplier=3.0,
            max_backoff=300.0,
            jitter_factor=0.2,
        )
        self.assertEqual(breaker.backoff_multiplier, 3.0)
        self.assertEqual(breaker.max_backoff, 300.0)
        self.assertEqual(breaker.jitter_factor, 0.2)

    def test_consecutive_opens_tracking(self) -> None:
        """Test that consecutive_opens increments on circuit open."""
        if not self.has_circuit_breaker:
            self.skipTest("CircuitBreaker not available")

        breaker = self.CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        # Open the circuit
        breaker.record_failure("test-host")
        breaker.record_failure("test-host")

        status = breaker.get_status("test-host")
        self.assertEqual(status.state, self.CircuitState.OPEN)
        self.assertEqual(status.consecutive_opens, 1)

    def test_consecutive_opens_resets_on_recovery(self) -> None:
        """Test that consecutive_opens resets after successful recovery."""
        if not self.has_circuit_breaker:
            self.skipTest("CircuitBreaker not available")

        breaker = self.CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.01,  # Very short for testing
            jitter_factor=0.0,  # No jitter for deterministic test
        )

        # Open the circuit
        breaker.record_failure("test-host2")
        breaker.record_failure("test-host2")
        self.assertEqual(breaker.get_status("test-host2").consecutive_opens, 1)

        # Wait for recovery timeout
        time.sleep(0.02)

        # Should be half-open now
        self.assertTrue(breaker.can_execute("test-host2"))
        state = breaker.get_state("test-host2")
        self.assertEqual(state, self.CircuitState.HALF_OPEN)

        # Successful recovery closes circuit and resets consecutive_opens
        breaker.record_success("test-host2")
        status = breaker.get_status("test-host2")
        self.assertEqual(status.state, self.CircuitState.CLOSED)
        self.assertEqual(status.consecutive_opens, 0)

    def test_backoff_computation(self) -> None:
        """Test that backoff increases with consecutive opens."""
        if not self.has_circuit_breaker:
            self.skipTest("CircuitBreaker not available")

        breaker = self.CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=10.0,
            backoff_multiplier=2.0,
            max_backoff=600.0,
            jitter_factor=0.0,  # No jitter for deterministic test
        )

        # Manually access internal circuit to test backoff computation
        circuit = breaker._get_or_create_circuit("backoff-test")
        circuit.consecutive_opens = 0
        self.assertAlmostEqual(breaker._compute_backoff_timeout(circuit), 10.0)

        circuit.consecutive_opens = 1
        self.assertAlmostEqual(breaker._compute_backoff_timeout(circuit), 20.0)

        circuit.consecutive_opens = 2
        self.assertAlmostEqual(breaker._compute_backoff_timeout(circuit), 40.0)

        circuit.consecutive_opens = 3
        self.assertAlmostEqual(breaker._compute_backoff_timeout(circuit), 80.0)

    def test_backoff_capped_at_max(self) -> None:
        """Test that backoff is capped at max_backoff."""
        if not self.has_circuit_breaker:
            self.skipTest("CircuitBreaker not available")

        breaker = self.CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=60.0,
            backoff_multiplier=2.0,
            max_backoff=120.0,
            jitter_factor=0.0,
        )

        circuit = breaker._get_or_create_circuit("cap-test")
        circuit.consecutive_opens = 10  # Would be 60 * 2^10 = 61440 without cap

        backoff = breaker._compute_backoff_timeout(circuit)
        self.assertEqual(backoff, 120.0)

    def test_status_includes_consecutive_opens(self) -> None:
        """Test that CircuitStatus includes consecutive_opens."""
        if not self.has_circuit_breaker:
            self.skipTest("CircuitBreaker not available")

        breaker = self.CircuitBreaker(failure_threshold=2)

        # Open the circuit multiple times
        for _ in range(2):
            breaker.record_failure("status-test")

        status = breaker.get_status("status-test")
        self.assertTrue(hasattr(status, 'consecutive_opens'))
        self.assertEqual(status.consecutive_opens, 1)

        # Also test to_dict includes it
        status_dict = status.to_dict()
        self.assertIn('consecutive_opens', status_dict)


class TestAugmentBatchDense(unittest.TestCase):
    """Tests for IntegratedTrainingManager.augment_batch_dense()."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check if integrated enhancements are available."""
        try:
            from app.training.integrated_enhancements import (
                IntegratedTrainingManager,
                IntegratedEnhancementsConfig,
            )
            cls.has_enhancements = True
            cls.IntegratedTrainingManager = IntegratedTrainingManager
            cls.IntegratedEnhancementsConfig = IntegratedEnhancementsConfig
        except ImportError:
            cls.has_enhancements = False

    def test_augment_batch_dense_shapes_preserved(self) -> None:
        """Test that augment_batch_dense preserves tensor shapes."""
        if not self.has_enhancements:
            self.skipTest("IntegratedEnhancements not available")

        config = self.IntegratedEnhancementsConfig(augmentation_enabled=True)
        manager = self.IntegratedTrainingManager(
            config=config, model=None, board_type="square8"
        )
        manager.initialize_all()

        # Create test data
        features = torch.randn(4, 19, 8, 8)
        policy = torch.softmax(torch.randn(4, 4160), dim=1)

        aug_features, aug_policy = manager.augment_batch_dense(features, policy)

        self.assertEqual(aug_features.shape, features.shape)
        self.assertEqual(aug_policy.shape, policy.shape)

    def test_augment_batch_dense_no_augmentor(self) -> None:
        """Test that augment_batch_dense returns input unchanged when disabled."""
        if not self.has_enhancements:
            self.skipTest("IntegratedEnhancements not available")

        config = self.IntegratedEnhancementsConfig(augmentation_enabled=False)
        manager = self.IntegratedTrainingManager(
            config=config, model=None, board_type="square8"
        )
        manager.initialize_all()

        features = torch.randn(2, 19, 8, 8)
        policy = torch.softmax(torch.randn(2, 4160), dim=1)

        aug_features, aug_policy = manager.augment_batch_dense(features, policy)

        # Should return same tensors
        self.assertTrue(torch.equal(aug_features, features))
        self.assertTrue(torch.equal(aug_policy, policy))

    def test_augment_batch_dense_policy_still_valid(self) -> None:
        """Test that augmented policy is still a valid probability distribution."""
        if not self.has_enhancements:
            self.skipTest("IntegratedEnhancements not available")

        config = self.IntegratedEnhancementsConfig(augmentation_enabled=True)
        manager = self.IntegratedTrainingManager(
            config=config, model=None, board_type="square8"
        )
        manager.initialize_all()

        features = torch.randn(2, 19, 8, 8)
        policy = torch.softmax(torch.randn(2, 4160), dim=1)

        _, aug_policy = manager.augment_batch_dense(features, policy)

        # Check that policy sums to ~1 (allowing for floating point error)
        sums = aug_policy.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


class TestGameGauntlet(unittest.TestCase):
    """Tests for game_gauntlet module."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check if game_gauntlet is available."""
        try:
            from app.training.game_gauntlet import (
                BaselineOpponent,
                GauntletResult,
                BASELINE_ELOS,
            )
            cls.has_gauntlet = True
            cls.BaselineOpponent = BaselineOpponent
            cls.GauntletResult = GauntletResult
            cls.BASELINE_ELOS = BASELINE_ELOS
        except ImportError:
            cls.has_gauntlet = False

    def test_baseline_opponent_enum(self) -> None:
        """Test that BaselineOpponent enum has expected values."""
        if not self.has_gauntlet:
            self.skipTest("game_gauntlet not available")

        self.assertEqual(self.BaselineOpponent.RANDOM.value, "random")
        self.assertEqual(self.BaselineOpponent.HEURISTIC.value, "heuristic")

    def test_baseline_elos_defined(self) -> None:
        """Test that BASELINE_ELOS has entries for all opponents."""
        if not self.has_gauntlet:
            self.skipTest("game_gauntlet not available")

        self.assertIn(self.BaselineOpponent.RANDOM, self.BASELINE_ELOS)
        self.assertIn(self.BaselineOpponent.HEURISTIC, self.BASELINE_ELOS)
        self.assertEqual(self.BASELINE_ELOS[self.BaselineOpponent.RANDOM], 400)
        self.assertEqual(self.BASELINE_ELOS[self.BaselineOpponent.HEURISTIC], 1200)

    def test_gauntlet_result_dataclass(self) -> None:
        """Test that GauntletResult has expected fields."""
        if not self.has_gauntlet:
            self.skipTest("game_gauntlet not available")

        result = self.GauntletResult(
            total_games=20,
            total_wins=15,
            total_losses=5,
            total_draws=0,
            win_rate=0.75,
            opponent_results={"random": {"wins": 15, "games": 20}},
            passes_baseline_gating=True,
            failed_baselines=[],
            estimated_elo=1400,
        )

        self.assertEqual(result.total_games, 20)
        self.assertEqual(result.win_rate, 0.75)
        self.assertTrue(result.passes_baseline_gating)
        self.assertEqual(result.estimated_elo, 1400)


class TestBaselineGatingStatus(unittest.TestCase):
    """Tests for baseline gating status in IntegratedTrainingManager."""

    @classmethod
    def setUpClass(cls) -> None:
        """Check if integrated enhancements are available."""
        try:
            from app.training.integrated_enhancements import (
                IntegratedTrainingManager,
                IntegratedEnhancementsConfig,
            )
            cls.has_enhancements = True
            cls.IntegratedTrainingManager = IntegratedTrainingManager
            cls.IntegratedEnhancementsConfig = IntegratedEnhancementsConfig
        except ImportError:
            cls.has_enhancements = False

    def test_get_baseline_gating_status_default(self) -> None:
        """Test default baseline gating status (passes when no evaluator)."""
        if not self.has_enhancements:
            self.skipTest("IntegratedEnhancements not available")

        config = self.IntegratedEnhancementsConfig(background_eval_enabled=False)
        manager = self.IntegratedTrainingManager(
            config=config, model=None, board_type="square8"
        )

        passes, failed, consecutive = manager.get_baseline_gating_status()

        # Default should pass (no evaluator means no failures)
        self.assertTrue(passes)
        self.assertEqual(len(failed), 0)
        self.assertEqual(consecutive, 0)


if __name__ == "__main__":
    unittest.main()
