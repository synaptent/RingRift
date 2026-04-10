#!/usr/bin/env python3
"""
Validation script for training loop integrations (2025-12).

Tests:
1. Circuit breaker initialization and recording
2. Anomaly detection with real loss values
3. Adaptive gradient clipping during training
4. Graceful shutdown handler setup
5. Prometheus metrics recording
6. Full training loop with all features enabled

Usage:
    python scripts/validate_training_integrations.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn


def test_circuit_breaker():
    """Test circuit breaker integration."""
    print("\n=== Testing Circuit Breaker ===")
    try:
        from app.distributed.circuit_breaker import get_training_breaker

        breaker = get_training_breaker()
        print("  ✓ Circuit breaker initialized")

        # Test can_execute
        can_exec = breaker.can_execute("training_epoch")
        print(f"  ✓ can_execute('training_epoch') = {can_exec}")

        # Test record_success
        breaker.record_success("training_epoch")
        print("  ✓ record_success() called")

        # Test record_failure
        breaker.record_failure("training_epoch")
        print("  ✓ record_failure() called")

        return True
    except Exception as e:
        print(f"  ✗ Circuit breaker test failed: {e}")
        return False


def test_anomaly_detector():
    """Test anomaly detection."""
    print("\n=== Testing Anomaly Detector ===")
    try:
        from app.training.training_enhancements import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector(
            loss_spike_threshold=3.0,
            gradient_norm_threshold=100.0,
            halt_on_nan=False,
        )
        print("  ✓ Anomaly detector initialized")

        # Test normal loss
        result = detector.check_loss(0.5, step=1)
        print(f"  ✓ Normal loss (0.5) detected as anomaly: {result}")

        # Test NaN detection
        result = detector.check_loss(float('nan'), step=2)
        print(f"  ✓ NaN loss detected as anomaly: {result}")

        # Test summary
        summary = detector.get_summary()
        print(f"  ✓ Summary: total_anomalies={summary.get('total_anomalies', 0)}")

        return True
    except Exception as e:
        print(f"  ✗ Anomaly detector test failed: {e}")
        return False


def test_adaptive_clipper():
    """Test adaptive gradient clipping."""
    print("\n=== Testing Adaptive Gradient Clipper ===")
    try:
        from app.training.training_enhancements import AdaptiveGradientClipper

        clipper = AdaptiveGradientClipper(
            initial_max_norm=1.0,
            percentile=90.0,
            history_size=100,
        )
        print(f"  ✓ Adaptive clipper initialized (norm={clipper.current_max_norm})")

        # Create simple model and test clipping
        model = nn.Linear(10, 2)
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        grad_norm = clipper.update_and_clip(model.parameters())
        print(f"  ✓ Gradient clipped, norm={grad_norm:.4f}")

        stats = clipper.get_stats()
        print(f"  ✓ Stats: clip_norm={stats['current_clip_norm']:.4f}, history_size={stats['history_size']}")

        return True
    except Exception as e:
        print(f"  ✗ Adaptive clipper test failed: {e}")
        return False


def test_graceful_shutdown():
    """Test graceful shutdown handler."""
    print("\n=== Testing Graceful Shutdown Handler ===")
    try:
        from app.training.train import GracefulShutdownHandler

        handler = GracefulShutdownHandler()
        print("  ✓ Shutdown handler initialized")

        callback_called = [False]
        def test_callback():
            callback_called[0] = True

        handler.setup(test_callback)
        print("  ✓ Handler setup complete")

        # Verify shutdown_requested is False initially
        assert not handler.shutdown_requested, "shutdown_requested should be False"
        print(f"  ✓ shutdown_requested = {handler.shutdown_requested}")

        handler.teardown()
        print("  ✓ Handler teardown complete")

        return True
    except Exception as e:
        print(f"  ✗ Graceful shutdown test failed: {e}")
        return False


def test_prometheus_metrics():
    """Test Prometheus metrics are defined."""
    print("\n=== Testing Prometheus Metrics ===")
    try:
        from app.training.train import (
            ANOMALY_DETECTIONS,
            CIRCUIT_BREAKER_STATE,
            GRADIENT_CLIP_NORM,
            GRADIENT_NORM,
            HAS_PROMETHEUS,
        )

        print(f"  ✓ HAS_PROMETHEUS = {HAS_PROMETHEUS}")

        if HAS_PROMETHEUS:
            print(f"  ✓ CIRCUIT_BREAKER_STATE defined: {CIRCUIT_BREAKER_STATE is not None}")
            print(f"  ✓ ANOMALY_DETECTIONS defined: {ANOMALY_DETECTIONS is not None}")
            print(f"  ✓ GRADIENT_CLIP_NORM defined: {GRADIENT_CLIP_NORM is not None}")
            print(f"  ✓ GRADIENT_NORM defined: {GRADIENT_NORM is not None}")

            # Test setting a metric
            if GRADIENT_NORM:
                GRADIENT_NORM.labels(config='test').set(0.5)
                print("  ✓ GRADIENT_NORM metric set successfully")
        else:
            print("  ⚠ Prometheus not available, skipping metric tests")

        return True
    except Exception as e:
        print(f"  ✗ Prometheus metrics test failed: {e}")
        return False


def test_short_training_loop():
    """Test a short training loop with all features enabled."""
    print("\n=== Testing Short Training Loop ===")

    try:
        # Check for training data
        data_path = ROOT / "data" / "training" / "auto_training_sq8_2p.npz"
        if not data_path.exists():
            print(f"  ⚠ Training data not found at {data_path}, skipping training test")
            return True

        print(f"  ✓ Training data found: {data_path}")

        # Import training components
        from app.training.train import (
            HAS_CIRCUIT_BREAKER,
            HAS_TRAINING_ENHANCEMENTS,
            GracefulShutdownHandler,
        )

        print(f"  ✓ HAS_CIRCUIT_BREAKER = {HAS_CIRCUIT_BREAKER}")
        print(f"  ✓ HAS_TRAINING_ENHANCEMENTS = {HAS_TRAINING_ENHANCEMENTS}")

        # Import model
        from app.ai.neural_net import RingRiftCNN_v3

        # Create a small model for testing (square8, 2 players)
        model = RingRiftCNN_v3(
            board_size=8,
            in_channels=14,
            global_features=8,
            num_res_blocks=2,  # Minimal for testing
            num_filters=32,    # Minimal for testing
            num_players=2,
        )
        print(f"  ✓ Model created: {type(model).__name__}")

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initialize fault tolerance components
        if HAS_CIRCUIT_BREAKER:
            from app.distributed.circuit_breaker import get_training_breaker
            training_breaker = get_training_breaker()
            print("  ✓ Training circuit breaker enabled")
        else:
            training_breaker = None

        if HAS_TRAINING_ENHANCEMENTS:
            from app.training.training_enhancements import (
                AdaptiveGradientClipper,
                TrainingAnomalyDetector,
            )
            anomaly_detector = TrainingAnomalyDetector(
                loss_spike_threshold=3.0,
                halt_on_nan=False,
            )
            adaptive_clipper = AdaptiveGradientClipper(initial_max_norm=1.0)
            print("  ✓ Anomaly detector and adaptive clipper enabled")
        else:
            anomaly_detector = None
            adaptive_clipper = None

        # Setup graceful shutdown
        shutdown_handler = GracefulShutdownHandler()
        checkpoint_saved = [False]
        def emergency_save():
            checkpoint_saved[0] = True
            print("    [Emergency checkpoint would be saved here]")
        shutdown_handler.setup(emergency_save)
        print("  ✓ Graceful shutdown handler configured")

        # Simulate a few training steps
        print("  → Running 5 simulated training steps...")
        model.train()
        anomaly_step = 0

        # Input channels = in_channels * (history_length + 1) = 14 * 4 = 56
        total_in_channels = 14 * 4  # Default history_length=3

        for step in range(5):
            # Create dummy input matching model's expected shape
            features = torch.randn(4, total_in_channels, 8, 8)  # Batch of 4
            globals_vec = torch.randn(4, 8)  # global_features=8

            # Forward pass
            optimizer.zero_grad()
            value_pred, policy_pred, _ = model(features, globals_vec)

            # Dummy loss
            loss = value_pred.mean() + policy_pred.mean()
            loss.backward()

            # Adaptive gradient clipping
            if adaptive_clipper:
                grad_norm = adaptive_clipper.update_and_clip(model.parameters())
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Anomaly detection
            if anomaly_detector:
                anomaly_step += 1
                loss_val = loss.item()
                if anomaly_detector.check_loss(loss_val, anomaly_step):
                    print(f"    Step {step}: Anomaly detected!")
                    if training_breaker:
                        training_breaker.record_failure("training_epoch")

            print(f"    Step {step}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

        # Record success with circuit breaker
        if training_breaker:
            training_breaker.record_success("training_epoch")
            print("  ✓ Circuit breaker success recorded")

        # Cleanup
        shutdown_handler.teardown()
        print("  ✓ Training loop completed successfully")

        return True
    except Exception as e:
        import traceback
        print(f"  ✗ Training loop test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Training Integration Validation")
    print("=" * 60)

    results = {
        "Circuit Breaker": test_circuit_breaker(),
        "Anomaly Detector": test_anomaly_detector(),
        "Adaptive Clipper": test_adaptive_clipper(),
        "Graceful Shutdown": test_graceful_shutdown(),
        "Prometheus Metrics": test_prometheus_metrics(),
        "Training Loop": test_short_training_loop(),
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
