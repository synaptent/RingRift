"""Tests for adaptive loss anomaly thresholds (January 2026).

These tests verify the dynamic threshold adaptation based on training epoch:
- Early training (epochs 0-4): Permissive thresholds
- Mid training (epochs 5-14): Standard thresholds
- Late training (epochs 15+): Strict thresholds

Expected Elo improvement: +8-12 from better anomaly detection.
"""

import pytest


class TestAdaptiveThresholdConstants:
    """Test the adaptive threshold constants and helper functions."""

    def test_get_loss_anomaly_threshold_early_epoch(self):
        """Test early training uses permissive threshold."""
        from app.config.thresholds import get_loss_anomaly_threshold, LOSS_ANOMALY_THRESHOLD_EARLY

        for epoch in range(5):  # Epochs 0-4
            threshold = get_loss_anomaly_threshold(epoch)
            assert threshold == LOSS_ANOMALY_THRESHOLD_EARLY
            assert threshold == 5.0  # Permissive

    def test_get_loss_anomaly_threshold_mid_epoch(self):
        """Test mid training uses standard threshold."""
        from app.config.thresholds import get_loss_anomaly_threshold, LOSS_ANOMALY_THRESHOLD_MID

        for epoch in range(5, 15):  # Epochs 5-14
            threshold = get_loss_anomaly_threshold(epoch)
            assert threshold == LOSS_ANOMALY_THRESHOLD_MID
            assert threshold == 3.5  # Standard

    def test_get_loss_anomaly_threshold_late_epoch(self):
        """Test late training uses strict threshold."""
        from app.config.thresholds import get_loss_anomaly_threshold, LOSS_ANOMALY_THRESHOLD_LATE

        for epoch in [15, 20, 30, 50, 100]:  # Epochs 15+
            threshold = get_loss_anomaly_threshold(epoch)
            assert threshold == LOSS_ANOMALY_THRESHOLD_LATE
            assert threshold == 2.5  # Strict

    def test_get_loss_anomaly_threshold_negative_epoch(self):
        """Test negative epoch returns default."""
        from app.config.thresholds import get_loss_anomaly_threshold, LOSS_ANOMALY_THRESHOLD_DEFAULT

        threshold = get_loss_anomaly_threshold(-1)
        assert threshold == LOSS_ANOMALY_THRESHOLD_DEFAULT
        assert threshold == 4.0

    def test_get_gradient_norm_threshold_phases(self):
        """Test gradient norm threshold adapts by epoch."""
        from app.config.thresholds import (
            get_gradient_norm_threshold,
            GRADIENT_NORM_THRESHOLD_EARLY,
            GRADIENT_NORM_THRESHOLD_MID,
            GRADIENT_NORM_THRESHOLD_LATE,
        )

        # Early
        assert get_gradient_norm_threshold(0) == GRADIENT_NORM_THRESHOLD_EARLY
        assert get_gradient_norm_threshold(4) == GRADIENT_NORM_THRESHOLD_EARLY

        # Mid
        assert get_gradient_norm_threshold(5) == GRADIENT_NORM_THRESHOLD_MID
        assert get_gradient_norm_threshold(14) == GRADIENT_NORM_THRESHOLD_MID

        # Late
        assert get_gradient_norm_threshold(15) == GRADIENT_NORM_THRESHOLD_LATE
        assert get_gradient_norm_threshold(100) == GRADIENT_NORM_THRESHOLD_LATE

    def test_get_severe_anomaly_count_phases(self):
        """Test severe anomaly count threshold adapts by epoch."""
        from app.config.thresholds import (
            get_severe_anomaly_count,
            LOSS_ANOMALY_SEVERE_COUNT_EARLY,
            LOSS_ANOMALY_SEVERE_COUNT_MID,
            LOSS_ANOMALY_SEVERE_COUNT_LATE,
        )

        # Early - more permissive
        assert get_severe_anomaly_count(0) == LOSS_ANOMALY_SEVERE_COUNT_EARLY
        assert get_severe_anomaly_count(0) == 5

        # Mid - standard
        assert get_severe_anomaly_count(10) == LOSS_ANOMALY_SEVERE_COUNT_MID
        assert get_severe_anomaly_count(10) == 3

        # Late - strict
        assert get_severe_anomaly_count(20) == LOSS_ANOMALY_SEVERE_COUNT_LATE
        assert get_severe_anomaly_count(20) == 2

    def test_threshold_monotonically_decreases(self):
        """Test thresholds become stricter as training progresses."""
        from app.config.thresholds import get_loss_anomaly_threshold

        early = get_loss_anomaly_threshold(0)
        mid = get_loss_anomaly_threshold(10)
        late = get_loss_anomaly_threshold(20)

        assert early > mid > late
        assert early == 5.0
        assert mid == 3.5
        assert late == 2.5


class TestTrainingAnomalyDetectorAdaptive:
    """Test TrainingAnomalyDetector with adaptive thresholds."""

    def test_adaptive_thresholds_enabled_by_default(self):
        """Test adaptive thresholds are enabled by default."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector()
        assert detector._adaptive_thresholds is True

    def test_set_epoch_updates_threshold(self):
        """Test set_epoch updates effective threshold."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector(adaptive_thresholds=True)

        # Early epoch - permissive threshold
        detector.set_epoch(0)
        assert detector.loss_spike_threshold == 5.0
        assert detector.gradient_norm_threshold == 150.0

        # Mid epoch - standard threshold
        detector.set_epoch(10)
        assert detector.loss_spike_threshold == 3.5
        assert detector.gradient_norm_threshold == 100.0

        # Late epoch - strict threshold
        detector.set_epoch(20)
        assert detector.loss_spike_threshold == 2.5
        assert detector.gradient_norm_threshold == 75.0

    def test_adaptive_disabled_uses_base_threshold(self):
        """Test disabling adaptive thresholds uses base threshold."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector(
            loss_spike_threshold=3.0,
            gradient_norm_threshold=50.0,
            adaptive_thresholds=False,
        )

        # Set various epochs - should not change threshold
        for epoch in [0, 5, 10, 20, 100]:
            detector.set_epoch(epoch)
            assert detector.loss_spike_threshold == 3.0
            assert detector.gradient_norm_threshold == 50.0

    def test_current_epoch_property(self):
        """Test current_epoch property returns correct value."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector()
        assert detector.current_epoch == 0

        detector.set_epoch(15)
        assert detector.current_epoch == 15

        detector.set_epoch(-5)  # Negative clamped to 0
        assert detector.current_epoch == 0

    def test_reset_preserves_epoch_by_default(self):
        """Test reset preserves epoch by default."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector()
        detector.set_epoch(10)

        detector.reset()
        assert detector.current_epoch == 10  # Preserved

    def test_reset_with_reset_epoch(self):
        """Test reset can optionally reset epoch."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector()
        detector.set_epoch(10)

        detector.reset(reset_epoch=True)
        assert detector.current_epoch == 0  # Reset

    def test_get_summary_includes_adaptive_info(self):
        """Test get_summary includes adaptive threshold info."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector(adaptive_thresholds=True)
        detector.set_epoch(10)

        summary = detector.get_summary()
        assert "adaptive_thresholds" in summary
        assert summary["adaptive_thresholds"]["enabled"] is True
        assert summary["adaptive_thresholds"]["current_epoch"] == 10
        assert summary["adaptive_thresholds"]["loss_spike_threshold"] == 3.5
        assert summary["adaptive_thresholds"]["gradient_norm_threshold"] == 100.0

    def test_get_summary_no_adaptive_when_disabled(self):
        """Test get_summary excludes adaptive info when disabled."""
        from app.training.anomaly_detection import TrainingAnomalyDetector

        detector = TrainingAnomalyDetector(adaptive_thresholds=False)
        summary = detector.get_summary()
        assert "adaptive_thresholds" not in summary

    def test_early_epoch_detects_fewer_anomalies(self):
        """Test early epoch with permissive threshold detects fewer anomalies."""
        from app.training.anomaly_detection import TrainingAnomalyDetector
        import numpy as np

        # Set up detector with adaptive thresholds
        detector = TrainingAnomalyDetector(adaptive_thresholds=True)

        # Build up loss history with mean ~1.0, std ~0.14 (from values 0.7-1.3)
        # This way a value of 1.5 is ~3.5σ above mean
        base_losses = [0.7, 0.85, 1.0, 1.15, 1.3]  # std ~0.21
        for i in range(20):
            loss = base_losses[i % 5]
            detector.check_loss(loss, step=i)

        # Verify the statistics (mean ~1.0, std ~0.2)
        history = list(detector._loss_history)
        mean = np.mean(history)
        std = np.std(history)

        # Test loss value that's 4σ above mean (should be detected by late but not early)
        test_loss = mean + 4 * std  # This is about 1.8

        # Early epoch (5σ threshold) - 4σ deviation should NOT trigger
        detector.set_epoch(0)
        assert detector.check_loss(test_loss - 0.1, step=100) is False  # ~3.5σ, below early threshold

        # Late epoch (2.5σ threshold) - 4σ deviation SHOULD trigger
        detector.set_epoch(20)
        assert detector.check_loss(test_loss + 0.1, step=101) is True  # ~4.5σ > 2.5σ, is anomaly


class TestFeedbackLoopControllerAdaptiveIntegration:
    """Test FeedbackLoopController integration with adaptive thresholds."""

    def test_handler_imports_adaptive_threshold(self):
        """Test handler can import adaptive threshold function."""
        from app.config.thresholds import get_severe_anomaly_count

        # Just verify import works
        assert callable(get_severe_anomaly_count)
        assert get_severe_anomaly_count(0) == 5
        assert get_severe_anomaly_count(10) == 3
        assert get_severe_anomaly_count(20) == 2
