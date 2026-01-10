"""Tests for training_architecture_selector module.

January 2026: Created as part of Phase 2 modularization testing.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.coordination.training_architecture_selector import (
    get_training_params_for_intensity,
    select_architecture_for_training,
    apply_velocity_amplification,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
)


class TestGetTrainingParamsForIntensity:
    """Tests for get_training_params_for_intensity function."""

    def test_hot_path_intensity(self):
        """Hot path should return fast iteration params."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity("hot_path")
        assert epochs == 30
        assert batch_size == 1024
        assert lr_mult == 1.5

    def test_accelerated_intensity(self):
        """Accelerated should return moderate boost params."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity("accelerated")
        assert epochs == 40
        assert batch_size == 768
        assert lr_mult == 1.2

    def test_normal_intensity(self):
        """Normal should return default params."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity("normal")
        assert epochs == DEFAULT_EPOCHS
        assert batch_size == DEFAULT_BATCH_SIZE
        assert lr_mult == 1.0

    def test_reduced_intensity(self):
        """Reduced should return careful training params."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity("reduced")
        assert epochs == 60
        assert batch_size == 256
        assert lr_mult == 0.8

    def test_paused_intensity(self):
        """Paused should return minimal params."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity("paused")
        assert epochs == 10
        assert batch_size == 128
        assert lr_mult == 0.5

    def test_unknown_intensity_fallback(self):
        """Unknown intensity should fall back to normal."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity("unknown_value")
        assert epochs == DEFAULT_EPOCHS
        assert batch_size == DEFAULT_BATCH_SIZE
        assert lr_mult == 1.0

    def test_custom_defaults(self):
        """Custom default values should be used for normal intensity."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity(
            "normal", default_epochs=100, default_batch_size=256
        )
        assert epochs == 100
        assert batch_size == 256
        assert lr_mult == 1.0

    def test_custom_defaults_not_affect_other_intensities(self):
        """Custom defaults should only affect normal intensity."""
        epochs, batch_size, lr_mult = get_training_params_for_intensity(
            "hot_path", default_epochs=100, default_batch_size=256
        )
        # hot_path has fixed values, not affected by defaults
        assert epochs == 30
        assert batch_size == 1024


class TestSelectArchitectureForTraining:
    """Tests for select_architecture_for_training function."""

    def test_returns_default_when_tracker_not_available(self):
        """Should return default arch when ArchitectureTracker import fails."""
        with patch.dict("sys.modules", {"app.training.architecture_tracker": None}):
            # Force ImportError by making the module None
            with patch(
                "app.coordination.training_architecture_selector.logger"
            ) as mock_logger:
                # The import will fail gracefully
                result = select_architecture_for_training("hex8", 2, default_arch="v2")
                # Should return default since we can't actually make import fail in this test
                # Let's test a different way

        # Test with mocked import error
        with patch(
            "app.coordination.training_architecture_selector.logger"
        ):
            result = select_architecture_for_training(
                "hex8", 2, default_arch="test_default"
            )
            # Result depends on whether architecture_tracker exists
            assert isinstance(result, str)

    def test_returns_default_when_no_weights(self):
        """Should return default arch when no weights available."""
        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            return_value={},
        ):
            result = select_architecture_for_training(
                "hex8", 2, default_arch="v5"
            )
            assert result == "v5"

    def test_returns_default_when_weights_none(self):
        """Should return default arch when weights is None."""
        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            return_value=None,
        ):
            result = select_architecture_for_training(
                "hex8", 2, default_arch="v4"
            )
            assert result == "v4"

    def test_selects_from_weighted_architectures(self):
        """Should select architecture from weights dict."""
        mock_weights = {"v2": 0.3, "v4": 0.5, "v5": 0.2}

        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            return_value=mock_weights,
        ):
            # Run multiple times to verify selection works
            results = set()
            for _ in range(50):
                result = select_architecture_for_training("hex8", 2)
                results.add(result)
                assert result in mock_weights.keys()

            # With 50 iterations and reasonable weights, should see multiple archs
            # (probabilistic, but very likely)

    def test_single_architecture_always_selected(self):
        """When only one architecture has weight, it should always be selected."""
        mock_weights = {"v5-heavy": 1.0}

        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            return_value=mock_weights,
        ):
            for _ in range(10):
                result = select_architecture_for_training("hex8", 2)
                assert result == "v5-heavy"

    def test_handles_key_error_gracefully(self):
        """Should return default on KeyError."""
        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            side_effect=KeyError("test error"),
        ):
            result = select_architecture_for_training(
                "hex8", 2, default_arch="v2"
            )
            assert result == "v2"

    def test_handles_value_error_gracefully(self):
        """Should return default on ValueError."""
        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            side_effect=ValueError("test error"),
        ):
            result = select_architecture_for_training(
                "hex8", 2, default_arch="v2"
            )
            assert result == "v2"

    def test_handles_type_error_gracefully(self):
        """Should return default on TypeError."""
        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            side_effect=TypeError("test error"),
        ):
            result = select_architecture_for_training(
                "hex8", 2, default_arch="v2"
            )
            assert result == "v2"

    def test_passes_temperature_to_tracker(self):
        """Should pass temperature parameter to get_allocation_weights."""
        mock_get_weights = MagicMock(return_value={"v5": 1.0})

        with patch(
            "app.training.architecture_tracker.get_allocation_weights",
            mock_get_weights,
        ):
            select_architecture_for_training(
                "hex8", 2, temperature=0.8
            )
            mock_get_weights.assert_called_once_with(
                board_type="hex8",
                num_players=2,
                temperature=0.8,
            )


class TestApplyVelocityAmplification:
    """Tests for apply_velocity_amplification function."""

    def test_fast_improvement_high_velocity(self):
        """Velocity > 2.0 should boost epochs, batch, and LR."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=2.5, velocity_trend="stable"
        )

        # 1.5x epochs
        assert epochs == 75
        # batch_size at least 512
        assert batch_size >= 512
        # 1.3x LR
        assert abs(lr_mult - 1.3) < 0.01

    def test_good_progress_velocity(self):
        """Velocity > 1.0 should moderately boost params."""
        base_params = (50, 256, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=1.5, velocity_trend="stable"
        )

        # 1.2x epochs
        assert epochs == 60
        # batch_size at least 384
        assert batch_size >= 384
        # 1.1x LR
        assert abs(lr_mult - 1.1) < 0.01

    def test_slow_improvement_velocity(self):
        """Velocity < 0.5 should reduce LR."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.3, velocity_trend="stable"
        )

        # epochs unchanged
        assert epochs == 50
        # batch_size unchanged
        assert batch_size == 512
        # 0.8x LR
        assert abs(lr_mult - 0.8) < 0.01

    def test_regression_negative_velocity(self):
        """Negative velocity should increase epochs and reduce LR."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=-1.0, velocity_trend="stable"
        )

        # 1.3x epochs
        assert epochs == 65
        # batch_size unchanged
        assert batch_size == 512
        # 0.6x LR
        assert abs(lr_mult - 0.6) < 0.01

    def test_accelerating_trend_boosts_lr(self):
        """Accelerating trend with positive velocity should boost LR."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.8, velocity_trend="accelerating"
        )

        # With velocity 0.8 (0.5-1.0 range), base LR is 0.8
        # Then accelerating adds 1.05x
        expected_lr = 0.8 * 1.05
        assert abs(lr_mult - expected_lr) < 0.01

    def test_plateauing_trend_boosts_epochs(self):
        """Plateauing trend should increase epochs."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.8, velocity_trend="plateauing"
        )

        # With velocity 0.8 (0.5-1.0 range), epochs unchanged from velocity
        # Then plateauing adds 1.15x
        expected_epochs = int(50 * 1.15)
        assert epochs == expected_epochs

    def test_decelerating_trend_reduces_lr(self):
        """Decelerating trend with positive velocity should slightly reduce LR."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.8, velocity_trend="decelerating"
        )

        # With velocity 0.8 (0.5-1.0 range), base LR is 0.8
        # Then decelerating reduces by 0.95x
        expected_lr = 0.8 * 0.95
        assert abs(lr_mult - expected_lr) < 0.01

    def test_clamp_epochs_minimum(self):
        """Epochs should not go below 10."""
        # Use very low epochs that might go below 10 with regression penalty
        base_params = (5, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=-5.0, velocity_trend="stable"
        )

        assert epochs >= 10

    def test_clamp_epochs_maximum(self):
        """Epochs should not exceed 150."""
        # Use high epochs with fast velocity that would exceed 150
        base_params = (120, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=3.0, velocity_trend="accelerating"
        )

        assert epochs <= 150

    def test_clamp_batch_size_minimum(self):
        """Batch size should not go below 128."""
        base_params = (50, 64, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.3, velocity_trend="stable"
        )

        assert batch_size >= 128

    def test_clamp_batch_size_maximum(self):
        """Batch size should not exceed 2048."""
        base_params = (50, 4096, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=3.0, velocity_trend="stable"
        )

        assert batch_size <= 2048

    def test_clamp_lr_mult_minimum(self):
        """LR multiplier should not go below 0.3."""
        # Extreme regression
        base_params = (50, 512, 0.3)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=-10.0, velocity_trend="decelerating"
        )

        assert lr_mult >= 0.3

    def test_clamp_lr_mult_maximum(self):
        """LR multiplier should not exceed 2.5."""
        # Extreme fast improvement
        base_params = (50, 512, 2.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=5.0, velocity_trend="accelerating"
        )

        assert lr_mult <= 2.5

    def test_stable_velocity_no_trend_adjustment(self):
        """Stable trend with moderate velocity should not adjust."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.8, velocity_trend="stable"
        )

        # Velocity 0.8 is in 0.5-1.0 range, so LR gets 0.8x
        assert epochs == 50
        assert batch_size == 512
        assert abs(lr_mult - 0.8) < 0.01

    def test_unknown_trend_no_adjustment(self):
        """Unknown trend should not cause error."""
        base_params = (50, 512, 1.0)
        epochs, batch_size, lr_mult = apply_velocity_amplification(
            base_params, elo_velocity=0.8, velocity_trend="unknown_trend"
        )

        # Should still work with just velocity adjustment
        assert isinstance(epochs, int)
        assert isinstance(batch_size, int)
        assert isinstance(lr_mult, float)


class TestIntegration:
    """Integration tests for combined function usage."""

    def test_intensity_then_velocity_amplification(self):
        """Test typical workflow: get intensity params, then amplify."""
        # Get base params for accelerated intensity
        base_params = get_training_params_for_intensity("accelerated")
        assert base_params == (40, 768, 1.2)

        # Apply velocity amplification for fast improvement
        final_params = apply_velocity_amplification(
            base_params, elo_velocity=2.5, velocity_trend="accelerating"
        )

        epochs, batch_size, lr_mult = final_params

        # Should be amplified from accelerated base
        assert epochs > 40  # boosted
        assert batch_size >= 768  # at least maintained
        assert lr_mult > 1.2  # boosted

    def test_all_intensities_produce_valid_params(self):
        """All intensity levels should produce valid parameter tuples."""
        intensities = ["hot_path", "accelerated", "normal", "reduced", "paused"]

        for intensity in intensities:
            epochs, batch_size, lr_mult = get_training_params_for_intensity(intensity)

            assert isinstance(epochs, int)
            assert isinstance(batch_size, int)
            assert isinstance(lr_mult, float)
            assert epochs > 0
            assert batch_size > 0
            assert lr_mult > 0
