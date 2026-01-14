"""Unit tests for cluster deficit boost functionality.

Jan 2026: Tests for Phase 2 of Cluster Manifest Training Integration.
"""

from __future__ import annotations

import pytest


class TestComputeClusterDeficitBoost:
    """Tests for compute_cluster_deficit_boost function."""

    def test_high_cluster_count_no_boost(self):
        """Test that well-represented configs get no boost."""
        from app.coordination.priority_calculator import compute_cluster_deficit_boost

        boost = compute_cluster_deficit_boost(
            local_game_count=5000,
            cluster_game_count=15000,
        )
        assert boost == 1.0  # No boost for well-represented configs

    def test_medium_cluster_count_slight_boost(self):
        """Test that moderately represented configs get slight boost."""
        from app.coordination.priority_calculator import compute_cluster_deficit_boost

        boost = compute_cluster_deficit_boost(
            local_game_count=3000,
            cluster_game_count=7000,  # Between 5000 and 10000
        )
        assert boost == 1.2  # Slight boost

    def test_low_cluster_count_medium_boost(self):
        """Test that under-represented configs get medium boost."""
        from app.coordination.priority_calculator import compute_cluster_deficit_boost

        boost = compute_cluster_deficit_boost(
            local_game_count=1000,
            cluster_game_count=3000,  # Between 1000 and 5000
        )
        assert boost == 1.5  # Medium boost

    def test_very_low_cluster_count_high_boost(self):
        """Test that severely under-represented configs get high boost."""
        from app.coordination.priority_calculator import compute_cluster_deficit_boost

        boost = compute_cluster_deficit_boost(
            local_game_count=100,
            cluster_game_count=500,  # Below 1000
        )
        assert boost == 2.0  # High boost

    def test_zero_cluster_count_high_boost(self):
        """Test that configs with no cluster data get high boost."""
        from app.coordination.priority_calculator import compute_cluster_deficit_boost

        boost = compute_cluster_deficit_boost(
            local_game_count=0,
            cluster_game_count=0,
        )
        assert boost == 2.0  # High boost for bootstrapping

    def test_custom_thresholds(self):
        """Test custom thresholds override defaults."""
        from app.coordination.priority_calculator import compute_cluster_deficit_boost

        boost = compute_cluster_deficit_boost(
            local_game_count=100,
            cluster_game_count=2000,  # Would be 1.5x with defaults
            low_cluster_threshold=500,
            medium_cluster_threshold=3000,
            high_cluster_threshold=5000,
        )
        assert boost == 1.5  # Between custom low (500) and medium (3000)


class TestPriorityInputsClusterField:
    """Tests for cluster_game_count field in PriorityInputs."""

    def test_default_cluster_game_count(self):
        """Test that cluster_game_count defaults to 0."""
        from app.coordination.priority_calculator import PriorityInputs

        inputs = PriorityInputs(config_key="hex8_2p")
        assert inputs.cluster_game_count == 0

    def test_custom_cluster_game_count(self):
        """Test setting cluster_game_count."""
        from app.coordination.priority_calculator import PriorityInputs

        inputs = PriorityInputs(
            config_key="hex8_2p",
            cluster_game_count=10000,
        )
        assert inputs.cluster_game_count == 10000


class TestDynamicWeightsClusterField:
    """Tests for cluster weight in DynamicWeights."""

    def test_default_cluster_weight(self):
        """Test that cluster weight has correct default."""
        from app.coordination.priority_calculator import DynamicWeights

        weights = DynamicWeights()
        assert weights.cluster == 0.15  # Default weight

    def test_cluster_in_to_dict(self):
        """Test that cluster weight is included in to_dict()."""
        from app.coordination.priority_calculator import DynamicWeights

        weights = DynamicWeights(cluster=0.20)
        weights_dict = weights.to_dict()

        assert "cluster" in weights_dict
        assert weights_dict["cluster"] == 0.20


class TestPriorityCalculatorClusterIntegration:
    """Tests for cluster factor in priority calculation."""

    def test_cluster_factor_affects_score(self):
        """Test that cluster factor is included in priority score."""
        from app.coordination.priority_calculator import (
            PriorityCalculator,
            PriorityInputs,
            DynamicWeights,
        )

        # Use only cluster weight to isolate the effect
        weights = DynamicWeights(
            staleness=0.0,
            velocity=0.0,
            training=0.0,
            exploration=0.0,
            curriculum=0.0,
            improvement=0.0,
            data_deficit=0.0,
            quality=0.0,
            voi=0.0,
            diversity=0.0,
            cluster=1.0,  # Only cluster weight enabled
        )
        calculator = PriorityCalculator(dynamic_weights=weights)

        # Low cluster count = high boost (2.0x)
        inputs_low_cluster = PriorityInputs(
            config_key="hex8_2p",
            game_count=100,
            cluster_game_count=500,  # Low - will get 2.0x boost
        )

        # High cluster count = no boost (1.0x)
        inputs_high_cluster = PriorityInputs(
            config_key="hex8_2p",  # Same config for fair comparison
            game_count=100,
            cluster_game_count=15000,  # High - will get 1.0x boost
        )

        score_low = calculator.compute_priority_score(inputs_low_cluster)
        score_high = calculator.compute_priority_score(inputs_high_cluster)

        # Low cluster count config should have higher score (more priority)
        # cluster = (boost - 1.0) * weight
        # Low: (2.0 - 1.0) * 1.0 = 1.0 base, then exploration_boost (1.0) multiplied
        # High: (1.0 - 1.0) * 1.0 = 0.0
        # Final low score includes momentum/exploration multipliers
        assert score_low > score_high
        assert score_high == 0.0  # No cluster boost for well-represented configs
        assert score_low > 1.0  # Has cluster boost

    def test_zero_cluster_weight_disables_factor(self):
        """Test that zero cluster weight effectively disables the factor."""
        from app.coordination.priority_calculator import (
            PriorityCalculator,
            PriorityInputs,
            DynamicWeights,
        )

        # All weights zero to isolate effect
        weights = DynamicWeights(
            staleness=0.0,
            velocity=0.0,
            training=0.0,
            exploration=0.0,
            curriculum=0.0,
            improvement=0.0,
            data_deficit=0.0,
            quality=0.0,
            voi=0.0,
            diversity=0.0,
            cluster=0.0,  # Disabled
        )
        calculator = PriorityCalculator(dynamic_weights=weights)

        inputs_low = PriorityInputs(
            config_key="hex8_2p",
            cluster_game_count=100,
        )
        inputs_high = PriorityInputs(
            config_key="hex8_2p",  # Same config for fair comparison
            cluster_game_count=15000,
        )

        score_low = calculator.compute_priority_score(inputs_low)
        score_high = calculator.compute_priority_score(inputs_high)

        # With cluster weight = 0, scores should be equal
        assert score_low == score_high
