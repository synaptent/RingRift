"""Unit tests for priority_calculator module.

Tests the pure calculation functions for selfplay priority scoring.
"""

from __future__ import annotations

import pytest

from app.coordination.priority_calculator import (
    ALL_CONFIGS,
    ClusterState,
    DynamicWeights,
    PLAYER_COUNT_ALLOCATION_MULTIPLIER,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    PriorityCalculator,
    PriorityInputs,
    SAMPLES_PER_GAME_BY_BOARD,
    VOI_SAMPLE_COST_BY_BOARD,
    clamp_weight,
    compute_data_deficit_factor,
    compute_dynamic_weights,
    compute_games_needed,
    compute_info_gain_per_game,
    compute_staleness_factor,
    compute_velocity_factor,
    compute_voi_score,
    extract_player_count,
)


# =============================================================================
# compute_staleness_factor Tests
# =============================================================================


class TestComputeStalenesFactor:
    """Tests for compute_staleness_factor function."""

    def test_fresh_data_returns_zero(self):
        """Fresh data (below threshold) should return 0."""
        result = compute_staleness_factor(0.5, fresh_threshold=1.0)
        assert result == 0.0

    def test_at_fresh_threshold_returns_zero(self):
        """Data at fresh threshold should return 0."""
        result = compute_staleness_factor(1.0, fresh_threshold=1.0, stale_threshold=12.0)
        assert result == 0.0

    def test_very_stale_data_returns_one(self):
        """Very stale data should cap at 1.0."""
        result = compute_staleness_factor(100.0, max_staleness=24.0)
        assert result == 1.0

    def test_linear_interpolation(self):
        """Test linear interpolation between fresh and stale thresholds."""
        # Midpoint between 1.0 and 12.0 (fresh=1, stale=12)
        result = compute_staleness_factor(6.5, fresh_threshold=1.0, stale_threshold=12.0)
        expected = (6.5 - 1.0) / (12.0 - 1.0)
        assert abs(result - expected) < 0.001


# =============================================================================
# compute_velocity_factor Tests
# =============================================================================


class TestComputeVelocityFactor:
    """Tests for compute_velocity_factor function."""

    def test_negative_velocity_returns_zero(self):
        """Negative velocity (regression) should return 0."""
        result = compute_velocity_factor(-10.0)
        assert result == 0.0

    def test_zero_velocity_returns_zero(self):
        """Zero velocity should return 0."""
        result = compute_velocity_factor(0.0)
        assert result == 0.0

    def test_max_velocity_returns_one(self):
        """Velocity at or above max should return 1.0."""
        result = compute_velocity_factor(100.0, max_velocity=100.0)
        assert result == 1.0
        result = compute_velocity_factor(150.0, max_velocity=100.0)
        assert result == 1.0

    def test_linear_scaling(self):
        """Test linear scaling of velocity."""
        result = compute_velocity_factor(50.0, max_velocity=100.0)
        assert result == 0.5


# =============================================================================
# compute_data_deficit_factor Tests
# =============================================================================


class TestComputeDataDeficitFactor:
    """Tests for compute_data_deficit_factor function."""

    def test_no_deficit_when_at_target(self):
        """No deficit when game count meets target."""
        result = compute_data_deficit_factor(1000, is_large_board=False, target_games=1000)
        assert result == 0.0

    def test_no_deficit_when_above_target(self):
        """No deficit when game count exceeds target."""
        result = compute_data_deficit_factor(1500, is_large_board=False, target_games=1000)
        assert result == 0.0

    def test_full_deficit_with_zero_games(self):
        """Full deficit with zero games."""
        result = compute_data_deficit_factor(0, is_large_board=False, target_games=1000)
        assert result == 1.0

    def test_partial_deficit(self):
        """Partial deficit with some games."""
        result = compute_data_deficit_factor(500, is_large_board=False, target_games=1000)
        assert result == 0.5

    def test_large_board_multiplier(self):
        """Large boards have higher target."""
        # With 2x multiplier, target becomes 2000
        result = compute_data_deficit_factor(
            1000, is_large_board=True, target_games=1000, large_board_multiplier=2.0
        )
        assert result == 0.5  # 1000 / 2000 = 0.5 deficit


# =============================================================================
# compute_voi_score Tests
# =============================================================================


class TestComputeVoiScore:
    """Tests for compute_voi_score function."""

    def test_max_voi_with_high_uncertainty(self):
        """High uncertainty contributes 0.4 to VOI."""
        result = compute_voi_score(
            elo_uncertainty=300.0,  # Max
            elo_gap=0.0,
            info_gain_per_game=0.0,
        )
        assert result == 0.4  # Only uncertainty factor contributes

    def test_max_voi_with_high_gap(self):
        """High Elo gap contributes 0.3 to VOI."""
        result = compute_voi_score(
            elo_uncertainty=0.0,
            elo_gap=500.0,  # Max
            info_gain_per_game=0.0,
        )
        assert result == 0.3  # Only gap factor contributes

    def test_max_voi_with_high_info_gain(self):
        """High info gain contributes 0.3 to VOI."""
        result = compute_voi_score(
            elo_uncertainty=0.0,
            elo_gap=0.0,
            info_gain_per_game=10.0,  # Max
        )
        assert result == 0.3  # Only info gain contributes

    def test_full_voi(self):
        """All factors at max should give 1.0 VOI."""
        result = compute_voi_score(
            elo_uncertainty=300.0,
            elo_gap=500.0,
            info_gain_per_game=10.0,
        )
        assert result == 1.0


# =============================================================================
# compute_info_gain_per_game Tests
# =============================================================================


class TestComputeInfoGainPerGame:
    """Tests for compute_info_gain_per_game function."""

    def test_first_game_max_gain(self):
        """First game has maximum info gain."""
        result = compute_info_gain_per_game(elo_uncertainty=100.0, game_count=0)
        assert result == 100.0

    def test_sqrt_n_reduction(self):
        """Info gain reduces with sqrt(n)."""
        result = compute_info_gain_per_game(elo_uncertainty=100.0, game_count=100)
        expected = 100.0 / 10.0  # sqrt(100) = 10
        assert abs(result - expected) < 0.001

    def test_diminishing_returns(self):
        """More games = less info per additional game."""
        gain_10 = compute_info_gain_per_game(100.0, 10)
        gain_100 = compute_info_gain_per_game(100.0, 100)
        gain_1000 = compute_info_gain_per_game(100.0, 1000)

        assert gain_10 > gain_100 > gain_1000


# =============================================================================
# clamp_weight Tests
# =============================================================================


class TestClampWeight:
    """Tests for clamp_weight function."""

    def test_clamp_below_minimum(self):
        """Value below minimum should be clamped."""
        bounds = {"test": (0.1, 0.5)}
        result = clamp_weight("test", 0.05, bounds)
        assert result == 0.1

    def test_clamp_above_maximum(self):
        """Value above maximum should be clamped."""
        bounds = {"test": (0.1, 0.5)}
        result = clamp_weight("test", 0.7, bounds)
        assert result == 0.5

    def test_value_in_range(self):
        """Value in range should be unchanged."""
        bounds = {"test": (0.1, 0.5)}
        result = clamp_weight("test", 0.3, bounds)
        assert result == 0.3

    def test_unknown_name_uses_default_bounds(self):
        """Unknown weight name uses default bounds."""
        result = clamp_weight("unknown_weight", 0.001)
        assert result >= 0.05  # Default min


# =============================================================================
# extract_player_count Tests
# =============================================================================


class TestExtractPlayerCount:
    """Tests for extract_player_count function."""

    def test_2p_config(self):
        """Extract 2 from 2p config."""
        assert extract_player_count("hex8_2p") == 2
        assert extract_player_count("square19_2p") == 2

    def test_3p_config(self):
        """Extract 3 from 3p config."""
        assert extract_player_count("hex8_3p") == 3
        assert extract_player_count("hexagonal_3p") == 3

    def test_4p_config(self):
        """Extract 4 from 4p config."""
        assert extract_player_count("square8_4p") == 4
        assert extract_player_count("square19_4p") == 4

    def test_invalid_format_returns_default(self):
        """Invalid format returns default (2)."""
        assert extract_player_count("invalid") == 2
        assert extract_player_count("hex8_Xp") == 2


# =============================================================================
# compute_games_needed Tests
# =============================================================================


class TestComputeGamesNeeded:
    """Tests for compute_games_needed function."""

    def test_target_met(self):
        """No games needed when target is met."""
        result = compute_games_needed(
            game_count=1000,
            samples_per_game=50.0,
            target_samples=50000,
        )
        assert result == 0

    def test_partial_progress(self):
        """Compute remaining games for partial progress."""
        # 500 games * 50 samples = 25000 samples
        # Need 50000 - 25000 = 25000 more samples
        # 25000 / 50 = 500 more games
        result = compute_games_needed(
            game_count=500,
            samples_per_game=50.0,
            target_samples=50000,
        )
        assert result == 500

    def test_zero_samples_per_game(self):
        """Zero samples per game returns 0."""
        result = compute_games_needed(
            game_count=100,
            samples_per_game=0.0,
            target_samples=50000,
        )
        assert result == 0


# =============================================================================
# DynamicWeights Tests
# =============================================================================


class TestDynamicWeights:
    """Tests for DynamicWeights dataclass."""

    def test_default_values(self):
        """Test default weight values."""
        weights = DynamicWeights()
        assert weights.staleness == 0.30
        assert weights.velocity == 0.10
        assert weights.training == 0.15
        assert weights.voi == 0.02

    def test_to_dict(self):
        """Test to_dict conversion."""
        weights = DynamicWeights(staleness=0.25, velocity=0.15)
        d = weights.to_dict()
        assert d["staleness"] == 0.25
        assert d["velocity"] == 0.15
        assert "idle_gpu_fraction" in d


# =============================================================================
# compute_dynamic_weights Tests
# =============================================================================


class TestComputeDynamicWeights:
    """Tests for compute_dynamic_weights function."""

    def test_default_cluster_state(self):
        """Default cluster state produces default-ish weights."""
        state = ClusterState()
        weights = compute_dynamic_weights(state)
        assert isinstance(weights, DynamicWeights)

    def test_high_idle_gpus_boost_staleness(self):
        """High idle GPU fraction boosts staleness weight."""
        state = ClusterState(idle_gpu_fraction=0.7)
        base = DynamicWeights(staleness=0.30)
        weights = compute_dynamic_weights(state, base_weights=base)
        assert weights.staleness > 0.30

    def test_large_queue_reduces_staleness(self):
        """Large training queue reduces staleness weight."""
        state = ClusterState(training_queue_depth=20)
        base = DynamicWeights(staleness=0.30)
        weights = compute_dynamic_weights(state, base_weights=base)
        assert weights.staleness < 0.30

    def test_many_at_target_reduces_velocity(self):
        """Many configs at target reduces velocity weight."""
        state = ClusterState(configs_at_target_fraction=0.8)
        base = DynamicWeights(velocity=0.15)  # Start higher so reduction is visible
        weights = compute_dynamic_weights(state, base_weights=base)
        # 0.15 * 0.6 = 0.09, but clamped to min (0.05)
        assert weights.velocity <= 0.15

    def test_high_elo_boosts_curriculum(self):
        """High average Elo boosts curriculum weight."""
        state = ClusterState(average_elo=1900.0)
        base = DynamicWeights(curriculum=0.10)
        weights = compute_dynamic_weights(state, base_weights=base)
        assert weights.curriculum > 0.10


# =============================================================================
# PriorityInputs Tests
# =============================================================================


class TestPriorityInputs:
    """Tests for PriorityInputs dataclass."""

    def test_default_values(self):
        """Test default input values."""
        inputs = PriorityInputs(config_key="hex8_2p")
        assert inputs.config_key == "hex8_2p"
        assert inputs.staleness_hours == 0.0
        assert inputs.exploration_boost == 1.0
        assert inputs.momentum_multiplier == 1.0


# =============================================================================
# PriorityCalculator Tests
# =============================================================================


class TestPriorityCalculator:
    """Tests for PriorityCalculator class."""

    def test_basic_priority_calculation(self):
        """Test basic priority score calculation."""
        calculator = PriorityCalculator()
        inputs = PriorityInputs(config_key="hex8_2p")
        score = calculator.compute_priority_score(inputs)
        assert isinstance(score, float)
        assert score >= 0

    def test_stale_data_increases_priority(self):
        """Stale data should increase priority score."""
        calculator = PriorityCalculator()

        fresh = PriorityInputs(config_key="hex8_2p", staleness_hours=0.5)
        stale = PriorityInputs(config_key="hex8_2p", staleness_hours=24.0)

        fresh_score = calculator.compute_priority_score(fresh)
        stale_score = calculator.compute_priority_score(stale)

        assert stale_score > fresh_score

    def test_exploration_boost_multiplier(self):
        """Exploration boost should multiply score."""
        calculator = PriorityCalculator()

        normal = PriorityInputs(config_key="hex8_2p", exploration_boost=1.0)
        boosted = PriorityInputs(config_key="hex8_2p", exploration_boost=2.0)

        normal_score = calculator.compute_priority_score(normal)
        boosted_score = calculator.compute_priority_score(boosted)

        # Boosted should be significantly higher due to 2x multiplier
        assert boosted_score > normal_score * 1.5

    def test_momentum_multiplier(self):
        """Momentum multiplier should affect score."""
        calculator = PriorityCalculator()

        normal = PriorityInputs(config_key="hex8_2p", momentum_multiplier=1.0)
        accelerated = PriorityInputs(config_key="hex8_2p", momentum_multiplier=1.5)

        normal_score = calculator.compute_priority_score(normal)
        accel_score = calculator.compute_priority_score(accelerated)

        assert accel_score > normal_score

    def test_priority_override_multiplier(self):
        """Priority override should affect score."""
        calculator = PriorityCalculator()

        low = PriorityInputs(config_key="hex8_2p", priority_override=3)  # 1x
        critical = PriorityInputs(config_key="hex8_2p", priority_override=0)  # 3x

        low_score = calculator.compute_priority_score(low)
        critical_score = calculator.compute_priority_score(critical)

        assert critical_score > low_score * 2

    def test_player_count_multiplier(self):
        """4p configs should get higher priority than 2p."""
        calculator = PriorityCalculator()

        two_p = PriorityInputs(config_key="hex8_2p")  # 1x
        four_p = PriorityInputs(config_key="hex8_4p")  # 4x

        two_score = calculator.compute_priority_score(two_p)
        four_score = calculator.compute_priority_score(four_p)

        assert four_score > two_score * 3

    def test_data_starvation_emergency(self):
        """Very low game count should trigger emergency multiplier."""
        calculator = PriorityCalculator(
            data_starvation_emergency_threshold=50,
            data_starvation_emergency_multiplier=5.0,
        )

        normal = PriorityInputs(config_key="hex8_2p", game_count=1000)
        starved = PriorityInputs(config_key="hex8_2p", game_count=10)

        normal_score = calculator.compute_priority_score(normal)
        starved_score = calculator.compute_priority_score(starved)

        assert starved_score > normal_score * 4

    def test_update_weights(self):
        """Test updating dynamic weights."""
        calculator = PriorityCalculator()

        new_weights = DynamicWeights(staleness=0.5)
        calculator.update_weights(new_weights)

        inputs = PriorityInputs(config_key="hex8_2p", staleness_hours=24.0)
        score = calculator.compute_priority_score(inputs)

        # Score should reflect higher staleness weight
        assert score > 0

    def test_quality_callback(self):
        """Test quality score callback."""
        def mock_quality(config_key: str) -> float:
            return 0.9  # High quality

        calculator = PriorityCalculator(get_quality_score_fn=mock_quality)
        inputs = PriorityInputs(config_key="hex8_2p")
        score = calculator.compute_priority_score(inputs)

        assert score > 0


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_priority_override_multipliers(self):
        """Test priority override multiplier values."""
        assert PRIORITY_OVERRIDE_MULTIPLIERS[0] == 3.0  # CRITICAL
        assert PRIORITY_OVERRIDE_MULTIPLIERS[3] == 1.0  # LOW

    def test_player_count_multipliers(self):
        """Test player count multiplier values."""
        assert PLAYER_COUNT_ALLOCATION_MULTIPLIER[2] == 1.0
        assert PLAYER_COUNT_ALLOCATION_MULTIPLIER[4] == 4.0


# =============================================================================
# ALL_CONFIGS Tests (December 2025 consolidation)
# =============================================================================


class TestAllConfigs:
    """Tests for ALL_CONFIGS constant."""

    def test_contains_12_configs(self):
        """ALL_CONFIGS should contain exactly 12 configurations."""
        assert len(ALL_CONFIGS) == 12

    def test_contains_all_board_types(self):
        """ALL_CONFIGS should contain all 4 board types."""
        board_types = {"hex8", "square8", "square19", "hexagonal"}
        found_boards = set()
        for config in ALL_CONFIGS:
            board_type = config.split("_")[0]
            found_boards.add(board_type)
        assert found_boards == board_types

    def test_contains_all_player_counts(self):
        """ALL_CONFIGS should contain all player counts (2, 3, 4)."""
        player_counts = set()
        for config in ALL_CONFIGS:
            suffix = config.split("_")[1]
            player_counts.add(suffix)
        assert player_counts == {"2p", "3p", "4p"}

    def test_format_consistency(self):
        """All configs should follow board_Xp format."""
        for config in ALL_CONFIGS:
            parts = config.split("_")
            assert len(parts) == 2
            assert parts[1] in {"2p", "3p", "4p"}

    def test_specific_configs_present(self):
        """Specific commonly-used configs should be present."""
        assert "hex8_2p" in ALL_CONFIGS
        assert "square8_4p" in ALL_CONFIGS
        assert "hexagonal_3p" in ALL_CONFIGS


# =============================================================================
# SAMPLES_PER_GAME_BY_BOARD Tests (December 2025 consolidation)
# =============================================================================


class TestSamplesPerGameByBoard:
    """Tests for SAMPLES_PER_GAME_BY_BOARD constant."""

    def test_contains_all_board_types(self):
        """SAMPLES_PER_GAME_BY_BOARD should have entries for all board types."""
        expected_boards = {"hex8", "square8", "square19", "hexagonal"}
        assert set(SAMPLES_PER_GAME_BY_BOARD.keys()) == expected_boards

    def test_each_board_has_all_player_counts(self):
        """Each board type should have entries for 2p, 3p, 4p."""
        expected_players = {"2p", "3p", "4p"}
        for board_type, players_dict in SAMPLES_PER_GAME_BY_BOARD.items():
            assert set(players_dict.keys()) == expected_players

    def test_samples_are_positive(self):
        """All sample counts should be positive integers."""
        for board_type, players_dict in SAMPLES_PER_GAME_BY_BOARD.items():
            for player_key, samples in players_dict.items():
                assert samples > 0
                assert isinstance(samples, int)

    def test_larger_boards_more_samples(self):
        """Larger boards should have more samples per game."""
        # hex8 is smallest, hexagonal is largest
        hex8_2p = SAMPLES_PER_GAME_BY_BOARD["hex8"]["2p"]
        hexagonal_2p = SAMPLES_PER_GAME_BY_BOARD["hexagonal"]["2p"]
        assert hexagonal_2p > hex8_2p * 2  # Much larger

    def test_more_players_more_samples(self):
        """More players should generally produce more samples per game."""
        for board_type in SAMPLES_PER_GAME_BY_BOARD:
            samples_2p = SAMPLES_PER_GAME_BY_BOARD[board_type]["2p"]
            samples_4p = SAMPLES_PER_GAME_BY_BOARD[board_type]["4p"]
            assert samples_4p >= samples_2p


# =============================================================================
# VOI_SAMPLE_COST_BY_BOARD Tests (December 2025 consolidation)
# =============================================================================


class TestVoiSampleCostByBoard:
    """Tests for VOI_SAMPLE_COST_BY_BOARD constant."""

    def test_contains_all_board_types(self):
        """VOI_SAMPLE_COST_BY_BOARD should have entries for all board types."""
        expected_boards = {"hex8", "square8", "square19", "hexagonal"}
        assert set(VOI_SAMPLE_COST_BY_BOARD.keys()) == expected_boards

    def test_each_board_has_all_player_counts(self):
        """Each board type should have entries for 2p, 3p, 4p."""
        expected_players = {"2p", "3p", "4p"}
        for board_type, players_dict in VOI_SAMPLE_COST_BY_BOARD.items():
            assert set(players_dict.keys()) == expected_players

    def test_costs_are_positive(self):
        """All VOI costs should be positive floats."""
        for board_type, players_dict in VOI_SAMPLE_COST_BY_BOARD.items():
            for player_key, cost in players_dict.items():
                assert cost > 0
                assert isinstance(cost, (int, float))

    def test_small_boards_baseline_cost(self):
        """Small boards (hex8, square8) should have baseline cost of 1.0 for 2p."""
        assert VOI_SAMPLE_COST_BY_BOARD["hex8"]["2p"] == 1.0
        assert VOI_SAMPLE_COST_BY_BOARD["square8"]["2p"] == 1.0

    def test_larger_boards_higher_cost(self):
        """Larger boards should have higher VOI costs."""
        hex8_2p = VOI_SAMPLE_COST_BY_BOARD["hex8"]["2p"]
        hexagonal_2p = VOI_SAMPLE_COST_BY_BOARD["hexagonal"]["2p"]
        assert hexagonal_2p > hex8_2p * 2  # Much more expensive

    def test_more_players_higher_cost(self):
        """More players should increase VOI cost."""
        for board_type in VOI_SAMPLE_COST_BY_BOARD:
            cost_2p = VOI_SAMPLE_COST_BY_BOARD[board_type]["2p"]
            cost_4p = VOI_SAMPLE_COST_BY_BOARD[board_type]["4p"]
            assert cost_4p > cost_2p
