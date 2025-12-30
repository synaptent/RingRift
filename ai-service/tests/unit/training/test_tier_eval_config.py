"""Unit tests for tier evaluation configuration module.

Tests the tier evaluation profiles used for AI difficulty promotion gating.
This module is critical for the model promotion pipeline (used by 17 files).
"""

from __future__ import annotations

import pytest

from app.models import AIType, BoardType
from app.training.tier_eval_config import (
    HEURISTIC_TIER_SPECS,
    TIER_EVAL_CONFIGS,
    HeuristicTierSpec,
    TierEvaluationConfig,
    TierOpponentConfig,
    TierRole,
    get_tier_config,
)


class TestTierOpponentConfig:
    """Tests for TierOpponentConfig dataclass."""

    def test_create_basic_opponent(self):
        """Test creating a basic opponent config."""
        opponent = TierOpponentConfig(
            id="random_d1",
            description="Random baseline",
            difficulty=1,
        )
        assert opponent.id == "random_d1"
        assert opponent.description == "Random baseline"
        assert opponent.difficulty == 1
        assert opponent.ai_type is None
        assert opponent.role == "baseline"
        assert opponent.weight == 1.0
        assert opponent.games is None

    def test_create_opponent_with_all_fields(self):
        """Test creating opponent with all fields specified."""
        opponent = TierOpponentConfig(
            id="heuristic_d3",
            description="Heuristic at difficulty 3",
            difficulty=3,
            ai_type=AIType.HEURISTIC,
            role="previous_tier",
            weight=0.5,
            games=100,
        )
        assert opponent.id == "heuristic_d3"
        assert opponent.ai_type == AIType.HEURISTIC
        assert opponent.role == "previous_tier"
        assert opponent.weight == 0.5
        assert opponent.games == 100

    def test_opponent_is_frozen(self):
        """Test that opponent config is immutable."""
        opponent = TierOpponentConfig(
            id="test",
            description="Test",
            difficulty=1,
        )
        with pytest.raises(AttributeError):
            opponent.id = "modified"  # type: ignore[misc]

    def test_opponent_roles(self):
        """Test all valid opponent roles."""
        valid_roles: list[TierRole] = ["baseline", "previous_tier", "peer", "other"]
        for role in valid_roles:
            opponent = TierOpponentConfig(
                id=f"test_{role}",
                description=f"Test {role}",
                difficulty=1,
                role=role,
            )
            assert opponent.role == role

    def test_opponent_with_random_ai_type(self):
        """Test opponent with RANDOM AI type."""
        opponent = TierOpponentConfig(
            id="random",
            description="Random opponent",
            difficulty=1,
            ai_type=AIType.RANDOM,
        )
        assert opponent.ai_type == AIType.RANDOM

    def test_opponent_with_mcts_ai_type(self):
        """Test opponent with MCTS AI type."""
        opponent = TierOpponentConfig(
            id="mcts",
            description="MCTS opponent",
            difficulty=5,
            ai_type=AIType.MCTS,
        )
        assert opponent.ai_type == AIType.MCTS


class TestTierEvaluationConfig:
    """Tests for TierEvaluationConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating a basic evaluation config."""
        config = TierEvaluationConfig(
            tier_name="D1",
            display_name="D1 - Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=1,
            time_budget_ms=None,
        )
        assert config.tier_name == "D1"
        assert config.board_type == BoardType.SQUARE8
        assert config.num_players == 2
        assert config.num_games == 100
        assert config.candidate_difficulty == 1
        assert config.opponents == []
        assert config.min_win_rate_vs_baseline is None
        assert config.min_win_rate_vs_previous_tier == 0.55
        assert config.promotion_confidence == 0.95
        assert config.description == ""

    def test_create_config_with_opponents(self):
        """Test creating config with opponents list."""
        opponents = [
            TierOpponentConfig(
                id="random",
                description="Random baseline",
                difficulty=1,
                ai_type=AIType.RANDOM,
                role="baseline",
            ),
            TierOpponentConfig(
                id="prev_tier",
                description="Previous tier",
                difficulty=1,
                role="previous_tier",
            ),
        ]
        config = TierEvaluationConfig(
            tier_name="D2",
            display_name="D2 - Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=200,
            candidate_difficulty=2,
            time_budget_ms=None,
            opponents=opponents,
            min_win_rate_vs_baseline=0.60,
        )
        assert len(config.opponents) == 2
        assert config.opponents[0].id == "random"
        assert config.opponents[1].role == "previous_tier"
        assert config.min_win_rate_vs_baseline == 0.60

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        config = TierEvaluationConfig(
            tier_name="D1",
            display_name="D1",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=100,
            candidate_difficulty=1,
            time_budget_ms=None,
        )
        with pytest.raises(AttributeError):
            config.tier_name = "modified"  # type: ignore[misc]

    def test_config_with_time_budget(self):
        """Test config with time budget specified."""
        config = TierEvaluationConfig(
            tier_name="D5",
            display_name="D5 - MCTS",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=300,
            candidate_difficulty=5,
            time_budget_ms=5000,
        )
        assert config.time_budget_ms == 5000

    def test_config_with_regression_threshold(self):
        """Test config with max regression threshold."""
        config = TierEvaluationConfig(
            tier_name="D3",
            display_name="D3",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=200,
            candidate_difficulty=3,
            time_budget_ms=None,
            max_regression_vs_previous_tier=0.05,
        )
        assert config.max_regression_vs_previous_tier == 0.05

    def test_config_with_custom_promotion_confidence(self):
        """Test config with custom promotion confidence."""
        config = TierEvaluationConfig(
            tier_name="D8",
            display_name="D8",
            board_type=BoardType.SQUARE8,
            num_players=2,
            num_games=400,
            candidate_difficulty=8,
            time_budget_ms=None,
            promotion_confidence=0.90,
        )
        assert config.promotion_confidence == 0.90

    def test_config_hex8_board(self):
        """Test config for hex8 board."""
        config = TierEvaluationConfig(
            tier_name="D2_HEX8",
            display_name="D2 hex8",
            board_type=BoardType.HEX8,
            num_players=2,
            num_games=100,
            candidate_difficulty=2,
            time_budget_ms=None,
        )
        assert config.board_type == BoardType.HEX8

    def test_config_multiplayer(self):
        """Test config for 3 and 4 player games."""
        config_3p = TierEvaluationConfig(
            tier_name="D2_3P",
            display_name="D2 3p",
            board_type=BoardType.SQUARE8,
            num_players=3,
            num_games=200,
            candidate_difficulty=2,
            time_budget_ms=None,
        )
        config_4p = TierEvaluationConfig(
            tier_name="D2_4P",
            display_name="D2 4p",
            board_type=BoardType.SQUARE8,
            num_players=4,
            num_games=200,
            candidate_difficulty=2,
            time_budget_ms=None,
        )
        assert config_3p.num_players == 3
        assert config_4p.num_players == 4


class TestTierEvalConfigs:
    """Tests for the TIER_EVAL_CONFIGS dictionary."""

    def test_configs_not_empty(self):
        """Test that tier configs exist."""
        assert len(TIER_EVAL_CONFIGS) > 0

    def test_has_canonical_square8_tiers(self):
        """Test that canonical square8 2p tiers D1-D11 exist."""
        for i in range(1, 12):
            tier_name = f"D{i}"
            assert tier_name in TIER_EVAL_CONFIGS, f"Missing tier {tier_name}"
            config = TIER_EVAL_CONFIGS[tier_name]
            assert config.board_type == BoardType.SQUARE8
            assert config.num_players == 2
            assert config.candidate_difficulty == i

    def test_has_hex8_2p_tiers(self):
        """Test that hex8 2p tiers exist for ladder validation."""
        hex8_tiers = [
            "D2_HEX8_2P",
            "D3_HEX8_2P",
            "D4_HEX8_2P",
            "D5_HEX8_2P",
            "D6_HEX8_2P",
            "D7_HEX8_2P",
            "D8_HEX8_2P",
            "D9_HEX8_2P",
            "D10_HEX8_2P",
        ]
        for tier_name in hex8_tiers:
            assert tier_name in TIER_EVAL_CONFIGS, f"Missing tier {tier_name}"
            config = TIER_EVAL_CONFIGS[tier_name]
            assert config.board_type == BoardType.HEX8
            assert config.num_players == 2

    def test_has_square19_tiers(self):
        """Test that square19 tiers exist."""
        assert "D2_SQ19_2P" in TIER_EVAL_CONFIGS
        assert "D4_SQ19_2P" in TIER_EVAL_CONFIGS
        for tier_name in ["D2_SQ19_2P", "D4_SQ19_2P"]:
            config = TIER_EVAL_CONFIGS[tier_name]
            assert config.board_type == BoardType.SQUARE19
            assert config.num_players == 2

    def test_has_hexagonal_tiers(self):
        """Test that legacy hexagonal tiers exist."""
        assert "D2_HEX_2P" in TIER_EVAL_CONFIGS
        assert "D4_HEX_2P" in TIER_EVAL_CONFIGS
        for tier_name in ["D2_HEX_2P", "D4_HEX_2P"]:
            config = TIER_EVAL_CONFIGS[tier_name]
            assert config.board_type == BoardType.HEXAGONAL
            assert config.num_players == 2

    def test_has_multiplayer_tiers(self):
        """Test that 3p and 4p tiers exist."""
        assert "D2_SQ8_3P" in TIER_EVAL_CONFIGS
        assert "D2_SQ8_4P" in TIER_EVAL_CONFIGS
        config_3p = TIER_EVAL_CONFIGS["D2_SQ8_3P"]
        config_4p = TIER_EVAL_CONFIGS["D2_SQ8_4P"]
        assert config_3p.num_players == 3
        assert config_4p.num_players == 4

    def test_all_configs_have_required_fields(self):
        """Test that all configs have required fields set."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            assert config.tier_name, f"{tier_name} missing tier_name"
            assert config.display_name, f"{tier_name} missing display_name"
            assert config.board_type is not None, f"{tier_name} missing board_type"
            assert config.num_players >= 2, f"{tier_name} invalid num_players"
            assert config.num_games > 0, f"{tier_name} invalid num_games"
            assert config.candidate_difficulty >= 1, f"{tier_name} invalid difficulty"

    def test_d1_has_no_gating(self):
        """Test that D1 (random baseline) has no gating requirements."""
        d1 = TIER_EVAL_CONFIGS["D1"]
        assert d1.min_win_rate_vs_baseline is None
        assert d1.max_regression_vs_previous_tier is None
        assert len(d1.opponents) == 0
        assert "Entry tier" in d1.description or "random baseline" in d1.description.lower()

    def test_higher_tiers_have_opponents(self):
        """Test that D2+ tiers have opponents configured."""
        for i in range(2, 12):
            tier_name = f"D{i}"
            config = TIER_EVAL_CONFIGS[tier_name]
            assert len(config.opponents) > 0, f"{tier_name} has no opponents"

    def test_previous_tier_opponent_configured(self):
        """Test that D2+ tiers have previous_tier opponent."""
        for i in range(2, 12):
            tier_name = f"D{i}"
            config = TIER_EVAL_CONFIGS[tier_name]
            previous_tier_opponents = [
                o for o in config.opponents if o.role == "previous_tier"
            ]
            assert len(previous_tier_opponents) >= 1, f"{tier_name} missing previous_tier opponent"

    def test_win_rate_thresholds_reasonable(self):
        """Test that win rate thresholds are in valid range."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            if config.min_win_rate_vs_baseline is not None:
                assert 0.5 <= config.min_win_rate_vs_baseline <= 1.0, (
                    f"{tier_name} baseline threshold out of range"
                )
            assert 0.5 <= config.min_win_rate_vs_previous_tier <= 1.0, (
                f"{tier_name} previous tier threshold out of range"
            )

    def test_promotion_confidence_reasonable(self):
        """Test that promotion confidence is in valid range."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            assert 0.0 <= config.promotion_confidence <= 1.0, (
                f"{tier_name} promotion_confidence out of range"
            )


class TestGetTierConfig:
    """Tests for get_tier_config function."""

    def test_get_existing_tier(self):
        """Test getting an existing tier config."""
        config = get_tier_config("D2")
        assert config.tier_name == "D2"
        assert config.candidate_difficulty == 2

    def test_get_tier_case_insensitive(self):
        """Test that tier lookup is case insensitive."""
        config_upper = get_tier_config("D5")
        config_lower = get_tier_config("d5")
        config_mixed = get_tier_config("D5")
        assert config_upper == config_lower == config_mixed

    def test_get_extended_tier_name(self):
        """Test getting tier with extended name."""
        config = get_tier_config("D2_HEX8_2P")
        assert config.tier_name == "D2_HEX8_2P"
        assert config.board_type == BoardType.HEX8

    def test_get_nonexistent_tier_raises(self):
        """Test that getting nonexistent tier raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_tier_config("D99")
        assert "Unknown tier 'D99'" in str(exc_info.value)
        assert "Available tiers:" in str(exc_info.value)

    def test_get_tier_error_lists_available(self):
        """Test that error message lists available tiers."""
        with pytest.raises(KeyError) as exc_info:
            get_tier_config("INVALID")
        error_msg = str(exc_info.value)
        # Should list at least D1, D2
        assert "D1" in error_msg
        assert "D2" in error_msg

    def test_get_all_canonical_tiers(self):
        """Test getting all canonical tiers by name."""
        for tier_name in TIER_EVAL_CONFIGS.keys():
            config = get_tier_config(tier_name)
            assert config is not None
            assert config.tier_name.upper() == tier_name.upper()


class TestHeuristicTierSpec:
    """Tests for HeuristicTierSpec dataclass."""

    def test_create_heuristic_tier_spec(self):
        """Test creating a heuristic tier spec."""
        spec = HeuristicTierSpec(
            id="test_spec",
            name="Test Spec",
            board_type=BoardType.SQUARE8,
            num_players=2,
            eval_pool_id="v1",
            num_games=64,
            candidate_profile_id="candidate_v1",
            baseline_profile_id="baseline_v1",
        )
        assert spec.id == "test_spec"
        assert spec.name == "Test Spec"
        assert spec.board_type == BoardType.SQUARE8
        assert spec.num_players == 2
        assert spec.eval_pool_id == "v1"
        assert spec.num_games == 64
        assert spec.description == ""

    def test_heuristic_tier_spec_with_description(self):
        """Test creating spec with description."""
        spec = HeuristicTierSpec(
            id="test",
            name="Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            eval_pool_id="v1",
            num_games=64,
            candidate_profile_id="c",
            baseline_profile_id="b",
            description="Test description",
        )
        assert spec.description == "Test description"

    def test_heuristic_tier_spec_is_frozen(self):
        """Test that spec is immutable."""
        spec = HeuristicTierSpec(
            id="test",
            name="Test",
            board_type=BoardType.SQUARE8,
            num_players=2,
            eval_pool_id="v1",
            num_games=64,
            candidate_profile_id="c",
            baseline_profile_id="b",
        )
        with pytest.raises(AttributeError):
            spec.id = "modified"  # type: ignore[misc]


class TestHeuristicTierSpecs:
    """Tests for HEURISTIC_TIER_SPECS list."""

    def test_specs_list_not_empty(self):
        """Test that heuristic tier specs list exists."""
        assert isinstance(HEURISTIC_TIER_SPECS, list)
        assert len(HEURISTIC_TIER_SPECS) >= 1

    def test_all_specs_valid(self):
        """Test that all specs have required fields."""
        for spec in HEURISTIC_TIER_SPECS:
            assert spec.id, "Spec missing id"
            assert spec.name, "Spec missing name"
            assert spec.board_type is not None, "Spec missing board_type"
            assert spec.num_players >= 2, "Spec invalid num_players"
            assert spec.eval_pool_id, "Spec missing eval_pool_id"
            assert spec.num_games > 0, "Spec invalid num_games"
            assert spec.candidate_profile_id, "Spec missing candidate_profile_id"
            assert spec.baseline_profile_id, "Spec missing baseline_profile_id"

    def test_first_spec_is_square8_baseline(self):
        """Test that first spec is square8 baseline v1."""
        if len(HEURISTIC_TIER_SPECS) > 0:
            first_spec = HEURISTIC_TIER_SPECS[0]
            assert first_spec.board_type == BoardType.SQUARE8
            assert first_spec.num_players == 2
            assert "v1" in first_spec.eval_pool_id.lower()


class TestTierConfigIntegrity:
    """Integration tests for tier config consistency."""

    def test_difficulty_progression_square8(self):
        """Test that difficulty increases from D1 to D11."""
        prev_difficulty = 0
        for i in range(1, 12):
            config = TIER_EVAL_CONFIGS[f"D{i}"]
            assert config.candidate_difficulty == i
            assert config.candidate_difficulty > prev_difficulty
            prev_difficulty = config.candidate_difficulty

    def test_game_counts_increase_with_difficulty(self):
        """Test that higher tiers tend to have more games (for statistical significance)."""
        d2_games = TIER_EVAL_CONFIGS["D2"].num_games
        d10_games = TIER_EVAL_CONFIGS["D10"].num_games
        # Higher tiers should have at least as many games
        assert d10_games >= d2_games

    def test_hex8_tiers_have_previous_tier_opponents(self):
        """Test that hex8 tiers reference previous tier for ladder validation."""
        for i in range(2, 11):
            tier_name = f"D{i}_HEX8_2P"
            if tier_name in TIER_EVAL_CONFIGS:
                config = TIER_EVAL_CONFIGS[tier_name]
                prev_tier_opponents = [
                    o for o in config.opponents if o.role == "previous_tier"
                ]
                assert len(prev_tier_opponents) >= 1, f"{tier_name} missing previous tier opponent"
                # The opponent should reference difficulty i-1
                for opponent in prev_tier_opponents:
                    assert opponent.difficulty == i - 1, (
                        f"{tier_name} opponent has wrong difficulty"
                    )

    def test_55_percent_threshold_for_previous_tier(self):
        """Test that all tiers use 55% threshold for previous tier (canonical rule)."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            # D(n) must beat D(n-1) at 55%+ is the canonical rule
            assert config.min_win_rate_vs_previous_tier == 0.55, (
                f"{tier_name} should use 55% threshold for previous tier"
            )

    def test_no_duplicate_tier_names(self):
        """Test that all tier names are unique."""
        tier_names = list(TIER_EVAL_CONFIGS.keys())
        assert len(tier_names) == len(set(tier_names)), "Duplicate tier names found"

    def test_opponent_ids_unique_within_tier(self):
        """Test that opponent IDs are unique within each tier."""
        for tier_name, config in TIER_EVAL_CONFIGS.items():
            opponent_ids = [o.id for o in config.opponents]
            assert len(opponent_ids) == len(set(opponent_ids)), (
                f"{tier_name} has duplicate opponent IDs"
            )
