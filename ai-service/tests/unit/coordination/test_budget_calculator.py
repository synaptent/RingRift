"""Tests for budget_calculator module.

December 2025: Tests for extracted budget calculation logic from selfplay_scheduler.py.
"""

import pytest

from app.coordination.budget_calculator import (
    # Functions
    get_adaptive_budget_for_elo,
    get_adaptive_budget_for_games,
    compute_target_games,
    parse_config_key,
    get_budget_tier_name,
    # Constants
    BOARD_DIFFICULTY_MULTIPLIERS,
    PLAYER_COUNT_MULTIPLIERS,
    ELO_TIER_MASTER,
    ELO_TIER_ULTIMATE,
    ELO_TIER_QUALITY,
    TARGET_ELO_THRESHOLD,
    GAMES_PER_ELO_POINT,
    MAX_TARGET_GAMES,
)
from app.config.thresholds import (
    GUMBEL_BUDGET_STANDARD,
    GUMBEL_BUDGET_QUALITY,
    GUMBEL_BUDGET_ULTIMATE,
    GUMBEL_BUDGET_MASTER,
    GUMBEL_BUDGET_BOOTSTRAP_TIER1,
    GUMBEL_BUDGET_BOOTSTRAP_TIER2,
    GUMBEL_BUDGET_BOOTSTRAP_TIER3,
)


class TestGetAdaptiveBudgetForElo:
    """Tests for get_adaptive_budget_for_elo function."""

    def test_master_tier_at_2000_elo(self):
        """Elo >= 2000 returns master budget (3200)."""
        assert get_adaptive_budget_for_elo(2000) == GUMBEL_BUDGET_MASTER
        assert get_adaptive_budget_for_elo(2100) == GUMBEL_BUDGET_MASTER
        assert get_adaptive_budget_for_elo(2500) == GUMBEL_BUDGET_MASTER

    def test_ultimate_tier_at_1800_elo(self):
        """Elo >= 1800 and < 2000 returns ultimate budget (1600)."""
        assert get_adaptive_budget_for_elo(1800) == GUMBEL_BUDGET_ULTIMATE
        assert get_adaptive_budget_for_elo(1900) == GUMBEL_BUDGET_ULTIMATE
        assert get_adaptive_budget_for_elo(1999) == GUMBEL_BUDGET_ULTIMATE

    def test_quality_tier_at_1500_elo(self):
        """Elo >= 1500 and < 1800 returns quality budget (800)."""
        assert get_adaptive_budget_for_elo(1500) == GUMBEL_BUDGET_QUALITY
        assert get_adaptive_budget_for_elo(1650) == GUMBEL_BUDGET_QUALITY
        assert get_adaptive_budget_for_elo(1799) == GUMBEL_BUDGET_QUALITY

    def test_standard_tier_below_1500_elo(self):
        """Elo < 1500 returns standard budget (800)."""
        assert get_adaptive_budget_for_elo(1499) == GUMBEL_BUDGET_STANDARD
        assert get_adaptive_budget_for_elo(1200) == GUMBEL_BUDGET_STANDARD
        assert get_adaptive_budget_for_elo(1000) == GUMBEL_BUDGET_STANDARD

    def test_boundary_values(self):
        """Test exact boundary values."""
        # Just below each tier
        assert get_adaptive_budget_for_elo(1999.9) == GUMBEL_BUDGET_ULTIMATE
        assert get_adaptive_budget_for_elo(1799.9) == GUMBEL_BUDGET_QUALITY
        assert get_adaptive_budget_for_elo(1499.9) == GUMBEL_BUDGET_STANDARD


class TestGetAdaptiveBudgetForGames:
    """Tests for get_adaptive_budget_for_games function."""

    def test_bootstrap_tier1_under_100_games(self):
        """< 100 games returns bootstrap tier 1 budget (64)."""
        assert get_adaptive_budget_for_games(0, 1200) == GUMBEL_BUDGET_BOOTSTRAP_TIER1
        assert get_adaptive_budget_for_games(50, 1500) == GUMBEL_BUDGET_BOOTSTRAP_TIER1
        assert get_adaptive_budget_for_games(99, 1800) == GUMBEL_BUDGET_BOOTSTRAP_TIER1

    def test_bootstrap_tier2_100_to_500_games(self):
        """100-499 games returns bootstrap tier 2 budget (150)."""
        assert get_adaptive_budget_for_games(100, 1200) == GUMBEL_BUDGET_BOOTSTRAP_TIER2
        assert get_adaptive_budget_for_games(250, 1500) == GUMBEL_BUDGET_BOOTSTRAP_TIER2
        assert get_adaptive_budget_for_games(499, 1800) == GUMBEL_BUDGET_BOOTSTRAP_TIER2

    def test_bootstrap_tier3_500_to_1000_games(self):
        """500-999 games returns bootstrap tier 3 budget (200)."""
        assert get_adaptive_budget_for_games(500, 1200) == GUMBEL_BUDGET_BOOTSTRAP_TIER3
        assert get_adaptive_budget_for_games(750, 1500) == GUMBEL_BUDGET_BOOTSTRAP_TIER3
        assert get_adaptive_budget_for_games(999, 1800) == GUMBEL_BUDGET_BOOTSTRAP_TIER3

    def test_mature_phase_uses_elo_budget(self):
        """>=1000 games uses Elo-based budget."""
        # Master tier (Elo >= 2000)
        assert get_adaptive_budget_for_games(1000, 2000) == GUMBEL_BUDGET_MASTER

        # Ultimate tier (Elo >= 1800)
        assert get_adaptive_budget_for_games(2000, 1850) == GUMBEL_BUDGET_ULTIMATE

        # Quality tier (Elo >= 1500)
        assert get_adaptive_budget_for_games(5000, 1600) == GUMBEL_BUDGET_QUALITY

        # Standard tier (Elo < 1500)
        assert get_adaptive_budget_for_games(1500, 1400) == GUMBEL_BUDGET_STANDARD

    def test_elo_ignored_during_bootstrap(self):
        """High Elo doesn't affect bootstrap budget."""
        # Even with 2000+ Elo, bootstrap budget used when games < 1000
        assert get_adaptive_budget_for_games(50, 2100) == GUMBEL_BUDGET_BOOTSTRAP_TIER1
        assert get_adaptive_budget_for_games(300, 2000) == GUMBEL_BUDGET_BOOTSTRAP_TIER2
        assert get_adaptive_budget_for_games(800, 1900) == GUMBEL_BUDGET_BOOTSTRAP_TIER3


class TestParseConfigKey:
    """Tests for parse_config_key function."""

    def test_standard_configs(self):
        """Test standard config key formats."""
        assert parse_config_key("hex8_2p") == ("hex8", 2)
        assert parse_config_key("square8_4p") == ("square8", 4)
        assert parse_config_key("square19_3p") == ("square19", 3)
        assert parse_config_key("hexagonal_2p") == ("hexagonal", 2)

    def test_malformed_config_returns_defaults(self):
        """Malformed or partial configs return defaults for missing parts."""
        # Empty string -> default board
        assert parse_config_key("") == ("hex8", 2)
        # Single word -> uses it as board, defaults player count
        assert parse_config_key("invalid") == ("invalid", 2)
        assert parse_config_key("hex8") == ("hex8", 2)

    def test_edge_cases(self):
        """Test edge case inputs."""
        assert parse_config_key("board_5p") == ("board", 5)
        assert parse_config_key("a_1p") == ("a", 1)


class TestComputeTargetGames:
    """Tests for compute_target_games function."""

    def test_already_at_target_returns_zero(self):
        """Returns 0 when already at or above target Elo."""
        assert compute_target_games("hex8_2p", 1900) == 0
        assert compute_target_games("hex8_2p", 2000) == 0
        assert compute_target_games("square19_4p", 1950) == 0

    def test_elo_gap_calculation(self):
        """Target games scale with Elo gap."""
        # hex8_2p with 400 Elo gap (1500 -> 1900)
        # 400 * 500 * 1.0 (board) * 1.0 (players) = 200,000
        target = compute_target_games("hex8_2p", 1500)
        assert target == 200_000

        # hex8_2p with 100 Elo gap (1800 -> 1900)
        # 100 * 500 * 1.0 * 1.0 = 50,000
        target = compute_target_games("hex8_2p", 1800)
        assert target == 50_000

    def test_board_difficulty_multipliers(self):
        """Board difficulty affects target games."""
        elo = 1700  # 200 Elo gap
        base = 200 * 500  # 100,000

        # hex8 (1.0x) = 100,000
        assert compute_target_games("hex8_2p", elo) == int(base * 1.0)

        # square8 (1.2x) = 120,000
        assert compute_target_games("square8_2p", elo) == int(base * 1.2)

        # square19 (2.0x) = 200,000
        assert compute_target_games("square19_2p", elo) == int(base * 2.0)

        # hexagonal (2.5x) = 250,000
        assert compute_target_games("hexagonal_2p", elo) == int(base * 2.5)

    def test_player_count_multipliers(self):
        """Player count affects target games."""
        elo = 1700  # 200 Elo gap
        base = 200 * 500  # 100,000 for hex8

        # 2p (1.0x)
        assert compute_target_games("hex8_2p", elo) == int(base * 1.0)

        # 3p (1.5x)
        assert compute_target_games("hex8_3p", elo) == int(base * 1.5)

        # 4p (2.5x)
        assert compute_target_games("hex8_4p", elo) == int(base * 2.5)

    def test_combined_multipliers(self):
        """Board and player multipliers combine."""
        elo = 1700  # 200 Elo gap
        base = 200 * 500  # 100,000

        # hexagonal_4p: 2.5 (board) * 2.5 (players) = 6.25x
        expected = int(base * 2.5 * 2.5)
        assert compute_target_games("hexagonal_4p", elo) == min(expected, MAX_TARGET_GAMES)

    def test_max_target_games_cap(self):
        """Target games capped at MAX_TARGET_GAMES (500K)."""
        # Very low Elo with hardest config would exceed cap
        # 1000 Elo gap * 500 * 2.5 * 2.5 = 3,125,000 (capped to 500K)
        target = compute_target_games("hexagonal_4p", 900)
        assert target == MAX_TARGET_GAMES


class TestGetBudgetTierName:
    """Tests for get_budget_tier_name function."""

    def test_named_tiers(self):
        """Named budget tiers return correct names."""
        assert get_budget_tier_name(GUMBEL_BUDGET_MASTER) == "MASTER"
        assert get_budget_tier_name(GUMBEL_BUDGET_ULTIMATE) == "ULTIMATE"
        # Note: QUALITY and STANDARD both equal 800, so get_budget_tier_name
        # returns whichever appears first in the dict. Both are valid for 800.
        assert get_budget_tier_name(GUMBEL_BUDGET_STANDARD) in ("QUALITY", "STANDARD")
        assert get_budget_tier_name(GUMBEL_BUDGET_BOOTSTRAP_TIER1) == "BOOTSTRAP_TIER1"
        assert get_budget_tier_name(GUMBEL_BUDGET_BOOTSTRAP_TIER2) == "BOOTSTRAP_TIER2"
        assert get_budget_tier_name(GUMBEL_BUDGET_BOOTSTRAP_TIER3) == "BOOTSTRAP_TIER3"

    def test_custom_budget(self):
        """Unknown budget returns CUSTOM(value)."""
        assert get_budget_tier_name(100) == "CUSTOM(100)"
        assert get_budget_tier_name(999) == "CUSTOM(999)"


class TestConstants:
    """Tests for module constants."""

    def test_elo_tier_thresholds(self):
        """Elo tier thresholds are properly ordered."""
        assert ELO_TIER_QUALITY < ELO_TIER_ULTIMATE < ELO_TIER_MASTER

    def test_board_difficulty_multipliers(self):
        """Board multipliers exist for all board types."""
        assert "hex8" in BOARD_DIFFICULTY_MULTIPLIERS
        assert "square8" in BOARD_DIFFICULTY_MULTIPLIERS
        assert "square19" in BOARD_DIFFICULTY_MULTIPLIERS
        assert "hexagonal" in BOARD_DIFFICULTY_MULTIPLIERS

        # Larger boards have higher multipliers
        assert BOARD_DIFFICULTY_MULTIPLIERS["hex8"] < BOARD_DIFFICULTY_MULTIPLIERS["hexagonal"]

    def test_player_count_multipliers(self):
        """Player multipliers exist for 2, 3, 4 players."""
        assert 2 in PLAYER_COUNT_MULTIPLIERS
        assert 3 in PLAYER_COUNT_MULTIPLIERS
        assert 4 in PLAYER_COUNT_MULTIPLIERS

        # More players = higher multiplier
        assert PLAYER_COUNT_MULTIPLIERS[2] < PLAYER_COUNT_MULTIPLIERS[3]
        assert PLAYER_COUNT_MULTIPLIERS[3] < PLAYER_COUNT_MULTIPLIERS[4]

    def test_games_per_elo_point(self):
        """GAMES_PER_ELO_POINT is reasonable."""
        assert GAMES_PER_ELO_POINT == 500

    def test_target_elo_threshold(self):
        """TARGET_ELO_THRESHOLD is set correctly."""
        assert TARGET_ELO_THRESHOLD == 1900

    def test_max_target_games(self):
        """MAX_TARGET_GAMES is half a million."""
        assert MAX_TARGET_GAMES == 500_000
