"""Tests for dynamic priority override computation (Feb 2026).

Verifies compute_config_priority_override() returns correct priority tiers
based on live game counts and Elo ratings, and falls back to
CONFIG_PRIORITY_FALLBACK when live data is unavailable.
"""

from __future__ import annotations

import pytest

from app.coordination.priority_calculator import (
    CONFIG_PRIORITY_FALLBACK,
    PRIORITY_OVERRIDE_MULTIPLIERS,
    compute_config_priority_override,
)


class TestComputeConfigPriorityOverride:
    """Tests for compute_config_priority_override()."""

    def test_zero_games_returns_emergency(self) -> None:
        """0 games should be EMERGENCY (-1) — bootstrap crisis."""
        assert compute_config_priority_override("hex8_2p", game_count=0, elo=None) == -1

    def test_50_games_returns_emergency(self) -> None:
        """50 games (< 100) should be EMERGENCY (-1)."""
        assert compute_config_priority_override("hex8_2p", game_count=50, elo=None) == -1

    def test_99_games_returns_emergency(self) -> None:
        """99 games (< 100) should still be EMERGENCY (-1)."""
        assert compute_config_priority_override("square8_3p", game_count=99, elo=1500.0) == -1

    def test_100_games_returns_critical(self) -> None:
        """100 games (>= 100, < 500) should be CRITICAL (0)."""
        assert compute_config_priority_override("hex8_2p", game_count=100, elo=None) == 0

    def test_300_games_returns_critical(self) -> None:
        """300 games (< 500) should be CRITICAL (0)."""
        assert compute_config_priority_override("hex8_4p", game_count=300, elo=1400.0) == 0

    def test_1000_games_elo_1600_returns_high(self) -> None:
        """1000 games with Elo 1600 (< 1800, < 2000 games) should be HIGH (1)."""
        assert compute_config_priority_override("hex8_2p", game_count=1000, elo=1600.0) == 1

    def test_1999_games_elo_1799_returns_high(self) -> None:
        """1999 games, Elo 1799 — boundary of HIGH (1)."""
        assert compute_config_priority_override("square19_2p", game_count=1999, elo=1799.0) == 1

    def test_2000_games_elo_1600_returns_medium(self) -> None:
        """2000 games with Elo 1600 — past game threshold, falls to MEDIUM (2)."""
        assert compute_config_priority_override("hex8_2p", game_count=2000, elo=1600.0) == 2

    def test_1000_games_elo_1800_returns_medium(self) -> None:
        """1000 games with Elo 1800 — Elo not < 1800, falls to MEDIUM (2)."""
        assert compute_config_priority_override("square8_2p", game_count=1000, elo=1800.0) == 2

    def test_5000_games_elo_1900_returns_medium(self) -> None:
        """5000 games, Elo 1900 — solid progress but not at target."""
        assert compute_config_priority_override("square8_2p", game_count=5000, elo=1900.0) == 2

    def test_1000_games_elo_2050_returns_low(self) -> None:
        """1000 games with Elo 2050 — target met, should be LOW (3)."""
        assert compute_config_priority_override("hex8_2p", game_count=1000, elo=2050.0) == 3

    def test_500_games_elo_2000_returns_low(self) -> None:
        """Even with few games, Elo >= 2000 means target met → LOW (3)."""
        assert compute_config_priority_override("hex8_3p", game_count=500, elo=2000.0) == 3

    def test_none_game_count_falls_back_to_table(self) -> None:
        """game_count=None should fall back to CONFIG_PRIORITY_FALLBACK."""
        # hexagonal_3p is 0 (CRITICAL) in the fallback table
        assert compute_config_priority_override("hexagonal_3p", game_count=None, elo=None) == 0
        assert CONFIG_PRIORITY_FALLBACK.get("hexagonal_3p") == 0

    def test_none_game_count_unknown_config_defaults_to_medium(self) -> None:
        """Unknown config with None game_count should default to MEDIUM (2)."""
        assert compute_config_priority_override("unknown_config", game_count=None, elo=None) == 2

    def test_none_elo_with_low_games_returns_emergency(self) -> None:
        """None Elo with < 100 games is still EMERGENCY (Elo check skipped)."""
        assert compute_config_priority_override("hex8_2p", game_count=50, elo=None) == -1

    def test_none_elo_with_moderate_games_returns_medium(self) -> None:
        """None Elo with >= 500 games and no Elo data → MEDIUM (2)."""
        # Elo < 1800 check requires elo is not None, so falls through to MEDIUM
        assert compute_config_priority_override("hex8_2p", game_count=1000, elo=None) == 2

    def test_emergency_elo_target_met_still_low(self) -> None:
        """Even 10 games with Elo >= 2000 returns LOW — target check comes first."""
        assert compute_config_priority_override("hex8_2p", game_count=10, elo=2100.0) == 3


class TestEmergencyMultiplier:
    """Verify EMERGENCY tier is correctly configured in multiplier table."""

    def test_emergency_multiplier_is_4x(self) -> None:
        """EMERGENCY (-1) should have 4.0x multiplier."""
        assert PRIORITY_OVERRIDE_MULTIPLIERS[-1] == 4.0

    def test_critical_multiplier_is_3x(self) -> None:
        """CRITICAL (0) should have 3.0x multiplier."""
        assert PRIORITY_OVERRIDE_MULTIPLIERS[0] == 3.0

    def test_multiplier_ordering(self) -> None:
        """Multipliers should decrease as priority tier increases."""
        tiers = sorted(PRIORITY_OVERRIDE_MULTIPLIERS.keys())
        multipliers = [PRIORITY_OVERRIDE_MULTIPLIERS[t] for t in tiers]
        for i in range(len(multipliers) - 1):
            assert multipliers[i] >= multipliers[i + 1], (
                f"Multiplier for tier {tiers[i]} ({multipliers[i]}) should be >= "
                f"tier {tiers[i+1]} ({multipliers[i+1]})"
            )
