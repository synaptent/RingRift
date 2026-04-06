"""Tests for the Model Quality Gate (behavioral diversity + value head health).

Verifies mode collapse detection, value head pathology checks, move diversity
tracking, and the aggregated quality verdict logic.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.lib.model_quality_gate import (
    DEAD_VALUE_STD_THRESHOLD,
    LOW_DIVERSITY_THRESHOLD,
    MODE_COLLAPSE_THRESHOLD,
    OPENING_LENGTH,
    QualityGateTracker,
    QualityGateVerdict,
    _check_behavioral_diversity,
    _check_value_head_health,
    _move_key,
    check_model_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_move(type_val: str = "place", x: int = 0, y: int = 0) -> SimpleNamespace:
    """Create a mock Move-like object."""
    return SimpleNamespace(
        type=SimpleNamespace(value=type_val),
        from_pos=None,
        to=SimpleNamespace(x=x, y=y),
    )


def _make_legal_moves(n: int = 20) -> list:
    """Create n distinct mock legal moves."""
    return [_make_move(x=i % 10, y=i // 10) for i in range(n)]


def _play_diverse_games(
    tracker: QualityGateTracker,
    n_games: int = 20,
    moves_per_game: int = 10,
    n_legal: int = 30,
    value_fn=None,
) -> None:
    """Simulate diverse evaluation games where each game plays different moves."""
    legal = _make_legal_moves(n_legal)
    for g in range(n_games):
        for m in range(moves_per_game):
            # Pick a different move based on (game, move_number) so openings differ
            idx = (g * 7 + m * 3) % n_legal
            move = legal[idx]
            root_value = value_fn(g, m) if value_fn else (0.1 * (g % 5) - 0.2)
            tracker.record_move(g, m, move, legal, root_value=root_value)
        tracker.finish_game(g)


def _play_collapsed_games(
    tracker: QualityGateTracker,
    n_games: int = 20,
    moves_per_game: int = 10,
    n_legal: int = 30,
) -> None:
    """Simulate games where ALL games play the exact same opening sequence."""
    legal = _make_legal_moves(n_legal)
    for g in range(n_games):
        for m in range(moves_per_game):
            # Always pick the same move at each position -> identical openings
            move = legal[m % n_legal]
            tracker.record_move(g, m, move, legal, root_value=0.5)
        tracker.finish_game(g)


# ---------------------------------------------------------------------------
# move_key tests
# ---------------------------------------------------------------------------

class TestMoveKey:
    def test_basic_placement_key(self):
        m = _make_move(type_val="place", x=3, y=4)
        key = _move_key(m)
        assert "place" in key
        assert "3" in key
        assert "4" in key

    def test_different_moves_different_keys(self):
        m1 = _make_move(x=0, y=0)
        m2 = _make_move(x=1, y=1)
        assert _move_key(m1) != _move_key(m2)


# ---------------------------------------------------------------------------
# QualityGateVerdict tests
# ---------------------------------------------------------------------------

class TestQualityGateVerdict:
    def test_default_is_passing(self):
        v = QualityGateVerdict()
        assert v.passed
        assert not v.critical
        assert v.summary == "quality gate passed"

    def test_critical_summary(self):
        v = QualityGateVerdict(
            passed=False, critical=True, warnings=["MODE_COLLAPSE: 90%"]
        )
        assert "CRITICAL" in v.summary
        assert "MODE_COLLAPSE" in v.summary

    def test_warn_only_summary(self):
        v = QualityGateVerdict(warnings=["BIASED_VALUE_HEAD: all positive"])
        assert v.passed
        assert "BIASED_VALUE_HEAD" in v.summary


# ---------------------------------------------------------------------------
# Behavioral Diversity tests
# ---------------------------------------------------------------------------

class TestBehavioralDiversity:
    def test_diverse_games_pass(self):
        """Games with varied openings should not trigger mode collapse."""
        tracker = QualityGateTracker()
        _play_diverse_games(tracker, n_games=20, moves_per_game=10, n_legal=30)

        crit, warns, details = _check_behavioral_diversity(tracker)
        assert not crit
        collapse_warns = [w for w in warns if "MODE_COLLAPSE" in w]
        assert len(collapse_warns) == 0

    def test_mode_collapse_detected(self):
        """All games playing the same opening triggers MODE_COLLAPSE."""
        tracker = QualityGateTracker()
        _play_collapsed_games(tracker, n_games=20, moves_per_game=10)

        crit, warns, details = _check_behavioral_diversity(tracker)
        assert crit
        assert any("MODE_COLLAPSE" in w for w in warns)
        assert details["opening_repeat_rate"] > MODE_COLLAPSE_THRESHOLD

    def test_low_diversity_detected(self):
        """Choosing only 1 unique move from 100 legal moves flags LOW_DIVERSITY."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(100)
        single_move = legal[0]  # always pick the same move
        for g in range(10):
            for m in range(10):
                tracker.record_move(g, m, single_move, legal, root_value=0.0)
            tracker.finish_game(g)

        crit, warns, details = _check_behavioral_diversity(tracker)
        assert details["unique_moves_chosen"] == 1
        assert details["diversity_ratio"] < LOW_DIVERSITY_THRESHOLD
        assert any("LOW_DIVERSITY" in w for w in warns)

    def test_adequate_diversity_passes(self):
        """Choosing many distinct moves from the legal set passes diversity."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(20)
        for g in range(10):
            for m in range(10):
                move = legal[(g + m) % 20]  # cycle through all legal moves
                tracker.record_move(g, m, move, legal, root_value=0.0)
            tracker.finish_game(g)

        crit, warns, details = _check_behavioral_diversity(tracker)
        assert not crit
        diversity_warns = [w for w in warns if "LOW_DIVERSITY" in w]
        assert len(diversity_warns) == 0
        assert details["diversity_ratio"] >= LOW_DIVERSITY_THRESHOLD

    def test_too_few_games_skips_opening_check(self):
        """With fewer than 3 games, opening collapse check is skipped."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(10)
        # Only 2 games
        for g in range(2):
            for m in range(5):
                tracker.record_move(g, m, legal[0], legal, root_value=0.0)
            tracker.finish_game(g)

        crit, warns, details = _check_behavioral_diversity(tracker)
        # Should not be critical due to opening check (too few games)
        collapse_warns = [w for w in warns if "MODE_COLLAPSE" in w]
        assert len(collapse_warns) == 0


# ---------------------------------------------------------------------------
# Value Head Health tests
# ---------------------------------------------------------------------------

class TestValueHeadHealth:
    def test_healthy_value_head_passes(self):
        """Normal spread of values passes all checks."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=20, moves_per_game=10,
            value_fn=lambda g, m: 0.3 * (g % 7) - 0.9,
        )

        crit, warns, details = _check_value_head_health(tracker)
        assert not crit
        assert not any("DEAD_VALUE_HEAD" in w for w in warns)

    def test_dead_value_head_detected(self):
        """Constant value output (std < 0.01) triggers DEAD_VALUE_HEAD."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=10, moves_per_game=10,
            value_fn=lambda g, m: 0.5,  # always the same value
        )

        crit, warns, details = _check_value_head_health(tracker)
        assert crit
        assert any("DEAD_VALUE_HEAD" in w for w in warns)
        assert details["value_std"] < DEAD_VALUE_STD_THRESHOLD

    def test_biased_value_head_detected(self):
        """All-positive values trigger BIASED_VALUE_HEAD warning."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=10, moves_per_game=10,
            value_fn=lambda g, m: 0.2 + 0.1 * (g % 5),  # all positive
        )

        crit, warns, details = _check_value_head_health(tracker)
        # Biased is WARN, not CRITICAL
        assert not crit
        assert any("BIASED_VALUE_HEAD" in w for w in warns)
        assert details["all_positive"]

    def test_biased_negative_detected(self):
        """All-negative values trigger BIASED_VALUE_HEAD warning."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=10, moves_per_game=10,
            value_fn=lambda g, m: -0.2 - 0.1 * (g % 5),  # all negative
        )

        crit, warns, details = _check_value_head_health(tracker)
        assert not crit
        assert any("BIASED_VALUE_HEAD" in w for w in warns)
        assert details["all_negative"]

    def test_too_few_values_skips_check(self):
        """With fewer than MIN_VALUES_FOR_CHECK samples, check is skipped."""
        tracker = QualityGateTracker()
        # Record only 2 values (below threshold)
        legal = _make_legal_moves(5)
        for m in range(2):
            tracker.record_move(0, m, legal[0], legal, root_value=0.5)
        tracker.finish_game(0)

        crit, warns, details = _check_value_head_health(tracker)
        assert not crit
        assert len(warns) == 0
        assert "skipped" in details

    def test_no_values_skips_check(self):
        """When root_value is never provided, check is skipped gracefully."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(5)
        for g in range(5):
            for m in range(5):
                tracker.record_move(g, m, legal[m], legal, root_value=None)
            tracker.finish_game(g)

        crit, warns, details = _check_value_head_health(tracker)
        assert not crit
        assert details["value_samples"] == 0


# ---------------------------------------------------------------------------
# Aggregated Quality Verdict tests
# ---------------------------------------------------------------------------

class TestCheckModelQuality:
    def test_healthy_model_passes(self):
        """A well-behaved model passes the quality gate."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=20, moves_per_game=10, n_legal=30,
            value_fn=lambda g, m: 0.2 * (g % 5) - 0.4,
        )
        verdict = check_model_quality(tracker)
        assert verdict.passed
        assert not verdict.critical
        assert "quality gate passed" in verdict.summary

    def test_mode_collapse_is_critical(self):
        """Mode collapse produces a CRITICAL verdict blocking promotion."""
        tracker = QualityGateTracker()
        _play_collapsed_games(tracker, n_games=20)
        verdict = check_model_quality(tracker)
        assert verdict.critical
        assert not verdict.passed
        assert "MODE_COLLAPSE" in verdict.summary

    def test_dead_value_head_is_critical(self):
        """Dead value head produces a CRITICAL verdict."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=20, moves_per_game=10, n_legal=30,
            value_fn=lambda g, m: 0.42,  # constant
        )
        verdict = check_model_quality(tracker)
        assert verdict.critical
        assert not verdict.passed
        assert "DEAD_VALUE_HEAD" in verdict.summary

    def test_biased_value_is_warn_not_critical(self):
        """Biased value head is WARN only — still passes."""
        tracker = QualityGateTracker()
        _play_diverse_games(
            tracker, n_games=20, moves_per_game=10, n_legal=30,
            value_fn=lambda g, m: 0.1 + 0.05 * (g % 10),  # all positive, spread
        )
        verdict = check_model_quality(tracker)
        assert verdict.passed
        assert not verdict.critical
        assert any("BIASED_VALUE_HEAD" in w for w in verdict.warnings)

    def test_empty_tracker_passes(self):
        """An empty tracker (no games played) should pass without errors."""
        tracker = QualityGateTracker()
        verdict = check_model_quality(tracker)
        assert verdict.passed
        assert not verdict.critical


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for quality gate hardening."""

    def test_game_count_auto_tracked_without_finish_game(self):
        """Game count should be auto-tracked from record_move even if finish_game is never called."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(10)
        # Record moves for 5 games but never call finish_game
        for g in range(5):
            for m in range(3):
                idx = (g * 3 + m) % 10
                tracker.record_move(g, m, legal[idx], legal, root_value=0.1 * g)

        assert tracker._game_count == 5
        # The behavioral diversity check should work
        crit, warns, details = _check_behavioral_diversity(tracker)
        assert details["games_tracked"] == 5

    def test_mode_collapse_threshold_exactly_at_boundary(self):
        """Test behavior at exactly the MODE_COLLAPSE_THRESHOLD boundary (80%).

        With 10 games, 8 having the same opening (8/10=0.80) should NOT trigger
        mode collapse since the threshold check is strict (>0.80, not >=).
        The remaining 2 games must each have unique openings.
        """
        tracker = QualityGateTracker()
        legal = _make_legal_moves(20)
        n_games = 10
        for g in range(n_games):
            for m in range(OPENING_LENGTH):
                if g < 8:
                    # Same opening for first 8 games
                    move = legal[m]
                else:
                    # Unique openings for games 8 and 9 (must differ from each
                    # other AND from the common opening)
                    move = legal[(m + 10 + g * 3) % 20]
                tracker.record_move(g, m, move, legal, root_value=0.1 * g - 0.3)
            tracker.finish_game(g)

        crit, warns, details = _check_behavioral_diversity(tracker)
        assert details["opening_repeat_rate"] == pytest.approx(0.8, abs=0.01)
        # The check uses > (strict), not >=, so exactly 80% should NOT trigger
        assert not crit

    def test_fewer_moves_than_opening_length(self):
        """Games with fewer moves than OPENING_LENGTH should still work."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(10)
        # 5 games with only 2 moves each (< OPENING_LENGTH=5)
        for g in range(5):
            for m in range(2):
                tracker.record_move(g, m, legal[(g + m) % 10], legal, root_value=0.1)
            tracker.finish_game(g)

        crit, warns, details = _check_behavioral_diversity(tracker)
        # Should not crash, and opening comparison should use the shorter sequences
        assert details["games_tracked"] == 5

    def test_move_key_with_no_type_attribute(self):
        """Move object without standard attributes should produce a usable key."""
        # A move with just a string representation
        move = SimpleNamespace(type="pass", from_pos=None, to=None)
        key = _move_key(move)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_value_head_with_exactly_min_samples(self):
        """Value head check with exactly MIN_VALUES_FOR_CHECK samples should not skip."""
        from scripts.lib.model_quality_gate import MIN_VALUES_FOR_CHECK
        tracker = QualityGateTracker()
        legal = _make_legal_moves(5)
        # Record exactly MIN_VALUES_FOR_CHECK values
        for m in range(MIN_VALUES_FOR_CHECK):
            tracker.record_move(0, m, legal[m % 5], legal, root_value=0.1 * m - 0.2)
        tracker.finish_game(0)

        crit, warns, details = _check_value_head_health(tracker)
        assert details["value_samples"] == MIN_VALUES_FOR_CHECK
        assert "skipped" not in details

    def test_single_game_all_same_moves_no_crash(self):
        """Single game with all same moves should not crash.

        Note: this triggers DEAD_VALUE_HEAD (constant root_value=0.5 across
        all moves) and LOW_DIVERSITY (only 1 unique move from 20 legal), both
        of which are correct detections.
        """
        tracker = QualityGateTracker()
        legal = _make_legal_moves(20)
        for m in range(10):
            tracker.record_move(0, m, legal[0], legal, root_value=0.5)
        tracker.finish_game(0)

        verdict = check_model_quality(tracker)
        # Correctly flags dead value head and low diversity
        assert verdict.critical
        assert any("DEAD_VALUE_HEAD" in w for w in verdict.warnings)

    def test_value_head_all_zero(self):
        """All-zero values should trigger DEAD_VALUE_HEAD (std < threshold)."""
        tracker = QualityGateTracker()
        legal = _make_legal_moves(10)
        for g in range(5):
            for m in range(10):
                tracker.record_move(g, m, legal[(g + m) % 10], legal, root_value=0.0)
            tracker.finish_game(g)

        crit, warns, details = _check_value_head_health(tracker)
        assert crit
        assert any("DEAD_VALUE_HEAD" in w for w in warns)
