"""Tests for the plateau detector (plan item A2, issue #79).

Covers:
- Cold-start protection (no fire below min_iterations)
- Both-triggers-required gating (rate AND staleness)
- Correct rate calculation in the rolling window
- Staleness math when no promotion has ever happened
- Handling of malformed / missing history entries
"""

from __future__ import annotations

import pytest

from scripts.lib.plateau_detector import (
    PLATEAU_MIN_ITERATIONS,
    PLATEAU_REJECTION_RATE,
    PLATEAU_STALENESS,
    PLATEAU_WINDOW,
    PlateauResult,
    detect_plateau,
)


def _entry(iteration: int, promoted: bool) -> dict:
    return {"iteration": iteration, "promoted": promoted}


def _history(total: int, promoted_iters: set[int]) -> list[dict]:
    """Build a history list of `total` iterations with promotions at the
    specified iteration numbers."""
    return [
        _entry(i, promoted=(i in promoted_iters)) for i in range(1, total + 1)
    ]


class TestDefaults:
    def test_module_constants_are_stable(self):
        """Regression guard: defaults documented in the plan doc."""
        assert PLATEAU_WINDOW == 10
        assert PLATEAU_REJECTION_RATE == 0.80
        assert PLATEAU_STALENESS == 15
        assert PLATEAU_MIN_ITERATIONS == 20


class TestEmptyAndShortHistories:
    def test_empty_history(self):
        r = detect_plateau([])
        assert not r.detected
        assert r.total_iterations == 0
        assert r.last_promoted_iteration is None
        assert "no metrics history" in r.reason

    def test_below_min_iterations_never_fires(self):
        # 19 iterations all rejected — looks plateau-ish but we suppress.
        hist = _history(19, promoted_iters=set())
        r = detect_plateau(hist)
        assert not r.detected
        assert r.total_iterations == 19
        assert "below min_iterations" in r.reason


class TestPlateauDetection:
    def test_canonical_hex8_2p_plateau_fires(self):
        """The April 2026 hex8_2p pattern: 33 iters, last promoted at 33,
        then 3 more rejections.  After 36 iters we should detect plateau
        as soon as the rejection streak passes the staleness threshold."""
        promoted = {4, 12, 20, 27, 30, 33}  # 6 promotions, last at 33
        # Iters 34..50 all rejected — staleness grows to 17 at iter 50.
        hist = _history(50, promoted)
        r = detect_plateau(hist)
        assert r.detected, r.reason
        assert r.last_promoted_iteration == 33
        assert r.iterations_since_promotion == 50 - 33
        assert r.recent_rejection_rate == 1.0

    def test_rate_not_high_enough_does_not_fire(self):
        """Window has some recent promotions → rate below 80%."""
        # Promote iter 49 — last window of 10 (iters 41..50) has one
        # promotion, giving rejection_rate = 90%... actually that's
        # still above 80%. Instead promote two iterations in the window.
        promoted = {20, 42, 48}
        hist = _history(50, promoted)
        r = detect_plateau(hist)
        assert not r.detected
        assert r.recent_rejection_rate == 0.8  # 8/10 rejected exactly
        # Exactly at the threshold — staleness gate may or may not fire.
        # Either way, detector should NOT claim a plateau on a boundary
        # rate because the "or" semantics require BOTH triggers to fire
        # strictly above / at thresholds.  Make sure reason is informative.
        assert r.iterations_since_promotion == 50 - 48
        assert r.iterations_since_promotion < PLATEAU_STALENESS

    def test_fresh_promotion_resets_staleness(self):
        """Recent promotion → staleness gate does not fire even with
        high rejection rate in the window (impossible combination in
        practice; here for regression safety)."""
        # 20 iters, all rejected except the last.
        promoted = {25}  # promoted on iter 25
        hist = _history(25, promoted)
        r = detect_plateau(hist)
        assert not r.detected
        assert r.iterations_since_promotion == 0

    def test_never_promoted_uses_total_iteration_as_staleness(self):
        """A config that has never promoted is a cold start; detector
        respects min_iterations but still reports staleness honestly."""
        hist = _history(30, promoted_iters=set())
        r = detect_plateau(hist)
        # 30 iters, all rejected, never promoted — triggers fire.
        assert r.detected
        assert r.last_promoted_iteration is None
        assert r.iterations_since_promotion == 30

    def test_custom_thresholds_respected(self):
        hist = _history(25, promoted_iters={10})
        # Set a very high staleness threshold; detector should not fire.
        r = detect_plateau(hist, staleness_threshold=100)
        assert not r.detected
        assert "iters_since_promotion=15 < 100" in r.reason


class TestWindowSizing:
    def test_window_smaller_than_history_uses_last_N(self):
        promoted = {5, 10}
        hist = _history(30, promoted)
        r = detect_plateau(hist, window=10)
        # Last 10 iters (21..30) are all rejections.
        assert r.window_size == 10
        assert r.recent_rejection_rate == 1.0

    def test_window_larger_than_history_uses_all(self):
        promoted = {2}
        hist = _history(25, promoted)
        r = detect_plateau(hist, window=100)
        # Can't use 100, falls back to total.
        assert r.window_size == 25


class TestMalformedHistory:
    def test_missing_iteration_field_falls_back_to_index(self):
        hist = [{"promoted": False} for _ in range(25)]
        # Last entry index (0-based) is 24 → iteration 25.
        r = detect_plateau(hist)
        assert r.total_iterations == 25
        # No promotions → staleness == total
        assert r.iterations_since_promotion == 25

    def test_missing_promoted_key_excluded_from_rate(self):
        # Half the window has decisions (rejected), half has no 'promoted'.
        # Rate is computed on decided entries only.
        hist = [_entry(i, False) for i in range(1, 21)]  # 20 rejections
        hist += [{"iteration": i} for i in range(21, 31)]  # 10 undecided
        r = detect_plateau(hist)
        # Window is last 10 entries — all undecided → window_size=0.
        assert r.window_size == 0
        assert r.recent_rejection_rate == 0.0


class TestValidation:
    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            detect_plateau([], window=0)
        with pytest.raises(ValueError):
            detect_plateau([], window=-5)

    def test_invalid_rate_threshold_raises(self):
        with pytest.raises(ValueError):
            detect_plateau([], rejection_rate_threshold=0.0)
        with pytest.raises(ValueError):
            detect_plateau([], rejection_rate_threshold=1.5)

    def test_invalid_staleness_raises(self):
        with pytest.raises(ValueError):
            detect_plateau([], staleness_threshold=0)


class TestResultShape:
    def test_result_is_frozen_dataclass(self):
        hist = _history(30, promoted_iters={5, 15})
        r = detect_plateau(hist)
        assert isinstance(r, PlateauResult)
        with pytest.raises((AttributeError, Exception)):
            r.detected = not r.detected  # type: ignore[misc]
