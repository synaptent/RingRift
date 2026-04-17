"""Model Quality Gate — behavioral diversity and value head health checks.

Runs during evaluation games in the minimal AlphaZero loop to detect mode
collapse, dead/biased value heads, and narrow move selection.  Produces a
PASS/WARN/CRITICAL verdict that can block promotion of degenerate candidates.

Usage from minimal_alphazero_loop.py:
    from scripts.lib.model_quality_gate import (
        QualityGateTracker, check_model_quality,
    )

    tracker = QualityGateTracker()
    # inside each eval game, after every move by the candidate:
    tracker.record_move(game_idx, move_number, move, legal_moves, root_value)
    # after all games:
    verdict = check_model_quality(tracker)
    if verdict.critical:
        promoted = False
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QualityGateVerdict:
    """Aggregated quality gate result."""

    passed: bool = True
    critical: bool = False
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.critical:
            parts.append("CRITICAL")
        for w in self.warnings:
            parts.append(w)
        return "; ".join(parts) if parts else "quality gate passed"


# ---------------------------------------------------------------------------
# Move key helper
# ---------------------------------------------------------------------------

def _move_key(move: object) -> str:
    """Convert a Move object to a short string key for deduplication.

    Works with any object that has .type, .from_pos, .to attributes
    (matching the RingRift Move pydantic model).
    """
    mtype = getattr(move, "type", None)
    key = mtype.value if hasattr(mtype, "value") else str(mtype)
    from_pos = getattr(move, "from_pos", None)
    if from_pos is not None:
        key += f"_{getattr(from_pos, 'x', '?')},{getattr(from_pos, 'y', '?')}"
    to_pos = getattr(move, "to", None)
    if to_pos is not None:
        key += f"_{getattr(to_pos, 'x', '?')},{getattr(to_pos, 'y', '?')}"
    return key


# ---------------------------------------------------------------------------
# 1. Behavioral Diversity Tracker
# ---------------------------------------------------------------------------

OPENING_LENGTH = 5
MODE_COLLAPSE_THRESHOLD = 0.80
LOW_DIVERSITY_THRESHOLD = 0.10


class QualityGateTracker:
    """Accumulates move and value data across evaluation games.

    Lightweight — adds negligible overhead per move (just dict/set updates).
    """

    def __init__(self) -> None:
        # Opening sequences: game_idx -> list of move keys for first N moves
        self._openings: dict[int, list[str]] = {}
        # All unique moves chosen by the candidate across all games
        self._unique_moves_chosen: set[str] = set()
        # All unique legal moves seen across all games
        self._unique_legal_moves_seen: set[str] = set()
        # Value head outputs
        self._values: list[float] = []
        self._game_count: int = 0
        # Per-seat outcome tracking: seat (1..num_players) -> wins/games by candidate
        # when it played that seat.  Used to detect structural seat-fairness bugs
        # in multiplayer evaluation (e.g. the square8_3p 20-30% WR pattern where
        # the candidate consistently plays a disadvantaged seat).
        self._seat_wins: dict[int, int] = {}
        self._seat_games: dict[int, int] = {}

    def record_move(
        self,
        game_idx: int,
        move_number: int,
        move: object,
        legal_moves: list,
        root_value: Optional[float] = None,
    ) -> None:
        """Record a single move made by the candidate model.

        Args:
            game_idx: Which evaluation game this move belongs to.
            move_number: 0-based move number within the game.
            move: The Move object chosen by the candidate.
            legal_moves: All legal moves at this position.
            root_value: Value head output at the root (if available).
        """
        mk = _move_key(move)

        # Track opening sequence
        if move_number < OPENING_LENGTH:
            self._openings.setdefault(game_idx, []).append(mk)

        # Track move diversity
        self._unique_moves_chosen.add(mk)
        for lm in legal_moves:
            self._unique_legal_moves_seen.add(_move_key(lm))

        # Track value head output
        if root_value is not None:
            self._values.append(float(root_value))

        # Auto-track game count from game_idx (finish_game is optional)
        self._game_count = max(self._game_count, game_idx + 1)

    def finish_game(self, game_idx: int) -> None:
        """Mark a game as finished (for game counting).

        Optional: game count is also auto-tracked by record_move.
        """
        self._game_count = max(self._game_count, game_idx + 1)

    def record_game_outcome(
        self,
        game_idx: int,
        candidate_seat: int,
        candidate_won: bool,
    ) -> None:
        """Record which seat the candidate played and whether it won.

        Args:
            game_idx: Which evaluation game this outcome belongs to.
            candidate_seat: The seat (player number, 1-indexed) the candidate
                played in this game.
            candidate_won: True if the candidate won, False otherwise (loss or
                draw).

        Used by the seat-fairness check to detect structural imbalance in
        multiplayer evaluation where the candidate performs very differently
        across seats even though seats are rotated fairly across games.
        """
        self._seat_games[candidate_seat] = self._seat_games.get(candidate_seat, 0) + 1
        if candidate_won:
            self._seat_wins[candidate_seat] = self._seat_wins.get(candidate_seat, 0) + 1
        # Game count also advances, in case this is the last signal for the game.
        self._game_count = max(self._game_count, game_idx + 1)


# ---------------------------------------------------------------------------
# 2. Behavioral Diversity Check
# ---------------------------------------------------------------------------

def _check_behavioral_diversity(
    tracker: QualityGateTracker,
) -> tuple[bool, list[str], dict]:
    """Check for mode collapse and narrow move selection.

    Returns (critical, warnings, details).
    """
    warnings: list[str] = []
    critical = False
    details: dict = {}

    n_games = tracker._game_count
    details["games_tracked"] = n_games

    # --- Opening sequence repetition ---
    if n_games >= 3:
        # Stringify opening sequences for comparison
        opening_strs: list[str] = []
        for gidx in sorted(tracker._openings.keys()):
            seq = tracker._openings[gidx]
            opening_strs.append("|".join(seq))

        if opening_strs:
            from collections import Counter
            counts = Counter(opening_strs)
            most_common_seq, most_common_count = counts.most_common(1)[0]
            repeat_rate = most_common_count / len(opening_strs)
            details["opening_repeat_rate"] = round(repeat_rate, 3)
            details["unique_openings"] = len(counts)

            if repeat_rate > MODE_COLLAPSE_THRESHOLD:
                critical = True
                warnings.append(
                    f"MODE_COLLAPSE: {repeat_rate:.0%} of games share the same "
                    f"opening ({most_common_count}/{len(opening_strs)} games)"
                )

    # --- Move diversity ---
    n_chosen = len(tracker._unique_moves_chosen)
    n_legal = len(tracker._unique_legal_moves_seen)
    details["unique_moves_chosen"] = n_chosen
    details["unique_legal_moves_seen"] = n_legal

    if n_legal > 0:
        diversity_ratio = n_chosen / n_legal
        details["diversity_ratio"] = round(diversity_ratio, 3)
        if diversity_ratio < LOW_DIVERSITY_THRESHOLD:
            warnings.append(
                f"LOW_DIVERSITY: only {n_chosen}/{n_legal} legal moves ever "
                f"chosen ({diversity_ratio:.1%})"
            )

    return critical, warnings, details


# ---------------------------------------------------------------------------
# 3. Value Head Health Check
# ---------------------------------------------------------------------------

DEAD_VALUE_STD_THRESHOLD = 0.01
MIN_VALUES_FOR_CHECK = 5


def _check_value_head_health(
    tracker: QualityGateTracker,
) -> tuple[bool, list[str], dict]:
    """Check value head outputs for pathological patterns.

    Returns (critical, warnings, details).
    """
    warnings: list[str] = []
    critical = False
    details: dict = {}

    values = tracker._values
    n = len(values)
    details["value_samples"] = n

    if n < MIN_VALUES_FOR_CHECK:
        details["skipped"] = "too few value samples"
        return False, [], details

    mean_val = sum(values) / n
    var_val = sum((v - mean_val) ** 2 for v in values) / n
    std_val = var_val ** 0.5
    details["value_mean"] = round(mean_val, 4)
    details["value_std"] = round(std_val, 4)

    # Dead value head: near-zero variance
    if std_val < DEAD_VALUE_STD_THRESHOLD:
        critical = True
        warnings.append(
            f"DEAD_VALUE_HEAD: value std={std_val:.6f} < {DEAD_VALUE_STD_THRESHOLD} "
            f"across {n} positions (mean={mean_val:.4f})"
        )
        return critical, warnings, details

    # Biased value head: all values same sign
    all_positive = all(v > 0 for v in values)
    all_negative = all(v < 0 for v in values)
    details["all_positive"] = all_positive
    details["all_negative"] = all_negative

    if all_positive or all_negative:
        sign = "positive" if all_positive else "negative"
        warnings.append(
            f"BIASED_VALUE_HEAD: all {n} values are {sign} "
            f"(mean={mean_val:.4f}, std={std_val:.4f})"
        )

    return critical, warnings, details


# ---------------------------------------------------------------------------
# 4. Seat Fairness Check (multiplayer)
# ---------------------------------------------------------------------------

# Minimum games per seat before seat-fairness analysis runs.  Below this we
# cannot distinguish real imbalance from sampling noise.
SEAT_FAIRNESS_MIN_GAMES_PER_SEAT = 10
# Ratio of highest per-seat WR to lowest per-seat WR above which we flag an
# imbalance warning.  A fair model playing all seats should have a ratio close
# to 1.0; 1.5 captures structural bias without firing on normal seat variance.
SEAT_FAIRNESS_MAX_RATIO = 1.5


def _check_seat_fairness(
    tracker: QualityGateTracker,
) -> tuple[bool, list[str], dict]:
    """Check whether the candidate's win rate differs sharply across seats.

    For 3p/4p multiplayer this is the single most diagnostic signal for the
    "value head is seat-biased" failure mode observed on square8_3p (iters
    9-23 rejected at 20-30% WR).  Staged evaluation rotates the candidate
    through every seat, so a fair model should win at roughly the same rate
    from each seat.  Wildly different per-seat WRs indicate the value head
    has learned "seat N usually wins" instead of evaluating positions.

    Returns (critical, warnings, details).  Never critical today — this is a
    diagnostic signal; callers decide whether to act on it.
    """
    warnings: list[str] = []
    details: dict = {}

    seat_games = dict(tracker._seat_games)
    seat_wins = dict(tracker._seat_wins)

    if not seat_games:
        details["skipped"] = "no seat outcomes recorded"
        return False, [], details

    # Compute per-seat WR
    seat_wr: dict[int, float] = {}
    for seat, games in sorted(seat_games.items()):
        wins = seat_wins.get(seat, 0)
        seat_wr[seat] = wins / games if games > 0 else 0.0
    details["seat_games"] = seat_games
    details["seat_wr"] = {s: round(wr, 3) for s, wr in seat_wr.items()}

    # Only analyze when every seat has enough samples.  With fewer than
    # SEAT_FAIRNESS_MIN_GAMES_PER_SEAT per seat, the ratio is noisy.
    min_games = min(seat_games.values())
    details["min_games_per_seat"] = min_games
    if min_games < SEAT_FAIRNESS_MIN_GAMES_PER_SEAT:
        details["skipped"] = (
            f"min {min_games} games/seat < {SEAT_FAIRNESS_MIN_GAMES_PER_SEAT}"
        )
        return False, [], details

    # Single-seat tracking (2p games where candidate played only seat 1 or 2):
    # We can still report the WR but cannot compute a ratio.
    if len(seat_wr) < 2:
        details["note"] = "only one seat observed, ratio N/A"
        return False, [], details

    # Avoid division by zero when some seat has 0 wins.  Use a small epsilon
    # that still flags imbalance cleanly.
    min_wr = min(seat_wr.values())
    max_wr = max(seat_wr.values())
    eff_min = max(min_wr, 1e-3)
    ratio = max_wr / eff_min
    details["wr_ratio"] = round(ratio, 3)

    if ratio > SEAT_FAIRNESS_MAX_RATIO:
        seat_str = ", ".join(
            f"seat{s}={seat_wr[s]:.0%} ({seat_wins.get(s, 0)}/{seat_games[s]})"
            for s in sorted(seat_wr)
        )
        warnings.append(
            f"SEAT_WR_IMBALANCE: max/min per-seat WR ratio "
            f"{ratio:.2f} > {SEAT_FAIRNESS_MAX_RATIO} ({seat_str})"
        )

    return False, warnings, details


# ---------------------------------------------------------------------------
# 5. Quality Verdict
# ---------------------------------------------------------------------------

def check_model_quality(tracker: QualityGateTracker) -> QualityGateVerdict:
    """Aggregate behavioral diversity and value head health into a verdict.

    Returns QualityGateVerdict with passed/critical flags.
    """
    verdict = QualityGateVerdict()

    # Behavioral diversity
    try:
        crit, warns, details = _check_behavioral_diversity(tracker)
        verdict.details["behavioral_diversity"] = details
        if crit:
            verdict.critical = True
            verdict.passed = False
        verdict.warnings.extend(warns)
    except Exception as e:
        logger.warning("Behavioral diversity check error: %s", e)
        verdict.details["behavioral_diversity"] = {"error": str(e)}

    # Value head health
    try:
        crit, warns, details = _check_value_head_health(tracker)
        verdict.details["value_head_health"] = details
        if crit:
            verdict.critical = True
            verdict.passed = False
        verdict.warnings.extend(warns)
    except Exception as e:
        logger.warning("Value head health check error: %s", e)
        verdict.details["value_head_health"] = {"error": str(e)}

    # Seat fairness (multiplayer diagnostic; never critical by itself)
    try:
        crit, warns, details = _check_seat_fairness(tracker)
        verdict.details["seat_fairness"] = details
        if crit:
            verdict.critical = True
            verdict.passed = False
        verdict.warnings.extend(warns)
    except Exception as e:
        logger.warning("Seat fairness check error: %s", e)
        verdict.details["seat_fairness"] = {"error": str(e)}

    # Warnings without critical still pass, but flag for logging
    if verdict.warnings and not verdict.critical:
        verdict.passed = True

    return verdict
