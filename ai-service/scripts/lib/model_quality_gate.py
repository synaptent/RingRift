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

    def finish_game(self, game_idx: int) -> None:
        """Mark a game as finished (for game counting)."""
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
# 4. Quality Verdict
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

    # Warnings without critical still pass, but flag for logging
    if verdict.warnings and not verdict.critical:
        verdict.passed = True

    return verdict
