"""Lightweight canonical history validator for GameReplayDB recordings.

Scope:
- Enforces a minimal subset of RR-CANON-R075 at the recording level by
  checking consistency between stored phase strings and move types.
- Intended to be shared by parity tooling, DB ingestion, and replay
  helpers as a fast pre-flight gate before doing heavier work.

This module does *not* attempt to reconstruct full per-turn phase
sequences from engine simulation. Instead, it treats each recorded
move as authoritative for:

- phase: the GamePhase string stored in game_moves.phase.
- moveType: the MoveType string stored in game_moves.move_type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.game_replay import GameReplayDB
from app.models import GameState
from app.rules.core import (
    BOARD_CONFIGS,
    get_territory_victory_threshold,
    get_victory_threshold,
)
from app.rules.history_contract import (
    CanonicalMoveCheckResult,
    validate_canonical_move,
)


@dataclass
class CanonicalHistoryIssue:
    game_id: str
    move_number: int
    phase: str
    move_type: str
    reason: str


@dataclass
class CanonicalHistoryReport:
    game_id: str
    is_canonical: bool
    issues: list[CanonicalHistoryIssue]


@dataclass
class CanonicalConfigIssue:
    game_id: str
    reason: str
    observed: Any | None = None
    expected: Any | None = None


@dataclass
class CanonicalConfigReport:
    game_id: str
    is_canonical: bool
    issues: list[CanonicalConfigIssue]


def validate_canonical_config_for_game(
    db: GameReplayDB,
    game_id: str,
) -> CanonicalConfigReport:
    """Validate canonical rules parameters from the game's initial state.

    This is intentionally separate from :func:`validate_canonical_history_for_game`.
    The phase↔move contract is useful for parity/debug tooling even when running
    experimental non-canonical rulesOptions (e.g. ringsPerPlayer ablations).
    """
    issues: list[CanonicalConfigIssue] = []

    state: GameState | None = db.get_initial_state(game_id)
    if state is None:
        issues.append(
            CanonicalConfigIssue(
                game_id=game_id,
                reason="missing_initial_state",
            )
        )
        return CanonicalConfigReport(game_id=game_id, is_canonical=False, issues=issues)

    board_type = state.board_type
    num_players = len(state.players)
    config = BOARD_CONFIGS[board_type]

    expected_rings = config.rings_per_player
    expected_victory_threshold = get_victory_threshold(board_type, num_players)
    expected_territory_threshold = get_territory_victory_threshold(board_type)
    expected_lps_rounds_required = 3

    if state.victory_threshold != expected_victory_threshold:
        issues.append(
            CanonicalConfigIssue(
                game_id=game_id,
                reason="victory_threshold_mismatch",
                observed=state.victory_threshold,
                expected=expected_victory_threshold,
            )
        )

    if state.territory_victory_threshold != expected_territory_threshold:
        issues.append(
            CanonicalConfigIssue(
                game_id=game_id,
                reason="territory_victory_threshold_mismatch",
                observed=state.territory_victory_threshold,
                expected=expected_territory_threshold,
            )
        )

    if state.lps_rounds_required != expected_lps_rounds_required:
        issues.append(
            CanonicalConfigIssue(
                game_id=game_id,
                reason="lps_rounds_required_mismatch",
                observed=state.lps_rounds_required,
                expected=expected_lps_rounds_required,
            )
        )

    for player in state.players:
        if player.rings_in_hand != expected_rings:
            issues.append(
                CanonicalConfigIssue(
                    game_id=game_id,
                    reason=f"starting_rings_in_hand_mismatch_p{player.player_number}",
                    observed=player.rings_in_hand,
                    expected=expected_rings,
                )
            )

    rules_options = state.rules_options
    if isinstance(rules_options, dict):
        rings_override = rules_options.get("ringsPerPlayer")
        if rings_override is not None and rings_override != expected_rings:
            issues.append(
                CanonicalConfigIssue(
                    game_id=game_id,
                    reason="noncanonical_rules_options.ringsPerPlayer",
                    observed=rings_override,
                    expected=expected_rings,
                )
            )

        lps_override = rules_options.get("lpsRoundsRequired")
        if lps_override is not None and lps_override != expected_lps_rounds_required:
            issues.append(
                CanonicalConfigIssue(
                    game_id=game_id,
                    reason="noncanonical_rules_options.lpsRoundsRequired",
                    observed=lps_override,
                    expected=expected_lps_rounds_required,
                )
            )

    return CanonicalConfigReport(game_id=game_id, is_canonical=len(issues) == 0, issues=issues)


def validate_canonical_history_for_game(db: GameReplayDB, game_id: str) -> CanonicalHistoryReport:
    """
    Validate that the recorded (phase, moveType) pairs for a game are
    consistent with the canonical phase/move contract.

    Returns a CanonicalHistoryReport; callers can treat any game with
    is_canonical == False as non-canonical for parity/training purposes.

    Note: When the stored phase is empty (legacy recordings), the phase is
    inferred from the move_type using derive_phase_from_move_type(). This
    allows validation of older DBs that predate phase tracking.
    """
    issues: list[CanonicalHistoryIssue] = []

    records = db.get_move_records(game_id)

    for rec in records:
        move_number = int(rec.get("moveNumber", 0) or 0)
        phase = str(rec.get("phase") or "")
        move_type = str(rec.get("moveType") or "")

        check: CanonicalMoveCheckResult = validate_canonical_move(phase, move_type)
        if not check.ok and check.reason:
            issues.append(
                CanonicalHistoryIssue(
                    game_id=game_id,
                    move_number=move_number,
                    phase=check.effective_phase or phase,
                    move_type=move_type,
                    reason=check.reason,
                )
            )

    return CanonicalHistoryReport(game_id=game_id, is_canonical=len(issues) == 0, issues=issues)
