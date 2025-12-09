from __future__ import annotations

"""
Lightweight canonical history validator for GameReplayDB recordings.

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

from dataclasses import dataclass
from typing import List

from app.db.game_replay import GameReplayDB
from app.rules.history_contract import (
    CanonicalMoveCheckResult,
    phase_move_contract,
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
    issues: List[CanonicalHistoryIssue]


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
    _phase_contract = phase_move_contract()  # Contract lookup, used via validate_canonical_move
    issues: List[CanonicalHistoryIssue] = []

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
