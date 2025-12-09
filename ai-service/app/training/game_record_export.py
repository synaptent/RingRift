"""
Utilities for exporting training episodes as canonical GameRecord objects.

These helpers bridge between the live GameState / Move models used by the
Python rules engine and the GameRecord schema in app.models.game_record.

They are intended for:
- self-play data generation (generate_data.py),
- heuristic training/evaluation scripts (CMA-ES, GA), and
- future RL loops that want per-game JSONL exports.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from app.models import (
    GameState,
    GameStatus,
    Move,
)
from app.models.game_record import (
    FinalScore,
    GameOutcome,
    GameRecord,
    GameRecordMetadata,
    MoveRecord,
    PlayerRecordInfo,
    RecordSource,
)
from app.training.tournament import infer_victory_reason


def _compute_final_score(final_state: GameState) -> FinalScore:
    """Derive FinalScore from a terminal GameState."""
    # Rings eliminated keyed by player number.
    rings_eliminated: Dict[int, int] = {}
    for pid_str, count in final_state.board.eliminated_rings.items():
        try:
            pid = int(pid_str)
        except (TypeError, ValueError):
            continue
        rings_eliminated[pid] = count

    # Territory spaces from the Player models.
    territory_spaces: Dict[int, int] = {}
    for player in final_state.players:
        territory_spaces[player.player_number] = player.territory_spaces

    # Rings remaining = in-hand + on-board per player.
    rings_remaining: Dict[int, int] = {
        player.player_number: player.rings_in_hand for player in final_state.players
    }
    for stack in final_state.board.stacks.values():
        for owner in stack.rings:
            rings_remaining[owner] = rings_remaining.get(owner, 0) + 1

    return FinalScore(
        rings_eliminated=rings_eliminated,
        territory_spaces=territory_spaces,
        rings_remaining=rings_remaining,
    )


def _infer_game_outcome(
    final_state: GameState,
    *,
    terminated_by_budget_only: bool = False,
) -> GameOutcome:
    """Map final state + termination mode to a GameOutcome enum."""
    # Abandoned games map directly.
    if final_state.game_status == GameStatus.ABANDONED:
        return GameOutcome.ABANDONMENT

    # Max-move cutoffs without a winner are treated as timeouts from the
    # training / record perspective.
    if (
        terminated_by_budget_only
        and final_state.game_status == GameStatus.ACTIVE
        and final_state.winner is None
    ):
        return GameOutcome.TIMEOUT

    # Finished / completed games: infer canonical victory category.
    reason = infer_victory_reason(final_state)
    winner = final_state.winner

    if winner is None:
        # Structural stalemates and unknown-but-no-winner are draws.
        return GameOutcome.DRAW

    mapping = {
        "elimination": GameOutcome.RING_ELIMINATION,
        "territory": GameOutcome.TERRITORY_CONTROL,
        "last_player_standing": GameOutcome.LAST_PLAYER_STANDING,
        "structural": GameOutcome.DRAW,
        "unknown": GameOutcome.RING_ELIMINATION,
    }
    return mapping.get(reason, GameOutcome.RING_ELIMINATION)


def build_training_game_record(
    *,
    game_id: str,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    source: RecordSource,
    rng_seed: Optional[int],
    terminated_by_budget_only: bool = False,
    created_at: Optional[datetime] = None,
    tags: Optional[List[str]] = None,
) -> GameRecord:
    """Construct a canonical GameRecord for a completed training episode.

    This helper makes a best-effort mapping from the live GameState / Move
    models to the storage-oriented GameRecord schema. It is intentionally
    conservative:

    - Outcome uses the victory ladder from infer_victory_reason plus an
      explicit TIMEOUT outcome for max-move cutoffs.
    - FinalScore is derived from board + player tallies at the end state.
    - Timing is taken from createdAt/lastMoveAt when available.
    """
    # Timing and duration.
    started_at = initial_state.created_at
    ended_at = final_state.last_move_at
    if isinstance(started_at, datetime) and isinstance(ended_at, datetime):
        total_duration_ms = int((ended_at - started_at).total_seconds() * 1000)
        if total_duration_ms < 0:
            total_duration_ms = 0
    else:
        # Fallback to "now" if timestamps are missing or invalid.
        now = datetime.utcnow()
        started_at = started_at or now
        ended_at = ended_at or now
        total_duration_ms = 0

    # Players.
    players: List[PlayerRecordInfo] = []
    for p in initial_state.players:
        players.append(
            PlayerRecordInfo(
                playerNumber=p.player_number,
                username=p.username,
                playerType="ai" if p.type == "ai" else "human",
                ratingBefore=None,
                ratingAfter=None,
                aiDifficulty=p.ai_difficulty,
                aiType=None,
            )
        )

    final_score = _compute_final_score(final_state)
    outcome = _infer_game_outcome(
        final_state,
        terminated_by_budget_only=terminated_by_budget_only,
    )

    metadata = GameRecordMetadata(
        recordVersion="1.0",
        createdAt=created_at or ended_at,
        source=source,
        sourceId=None,
        generation=None,
        candidateId=None,
        tags=tags or [],
    )

    return GameRecord(
        id=game_id,
        board_type=initial_state.board_type,
        num_players=len(initial_state.players),
        rng_seed=rng_seed,
        is_rated=initial_state.is_rated,
        players=players,
        winner=final_state.winner,
        outcome=outcome,
        final_score=final_score,
        started_at=started_at,
        ended_at=ended_at,
        total_moves=len(moves),
        total_duration_ms=total_duration_ms,
        moves=[MoveRecord.from_move(m) for m in moves],
        metadata=metadata,
        initial_state_hash=None,
        final_state_hash=None,
        progress_snapshots=None,
    )
