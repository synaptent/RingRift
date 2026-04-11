"""Incremental writer helpers for replay database storage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.models import GameState, Move

from .replay_serialization import _compute_state_hash

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("app.db.game_replay")


class GameWriter:
    """Incremental game writer for live games."""

    def __init__(
        self,
        db,
        game_id: str,
        initial_state: GameState,
        snapshot_interval: int = 20,
        all_snapshots: bool = False,
        store_history_entries: bool = False,
    ) -> None:
        self._db = db
        self._game_id = game_id
        self._initial_state = initial_state
        self._snapshot_interval = snapshot_interval
        self._all_snapshots = all_snapshots
        self._store_history_entries = store_history_entries
        self._move_count = 0
        self._turn_count = 0
        self._current_player = initial_state.current_player
        self._finalized = False
        self._prev_state: GameState | None = initial_state
        self._prev_state_hash: str | None = (
            _compute_state_hash(initial_state) if store_history_entries else None
        )

        self._db._create_placeholder_game(game_id, initial_state)
        self._db._store_initial_state(game_id, initial_state)

    def __enter__(self) -> "GameWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._finalized:
            reason = f"exception: {exc_val}" if exc_type is not None else "not finalized"
            try:
                self._db._delete_game(self._game_id)
                logger.warning(
                    "Cleaned up incomplete game %s (%s)",
                    self._game_id,
                    reason,
                )
            except Exception as cleanup_error:
                logger.error(
                    "Failed to clean up incomplete game %s: %s",
                    self._game_id,
                    cleanup_error,
                )
            self._finalized = True
        return False

    def add_move(
        self,
        move: Move,
        state_after: GameState | None = None,
        state_before: GameState | None = None,
        available_moves: list[Move] | None = None,
        available_moves_count: int | None = None,
        engine_eval: float | None = None,
        engine_depth: int | None = None,
        fsm_valid: bool | None = None,
        fsm_error_code: str | None = None,
        move_probs: dict[str, float] | None = None,
        search_stats: dict | None = None,
        heuristic_features: "np.ndarray | None" = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("GameWriter has been finalized")

        if move.player != self._current_player:
            self._turn_count += 1
            self._current_player = move.player

        phase_hint: str | None = None
        phase_source = state_before if state_before is not None else self._prev_state
        if phase_source is not None:
            current_phase = getattr(phase_source, "current_phase", None)
            if current_phase is not None:
                phase_hint = (
                    current_phase.value
                    if hasattr(current_phase, "value")
                    else str(current_phase)
                )

        self._db._store_move(
            game_id=self._game_id,
            move_number=self._move_count,
            turn_number=self._turn_count,
            move=move,
            phase=phase_hint,
            move_probs=move_probs,
            search_stats=search_stats,
        )

        if heuristic_features is not None:
            self._db.store_move_heuristics(
                self._game_id,
                self._move_count,
                heuristic_features,
            )

        should_snapshot = False
        if state_after is not None:
            if self._all_snapshots:
                should_snapshot = True
            elif self._move_count > 0 and self._move_count % self._snapshot_interval == 0:
                should_snapshot = True

        if should_snapshot and state_after is not None:
            state_hash = _compute_state_hash(state_after) if self._all_snapshots else None
            self._db._store_snapshot(
                game_id=self._game_id,
                move_number=self._move_count,
                state=state_after,
                state_hash=state_hash,
            )

        if self._store_history_entries and state_after is not None:
            before = state_before if state_before is not None else self._prev_state
            if before is not None:
                after_hash = _compute_state_hash(state_after)
                self._db._store_history_entry(
                    game_id=self._game_id,
                    move_number=self._move_count,
                    move=move,
                    state_before=before,
                    state_after=state_after,
                    state_hash_before=self._prev_state_hash,
                    state_hash_after=after_hash,
                    available_moves=available_moves,
                    available_moves_count=available_moves_count,
                    engine_eval=engine_eval,
                    engine_depth=engine_depth,
                    fsm_valid=fsm_valid,
                    fsm_error_code=fsm_error_code,
                )
                self._prev_state = state_after
                self._prev_state_hash = after_hash
        elif state_after is not None:
            self._prev_state = state_after

        self._move_count += 1

    def add_choice(
        self,
        move_number: int,
        choice_type: str,
        player: int,
        options: list[dict],
        selected: dict,
        reasoning: str | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("GameWriter has been finalized")

        self._db._store_choice(
            game_id=self._game_id,
            move_number=move_number,
            choice_type=choice_type,
            player=player,
            options=options,
            selected=selected,
            reasoning=reasoning,
        )

    def finalize(
        self,
        final_state: GameState,
        metadata: dict | None = None,
    ) -> None:
        from app.db.move_data_validator import MIN_MOVES_REQUIRED
        from app.errors import InvalidGameError

        if self._finalized:
            raise RuntimeError("GameWriter already finalized")

        if self._move_count < MIN_MOVES_REQUIRED:
            logger.error(
                "Attempted to finalize game %s with only %s moves (minimum required: %s). Aborting instead.",
                self._game_id,
                self._move_count,
                MIN_MOVES_REQUIRED,
            )
            self.abort()
            raise InvalidGameError(
                f"Cannot finalize game with {self._move_count} moves "
                f"(minimum required: {MIN_MOVES_REQUIRED}). "
                "Games without sufficient move data are useless for training.",
                game_id=self._game_id,
                move_count=self._move_count,
            )

        metadata = metadata or {}
        self._db._store_snapshot(
            game_id=self._game_id,
            move_number=self._move_count - 1,
            state=final_state,
        )
        self._db._finalize_game(
            game_id=self._game_id,
            initial_state=self._initial_state,
            final_state=final_state,
            total_moves=self._move_count,
            total_turns=self._turn_count + 1,
            metadata=metadata,
        )
        self._finalized = True

    def abort(self) -> None:
        if self._finalized:
            return
        self._db._delete_game(self._game_id)
        self._finalized = True
