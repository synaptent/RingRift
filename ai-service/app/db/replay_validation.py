"""Replay reconstruction helpers for GameReplayDB."""

from __future__ import annotations

from app.models import GameState, Move


class ReplayValidationMixin:
    """Replay reconstruction and legacy phase-injection helpers."""

    def get_state_at_move(
        self,
        game_id: str,
        move_number: int,
        auto_inject: bool | None = None,
    ) -> GameState | None:
        """Reconstruct state at a specific move number."""
        if auto_inject is None:
            auto_inject = not self._enforce_canonical_history

        from app.game_engine import GameEngine

        state = self.get_initial_state(game_id)
        if state is None:
            return None
        if move_number < 0:
            return state

        moves = self.get_moves(game_id, start=0, end=move_number + 1)
        for move in moves:
            if auto_inject:
                state = self._auto_inject_before_move(state, move)
            state = GameEngine.apply_move(state, move, trace_mode=True)
        return state

    def get_state_at_move_legacy(
        self,
        game_id: str,
        move_number: int,
    ) -> GameState | None:
        """Legacy replay helper that always enables phase injection."""
        return self.get_state_at_move(game_id, move_number, auto_inject=True)

    def _get_game_move_count(self, game_id: str) -> int:
        """Get total number of moves recorded for a game."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM game_moves WHERE game_id = ?",
                (game_id,),
            ).fetchone()
            return row["count"] if row else 0

    def _is_move_redundant_for_phase(self, state: GameState, move: Move) -> bool:
        """Check if a bookkeeping move is redundant for the current phase."""
        current_phase = (
            state.current_phase.value
            if hasattr(state.current_phase, "value")
            else str(state.current_phase)
        )
        move_type = move.type.value if hasattr(move.type, "value") else str(move.type)
        valid_phases = {
            "no_placement_action": ("ring_placement",),
            "no_movement_action": ("movement", "capture", "chain_capture"),
            "no_line_action": ("line_processing",),
            "no_territory_action": ("territory_processing",),
        }
        return move_type in valid_phases and current_phase not in valid_phases[move_type]

    def _auto_inject_before_move(self, state: GameState, next_move: Move) -> GameState:
        """Delegate legacy bookkeeping injection before replaying a move."""
        from app.rules.legacy import auto_inject_before_move

        return auto_inject_before_move(state, next_move)

    def _auto_inject_no_action_moves(self, state: GameState) -> GameState:
        """Delegate legacy NO_LINE_ACTION/NO_TERRITORY_ACTION injection."""
        from app.rules.legacy import auto_inject_no_action_moves

        return auto_inject_no_action_moves(state)
