"""Small helpers for working with GameState objects in AI code.

The Python AIs sometimes receive synthetic or partially-populated GameState
instances (for example in unit tests or data tooling). These helpers provide
robust defaults so search/evaluation code does not accidentally mis-classify
states based on missing fields.
"""

from __future__ import annotations

from typing import Optional

from app.models import GameState


def infer_num_players(game_state: GameState) -> int:
    """Infer the number of players for a game state.

    Prefer the concrete ``players`` list when present; fall back to the
    declared ``max_players`` for synthetic states (for example unit tests).
    """
    players = getattr(game_state, "players", None)
    if players:
        return len(players)

    max_players: Optional[int] = getattr(game_state, "max_players", None)
    if isinstance(max_players, int) and max_players > 0:
        return max_players

    # Defensive default: treat unknown as 2p so legacy unit tests and small
    # tooling states do not trip multi-player guards.
    return 2

