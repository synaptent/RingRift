"""Small helpers for working with GameState objects in AI code.

The Python AIs sometimes receive synthetic or partially-populated GameState
instances (for example in unit tests or data tooling). These helpers provide
robust defaults so search/evaluation code does not accidentally mis-classify
states based on missing fields.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Iterable

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


def _get_player_counts(state: Any, player_number: int) -> Tuple[int, int]:
    """Return (eliminated_rings, territory_spaces) for a player.

    Supports both immutable GameState (players list) and MutableGameState
    (players dict) used in search.
    """
    players = getattr(state, "players", None)
    if isinstance(players, dict):
        player = players.get(player_number)
        if player is None:
            return 0, 0
        eliminated = int(getattr(player, "eliminated_rings", 0))
        territory = int(getattr(player, "territory_spaces", 0))
        return eliminated, territory

    if players:
        for player in players:
            if getattr(player, "player_number", None) == player_number:
                eliminated = int(getattr(player, "eliminated_rings", 0))
                territory = int(getattr(player, "territory_spaces", 0))
                return eliminated, territory

    return 0, 0


def victory_progress_for_player(state: Any, player_number: int) -> float:
    """Estimate how close a player is to winning by any canonical path.

    We treat "victory progress" as the maximum of:
      - ring-elimination progress (eliminated_rings / victory_threshold)
      - territory-control progress (territory_spaces / territory_victory_threshold)
      - LPS proximity (exclusive-action streak nearing 2 consecutive rounds)

    This is used for Paranoid-style multi-player reductions to pick the
    "most dangerous opponent" without privileging any single victory type.
    """
    eliminated, territory = _get_player_counts(state, player_number)

    victory_threshold = getattr(
        state, "victory_threshold", getattr(state, "_victory_threshold", 1)
    )
    if not isinstance(victory_threshold, int) or victory_threshold <= 0:
        victory_threshold = 1

    territory_threshold = getattr(
        state,
        "territory_victory_threshold",
        getattr(state, "_territory_victory_threshold", 1),
    )
    if not isinstance(territory_threshold, int) or territory_threshold <= 0:
        territory_threshold = 1

    elim_progress = float(eliminated) / float(victory_threshold)
    terr_progress = float(territory) / float(territory_threshold)
    base_progress = max(elim_progress, terr_progress)

    lps_player = getattr(
        state,
        "lps_consecutive_exclusive_player",
        getattr(state, "_lps_consecutive_exclusive_player", None),
    )
    lps_rounds = getattr(
        state,
        "lps_consecutive_exclusive_rounds",
        getattr(state, "_lps_consecutive_exclusive_rounds", 0),
    )

    lps_progress = 0.0
    if lps_player == player_number and isinstance(lps_rounds, int) and lps_rounds > 0:
        # One completed exclusive round is a high-threat precursor to LPS victory.
        lps_progress = 1.0 if lps_rounds >= 2 else 0.95

    return max(base_progress, lps_progress)


def _iter_player_numbers(state: Any) -> Iterable[int]:
    players = getattr(state, "players", None)
    if isinstance(players, dict):
        return players.keys()
    if players:
        return [
            int(getattr(p, "player_number"))
            for p in players
            if getattr(p, "player_number", None) is not None
        ]
    return []


def select_threat_opponent(
    state: Any,
    perspective_player_number: int,
) -> Optional[int]:
    """Select the opponent most likely to win soon.

    For 3p/4p Paranoid reductions we treat the leading opponent (by
    victory_progress_for_player) as the primary adversary.
    """
    best_player: Optional[int] = None
    best_progress = -1.0

    for pid in _iter_player_numbers(state):
        if pid == perspective_player_number:
            continue
        progress = victory_progress_for_player(state, pid)
        if progress > best_progress:
            best_progress = progress
            best_player = pid

    return best_player
