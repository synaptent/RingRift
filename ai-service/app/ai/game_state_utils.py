"""Small helpers for working with GameState objects in AI code.

The Python AIs sometimes receive synthetic or partially-populated GameState
instances (for example in unit tests or data tooling). These helpers provide
robust defaults so search/evaluation code does not accidentally mis-classify
states based on missing fields.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Iterable

from app.models import GameState, BoardType
from app.rules.core import get_territory_victory_threshold, get_victory_threshold


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


def _infer_board_type(state: Any) -> Optional[BoardType]:
    raw = getattr(state, "board_type", getattr(state, "_board_type", None))
    if raw is not None:
        if isinstance(raw, BoardType):
            return raw
        try:
            return BoardType(str(getattr(raw, "value", raw)))
        except Exception:
            return None

    board = getattr(state, "board", getattr(state, "_board", None))
    if board is None:
        return None

    raw = getattr(board, "type", None)
    if raw is None:
        return None
    if isinstance(raw, BoardType):
        return raw
    try:
        return BoardType(str(getattr(raw, "value", raw)))
    except Exception:
        return None


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
        state, "victory_threshold", getattr(state, "_victory_threshold", None)
    )
    territory_threshold = getattr(
        state,
        "territory_victory_threshold",
        getattr(state, "_territory_victory_threshold", None),
    )

    board_type = _infer_board_type(state)
    player_numbers = list(_iter_player_numbers(state))
    num_players = len(player_numbers) if player_numbers else 2

    if not isinstance(victory_threshold, int) or victory_threshold <= 0:
        if board_type is not None:
            victory_threshold = get_victory_threshold(board_type, num_players)
        else:
            victory_threshold = 1

    if not isinstance(territory_threshold, int) or territory_threshold <= 0:
        if board_type is not None:
            territory_threshold = get_territory_victory_threshold(board_type)
        else:
            territory_threshold = 1

    elim_progress = float(eliminated) / float(victory_threshold)
    terr_progress = float(territory) / float(territory_threshold)
    # Clamp for safety (synthetic states / legacy data can exceed thresholds).
    elim_progress = max(0.0, min(1.0, elim_progress))
    terr_progress = max(0.0, min(1.0, terr_progress))
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
        required_rounds = getattr(
            state,
            "lps_rounds_required",
            getattr(state, "_lps_rounds_required", getattr(state, "lpsRoundsRequired", 2)),
        )
        if not isinstance(required_rounds, int) or required_rounds <= 0:
            required_rounds = 2

        # LPS can convert quickly once a player has any exclusive-round momentum.
        # Use a high-threat scale rather than a simple linear fraction.
        if lps_rounds >= required_rounds:
            lps_progress = 1.0
        elif required_rounds == 1:
            lps_progress = 1.0
        else:
            # Map 1..(required-1) → [0.90, 0.99] with the final pre-win round
            # treated as near-certain imminent threat.
            denom = float(required_rounds - 1)
            frac = min(1.0, max(0.0, float(lps_rounds) / denom))
            lps_progress = 0.90 + 0.09 * frac

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
