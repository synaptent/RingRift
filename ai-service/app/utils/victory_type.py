"""Standardized victory type derivation for RingRift game records.

This module provides canonical victory type categorization per GAME_RECORD_SPEC.md.
All scripts that generate game records should use these functions to ensure
consistent victory type reporting across DBs, JSONL, and training data.

Victory Types (canonical):
- ring_elimination: Winner reached ring elimination threshold (default: 3)
- territory: Winner reached territory threshold (default: 10)
- timeout: Game hit max_moves limit without decisive victory
- lps: Last-player-standing (opponent(s) have 0 total rings)
- stalemate: Bare-board stalemate resolved by tiebreaker ladder

Stalemate Tiebreakers:
- territory: Most collapsed spaces
- eliminated_rings: Most eliminated rings
- markers: Most markers on board
- last_actor: Last player to complete a valid turn action
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "STALEMATE_TIEBREAKERS",
    "VICTORY_TYPES",
    "derive_stalemate_tiebreaker",
    "derive_victory_type",
    "validate_stalemate_tiebreaker",
    "validate_victory_type",
]

if TYPE_CHECKING:
    from app.models import GameState


def derive_stalemate_tiebreaker(game_state: GameState) -> str:
    """Determine which tiebreaker resolved a stalemate victory.

    The tiebreaker ladder per game_engine.py is:
    1. Most collapsed spaces (territory) -> "territory"
    2. Most eliminated rings -> "eliminated_rings"
    3. Most markers -> "markers"
    4. Last player to complete a valid turn action -> "last_actor"

    Args:
        game_state: The final GameState after game completion

    Returns:
        The tiebreaker that determined the winner (the first one
        where the winner has a unique maximum).
    """
    winner = game_state.winner
    if winner is None:
        return "unknown"

    # Calculate scores for all players
    scores = {}
    for player in game_state.players:
        pid = player.player_number
        collapsed = player.territory_spaces

        # Eliminated rings
        eliminated = player.eliminated_rings

        # Markers
        markers = 0
        for m in game_state.board.markers.values():
            if m.player == pid:
                markers += 1

        # Last actor (1 if this player made the last move, 0 otherwise)
        last_actor = 0
        if game_state.move_history and game_state.move_history[-1].player == pid:
            last_actor = 1

        scores[pid] = {
            "collapsed": collapsed,
            "eliminated": eliminated,
            "markers": markers,
            "last_actor": last_actor,
        }

    winner_scores = scores[winner]

    # Check each tiebreaker in order - the winner has the highest value
    # A tiebreaker "determines" the winner if winner has strictly higher than all others
    def winner_has_unique_max(key: str) -> bool:
        winner_val = winner_scores[key]
        return all(not (pid != winner and s[key] >= winner_val) for pid, s in scores.items())

    if winner_has_unique_max("collapsed"):
        return "territory"
    if winner_has_unique_max("eliminated"):
        return "eliminated_rings"
    if winner_has_unique_max("markers"):
        return "markers"
    if winner_has_unique_max("last_actor"):
        return "last_actor"

    # Fallback - couldn't determine (shouldn't happen)
    return "unknown"


def derive_victory_type(
    game_state: GameState,
    max_moves: int | None = None,
) -> tuple[str, str | None]:
    """Derive the victory type from the final game state.

    Per GAME_RECORD_SPEC.md, canonical termination reasons are:
    - "ring_elimination": Winner reached ring elimination threshold
    - "territory": Winner reached territory threshold
    - "timeout": Game hit max_moves limit
    - "lps": Last-player-standing (opponent(s) eliminated)
    - "stalemate": Bare-board stalemate resolved by tiebreaker ladder

    Args:
        game_state: The final GameState after game completion
        max_moves: The max_moves limit used for this game (None if unlimited)

    Returns:
        Tuple of (victory_type, stalemate_tiebreaker).
        stalemate_tiebreaker is None unless victory_type is "stalemate".
    """
    winner = game_state.winner
    move_count = len(game_state.move_history) if game_state.move_history else 0

    # Check timeout first (max_moves reached without victory)
    if max_moves and move_count >= max_moves and winner is None:
        return ("timeout", None)

    # If no winner, it's a stalemate (trapped or drawn, resolved by tiebreakers)
    if winner is None:
        return ("stalemate", derive_stalemate_tiebreaker(game_state))

    # Check ring elimination victory
    for p in game_state.players:
        if p.player_number == winner:
            if p.eliminated_rings >= game_state.victory_threshold:
                return ("ring_elimination", None)
            break

    # Check territory victory
    territory_counts: dict[int, int] = {}
    for p_id in game_state.board.collapsed_spaces.values():
        territory_counts[p_id] = territory_counts.get(p_id, 0) + 1

    winner_territory = territory_counts.get(winner, 0)
    if winner_territory >= game_state.territory_victory_threshold:
        return ("territory", None)

    # Check if this is LPS (opponent(s) eliminated - total rings = 0)
    # vs stalemate (bare board resolved by tiebreaker ladder)
    def count_total_rings(player_number: int) -> int:
        """Count total rings for player (on board + in hand)."""
        player = next(
            (p for p in game_state.players if p.player_number == player_number),
            None,
        )
        rings_in_hand = player.rings_in_hand if player else 0
        rings_on_board = sum(
            1
            for stack in game_state.board.stacks.values()
            for ring_owner in stack.rings
            if ring_owner == player_number
        )
        return rings_in_hand + rings_on_board

    # If any opponent has 0 total rings, this is LPS
    for p in game_state.players:
        if p.player_number != winner and count_total_rings(p.player_number) == 0:
            return ("lps", None)

    # Otherwise it's a stalemate victory (tiebreaker resolution)
    return ("stalemate", derive_stalemate_tiebreaker(game_state))


# Canonical victory type constants for validation
VICTORY_TYPES = frozenset({
    "ring_elimination",
    "territory",
    "timeout",
    "lps",
    "stalemate",
})

STALEMATE_TIEBREAKERS = frozenset({
    "territory",
    "eliminated_rings",
    "markers",
    "last_actor",
    "unknown",
})


def validate_victory_type(victory_type: str) -> bool:
    """Check if a victory_type string is valid."""
    return victory_type in VICTORY_TYPES


def validate_stalemate_tiebreaker(tiebreaker: str | None) -> bool:
    """Check if a stalemate_tiebreaker string is valid."""
    return tiebreaker is None or tiebreaker in STALEMATE_TIEBREAKERS
