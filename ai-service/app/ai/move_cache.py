"""
Move generation caching for faster repeated lookups.

Caches valid moves based on board state hash to avoid redundant computation.
Uses LRU eviction to bound memory usage.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

from app.utils.checksum_utils import compute_string_checksum

if TYPE_CHECKING:
    from ..models import GameState, Move

# Maximum cache size (number of entries)
MOVE_CACHE_SIZE = int(os.getenv('RINGRIFT_MOVE_CACHE_SIZE', '1000'))

# Enable/disable move caching
# Caches valid moves by board state hash to avoid redundant move generation.
# Low risk optimization - only caches move generation, not evaluation.
USE_MOVE_CACHE = os.getenv('RINGRIFT_USE_MOVE_CACHE', 'true').lower() == 'true'


def _move_type_str(move: object) -> str:
    raw = getattr(move, "type", None)
    if raw is None and isinstance(move, dict):
        raw = move.get("type")
    if raw is None:
        return ""
    if hasattr(raw, "value"):
        try:
            return str(raw.value)
        except (AttributeError, TypeError):
            pass
    return str(raw)


def _move_player_int(move: object) -> int | None:
    raw = getattr(move, "player", None)
    if raw is None and isinstance(move, dict):
        raw = move.get("player")
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _pos_key(pos: object) -> str:
    if pos is None:
        return ""
    if hasattr(pos, "to_key"):
        try:
            return str(pos.to_key())
        except (AttributeError, TypeError, ValueError):
            return ""
    if isinstance(pos, dict):
        try:
            x = pos.get("x", pos.get("col", 0))
            y = pos.get("y", pos.get("row", 0))
            z = pos.get("z")
            if z is None:
                return f"{int(x)},{int(y)}"
            return f"{int(x)},{int(y)},{int(z)}"
        except (KeyError, ValueError, TypeError):
            return ""
    try:
        return str(pos)
    except (TypeError, ValueError):
        return ""


class MoveCache:
    """
    LRU cache for valid moves keyed by board state hash.

    The cache key is computed from:
    - Board type and size
    - Stack positions and controlling players
    - Marker positions and owners
    - Collapsed spaces
    - Current player and phase
    - must_move_from_stack_key (movement constraints)
    - rulesOptions (e.g. swapRuleEnabled)
    - move_history length (meta-move eligibility, e.g. swap_sides)
    """

    def __init__(self, max_size: int = MOVE_CACHE_SIZE):
        self.max_size = max_size
        self._cache: OrderedDict[str, list] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, state: GameState, player: int) -> list[Move] | None:
        """
        Get cached moves for a game state.

        Returns None if not cached.
        """
        if not USE_MOVE_CACHE:
            return None

        phase = getattr(state, "current_phase", None)
        phase_value = phase.value if hasattr(phase, "value") else str(phase)
        # In decision phases (line/territory), legal move generation can depend
        # on move_history content (e.g., which region was processed last) and
        # other transient metadata not reliably captured by board/zobrist alone.
        # Prefer correctness over caching here.
        if phase_value in {"line_processing", "territory_processing"}:
            return None

        # chainCaptureState metadata (visited positions / remaining captures)
        # affects legal move generation but is not encoded in board geometry or
        # zobristHash. Bypass caching for these transient states to avoid stale
        # move lists during chain-capture sequences.
        if getattr(state, "chain_capture_state", None) is not None:
            return None

        key = self._compute_key(state, player)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        return None

    def put(self, state: GameState, player: int, moves: list[Move]):
        """Cache moves for a game state."""
        if not USE_MOVE_CACHE:
            return

        phase = getattr(state, "current_phase", None)
        phase_value = phase.value if hasattr(phase, "value") else str(phase)
        if phase_value in {"line_processing", "territory_processing"}:
            return

        if getattr(state, "chain_capture_state", None) is not None:
            return

        key = self._compute_key(state, player)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = moves

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'size': len(self._cache),
            'hit_rate': hit_rate,
        }

    def _compute_key(self, state: GameState, player: int) -> str:
        """Compute a hash key for the game state.

        CRITICAL: Legal move generation depends on small pieces of metadata
        that are NOT captured by board structure alone:

        - move_history: some move families depend on small pieces of history
          context that do not change the structural board hash, including:
            - swap_sides eligibility (pie rule),
            - territory-processing follow-ups (e.g. eliminate_rings_from_stack
              after choose_territory_option), and
            - recovery-in-turn context (RR-CANON-R114) which affects territory
              prerequisites.
        - must_move_from_stack_key: constrains legal movement/capture options
          after certain placements; this is not encoded in board geometry.
        - rulesOptions: rule toggles like swapRuleEnabled can change the move
          surface even when the board is identical.
        """
        history_len = len(state.move_history) if state.move_history else 0
        must_move_key = state.must_move_from_stack_key or ""

        board = state.board
        board_type = board.type.value if hasattr(board.type, "value") else str(board.type)
        board_size = getattr(board, "size", 0)

        # Player meta affects legal moves (e.g. rings_in_hand determines whether
        # PLACE_RING moves are legal). Encode a small, deterministic digest so
        # cached move lists cannot bleed between positions with identical board
        # structure but different per-player counters.
        players_digest = ""
        players = getattr(state, "players", None) or []
        if players:
            players_meta = sorted(
                f"{p.player_number}:{p.rings_in_hand}:{p.eliminated_rings}:{p.territory_spaces}"
                for p in players
            )
            players_digest = compute_string_checksum("|".join(players_meta), algorithm="md5")

        max_players = getattr(state, "max_players", None)
        if max_players is None:
            max_players = len(players) if players else 0

        phase_value = (
            state.current_phase.value
            if hasattr(state.current_phase, "value")
            else str(state.current_phase)
        )

        rules_options = state.rules_options or {}
        if rules_options:
            rules_str = json.dumps(
                rules_options,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            rules_digest = compute_string_checksum(rules_str, algorithm="md5")
        else:
            rules_digest = ""

        # Last-move signature: some phases derive their interactive move surface
        # from the immediately preceding move (e.g., territory follow-ups).
        last_move_sig = ""
        try:
            last_move = (state.move_history or [])[-1]
        except (IndexError, TypeError, AttributeError):
            last_move = None
        if last_move is not None:
            last_type = _move_type_str(last_move)
            last_player = _move_player_int(last_move) or 0
            last_from = getattr(last_move, "from_pos", None)
            if last_from is None and isinstance(last_move, dict):
                last_from = last_move.get("from") or last_move.get("from_pos")
            last_to = getattr(last_move, "to", None)
            if last_to is None and isinstance(last_move, dict):
                last_to = last_move.get("to")
            last_move_sig = (
                f"lm:{last_player}:{last_type}:{_pos_key(last_from)}->{_pos_key(last_to)}"
            )

        # Territory-processing context: whether the current player's turn
        # included a recovery_slide affects eligibility rules (RR-CANON-R114).
        territory_turn_sig = ""
        if phase_value == "territory_processing":
            has_recovery = False
            for mv in reversed(state.move_history or []):
                mv_player = _move_player_int(mv)
                if mv_player != player:
                    break
                if _move_type_str(mv) == "recovery_slide":
                    has_recovery = True
                    break
            territory_turn_sig = f"tr:{int(has_recovery)}"

        # swap_sides eligibility depends on move history composition (not just length).
        # Without encoding this, a cached move surface can leak swap_sides into a
        # position where Player 2 has already taken a non-swap move (or where swap
        # has already been applied), causing illegal move selection in self-play.
        swap_sig = ""
        try:
            swap_enabled = bool(rules_options.get("swapRuleEnabled", False))
        except (AttributeError, TypeError):
            swap_enabled = False

        if swap_enabled and player == 2 and max_players == 2:
            has_p1_move = False
            has_p2_non_swap_move = False
            has_swap_move = False
            for mv in (state.move_history or []):
                mv_type = _move_type_str(mv)
                mv_player = _move_player_int(mv)
                if mv_type == "swap_sides":
                    has_swap_move = True
                if mv_player == 1:
                    has_p1_move = True
                elif mv_player == 2 and mv_type != "swap_sides":
                    has_p2_non_swap_move = True
                if has_swap_move and has_p2_non_swap_move and has_p1_move:
                    break
            swap_sig = (
                f"sw:{int(has_p1_move)}:{int(has_p2_non_swap_move)}:{int(has_swap_move)}"
            )

        # Use Zobrist hash if available (fast)
        if state.zobrist_hash is not None:
            # Include board metadata so identical hashes on different boards
            # (e.g. square8 vs square19) never reuse move surfaces.
            return (
                f"{board_type}:{board_size}:{max_players}:{players_digest}:"
                f"{state.zobrist_hash}:{player}:{phase_value}:{history_len}:"
                f"{must_move_key}:{rules_digest}:{last_move_sig}:{territory_turn_sig}:{swap_sig}"
            )

        # Fallback: compute hash from board state
        # Build a deterministic string representation
        parts = [
            f"t:{board_type}",
            f"s:{board_size}",
            f"mp:{max_players}",
            f"pm:{players_digest}",
            f"p:{player}",
            f"ph:{phase_value}",
            f"hl:{history_len}",  # History length for swap_sides eligibility
            f"mm:{must_move_key}",
            f"ro:{rules_digest}",
            f"{last_move_sig}",
            f"{territory_turn_sig}",
            f"{swap_sig}",
        ]

        # Add stack positions (sorted for determinism)
        stack_keys = sorted(board.stacks.keys())
        for key in stack_keys:
            stack = board.stacks[key]
            rings_str = ",".join(str(r) for r in stack.rings)
            parts.append(
                f"st:{key}:{stack.controlling_player}:{stack.stack_height}:{rings_str}"
            )

        # Add marker positions
        marker_keys = sorted(board.markers.keys())
        for key in marker_keys:
            marker = board.markers[key]
            parts.append(f"mk:{key}:{marker.player}")

        # Add collapsed spaces
        collapsed_keys = sorted(board.collapsed_spaces.keys())
        parts.append(f"cl:{','.join(collapsed_keys)}")

        # Compute hash
        key_str = "|".join(parts)
        return compute_string_checksum(key_str, algorithm="md5")


# Global move cache instance
_global_cache: MoveCache | None = None


def get_move_cache() -> MoveCache:
    """Get the global move cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MoveCache()
    return _global_cache


def clear_move_cache():
    """Clear the global move cache."""
    if _global_cache is not None:
        _global_cache.clear()


def get_cached_moves(state: GameState, player: int) -> list[Move] | None:
    """Get cached moves from global cache."""
    return get_move_cache().get(state, player)


def cache_moves(state: GameState, player: int, moves: list[Move]):
    """Store moves in global cache."""
    get_move_cache().put(state, player, moves)
