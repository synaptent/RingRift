"""Copy-on-Write (Lazy) State Wrapper for RingRift.

Provides a lightweight wrapper around immutable GameState that defers copying
until the first write operation. This reduces allocation overhead in search
algorithms where we frequently clone state, make one move, evaluate, and discard.

Performance Impact:
- O(1) clone instead of O(N) deep copy
- Only modified positions are allocated
- Ideal for minimax/MCTS where most clones are discarded

Usage:
    from app.rules.lazy_state import LazyMutableState

    # Wrap immutable state (O(1))
    lazy = LazyMutableState(game_state)

    # Read operations delegate to base state
    stack = lazy.get_stack("3,4")  # No copy needed

    # Write operations only copy what changes
    lazy.set_stack("3,4", new_stack)  # Only this position is copied

    # Get the modified state
    new_immutable = lazy.to_immutable()
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from app.rules.game_types import (
    BoardState,
    GamePhase,
    GameState,
    MarkerInfo,
    Move,
    Player,
    RingStack,
    TimeControl,
)


@dataclass
class MutableStackData:
    """Lightweight mutable stack data (avoiding full MutableStack overhead)."""
    controlling_player: int
    stack_height: int
    rings: list[int]

    @classmethod
    def from_ring_stack(cls, stack: RingStack) -> "MutableStackData":
        return cls(
            controlling_player=stack.controlling_player,
            stack_height=stack.stack_height,
            rings=list(stack.rings),
        )

    def to_ring_stack(self) -> RingStack:
        return RingStack(
            controlling_player=self.controlling_player,
            stack_height=self.stack_height,
            rings=tuple(self.rings),
        )


@dataclass
class MutableMarkerData:
    """Lightweight mutable marker data."""
    player: int
    is_pinned: bool = False

    @classmethod
    def from_marker_info(cls, marker: MarkerInfo) -> "MutableMarkerData":
        return cls(player=marker.player, is_pinned=marker.is_pinned)

    def to_marker_info(self) -> MarkerInfo:
        return MarkerInfo(player=self.player, is_pinned=self.is_pinned)


class LazyMutableState:
    """Copy-on-write wrapper for GameState.

    Provides lazy copying semantics - the base state is not modified, and
    we only allocate new data structures when positions are actually changed.

    This is optimized for search algorithms where we frequently:
    1. Clone state (now O(1) instead of O(N))
    2. Make one move (only affected positions are copied)
    3. Evaluate
    4. Discard or undo

    Thread Safety: Not thread-safe. Each thread should create its own wrapper.
    """

    __slots__ = (
        '_base',
        '_modified_stacks',
        '_deleted_stacks',
        '_modified_markers',
        '_deleted_markers',
        '_modified_collapsed',
        '_deleted_collapsed',
        '_active_player',
        '_phase',
        '_chain_capture_state',
        '_must_move_from_stack_key',
        '_lps_round_index',
        '_lps_current_round_actor_mask',
        '_lps_current_round_first_player',
        '_game_status',
        '_winner',
        '_total_rings_eliminated',
        '_zobrist_hash',
        '_move_history',
        '_move_history_dirty',
    )

    def __init__(self, base_state: GameState):
        """Create a lazy wrapper around an immutable GameState.

        Args:
            base_state: The immutable GameState to wrap
        """
        self._base = base_state

        # Modified data - only positions that have been written
        self._modified_stacks: dict[str, MutableStackData] = {}
        self._deleted_stacks: set[str] = set()
        self._modified_markers: dict[str, MutableMarkerData] = {}
        self._deleted_markers: set[str] = set()
        self._modified_collapsed: dict[str, bool] = {}
        self._deleted_collapsed: set[str] = set()

        # Scalar state (cheap to copy)
        self._active_player = base_state.current_player
        self._phase = base_state.current_phase
        self._chain_capture_state = base_state.chain_capture_state
        self._must_move_from_stack_key = base_state.must_move_from_stack_key
        self._lps_round_index = base_state.lps_round_index
        self._lps_current_round_actor_mask = None  # Lazy copy
        self._lps_current_round_first_player = base_state.lps_current_round_first_player
        self._game_status = base_state.game_status
        self._winner = base_state.winner
        self._total_rings_eliminated = base_state.total_rings_eliminated
        self._zobrist_hash = base_state.zobrist_hash or 0

        # Move history (lazy copy)
        self._move_history: list[Move] | None = None
        self._move_history_dirty = False

    # =========================================================================
    # Stack Operations (Copy-on-Write)
    # =========================================================================

    def get_stack(self, pos_key: str) -> RingStack | None:
        """Get stack at position. O(1), no copy."""
        if pos_key in self._deleted_stacks:
            return None
        if pos_key in self._modified_stacks:
            return self._modified_stacks[pos_key].to_ring_stack()
        return self._base.board.stacks.get(pos_key)

    def get_stack_mutable(self, pos_key: str) -> MutableStackData | None:
        """Get mutable stack data. Creates copy on first access for writing."""
        if pos_key in self._deleted_stacks:
            return None
        if pos_key in self._modified_stacks:
            return self._modified_stacks[pos_key]
        # Copy on first write access
        base_stack = self._base.board.stacks.get(pos_key)
        if base_stack is None:
            return None
        mutable = MutableStackData.from_ring_stack(base_stack)
        self._modified_stacks[pos_key] = mutable
        return mutable

    def set_stack(self, pos_key: str, stack: MutableStackData | None) -> None:
        """Set stack at position. O(1) for first write, O(1) for subsequent."""
        if stack is None:
            self._deleted_stacks.add(pos_key)
            self._modified_stacks.pop(pos_key, None)
        else:
            self._deleted_stacks.discard(pos_key)
            self._modified_stacks[pos_key] = stack

    def has_stack(self, pos_key: str) -> bool:
        """Check if position has a stack. O(1)."""
        if pos_key in self._deleted_stacks:
            return False
        if pos_key in self._modified_stacks:
            return True
        return pos_key in self._base.board.stacks

    # =========================================================================
    # Marker Operations (Copy-on-Write)
    # =========================================================================

    def get_marker(self, pos_key: str) -> MarkerInfo | None:
        """Get marker at position. O(1), no copy."""
        if pos_key in self._deleted_markers:
            return None
        if pos_key in self._modified_markers:
            return self._modified_markers[pos_key].to_marker_info()
        return self._base.board.markers.get(pos_key)

    def set_marker(self, pos_key: str, player: int | None, is_pinned: bool = False) -> None:
        """Set or remove marker at position. O(1)."""
        if player is None:
            self._deleted_markers.add(pos_key)
            self._modified_markers.pop(pos_key, None)
        else:
            self._deleted_markers.discard(pos_key)
            self._modified_markers[pos_key] = MutableMarkerData(player=player, is_pinned=is_pinned)

    def has_marker(self, pos_key: str) -> bool:
        """Check if position has a marker. O(1)."""
        if pos_key in self._deleted_markers:
            return False
        if pos_key in self._modified_markers:
            return True
        return pos_key in self._base.board.markers

    # =========================================================================
    # Collapsed Space Operations (Copy-on-Write)
    # =========================================================================

    def is_collapsed(self, pos_key: str) -> bool:
        """Check if position is collapsed. O(1)."""
        if pos_key in self._deleted_collapsed:
            return False
        if pos_key in self._modified_collapsed:
            return self._modified_collapsed[pos_key]
        return pos_key in self._base.board.collapsed_spaces

    def set_collapsed(self, pos_key: str, collapsed: bool) -> None:
        """Set collapsed state for position. O(1)."""
        if collapsed:
            self._deleted_collapsed.discard(pos_key)
            self._modified_collapsed[pos_key] = True
        else:
            if pos_key in self._base.board.collapsed_spaces:
                self._deleted_collapsed.add(pos_key)
            self._modified_collapsed.pop(pos_key, None)

    # =========================================================================
    # Scalar State Accessors
    # =========================================================================

    @property
    def current_player(self) -> int:
        return self._active_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        self._active_player = value

    @property
    def current_phase(self) -> GamePhase:
        return self._phase

    @current_phase.setter
    def current_phase(self, value: GamePhase) -> None:
        self._phase = value

    @property
    def game_status(self) -> str:
        return self._game_status

    @game_status.setter
    def game_status(self, value: str) -> None:
        self._game_status = value

    @property
    def winner(self) -> int | None:
        return self._winner

    @winner.setter
    def winner(self, value: int | None) -> None:
        self._winner = value

    @property
    def zobrist_hash(self) -> int:
        return self._zobrist_hash

    @zobrist_hash.setter
    def zobrist_hash(self, value: int) -> None:
        self._zobrist_hash = value

    @property
    def board_type(self) -> str:
        return self._base.board_type

    @property
    def board_size(self) -> int:
        return self._base.board.size

    # =========================================================================
    # Move History (Lazy Copy)
    # =========================================================================

    @property
    def move_history(self) -> list[Move]:
        if self._move_history is None:
            self._move_history = list(self._base.move_history)
        return self._move_history

    def append_move(self, move: Move) -> None:
        """Append move to history. Lazy copies on first write."""
        if self._move_history is None:
            self._move_history = list(self._base.move_history)
        self._move_history.append(move)
        self._move_history_dirty = True

    # =========================================================================
    # LPS State (Lazy Copy)
    # =========================================================================

    @property
    def lps_current_round_actor_mask(self) -> dict[int, bool]:
        if self._lps_current_round_actor_mask is None:
            self._lps_current_round_actor_mask = dict(
                self._base.lps_current_round_actor_mask
            )
        return self._lps_current_round_actor_mask

    # =========================================================================
    # Player State (Read-Only for now)
    # =========================================================================

    def get_player(self, player_num: int) -> Player | None:
        """Get player info. Note: returns immutable Player for now."""
        for p in self._base.players:
            if p.player_number == player_num:
                return p
        return None

    @property
    def players(self) -> list[Player]:
        return self._base.players

    # =========================================================================
    # Iteration
    # =========================================================================

    def iter_stacks(self):
        """Iterate over all stacks (pos_key, stack). Handles modifications."""
        # First yield modified stacks
        for pos_key, stack_data in self._modified_stacks.items():
            yield pos_key, stack_data.to_ring_stack()

        # Then yield base stacks that weren't modified or deleted
        for pos_key, stack in self._base.board.stacks.items():
            if pos_key not in self._modified_stacks and pos_key not in self._deleted_stacks:
                yield pos_key, stack

    def iter_markers(self):
        """Iterate over all markers (pos_key, marker). Handles modifications."""
        for pos_key, marker_data in self._modified_markers.items():
            yield pos_key, marker_data.to_marker_info()

        for pos_key, marker in self._base.board.markers.items():
            if pos_key not in self._modified_markers and pos_key not in self._deleted_markers:
                yield pos_key, marker

    # =========================================================================
    # Conversion
    # =========================================================================

    def to_immutable(self) -> GameState:
        """Convert to immutable GameState. Only allocates for changed data."""
        # Build stacks dict
        stacks = {}
        for pos_key, stack in self._base.board.stacks.items():
            if pos_key not in self._deleted_stacks:
                if pos_key in self._modified_stacks:
                    stacks[pos_key] = self._modified_stacks[pos_key].to_ring_stack()
                else:
                    stacks[pos_key] = stack
        for pos_key, stack_data in self._modified_stacks.items():
            if pos_key not in stacks:
                stacks[pos_key] = stack_data.to_ring_stack()

        # Build markers dict
        markers = {}
        for pos_key, marker in self._base.board.markers.items():
            if pos_key not in self._deleted_markers:
                if pos_key in self._modified_markers:
                    markers[pos_key] = self._modified_markers[pos_key].to_marker_info()
                else:
                    markers[pos_key] = marker
        for pos_key, marker_data in self._modified_markers.items():
            if pos_key not in markers:
                markers[pos_key] = marker_data.to_marker_info()

        # Build collapsed set
        collapsed = {}
        for pos_key in self._base.board.collapsed_spaces:
            if pos_key not in self._deleted_collapsed:
                collapsed[pos_key] = True
        for pos_key, is_collapsed in self._modified_collapsed.items():
            if is_collapsed:
                collapsed[pos_key] = True

        board = BoardState(
            type=self._base.board_type,
            size=self._base.board.size,
            stacks=stacks,
            markers=markers,
            collapsedSpaces=collapsed,
            eliminatedRings=dict(self._base.board.eliminated_rings),
            formedLines=list(self._base.board.formed_lines),
            territories=dict(self._base.board.territories),
        )

        move_history = self._move_history if self._move_history_dirty else self._base.move_history

        return GameState(
            id=self._base.id,
            boardType=self._base.board_type,
            rngSeed=self._base.rng_seed,
            board=board,
            players=self._base.players,
            currentPhase=self._phase,
            currentPlayer=self._active_player,
            moveHistory=move_history,
            timeControl=self._base.time_control,
            spectators=self._base.spectators,
            gameStatus=self._game_status,
            winner=self._winner,
            createdAt=self._base.created_at,
            lastMoveAt=self._base.last_move_at,
            isRated=self._base.is_rated,
            maxPlayers=self._base.max_players,
            totalRingsInPlay=self._base.total_rings_in_play,
            victoryThreshold=self._base.victory_threshold,
            territoryVictoryThreshold=self._base.territory_victory_threshold,
            chainCaptureState=self._chain_capture_state,
            mustMoveFromStackKey=self._must_move_from_stack_key,
            totalRingsEliminated=self._total_rings_eliminated,
            zobristHash=self._zobrist_hash,
            lpsRoundIndex=self._lps_round_index,
            lpsCurrentRoundActorMask=(
                self._lps_current_round_actor_mask
                if self._lps_current_round_actor_mask is not None
                else self._base.lps_current_round_actor_mask
            ),
            lpsCurrentRoundFirstPlayer=self._lps_current_round_first_player,
            lpsExclusivePlayerForCompletedRound=self._base.lps_exclusive_player_for_completed_round,
            lpsConsecutiveExclusiveRounds=self._base.lps_consecutive_exclusive_rounds,
            lpsConsecutiveExclusivePlayer=self._base.lps_consecutive_exclusive_player,
        )

    def clone(self) -> "LazyMutableState":
        """Create a shallow clone for further modifications.

        This is O(M) where M is the number of modifications, not O(N) board size.
        """
        new = LazyMutableState.__new__(LazyMutableState)
        new._base = self._base

        # Copy modifications (shallow - the MutableStackData/MarkerData are copied)
        new._modified_stacks = dict(self._modified_stacks)
        new._deleted_stacks = set(self._deleted_stacks)
        new._modified_markers = dict(self._modified_markers)
        new._deleted_markers = set(self._deleted_markers)
        new._modified_collapsed = dict(self._modified_collapsed)
        new._deleted_collapsed = set(self._deleted_collapsed)

        # Copy scalars
        new._active_player = self._active_player
        new._phase = self._phase
        new._chain_capture_state = self._chain_capture_state
        new._must_move_from_stack_key = self._must_move_from_stack_key
        new._lps_round_index = self._lps_round_index
        new._lps_current_round_actor_mask = (
            dict(self._lps_current_round_actor_mask)
            if self._lps_current_round_actor_mask is not None
            else None
        )
        new._lps_current_round_first_player = self._lps_current_round_first_player
        new._game_status = self._game_status
        new._winner = self._winner
        new._total_rings_eliminated = self._total_rings_eliminated
        new._zobrist_hash = self._zobrist_hash

        new._move_history = list(self._move_history) if self._move_history is not None else None
        new._move_history_dirty = self._move_history_dirty

        return new


# Convenience function
def wrap_immutable(state: GameState) -> LazyMutableState:
    """Create a copy-on-write wrapper around an immutable GameState.

    This is the recommended entry point for search algorithms.
    """
    return LazyMutableState(state)
