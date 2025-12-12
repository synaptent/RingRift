"""
Lightweight state representation for fast make/unmake move evaluation.

This module provides a Pydantic-free internal state representation that enables
efficient in-place mutation with undo capability, avoiding the overhead of
deep copying GameState objects for each candidate move evaluation.

Performance gains:
- Eliminates Pydantic validation overhead (~0.07s per game)
- Eliminates dict copying overhead (~60 copies per move)
- Enables O(1) undo instead of O(n) state recreation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class LightweightBoardType(Enum):
    """Board type for lightweight state."""

    SQUARE8 = "square8"
    SQUARE19 = "square19"
    HEXAGONAL = "hexagonal"


class LightweightPhase(Enum):
    """Game phase for lightweight state - 7 canonical phases per RR-CANON-R070."""

    RING_PLACEMENT = "ring_placement"
    MOVEMENT = "movement"
    CAPTURE = "capture"
    CHAIN_CAPTURE = "chain_capture"
    LINE_PROCESSING = "line_processing"
    TERRITORY_PROCESSING = "territory_processing"
    # Final phase: entered only when player had no actions in all prior phases
    # but still controls stacks. Records forced_elimination move then advances
    # to next player. See RR-CANON-R100, RR-CANON-R204.
    FORCED_ELIMINATION = "forced_elimination"


@dataclass(slots=True)
class LightweightStack:
    """Minimal stack representation."""

    position_key: str
    rings: List[int]  # Player numbers from bottom to top
    controlling_player: int

    @property
    def stack_height(self) -> int:
        return len(self.rings)


@dataclass(slots=True)
class LightweightMarker:
    """Minimal marker representation."""

    position_key: str
    player: int


@dataclass(slots=True)
class LightweightPlayer:
    """Minimal player representation for evaluation."""

    player_number: int
    rings_in_hand: int
    eliminated_rings: int
    territory_spaces: int


@dataclass(slots=True)
class MoveUndo:
    """Captures state delta for undoing a move.

    This records the minimal information needed to reverse a move:
    - What was added (to be removed on undo)
    - What was removed (to be restored on undo)
    - What was modified (original values to restore)
    """

    # Stack changes
    added_stack_keys: List[str] = field(default_factory=list)
    removed_stacks: Dict[str, LightweightStack] = field(default_factory=dict)
    modified_stacks: Dict[str, LightweightStack] = field(default_factory=dict)  # Original state

    # Marker changes
    added_marker_keys: List[str] = field(default_factory=list)
    removed_markers: Dict[str, LightweightMarker] = field(default_factory=dict)

    # Player changes (player_number -> original values)
    player_rings_in_hand: Dict[int, int] = field(default_factory=dict)
    player_eliminated_rings: Dict[int, int] = field(default_factory=dict)
    player_territory_spaces: Dict[int, int] = field(default_factory=dict)

    # Territory changes
    added_territory_keys: List[str] = field(default_factory=list)
    removed_territories: Dict[str, int] = field(default_factory=dict)  # key -> owner
    modified_territories: Dict[str, int] = field(default_factory=dict)  # key -> original owner

    # Collapsed space changes
    added_collapsed_keys: List[str] = field(default_factory=list)

    # Phase/turn changes
    original_current_player: Optional[int] = None
    original_phase: Optional[LightweightPhase] = None


class LightweightState:
    """
    Pydantic-free game state for fast make/unmake evaluation.

    This class provides:
    - O(1) conversion from GameState (shallow copies only)
    - In-place mutation with undo tracking
    - Fast access to board elements via dict lookups

    Usage:
        state = LightweightState.from_game_state(game_state)
        undo = state.make_move(move)
        score = evaluate(state)
        state.unmake_move(undo)
    """

    __slots__ = [
        "board_type",
        "board_size",
        "stacks",
        "markers",
        "collapsed_spaces",
        "territories",
        "players",
        "current_player",
        "current_phase",
        "victory_rings",
        "victory_territory",
        "_position_key_cache",
    ]

    def __init__(self):
        self.board_type: LightweightBoardType = LightweightBoardType.SQUARE8
        self.board_size: int = 8
        self.stacks: Dict[str, LightweightStack] = {}
        self.markers: Dict[str, LightweightMarker] = {}
        self.collapsed_spaces: Dict[str, bool] = {}
        self.territories: Dict[str, int] = {}  # position_key -> owner player
        self.players: Dict[int, LightweightPlayer] = {}
        self.current_player: int = 1
        self.current_phase: LightweightPhase = LightweightPhase.RING_PLACEMENT
        # Default for square8 2-player (RR-CANON-R061: ringsPerPlayer for 2p).
        self.victory_rings: int = 18
        self.victory_territory: int = 33
        self._position_key_cache: Dict[Tuple[int, int], str] = {}

    @classmethod
    def from_game_state(cls, game_state) -> "LightweightState":
        """Convert a Pydantic GameState to lightweight representation.

        This is O(n) where n is board elements, but only done once per
        select_move call instead of once per candidate move.
        """
        state = cls()

        # Board type
        board = game_state.board
        board_type_str = board.type.value if hasattr(board.type, "value") else str(board.type)
        state.board_type = LightweightBoardType(board_type_str)
        state.board_size = board.size

        # Convert stacks
        for key, stack in board.stacks.items():
            state.stacks[key] = LightweightStack(
                position_key=key,
                rings=list(stack.rings),  # Copy the list
                controlling_player=stack.controlling_player,
            )

        # Convert markers
        for key, marker in board.markers.items():
            state.markers[key] = LightweightMarker(
                position_key=key,
                player=marker.player,
            )

        # Collapsed spaces (just keys)
        state.collapsed_spaces = dict(board.collapsed_spaces)

        # Territories
        state.territories = dict(board.territories)

        # Players
        for player in game_state.players:
            pnum = player.player_number
            state.players[pnum] = LightweightPlayer(
                player_number=pnum,
                rings_in_hand=player.rings_in_hand,
                eliminated_rings=player.eliminated_rings,
                territory_spaces=player.territory_spaces,
            )

        # Game state
        state.current_player = game_state.current_player
        phase_str = (
            game_state.current_phase.value
            if hasattr(game_state.current_phase, "value")
            else str(game_state.current_phase)
        )
        state.current_phase = LightweightPhase(phase_str)

        # Victory conditions (RR-CANON-R061/R062).
        from app.rules.core import get_territory_victory_threshold, get_victory_threshold

        num_players = len(game_state.players)
        default_victory_threshold = get_victory_threshold(game_state.board.type, num_players)
        default_territory_threshold = get_territory_victory_threshold(game_state.board.type)

        state.victory_rings = getattr(game_state, "victory_threshold", default_victory_threshold)
        state.victory_territory = getattr(
            game_state, "territory_victory_threshold", default_territory_threshold
        )

        return state

    def get_position_key(self, x: int, y: int) -> str:
        """Cached position key generation."""
        coord = (x, y)
        if coord not in self._position_key_cache:
            self._position_key_cache[coord] = f"{x},{y}"
        return self._position_key_cache[coord]

    def make_place_ring(self, to_key: str, player: int) -> MoveUndo:
        """Apply a PLACE_RING move and return undo information."""
        undo = MoveUndo()

        # Record original player state
        p = self.players[player]
        undo.player_rings_in_hand[player] = p.rings_in_hand

        # Check if placing on existing stack
        if to_key in self.stacks:
            # Save original stack state
            orig = self.stacks[to_key]
            undo.modified_stacks[to_key] = LightweightStack(
                position_key=to_key,
                rings=list(orig.rings),
                controlling_player=orig.controlling_player,
            )
            # Add ring to stack
            orig.rings.append(player)
            orig.controlling_player = player
        else:
            # Create new stack
            self.stacks[to_key] = LightweightStack(
                position_key=to_key,
                rings=[player],
                controlling_player=player,
            )
            undo.added_stack_keys.append(to_key)

        # Decrement rings in hand
        p.rings_in_hand -= 1

        return undo

    def make_move_stack(self, from_key: str, to_key: str, player: int) -> MoveUndo:
        """Apply a MOVE_STACK move and return undo information."""
        undo = MoveUndo()

        if from_key not in self.stacks:
            return undo  # Invalid move, no-op

        from_stack = self.stacks[from_key]

        # Save original from_stack
        undo.modified_stacks[from_key] = LightweightStack(
            position_key=from_key,
            rings=list(from_stack.rings),
            controlling_player=from_stack.controlling_player,
        )

        # Move leaves a marker
        if from_key not in self.markers:
            self.markers[from_key] = LightweightMarker(
                position_key=from_key,
                player=player,
            )
            undo.added_marker_keys.append(from_key)

        if to_key in self.stacks:
            # Moving onto existing stack
            to_stack = self.stacks[to_key]
            undo.modified_stacks[to_key] = LightweightStack(
                position_key=to_key,
                rings=list(to_stack.rings),
                controlling_player=to_stack.controlling_player,
            )
            # Merge stacks: from_stack goes on top of to_stack
            to_stack.rings.extend(from_stack.rings)
            to_stack.controlling_player = from_stack.controlling_player
        else:
            # Moving to empty space - create new stack
            self.stacks[to_key] = LightweightStack(
                position_key=to_key,
                rings=list(from_stack.rings),
                controlling_player=from_stack.controlling_player,
            )
            undo.added_stack_keys.append(to_key)

        # Remove from_stack
        del self.stacks[from_key]
        undo.removed_stacks[from_key] = undo.modified_stacks[from_key]
        del undo.modified_stacks[from_key]  # It was removed, not modified

        return undo

    def make_capture(self, from_key: str, to_key: str, player: int) -> MoveUndo:
        """Apply a capture move and return undo information."""
        undo = MoveUndo()

        if from_key not in self.stacks or to_key not in self.stacks:
            return undo  # Invalid move

        from_stack = self.stacks[from_key]
        to_stack = self.stacks[to_key]

        # Save original states
        undo.modified_stacks[from_key] = LightweightStack(
            position_key=from_key,
            rings=list(from_stack.rings),
            controlling_player=from_stack.controlling_player,
        )
        undo.modified_stacks[to_key] = LightweightStack(
            position_key=to_key,
            rings=list(to_stack.rings),
            controlling_player=to_stack.controlling_player,
        )

        # Leave marker at from position
        if from_key not in self.markers:
            self.markers[from_key] = LightweightMarker(
                position_key=from_key,
                player=player,
            )
            undo.added_marker_keys.append(from_key)

        # Capture: from_stack lands on to_stack and takes control
        to_stack.rings.extend(from_stack.rings)
        to_stack.controlling_player = from_stack.controlling_player

        # Remove from_stack
        del self.stacks[from_key]
        undo.removed_stacks[from_key] = undo.modified_stacks[from_key]
        del undo.modified_stacks[from_key]

        return undo

    def unmake_move(self, undo: MoveUndo) -> None:
        """Restore state to before the move using undo information."""

        # Restore removed stacks
        for key, stack in undo.removed_stacks.items():
            self.stacks[key] = LightweightStack(
                position_key=stack.position_key,
                rings=list(stack.rings),
                controlling_player=stack.controlling_player,
            )

        # Remove added stacks
        for key in undo.added_stack_keys:
            if key in self.stacks:
                del self.stacks[key]

        # Restore modified stacks
        for key, stack in undo.modified_stacks.items():
            self.stacks[key] = LightweightStack(
                position_key=stack.position_key,
                rings=list(stack.rings),
                controlling_player=stack.controlling_player,
            )

        # Restore removed markers
        for key, marker in undo.removed_markers.items():
            self.markers[key] = LightweightMarker(
                position_key=marker.position_key,
                player=marker.player,
            )

        # Remove added markers
        for key in undo.added_marker_keys:
            if key in self.markers:
                del self.markers[key]

        # Restore player states
        for pnum, rings in undo.player_rings_in_hand.items():
            self.players[pnum].rings_in_hand = rings
        for pnum, elim in undo.player_eliminated_rings.items():
            self.players[pnum].eliminated_rings = elim
        for pnum, terr in undo.player_territory_spaces.items():
            self.players[pnum].territory_spaces = terr

        # Restore territories
        for key in undo.added_territory_keys:
            if key in self.territories:
                del self.territories[key]
        for key, owner in undo.removed_territories.items():
            self.territories[key] = owner
        for key, owner in undo.modified_territories.items():
            self.territories[key] = owner

        # Restore collapsed spaces
        for key in undo.added_collapsed_keys:
            if key in self.collapsed_spaces:
                del self.collapsed_spaces[key]

        # Restore phase/turn
        if undo.original_current_player is not None:
            self.current_player = undo.original_current_player
        if undo.original_phase is not None:
            self.current_phase = undo.original_phase

    def get_player(self, player_number: int) -> Optional[LightweightPlayer]:
        """Get player by number."""
        return self.players.get(player_number)

    def count_player_stacks(self, player_number: int) -> int:
        """Count stacks controlled by player."""
        return sum(1 for s in self.stacks.values() if s.controlling_player == player_number)

    def count_player_markers(self, player_number: int) -> int:
        """Count markers owned by player."""
        return sum(1 for m in self.markers.values() if m.player == player_number)

    def count_player_territory(self, player_number: int) -> int:
        """Count territory spaces owned by player."""
        return sum(1 for owner in self.territories.values() if owner == player_number)

    def total_rings_on_board(self, player_number: int) -> int:
        """Count total rings on board for player."""
        total = 0
        for stack in self.stacks.values():
            total += sum(1 for r in stack.rings if r == player_number)
        return total
