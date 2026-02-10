"""
Mutable Game State for efficient make/unmake move operations.

This module implements the make/unmake move pattern for tree-search algorithms
(Minimax, Alpha-Beta, MCTS) to avoid object creation overhead. Instead of
creating new GameState copies for each move, we modify state in-place and
capture undo information in a MoveUndo token.

See design doc: ai-service/docs/MAKE_UNMAKE_DESIGN.md

Thread Safety: NOT thread-safe. Each search thread should have its own
MutableGameState instance.

Usage:
    state = MutableGameState.from_immutable(game_state)
    undo = state.make_move(move)
    score = evaluate(state)
    state.unmake_move(undo)
"""
from __future__ import annotations


import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

from app.core.zobrist import ZobristHash
from app.models import (
    BoardState,
    BoardType,
    ChainCaptureState,
    GamePhase,
    GameState,
    GameStatus,
    LineInfo,
    MarkerInfo,
    Move,
    MoveType,
    Player,
    Position,
    RingStack,
    Territory,
    TimeControl,
)
from app.rules.core import get_effective_line_length

# =============================================================================
# Helper Classes
# =============================================================================


@dataclass
class MutableStack:
    """Mutable representation of a ring stack for efficient in-place updates.

    Unlike the Pydantic RingStack model, this class allows direct mutation
    of all fields without validation overhead.
    """
    position: Position
    rings: list[int]  # Player numbers from bottom to top
    stack_height: int
    cap_height: int
    controlling_player: int

    @classmethod
    def from_ring_stack(cls, stack: RingStack) -> "MutableStack":
        """Create mutable stack from immutable RingStack."""
        return cls(
            position=stack.position,
            rings=list(stack.rings),  # Make a copy
            stack_height=stack.stack_height,
            cap_height=stack.cap_height,
            controlling_player=stack.controlling_player,
        )

    def to_ring_stack(self) -> RingStack:
        """Convert back to immutable RingStack."""
        return RingStack(
            position=self.position,
            rings=list(self.rings),
            stackHeight=self.stack_height,
            capHeight=self.cap_height,
            controllingPlayer=self.controlling_player,
        )

    def copy(self) -> "MutableStack":
        """Create a deep copy of this stack."""
        return MutableStack(
            position=self.position,
            rings=list(self.rings),
            stack_height=self.stack_height,
            cap_height=self.cap_height,
            controlling_player=self.controlling_player,
        )

    def recompute_properties(self) -> None:
        """Recompute stack properties from rings list."""
        self.stack_height = len(self.rings)
        if self.stack_height == 0:
            self.controlling_player = 0
            self.cap_height = 0
        else:
            self.controlling_player = self.rings[-1]
            h = 0
            for r in reversed(self.rings):
                if r == self.controlling_player:
                    h += 1
                else:
                    break
            self.cap_height = h


@dataclass
class MutableMarker:
    """Mutable representation of a marker for efficient in-place updates."""
    player: int
    position: Position
    marker_type: str = "regular"

    @classmethod
    def from_marker_info(cls, marker: MarkerInfo) -> "MutableMarker":
        """Create mutable marker from immutable MarkerInfo."""
        return cls(
            player=marker.player,
            position=marker.position,
            marker_type=marker.type,
        )

    def to_marker_info(self) -> MarkerInfo:
        """Convert back to immutable MarkerInfo."""
        return MarkerInfo(
            player=self.player,
            position=self.position,
            type=self.marker_type,
        )

    def copy(self) -> "MutableMarker":
        """Create a copy of this marker."""
        return MutableMarker(
            player=self.player,
            position=self.position,
            marker_type=self.marker_type,
        )


@dataclass
class MutablePlayerState:
    """Mutable player state for efficient in-place updates."""
    player_number: int
    rings_in_hand: int
    eliminated_rings: int
    territory_spaces: int

    # Reference fields (not typically mutated during search)
    id: str = ""
    username: str = ""
    player_type: str = "ai"
    is_ready: bool = True
    time_remaining: int = 0
    ai_difficulty: int | None = None

    @classmethod
    def from_player(cls, player: Player) -> "MutablePlayerState":
        """Create mutable player state from immutable Player."""
        return cls(
            player_number=player.player_number,
            rings_in_hand=player.rings_in_hand,
            eliminated_rings=player.eliminated_rings,
            territory_spaces=player.territory_spaces,
            id=player.id,
            username=player.username,
            player_type=player.type,
            is_ready=player.is_ready,
            time_remaining=player.time_remaining,
            ai_difficulty=player.ai_difficulty,
        )

    def to_player(self) -> Player:
        """Convert back to immutable Player."""
        return Player(
            id=self.id,
            username=self.username,
            type=self.player_type,
            playerNumber=self.player_number,
            isReady=self.is_ready,
            timeRemaining=self.time_remaining,
            aiDifficulty=self.ai_difficulty,
            ringsInHand=self.rings_in_hand,
            eliminatedRings=self.eliminated_rings,
            territorySpaces=self.territory_spaces,
        )

    def copy(self) -> "MutablePlayerState":
        """Create a copy of this player state."""
        return MutablePlayerState(
            player_number=self.player_number,
            rings_in_hand=self.rings_in_hand,
            eliminated_rings=self.eliminated_rings,
            territory_spaces=self.territory_spaces,
            id=self.id,
            username=self.username,
            player_type=self.player_type,
            is_ready=self.is_ready,
            time_remaining=self.time_remaining,
            ai_difficulty=self.ai_difficulty,
        )


# =============================================================================
# MoveUndo Token
# =============================================================================


@dataclass
class MoveUndo:
    """Token capturing all state changes for move reversal.

    This structure is designed to capture the minimal information needed
    to reverse any move type. It uses dictionaries to store original
    values only for fields that were actually modified.

    Memory Footprint Estimate:
    - Simple placement: ~100-200 bytes
    - Movement with path markers: ~200-500 bytes
    - Complex capture chain: ~500-1000 bytes

    This is much smaller than a full GameState copy (~5-20KB).
    """

    # The move being undone (for debugging/validation)
    move: Move

    # === Stack Changes ===
    # Stacks that were removed (key -> original stack copy)
    removed_stacks: dict[str, MutableStack] = field(default_factory=dict)
    # Keys of stacks that were added (will be deleted on undo)
    added_stacks: set[str] = field(default_factory=set)
    # Stacks modified (key -> original stack copy before modification)
    modified_stacks: dict[str, MutableStack] = field(default_factory=dict)

    # === Marker Changes ===
    # Markers that were removed (key -> original marker copy)
    removed_markers: dict[str, MutableMarker] = field(default_factory=dict)
    # Keys of markers that were added (will be deleted on undo)
    added_markers: set[str] = field(default_factory=set)
    # Markers that were modified (key -> original marker copy)
    modified_markers: dict[str, MutableMarker] = field(default_factory=dict)

    # === Collapsed Space Changes ===
    # Collapsed spaces that were removed (key -> original owner)
    removed_collapsed: dict[str, int] = field(default_factory=dict)
    # Keys of collapsed spaces that were added
    added_collapsed: set[str] = field(default_factory=set)

    # === Player State Changes ===
    # Previous rings_in_hand for modified players (player_number -> count)
    prev_rings_in_hand: dict[int, int] = field(default_factory=dict)
    # Previous eliminated_rings for modified players (player_number -> count)
    prev_eliminated_rings: dict[int, int] = field(default_factory=dict)
    # Previous territory_spaces for modified players (player_number -> count)
    prev_territory_spaces: dict[int, int] = field(default_factory=dict)

    # === Board Aggregate Changes ===
    # Previous board.eliminated_rings dict entries (player_id_str -> count)
    prev_board_eliminated_rings: dict[str, int] = field(default_factory=dict)
    # Previous total_rings_eliminated
    prev_total_rings_eliminated: int = 0

    # === Turn/Phase State ===
    prev_zobrist_hash: int = 0
    prev_phase: GamePhase | None = None
    prev_player: int = 0
    prev_chain_capture_state: ChainCaptureState | None = None
    prev_must_move_from_stack_key: str | None = None
    prev_game_status: GameStatus | None = None
    prev_winner: int | None = None

    # === LPS Tracking (for victory detection) ===
    prev_lps_round_index: int = 0
    prev_lps_actor_mask: dict[int, bool] = field(default_factory=dict)
    prev_lps_current_round_first_player: int | None = None
    prev_lps_exclusive_player: int | None = None
    prev_lps_consecutive_exclusive_rounds: int = 0
    prev_lps_consecutive_exclusive_player: int | None = None

    # === Optional for chain captures ===
    chain_capture_stack: list["MoveUndo"] | None = None


# =============================================================================
# MutableGameState
# =============================================================================


class MutableGameState:
    """Mutable wrapper around GameState for efficient search.

    This class wraps an existing GameState and provides make_move/unmake_move
    operations that modify the underlying state in-place, avoiding object
    creation overhead during tree search.

    Thread Safety: NOT thread-safe. Each search thread should have its own
    MutableGameState instance.

    Usage:
        state = MutableGameState.from_immutable(game_state)
        undo = state.make_move(move)
        score = evaluate(state)
        state.unmake_move(undo)
    """

    def __init__(self) -> None:
        """Initialize empty mutable state. Use from_immutable() to create."""
        # Internal mutable state
        self._stacks: dict[str, MutableStack] = {}
        self._markers: dict[str, MutableMarker] = {}
        self._collapsed: dict[str, int] = {}  # pos_key -> owner
        # Board eliminated rings: player_id_str -> count
        self._board_eliminated_rings: dict[str, int] = {}

        # Player state (mutable): player_number -> state
        self._players: dict[int, MutablePlayerState] = {}

        # Turn/phase state
        self._phase: GamePhase = GamePhase.RING_PLACEMENT
        self._active_player: int = 1
        self._chain_capture_state: ChainCaptureState | None = None
        self._must_move_from_stack_key: str | None = None

        # Aggregate tracking
        self._total_rings_eliminated: int = 0
        self._zobrist_hash: int = 0

        # LPS tracking for victory detection
        self._lps_round_index: int = 0
        self._lps_current_round_actor_mask: dict[int, bool] = {}
        self._lps_current_round_first_player: int | None = None
        self._lps_exclusive_player_for_completed_round: int | None = None
        self._lps_consecutive_exclusive_rounds: int = 0
        self._lps_consecutive_exclusive_player: int | None = None
        self._lps_rounds_required: int = 3  # Configurable LPS threshold

        # Immutable reference fields (for context and conversion back)
        self._id: str = ""
        self._board_type: BoardType = BoardType.SQUARE8
        self._board_size: int = 8
        self._rng_seed: int | None = None
        self._game_status: GameStatus = GameStatus.ACTIVE
        self._winner: int | None = None
        self._victory_threshold: int = 6
        self._territory_victory_threshold: int = 20
        self._total_rings_in_play: int = 36
        self._time_control: TimeControl | None = None
        self._spectators: list[str] = []
        self._created_at: datetime = datetime.now()
        self._last_move_at: datetime = datetime.now()
        self._is_rated: bool = False
        self._max_players: int = 2
        self._move_history: list[Move] = []

        # Lines and territories (usually not mutated during search)
        self._formed_lines: list[LineInfo] = []
        self._territories: dict[str, Territory] = {}

        # Zobrist hash helper (singleton)
        self._zobrist: ZobristHash = ZobristHash()

    @classmethod
    def from_immutable(cls, state: GameState) -> "MutableGameState":
        """Create mutable state from immutable GameState.

        NOTE: Copies dictionary contents for independent mutable state.
        """
        mutable = cls()

        # Copy board state
        for pos_key, stack in state.board.stacks.items():
            mutable._stacks[pos_key] = MutableStack.from_ring_stack(stack)

        for pos_key, marker in state.board.markers.items():
            mutable._markers[pos_key] = MutableMarker.from_marker_info(marker)

        mutable._collapsed = dict(state.board.collapsed_spaces)
        mutable._board_eliminated_rings = dict(state.board.eliminated_rings)

        # Copy player state
        for player in state.players:
            pnum = player.player_number
            mutable._players[pnum] = MutablePlayerState.from_player(player)

        # Copy turn/phase state
        mutable._phase = state.current_phase
        mutable._active_player = state.current_player
        mutable._chain_capture_state = state.chain_capture_state
        mutable._must_move_from_stack_key = state.must_move_from_stack_key

        # Copy aggregate tracking
        mutable._total_rings_eliminated = state.total_rings_eliminated
        mutable._zobrist_hash = state.zobrist_hash or 0

        # Copy LPS tracking
        mutable._lps_round_index = state.lps_round_index
        mask = state.lps_current_round_actor_mask
        mutable._lps_current_round_actor_mask = dict(mask)
        mutable._lps_current_round_first_player = (
            state.lps_current_round_first_player
        )
        mutable._lps_exclusive_player_for_completed_round = (
            state.lps_exclusive_player_for_completed_round
        )
        mutable._lps_consecutive_exclusive_rounds = (
            state.lps_consecutive_exclusive_rounds
        )
        mutable._lps_consecutive_exclusive_player = (
            state.lps_consecutive_exclusive_player
        )
        mutable._lps_rounds_required = getattr(state, 'lps_rounds_required', 3)

        # Copy immutable reference fields
        mutable._id = state.id
        mutable._board_type = state.board_type
        mutable._board_size = state.board.size
        mutable._rng_seed = state.rng_seed
        mutable._game_status = state.game_status
        mutable._winner = state.winner
        mutable._victory_threshold = state.victory_threshold
        mutable._territory_victory_threshold = (
            state.territory_victory_threshold
        )
        mutable._total_rings_in_play = state.total_rings_in_play
        mutable._time_control = state.time_control
        mutable._spectators = list(state.spectators)
        mutable._created_at = state.created_at
        mutable._last_move_at = state.last_move_at
        mutable._is_rated = state.is_rated
        mutable._max_players = state.max_players
        mutable._move_history = list(state.move_history)

        # Copy lines and territories
        mutable._formed_lines = list(state.board.formed_lines)
        mutable._territories = dict(state.board.territories)

        # Initialize Zobrist hash if not present
        has_board_content = mutable._stacks or mutable._markers
        if mutable._zobrist_hash == 0 and has_board_content:
            mutable._zobrist_hash = mutable._compute_zobrist_hash()

        return mutable

    def to_immutable(self) -> GameState:
        """Convert back to immutable GameState.

        This is typically used when we need to pass the state to external APIs
        or for debugging/validation.
        """
        # Build board state
        stacks = {
            pos_key: stack.to_ring_stack()
            for pos_key, stack in self._stacks.items()
        }
        markers = {
            pos_key: marker.to_marker_info()
            for pos_key, marker in self._markers.items()
        }

        board = BoardState(
            type=self._board_type,
            size=self._board_size,
            stacks=stacks,
            markers=markers,
            collapsedSpaces=dict(self._collapsed),
            eliminatedRings=dict(self._board_eliminated_rings),
            formedLines=list(self._formed_lines),
            territories=dict(self._territories),
        )

        # Build player list (sorted by player_number for consistency)
        players = [
            self._players[pn].to_player()
            for pn in sorted(self._players.keys())
        ]

        # Build time control (use default if None)
        time_ctrl = self._time_control
        if time_ctrl is None:
            time_ctrl = TimeControl(
                initialTime=600000,
                increment=5000,
                type="fischer",
            )

        lps_exclusive = self._lps_exclusive_player_for_completed_round

        return GameState(
            id=self._id,
            boardType=self._board_type,
            rngSeed=self._rng_seed,
            board=board,
            players=players,
            currentPhase=self._phase,
            currentPlayer=self._active_player,
            moveHistory=list(self._move_history),
            timeControl=time_ctrl,
            spectators=list(self._spectators),
            gameStatus=self._game_status,
            winner=self._winner,
            createdAt=self._created_at,
            lastMoveAt=self._last_move_at,
            isRated=self._is_rated,
            maxPlayers=self._max_players,
            totalRingsInPlay=self._total_rings_in_play,
            totalRingsEliminated=self._total_rings_eliminated,
            victoryThreshold=self._victory_threshold,
            territoryVictoryThreshold=self._territory_victory_threshold,
            chainCaptureState=self._chain_capture_state,
            mustMoveFromStackKey=self._must_move_from_stack_key,
            zobristHash=self._zobrist_hash,
            lpsRoundIndex=self._lps_round_index,
            lpsCurrentRoundActorMask=dict(self._lps_current_round_actor_mask),
            lpsCurrentRoundFirstPlayer=self._lps_current_round_first_player,
            lpsExclusivePlayerForCompletedRound=lps_exclusive,
            lpsConsecutiveExclusiveRounds=self._lps_consecutive_exclusive_rounds,
            lpsConsecutiveExclusivePlayer=self._lps_consecutive_exclusive_player,
            lpsRoundsRequired=self._lps_rounds_required,
        )

    def _compute_zobrist_hash(self) -> int:
        """Compute full Zobrist hash from scratch (expensive, O(N))."""
        h = 0

        # Stacks
        for pos_key, stack in self._stacks.items():
            h ^= self._zobrist.get_stack_hash(
                pos_key,
                stack.controlling_player,
                stack.stack_height,
                tuple(stack.rings)
            )

        # Markers
        for pos_key, marker in self._markers.items():
            h ^= self._zobrist.get_marker_hash(pos_key, marker.player)

        # Collapsed
        for pos_key in self._collapsed:
            h ^= self._zobrist.get_collapsed_hash(pos_key)

        # Global state
        h ^= self._zobrist.get_player_hash(self._active_player)
        h ^= self._zobrist.get_phase_hash(self._phase.value)

        return h

    # =========================================================================
    # Properties for read access
    # =========================================================================

    @property
    def zobrist_hash(self) -> int:
        """Current Zobrist hash for transposition table lookup."""
        return self._zobrist_hash

    @property
    def current_player(self) -> int:
        """Current player number."""
        return self._active_player

    @property
    def current_phase(self) -> GamePhase:
        """Current game phase."""
        return self._phase

    @property
    def board_type(self) -> BoardType:
        """Board type (immutable)."""
        return self._board_type

    @property
    def board_size(self) -> int:
        """Board size (immutable)."""
        return self._board_size

    @property
    def stacks(self) -> dict[str, MutableStack]:
        """Read access to stacks dictionary."""
        return self._stacks

    @property
    def markers(self) -> dict[str, MutableMarker]:
        """Read access to markers dictionary."""
        return self._markers

    @property
    def collapsed_spaces(self) -> dict[str, int]:
        """Read access to collapsed spaces dictionary."""
        return self._collapsed

    @property
    def players(self) -> dict[int, MutablePlayerState]:
        """Read access to players dictionary."""
        return self._players

    @property
    def game_status(self) -> GameStatus:
        """Current game status."""
        return self._game_status

    @property
    def winner(self) -> int | None:
        """Winner player number if game is over."""
        return self._winner

    @property
    def total_rings_eliminated(self) -> int:
        """Total rings eliminated."""
        return self._total_rings_eliminated

    @property
    def chain_capture_state(self) -> ChainCaptureState | None:
        """Current chain capture state."""
        return self._chain_capture_state

    @property
    def must_move_from_stack_key(self) -> str | None:
        """Stack key that must be moved from (after placement)."""
        return self._must_move_from_stack_key

    # LPS tracking properties (for game_engine.py compatibility)
    @property
    def lps_consecutive_exclusive_rounds(self) -> int:
        """Number of consecutive rounds with exclusive player."""
        return self._lps_consecutive_exclusive_rounds

    @lps_consecutive_exclusive_rounds.setter
    def lps_consecutive_exclusive_rounds(self, value: int) -> None:
        self._lps_consecutive_exclusive_rounds = value

    @property
    def lps_consecutive_exclusive_player(self) -> int | None:
        """Player who has been exclusive for consecutive rounds."""
        return self._lps_consecutive_exclusive_player

    @lps_consecutive_exclusive_player.setter
    def lps_consecutive_exclusive_player(self, value: int | None) -> None:
        self._lps_consecutive_exclusive_player = value

    @property
    def lps_rounds_required(self) -> int:
        """Number of consecutive exclusive rounds required for LPS victory."""
        return self._lps_rounds_required

    @property
    def lps_round_index(self) -> int:
        """Current LPS round index."""
        return self._lps_round_index

    @lps_round_index.setter
    def lps_round_index(self, value: int) -> None:
        self._lps_round_index = value

    @property
    def lps_current_round_actor_mask(self) -> dict[int, bool]:
        """Actor mask for current LPS round."""
        return self._lps_current_round_actor_mask

    @property
    def lps_current_round_first_player(self) -> int | None:
        """First player of current LPS round."""
        return self._lps_current_round_first_player

    @lps_current_round_first_player.setter
    def lps_current_round_first_player(self, value: int | None) -> None:
        self._lps_current_round_first_player = value

    @property
    def lps_exclusive_player_for_completed_round(self) -> int | None:
        """Exclusive player for last completed round."""
        return self._lps_exclusive_player_for_completed_round

    @lps_exclusive_player_for_completed_round.setter
    def lps_exclusive_player_for_completed_round(self, value: int | None) -> None:
        self._lps_exclusive_player_for_completed_round = value

    @property
    def num_players(self) -> int:
        """Number of players in the game."""
        return self._max_players

    @property
    def max_players(self) -> int:
        """Maximum number of players in the game."""
        return self._max_players

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_cap_height(self, rings: list[int]) -> int:
        """Calculate the cap height of a stack from its rings list."""
        if not rings:
            return 0
        controlling_player = rings[-1]
        height = 0
        for r in reversed(rings):
            if r == controlling_player:
                height += 1
            else:
                break
        return height

    def get_stack_at(self, pos: Position) -> MutableStack | None:
        """Get stack at position, or None."""
        return self._stacks.get(pos.to_key())

    def get_marker_at(self, pos: Position) -> MutableMarker | None:
        """Get marker at position, or None."""
        return self._markers.get(pos.to_key())

    def is_collapsed(self, pos: Position) -> bool:
        """Check if position is a collapsed space."""
        return pos.to_key() in self._collapsed

    def get_player(self, player_number: int) -> MutablePlayerState | None:
        """Get player state by player number."""
        return self._players.get(player_number)

    # =========================================================================
    # Make/Unmake Move Methods
    # =========================================================================

    def make_move(self, move: Move) -> MoveUndo:
        """Apply move in-place and return undo token.

        This modifies the internal state directly without creating
        new objects. The returned MoveUndo token captures all changes
        needed to reverse the operation.

        Args:
            move: The move to apply

        Returns:
            MoveUndo token for reversing this move

        Note:
            For Phase 1, this implements placement moves and provides
            skeleton implementations for other move types that raise
            NotImplementedError.
        """
        undo = MoveUndo(move=move)

        # Capture pre-move state for undo
        undo.prev_zobrist_hash = self._zobrist_hash
        undo.prev_phase = self._phase
        undo.prev_player = self._active_player
        undo.prev_chain_capture_state = self._chain_capture_state
        undo.prev_must_move_from_stack_key = self._must_move_from_stack_key
        undo.prev_total_rings_eliminated = self._total_rings_eliminated
        undo.prev_game_status = self._game_status
        undo.prev_winner = self._winner

        # Capture LPS state
        undo.prev_lps_round_index = self._lps_round_index
        undo.prev_lps_actor_mask = dict(self._lps_current_round_actor_mask)
        undo.prev_lps_current_round_first_player = (
            self._lps_current_round_first_player
        )
        lps_excl = self._lps_exclusive_player_for_completed_round
        undo.prev_lps_exclusive_player = lps_excl
        undo.prev_lps_consecutive_exclusive_rounds = (
            self._lps_consecutive_exclusive_rounds
        )
        undo.prev_lps_consecutive_exclusive_player = (
            self._lps_consecutive_exclusive_player
        )

        # Remove phase/player from hash before changes
        player_hash = self._zobrist.get_player_hash(self._active_player)
        phase_hash = self._zobrist.get_phase_hash(self._phase.value)
        self._zobrist_hash ^= player_hash
        self._zobrist_hash ^= phase_hash

        # Dispatch to move-type-specific handler
        if move.type == MoveType.PLACE_RING:
            self._make_place_ring(move, undo)
        elif move.type == MoveType.SKIP_PLACEMENT:
            # No board changes; just update must_move_from_stack_key
            pass
        elif move.type == MoveType.MOVE_STACK:
            self._make_move_stack(move, undo)
        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.CHAIN_CAPTURE,
        ):
            self._make_chain_capture(move, undo)
        elif move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            self._make_process_line(move, undo)
        elif move.type in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            self._make_process_territory(move, undo)
        elif move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            self._make_eliminate_rings_from_stack(move, undo)
        elif move.type == MoveType.FORCED_ELIMINATION:
            self._make_forced_elimination(move, undo)
        elif move.type == MoveType.RECOVERY_SLIDE:
            self._make_recovery_slide(move, undo)
        elif move.type == MoveType.SKIP_RECOVERY:
            # No board changes - just phase transition (handled below)
            pass

        # Update must_move_from_stack_key
        if (move.type == MoveType.PLACE_RING and move.to) or ((
            self._must_move_from_stack_key is not None
            and move.from_pos is not None
            and move.to is not None
            and move.type in (
                MoveType.MOVE_STACK,
                MoveType.OVERTAKING_CAPTURE,
                MoveType.CONTINUE_CAPTURE_SEGMENT,
                MoveType.CHAIN_CAPTURE,
            )
        ) and move.from_pos.to_key() == self._must_move_from_stack_key):
            self._must_move_from_stack_key = move.to.to_key()

        # Update phase (simplified for search)
        self._update_phase_for_search(move)

        # Add new phase/player to hash
        player_hash = self._zobrist.get_player_hash(self._active_player)
        phase_hash = self._zobrist.get_phase_hash(self._phase.value)
        self._zobrist_hash ^= player_hash
        self._zobrist_hash ^= phase_hash

        return undo

    def unmake_move(self, undo: MoveUndo) -> None:
        """Reverse a move using the undo token.

        Restores the state to exactly what it was before make_move()
        was called with the corresponding move.

        Args:
            undo: The MoveUndo token returned from make_move()
        """
        # Restore stacks
        for key, stack in undo.removed_stacks.items():
            self._stacks[key] = stack.copy()
        for key in undo.added_stacks:
            if key in self._stacks:
                del self._stacks[key]
        for key, stack in undo.modified_stacks.items():
            self._stacks[key] = stack.copy()

        # Restore markers
        for key, marker in undo.removed_markers.items():
            self._markers[key] = marker.copy()
        for key in undo.added_markers:
            if key in self._markers:
                del self._markers[key]
        for key, marker in undo.modified_markers.items():
            self._markers[key] = marker.copy()

        # Restore collapsed spaces
        for key, owner in undo.removed_collapsed.items():
            self._collapsed[key] = owner
        for key in undo.added_collapsed:
            if key in self._collapsed:
                del self._collapsed[key]

        # Restore player state
        for player_num, rings in undo.prev_rings_in_hand.items():
            if player_num in self._players:
                self._players[player_num].rings_in_hand = rings
        for player_num, elim in undo.prev_eliminated_rings.items():
            if player_num in self._players:
                self._players[player_num].eliminated_rings = elim
        for player_num, terr in undo.prev_territory_spaces.items():
            if player_num in self._players:
                self._players[player_num].territory_spaces = terr

        # Restore aggregate tracking
        self._total_rings_eliminated = undo.prev_total_rings_eliminated

        # Restore board eliminated_rings dict
        for key, count in undo.prev_board_eliminated_rings.items():
            self._board_eliminated_rings[key] = count

        # Restore turn/phase state
        self._zobrist_hash = undo.prev_zobrist_hash
        self._phase = undo.prev_phase or GamePhase.RING_PLACEMENT
        self._active_player = undo.prev_player
        self._chain_capture_state = undo.prev_chain_capture_state
        self._must_move_from_stack_key = undo.prev_must_move_from_stack_key
        self._game_status = undo.prev_game_status or GameStatus.ACTIVE
        self._winner = undo.prev_winner

        # Restore LPS state
        self._lps_round_index = undo.prev_lps_round_index
        self._lps_current_round_actor_mask = dict(undo.prev_lps_actor_mask)
        self._lps_current_round_first_player = (
            undo.prev_lps_current_round_first_player
        )
        lps_excl = undo.prev_lps_exclusive_player
        self._lps_exclusive_player_for_completed_round = lps_excl
        self._lps_consecutive_exclusive_rounds = (
            undo.prev_lps_consecutive_exclusive_rounds
        )
        self._lps_consecutive_exclusive_player = (
            undo.prev_lps_consecutive_exclusive_player
        )

    # =========================================================================
    # Move-Type-Specific Make Methods
    # =========================================================================

    def _make_place_ring(self, move: Move, undo: MoveUndo) -> None:
        """Apply PLACE_RING move in-place."""
        pos_key = move.to.to_key()
        placement_count = move.placement_count or 1
        player = move.player

        # Capture player's previous rings_in_hand
        if player in self._players:
            prev_rings = self._players[player].rings_in_hand
            undo.prev_rings_in_hand[player] = prev_rings

        existing = self._stacks.get(pos_key)

        if existing and existing.stack_height > 0:
            # Capture original stack for undo
            undo.modified_stacks[pos_key] = existing.copy()

            # Remove old stack hash
            self._zobrist_hash ^= self._zobrist.get_stack_hash(
                pos_key,
                existing.controlling_player,
                existing.stack_height,
                tuple(existing.rings)
            )

            # Modify stack in-place
            existing.rings.extend([player] * placement_count)
            existing.recompute_properties()
        else:
            # New stack
            undo.added_stacks.add(pos_key)

            new_rings = [player] * placement_count
            self._stacks[pos_key] = MutableStack(
                position=move.to,
                rings=new_rings,
                stack_height=len(new_rings),
                cap_height=len(new_rings),
                controlling_player=player,
            )

        # Add new stack hash
        new_stack = self._stacks[pos_key]
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            pos_key,
            new_stack.controlling_player,
            new_stack.stack_height,
            tuple(new_stack.rings)
        )

        # Update player rings
        if player in self._players:
            self._players[player].rings_in_hand -= placement_count

    def _make_move_stack(self, move: Move, undo: MoveUndo) -> None:
        """Apply MOVE_STACK move in-place."""
        if not move.from_pos:
            return

        from_key = move.from_pos.to_key()
        to_key = move.to.to_key()

        source_stack = self._stacks.get(from_key)
        if not source_stack:
            return

        # Capture source stack for undo
        undo.removed_stacks[from_key] = source_stack.copy()

        # Remove source stack hash
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            from_key,
            source_stack.controlling_player,
            source_stack.stack_height,
            tuple(source_stack.rings)
        )

        # Remove source stack
        del self._stacks[from_key]

        # Leave marker at departure
        undo.added_markers.add(from_key)
        self._markers[from_key] = MutableMarker(
            player=move.player,
            position=move.from_pos,
            marker_type="regular",
        )
        marker_hash = self._zobrist.get_marker_hash(from_key, move.player)
        self._zobrist_hash ^= marker_hash

        # Process markers along path (simplified for now)
        self._make_process_path_markers(
            move.from_pos, move.to, move.player, undo
        )

        # Handle landing marker - landing on ANY marker (own or opponent)
        # removes the marker and eliminates the top ring of the landing stack.
        # Per canonical rules: "Landing on a marker (own or opponent) is legal;
        # the marker is removed and the top ring of the moving stack's cap is
        # immediately Eliminated (credited to the mover)."
        landing_marker = self._markers.get(to_key)
        landed_on_marker = landing_marker is not None

        if landed_on_marker:
            # Record marker removal
            undo.removed_markers[to_key] = landing_marker.copy()
            lm_hash = self._zobrist.get_marker_hash(
                to_key, landing_marker.player
            )
            self._zobrist_hash ^= lm_hash
            del self._markers[to_key]

        # Handle destination (merge or simple move)
        dest_stack = self._stacks.get(to_key)
        if dest_stack and dest_stack.stack_height > 0:
            undo.modified_stacks[to_key] = dest_stack.copy()

            # Remove old dest hash
            self._zobrist_hash ^= self._zobrist.get_stack_hash(
                to_key,
                dest_stack.controlling_player,
                dest_stack.stack_height,
                tuple(dest_stack.rings)
            )

            # Merge stacks (destination rings at bottom, source on top)
            dest_stack.rings.extend(source_stack.rings)
            dest_stack.recompute_properties()
        else:
            # Simple move
            undo.added_stacks.add(to_key)
            self._stacks[to_key] = MutableStack(
                position=move.to,
                rings=list(source_stack.rings),
                stack_height=source_stack.stack_height,
                cap_height=source_stack.cap_height,
                controlling_player=source_stack.controlling_player,
            )

        # Add new dest hash
        new_dest = self._stacks[to_key]
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            to_key,
            new_dest.controlling_player,
            new_dest.stack_height,
            tuple(new_dest.rings)
        )

        # Self-elimination if landed on ANY marker (own or opponent)
        if landed_on_marker:
            self._make_eliminate_top_ring(move.to, move.player, undo)

    def _make_process_path_markers(
        self,
        from_pos: Position,
        to_pos: Position,
        player: int,
        undo: MoveUndo
    ) -> None:
        """Process markers along movement path (flip or collapse)."""
        # Get path positions (excluding endpoints)
        path = self._get_path_positions(from_pos, to_pos)

        for pos in path[1:-1]:
            pos_key = pos.to_key()
            marker = self._markers.get(pos_key)
            if marker is None:
                continue

            if marker.player == player:
                # Own marker -> collapse
                undo.removed_markers[pos_key] = marker.copy()
                old_hash = self._zobrist.get_marker_hash(
                    pos_key, marker.player
                )
                self._zobrist_hash ^= old_hash
                del self._markers[pos_key]

                undo.added_collapsed.add(pos_key)
                self._collapsed[pos_key] = player
                coll_hash = self._zobrist.get_collapsed_hash(pos_key)
                self._zobrist_hash ^= coll_hash
            else:
                # Opponent marker -> flip
                undo.modified_markers[pos_key] = marker.copy()
                old_hash = self._zobrist.get_marker_hash(
                    pos_key, marker.player
                )
                self._zobrist_hash ^= old_hash
                marker.player = player
                new_hash = self._zobrist.get_marker_hash(pos_key, player)
                self._zobrist_hash ^= new_hash

    def _make_eliminate_top_ring(
        self, position: Position, credited_player: int, undo: MoveUndo
    ) -> None:
        """Eliminate top ring from stack at position."""
        pos_key = position.to_key()
        stack = self._stacks.get(pos_key)
        if not stack or stack.stack_height == 0:
            return

        # Capture original stack BEFORE any modifications if not already recorded
        if pos_key not in undo.modified_stacks:
            undo.modified_stacks[pos_key] = stack.copy()

        # Remove old stack hash
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            pos_key,
            stack.controlling_player,
            stack.stack_height,
            tuple(stack.rings)
        )

        # Record board eliminated rings change
        player_id_str = str(credited_player)
        if player_id_str not in undo.prev_board_eliminated_rings:
            undo.prev_board_eliminated_rings[player_id_str] = (
                self._board_eliminated_rings.get(player_id_str, 0)
            )

        # Record player eliminated rings change
        if credited_player not in undo.prev_eliminated_rings and credited_player in self._players:
            undo.prev_eliminated_rings[credited_player] = (
                self._players[credited_player].eliminated_rings
            )

        # Pop top ring
        stack.rings.pop()
        stack.stack_height -= 1

        # Update eliminated counts
        self._board_eliminated_rings[player_id_str] = (
            self._board_eliminated_rings.get(player_id_str, 0) + 1
        )
        self._total_rings_eliminated += 1

        if credited_player in self._players:
            self._players[credited_player].eliminated_rings += 1

        # Update or remove stack
        if stack.stack_height == 0:
            del self._stacks[pos_key]
        else:
            stack.recompute_properties()
            # Add new stack hash
            self._zobrist_hash ^= self._zobrist.get_stack_hash(
                pos_key,
                stack.controlling_player,
                stack.stack_height,
                tuple(stack.rings)
            )

    def _get_path_positions(
        self, from_pos: Position, to_pos: Position
    ) -> list[Position]:
        """Get all positions along a straight-line path, inclusive."""
        path = [from_pos]

        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        dz_from = from_pos.z or 0
        dz_to = to_pos.z or 0
        dz = dz_to - dz_from

        steps = max(abs(dx), abs(dy), abs(dz))
        if steps == 0:
            return path

        step_x = dx / steps
        step_y = dy / steps
        step_z = dz / steps

        for i in range(1, steps + 1):
            x = round(from_pos.x + step_x * i)
            y = round(from_pos.y + step_y * i)
            pos_kwargs: dict[str, Any] = {"x": x, "y": y}
            # Mirror TS: only include z when the destination carries z.
            if to_pos.z is not None:
                z = round(dz_from + step_z * i)
                pos_kwargs["z"] = z
            path.append(Position(**pos_kwargs))

        return path

    def _update_phase_for_search(self, move: Move) -> None:
        """Phase transition for search with victory detection.

        Determines the next phase based on the move type and current state.
        Also checks for victory conditions.

        Note on RR-CANON-R075 Compliance:
            This method directly assigns self._phase for performance in tree search
            (make/unmake pattern). This is an intentional optimization that bypasses
            the FSM state machine. The phase transitions here MUST mirror the
            canonical FSM transitions in app/rules/fsm.py.

            Validation: Parity tests in tests/unit/fsm/ verify that phase sequences
            produced by this method match canonical FSM transitions.
        """
        # First check for game over after the move
        if self._check_victory_conditions():
            return

        if move.type == MoveType.PLACE_RING:
            # After placement, move to MOVEMENT phase
            self._phase = GamePhase.MOVEMENT
        elif move.type == MoveType.SKIP_PLACEMENT:
            self._phase = GamePhase.MOVEMENT
        elif move.type == MoveType.NO_PLACEMENT_ACTION:
            # Player has no rings to place - advance to MOVEMENT
            self._phase = GamePhase.MOVEMENT
        elif move.type == MoveType.NO_MOVEMENT_ACTION:
            # No movement available - check if player controls stacks for FE
            # Per RR-CANON-R200: if player has stacks but no moves, they must
            # do forced elimination before their turn ends.
            player_stacks = [
                k for k, s in self._stacks.items()
                if s.controlling_player == move.player
            ]
            if player_stacks:
                # Player has stacks but no moves -> FORCED_ELIMINATION phase
                self._phase = GamePhase.FORCED_ELIMINATION
            else:
                # No stacks - end turn
                self._end_turn_for_search()
        elif move.type in (MoveType.MOVE_STACK, MoveType.MOVE_RING):
            # After movement, check for lines or end turn
            # Simplified: end turn (line detection would need BoardManager)
            self._end_turn_for_search()
        elif move.type in (
            MoveType.OVERTAKING_CAPTURE,
            MoveType.CONTINUE_CAPTURE_SEGMENT,
            MoveType.CHAIN_CAPTURE,
        ):
            # After capture, could continue capture chain or end turn
            # Simplified: end turn
            self._end_turn_for_search()
        elif move.type == MoveType.SKIP_CAPTURE:
            # Player declined capture opportunity - proceed to line processing
            # Simplified: end turn (line detection would need BoardManager)
            self._end_turn_for_search()
        elif move.type in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.LINE_FORMATION,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            # After line processing, could go to territory or end turn
            self._end_turn_for_search()
        elif move.type == MoveType.NO_LINE_ACTION:
            # No line to process - transition to territory or end turn
            # Per RR-CANON: NO_LINE_ACTION indicates line phase complete
            self._end_turn_for_search()
        elif move.type in (
            MoveType.PROCESS_TERRITORY_REGION,
            MoveType.TERRITORY_CLAIM,
            MoveType.CHOOSE_TERRITORY_OPTION,
        ):
            # After territory processing, end turn
            self._end_turn_for_search()
        elif move.type in (
            MoveType.SKIP_TERRITORY_PROCESSING,
            MoveType.NO_TERRITORY_ACTION,
        ):
            # Territory phase skipped or no action available - end turn
            self._end_turn_for_search()
        elif move.type == MoveType.ELIMINATE_RINGS_FROM_STACK:
            # After explicit elimination, end turn
            self._end_turn_for_search()
        elif move.type == MoveType.FORCED_ELIMINATION:
            # After forced elimination, end turn
            self._end_turn_for_search()
        elif move.type == MoveType.RECOVERY_SLIDE:
            # After recovery slide, end turn (per RR-CANON-R115)
            self._end_turn_for_search()
        elif move.type == MoveType.SKIP_RECOVERY:
            # Skip recovery preserves buried rings, ends turn (per RR-CANON-R115)
            self._end_turn_for_search()

    def _end_turn_for_search(self) -> None:
        """End turn for search - rotate to next non-eliminated player."""
        # Check for victory before rotating
        if self._check_victory_conditions():
            return

        # Get list of non-eliminated players
        active_players = self._get_active_player_numbers()
        if not active_players:
            return

        # If only one player remains, they win
        if len(active_players) == 1:
            self._winner = active_players[0]
            self._game_status = GameStatus.COMPLETED
            return

        # Find next active player
        current_idx = (
            active_players.index(self._active_player)
            if self._active_player in active_players
            else 0
        )
        next_idx = (current_idx + 1) % len(active_players)
        self._active_player = active_players[next_idx]
        self._phase = GamePhase.RING_PLACEMENT
        self._must_move_from_stack_key = None

    # =========================================================================
    # Victory Detection Methods
    # =========================================================================

    def _count_rings_for_player(self, player_number: int) -> int:
        """Count all rings for a player (on board + in hand).

        Args:
            player_number: The player number to count rings for.

        Returns:
            Total ring count (board rings + hand rings).
        """
        rings_on_board = 0
        for stack in self._stacks.values():
            for ring_owner in stack.rings:
                if ring_owner == player_number:
                    rings_on_board += 1

        rings_in_hand = 0
        if player_number in self._players:
            rings_in_hand = self._players[player_number].rings_in_hand

        return rings_on_board + rings_in_hand

    def _is_player_eliminated(self, player_number: int) -> bool:
        """Check if a player is eliminated.

        A player is eliminated when they have 0 rings on board AND 0 in hand.

        Args:
            player_number: The player number to check.

        Returns:
            True if player is eliminated, False otherwise.
        """
        return self._count_rings_for_player(player_number) == 0

    def _get_active_player_numbers(self) -> list[int]:
        """Get list of non-eliminated player numbers in order.

        Returns:
            List of player numbers who still have rings.
        """
        return [
            pn for pn in sorted(self._players.keys())
            if not self._is_player_eliminated(pn)
        ]

    def get_eliminated_players(self) -> set[int]:
        """Return set of eliminated player IDs.

        Returns:
            Set of player numbers who have been eliminated.
        """
        return {
            pn for pn in self._players
            if self._is_player_eliminated(pn)
        }

    def is_game_over(self) -> bool:
        """Check if game has ended.

        Game ends when:
        - Only one player remains (has rings)
        - GameStatus is COMPLETED

        Returns:
            True if game has ended, False otherwise.
        """
        if self._game_status == GameStatus.COMPLETED:
            return True

        active_players = self._get_active_player_numbers()
        return len(active_players) <= 1

    def get_winner(self) -> int | None:
        """Return winning player ID or None if not over/draw.

        Returns:
            The player number of the winner, or None if game is not over
            or if there's no clear winner.
        """
        if self._winner is not None:
            return self._winner

        if not self.is_game_over():
            return None

        active_players = self._get_active_player_numbers()
        if len(active_players) == 1:
            return active_players[0]

        return None

    def _check_victory_conditions(self) -> bool:
        """Check if victory conditions are met and update state.

        Called after each move to determine if the game has ended.
        Mirrors the checks in GameEngine._check_victory to prevent games
        from continuing past valid endpoints in the search tree.

        Victory conditions checked (in order, matching GameEngine):
        1. Ring elimination threshold (RR-CANON-R061)
        2. Territory victory (RR-CANON-R062-v2)
        3. Last player standing (all others eliminated)

        Returns:
            True if game is over, False otherwise.
        """
        # 1. Ring Elimination Victory (RR-CANON-R061)
        # Check if any player has eliminated enough rings to win.
        # Matches GameEngine._check_victory lines 1264-1270.
        for ps in self._players.values():
            if ps.eliminated_rings >= self._victory_threshold:
                self._winner = ps.player_number
                self._game_status = GameStatus.COMPLETED
                return True

        # 2. Territory Victory (RR-CANON-R062-v2)
        # Victory requires BOTH:
        #   a) Territory >= territory_victory_minimum (floor(totalSpaces/numPlayers) + 1)
        #   b) Territory > sum of all opponent territories
        # Matches GameEngine._check_victory lines 1272-1308.
        if self._collapsed:
            territory_counts: dict[int, int] = {}
            for p_id in self._collapsed.values():
                territory_counts[p_id] = territory_counts.get(p_id, 0) + 1

            total_spaces = self._board_size * self._board_size
            territory_minimum = total_spaces // self._max_players + 1

            for pn, player_territory in territory_counts.items():
                if player_territory < territory_minimum:
                    continue
                opponent_territory = sum(
                    tc for op, tc in territory_counts.items() if op != pn
                )
                if player_territory > opponent_territory:
                    self._winner = pn
                    self._game_status = GameStatus.COMPLETED
                    return True

        # 3. Last Player Standing / All Eliminated
        active_players = self._get_active_player_numbers()

        if len(active_players) == 0:
            # All players eliminated - use deterministic tiebreaker (lowest player number)
            # This mirrors VictoryAggregate.ts - games must ALWAYS have a winner.
            # Per RR-CANON: The game must always produce a winner.
            all_player_numbers = list(range(1, self._max_players + 1))
            self._winner = min(all_player_numbers)
            self._game_status = GameStatus.COMPLETED
            logger.warning(
                f"Degenerate game state: all players eliminated. "
                f"Winner by tiebreaker: {self._winner}"
            )
            return True

        if len(active_players) == 1:
            # Single player remaining - they win
            self._winner = active_players[0]
            self._game_status = GameStatus.COMPLETED
            return True

        return False

    def _make_chain_capture(self, move: Move, undo: MoveUndo) -> None:
        """Apply capture move (OVERTAKING_CAPTURE, CONTINUE_CAPTURE_SEGMENT, CHAIN_CAPTURE).

        Mirrors the TS GameEngine.performOvertakingCapture semantics:
        - Leave a marker at the departure space.
        - Process markers along both legs of the path (from→target and
          target→landing), flipping or collapsing as needed.
        - Remove exactly one ring from the top of the target stack and insert
          it at the bottom of the attacker.
        - Move the attacker stack to the landing space.
        - If landing on own marker, remove it and self-eliminate one ring.
        """
        if not move.from_pos or not move.capture_target:
            return

        from_key = move.from_pos.to_key()
        target_key = move.capture_target.to_key()
        to_key = move.to.to_key()

        attacker = self._stacks.get(from_key)
        target_stack = self._stacks.get(target_key)
        if not attacker or not target_stack:
            return

        # Capture original attacker for undo
        undo.removed_stacks[from_key] = attacker.copy()

        # Remove attacker stack hash
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            from_key,
            attacker.controlling_player,
            attacker.stack_height,
            tuple(attacker.rings)
        )

        # Remove attacker from source
        del self._stacks[from_key]

        # Leave marker at departure
        undo.added_markers.add(from_key)
        self._markers[from_key] = MutableMarker(
            player=move.player,
            position=move.from_pos,
            marker_type="regular",
        )
        marker_hash = self._zobrist.get_marker_hash(from_key, move.player)
        self._zobrist_hash ^= marker_hash

        # Process markers along both path segments
        self._make_process_path_markers(
            move.from_pos, move.capture_target, move.player, undo
        )
        self._make_process_path_markers(
            move.capture_target, move.to, move.player, undo
        )

        # Capture top ring from target
        if not target_stack.rings:
            # Defensive: nothing to capture; restore attacker and exit
            self._stacks[from_key] = attacker
            return

        # Capture original target for undo
        undo.modified_stacks[target_key] = target_stack.copy()

        # Remove target stack hash
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            target_key,
            target_stack.controlling_player,
            target_stack.stack_height,
            tuple(target_stack.rings)
        )

        # Pop top ring from target (bottom→top, so pop from end)
        captured_ring = target_stack.rings.pop()
        target_stack.stack_height -= 1

        if target_stack.stack_height == 0:
            if target_key in self._stacks:
                del self._stacks[target_key]
        else:
            # Recompute target properties
            target_stack.recompute_properties()
            # Add new target stack hash
            self._zobrist_hash ^= self._zobrist.get_stack_hash(
                target_key,
                target_stack.controlling_player,
                target_stack.stack_height,
                tuple(target_stack.rings)
            )

        # Insert captured ring at bottom of attacker (index 0)
        prev_controlling = attacker.controlling_player
        prev_cap_height = attacker.cap_height

        attacker.rings.insert(0, captured_ring)
        attacker.stack_height += 1
        attacker.controlling_player = prev_controlling
        attacker.cap_height = prev_cap_height

        # Check for landing marker - landing on ANY marker (own or opponent)
        # removes the marker and eliminates the top ring of the landing stack.
        landing_marker = self._markers.get(to_key)
        landed_on_marker = landing_marker is not None

        if landed_on_marker:
            # Record marker removal
            undo.removed_markers[to_key] = landing_marker.copy()
            lm_hash = self._zobrist.get_marker_hash(to_key, landing_marker.player)
            self._zobrist_hash ^= lm_hash
            del self._markers[to_key]

        # Handle destination (merge or simple move)
        dest_stack = self._stacks.get(to_key)
        if dest_stack and dest_stack.stack_height > 0:
            undo.modified_stacks[to_key] = dest_stack.copy()

            # Remove old dest hash
            self._zobrist_hash ^= self._zobrist.get_stack_hash(
                to_key,
                dest_stack.controlling_player,
                dest_stack.stack_height,
                tuple(dest_stack.rings)
            )

            # Merge stacks (destination at bottom, attacker on top)
            dest_stack.rings.extend(attacker.rings)
            dest_stack.recompute_properties()
        else:
            # Simple move to destination
            undo.added_stacks.add(to_key)
            attacker.position = move.to
            self._stacks[to_key] = attacker

        # Add new dest hash
        new_dest = self._stacks[to_key]
        self._zobrist_hash ^= self._zobrist.get_stack_hash(
            to_key,
            new_dest.controlling_player,
            new_dest.stack_height,
            tuple(new_dest.rings)
        )

        # Self-elimination if landed on ANY marker (own or opponent)
        if landed_on_marker:
            self._make_eliminate_top_ring(move.to, move.player, undo)

    def _make_eliminate_rings_from_stack(
        self, move: Move, undo: MoveUndo
    ) -> None:
        """Apply ELIMINATE_RINGS_FROM_STACK move.

        Eliminates the **entire cap** (all consecutive top rings of the
        controlling player's colour) from the stack at move.to position.
        For mixed-colour stacks, this exposes buried rings of other colours;
        for single-colour stacks with height > 1, this eliminates all rings
        (removing the stack entirely).

        Exception: Recovery actions use buried ring extraction (one ring)
        instead - that is handled separately.
        """
        pos_key = move.to.to_key()
        stack = self._stacks.get(pos_key)
        if not stack or stack.cap_height <= 0:
            return

        cap_height = stack.cap_height
        for _ in range(cap_height):
            self._make_eliminate_top_ring(move.to, move.player, undo)

    def _make_forced_elimination(self, move: Move, undo: MoveUndo) -> None:
        """Apply FORCED_ELIMINATION move (RR-CANON-R070).

        Eliminates the **entire cap** from the stack at move.to position.

        FORCED ELIMINATION and TERRITORY PROCESSING (RR-CANON-R022/R145):
        Both contexts allow ANY controlled stack as eligible, including
        height-1 standalone rings. They use the same underlying cap
        elimination logic (eliminating the entire cap from the stack).
        """
        self._make_eliminate_rings_from_stack(move, undo)

    def _make_recovery_slide(self, move: Move, undo: MoveUndo) -> None:
        """Apply RECOVERY_SLIDE move (RR-CANON-R110-R115).

        Recovery slides allow temporarily eliminated players to reenter by
        sliding markers to form lines or reposition.
        """
        player = move.player
        from_key = move.from_pos.to_key() if move.from_pos else None
        to_key = move.to.to_key() if move.to else None

        if not from_key or not to_key:
            return

        recovery_mode = getattr(move, 'recoveryMode', None)

        # Handle stack-strike recovery: marker attacks adjacent stack
        if recovery_mode == "stack_strike":
            # Remove marker
            if from_key in self._markers:
                undo.removed_markers[from_key] = self._markers[from_key].copy()
                marker_hash = self._zobrist.get_marker_hash(from_key, player)
                self._zobrist_hash ^= marker_hash
                del self._markers[from_key]

            # Strike the stack - eliminate top ring
            if to_key in self._stacks:
                stack = self._stacks[to_key]
                undo.modified_stacks[to_key] = stack.copy()
                if stack.rings:
                    stack.rings.pop()
                    stack.stack_height = len(stack.rings)
                    # Credit elimination to recovering player
                    if player not in undo.prev_eliminated_rings:
                        undo.prev_eliminated_rings[player] = (
                            self._players[player].eliminated_rings
                            if player in self._players else 0
                        )
                    if player in self._players:
                        self._players[player].eliminated_rings += 1
                    self._total_rings_eliminated += 1

                    if stack.stack_height == 0:
                        undo.removed_stacks[to_key] = undo.modified_stacks.pop(to_key)
                        del self._stacks[to_key]
                    else:
                        # Recalculate cap
                        stack.controlling_player = stack.rings[-1]
                        cap_height = 0
                        for r in reversed(stack.rings):
                            if r == stack.controlling_player:
                                cap_height += 1
                            else:
                                break
                        stack.cap_height = cap_height
            return

        # Move marker from from_pos to to_pos
        if from_key in self._markers:
            marker = self._markers[from_key]
            undo.removed_markers[from_key] = marker.copy()
            marker_hash = self._zobrist.get_marker_hash(from_key, marker.player)
            self._zobrist_hash ^= marker_hash
            del self._markers[from_key]

            # Add marker at new position
            new_marker = MutableMarker(position=move.to, player=player)
            self._markers[to_key] = new_marker
            undo.added_markers.add(to_key)
            new_hash = self._zobrist.get_marker_hash(to_key, player)
            self._zobrist_hash ^= new_hash

        # For line recovery, collapse markers
        if recovery_mode != "fallback":
            collapse_positions = getattr(move, 'collapsePositions', None)
            if collapse_positions:
                for pos in collapse_positions:
                    pos_key = pos.to_key()
                    if pos_key in self._markers:
                        m = self._markers[pos_key]
                        undo.removed_markers[pos_key] = m.copy()
                        mh = self._zobrist.get_marker_hash(pos_key, m.player)
                        self._zobrist_hash ^= mh
                        del self._markers[pos_key]

                    # Add to collapsed
                    if pos_key not in self._collapsed:
                        self._collapsed[pos_key] = player
                        undo.added_collapsed.add(pos_key)
                        # Update territory count
                        if player not in undo.prev_territory_spaces:
                            undo.prev_territory_spaces[player] = (
                                self._players[player].territory_spaces
                                if player in self._players else 0
                            )
                        if player in self._players:
                            self._players[player].territory_spaces += 1

        # Extract buried rings for cost (Option 1 = 1, Option 2 = 0)
        recovery_option = getattr(move, 'recoveryOption', 1)
        cost = 1 if recovery_option == 1 else 0

        # Fallback recovery always costs 1
        if recovery_mode == "fallback":
            cost = 1

        if cost > 0:
            extraction_stacks = getattr(move, 'extraction_stacks', None)
            if extraction_stacks:
                for stack_key in extraction_stacks:
                    if stack_key in self._stacks:
                        stack = self._stacks[stack_key]
                        undo.modified_stacks[stack_key] = stack.copy()
                        # Find player's bottommost ring
                        try:
                            idx = stack.rings.index(player)
                            stack.rings.pop(idx)
                            stack.stack_height = len(stack.rings)
                            # Credit self-elimination
                            if player not in undo.prev_eliminated_rings:
                                undo.prev_eliminated_rings[player] = (
                                    self._players[player].eliminated_rings
                                    if player in self._players else 0
                                )
                            if player in self._players:
                                self._players[player].eliminated_rings += 1
                            self._total_rings_eliminated += 1

                            if stack.stack_height == 0:
                                undo.removed_stacks[stack_key] = (
                                    undo.modified_stacks.pop(stack_key)
                                )
                                del self._stacks[stack_key]
                            else:
                                stack.controlling_player = stack.rings[-1]
                                cap_height = 0
                                for r in reversed(stack.rings):
                                    if r == stack.controlling_player:
                                        cap_height += 1
                                    else:
                                        break
                                stack.cap_height = cap_height
                            break
                        except ValueError:
                            pass

    def _make_process_line(self, move: Move, undo: MoveUndo) -> None:
        """Apply line processing move (PROCESS_LINE, CHOOSE_LINE_OPTION, etc.).

        Converts marker positions into collapsed territory.
        """
        # Find the line to process
        # Look for line info on the move
        line_positions: list[Position] = []

        if hasattr(move, 'formed_lines') and move.formed_lines:
            lines = list(move.formed_lines)
            if lines:
                line_positions = list(lines[0].positions)
        elif hasattr(move, 'collapsed_markers') and move.collapsed_markers:
            line_positions = list(move.collapsed_markers)

        if not line_positions:
            # Try to find line from move.to position
            # For simplicity, without access to BoardManager directly,
            # we just collapse the single position if no line info
            line_positions = [move.to]

        # Determine which positions to collapse based on move type.
        # Use canonical effective line length (board + player count aware).
        required_len = get_effective_line_length(self._board_type, self._max_players)

        positions_to_collapse: list[Position]
        if move.type in (MoveType.PROCESS_LINE, MoveType.LINE_FORMATION):
            # Full line collapse
            positions_to_collapse = list(line_positions)
        elif move.type in (MoveType.CHOOSE_LINE_OPTION, MoveType.CHOOSE_LINE_REWARD):
            # CHOOSE_LINE_OPTION is canonical; CHOOSE_LINE_REWARD is a legacy alias.
            #
            # Prefer collapsed_markers when present (explicit option encoding);
            # otherwise fall back to placement_count semantics:
            # 1 → minimum collapse, >1 → collapse all.
            markers = getattr(move, 'collapsed_markers', None)
            if markers:
                positions_to_collapse = list(markers)
            else:
                option = move.placement_count or 1
                if option == 1:
                    positions_to_collapse = list(line_positions[:required_len])
                else:
                    positions_to_collapse = list(line_positions)

        # Collapse each position
        # IMPORTANT: TS's collapseLinePositions returns rings from any stacks
        # on collapsed positions back to their owners' hands. We must do the
        # same for parity.
        seen_keys: set = set()
        for pos in positions_to_collapse:
            key = pos.to_key()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Return rings from any stack at this position to their owners' hands.
            # This matches TS LineAggregate.collapseLinePositions behavior.
            stack = self._stacks.get(key)
            if stack and stack.rings:
                for ring_owner in stack.rings:
                    if ring_owner in self._players:
                        ps = self._players[ring_owner]
                        # Save previous value for undo if not already saved
                        if ring_owner not in undo.prev_rings_in_hand:
                            undo.prev_rings_in_hand[ring_owner] = ps.rings_in_hand
                        ps.rings_in_hand += 1
                # Remove the stack
                undo.removed_stacks[key] = stack.copy()
                del self._stacks[key]

            # Check if there was a marker
            marker = self._markers.get(key)
            if marker:
                undo.removed_markers[key] = marker.copy()
                self._zobrist_hash ^= self._zobrist.get_marker_hash(
                    key, marker.player
                )
                del self._markers[key]

            # Add collapsed space
            if key not in self._collapsed:
                undo.added_collapsed.add(key)
            self._collapsed[key] = move.player
            self._zobrist_hash ^= self._zobrist.get_collapsed_hash(key)

        # RR-CANON-R121-R122: Apply elimination cost for Option 1 (collapse all).
        # Option 1 is used when we collapse the ENTIRE line (all positions).
        # Option 2 (minimum collapse) does NOT require elimination.
        # This matches GPU process_lines_batch behavior for parity.
        is_option_1 = len(positions_to_collapse) >= len(line_positions)
        if is_option_1:
            # Eliminate one ring from any controlled stack (matches GPU behavior).
            # GPU uses _eliminate_one_ring_from_any_stack which picks first eligible
            # in row-major order (y, x). Sort stack keys to match.
            player = move.player
            # Parse keys like "x,y" and sort by (y, x) to match GPU's numpy order
            def parse_key(k: str) -> tuple[int, int]:
                parts = k.split(',')
                x, y = int(parts[0]), int(parts[1])
                return (y, x)  # GPU uses row-major (y first)

            sorted_keys = sorted(self._stacks.keys(), key=parse_key)
            for stack_key in sorted_keys:
                stack = self._stacks[stack_key]
                if stack.controlling_player == player and stack.stack_height > 0:
                    # Save undo info
                    if stack_key not in undo.modified_stacks:
                        undo.modified_stacks[stack_key] = stack.copy()

                    # Eliminate one ring from this stack
                    if stack.stack_height == 1:
                        # Stack is fully eliminated
                        if stack_key not in undo.removed_stacks:
                            undo.removed_stacks[stack_key] = stack.copy()
                        del self._stacks[stack_key]
                    else:
                        # Reduce stack height
                        stack.stack_height -= 1
                        if stack.rings:
                            stack.rings = stack.rings[:-1]
                        # Recalculate controlling_player and cap_height
                        if stack.rings:
                            stack.controlling_player = stack.rings[-1]
                        cap_height = 0
                        for r in reversed(stack.rings):
                            if r == stack.controlling_player:
                                cap_height += 1
                            else:
                                break
                        stack.cap_height = cap_height

                    # Update player eliminated_rings count
                    if player in self._players:
                        ps = self._players[player]
                        if player not in undo.prev_eliminated_rings:
                            undo.prev_eliminated_rings[player] = ps.eliminated_rings
                        ps.eliminated_rings += 1

                    # Update game total_rings_eliminated
                    undo.prev_total_rings_eliminated = self._total_rings_eliminated
                    self._total_rings_eliminated += 1
                    break  # Only eliminate one ring

    def _make_process_territory(self, move: Move, undo: MoveUndo) -> None:
        """Apply territory processing move.

        Handles PROCESS_TERRITORY_REGION, TERRITORY_CLAIM, CHOOSE_TERRITORY_OPTION.

        This is a simplified version that:
        1. Collapses all positions in the region
        2. Updates territory_spaces for the player
        """
        # Get region from move if available
        region_spaces: list[Position] = []

        if hasattr(move, 'disconnected_regions') and move.disconnected_regions:
            regions = list(move.disconnected_regions)
            if regions:
                region_spaces = list(regions[0].spaces)

        if not region_spaces:
            # If no region info, just use move.to
            region_spaces = [move.to]

        region_keys = {p.to_key() for p in region_spaces}
        player = move.player

        # Track territory spaces change
        if player not in undo.prev_territory_spaces and player in self._players:
            undo.prev_territory_spaces[player] = (
                self._players[player].territory_spaces
            )

        spaces_gained = 0

        # Eliminate all rings within the region
        for pos in region_spaces:
            pos_key = pos.to_key()
            stack = self._stacks.get(pos_key)
            while stack and stack.stack_height > 0:
                self._make_eliminate_top_ring(pos, player, undo)
                stack = self._stacks.get(pos_key)

        # Collapse all spaces in the region
        for pos in region_spaces:
            key = pos.to_key()

            # Remove marker if present
            marker = self._markers.get(key)
            if marker:
                if key not in undo.removed_markers:
                    undo.removed_markers[key] = marker.copy()
                self._zobrist_hash ^= self._zobrist.get_marker_hash(
                    key, marker.player
                )
                del self._markers[key]

            # Add collapsed space
            if key not in self._collapsed:
                undo.added_collapsed.add(key)
                spaces_gained += 1
            self._collapsed[key] = player
            self._zobrist_hash ^= self._zobrist.get_collapsed_hash(key)

        # Get border markers for the region and collapse them too
        border_markers = self._get_border_marker_positions(region_spaces)
        for pos in border_markers:
            key = pos.to_key()
            if key in region_keys:
                continue  # Already processed

            marker = self._markers.get(key)
            if marker:
                if key not in undo.removed_markers:
                    undo.removed_markers[key] = marker.copy()
                self._zobrist_hash ^= self._zobrist.get_marker_hash(
                    key, marker.player
                )
                del self._markers[key]

            if key not in self._collapsed:
                undo.added_collapsed.add(key)
                spaces_gained += 1
            self._collapsed[key] = player
            self._zobrist_hash ^= self._zobrist.get_collapsed_hash(key)

        # Update territory_spaces for the player
        if player in self._players:
            self._players[player].territory_spaces += spaces_gained

        # Mandatory self-elimination
        # Find player stacks outside the region
        player_stacks = [
            (k, s) for k, s in self._stacks.items()
            if s.controlling_player == player
        ]
        outside_stacks = [
            (k, s) for k, s in player_stacks
            if k not in region_keys
        ]

        if outside_stacks:
            # Eliminate cap from the first outside stack
            _chosen_key, chosen_stack = outside_stacks[0]
            cap_height = chosen_stack.cap_height
            for _ in range(cap_height):
                self._make_eliminate_top_ring(
                    chosen_stack.position, player, undo
                )
        elif player_stacks:
            # Use any stack
            _chosen_key, chosen_stack = player_stacks[0]
            cap_height = chosen_stack.cap_height
            for _ in range(cap_height):
                self._make_eliminate_top_ring(
                    chosen_stack.position, player, undo
                )
        else:
            # No stacks - eliminate from rings_in_hand
            if player in self._players:
                ps = self._players[player]
                if ps.rings_in_hand > 0:
                    if player not in undo.prev_rings_in_hand:
                        undo.prev_rings_in_hand[player] = ps.rings_in_hand
                    if player not in undo.prev_eliminated_rings:
                        undo.prev_eliminated_rings[player] = ps.eliminated_rings

                    ps.rings_in_hand -= 1
                    ps.eliminated_rings += 1
                    self._total_rings_eliminated += 1

                    # Update board eliminated rings
                    player_id_str = str(player)
                    if player_id_str not in undo.prev_board_eliminated_rings:
                        undo.prev_board_eliminated_rings[player_id_str] = (
                            self._board_eliminated_rings.get(player_id_str, 0)
                        )
                    self._board_eliminated_rings[player_id_str] = (
                        self._board_eliminated_rings.get(player_id_str, 0) + 1
                    )

    def _get_border_marker_positions(
        self, region_spaces: list[Position]
    ) -> list[Position]:
        """Get border marker positions for a disconnected region.

        Returns markers that are adjacent to the region but not inside it.
        """
        region_keys = {p.to_key() for p in region_spaces}
        border_markers: list[Position] = []

        for space in region_spaces:
            neighbors = self._get_territory_neighbors(space)
            for neighbor in neighbors:
                n_key = neighbor.to_key()
                if n_key in region_keys:
                    continue
                marker = self._markers.get(n_key)
                if marker is not None and not any(p.to_key() == n_key for p in border_markers):
                    border_markers.append(neighbor)

        return border_markers

    def _get_territory_neighbors(self, pos: Position) -> list[Position]:
        """Get territory-adjacent neighbors (Von Neumann for square, hex for hex)."""
        if self._board_type in (BoardType.HEXAGONAL, BoardType.HEX8):
            return [
                Position(
                    x=pos.x+1, y=pos.y,
                    z=pos.z-1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x, y=pos.y+1,
                    z=pos.z-1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x-1, y=pos.y+1,
                    z=pos.z if pos.z is not None else None
                ),
                Position(
                    x=pos.x-1, y=pos.y,
                    z=pos.z+1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x, y=pos.y-1,
                    z=pos.z+1 if pos.z is not None else None
                ),
                Position(
                    x=pos.x+1, y=pos.y-1,
                    z=pos.z if pos.z is not None else None
                )
            ]
        else:
            # Von Neumann (4-direction)
            return [
                Position(x=pos.x+1, y=pos.y),
                Position(x=pos.x-1, y=pos.y),
                Position(x=pos.x, y=pos.y+1),
                Position(x=pos.x, y=pos.y-1)
            ]
