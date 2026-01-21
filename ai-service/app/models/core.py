"""
Pydantic Models for RingRift Game State
Mirrors TypeScript types from src/shared/types/game.ts
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class BoardType(str, Enum):
    """Board type enumeration"""
    SQUARE8 = "square8"
    SQUARE19 = "square19"
    HEX8 = "hex8"  # Radius-4 hexagonal board (61 cells) - parallel to square8
    HEXAGONAL = "hexagonal"
    # Aliases for the full 25x25 hexagonal board
    FULL_HEX = "hexagonal"
    FULL_HEXAGONAL = "hexagonal"


class GamePhase(str, Enum):
    """Game phase enumeration - 7 canonical phases per RR-CANON-R070 + terminal phase."""
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
    # Terminal phase: entered when the game ends (victory detected).
    # This provides semantic clarity that the game is over without relying
    # solely on gameStatus, and ensures TS↔Python phase parity at game end.
    GAME_OVER = "game_over"


class GameStatus(str, Enum):
    """Game status enumeration - aligned with TypeScript semantics"""
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


class MoveType(str, Enum):
    """
    Canonical + legacy MoveType enumeration mirroring src/shared/types/game.ts.

    Legacy aliases are retained for replay compatibility only; canonical
    recordings must use the canonical names (e.g., move_stack, choose_line_option).
    """
    # Ring Placement
    PLACE_RING = "place_ring"
    SKIP_PLACEMENT = "skip_placement"
    # Forced no-op: player entered RING_PLACEMENT but had no legal placement
    # anywhere (e.g. rings_in_hand == 0 or no positions allowed by caps /
    # no-dead-placement). Mirrors TS MoveType 'no_placement_action' and
    # RR-CANON-R075.
    NO_PLACEMENT_ACTION = "no_placement_action"

    # Movement
    MOVE_STACK = "move_stack"
    MOVE_RING = "move_ring"  # Legacy alias
    BUILD_STACK = "build_stack"  # Legacy alias (deprecated)
    # Forced no-op in movement: player entered MOVEMENT but had no legal
    # movement or capture anywhere. Mirrors TS MoveType 'no_movement_action'
    # and RR-CANON-R075.
    NO_MOVEMENT_ACTION = "no_movement_action"

    # Capture
    OVERTAKING_CAPTURE = "overtaking_capture"
    CONTINUE_CAPTURE_SEGMENT = "continue_capture_segment"
    # Voluntary skip: decline optional post-movement capture and proceed to
    # line_processing. Mirrors TS MoveType 'skip_capture' and RR-CANON-R073.
    SKIP_CAPTURE = "skip_capture"

    # Meta-move: pie rule / swap colours in 2-player games. This mirrors the
    # 'swap_sides' MoveType in src/shared/types/game.ts and is treated as a
    # pure seat/identity swap with no board geometry change.
    SWAP_SIDES = "swap_sides"

    # Line Processing
    PROCESS_LINE = "process_line"
    # Legacy alias for CHOOSE_LINE_OPTION; retained for replay compatibility.
    CHOOSE_LINE_REWARD = "choose_line_reward"
    # Forced no-op: player entered line_processing but had no lines or line
    # rewards. Mirrors TS MoveType 'no_line_action' and RR-CANON-R075.
    NO_LINE_ACTION = "no_line_action"

    # Territory Processing
    # Legacy alias for CHOOSE_TERRITORY_OPTION; retained for replay compatibility.
    PROCESS_TERRITORY_REGION = "process_territory_region"
    # Voluntary skip: player has eligible regions but chooses not to
    # process them.
    SKIP_TERRITORY_PROCESSING = "skip_territory_processing"
    # Forced no-op: player entered territory_processing but has no
    # eligible regions. Semantically distinct from
    # SKIP_TERRITORY_PROCESSING per RR-CANON-R075.
    NO_TERRITORY_ACTION = "no_territory_action"
    ELIMINATE_RINGS_FROM_STACK = "eliminate_rings_from_stack"

    # Legacy / Experimental (to be phased out or kept for compatibility)
    LINE_FORMATION = "line_formation"
    TERRITORY_CLAIM = "territory_claim"
    CHAIN_CAPTURE = "chain_capture"
    # Canonical forced-elimination move in the dedicated forced_elimination
    # phase (RR-CANON-R070, RR-CANON-R072, RR-CANON-R100). Hosts may continue
    # to express the underlying ring removal via ELIMINATE_RINGS_FROM_STACK
    # for now; this discriminant exists so parity tooling and DBs can record
    # the final turn phase explicitly.
    FORCED_ELIMINATION = "forced_elimination"

    # Canonical choice moves
    CHOOSE_LINE_OPTION = "choose_line_option"
    # Canonical territory region decision (legacy alias: PROCESS_TERRITORY_REGION).
    CHOOSE_TERRITORY_OPTION = "choose_territory_option"

    # Recovery (RR-CANON-R110–R115)
    # Temporarily eliminated players (no stacks, no rings in hand, has markers
    # and buried rings) can perform recovery actions.
    RECOVERY_SLIDE = "recovery_slide"
    # Skip recovery: player chooses to pass their turn without moving a marker
    # or extracting a buried ring. Per RR-CANON-R112 skip option.
    SKIP_RECOVERY = "skip_recovery"
    # Game termination: player forfeits or runs out of time.
    RESIGN = "resign"
    TIMEOUT = "timeout"


class AIType(str, Enum):
    """AI type enumeration.

    The NEURAL_DEMO variant is reserved for experimental / sandbox use and
    is never selected by the canonical difficulty ladder by default. It is
    gated behind an environment flag on the Python AI service to ensure that
    neural-only demo engines cannot be enabled accidentally in production
    ladders.

    GPU_MINIMAX is a GPU-accelerated minimax variant that uses batched leaf
    evaluation for improved performance on CUDA/MPS hardware.

    Multiplayer minimax variants:
    - MINIMAX: Paranoid search (assumes all opponents collude against you)
    - MAXN: Max-N search (each player maximizes their own score)
    - BRS: Best-Reply Search (greedy best replies, faster but shallower)
    """
    RANDOM = "random"
    HEURISTIC = "heuristic"
    MINIMAX = "minimax"
    GPU_MINIMAX = "gpu_minimax"
    MAXN = "maxn"
    BRS = "brs"
    MCTS = "mcts"
    DESCENT = "descent"
    NEURAL_DEMO = "neural_demo"
    POLICY_ONLY = "policy_only"  # Direct NN policy without search
    GUMBEL_MCTS = "gumbel_mcts"  # Gumbel AlphaZero with Sequential Halving
    EBMO = "ebmo"  # Energy-Based Move Optimization (gradient descent on action embeddings)
    GMO = "gmo"  # Gradient Move Optimization (entropy-guided gradient ascent in move embedding space)
    GMO_V2 = "gmo_v2"  # GMO v2: Enhanced with attention encoder, ensemble optimization, temperature scheduling
    GMO_MCTS = "gmo_mcts"  # GMO-guided MCTS (uses GMO for move priors in tree search)
    GMO_GUMBEL = "gmo_gumbel"  # GMO + Gumbel MCTS (uses GMO value network with Gumbel search)
    IG_GMO = "ig_gmo"  # Experimental: Information-Gain GMO (MI-based exploration + GNN)
    CAGE = "cage"  # Constraint-Aware Graph Energy-based move optimization (GNN + primal-dual)
    IMPROVED_MCTS = "improved_mcts"  # Advanced MCTS with PUCT, progressive widening, transposition tables
    HYBRID_NN = "hybrid_nn"  # Fast heuristic + NN value ranking (5-10x faster than full MCTS)
    GNN = "gnn"  # Graph Neural Network policy (message passing for territory connectivity)
    HYBRID = "hybrid"  # Hybrid CNN-GNN policy (CNN patterns + GNN connectivity)

    # NNUE-based AI types (Dec 2025 - Unified AI Evaluation Architecture)
    NNUE_GUMBEL = "nnue_gumbel"  # NNUE + Gumbel MCTS (native policy or value-derived)
    NNUE_MCTS = "nnue_mcts"  # NNUE + standard MCTS
    NNUE_BRS = "nnue_brs"  # NNUE + Best Reply Search (multiplayer)
    NNUE_MAXN = "nnue_maxn"  # NNUE + MaxN search (multiplayer)


class Position(BaseModel):
    """Board position (2D or 3D for hexagonal)"""
    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    z: Optional[int] = None

    def to_key(self) -> str:
        """Convert position to string key.

        Uses module-level cache for O(1) repeated lookups.
        Profiling showed this method called 314K+ times per game.
        """
        cache_key = (self.x, self.y, self.z)
        cached = _position_key_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.z is not None:
            key = f"{self.x},{self.y},{self.z}"
        else:
            key = f"{self.x},{self.y}"

        _position_key_cache[cache_key] = key
        return key


# Module-level cache for position keys (outside Pydantic model)
_position_key_cache: dict[tuple[int, int, Optional[int]], str] = {}


def clear_position_key_cache() -> None:
    """Clear the position key cache (useful for testing)."""
    _position_key_cache.clear()


class LineInfo(BaseModel):
    """Information about a formed line"""
    model_config = ConfigDict(populate_by_name=True)

    positions: list[Position]
    player: int
    length: int
    direction: Position  # Direction vector


class Territory(BaseModel):
    """Information about a territory region"""
    model_config = ConfigDict(populate_by_name=True)

    spaces: list[Position]
    controlling_player: int = Field(alias="controllingPlayer")
    is_disconnected: bool = Field(alias="isDisconnected")


class RingStack(BaseModel):
    """Ring stack on the board"""
    model_config = ConfigDict(populate_by_name=True)

    position: Position
    rings: list[int]  # Player numbers from bottom to top
    stack_height: int = Field(alias="stackHeight")
    cap_height: int = Field(alias="capHeight")
    controlling_player: int = Field(alias="controllingPlayer")


class MarkerInfo(BaseModel):
    """Marker information"""
    player: int
    position: Position
    type: str  # 'regular' or 'collapsed'


class Player(BaseModel):
    """Player state"""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    username: str
    type: str
    player_number: int = Field(alias="playerNumber")
    is_ready: bool = Field(alias="isReady")
    time_remaining: int = Field(alias="timeRemaining")
    ai_difficulty: Optional[int] = Field(None, alias="aiDifficulty")
    rings_in_hand: int = Field(alias="ringsInHand")
    eliminated_rings: int = Field(alias="eliminatedRings")
    territory_spaces: int = Field(alias="territorySpaces")


class TimeControl(BaseModel):
    """Time control settings"""
    model_config = ConfigDict(populate_by_name=True)

    initial_time: int = Field(alias="initialTime")
    increment: int
    type: str


class Move(BaseModel):
    """Move representation.

    This mirrors the TypeScript Move type closely enough for the AI
    service to participate in placement, movement, capture, and the
    newer line/territory decision phases:

    - `type` is a MoveType enum.
    - For placement moves, `placement_count` and `placed_on_stack`
      carry the multi-ring and stacking metadata introduced on the
      backend.
    - For capture moves, `capture_target` identifies the overtaken stack.
    - For line/territory decision moves, `formed_lines` and
      `disconnected_regions` allow the Python parity harness to
      construct canonical `process_line` and `choose_territory_option`
      Moves that match the TS engines (legacy alias: `process_territory_region`).
    """
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    id: str
    type: MoveType
    player: int

    # Spatial metadata (both optional for certain move types like
    # forced_elimination or swap_sides).
    from_pos: Optional[Position] = Field(None, alias="from")
    to: Optional[Position] = None

    # Capture metadata
    capture_target: Optional[Position] = Field(None, alias="captureTarget")
    captured_stacks: Optional[tuple[RingStack, ...]] = Field(
        None, alias="capturedStacks"
    )
    capture_chain: Optional[tuple[Position, ...]] = Field(
        None, alias="captureChain"
    )
    overtaken_rings: Optional[tuple[int, ...]] = Field(
        None, alias="overtakenRings"
    )

    # Ring placement specific metadata (optional for non-placement moves)
    placed_on_stack: Optional[bool] = Field(None, alias="placedOnStack")
    placement_count: Optional[int] = Field(None, alias="placementCount")

    # Movement specific
    stack_moved: Optional[RingStack] = Field(None, alias="stackMoved")
    minimum_distance: Optional[int] = Field(None, alias="minimumDistance")
    actual_distance: Optional[int] = Field(None, alias="actualDistance")
    marker_left: Optional[Position] = Field(None, alias="markerLeft")

    # Line / territory metadata for decision phases
    line_index: Optional[int] = Field(None, alias="lineIndex")
    formed_lines: Optional[tuple[LineInfo, ...]] = Field(
        default=None, alias="formedLines"
    )
    collapsed_markers: Optional[tuple[Position, ...]] = Field(
        default=None, alias="collapsedMarkers"
    )
    claimed_territory: Optional[tuple[Territory, ...]] = Field(
        default=None, alias="claimedTerritory"
    )
    disconnected_regions: Optional[tuple[Territory, ...]] = Field(
        default=None, alias="disconnectedRegions"
    )
    # Recovery-specific metadata
    recovery_option: Optional[int] = Field(None, alias="recoveryOption")
    # Recovery mode: which success criterion was met (RR-CANON-R112)
    # "line" (a), "fallback" (b1), or "stack_strike" (b2)
    recovery_mode: Optional[str] = Field(None, alias="recoveryMode")
    collapse_positions: Optional[tuple[Position, ...]] = Field(
        default=None, alias="collapsePositions"
    )
    # Stack position keys from which to extract buried rings (Option 1/fallback/stack_strike).
    # Option 2 uses no extraction (empty list).
    extraction_stacks: Optional[tuple[str, ...]] = Field(
        default=None, alias="extractionStacks"
    )
    # Summary of rings eliminated by this action, grouped per player.
    eliminated_rings: Optional[tuple[dict[str, int], ...]] = Field(
        default=None, alias="eliminatedRings"
    )
    # Elimination context for eliminate_rings_from_stack moves (RR-CANON-R022, R122):
    # - 'line': Eliminate exactly ONE ring (any controlled stack eligible)
    # - 'territory': Eliminate entire cap (any controlled stack eligible, including height-1)
    # - 'forced': Eliminate entire cap (any controlled stack eligible)
    # - 'recovery': Extract exactly ONE buried ring from any stack that contains a buried ring
    #   of the eliminating player (stack need not be controlled by that player)
    elimination_context: Optional[str] = Field(
        default=None, alias="eliminationContext"
    )

    # Timing / ordering (optional for backward compatibility with JSONL imports)
    timestamp: Optional[datetime] = None
    think_time: Optional[int] = Field(default=None, alias="thinkTime")
    move_number: Optional[int] = Field(default=None, alias="moveNumber")

    # Phase at which this move was made (optional - for canonical export/import)
    # This is stored alongside the move to preserve phase information from GPU
    # selfplay exports without requiring full state reconstruction during import.
    phase: Optional[str] = None


class BoardState(BaseModel):
    """Current board state"""
    model_config = ConfigDict(populate_by_name=True)

    type: BoardType
    size: int
    stacks: dict[str, RingStack] = Field(default_factory=dict)
    markers: dict[str, MarkerInfo] = Field(default_factory=dict)
    collapsed_spaces: dict[str, int] = Field(
        default_factory=dict, alias="collapsedSpaces"
    )
    eliminated_rings: dict[str, int] = Field(
        default_factory=dict, alias="eliminatedRings"
    )
    formed_lines: list[LineInfo] = Field(
        default_factory=list, alias="formedLines"
    )
    territories: dict[str, Territory] = Field(
        default_factory=dict, alias="territories"
    )


class ChainCaptureSegment(BaseModel):
    """Segment of a chain capture"""
    model_config = ConfigDict(populate_by_name=True)

    from_pos: Position = Field(alias="from")
    target: Position
    landing: Position
    captured_cap_height: int = Field(alias="capturedCapHeight")


class ChainCaptureState(BaseModel):
    """State of an ongoing chain capture"""
    model_config = ConfigDict(populate_by_name=True)

    player_number: int = Field(alias="playerNumber")
    start_position: Position = Field(alias="startPosition")
    current_position: Position = Field(alias="currentPosition")
    segments: list[ChainCaptureSegment]
    available_moves: list[Move] = Field(alias="availableMoves")
    visited_positions: list[str] = Field(alias="visitedPositions")


class GameState(BaseModel):
    """Complete game state"""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    board_type: BoardType = Field(alias="boardType")
    rng_seed: Optional[int] = Field(None, alias="rngSeed")
    board: BoardState
    players: list[Player]
    current_phase: GamePhase = Field(alias="currentPhase")
    current_player: int = Field(alias="currentPlayer")
    move_history: list[Move] = Field(default_factory=list, alias="moveHistory")
    time_control: TimeControl = Field(alias="timeControl")
    spectators: list[str] = Field(default_factory=list)
    game_status: GameStatus = Field(alias="gameStatus")
    winner: Optional[int] = None
    created_at: datetime = Field(alias="createdAt")
    last_move_at: datetime = Field(alias="lastMoveAt")
    is_rated: bool = Field(alias="isRated")
    max_players: int = Field(alias="maxPlayers")
    total_rings_in_play: int = Field(alias="totalRingsInPlay")
    total_rings_eliminated: int = Field(alias="totalRingsEliminated")
    victory_threshold: int = Field(alias="victoryThreshold")
    # Legacy: >50% threshold, kept for backward compatibility
    territory_victory_threshold: int = Field(alias="territoryVictoryThreshold")
    # New: floor(totalSpaces / numPlayers) + 1. Victory also requires > opponents combined.
    territory_victory_minimum: Optional[int] = Field(None, alias="territoryVictoryMinimum")
    chain_capture_state: Optional[ChainCaptureState] = Field(
        None,
        alias="chainCaptureState",
    )
    must_move_from_stack_key: Optional[str] = Field(
        None,
        alias="mustMoveFromStackKey",
    )
    zobrist_hash: Optional[int] = Field(None, alias="zobristHash")
    lps_round_index: int = Field(0, alias="lpsRoundIndex")
    lps_current_round_actor_mask: dict[int, bool] = Field(
        default_factory=dict,
        alias="lpsCurrentRoundActorMask",
    )
    lps_exclusive_player_for_completed_round: Optional[int] = Field(
        None,
        alias="lpsExclusivePlayerForCompletedRound",
    )
    lps_current_round_first_player: Optional[int] = Field(
        None,
        alias="lpsCurrentRoundFirstPlayer",
    )
    # Number of consecutive completed rounds where the same player was the
    # exclusive real-action holder. LPS victory requires lps_rounds_required
    # consecutive rounds (default 3).
    lps_consecutive_exclusive_rounds: int = Field(
        0,
        alias="lpsConsecutiveExclusiveRounds",
    )
    # Configurable LPS threshold: how many consecutive exclusive rounds
    # are required for LPS victory. Default is 3 (canonical), can be
    # adjusted for shorter/longer games.
    lps_rounds_required: int = Field(
        3,
        alias="lpsRoundsRequired",
    )
    # The player who has been exclusive for consecutive rounds.
    lps_consecutive_exclusive_player: Optional[int] = Field(
        None,
        alias="lpsConsecutiveExclusivePlayer",
    )
    # Optional per-game rules configuration (e.g., swap rule / pie rule).
    # Mirrors the RulesOptions bag in src/shared/types/game.ts; callers
    # currently use this primarily for swapRuleEnabled in 2-player games.
    rules_options: Optional[dict[str, object]] = Field(
        default=None,
        alias="rulesOptions",
    )
    # Per RR-CANON-R123: When True, player must choose a stack for line
    # elimination via an explicit eliminate_rings_from_stack move.
    # Set after process_line or choose_line_option (Option 1) applies.
    pending_line_reward_elimination: bool = Field(
        default=False,
        alias="pendingLineRewardElimination",
    )


class AIConfig(BaseModel):
    """AI configuration.

    This is an internal configuration model for Python AIs and is not part of
    the TS/JSON wire contract, so we keep field names Pythonic and avoid
    aliases that confuse static analysis.

    The ``rng_seed`` field drives per-instance pseudo-random behaviour for all
    AI implementations. When unset, each AI derives a deterministic seed from
    its difficulty and player number so that search and sampling remain
    reproducible. When provided (for example via the `/ai/move` ``seed`` query
    parameter or a `GameState.rngSeed` propagated from the Node backend), this
    value is used directly so that TS and Python can share a single
    cross-language RNG stream for debugging, parity harnesses, and datasets.

    The optional ``heuristic_profile_id`` and ``nn_model_id`` fields allow
    higher-level orchestration (difficulty ladder, training scripts, or
    tournaments) to pin a particular HeuristicAI weight set or neural network
    checkpoint to an AI instance without changing the external difficulty
    ladder contract. When unset, callers fall back to the built-in defaults
    (current v1 heuristics and the default neural-network model path).

    The optional ``heuristic_eval_mode`` field is consulted only by
    :class:`HeuristicAI`. It controls whether the heuristic evaluator runs in
    ``"full"`` mode (all feature tiers enabled, current production default) or
    ``"light"`` mode (structural/global Tier-2 features disabled for faster
    evaluation). When unset or set to an unknown value, HeuristicAI treats the
    mode as ``"full"`` to preserve existing behaviour for all callers that do
    not opt in explicitly.

    The optional ``training_move_sample_limit`` field is consulted only by
    :class:`HeuristicAI` during training/evaluation contexts. When set to a
    positive integer, the AI will randomly sample at most that many moves
    for evaluation instead of evaluating all valid moves. This is intended
    for training performance optimization on large boards where move counts
    can exceed thousands. The sampling is deterministic when ``rng_seed`` is
    provided. This field has NO effect on other AI types (Minimax, MCTS, etc.)
    and NO effect on production game play (only training harnesses set this).
    The optional ``use_neural_net`` flag allows higher-level tooling to
    explicitly disable neural-network-backed evaluation for AIs that support
    it (for example :class:`NeuralNetAI`, :class:`DescentAI`, or MCTS-based
    agents). When set to ``False``, these agents should avoid instantiating
    heavy neural-network models entirely and fall back to pure heuristic
    evaluation. When unset, callers retain the current behaviour (neural nets
    enabled where available).
    """
    model_config = ConfigDict(populate_by_name=True)

    difficulty: int = Field(ge=1, le=10)
    think_time: Optional[int] = None
    randomness: Optional[float] = Field(None, ge=0, le=1)
    rng_seed: Optional[int] = Field(None, alias="rngSeed")
    heuristic_profile_id: Optional[str] = None
    nn_model_id: Optional[str] = None
    nn_model_version: Optional[str] = Field(
        default=None,
        description=(
            "When provided, specifies the neural network architecture version to use "
            "(e.g., 'v2', 'v3', 'v4', 'v5-heavy', 'v5-heavy-large'). This controls "
            "which model class is instantiated when loading checkpoints. When None, "
            "defaults to 'v2' for backwards compatibility. Required when loading "
            "non-v2 architecture checkpoints (e.g., v5-heavy models)."
        ),
    )
    nn_state_dict: Optional[dict] = Field(
        default=None,
        description=(
            "When provided, NeuralNetAI loads weights from this in-memory state "
            "dict instead of from disk. This enables zero-disk-IO evaluation for "
            "BackgroundEvaluator and other in-process inference paths. The dict "
            "should be a PyTorch state_dict (parameter name -> tensor mapping). "
            "Takes precedence over nn_model_id when both are set."
        ),
    )
    heuristic_eval_mode: Optional[str] = None
    use_neural_net: Optional[bool] = Field(
        default=None,
        description=(
            "When explicitly set to False, disables neural-network-backed "
            "evaluation for AIs that support it (e.g. DescentAI), forcing "
            "purely heuristic search. When True or None, neural nets remain "
            "enabled subject to model availability and environment flags."
        ),
    )
    use_incremental_search: bool = Field(
        default=True,
        description="When True (default), MinimaxAI uses make/unmake on "
        "MutableGameState for 10-50x faster search. When False, falls back "
        "to legacy immutable state cloning via apply_move()."
    )
    training_move_sample_limit: Optional[int] = Field(
        default=None,
        description="When set to a positive integer, HeuristicAI will "
        "randomly sample at most this many moves for evaluation during "
        "training. Set to None (default) to evaluate all moves. Only "
        "affects HeuristicAI in training contexts; other AIs ignore this."
    )
    allow_fresh_weights: bool = Field(
        default=False,
        description=(
            "When True, NeuralNetAI may proceed with randomly initialized "
            "weights when a checkpoint cannot be found or loaded. This is "
            "intended for offline training/experimentation only; production "
            "paths should keep this False so missing models fail loudly and "
            "callers can fall back to heuristic evaluation."
        ),
    )
    weight_noise: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "When set (0.0-1.0), HeuristicAI applies multiplicative noise to "
            "evaluation weights for training diversity. Each WEIGHT_* constant "
            "is multiplied by a factor in [1-noise, 1+noise]. For example, "
            "weight_noise=0.1 means ±10% variation. Uses the AI's per-instance "
            "RNG for reproducibility."
        ),
    )

    # ------------------------------------------------------------------
    # Self-play exploration knobs (internal only).
    #
    # These fields are used by search-based AIs (notably MCTSAI) to enable
    # AlphaZero-style exploration during offline self-play/training runs.
    # They are NOT part of the TS/JSON wire contract.
    # ------------------------------------------------------------------

    self_play: bool = Field(
        default=False,
        description=(
            "When True, enables self-play exploration features in search-based "
            "AIs (e.g., Dirichlet root noise and temperature sampling for MCTS). "
            "Production and evaluation paths should keep this False."
        ),
    )
    root_dirichlet_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Dirichlet alpha parameter for root noise in self-play MCTS. When "
            "None, a board-specific default is used."
        ),
    )
    root_noise_fraction: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Epsilon mix fraction for root Dirichlet noise in self-play MCTS. "
            "When None, defaults to 0.25."
        ),
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Override temperature for self-play MCTS move sampling. When None, "
            "a default schedule based on move count is used."
        ),
    )
    temperature_cutoff_moves: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Move-count cutoff for high-temperature self-play MCTS. When None, "
            "uses board-specific defaults."
        ),
    )

    # ------------------------------------------------------------------
    # NNUE Policy Model Configuration
    # ------------------------------------------------------------------

    use_policy_ordering: Optional[bool] = Field(
        default=None,
        description=(
            "When True, MinimaxAI uses NNUE policy model for move ordering, "
            "evaluating high-probability moves first for better pruning. "
            "When None, defaults to True for difficulty >= 4 if a policy "
            "model is available."
        ),
    )
    use_nnue_policy_priors: Optional[bool] = Field(
        default=None,
        description=(
            "When True, MCTSAI uses NNUE policy model for move priors when "
            "no neural network is available. Provides informed exploration "
            "guidance without the overhead of a full neural net. "
            "When None, defaults to True when no neural net is loaded."
        ),
    )
    prior_uniform_mix: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Blend factor between policy priors and uniform distribution for MCTS. "
            "0.0 = pure policy priors, 1.0 = pure uniform distribution. "
            "Useful for self-play with value-only models where policy accuracy is low. "
            "When None, defaults to 0.5 in MCTSAI."
        ),
    )

    # ------------------------------------------------------------------
    # Policy-Only AI Configuration
    # ------------------------------------------------------------------

    policy_temperature: Optional[float] = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description=(
            "Temperature for policy softmax in policy-only mode. Lower values "
            "(e.g., 0.1) make the policy more deterministic (greedy), higher "
            "values (e.g., 2.0) increase exploration. Default is 1.0."
        ),
    )

    # ------------------------------------------------------------------
    # Gumbel MCTS Configuration
    # ------------------------------------------------------------------

    gumbel_num_sampled_actions: Optional[int] = Field(
        default=16,
        ge=2,
        le=64,
        description=(
            "Number of actions to sample via Gumbel-Top-K at the root for "
            "Gumbel MCTS. Higher values explore more actions but require "
            "more simulations. Default is 16."
        ),
    )
    gumbel_simulation_budget: Optional[int] = Field(
        default=100,
        ge=10,
        le=5000,
        description=(
            "Total simulation budget for Gumbel MCTS Sequential Halving. "
            "Budget is divided across phases to progressively narrow down "
            "the best action. Default is 100. For quality selfplay on large "
            "boards (square19, hexagonal), use 2400. Max 5000."
        ),
    )
    use_gpu_tree: bool = Field(
        default=False,
        description=(
            "When True, uses GPU-accelerated tensor tree MCTS for 170x speedup. "
            "Requires CUDA and uses Structure-of-Arrays tensor tree with batched "
            "heuristic evaluation. Falls back to CPU MCTS on error."
        ),
    )
    gpu_tree_eval_mode: str = Field(
        default="hybrid",
        description=(
            "Evaluation mode for GPU tensor tree MCTS. Options: "
            "'heuristic' (fastest, ~100ms), 'nn' (most accurate, ~1-2s), "
            "'hybrid' (heuristic for phases + NN for final selection, ~300ms). "
            "Only used when use_gpu_tree=True."
        ),
    )

    # ------------------------------------------------------------------
    # HybridNN Configuration (fast heuristic + NN value ranking)
    # ------------------------------------------------------------------

    hybrid_top_k: int = Field(
        default=8,
        ge=2,
        le=32,
        description=(
            "Number of top candidate moves to evaluate with NN value head "
            "in HybridNN mode. Higher values increase accuracy but reduce speed. "
            "Default is 8 (good balance of speed and quality)."
        ),
    )
    hybrid_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description=(
            "Temperature for move selection in HybridNN mode. "
            "0.0 = greedy (always pick best), higher = more random. "
            "Default is 0.1 (mostly greedy with slight exploration)."
        ),
    )
    board_type: Optional[str] = Field(
        default=None,
        description=(
            "Board type for AI initialization (e.g., 'square8', 'hex8'). "
            "Used by HybridNNAI and other board-aware AIs. When None, "
            "defaults to 'square8'."
        ),
    )
    num_players: int = Field(
        default=2,
        ge=2,
        le=4,
        description=(
            "Number of players in the game. Used by HybridNNAI and other "
            "player-count-aware AIs. Default is 2."
        ),
    )

    # Hybrid NN + Heuristic Evaluation (RR-CANON-HYBRID-001)
    heuristic_blend_alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "When set, GumbelMCTSAI uses hybrid evaluation blending NN value with "
            "normalized heuristic score. Formula: alpha * nn_value + (1-alpha) * heuristic. "
            "Recommended value: 0.6 (60% NN, 40% heuristic). When None, uses pure NN."
        ),
    )
    heuristic_fallback_enabled: bool = Field(
        default=True,
        description=(
            "When True, GumbelMCTSAI uses heuristic evaluation as fallback when NN "
            "evaluation fails. When False, returns 0.0 on NN failure (legacy behavior)."
        ),
    )

    # ------------------------------------------------------------------
    # Minimax Quiescence Search Configuration (Dec 2025)
    #
    # These fields control quiescence search behavior in MinimaxAI.
    # Quiescence search configuration (Dec 2025)
    #
    # IMPORTANT: Quiescence is DISABLED by default for RingRift because:
    # 1. RingRift has multi-phase turns where "noisy" moves (captures, line
    #    formations) are MANDATORY phase completions, not optional extensions
    # 2. The main search already explores all phases of a turn naturally
    # 3. Quiescence is redundant and wastes computation on large boards
    #
    # Quiescence was designed for chess where captures are optional. In RingRift,
    # the phase machine requires these moves before the turn ends. Enable only
    # for experimentation.
    # ------------------------------------------------------------------

    quiescence_enabled: bool = Field(
        default=False,
        description=(
            "When True, MinimaxAI uses quiescence search at leaf nodes "
            "to explore captures and line formations. DISABLED by default because "
            "RingRift's multi-phase turns make quiescence redundant - the main "
            "search already explores all mandatory phase moves. Enable only for "
            "experimentation or if turn-based depth is disabled."
        ),
    )
    quiescence_depth: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description=(
            "Maximum depth for quiescence search. When None, uses adaptive depth "
            "based on board size: 3 for small boards (<100 cells), 2 for medium "
            "(100-200), 1 for large (>200). Explicit values override this."
        ),
    )
    quiescence_node_limit: Optional[int] = Field(
        default=None,
        ge=100,
        le=100000,
        description=(
            "Maximum nodes to visit in quiescence search across all leaf nodes. "
            "When None, uses a default of 10000 nodes. Helps enforce time limits "
            "on large boards where quiescence can explore thousands of positions."
        ),
    )

    # ------------------------------------------------------------------
    # Turn-Based Depth and Branching Factor Optimization (Dec 2025)
    #
    # RingRift has multi-phase turns where a player may make 5-10+ moves
    # before the turn passes. Traditional per-move depth counting means
    # "depth 3" might only represent 1 actual turn on complex boards.
    # ------------------------------------------------------------------

    turn_based_depth: bool = Field(
        default=True,
        description=(
            "When True (default), MinimaxAI counts depth by turns (player changes) "
            "rather than individual moves. This ensures 'depth 3' means 3 full turns "
            "of lookahead regardless of phase complexity. When False, uses traditional "
            "per-move depth counting where each move decrements depth by 1."
        ),
    )
    max_branching_factor: Optional[int] = Field(
        default=None,
        ge=10,
        le=1000,
        description=(
            "Maximum number of moves to consider at each node. When None, evaluates "
            "all legal moves. When set, uses move ordering (killer moves, policy, "
            "heuristics) to select the top N moves, pruning the rest. Helps manage "
            "large boards (hexagonal: 469 cells) where branching can exceed 100+."
        ),
    )
    forced_move_extension: bool = Field(
        default=True,
        description=(
            "When True (default), positions with only one legal move are handled "
            "instantly without decrementing depth. This accelerates search through "
            "forced sequences (single captures, territory decisions with one option)."
        ),
    )


class LineRewardChoiceOption(str, Enum):
    """Line reward choice options, mirroring TypeScript LineRewardChoice."""
    OPTION_1 = "option_1_collapse_all_and_eliminate"
    OPTION_2 = "option_2_min_collapse_no_elimination"


class LineRewardChoiceRequest(BaseModel):
    """Request model for AI-backed line reward choices.

    This mirrors the core fields of the TypeScript LineRewardChoice plus
    the AI configuration metadata the service needs. GameState is not
    required for the initial implementation but can be added later if we
    want the Python service to make more context-aware decisions.
    """
    model_config = ConfigDict(populate_by_name=True)

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: list[LineRewardChoiceOption]


class LineRewardChoiceResponse(BaseModel):
    """Response model for AI-backed line reward choices."""
    model_config = ConfigDict(populate_by_name=True)

    selected_option: LineRewardChoiceOption = Field(alias="selectedOption")
    ai_type: str = Field(alias="aiType")
    difficulty: int


class RingEliminationChoiceOption(BaseModel):
    """Option for ring elimination choice.

    Mirrors the TypeScript RingEliminationChoice option shape:
    { stackPosition, capHeight, totalHeight }.
    """
    model_config = ConfigDict(populate_by_name=True)

    stack_position: Position = Field(alias="stackPosition")
    cap_height: int = Field(alias="capHeight")
    total_height: int = Field(alias="totalHeight")


class RingEliminationChoiceRequest(BaseModel):
    """Request model for AI-backed ring elimination choices.

    Carries the same metadata as LineRewardChoiceRequest plus the
    ring-elimination specific option list.
    """
    model_config = ConfigDict(populate_by_name=True)

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: list[RingEliminationChoiceOption]


class RingEliminationChoiceResponse(BaseModel):
    """Response model for AI-backed ring elimination choices."""
    model_config = ConfigDict(populate_by_name=True)

    selected_option: RingEliminationChoiceOption = Field(
        alias="selectedOption"
    )
    ai_type: str = Field(alias="aiType")
    difficulty: int


class RegionOrderChoiceOption(BaseModel):
    """Option for region order choice.

    Mirrors the TypeScript RegionOrderChoice option shape:
    { regionId, size, representativePosition }.
    """
    model_config = ConfigDict(populate_by_name=True)

    region_id: str = Field(alias="regionId")
    size: int
    representative_position: Position = Field(alias="representativePosition")


class ProgressSnapshot(BaseModel):
    """
    Canonical, engine-agnostic progress snapshot used for invariant checks
    and history entries. S is defined as markers + collapsed + eliminated.
    """
    markers: int
    collapsed: int
    eliminated: int
    S: int


class RegionOrderChoiceRequest(BaseModel):
    """Request model for AI-backed region order choices."""
    model_config = ConfigDict(populate_by_name=True)

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: list[RegionOrderChoiceOption]


class RegionOrderChoiceResponse(BaseModel):
    """Response model for AI-backed region order choices."""
    model_config = ConfigDict(populate_by_name=True)

    selected_option: RegionOrderChoiceOption = Field(alias="selectedOption")
    ai_type: str = Field(alias="aiType")
    difficulty: int


class LineOrderChoiceLine(BaseModel):
    """Single line option for line_order choice.

    Mirrors the TypeScript LineOrderChoice option shape:
    { lineId, markerPositions }.
    """
    model_config = ConfigDict(populate_by_name=True)

    line_id: str = Field(alias="lineId")
    marker_positions: list[Position] = Field(alias="markerPositions")


class LineOrderChoiceRequest(BaseModel):
    """Request model for AI-backed line order choices.

    Carries the same metadata pattern as other choice requests. The
    current implementation does not require full GameState but accepts
    it for future, more context-aware heuristics.
    """
    model_config = ConfigDict(populate_by_name=True)

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: list[LineOrderChoiceLine]


class LineOrderChoiceResponse(BaseModel):
    """Response model for AI-backed line order choices."""
    model_config = ConfigDict(populate_by_name=True)

    selected_option: LineOrderChoiceLine = Field(alias="selectedOption")
    ai_type: str = Field(alias="aiType")
    difficulty: int


class CaptureDirectionChoiceOption(BaseModel):
    """Option for capture_direction choice.

    Mirrors the TypeScript CaptureDirectionChoice option shape:
    { targetPosition, landingPosition, capturedCapHeight }.
    """
    model_config = ConfigDict(populate_by_name=True)

    target_position: Position = Field(alias="targetPosition")
    landing_position: Position = Field(alias="landingPosition")
    captured_cap_height: int = Field(alias="capturedCapHeight")


class CaptureDirectionChoiceRequest(BaseModel):
    """Request model for AI-backed capture direction choices."""
    model_config = ConfigDict(populate_by_name=True)

    game_state: Optional[GameState] = Field(None, alias="gameState")
    player_number: int = Field(alias="playerNumber")
    difficulty: int = Field(ge=1, le=10, default=5)
    ai_type: Optional[AIType] = Field(None, alias="aiType")
    options: list[CaptureDirectionChoiceOption]


class CaptureDirectionChoiceResponse(BaseModel):
    """Response model for AI-backed capture direction choices."""
    model_config = ConfigDict(populate_by_name=True)

    selected_option: CaptureDirectionChoiceOption = Field(
        alias="selectedOption"
    )
    ai_type: str = Field(alias="aiType")
    difficulty: int
