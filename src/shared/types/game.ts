export type BoardType = 'square8' | 'square19' | 'hexagonal';
/**
 * High-level game phases for the RingRift turn engine.
 *
 * - 'capture' is the initial overtaking capture that may start a chain.
 * - 'chain_capture' is an interactive phase where the same player chooses
 *   follow-up capture segments to continue or end an existing chain.
 *
 * Canonical phase → MoveType contract (target for both backend GameEngine and
 * ClientSandboxEngine):
 *
 * - 'ring_placement'
 *   - Legal MoveType values:
 *     - 'place_ring'      – place one or more rings for the active player.
 *     - 'skip_placement'  – explicitly skip optional placement when movement
 *                            or capture is already available (no board change).
 * - 'movement'
 *   - Legal MoveType values:
 *     - 'move_stack' / 'move_ring' – non-capture movement of an entire stack.
 *     - 'build_stack'              – legacy intra-region stack reconfiguration (rare).
 *     - 'overtaking_capture'       – initial overtaking capture that may start a chain.
 * - 'capture'
 *   - Legal MoveType values:
 *     - 'overtaking_capture' – initial overtaking capture chosen directly from
 *                              the capture phase (alternative entry to chains).
 * - 'chain_capture'
 *   - Legal MoveType values:
 *     - 'continue_capture_segment' – mandatory follow-up capture segments in an
 *                                    existing chain until no further segments exist.
 * - 'line_processing'
 *   - Target MoveType values (future-unified model):
 *     - 'process_line'       – choose which detected line to process next.
 *     - 'choose_line_reward' – choose Option 1 vs Option 2 for a specific line.
 * - 'territory_processing'
 *   - Target MoveType values (future-unified model):
 *     - 'process_territory_region' – choose which disconnected region to resolve first.
 *     - 'eliminate_rings_from_stack' – choose which stack/cap to self-eliminate
 *                                      as part of the mandatory follow-up.
 *
 * Engines are expected to expose *only* the MoveType values listed above from
 * getValidMoves for a given phase. PlayerChoice is a transport/UI concern and
 * should conceptually be "choose one Move from getValidMoves(...)"; it must
 * not introduce additional semantics outside this Move space.
 */
export type GamePhase =
  | 'ring_placement'
  | 'movement'
  | 'capture'
  | 'chain_capture'
  | 'line_processing'
  | 'territory_processing';
export type GameStatus = 'waiting' | 'active' | 'finished' | 'paused' | 'abandoned' | 'completed';
export type MarkerType = 'regular' | 'collapsed';
/**
 * Discriminant for the canonical {@link Move} type.
 *
 * Notes:
 * - Some values (e.g. 'move_ring', 'line_formation', 'territory_claim') are
 *   legacy/experimental. New code should prefer their canonical equivalents
 *   ('move_stack', explicit line/territory processing moves).
 * - The phase → MoveType contract is documented above in {@link GamePhase}.
 */
export type MoveType =
  | 'place_ring'
  // Legacy alias for non-capture stack movement. The canonical type for
  // simple movement is 'move_stack'; existing clients/tests may still
  // emit or accept 'move_ring', and the backend RuleEngine/GameEngine
  // treat it equivalently to 'move_stack'.
  | 'move_ring'
  | 'build_stack'
  // Canonical non-capture movement type for moving entire stacks.
  | 'move_stack'
  // Capture and capture-chain moves.
  | 'overtaking_capture'
  | 'continue_capture_segment'
  // Line-processing decisions (see GamePhase 'line_processing').
  | 'process_line'
  | 'choose_line_reward'
  // Territory-processing decisions (see GamePhase 'territory_processing').
  | 'process_territory_region'
  | 'eliminate_rings_from_stack'
  // Legacy / experimental move types (not used by the unified Move model).
  | 'line_formation'
  | 'territory_claim'
  | 'skip_placement';
export type PlayerType = 'human' | 'ai';
export type CaptureType = 'overtaking' | 'elimination';
export type AdjacencyType = 'moore' | 'von_neumann' | 'hexagonal';

export interface Position {
  x: number;
  y: number;
  z?: number; // For hexagonal boards
}

export type AIControlMode = 'local_heuristic' | 'service';

export type AITacticType = 'random' | 'heuristic' | 'minimax' | 'mcts' | 'descent';

/**
 * Configuration for how many AI opponents should participate in a
 * newly created game and how they should behave. This shape is
 * intentionally mirrored by CreateGameSchema in Zod to keep the
 * HTTP API, validation layer, and shared types aligned.
 */
export interface AiOpponentsConfig {
  count: number;
  /**
   * Difficulty levels for the AI opponents (1-10). The length and
   * indexing semantics are interpreted by the server when assigning
   * AI players to seats.
   */
  difficulty: number[];
  /** Where the AI logic runs for these opponents. */
  mode?: AIControlMode | undefined;
  /** Which tactical engine should be used for these opponents. */
  aiType?: AITacticType | undefined;
}

/**
 * Shared create-game request payload used by the client, server
 * route handler, and validation schema. This is the long-term
 * source of truth for the game creation API shape.
 */
export interface CreateGameRequest {
  boardType: BoardType;
  timeControl: TimeControl;
  isRated: boolean;
  isPrivate: boolean;
  maxPlayers: number;
  aiOpponents?: AiOpponentsConfig;
}

export interface AIProfile {
  /** Primary difficulty knob for this AI player (1-10). */
  difficulty: number;
  /**
   * How this AI makes decisions about moves:
   * - 'service': rely primarily on the Python AI service via AIServiceClient/globalAIEngine.
   * - 'local_heuristic': use local TypeScript heuristics (future extension for moves).
   *
   * Note: Regardless of this mode, PlayerChoices are currently answered via
   * AIInteractionHandler (local heuristics) with the service as an optional
   * future enhancement.
   */
  mode?: AIControlMode;
  /** The underlying tactical engine type (random, heuristic, minimax, mcts). */
  aiType?: AITacticType;
}

export interface Player {
  id: string;
  username: string;
  type: PlayerType;
  playerNumber: number;
  rating?: number;
  isReady: boolean;
  timeRemaining: number;
  aiDifficulty?: number; // 1-10 for AI players (legacy, see aiProfile)
  aiProfile?: AIProfile; // Rich AI configuration for AI players
  ringsInHand: number; // Rings not yet placed on board
  eliminatedRings: number; // Rings permanently removed from game
  territorySpaces: number; // Spaces controlled as territory
}

// Ring stack representation
export interface RingStack {
  position: Position;
  rings: number[]; // Array of player numbers, bottom to top
  stackHeight: number; // Total rings in stack
  capHeight: number; // Consecutive rings of same color from top
  controllingPlayer: number; // Player number of top ring
}

export interface MarkerInfo {
  player: number; // Player who owns this marker
  position: Position;
  type: MarkerType; // 'regular' for standard marker, 'collapsed' for claimed territory
}

// Territory representation
export interface Territory {
  spaces: Position[];
  controllingPlayer: number;
  isDisconnected: boolean;
}

// Line formation for marker collapse
export interface LineInfo {
  positions: Position[];
  player: number;
  length: number;
  direction: Position; // Direction vector
}

/**
 * Canonical, engine-agnostic action applied by GameEngine and ClientSandboxEngine.
 *
 * Semantics by phase (see {@link GamePhase}) and type (see {@link MoveType}):
 *
 * - ring_placement
 *   - type: 'place_ring'
 *     - Required:
 *       - player      – active player.
 *       - to          – destination cell for placement.
 *     - Optional:
 *       - placementCount – number of rings to place (defaults to 1).
 *       - placedOnStack  – hint for UI/tests; true when placing onto an existing stack.
 *   - type: 'skip_placement'
 *     - No board coordinates are semantically meaningful; `to` is a sentinel.
 *
 * - movement
 *   - type: 'move_stack' | 'move_ring'
 *     - Required:
 *       - from, to    – origin and landing positions of the moved stack.
 *       - player      – active player (must control the stack at `from`).
 *     - Optional diagnostics (mirroring RuleEngine):
 *       - stackMoved      – snapshot of the moved stack before movement.
 *       - minimumDistance – required distance (stack height).
 *       - actualDistance  – realised distance.
 *       - markerLeft      – where a departure marker was placed (if any).
 *
 * - capture / chain_capture
 *   - type: 'overtaking_capture' (initial segment) or 'continue_capture_segment'
 *     - Required:
 *       - from          – origin of the capturing stack.
 *       - captureTarget – position of the stack being overtaken.
 *       - to            – landing position after the segment.
 *     - Optional diagnostics:
 *       - captureType     – usually 'overtaking'.
 *       - capturedStacks  – stacks affected by this move (before state).
 *       - captureChain    – historical list of visited capture targets/landings.
 *       - overtakenRings  – colours of rings overtaken so far in the chain.
 *
 * - line_processing (target unified model for line decisions)
 *   - type: 'process_line'
 *     - Required:
 *       - formedLines[0] – identifies the line to process (positions, owner, direction).
 *   - type: 'choose_line_reward'
 *     - Required:
 *       - formedLines[0] – identifies the line being rewarded.
 *       - collapsedMarkers – subset of marker positions chosen for collapse
 *                            when selecting Option 2 (minimum collapse).
 *
 * - territory_processing (target unified model for territory decisions)
 *   - type: 'process_territory_region'
 *     - Required:
 *       - disconnectedRegions[0] – identifies the region being processed
 *                                  (spaces, controllingPlayer, isDisconnected).
 *   - type: 'eliminate_rings_from_stack'
 *     - Required:
 *       - eliminatedRings[0] – { player, count } describing the explicit
 *                              self-elimination choice for this region.
 *
 * Engines are free to augment moves with additional diagnostic fields, but
 * should not rely on consumers interpreting fields outside the contract above.
 * Tests and parity tooling should treat {@link Move} as the single source of
 * truth for "what action happened at step N".
 */
export interface Move {
  /** Stable identifier for this move instance (UUID on the backend). */
  id: string;
  /** Discriminant describing the kind of action. */
  type: MoveType;
  /** Numeric player index performing the action. */
  player: number;

  /**
   * Origin position for movement/capture-style moves. Undefined for pure
   * placement and most bookkeeping-only actions.
   */
  from?: Position;

  /**
   * Destination/landing position for movement, capture, and placement moves.
   * For moves that have no spatial meaning (e.g. 'skip_placement'), engines
   * may supply a harmless sentinel.
   */
  to: Position;

  /** For 'build_stack' moves: how many rings are transferred. */
  buildAmount?: number;

  // Ring placement specific
  /** True if rings were placed onto an existing stack rather than an empty cell. */
  placedOnStack?: boolean;
  /** Number of rings placed in this action (defaults to 1 when omitted). */
  placementCount?: number;

  // Movement specific
  /** Snapshot of the moved stack before movement (for diagnostics/parity). */
  stackMoved?: RingStack;
  /** Required distance for the move (usually the pre-move stack height). */
  minimumDistance?: number;
  /** Actual realised distance of the move. */
  actualDistance?: number;
  /** Where a departure marker was left (if any). */
  markerLeft?: Position;

  // Capture specific
  /** High-level capture kind; currently always 'overtaking' for chain segments. */
  captureType?: CaptureType;
  /** Position of the stack being captured/overtaken (for overtaking captures). */
  captureTarget?: Position;
  /** Snapshot(s) of stacks captured by this action (before state). */
  capturedStacks?: RingStack[];
  /** Sequence of capture positions visited so far in the chain. */
  captureChain?: Position[];
  /** Player numbers of overtaken rings added to the capturing stack. */
  overtakenRings?: number[];

  // Line formation specific
  /** Lines detected/processed as part of this action (if any). */
  formedLines?: LineInfo[];
  /** Marker positions collapsed to territory as a consequence of this action. */
  collapsedMarkers?: Position[];

  // Territory specific
  /** Newly claimed territories as a result of this action. */
  claimedTerritory?: Territory[];
  /** Disconnected regions processed by this action (if any). */
  disconnectedRegions?: Territory[];
  /**
   * Summary of rings eliminated by this action, grouped per player. For
   * territory/self-elimination decisions, this is the canonical record of
   * which player lost how many rings.
   */
  eliminatedRings?: { player: number; count: number }[];

  /** Wall-clock timestamp when the move was created/applied. */
  timestamp: Date;
  /** Think time in milliseconds for the player/AI before this action. */
  thinkTime: number;
  /** Global, 1-based action index within the game history. */
  moveNumber: number;
}

/**
 * Canonical, engine-agnostic progress snapshot used for invariant checks
 * and history entries. S is defined as markers + collapsed + eliminated
 * (Section 13.5 of the complete rules).
 */
export interface ProgressSnapshot {
  markers: number;
  collapsed: number;
  eliminated: number;
  S: number;
}

/**
 * Lightweight, order-independent summary of board state used for parity
 * debugging and history logging. The concrete string formats are not
 * intended for UI consumption; they exist to make backend/sandbox diffs
 * stable and easy to compare in tests/logs.
 */
export interface BoardSummary {
  /** entries like `${key}:${controllingPlayer}:${stackHeight}:${capHeight}` */
  stacks: string[];
  /** entries like `${key}:${player}` */
  markers: string[];
  /** entries like `${key}:${owner}` */
  collapsedSpaces: string[];
}

/**
 * Structured history entry capturing a single canonical action and its
 * before/after context. Both backend GameEngine and client
 * ClientSandboxEngine can emit these entries so tests and tools can
 * compare traces step-by-step.
 */
export interface GameHistoryEntry {
  /** 1-based move/action index for the lifetime of the game. */
  moveNumber: number;
  /** Canonical action that was applied. */
  action: Move;
  /** Player who performed the action (same as action.player). */
  actor: number;

  phaseBefore: GamePhase;
  phaseAfter: GamePhase;
  statusBefore: GameStatus;
  statusAfter: GameStatus;

  progressBefore: ProgressSnapshot;
  progressAfter: ProgressSnapshot;

  /** Optional but highly recommended for parity debugging. */
  stateHashBefore?: string;
  stateHashAfter?: string;

  /** Optional shallow board summaries for diff-friendly diagnostics. */
  boardBeforeSummary?: BoardSummary;
  boardAfterSummary?: BoardSummary;
}

/**
 * Complete trace of a game from an initial state through a sequence of
 * history entries. Used by test harnesses and tooling to replay the same
 * action list into different engines (backend vs sandbox).
 */
export interface GameTrace {
  initialState: GameState;
  entries: GameHistoryEntry[];
}

/**
 * Wire-level move payload used by WebSockets and HTTP APIs. This
 * intentionally stays simpler than the internal Move type and is
 * validated by MoveSchema in src/shared/validation/schemas.ts.
 */
export interface MovePayload {
  moveType: MoveType;
  /**
   * Either a JSON string or a structured object describing the
   * move positions. The legacy shape uses a stringified
   * `{ from?: Position; to: Position }` object; newer clients
   * may send the structured object directly.
   */
  position: string | { from?: Position; to: Position };
  moveNumber?: number;
}

export interface BoardState {
  stacks: Map<string, RingStack>; // Position string -> RingStack
  markers: Map<string, MarkerInfo>; // Position string -> MarkerInfo (regular markers only)
  collapsedSpaces: Map<string, number>; // Position string -> player number (collapsed territory)
  territories: Map<string, Territory>; // Region ID -> Territory
  formedLines: LineInfo[]; // Completed lines awaiting collapse
  eliminatedRings: { [player: number]: number }; // Count of eliminated rings per player
  size: number;
  type: BoardType;
}

export interface TimeControl {
  initialTime: number; // seconds
  increment: number; // seconds per move
  type: 'blitz' | 'rapid' | 'classical';
}
export interface GameState {
  id: string;
  boardType: BoardType;
  board: BoardState;
  players: Player[];
  currentPhase: GamePhase;
  currentPlayer: number;
  /**
   * Legacy linear move log preserved for backward compatibility with
   * existing clients/tests. New tooling should prefer the richer
   * GameHistoryEntry-based history when available.
   */
  moveHistory: Move[];
  /**
   * Structured history entries emitted by engines that support
   * event-sourced tracing. This is the canonical source for parity and
   * S-invariant debugging. Engines that do not yet record history
   * initialise this as an empty array.
   */
  history: GameHistoryEntry[];
  timeControl: TimeControl;
  spectators: string[]; // User IDs
  gameStatus: GameStatus;
  winner?: number | undefined;
  createdAt: Date;
  lastMoveAt: Date;
  isRated: boolean;
  maxPlayers: number;

  // RingRift specific state
  totalRingsInPlay: number; // Total rings placed on board
  totalRingsEliminated: number; // Total rings eliminated from game
  victoryThreshold: number; // Rings needed to win (>50% of total)
  territoryVictoryThreshold: number; // Territory spaces needed to win (>50% of board)
}

export interface GameResult {
  winner?: number;
  reason:
    | 'ring_elimination'
    | 'territory_control'
    | 'last_player_standing'
    | 'timeout'
    | 'resignation'
    | 'draw'
    | 'abandonment'
    | 'game_completed';
  finalScore: {
    ringsEliminated: { [playerNumber: number]: number };
    territorySpaces: { [playerNumber: number]: number };
    ringsRemaining: { [playerNumber: number]: number };
  };
  ratingChanges?: { [playerId: string]: number };
}

// Legacy interface for compatibility - will be removed
export interface RowInfo {
  positions: Position[];
  player: number;
  isComplete: boolean;
  length: number;
}

export interface WinCondition {
  type: 'ring_elimination' | 'territory_control' | 'last_player_standing';
  progress: { [playerNumber: number]: number };
  threshold: number;
}

// Database Game interface for API compatibility
export interface Game {
  id: string;
  boardType: BoardType;
  maxPlayers: number;
  timeControl: TimeControl;
  isRated: boolean;
  allowSpectators: boolean;
  status: GameStatus;
  gameState: GameState;

  // Players
  player1Id?: string;
  player2Id?: string;
  player3Id?: string;
  player4Id?: string;

  // Player relations (populated by API)
  player1?: { id: string; username: string; rating?: number };
  player2?: { id: string; username: string; rating?: number };
  player3?: { id: string; username: string; rating?: number };
  player4?: { id: string; username: string; rating?: number };

  // Game result
  winnerId?: string;

  // Timestamps
  createdAt: Date;
  updatedAt: Date;
  startedAt?: Date;
  endedAt?: Date;
}

// Utility functions for position handling
export const positionToString = (pos: Position): string => {
  return pos.z !== undefined ? `${pos.x},${pos.y},${pos.z}` : `${pos.x},${pos.y}`;
};

export const stringToPosition = (str: string): Position => {
  const parts = str.split(',').map(Number);
  return parts.length === 3
    ? { x: parts[0], y: parts[1], z: parts[2] }
    : { x: parts[0], y: parts[1] };
};

export const positionsEqual = (pos1: Position, pos2: Position): boolean => {
  return pos1.x === pos2.x && pos1.y === pos2.y && (pos1.z || 0) === (pos2.z || 0);
};

// Board configuration constants for RingRift
export const BOARD_CONFIGS = {
  square8: {
    size: 8,
    totalSpaces: 64,
    ringsPerPlayer: 18,
    lineLength: 4, // Minimum line length for collapse
    movementAdjacency: 'moore' as AdjacencyType, // 8-direction movement
    lineAdjacency: 'moore' as AdjacencyType, // 8-direction line formation
    territoryAdjacency: 'von_neumann' as AdjacencyType, // 4-direction territory
    type: 'square' as const,
  },
  square19: {
    size: 19,
    totalSpaces: 361,
    ringsPerPlayer: 36,
    lineLength: 5, // Minimum line length for collapse
    movementAdjacency: 'moore' as AdjacencyType, // 8-direction movement
    lineAdjacency: 'moore' as AdjacencyType, // 8-direction line formation
    territoryAdjacency: 'von_neumann' as AdjacencyType, // 4-direction territory
    type: 'square' as const,
  },
  hexagonal: {
    size: 11, // Radius of hexagonal board
    totalSpaces: 331,
    ringsPerPlayer: 36,
    lineLength: 5, // Minimum line length for collapse
    movementAdjacency: 'hexagonal' as AdjacencyType, // 6-direction movement
    lineAdjacency: 'hexagonal' as AdjacencyType, // 6-direction line formation
    territoryAdjacency: 'hexagonal' as AdjacencyType, // 6-direction territory
    type: 'hexagonal' as const,
  },
} as const;

export type BoardConfig = (typeof BOARD_CONFIGS)[keyof typeof BOARD_CONFIGS];

// --- Player choice system types ---

export type PlayerChoiceType =
  | 'line_order'
  | 'line_reward_option'
  | 'ring_elimination'
  | 'region_order'
  | 'capture_direction';

export interface PlayerChoiceBase {
  id: string;
  gameId: string;
  playerNumber: number; // numeric player index, consistent with GameState.currentPlayer
  type: PlayerChoiceType;
  prompt: string;
  timeoutMs?: number;
}

export interface LineOrderChoice extends PlayerChoiceBase {
  type: 'line_order';
  options: Array<{
    lineId: string;
    markerPositions: Position[];
  }>;
}

export interface LineRewardChoice extends PlayerChoiceBase {
  type: 'line_reward_option';
  options: Array<'option_1_collapse_all_and_eliminate' | 'option_2_min_collapse_no_elimination'>;
}

export interface RingEliminationChoice extends PlayerChoiceBase {
  type: 'ring_elimination';
  options: Array<{
    stackPosition: Position;
    capHeight: number;
    totalHeight: number;
  }>;
}

export interface RegionOrderChoice extends PlayerChoiceBase {
  type: 'region_order';
  options: Array<{
    regionId: string;
    size: number;
    representativePosition: Position;
  }>;
}

export interface CaptureDirectionChoice extends PlayerChoiceBase {
  type: 'capture_direction';
  options: Array<{
    targetPosition: Position;
    landingPosition: Position;
    capturedCapHeight: number;
  }>;
}

export type PlayerChoice =
  | LineOrderChoice
  | LineRewardChoice
  | RingEliminationChoice
  | RegionOrderChoice
  | CaptureDirectionChoice;

export interface PlayerChoiceResponse<TOption = unknown> {
  choiceId: string;
  playerNumber: number;
  /**
   * Echoes the type of the originating PlayerChoice when available.
   * Optional for backward compatibility while the choice system
   * is still being integrated across transports.
   */
  choiceType?: PlayerChoiceType;
  selectedOption: TOption;
}

/**
 * Convenience helper: given a concrete PlayerChoice type, derive the
 * corresponding PlayerChoiceResponse type with a correctly-typed
 * selectedOption and a narrowed choiceType.
 */
export type PlayerChoiceResponseFor<TChoice extends PlayerChoice> = PlayerChoiceResponse<
  TChoice['options'][number]
> & { choiceType: TChoice['type'] };
