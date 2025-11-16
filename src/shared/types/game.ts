export type BoardType = 'square8' | 'square19' | 'hexagonal';
export type GamePhase = 'ring_placement' | 'movement' | 'capture' | 'line_processing' | 'territory_processing';
export type GameStatus = 'waiting' | 'active' | 'finished' | 'paused' | 'abandoned' | 'completed';
export type MarkerType = 'regular' | 'collapsed';
export type MoveType = 'place_ring' | 'move_ring' | 'build_stack' | 'move_stack' | 'overtaking_capture' | 'line_formation' | 'territory_claim';
export type PlayerType = 'human' | 'ai';
export type CaptureType = 'overtaking' | 'elimination';
export type AdjacencyType = 'moore' | 'von_neumann' | 'hexagonal';

export interface Position {
  x: number;
  y: number;
  z?: number; // For hexagonal boards
}

export type AIControlMode = 'local_heuristic' | 'service';

export type AITacticType = 'random' | 'heuristic' | 'minimax' | 'mcts';

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

export interface Move {
  id: string;
  type: MoveType;
  player: number;
  from?: Position;
  to: Position;
  buildAmount?: number; // For build_stack moves
  
  // Ring placement specific
  placedOnStack?: boolean;
  
  // Movement specific
  stackMoved?: RingStack;
  minimumDistance?: number;
  actualDistance?: number;
  markerLeft?: Position; // Where marker was left
  
  // Capture specific
  captureType?: CaptureType;
  captureTarget?: Position; // Position of the stack being captured (for overtaking)
  capturedStacks?: RingStack[];
  captureChain?: Position[]; // Sequence of capture positions
  overtakenRings?: number[]; // Player numbers of overtaken rings
  
  // Line formation specific
  formedLines?: LineInfo[];
  collapsedMarkers?: Position[];
  
  // Territory specific
  claimedTerritory?: Territory[];
  disconnectedRegions?: Territory[];
  eliminatedRings?: { player: number; count: number }[];
  
  timestamp: Date;
  thinkTime: number;
  moveNumber: number;
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
  moveHistory: Move[];
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
  reason: 'ring_elimination' | 'territory_control' | 'last_player_standing' | 'timeout' | 'resignation' | 'draw' | 'abandonment' | 'game_completed';
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
    type: 'square' as const
  },
  square19: {
    size: 19,
    totalSpaces: 361,
    ringsPerPlayer: 36,
    lineLength: 5, // Minimum line length for collapse
    movementAdjacency: 'moore' as AdjacencyType, // 8-direction movement
    lineAdjacency: 'moore' as AdjacencyType, // 8-direction line formation
    territoryAdjacency: 'von_neumann' as AdjacencyType, // 4-direction territory
    type: 'square' as const
  },
  hexagonal: {
    size: 11, // Radius of hexagonal board
    totalSpaces: 331,
    ringsPerPlayer: 36,
    lineLength: 5, // Minimum line length for collapse
    movementAdjacency: 'hexagonal' as AdjacencyType, // 6-direction movement
    lineAdjacency: 'hexagonal' as AdjacencyType, // 6-direction line formation
    territoryAdjacency: 'hexagonal' as AdjacencyType, // 6-direction territory
    type: 'hexagonal' as const
  }
} as const;

export type BoardConfig = typeof BOARD_CONFIGS[keyof typeof BOARD_CONFIGS];

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
