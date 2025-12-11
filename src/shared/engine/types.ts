import {
  BoardState,
  Player,
  GamePhase,
  GameStatus,
  Position,
  Move,
  BoardType,
  TimeControl,
} from '../types/game';

// Re-export types used in the engine interface
export type { Position, BoardType, GamePhase, GameStatus, BoardState };

/**
 * The GameState interface for the new engine architecture.
 * It emphasizes immutability (readonly fields) and serves as the single source of truth.
 */
export interface GameState {
  readonly id: string;
  readonly board: BoardState;
  readonly players: ReadonlyArray<Player>;
  readonly currentPhase: GamePhase;
  readonly currentPlayer: number;
  readonly moveHistory: ReadonlyArray<Move>;
  readonly gameStatus: GameStatus;
  readonly timeControl: TimeControl;
  readonly winner?: number;
  readonly createdAt: Date;
  readonly lastMoveAt: Date;
  readonly isRated: boolean;
  readonly maxPlayers: number;

  // RingRift specific state
  readonly totalRingsInPlay: number;
  readonly totalRingsEliminated: number;
  readonly victoryThreshold: number;
  readonly territoryVictoryThreshold: number;
}

/**
 * Action Types
 * All player interactions and system events are modeled as Actions.
 */
export type ActionType =
  | 'PLACE_RING'
  | 'MOVE_STACK'
  | 'OVERTAKING_CAPTURE'
  | 'CONTINUE_CHAIN'
  | 'PROCESS_LINE'
  | 'CHOOSE_LINE_REWARD'
  | 'PROCESS_TERRITORY'
  | 'ELIMINATE_STACK' // For Forced Elimination Choice
  | 'SKIP_PLACEMENT';

export interface BaseAction {
  type: ActionType;
  playerId: number;
}

export interface PlaceRingAction extends BaseAction {
  type: 'PLACE_RING';
  position: Position;
  count: number;
}

export interface MoveStackAction extends BaseAction {
  type: 'MOVE_STACK';
  from: Position;
  to: Position;
}

export interface OvertakingCaptureAction extends BaseAction {
  type: 'OVERTAKING_CAPTURE';
  from: Position;
  to: Position;
  captureTarget: Position;
}

export interface ContinueChainAction extends BaseAction {
  type: 'CONTINUE_CHAIN';
  from: Position;
  to: Position;
  captureTarget: Position;
}

export interface ProcessLineAction extends BaseAction {
  type: 'PROCESS_LINE';
  lineIndex: number; // Index in board.formedLines
}

export interface ChooseLineRewardAction extends BaseAction {
  type: 'CHOOSE_LINE_REWARD';
  lineIndex: number;
  selection: 'COLLAPSE_ALL' | 'MINIMUM_COLLAPSE';
  collapsedPositions?: Position[]; // Required for MINIMUM_COLLAPSE
}

export interface ProcessTerritoryAction extends BaseAction {
  type: 'PROCESS_TERRITORY';
  regionId: string; // Key in board.territories
}

export interface EliminateStackAction extends BaseAction {
  type: 'ELIMINATE_STACK';
  stackPosition: Position;
  /**
   * Context for elimination determines how many rings to eliminate:
   * - 'line': Eliminate exactly ONE ring from the top (RR-CANON-R122)
   * - 'territory': Eliminate entire cap from eligible stack (multicolor or height > 1) (RR-CANON-R145)
   * - 'forced': Eliminate entire cap from any controlled stack (RR-CANON-R100)
   * Defaults to 'territory' if not specified for backward compatibility.
   */
  eliminationContext?: 'line' | 'territory' | 'forced';
}

export interface SkipPlacementAction extends BaseAction {
  type: 'SKIP_PLACEMENT';
}

export type GameAction =
  | PlaceRingAction
  | MoveStackAction
  | OvertakingCaptureAction
  | ContinueChainAction
  | ProcessLineAction
  | ChooseLineRewardAction
  | ProcessTerritoryAction
  | EliminateStackAction
  | SkipPlacementAction;

/**
 * Validation
 */
export type ValidationResult = { valid: true } | { valid: false; reason: string; code: string };

export type Validator<T extends GameAction> = (state: GameState, action: T) => ValidationResult;

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED VALIDATION OUTCOME (for new validators)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Canonical validation error codes.
 *
 * New validators should use these codes for consistent error handling across
 * the codebase. Existing validators use string codes for backwards compatibility.
 *
 * Naming convention: DOMAIN_SPECIFIC_ERROR
 * - PLACEMENT_*: Ring placement errors
 * - MOVEMENT_*: Stack movement errors
 * - CAPTURE_*: Overtaking capture errors
 * - PHASE_*: Phase/turn errors
 * - GENERAL_*: Cross-domain errors
 */
export enum ValidationErrorCode {
  // General
  GENERAL_INVALID_PLAYER = 'GENERAL_INVALID_PLAYER',
  GENERAL_GAME_NOT_ACTIVE = 'GENERAL_GAME_NOT_ACTIVE',
  GENERAL_NOT_YOUR_TURN = 'GENERAL_NOT_YOUR_TURN',

  // Phase
  PHASE_INVALID_MOVE_TYPE = 'PHASE_INVALID_MOVE_TYPE',
  PHASE_DECISION_REQUIRED = 'PHASE_DECISION_REQUIRED',

  // Placement
  PLACEMENT_NO_RINGS_IN_HAND = 'PLACEMENT_NO_RINGS_IN_HAND',
  PLACEMENT_INVALID_POSITION = 'PLACEMENT_INVALID_POSITION',
  PLACEMENT_POSITION_COLLAPSED = 'PLACEMENT_POSITION_COLLAPSED',
  PLACEMENT_POSITION_HAS_MARKER = 'PLACEMENT_POSITION_HAS_MARKER',
  PLACEMENT_EXCEEDS_CAP = 'PLACEMENT_EXCEEDS_CAP',
  PLACEMENT_DEAD_PLACEMENT = 'PLACEMENT_DEAD_PLACEMENT',

  // Movement
  MOVEMENT_NO_STACK_AT_POSITION = 'MOVEMENT_NO_STACK_AT_POSITION',
  MOVEMENT_NOT_CONTROLLING_STACK = 'MOVEMENT_NOT_CONTROLLING_STACK',
  MOVEMENT_INVALID_DESTINATION = 'MOVEMENT_INVALID_DESTINATION',
  MOVEMENT_MUST_MOVE_FROM_STACK = 'MOVEMENT_MUST_MOVE_FROM_STACK',

  // Capture
  CAPTURE_NO_TARGET = 'CAPTURE_NO_TARGET',
  CAPTURE_CANNOT_OVERTAKE = 'CAPTURE_CANNOT_OVERTAKE',
  CAPTURE_INVALID_LANDING = 'CAPTURE_INVALID_LANDING',
  CAPTURE_CHAIN_REQUIRED = 'CAPTURE_CHAIN_REQUIRED',

  // Recovery
  RECOVERY_NO_BURIED_RINGS = 'RECOVERY_NO_BURIED_RINGS',
  RECOVERY_INVALID_SLIDE = 'RECOVERY_INVALID_SLIDE',

  // Line/Territory
  LINE_NO_PENDING_LINES = 'LINE_NO_PENDING_LINES',
  LINE_INVALID_CHOICE = 'LINE_INVALID_CHOICE',
  TERRITORY_NO_PENDING_REGIONS = 'TERRITORY_NO_PENDING_REGIONS',
  TERRITORY_INVALID_ELIMINATION = 'TERRITORY_INVALID_ELIMINATION',
}

/**
 * Unified validation outcome for new validators.
 *
 * This type provides a consistent structure for validation results across
 * all domain validators. It supports:
 *
 * - Type-safe success data via generic parameter T
 * - Structured error codes via ValidationErrorCode enum
 * - Human-readable error messages
 * - Optional context for debugging
 *
 * **Migration strategy**: New validators should use ValidationOutcome<T>.
 * Existing validators continue using ValidationResult for backwards compatibility.
 * Over time, as validators are touched for other reasons, they can be migrated.
 *
 * @example
 * ```typescript
 * // Success case
 * const success: ValidationOutcome<PlacementData> = {
 *   valid: true,
 *   data: { position: { x: 3, y: 3 }, count: 2 }
 * };
 *
 * // Failure case
 * const failure: ValidationOutcome<PlacementData> = {
 *   valid: false,
 *   code: ValidationErrorCode.PLACEMENT_DEAD_PLACEMENT,
 *   reason: 'Placement would leave stack with no legal moves',
 *   context: { position: { x: 3, y: 3 } }
 * };
 * ```
 */
export type ValidationOutcome<T = void> =
  | { valid: true; data: T }
  | { valid: false; code: ValidationErrorCode; reason: string; context?: Record<string, unknown> };

/**
 * Type guard to check if a ValidationOutcome is successful.
 */
export function isValidOutcome<T>(
  outcome: ValidationOutcome<T>
): outcome is { valid: true; data: T } {
  return outcome.valid === true;
}

/**
 * Helper to create a successful validation outcome.
 */
export function validOutcome<T>(data: T): ValidationOutcome<T> {
  return { valid: true, data };
}

/**
 * Helper to create a failed validation outcome.
 */
export function invalidOutcome<T = void>(
  code: ValidationErrorCode,
  reason: string,
  context?: Record<string, unknown>
): ValidationOutcome<T> {
  return { valid: false, code, reason, context };
}

/**
 * Mutation
 */
export type Mutator<T extends GameAction> = (state: GameState, action: T) => GameState;

/**
 * Events
 */
export type GameEventType =
  | 'GAME_INITIALIZED'
  | 'ACTION_PROCESSED'
  | 'PHASE_CHANGED'
  | 'GAME_COMPLETED'
  | 'ERROR_OCCURRED';

/** Payload for ACTION_PROCESSED events */
export interface ActionProcessedPayload {
  action: GameAction;
  newState: GameState;
}

/** Payload for ERROR_OCCURRED events */
export interface ErrorOccurredPayload {
  error: string;
  code: string;
}

/** Payload for PHASE_CHANGED events */
export interface PhaseChangedPayload {
  previousPhase: GamePhase;
  newPhase: GamePhase;
}

/** Payload for GAME_COMPLETED events */
export interface GameCompletedPayload {
  winner?: number;
  reason: string;
}

/** Payload for GAME_INITIALIZED events */
export interface GameInitializedPayload {
  initialState: GameState;
}

/** Union of all possible event payloads */
export type GameEventPayload =
  | ActionProcessedPayload
  | ErrorOccurredPayload
  | PhaseChangedPayload
  | GameCompletedPayload
  | GameInitializedPayload;

export interface GameEvent {
  type: GameEventType;
  gameId: string;
  timestamp: number;
  payload?: unknown;
}
