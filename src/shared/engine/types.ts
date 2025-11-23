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

export interface GameEvent {
  type: GameEventType;
  gameId: string;
  timestamp: number;
  payload?: any;
}
