/**
 * TurnStateMachine - Finite State Machine for RingRift turn phases
 *
 * This FSM provides explicit, type-safe phase transitions with guards.
 * All valid (state, event) → nextState transitions are declared upfront,
 * making invalid transitions impossible at compile time.
 *
 * Key design principles:
 * - Discriminated unions for states (phase-specific context)
 * - Exhaustive event handling per state
 * - Guards for conditional transitions
 * - Actions for side effects
 * - No implicit transitions or coercions
 *
 * @module TurnStateMachine
 */

import type { Position, BoardType } from '../../types/game';

// ═══════════════════════════════════════════════════════════════════════════
// CORE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/** Victory reasons */
export type VictoryReason =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'resignation'
  | 'timeout';

/** Direction for captures */
export type Direction = 'N' | 'NE' | 'E' | 'SE' | 'S' | 'SW' | 'W' | 'NW';

/** Line reward choice */
export type LineRewardChoice = 'eliminate' | 'territory';

// ═══════════════════════════════════════════════════════════════════════════
// PHASE CONTEXT TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface DetectedLine {
  positions: Position[];
  player: number;
  requiresChoice: boolean;
}

export interface DisconnectedRegion {
  positions: Position[];
  controllingPlayer: number;
  eliminationsRequired: number;
}

export interface CaptureContext {
  target: Position;
  capturingPlayer: number;
  isChainCapture: boolean;
}

export interface EliminationTarget {
  position: Position;
  player: number;
  count: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// STATES - Discriminated union with phase-specific context
// ═══════════════════════════════════════════════════════════════════════════

export type TurnState =
  | RingPlacementState
  | MovementState
  | CaptureState
  | ChainCaptureState
  | LineProcessingState
  | TerritoryProcessingState
  | ForcedEliminationState
  | TurnEndState
  | GameOverState;

export interface RingPlacementState {
  readonly phase: 'ring_placement';
  readonly player: number;
  /** Rings currently in hand for the active player. */
  readonly ringsInHand: number;
  readonly canPlace: boolean;
  readonly validPositions: Position[];
}

export interface MovementState {
  readonly phase: 'movement';
  readonly player: number;
  readonly canMove: boolean;
  readonly placedRingAt: Position | null;
}

export interface CaptureState {
  readonly phase: 'capture';
  readonly player: number;
  readonly pendingCaptures: CaptureContext[];
  readonly chainInProgress: boolean;
  readonly capturesMade: number;
}

/**
 * ChainCaptureState - Mandatory follow-up capture phase
 *
 * Entered after an initial capture when the attacking stack can continue
 * capturing. The player MUST continue capturing until no more segments
 * are available (captures are mandatory in a chain).
 */
export interface ChainCaptureState {
  readonly phase: 'chain_capture';
  readonly player: number;
  /** Current position of the attacking stack */
  readonly attackerPosition: Position;
  /** Targets already captured in this chain */
  readonly capturedTargets: Position[];
  /** Available continuation targets (computed from game state) */
  readonly availableContinuations: CaptureContext[];
  /** Total segments executed in this chain */
  readonly segmentCount: number;
  /** Whether this is the first segment of a new chain */
  readonly isFirstSegment: boolean;
}

export interface LineProcessingState {
  readonly phase: 'line_processing';
  readonly player: number;
  readonly detectedLines: DetectedLine[];
  readonly currentLineIndex: number;
  readonly awaitingReward: boolean;
}

export interface TerritoryProcessingState {
  readonly phase: 'territory_processing';
  readonly player: number;
  readonly disconnectedRegions: DisconnectedRegion[];
  readonly currentRegionIndex: number;
  readonly eliminationsPending: EliminationTarget[];
}

export interface ForcedEliminationState {
  readonly phase: 'forced_elimination';
  readonly player: number;
  readonly ringsOverLimit: number;
  readonly eliminationsDone: number;
}

export interface TurnEndState {
  readonly phase: 'turn_end';
  readonly completedPlayer: number;
  readonly nextPlayer: number;
}

export interface GameOverState {
  readonly phase: 'game_over';
  readonly winner: number | null;
  readonly reason: VictoryReason;
}

// ═══════════════════════════════════════════════════════════════════════════
// EVENTS - All valid inputs to the state machine
// ═══════════════════════════════════════════════════════════════════════════

export type TurnEvent =
  // Ring Placement
  | { readonly type: 'PLACE_RING'; readonly to: Position }
  | { readonly type: 'SKIP_PLACEMENT' }
  | { readonly type: 'NO_PLACEMENT_ACTION' }

  // Movement
  | { readonly type: 'MOVE_STACK'; readonly from: Position; readonly to: Position }
  | { readonly type: 'NO_MOVEMENT_ACTION' }

  // Capture
  | { readonly type: 'CAPTURE'; readonly target: Position; readonly direction?: Direction }
  | { readonly type: 'CONTINUE_CHAIN'; readonly target: Position }
  | { readonly type: 'END_CHAIN' }

  // Line Processing
  | { readonly type: 'PROCESS_LINE'; readonly lineIndex: number }
  | { readonly type: 'CHOOSE_LINE_REWARD'; readonly choice: LineRewardChoice }
  | { readonly type: 'NO_LINE_ACTION' }

  // Territory Processing
  | { readonly type: 'PROCESS_REGION'; readonly regionIndex: number }
  | { readonly type: 'ELIMINATE_FROM_STACK'; readonly target: Position; readonly count: number }
  | { readonly type: 'NO_TERRITORY_ACTION' }

  // Forced Elimination
  | { readonly type: 'FORCED_ELIMINATE'; readonly target: Position }

  // Meta events
  | { readonly type: 'RESIGN'; readonly player: number }
  | { readonly type: 'TIMEOUT'; readonly player: number }

  // Internal: advance to next player
  | { readonly type: '_ADVANCE_TURN' };

// ═══════════════════════════════════════════════════════════════════════════
// TRANSITION RESULT
// ═══════════════════════════════════════════════════════════════════════════

export type TransitionResult =
  | { readonly ok: true; readonly state: TurnState; readonly actions: Action[] }
  | { readonly ok: false; readonly error: TransitionError };

export interface TransitionError {
  readonly code: 'INVALID_EVENT' | 'GUARD_FAILED' | 'INVALID_STATE';
  readonly message: string;
  readonly currentPhase: string;
  readonly eventType: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// ACTIONS - Side effects to apply after transition
// ═══════════════════════════════════════════════════════════════════════════

export type Action =
  | { readonly type: 'PLACE_RING'; readonly position: Position; readonly player: number }
  | { readonly type: 'LEAVE_MARKER'; readonly position: Position; readonly player: number }
  | { readonly type: 'MOVE_STACK'; readonly from: Position; readonly to: Position }
  | { readonly type: 'EXECUTE_CAPTURE'; readonly target: Position; readonly capturer: number }
  | { readonly type: 'COLLAPSE_LINE'; readonly positions: Position[] }
  | {
      readonly type: 'APPLY_LINE_REWARD';
      readonly choice: LineRewardChoice;
      readonly line: DetectedLine;
    }
  | { readonly type: 'PROCESS_DISCONNECTION'; readonly region: DisconnectedRegion }
  | { readonly type: 'ELIMINATE_RINGS'; readonly target: Position; readonly count: number }
  | { readonly type: 'FORCED_ELIMINATE'; readonly target: Position; readonly player: number }
  | { readonly type: 'CHECK_VICTORY' }
  | { readonly type: 'ADVANCE_PLAYER'; readonly from: number; readonly to: number };

// ═══════════════════════════════════════════════════════════════════════════
// STATE MACHINE IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

export interface GameContext {
  readonly boardType: BoardType;
  readonly numPlayers: number;
  readonly ringsPerPlayer: number;
  readonly lineLength: number;
}

/**
 * Pure transition function - the heart of the FSM.
 * Takes current state + event, returns new state + actions (or error).
 */
export function transition(
  state: TurnState,
  event: TurnEvent,
  context: GameContext
): TransitionResult {
  switch (state.phase) {
    case 'ring_placement':
      return handleRingPlacement(state, event, context);
    case 'movement':
      return handleMovement(state, event, context);
    case 'capture':
      return handleCapture(state, event, context);
    case 'chain_capture':
      return handleChainCapture(state, event, context);
    case 'line_processing':
      return handleLineProcessing(state, event, context);
    case 'territory_processing':
      return handleTerritoryProcessing(state, event, context);
    case 'forced_elimination':
      return handleForcedElimination(state, event, context);
    case 'turn_end':
      return handleTurnEnd(state, event, context);
    case 'game_over':
      return invalidTransition(state, event, 'Game is over - no transitions allowed');
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

function handleRingPlacement(
  state: RingPlacementState,
  event: TurnEvent,
  _context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'PLACE_RING': {
      if (!state.canPlace) {
        return guardFailed(
          state,
          event,
          'Cannot place ring - no rings in hand or no valid positions'
        );
      }
      const isValidPosition = state.validPositions.some(
        (p) => p.x === event.to.x && p.y === event.to.y
      );
      if (!isValidPosition) {
        return guardFailed(
          state,
          event,
          `Invalid placement position: (${event.to.x}, ${event.to.y})`
        );
      }

      return ok<MovementState>(
        {
          phase: 'movement',
          player: state.player,
          canMove: true, // Will be computed by board analysis
          placedRingAt: event.to,
        },
        [
          { type: 'PLACE_RING', position: event.to, player: state.player },
          { type: 'LEAVE_MARKER', position: event.to, player: state.player },
        ]
      );
    }

    case 'SKIP_PLACEMENT': {
      if (state.ringsInHand <= 0) {
        return guardFailed(
          state,
          event,
          'Cannot skip placement when you have no rings in hand; use no_placement_action'
        );
      }
      if (state.canPlace) {
        return guardFailed(state, event, 'Cannot skip placement when valid placements exist');
      }
      return ok<MovementState>(
        {
          phase: 'movement',
          player: state.player,
          canMove: true,
          placedRingAt: null,
        },
        []
      );
    }

    case 'NO_PLACEMENT_ACTION': {
      if (state.canPlace) {
        return guardFailed(state, event, 'Cannot skip placement when valid placements exist');
      }
      // Skip directly to next player's turn if truly no actions
      return ok<TurnEndState>(
        {
          phase: 'turn_end',
          completedPlayer: state.player,
          nextPlayer: nextPlayer(state.player, _context.numPlayers),
        },
        [{ type: 'CHECK_VICTORY' }]
      );
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

function handleMovement(
  state: MovementState,
  event: TurnEvent,
  _context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'MOVE_STACK': {
      if (!state.canMove) {
        return guardFailed(state, event, 'No valid moves available');
      }

      // Movement leads to either capture, line processing, or territory
      // The actual next phase depends on board analysis after the move
      // For now, we go to line_processing as the default post-move phase
      return ok<LineProcessingState>(
        {
          phase: 'line_processing',
          player: state.player,
          detectedLines: [], // Will be populated by board analysis
          currentLineIndex: 0,
          awaitingReward: false,
        },
        [{ type: 'MOVE_STACK', from: event.from, to: event.to }]
      );
    }

    case 'NO_MOVEMENT_ACTION': {
      if (state.canMove) {
        return guardFailed(state, event, 'Cannot skip movement when valid moves exist');
      }
      return ok<LineProcessingState>(
        {
          phase: 'line_processing',
          player: state.player,
          detectedLines: [],
          currentLineIndex: 0,
          awaitingReward: false,
        },
        []
      );
    }

    // Per RR-CANON-R070: Overtaking captures are also valid in the movement phase
    // The player can choose to capture instead of (or after) simple movement
    case 'CAPTURE': {
      // Captures during movement lead to either chain_capture or line_processing
      // depending on whether continuation is available
      return ok<LineProcessingState>(
        {
          phase: 'line_processing',
          player: state.player,
          detectedLines: [],
          currentLineIndex: 0,
          awaitingReward: false,
        },
        [{ type: 'EXECUTE_CAPTURE', target: event.target, capturer: state.player }]
      );
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

function handleCapture(
  state: CaptureState,
  event: TurnEvent,
  _context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'CAPTURE': {
      const validTarget = state.pendingCaptures.some(
        (c) => c.target.x === event.target.x && c.target.y === event.target.y
      );
      if (!validTarget) {
        return guardFailed(state, event, 'Invalid capture target');
      }

      // After initial capture, transition to chain_capture phase
      // The chain_capture phase handles mandatory continuations
      return ok<ChainCaptureState>(
        {
          phase: 'chain_capture',
          player: state.player,
          attackerPosition: event.target, // Landing position after capture
          capturedTargets: [event.target],
          availableContinuations: [], // Will be recomputed by host
          segmentCount: 1,
          isFirstSegment: false, // Already executed first segment
        },
        [{ type: 'EXECUTE_CAPTURE', target: event.target, capturer: state.player }]
      );
    }

    case 'END_CHAIN': {
      // Skip capture phase (RR-CANON-R070) - proceed to line processing
      return ok<LineProcessingState>(
        {
          phase: 'line_processing',
          player: state.player,
          detectedLines: [],
          currentLineIndex: 0,
          awaitingReward: false,
        },
        []
      );
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

/**
 * Handle chain_capture phase - mandatory capture continuations
 *
 * In this phase, the player MUST continue capturing until no more
 * segments are available. The only way out is CONTINUE_CHAIN (to
 * keep capturing) or END_CHAIN (when no continuations exist).
 */
function handleChainCapture(
  state: ChainCaptureState,
  event: TurnEvent,
  _context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'CONTINUE_CHAIN': {
      // Validate that the target is in available continuations
      const validTarget = state.availableContinuations.some(
        (c) => c.target.x === event.target.x && c.target.y === event.target.y
      );
      if (!validTarget && state.availableContinuations.length > 0) {
        return guardFailed(state, event, 'Invalid chain continuation target');
      }

      // Continue the chain - update state with new capture
      return ok<ChainCaptureState>(
        {
          phase: 'chain_capture',
          player: state.player,
          attackerPosition: event.target, // New landing position
          capturedTargets: [...state.capturedTargets, event.target],
          availableContinuations: [], // Will be recomputed by host
          segmentCount: state.segmentCount + 1,
          isFirstSegment: false,
        },
        [{ type: 'EXECUTE_CAPTURE', target: event.target, capturer: state.player }]
      );
    }

    case 'END_CHAIN': {
      // Only valid when no continuations are available
      // (captures are mandatory in a chain)
      if (state.availableContinuations.length > 0) {
        return guardFailed(state, event, 'Cannot end chain - mandatory captures remain');
      }

      // Chain complete - proceed to line processing
      return ok<LineProcessingState>(
        {
          phase: 'line_processing',
          player: state.player,
          detectedLines: [],
          currentLineIndex: 0,
          awaitingReward: false,
        },
        []
      );
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

function handleLineProcessing(
  state: LineProcessingState,
  event: TurnEvent,
  _context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'PROCESS_LINE': {
      if (event.lineIndex >= state.detectedLines.length) {
        return guardFailed(state, event, 'Invalid line index');
      }

      const line = state.detectedLines[event.lineIndex];
      if (line.requiresChoice) {
        // Stay in line_processing, awaiting reward choice
        return ok<LineProcessingState>(
          {
            ...state,
            currentLineIndex: event.lineIndex,
            awaitingReward: true,
          },
          [{ type: 'COLLAPSE_LINE', positions: line.positions }]
        );
      }

      // No choice needed, proceed to next line or territory
      const nextIndex = event.lineIndex + 1;
      if (nextIndex < state.detectedLines.length) {
        return ok<LineProcessingState>({ ...state, currentLineIndex: nextIndex }, [
          { type: 'COLLAPSE_LINE', positions: line.positions },
        ]);
      }

      // All lines processed, move to territory
      return ok<TerritoryProcessingState>(
        {
          phase: 'territory_processing',
          player: state.player,
          disconnectedRegions: [],
          currentRegionIndex: 0,
          eliminationsPending: [],
        },
        [{ type: 'COLLAPSE_LINE', positions: line.positions }]
      );
    }

    case 'CHOOSE_LINE_REWARD': {
      if (!state.awaitingReward) {
        return guardFailed(state, event, 'Not awaiting line reward choice');
      }

      const line = state.detectedLines[state.currentLineIndex];
      const nextIndex = state.currentLineIndex + 1;

      if (nextIndex < state.detectedLines.length) {
        return ok<LineProcessingState>(
          {
            ...state,
            currentLineIndex: nextIndex,
            awaitingReward: false,
          },
          [{ type: 'APPLY_LINE_REWARD', choice: event.choice, line }]
        );
      }

      // All lines processed
      return ok<TerritoryProcessingState>(
        {
          phase: 'territory_processing',
          player: state.player,
          disconnectedRegions: [],
          currentRegionIndex: 0,
          eliminationsPending: [],
        },
        [{ type: 'APPLY_LINE_REWARD', choice: event.choice, line }]
      );
    }

    case 'NO_LINE_ACTION': {
      if (state.detectedLines.length > 0) {
        return guardFailed(state, event, 'Cannot skip line processing when lines exist');
      }
      return ok<TerritoryProcessingState>(
        {
          phase: 'territory_processing',
          player: state.player,
          disconnectedRegions: [],
          currentRegionIndex: 0,
          eliminationsPending: [],
        },
        []
      );
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

function handleTerritoryProcessing(
  state: TerritoryProcessingState,
  event: TurnEvent,
  context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'PROCESS_REGION': {
      if (event.regionIndex >= state.disconnectedRegions.length) {
        return guardFailed(state, event, 'Invalid region index');
      }

      const region = state.disconnectedRegions[event.regionIndex];

      // If eliminations needed, queue them
      if (region.eliminationsRequired > 0) {
        return ok<TerritoryProcessingState>(
          {
            ...state,
            currentRegionIndex: event.regionIndex,
            eliminationsPending: [
              {
                position: region.positions[0], // Will be computed properly
                player: region.controllingPlayer,
                count: region.eliminationsRequired,
              },
            ],
          },
          [{ type: 'PROCESS_DISCONNECTION', region }]
        );
      }

      // No eliminations, check for more regions
      const nextIndex = event.regionIndex + 1;
      if (nextIndex < state.disconnectedRegions.length) {
        return ok<TerritoryProcessingState>({ ...state, currentRegionIndex: nextIndex }, [
          { type: 'PROCESS_DISCONNECTION', region },
        ]);
      }

      // All regions processed, check forced elimination or end turn
      return ok<TurnEndState>(
        {
          phase: 'turn_end',
          completedPlayer: state.player,
          nextPlayer: nextPlayer(state.player, context.numPlayers),
        },
        [{ type: 'PROCESS_DISCONNECTION', region }, { type: 'CHECK_VICTORY' }]
      );
    }

    case 'ELIMINATE_FROM_STACK': {
      if (state.eliminationsPending.length === 0) {
        return guardFailed(state, event, 'No eliminations pending');
      }

      const remaining = state.eliminationsPending.slice(1);
      if (remaining.length > 0) {
        return ok<TerritoryProcessingState>({ ...state, eliminationsPending: remaining }, [
          { type: 'ELIMINATE_RINGS', target: event.target, count: event.count },
        ]);
      }

      // All eliminations done, check for more regions
      const nextIndex = state.currentRegionIndex + 1;
      if (nextIndex < state.disconnectedRegions.length) {
        return ok<TerritoryProcessingState>(
          {
            ...state,
            currentRegionIndex: nextIndex,
            eliminationsPending: [],
          },
          [{ type: 'ELIMINATE_RINGS', target: event.target, count: event.count }]
        );
      }

      // All done
      return ok<TurnEndState>(
        {
          phase: 'turn_end',
          completedPlayer: state.player,
          nextPlayer: nextPlayer(state.player, context.numPlayers),
        },
        [
          { type: 'ELIMINATE_RINGS', target: event.target, count: event.count },
          { type: 'CHECK_VICTORY' },
        ]
      );
    }

    case 'NO_TERRITORY_ACTION': {
      if (state.disconnectedRegions.length > 0) {
        return guardFailed(state, event, 'Cannot skip territory processing when regions exist');
      }
      return ok<TurnEndState>(
        {
          phase: 'turn_end',
          completedPlayer: state.player,
          nextPlayer: nextPlayer(state.player, context.numPlayers),
        },
        [{ type: 'CHECK_VICTORY' }]
      );
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

function handleForcedElimination(
  state: ForcedEliminationState,
  event: TurnEvent,
  context: GameContext
): TransitionResult {
  switch (event.type) {
    case 'FORCED_ELIMINATE': {
      const newCount = state.eliminationsDone + 1;

      if (newCount >= state.ringsOverLimit) {
        // All forced eliminations done
        return ok<TurnEndState>(
          {
            phase: 'turn_end',
            completedPlayer: state.player,
            nextPlayer: nextPlayer(state.player, context.numPlayers),
          },
          [
            { type: 'FORCED_ELIMINATE', target: event.target, player: state.player },
            { type: 'CHECK_VICTORY' },
          ]
        );
      }

      // More eliminations needed
      return ok<ForcedEliminationState>({ ...state, eliminationsDone: newCount }, [
        { type: 'FORCED_ELIMINATE', target: event.target, player: state.player },
      ]);
    }

    case 'RESIGN':
      return handleResignation(state, event);

    case 'TIMEOUT':
      return handleTimeout(state, event);

    default:
      return invalidTransition(state, event);
  }
}

function handleTurnEnd(
  state: TurnEndState,
  event: TurnEvent,
  _context: GameContext
): TransitionResult {
  switch (event.type) {
    case '_ADVANCE_TURN': {
      // Transition to next player's ring placement
      return ok<RingPlacementState>(
        {
          phase: 'ring_placement',
          player: state.nextPlayer,
          ringsInHand: 0,
          canPlace: true, // Will be computed
          validPositions: [], // Will be computed
        },
        [{ type: 'ADVANCE_PLAYER', from: state.completedPlayer, to: state.nextPlayer }]
      );
    }

    default:
      return invalidTransition(state, event);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

function ok<S extends TurnState>(state: S, actions: Action[]): TransitionResult {
  return { ok: true, state, actions };
}

function invalidTransition(state: TurnState, event: TurnEvent, message?: string): TransitionResult {
  return {
    ok: false,
    error: {
      code: 'INVALID_EVENT',
      message: message || `Event '${event.type}' not valid in phase '${state.phase}'`,
      currentPhase: state.phase,
      eventType: event.type,
    },
  };
}

function guardFailed(state: TurnState, event: TurnEvent, message: string): TransitionResult {
  return {
    ok: false,
    error: {
      code: 'GUARD_FAILED',
      message,
      currentPhase: state.phase,
      eventType: event.type,
    },
  };
}

function handleResignation(
  _state: TurnState,
  _event: { type: 'RESIGN'; player: number }
): TransitionResult {
  return ok<GameOverState>(
    {
      phase: 'game_over',
      winner: null, // Would need to compute remaining players
      reason: 'resignation',
    },
    []
  );
}

function handleTimeout(
  _state: TurnState,
  _event: { type: 'TIMEOUT'; player: number }
): TransitionResult {
  return ok<GameOverState>(
    {
      phase: 'game_over',
      winner: null,
      reason: 'timeout',
    },
    []
  );
}

function nextPlayer(current: number, numPlayers: number): number {
  return (current % numPlayers) + 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE MACHINE CLASS WRAPPER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * TurnStateMachine class - wraps pure transition function with state management.
 */
export class TurnStateMachine {
  private _state: TurnState;
  private readonly context: GameContext;
  private readonly history: Array<{ state: TurnState; event: TurnEvent }> = [];

  constructor(initialState: TurnState, context: GameContext) {
    this._state = initialState;
    this.context = context;
  }

  get state(): TurnState {
    return this._state;
  }

  get phase(): string {
    return this._state.phase;
  }

  get currentPlayer(): number {
    if (this._state.phase === 'game_over') {
      return 0;
    }
    if (this._state.phase === 'turn_end') {
      return this._state.completedPlayer;
    }
    return this._state.player;
  }

  /**
   * Send an event to the state machine.
   * Returns actions to apply if successful, or throws on invalid transition.
   */
  send(event: TurnEvent): Action[] {
    const result = transition(this._state, event, this.context);

    if (result.ok === false) {
      const { code, message, currentPhase, eventType } = result.error;
      throw new Error(`[FSM] ${code}: ${message} (phase=${currentPhase}, event=${eventType})`);
    }

    this.history.push({ state: this._state, event });
    this._state = result.state;
    return result.actions;
  }

  /**
   * Check if an event is valid in the current state.
   */
  canSend(event: TurnEvent): boolean {
    const result = transition(this._state, event, this.context);
    return result.ok;
  }

  /**
   * Get the transition history for debugging.
   */
  getHistory(): ReadonlyArray<{ state: TurnState; event: TurnEvent }> {
    return this.history;
  }

  /**
   * Create initial state for a new game.
   */
  static createInitialState(startingPlayer: number): RingPlacementState {
    return {
      phase: 'ring_placement',
      player: startingPlayer,
      ringsInHand: 0,
      canPlace: true,
      validPositions: [],
    };
  }
}
