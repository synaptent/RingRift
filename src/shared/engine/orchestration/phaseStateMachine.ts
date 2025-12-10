/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Phase State Machine
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Manages phase transitions during turn processing.
 * Built on top of the existing turnLogic module.
 *
 * @deprecated This module is superseded by the FSM-based orchestration in
 * `../fsm/TurnStateMachine.ts`. The FSM is now the canonical orchestrator
 * for all phase transitions and move validation. This module remains for
 * backwards compatibility but should not be used for new code.
 *
 * Migration path:
 * - Use `validateMoveWithFSM` from `../fsm/FSMAdapter` for move validation
 * - Use `computeFSMOrchestration` from `../fsm/FSMAdapter` for phase transitions
 * - Use `determineNextPhaseFromFSM` from `../fsm/FSMAdapter` instead of `determineNextPhase`
 */

import type { GameState, GamePhase, Move, Position } from '../../types/game';
import type { TurnProcessingState, PerTurnFlags } from './types';
import type { PerTurnState, TurnLogicDelegates } from '../turnLogic';
import { advanceTurnAndPhase } from '../turnLogic';

// ═══════════════════════════════════════════════════════════════════════════
// Phase Transition Logic
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Determine the next phase after processing a move.
 *
 * This function encapsulates the phase transition rules from the canonical
 * RingRift rules specification.
 *
 * @deprecated Use `determineNextPhaseFromFSM` from `../fsm/FSMAdapter` instead.
 * This function duplicates logic that is canonically expressed in TurnStateMachine.
 * The FSM-based version provides type-safe transitions and better validation.
 * Migration path: Import `determineNextPhaseFromFSM` from `../fsm` and adapt
 * the context parameter to `PhaseTransitionContext`.
 *
 * @param currentPhase Current phase
 * @param moveType Type of move that was processed
 * @param context Additional context for the transition
 * @returns The next phase
 */
export function determineNextPhase(
  currentPhase: GamePhase,
  _moveType: Move['type'],
  context: {
    hasMoreLinesToProcess: boolean;
    hasMoreRegionsToProcess: boolean;
    chainCapturesAvailable: boolean;
    hasAnyMovement: boolean;
    hasAnyCapture: boolean;
  }
): GamePhase {
  switch (currentPhase) {
    case 'ring_placement':
      // After placement, move to movement phase if player has moves/captures
      if (context.hasAnyMovement || context.hasAnyCapture) {
        return 'movement';
      }
      // Otherwise skip to line processing
      return 'line_processing';

    case 'movement':
      // After movement/capture, check for chain captures
      if (context.chainCapturesAvailable) {
        return 'chain_capture';
      }
      // Otherwise proceed to line processing
      return 'line_processing';

    case 'capture':
      // Same as movement
      if (context.chainCapturesAvailable) {
        return 'chain_capture';
      }
      return 'line_processing';

    case 'chain_capture':
      // After chain capture segment, check for more chains
      if (context.chainCapturesAvailable) {
        return 'chain_capture'; // Stay in chain capture
      }
      return 'line_processing';

    case 'line_processing':
      // After processing lines, move to territory
      if (context.hasMoreLinesToProcess) {
        return 'line_processing'; // Stay and process more
      }
      return 'territory_processing';

    case 'territory_processing':
      // After territory, turn ends - this triggers advanceTurnAndPhase
      // unless forced_elimination is required (handled by orchestrator)
      return 'territory_processing'; // Will be handled by turn advance

    case 'forced_elimination':
      // After forced elimination, turn ends
      return 'forced_elimination'; // Will be handled by turn advance

    default:
      return currentPhase;
  }
}

/**
 * Create delegates for the turnLogic module from the current processing state.
 *
 * These delegates are used by advanceTurnAndPhase to:
 * - Query player stacks
 * - Check for available placements/movements/captures
 * - Apply forced elimination
 * - Get next player number
 *
 * @deprecated Part of the deprecated phaseStateMachine module.
 * The FSM-based orchestration in TurnStateMachine.ts is now canonical.
 */
export function createTurnLogicDelegates(
  _processingState: TurnProcessingState,
  callbacks: {
    getPlayerStacks: (
      state: GameState,
      player: number
    ) => Array<{ position: Position; stackHeight: number }>;
    hasAnyPlacement: (state: GameState, player: number) => boolean;
    hasAnyMovement: (state: GameState, player: number, turn: PerTurnState) => boolean;
    hasAnyCapture: (state: GameState, player: number, turn: PerTurnState) => boolean;
    applyForcedElimination: (state: GameState, player: number) => GameState;
  }
): TurnLogicDelegates {
  return {
    getPlayerStacks: callbacks.getPlayerStacks,
    hasAnyPlacement: callbacks.hasAnyPlacement,
    hasAnyMovement: callbacks.hasAnyMovement,
    hasAnyCapture: callbacks.hasAnyCapture,
    applyForcedElimination: callbacks.applyForcedElimination,

    getNextPlayerNumber: (state: GameState, current: number): number => {
      const players = state.players;
      if (!players || players.length === 0) {
        return current;
      }

      const currentIdx = players.findIndex((p) => p.playerNumber === current);
      if (currentIdx === -1) {
        return players[0].playerNumber;
      }

      const nextIdx = (currentIdx + 1) % players.length;
      return players[nextIdx].playerNumber;
    },

    playerHasAnyRings: (state: GameState, player: number): boolean => {
      // Check if player has rings anywhere (hand + board including buried)
      const playerState = state.players.find((p) => p.playerNumber === player);
      if (!playerState) return false;
      if (playerState.ringsInHand > 0) return true;
      // Check stacks for player's rings
      for (const [, stack] of state.board.stacks) {
        if (stack.rings.includes(player)) return true;
      }
      return false;
    },
  };
}

/**
 * Convert per-turn flags to the PerTurnState format expected by turnLogic.
 *
 * @deprecated Part of the deprecated phaseStateMachine module.
 */
export function toPerTurnState(flags: PerTurnFlags): PerTurnState {
  return {
    hasPlacedThisTurn: flags.hasPlacedThisTurn,
    mustMoveFromStackKey: flags.mustMoveFromStackKey,
  };
}

/**
 * Update per-turn flags from a PerTurnState.
 *
 * @deprecated Part of the deprecated phaseStateMachine module.
 */
export function updateFlagsFromPerTurnState(
  flags: PerTurnFlags,
  state: PerTurnState
): PerTurnFlags {
  return {
    ...flags,
    hasPlacedThisTurn: state.hasPlacedThisTurn,
    mustMoveFromStackKey: state.mustMoveFromStackKey,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase Handlers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a phase requires player decisions before proceeding.
 *
 * @deprecated Part of the deprecated phaseStateMachine module.
 * Use FSM's `decisionSurface.pendingDecisionType` to determine if decisions are needed.
 */
export function phaseRequiresDecision(phase: GamePhase, context: PhaseContext): boolean {
  switch (phase) {
    case 'line_processing':
      // Multiple lines require ordering decision
      return context.pendingLineCount > 1;

    case 'territory_processing':
      // Multiple regions require ordering decision
      return context.pendingRegionCount > 1;

    case 'chain_capture':
      // Chain capture may require direction choice
      return context.chainCaptureOptionsCount > 1;

    case 'forced_elimination':
      // Forced elimination may require target choice if multiple stacks
      return context.forcedEliminationOptionsCount > 1;

    default:
      return false;
  }
}

/**
 * Context for phase decision checks.
 */
export interface PhaseContext {
  pendingLineCount: number;
  pendingRegionCount: number;
  chainCaptureOptionsCount: number;
  /** Number of stacks the player controls (for forced elimination target choice) */
  forcedEliminationOptionsCount: number;
}

/**
 * Determine if the current phase should auto-advance.
 *
 * Some phases auto-advance when there's only one option or no options.
 *
 * @deprecated Part of the deprecated phaseStateMachine module.
 * Use FSM's `decisionSurface.pendingDecisionType` to determine if auto-advance is needed.
 */
export function shouldAutoAdvancePhase(phase: GamePhase, context: PhaseContext): boolean {
  switch (phase) {
    case 'line_processing':
      // Auto-advance if 0 or 1 line
      return context.pendingLineCount <= 1;

    case 'territory_processing':
      // Auto-advance if 0 or 1 region
      return context.pendingRegionCount <= 1;

    case 'chain_capture':
      // Auto-advance if only 1 capture option (mandatory)
      return context.chainCaptureOptionsCount === 1;

    case 'forced_elimination':
      // Auto-advance if only 1 stack to eliminate from
      return context.forcedEliminationOptionsCount === 1;

    default:
      return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// State Machine
// ═══════════════════════════════════════════════════════════════════════════

/**
 * The phase state machine that coordinates phase transitions.
 *
 * @deprecated Use `TurnStateMachine` from `../fsm/TurnStateMachine` instead.
 * The FSM provides type-safe phase transitions with explicit guards and actions.
 */
export class PhaseStateMachine {
  private state: TurnProcessingState;

  constructor(initialState: TurnProcessingState) {
    this.state = initialState;
  }

  /**
   * Get current phase from game state.
   */
  get currentPhase(): GamePhase {
    return this.state.gameState.currentPhase;
  }

  /**
   * Get current game state.
   */
  get gameState(): GameState {
    return this.state.gameState;
  }

  /**
   * Get current processing state.
   */
  get processingState(): TurnProcessingState {
    return this.state;
  }

  /**
   * Update the game state.
   */
  updateGameState(newState: GameState): void {
    this.state.gameState = newState;
    this.state.phasesTraversed.push(newState.currentPhase);
  }

  /**
   * Update per-turn flags.
   */
  updateFlags(flags: Partial<PerTurnFlags>): void {
    this.state.perTurnFlags = { ...this.state.perTurnFlags, ...flags };
  }

  /**
   * Set chain capture state.
   */
  setChainCapture(inProgress: boolean, position?: Position): void {
    this.state.chainCaptureInProgress = inProgress;
    this.state.chainCapturePosition = position;
  }

  /**
   * Add a processing event.
   */
  addEvent(
    type: TurnProcessingState['events'][number]['type'],
    payload: Record<string, unknown>
  ): void {
    this.state.events.push({
      type,
      timestamp: new Date(),
      payload,
    });
  }

  /**
   * Transition to a specific phase.
   */
  transitionTo(phase: GamePhase): void {
    this.state.gameState = {
      ...this.state.gameState,
      currentPhase: phase,
    };
    this.state.phasesTraversed.push(phase);
  }

  /**
   * Apply turn advancement using the shared turnLogic.
   */
  advanceTurn(delegates: TurnLogicDelegates): void {
    const perTurn = toPerTurnState(this.state.perTurnFlags);
    const result = advanceTurnAndPhase(this.state.gameState, perTurn, delegates);

    this.state.gameState = result.nextState;
    this.state.perTurnFlags = updateFlagsFromPerTurnState(this.state.perTurnFlags, result.nextTurn);
    this.state.phasesTraversed.push(result.nextState.currentPhase);
  }

  /**
   * Check if the game has ended.
   */
  isGameOver(): boolean {
    return this.state.gameState.gameStatus !== 'active';
  }

  /**
   * Set pending lines for processing.
   */
  setPendingLines(lines: TurnProcessingState['pendingLines']): void {
    this.state.pendingLines = lines;
  }

  /**
   * Set pending regions for processing.
   */
  setPendingRegions(regions: TurnProcessingState['pendingRegions']): void {
    this.state.pendingRegions = regions;
  }

  /**
   * Get the number of pending lines.
   */
  get pendingLineCount(): number {
    return this.state.pendingLines.length;
  }

  /**
   * Get the number of pending regions.
   */
  get pendingRegionCount(): number {
    return this.state.pendingRegions.length;
  }

  /**
   * Pop the first pending line.
   */
  popPendingLine(): TurnProcessingState['pendingLines'][number] | undefined {
    return this.state.pendingLines.shift();
  }

  /**
   * Pop the first pending region.
   */
  popPendingRegion(): TurnProcessingState['pendingRegions'][number] | undefined {
    return this.state.pendingRegions.shift();
  }
}

/**
 * Create a fresh turn processing state.
 *
 * @deprecated Part of the deprecated phaseStateMachine module.
 * The FSM-based orchestration handles processing state internally.
 */
export function createTurnProcessingState(gameState: GameState, move: Move): TurnProcessingState {
  return {
    gameState,
    originalMove: move,
    perTurnFlags: {
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
      eliminationRewardPending: false,
      eliminationRewardCount: 0,
      hadActionThisTurn: false,
    },
    pendingLines: [],
    pendingRegions: [],
    chainCaptureInProgress: false,
    chainCapturePosition: undefined,
    events: [],
    phasesTraversed: [gameState.currentPhase],
    startTime: Date.now(),
  };
}
