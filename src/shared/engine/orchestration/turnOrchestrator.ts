/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Turn Orchestrator
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The canonical processTurn function that orchestrates all turn phases.
 * This is the single entry point for turn processing, delegating to
 * domain aggregates for actual logic.
 */

import type { GameState, GamePhase, Move, Territory, Position } from '../../types/game';
import { positionToString } from '../../types/game';

import { hashGameState, computeProgressSnapshot } from '../core';
import {
  isANMState,
  applyForcedEliminationForPlayer,
  computeGlobalLegalActionsSummary,
} from '../globalActions';

import type {
  ProcessTurnResult,
  PendingDecision,
  TurnProcessingDelegates,
  ProcessingMetadata,
  VictoryState,
  DetectedLineInfo,
} from './types';

import { PhaseStateMachine, createTurnProcessingState } from './phaseStateMachine';

// Import from domain aggregates
import {
  validatePlacement,
  mutatePlacement,
  enumeratePlacementPositions,
  evaluateSkipPlacementEligibility,
} from '../aggregates/PlacementAggregate';

import {
  validateMovement,
  enumerateSimpleMovesForPlayer,
  applySimpleMovement,
} from '../aggregates/MovementAggregate';

import {
  validateCapture,
  enumerateAllCaptureMoves,
  applyCaptureSegment,
  getChainCaptureContinuationInfo,
} from '../aggregates/CaptureAggregate';

import {
  findAllLines,
  enumerateProcessLineMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
} from '../aggregates/LineAggregate';

import {
  getProcessableTerritoryRegions,
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
} from '../aggregates/TerritoryAggregate';

import { evaluateVictory } from '../aggregates/VictoryAggregate';

import { countRingsOnBoardForPlayer } from '../core';

// ═══════════════════════════════════════════════════════════════════════════
// S-Invariant Computation
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute the S-invariant for the current game state.
 * S = markers + collapsedSpaces + eliminatedRings (should be non-decreasing)
 */
function computeSInvariant(state: GameState): number {
  return computeProgressSnapshot(state).S;
}

/**
 * Resolve an ANM(state, currentPlayer) situation by applying forced elimination
 * and/or re-running victory evaluation until either:
 *
 * - The game becomes terminal, or
 * - The active player has at least one global legal action and ANM no longer holds.
 *
 * This encodes the RR-CANON R202/R203 semantics used by
 * INV-ACTIVE-NO-MOVES / INV-ANM-TURN-MATERIAL-SKIP: no externally visible
 * ACTIVE state may satisfy isANMState(state).
 */
function resolveANMForCurrentPlayer(state: GameState): {
  nextState: GameState;
  victoryResult?: VictoryState;
} {
  let workingState = state;

  // Conservative safety bound: one step per ring currently recorded in play
  // plus a small constant. Forced elimination is monotone in E (and S), so
  // ANM-resolution chains are finite (INV-ELIMINATION-MONOTONIC / INV-TERMINATION).
  const totalRingsInPlay =
    (workingState as GameState & { totalRingsInPlay?: number }).totalRingsInPlay ??
    workingState.players.reduce((sum, p) => sum + p.ringsInHand, 0) +
      Array.from(workingState.board.stacks.values()).reduce(
        (sum, stack) => sum + stack.stackHeight,
        0
      );
  const maxSteps = Math.max(4, totalRingsInPlay + 4);

  for (let i = 0; i < maxSteps; i++) {
    if (workingState.gameStatus !== 'active') {
      return { nextState: workingState };
    }

    if (!isANMState(workingState)) {
      return { nextState: workingState };
    }

    const player = workingState.currentPlayer;
    const forced = applyForcedEliminationForPlayer(workingState, player);

    if (!forced) {
      // No forced-elimination action available; fall back to a victory /
      // stalemate evaluation (R170–R173, R203).
      const victory = toVictoryState(workingState);
      if (victory.isGameOver) {
        const terminalState: GameState = {
          ...workingState,
          gameStatus: 'completed',
          winner: victory.winner,
        };
        return { nextState: terminalState, victoryResult: victory };
      }
      // If victory logic cannot classify the position as terminal we stop
      // resolving here and return the ANM state to the caller; this should
      // not happen for canonical states and will be surfaced by invariants.
      return { nextState: workingState };
    }

    workingState = forced.nextState;

    const victoryAfter = toVictoryState(workingState);
    if (victoryAfter.isGameOver) {
      const terminalState: GameState = {
        ...workingState,
        gameStatus: 'completed',
        winner: victoryAfter.winner,
      };
      return { nextState: terminalState, victoryResult: victoryAfter };
    }
  }

  // Safety fallback: after exhausting the bound, return the latest state.
  return { nextState: workingState };
}

 // ═══════════════════════════════════════════════════════════════════════════
 // Victory State Conversion
 // ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert evaluateVictory result to VictoryState for the orchestrator.
 */
function toVictoryState(state: GameState): VictoryState {
  const result = evaluateVictory(state);

  const scores = state.players.map((p) => ({
    player: p.playerNumber,
    eliminatedRings: p.eliminatedRings,
    territorySpaces: p.territorySpaces,
    ringsOnBoard: countRingsOnBoardForPlayer(state.board, p.playerNumber),
    ringsInHand: p.ringsInHand,
    markerCount: 0, // Will be computed below
    isEliminated: false, // Will be computed below
  }));

  // Compute marker counts
  for (const marker of state.board.markers.values()) {
    const scoreEntry = scores.find((s) => s.player === marker.player);
    if (scoreEntry) {
      scoreEntry.markerCount++;
    }
  }

  // Compute elimination status
  for (const score of scores) {
    score.isEliminated = score.ringsOnBoard === 0 && score.ringsInHand === 0;
  }

  return {
    isGameOver: result.isGameOver,
    winner: result.winner,
    reason: result.reason as VictoryState['reason'],
    scores,
    tieBreaker: undefined,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Decision Creation Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a pending decision for line processing.
 */
function createLineOrderDecision(state: GameState, lines: DetectedLineInfo[]): PendingDecision {
  const moves = enumerateProcessLineMoves(state, state.currentPlayer);

  return {
    type: 'line_order',
    player: state.currentPlayer,
    options: moves,
    context: {
      description: `Choose which line to process first (${lines.length} lines available)`,
      relevantPositions: lines.flatMap((l) => l.positions),
    },
  };
}

/**
 * Create a pending decision for territory region processing.
 */
function createRegionOrderDecision(state: GameState, regions: Territory[]): PendingDecision {
  const moves = enumerateProcessTerritoryRegionMoves(state, state.currentPlayer);

  return {
    type: 'region_order',
    player: state.currentPlayer,
    options: moves,
    context: {
      description: `Choose which region to process (${regions.length} regions available)`,
      relevantPositions: regions.flatMap((r) => r.spaces),
    },
  };
}

/**
 * Create a pending decision for forced elimination when the current
 * player is blocked with stacks but has no legal placement, movement,
 * or capture actions (RR-CANON-R072/R100/R206).
 *
 * The options are `eliminate_rings_from_stack` Moves, one per eligible
 * stack/standalone ring controlled by the current player.
 */
function createForcedEliminationDecision(state: GameState): PendingDecision | undefined {
  const player = state.currentPlayer;

  const options: Move[] = [];

  for (const stack of state.board.stacks.values() as any) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }

    const capHeight: number =
      typeof stack.capHeight === 'number' && stack.capHeight > 0 ? stack.capHeight : 0;
    const count = Math.max(1, capHeight || 0);

    const move: Move = {
      id: `forced-elim-${positionToString(stack.position)}`,
      type: 'eliminate_rings_from_stack',
      player,
      to: stack.position,
      eliminatedRings: [{ player, count }],
      eliminationFromStack: {
        position: stack.position,
        capHeight,
        totalHeight: stack.stackHeight,
      },
      // Deterministic placeholders for orchestrator-driven decisions; hosts
      // are free to override timestamp/thinkTime/moveNumber as needed.
      timestamp: new Date(0),
      thinkTime: 0,
      moveNumber: state.moveHistory.length + 1,
    } as Move;

    options.push(move);
  }

  if (options.length === 0) {
    return undefined;
  }

  return {
    type: 'elimination_target',
    player,
    options,
    context: {
      description: 'Choose which stack to eliminate from (forced elimination)',
      relevantPositions: options
        .map((m) => m.to)
        .filter((p): p is Position => !!p),
      extra: {
        reason: 'forced_elimination',
      },
    },
  };
}

/**
 * Create a pending decision for elimination target.
 */
function createEliminationDecision(state: GameState): PendingDecision {
  const moves = enumerateTerritoryEliminationMoves(state, state.currentPlayer);

  return {
    type: 'elimination_target',
    player: state.currentPlayer,
    options: moves,
    context: {
      description: 'Choose which stack to eliminate from',
    },
  };
}

/**
 * Create a pending decision for chain capture continuation.
 */
function createChainCaptureDecision(state: GameState, continuations: Move[]): PendingDecision {
  return {
    type: 'chain_capture',
    player: state.currentPlayer,
    options: continuations,
    context: {
      description: 'Continue the capture chain',
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn (Synchronous Entry Point)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Process a single move synchronously.
 *
 * This is the main entry point when decisions can be made immediately
 * (e.g., single-line/single-region cases or AI auto-selection).
 *
 * For cases requiring async decision resolution (human player choices),
 * use processTurnAsync or check the 'awaiting_decision' status.
 */
export function processTurn(state: GameState, move: Move): ProcessTurnResult {
  const sInvariantBefore = computeSInvariant(state);
  const startTime = Date.now();
  const stateMachine = new PhaseStateMachine(createTurnProcessingState(state, move));

  // Compute hash before applying move to detect if the move actually changed state
  const hashBefore = hashGameState(stateMachine.gameState);

  // Apply the move based on type
  const applyResult = applyMoveWithChainInfo(stateMachine.gameState, move);
  stateMachine.updateGameState(applyResult.nextState);

  // Compute hash after applying move
  const hashAfter = hashGameState(stateMachine.gameState);
  const moveActuallyChangedState = hashBefore !== hashAfter;

  // If this was a capture move and chain continuation is required, set the chain capture state
  if (applyResult.chainCaptureRequired && applyResult.chainCapturePosition) {
    stateMachine.setChainCapture(true, applyResult.chainCapturePosition);
  }

  // For placement moves, don't process post-move phases - the player still needs to move
  // Post-move phases (lines, territory) only happen after movement/capture
  const isPlacementMove = move.type === 'place_ring' || move.type === 'skip_placement';

  // For decision moves (process_territory_region, eliminate_rings_from_stack, etc.),
  // if the move didn't actually change state (e.g., Q23 prerequisite not met), don't
  // process post-move phases - just return the unchanged state.
  const isDecisionMove =
    move.type === 'process_territory_region' ||
    move.type === 'eliminate_rings_from_stack' ||
    move.type === 'process_line' ||
    move.type === 'choose_line_reward';

  let result: { pendingDecision?: PendingDecision; victoryResult?: VictoryState } = {};
  if (!isPlacementMove && (moveActuallyChangedState || !isDecisionMove)) {
    // Process post-move phases only for movement/capture moves, or for decision moves
    // that actually changed state.
    result = processPostMovePhases(stateMachine);
  }

  // Finalize metadata
  const sInvariantAfter = computeSInvariant(stateMachine.gameState);
  const metadata: ProcessingMetadata = {
    processedMove: move,
    phasesTraversed: stateMachine.processingState.phasesTraversed,
    linesDetected: stateMachine.processingState.pendingLines.length,
    regionsProcessed: 0,
    durationMs: Date.now() - startTime,
    sInvariantBefore,
    sInvariantAfter,
  };

  // Under trace-debug runs, log any strict S-invariant decreases that occur wholly
  // inside the orchestrator. This helps distinguish shared-engine bookkeeping
  // issues from host/adapter integration drift.
  const TRACE_DEBUG_ENABLED =
    typeof process !== 'undefined' &&
    !!(process as any).env &&
    ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

  if (
    TRACE_DEBUG_ENABLED &&
    state.gameStatus === 'active' &&
    stateMachine.gameState.gameStatus === 'active' &&
    sInvariantAfter < sInvariantBefore
  ) {
    const progressBefore = computeProgressSnapshot(state);
    const progressAfter = computeProgressSnapshot(stateMachine.gameState);

    // eslint-disable-next-line no-console
    console.log('[turnOrchestrator.processTurn] STRICT_S_INVARIANT_DECREASE', {
      moveType: move.type,
      player: move.player,
      phaseBefore: state.currentPhase,
      phaseAfter: stateMachine.gameState.currentPhase,
      statusBefore: state.gameStatus,
      statusAfter: stateMachine.gameState.gameStatus,
      progressBefore,
      progressAfter,
      stateHashBefore: hashGameState(state),
      stateHashAfter: hashGameState(stateMachine.gameState),
    });
  }

  let finalState = stateMachine.gameState;
  let finalPendingDecision = result.pendingDecision;
  let finalVictory = result.victoryResult;
  let finalStatus: ProcessTurnResult['status'] =
    finalPendingDecision ? 'awaiting_decision' : 'complete';

  // If the turn is otherwise complete but the current player is blocked
  // with stacks and only a forced-elimination action is available, surface
  // that action as an explicit decision rather than applying a hidden
  // host-level tie-breaker. This implements RR-CANON-R206 for hosts that
  // drive decisions via PendingDecision/Move.
  if (finalStatus === 'complete' && finalState.gameStatus === 'active') {
    const player = finalState.currentPlayer;
    const summary = computeGlobalLegalActionsSummary(finalState, player);

    if (
      summary.hasTurnMaterial &&
      summary.hasForcedEliminationAction &&
      !summary.hasGlobalPlacementAction &&
      !summary.hasPhaseLocalInteractiveMove
    ) {
      const forcedDecision = createForcedEliminationDecision(finalState);
      if (forcedDecision && forcedDecision.options.length > 0) {
        finalPendingDecision = forcedDecision;
        finalStatus = 'awaiting_decision';
      }
    }
  }

  // Enforce INV-ACTIVE-NO-MOVES / INV-ANM-TURN-MATERIAL-SKIP at the orchestrator
  // boundary: no externally visible ACTIVE state may satisfy ANM(state) for the
  // current player. When such a state is detected and there is no pending
  // decision, resolve it immediately via forced elimination and/or victory
  // evaluation (RR-CANON R200–R205).
  if (finalStatus === 'complete' && finalState.gameStatus === 'active' && isANMState(finalState)) {
    const resolved = resolveANMForCurrentPlayer(finalState);
    finalState = resolved.nextState;
    if (resolved.victoryResult) {
      finalVictory = resolved.victoryResult;
    }
  }

  return {
    nextState: finalState,
    status: finalPendingDecision ? 'awaiting_decision' : 'complete',
    pendingDecision: finalPendingDecision,
    victoryResult: finalVictory,
    metadata,
  };
}

/**
 * Result of applying a move, including chain capture info for capture moves.
 */
interface ApplyMoveResult {
  nextState: GameState;
  chainCaptureRequired?: boolean;
  chainCapturePosition?: Position;
}

/**
 * Apply a move to the game state based on move type.
 * For capture moves, also returns chain capture continuation info.
 */
function applyMoveWithChainInfo(state: GameState, move: Move): ApplyMoveResult {
  switch (move.type) {
    case 'place_ring': {
      const action = {
        type: 'PLACE_RING' as const,
        playerId: move.player,
        position: move.to,
        count: move.placementCount ?? 1,
      };
      const newState = mutatePlacement(state, action);
      // After placement, advance to movement phase
      return {
        nextState: {
          ...newState,
          currentPhase: 'movement' as GamePhase,
        },
      };
    }

    case 'skip_placement': {
      // Skip placement is a no-op on state, just phase transition
      return {
        nextState: {
          ...state,
          currentPhase: 'movement' as GamePhase,
        },
      };
    }

    case 'move_stack':
    case 'move_ring': {
      if (!move.from) {
        throw new Error('Move.from is required for movement moves');
      }
      const outcome = applySimpleMovement(state, {
        from: move.from,
        to: move.to,
        player: move.player,
      });
      return { nextState: outcome.nextState };
    }

    case 'overtaking_capture':
    case 'continue_capture_segment': {
      if (!move.from || !move.captureTarget) {
        throw new Error('Move.from and Move.captureTarget are required for capture moves');
      }
      const outcome = applyCaptureSegment(state, {
        from: move.from,
        target: move.captureTarget,
        landing: move.to,
        player: move.player,
      });
      // Return chain capture info so the orchestrator can set state machine flags
      if (outcome.chainContinuationRequired) {
        return {
          nextState: outcome.nextState,
          chainCaptureRequired: true,
          chainCapturePosition: move.to,
        };
      }
      return { nextState: outcome.nextState };
    }

    case 'process_line': {
      const outcome = applyProcessLineDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'choose_line_reward': {
      const outcome = applyChooseLineRewardDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'process_territory_region': {
      const outcome = applyProcessTerritoryRegionDecision(state, move);
      return { nextState: outcome.nextState };
    }

    case 'eliminate_rings_from_stack': {
      const outcome = applyEliminateRingsFromStackDecision(state, move);
      return { nextState: outcome.nextState };
    }

    default: {
      // For unsupported move types, return state unchanged
      return { nextState: state };
    }
  }
}

/**
 * Process post-move phases (lines, territory, victory check).
 */
function processPostMovePhases(stateMachine: PhaseStateMachine): {
  pendingDecision?: PendingDecision;
  victoryResult?: VictoryState;
} {
  const state = stateMachine.gameState;

  // Check for chain capture continuation first
  if (stateMachine.processingState.chainCaptureInProgress) {
    const pos = stateMachine.processingState.chainCapturePosition;
    if (pos) {
      const info = getChainCaptureContinuationInfo(state, state.currentPlayer, pos);
      if (info.mustContinue) {
        // Transition phase to chain_capture and set chainCapturePosition in state
        // so that RuleEngine/AIEngine can enumerate valid continuations.
        stateMachine.transitionTo('chain_capture');
        stateMachine.updateGameState({
          ...stateMachine.gameState,
          chainCapturePosition: pos,
        });
        return {
          pendingDecision: createChainCaptureDecision(
            stateMachine.gameState,
            info.availableContinuations
          ),
        };
      }
    }
    // Chain capture complete, clear chainCapturePosition and continue to line processing
    stateMachine.setChainCapture(false, undefined);
    stateMachine.updateGameState({
      ...stateMachine.gameState,
      chainCapturePosition: undefined,
    });
  }

  // Transition to line processing phase
  if (state.currentPhase === 'movement' || state.currentPhase === 'capture') {
    stateMachine.transitionTo('line_processing');
  }

  // Process lines
  if (stateMachine.currentPhase === 'line_processing') {
    const lines = findAllLines(state.board).filter((l) => l.player === state.currentPlayer);

    if (lines.length > 1) {
      // Multiple lines: player must choose order
      const detectedLines = lines.map((l) => ({
        positions: l.positions,
        player: l.player,
        length: l.length,
        direction: l.direction,
        collapseOptions: [],
      }));
      stateMachine.setPendingLines(detectedLines);
      return {
        pendingDecision: createLineOrderDecision(state, detectedLines),
      };
    } else if (lines.length === 1) {
      // Single line: process automatically
      const line = lines[0];
      const move: Move = {
        id: `auto-process-line`,
        type: 'process_line',
        player: state.currentPlayer,
        to: line.positions[0],
        formedLines: [line],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: state.moveHistory.length + 1,
      };
      const outcome = applyProcessLineDecision(state, move);
      stateMachine.updateGameState(outcome.nextState);

      // Check if line reward decision is needed
      if (outcome.pendingLineRewardElimination) {
        // Need elimination decision
        return {
          pendingDecision: createEliminationDecision(stateMachine.gameState),
        };
      }
    }

    // No lines or lines processed, continue to territory
    stateMachine.transitionTo('territory_processing');
  }

  // Process territory
  if (stateMachine.currentPhase === 'territory_processing') {
    const regions = getProcessableTerritoryRegions(state.board, {
      player: state.currentPlayer,
    });

    if (regions.length > 1) {
      // Multiple regions: player must choose order
      return {
        pendingDecision: createRegionOrderDecision(state, regions),
      };
    } else if (regions.length === 1) {
      // Single region: process automatically
      const region = regions[0];
      const move: Move = {
        id: `auto-process-region`,
        type: 'process_territory_region',
        player: state.currentPlayer,
        to: region.spaces[0],
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: state.moveHistory.length + 1,
      };
      const outcome = applyProcessTerritoryRegionDecision(state, move);
      stateMachine.updateGameState(outcome.nextState);

      // Check if elimination decision is needed
      if (outcome.pendingSelfElimination) {
        return {
          pendingDecision: createEliminationDecision(stateMachine.gameState),
        };
      }
    }
  }

  // Check victory
  const victoryResult = toVictoryState(stateMachine.gameState);
  if (victoryResult.isGameOver) {
    stateMachine.updateGameState({
      ...stateMachine.gameState,
      gameStatus: 'completed',
    });
    return { victoryResult };
  }

  // All phases complete - advance to next player's turn
  const currentState = stateMachine.gameState;
  const players = currentState.players;
  const currentPlayerIndex = players.findIndex(
    (p) => p.playerNumber === currentState.currentPlayer
  );
  const nextPlayerIndex = (currentPlayerIndex + 1) % players.length;
  const nextPlayer = players[nextPlayerIndex].playerNumber;

  stateMachine.updateGameState({
    ...currentState,
    currentPlayer: nextPlayer,
    currentPhase: 'ring_placement' as GamePhase,
  });

  return {};
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Turn Async (For Human Decisions)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Process a turn with async decision resolution.
 *
 * When a decision is required, the delegates.resolveDecision function
 * is called to get the player's choice.
 */
export async function processTurnAsync(
  state: GameState,
  move: Move,
  delegates: TurnProcessingDelegates
): Promise<ProcessTurnResult> {
  let result = processTurn(state, move);

  // Process any pending decisions
  while (result.status === 'awaiting_decision' && result.pendingDecision) {
    const decision = result.pendingDecision;

    // Chain capture decisions should NOT be auto-resolved via delegates.
    // They must be handled via explicit continue_capture_segment moves
    // submitted through the normal makeMove() flow. Return with the pending
    // decision so GameEngine can set up chainCaptureState and allow the
    // caller to select a continuation move.
    if (decision.type === 'chain_capture') {
      return result;
    }

    // Emit decision event if handler provided
    delegates.onProcessingEvent?.({
      type: 'decision_required',
      timestamp: new Date(),
      payload: { decision },
    });

    // Resolve the decision
    const chosenMove = await delegates.resolveDecision(decision);

    // Emit decision resolved event
    delegates.onProcessingEvent?.({
      type: 'decision_resolved',
      timestamp: new Date(),
      payload: { decision, chosenMove },
    });

    // Continue processing with the chosen move
    result = processTurn(result.nextState, chosenMove);
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a move is valid for the current game state.
 */
export function validateMove(state: GameState, move: Move): { valid: boolean; reason?: string } {
  switch (move.type) {
    case 'place_ring': {
      const action = {
        type: 'PLACE_RING' as const,
        playerId: move.player,
        position: move.to,
        count: move.placementCount ?? 1,
      };
      return validatePlacement(state, action);
    }

    case 'skip_placement': {
      const result = evaluateSkipPlacementEligibility(state, move.player);
      if (!result.eligible && result.reason) {
        return { valid: false, reason: result.reason };
      }
      return { valid: result.eligible };
    }

    case 'move_stack':
    case 'move_ring': {
      if (!move.from) {
        return { valid: false, reason: 'Move.from is required' };
      }
      const action = {
        type: 'MOVE_STACK' as const,
        playerId: move.player,
        from: move.from,
        to: move.to,
      };
      return validateMovement(state, action);
    }

    case 'overtaking_capture':
    case 'continue_capture_segment': {
      if (!move.from || !move.captureTarget) {
        return { valid: false, reason: 'Move.from and Move.captureTarget are required' };
      }
      const action = {
        type: 'OVERTAKING_CAPTURE' as const,
        playerId: move.player,
        from: move.from,
        captureTarget: move.captureTarget,
        to: move.to,
      };
      return validateCapture(state, action);
    }

    default:
      return { valid: true };
  }
}

/**
 * Get all valid moves for the current player and phase.
 */
export function getValidMoves(state: GameState): Move[] {
  const player = state.currentPlayer;
  const phase = state.currentPhase;
  const moveNumber = state.moveHistory.length + 1;

  switch (phase) {
    case 'ring_placement': {
      const positions = enumeratePlacementPositions(state, player);
      const moves: Move[] = positions.map((pos) => ({
        id: `place-${positionToString(pos)}-${moveNumber}`,
        type: 'place_ring',
        player,
        to: pos,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      }));

      // Add skip placement if eligible
      const skipResult = evaluateSkipPlacementEligibility(state, player);
      if (skipResult.eligible) {
        moves.push({
          id: `skip-placement-${moveNumber}`,
          type: 'skip_placement',
          player,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber,
        });
      }

      return moves;
    }

    case 'movement': {
      const movements = enumerateSimpleMovesForPlayer(state, player);
      const captures = enumerateAllCaptureMoves(state, player);
      return [...movements, ...captures];
    }

    case 'chain_capture': {
      // Get capture continuations from current position
      // Note: This requires knowing the chain position, which needs context
      const captures = enumerateAllCaptureMoves(state, player);
      return captures;
    }

    case 'line_processing': {
      return enumerateProcessLineMoves(state, player);
    }

    case 'territory_processing': {
      const regionMoves = enumerateProcessTerritoryRegionMoves(state, player);
      const elimMoves = enumerateTerritoryEliminationMoves(state, player);
      return [...regionMoves, ...elimMoves];
    }

    default:
      return [];
  }
}

/**
 * Check if the current player has any valid moves.
 */
export function hasValidMoves(state: GameState): boolean {
  return getValidMoves(state).length > 0;
}
