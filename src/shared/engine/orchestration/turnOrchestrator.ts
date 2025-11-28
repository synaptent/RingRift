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

import { hashGameState } from '../core';

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
  const markers = state.board.markers.size;
  const collapsed = state.board.collapsedSpaces.size;
  const eliminated = state.players.reduce((sum, p) => sum + p.eliminatedRings, 0);
  return markers + collapsed + eliminated;
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
  const metadata: ProcessingMetadata = {
    processedMove: move,
    phasesTraversed: stateMachine.processingState.phasesTraversed,
    linesDetected: stateMachine.processingState.pendingLines.length,
    regionsProcessed: 0,
    durationMs: Date.now() - startTime,
    sInvariantBefore,
    sInvariantAfter: computeSInvariant(stateMachine.gameState),
  };

  return {
    nextState: stateMachine.gameState,
    status: result.pendingDecision ? 'awaiting_decision' : 'complete',
    pendingDecision: result.pendingDecision,
    victoryResult: result.victoryResult,
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
        return {
          pendingDecision: createChainCaptureDecision(state, info.availableContinuations),
        };
      }
    }
    // Chain capture complete, continue to line processing
    stateMachine.setChainCapture(false, undefined);
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
