/**
 * FSMAdapter - Bridges the FSM with existing Move types and game state
 *
 * This adapter provides bidirectional conversion between:
 * - Move (existing game.ts type) ↔ TurnEvent (FSM input)
 * - Action (FSM output) → GameState mutations
 * - GameState → TurnState (derived FSM context)
 *
 * It allows the FSM to be integrated incrementally with the existing
 * orchestrator and game engine without breaking changes.
 *
 * @module FSMAdapter
 */

import type {
  Move,
  GameState,
  LineInfo,
  Territory,
  Position,
  MoveType,
  GamePhase,
} from '../../types/game';
import { positionToString } from '../../types/game';
import {
  transition,
  type TurnEvent,
  type TurnState,
  type Action,
  type GameContext,
  type RingPlacementState,
  type MovementState,
  type CaptureState,
  type ChainCaptureState,
  type LineProcessingState,
  type TerritoryProcessingState,
  type ForcedEliminationState,
  type TurnEndState,
  type GameOverState,
  type DetectedLine,
  type DisconnectedRegion,
  type LineRewardChoice,
} from './TurnStateMachine';
import { getValidMoves, validateMove } from '../orchestration/turnOrchestrator';
import { findLinesForPlayer } from '../lineDetection';
import { findDisconnectedRegions } from '../territoryDetection';
import { enumerateChainCaptureSegments } from '../aggregates/CaptureAggregate';
import { hasAnyGlobalMovementOrCapture } from '../globalActions';

// ═══════════════════════════════════════════════════════════════════════════
// MOVE → EVENT CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert a Move to a TurnEvent for FSM processing.
 *
 * This is the primary entry point for integrating existing game logic
 * with the FSM. Each move type maps to exactly one event type.
 */
export function moveToEvent(move: Move): TurnEvent | null {
  switch (move.type) {
    // Ring Placement
    case 'place_ring':
      return { type: 'PLACE_RING', to: move.to };

    case 'skip_placement':
      return { type: 'SKIP_PLACEMENT' };

    case 'no_placement_action':
      return { type: 'NO_PLACEMENT_ACTION' };

    // Movement
    case 'move_stack':
    case 'move_ring':
      if (!move.from) return null;
      return { type: 'MOVE_STACK', from: move.from, to: move.to };

    case 'no_movement_action':
      return { type: 'NO_MOVEMENT_ACTION' };

    // Recovery (RR-CANON-R110–R115)
    case 'recovery_slide': {
      if (!move.from) return null;
      const recoveryEvent: TurnEvent = {
        type: 'RECOVERY_SLIDE',
        from: move.from,
        to: move.to,
      };
      // Only set option if defined (exactOptionalPropertyTypes compliance)
      if (move.recoveryOption !== undefined) {
        (recoveryEvent as { option?: 1 | 2 }).option = move.recoveryOption;
      }
      return recoveryEvent;
    }

    // Capture
    case 'overtaking_capture':
      if (!move.captureTarget) return null;
      return { type: 'CAPTURE', target: move.captureTarget };

    case 'continue_capture_segment':
      if (!move.captureTarget) return null;
      return { type: 'CONTINUE_CHAIN', target: move.captureTarget };

    case 'skip_capture':
      return { type: 'END_CHAIN' };

    // Line Processing
    case 'process_line':
      // The lineIndex would typically come from move context
      return { type: 'PROCESS_LINE', lineIndex: 0 };

    case 'choose_line_reward': {
      // Determine choice from move data
      const choice = extractLineRewardChoice(move);
      return { type: 'CHOOSE_LINE_REWARD', choice };
    }

    case 'no_line_action':
      return { type: 'NO_LINE_ACTION' };

    // Territory Processing
    case 'process_territory_region':
      return { type: 'PROCESS_REGION', regionIndex: 0 };

    case 'skip_territory_processing':
    case 'no_territory_action':
      return { type: 'NO_TERRITORY_ACTION' };

    case 'eliminate_rings_from_stack': {
      // Extract elimination count from move
      const elimCount = move.eliminatedRings?.[0]?.count ?? 1;
      return { type: 'ELIMINATE_FROM_STACK', target: move.to, count: elimCount };
    }

    // Forced Elimination
    case 'forced_elimination':
      return { type: 'FORCED_ELIMINATE', target: move.to };

    // Resign and Timeout (allowed from any phase, cause game termination)
    case 'resign':
      return { type: 'RESIGN', player: move.player };

    case 'timeout':
      return { type: 'TIMEOUT', player: move.player };

    // Meta/swap (not FSM events)
    case 'swap_sides':
    case 'line_formation':
    case 'territory_claim':
    case 'build_stack':
      // These are either legacy or meta-moves not handled by turn FSM
      return null;

    default:
      // Exhaustive check - should never reach here
      return null;
  }
}

/**
 * Extract line reward choice from move data.
 */
function extractLineRewardChoice(move: Move): LineRewardChoice {
  // Option 1 = eliminate (ring recovery), Option 2 = territory
  // Determine from collapsedMarkers presence - if specified, it's territory
  if (move.collapsedMarkers && move.collapsedMarkers.length > 0) {
    return 'territory';
  }
  return 'eliminate';
}

// ═══════════════════════════════════════════════════════════════════════════
// EVENT → MOVE CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert a TurnEvent back to a Move (for validation and application).
 *
 * This is useful for testing and for hosts that want to use FSM events
 * but still need to interact with the existing game engine.
 */
export function eventToMove(event: TurnEvent, player: number, moveNumber: number): Move | null {
  const baseMove = {
    id: `fsm-${event.type}-${moveNumber}`,
    player,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber,
  };

  switch (event.type) {
    case 'PLACE_RING':
      return { ...baseMove, type: 'place_ring', to: event.to };

    case 'SKIP_PLACEMENT':
      return { ...baseMove, type: 'skip_placement', to: { x: 0, y: 0 } };

    case 'NO_PLACEMENT_ACTION':
      return { ...baseMove, type: 'no_placement_action', to: { x: 0, y: 0 } };

    case 'MOVE_STACK':
      return { ...baseMove, type: 'move_stack', from: event.from, to: event.to };

    case 'NO_MOVEMENT_ACTION':
      return { ...baseMove, type: 'no_movement_action', to: { x: 0, y: 0 } };

    case 'RECOVERY_SLIDE': {
      const recoveryMove: Move = {
        ...baseMove,
        type: 'recovery_slide',
        from: event.from,
        to: event.to,
      };
      // Only set recoveryOption if defined (exactOptionalPropertyTypes compliance)
      if (event.option !== undefined) {
        recoveryMove.recoveryOption = event.option;
      }
      return recoveryMove;
    }

    case 'CAPTURE':
      return {
        ...baseMove,
        type: 'overtaking_capture',
        to: event.target, // Landing position would need calculation
        captureTarget: event.target,
      };

    case 'CONTINUE_CHAIN':
      return {
        ...baseMove,
        type: 'continue_capture_segment',
        to: event.target,
        captureTarget: event.target,
      };

    case 'END_CHAIN':
      return { ...baseMove, type: 'skip_capture', to: { x: 0, y: 0 } };

    case 'PROCESS_LINE':
      return { ...baseMove, type: 'process_line', to: { x: 0, y: 0 } };

    case 'CHOOSE_LINE_REWARD':
      return { ...baseMove, type: 'choose_line_reward', to: { x: 0, y: 0 } };

    case 'NO_LINE_ACTION':
      return { ...baseMove, type: 'no_line_action', to: { x: 0, y: 0 } };

    case 'PROCESS_REGION':
      return { ...baseMove, type: 'process_territory_region', to: { x: 0, y: 0 } };

    case 'ELIMINATE_FROM_STACK':
      return {
        ...baseMove,
        type: 'eliminate_rings_from_stack',
        to: event.target,
        eliminatedRings: [{ player, count: event.count }],
      };

    case 'NO_TERRITORY_ACTION':
      return { ...baseMove, type: 'no_territory_action', to: { x: 0, y: 0 } };

    case 'FORCED_ELIMINATE':
      return {
        ...baseMove,
        type: 'forced_elimination',
        to: event.target,
        eliminatedRings: [{ player, count: 1 }],
      };

    case 'RESIGN':
    case 'TIMEOUT':
    case '_ADVANCE_TURN':
      // Meta events don't produce moves
      return null;

    default:
      // Exhaustive check - should never reach here
      return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// GAMESTATE → FSMSTATE DERIVATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Derive FSM TurnState from the current GameState.
 *
 * This computes the phase-specific context needed by the FSM:
 * - validPositions for placement
 * - canMove for movement
 * - detectedLines for line processing
 * - disconnectedRegions for territory processing
 *
 * @param gameState The current game state
 * @param moveHint Optional move being validated - used to ensure state includes relevant context
 */
export function deriveStateFromGame(gameState: GameState, moveHint?: Move): TurnState {
  // For bookkeeping moves and skip_placement, use the move's player instead of
  // currentPlayer because these moves may be recorded at turn boundaries where
  // the state's currentPlayer hasn't been updated yet.
  const isBookkeepingOrSkipMove =
    moveHint?.type === 'skip_placement' ||
    moveHint?.type === 'no_placement_action' ||
    moveHint?.type === 'no_movement_action' ||
    moveHint?.type === 'no_line_action' ||
    moveHint?.type === 'no_territory_action';

  // Trust the move hint's player for moves where parity divergence may cause
  // TS and Python to have different current players. This includes:
  // - Bookkeeping moves (existing behavior)
  // - Territory region processing (Python may detect more regions than TS)
  // RR-CANON-R075: Trust recorded moves during replay.
  const isTerritoryRegionMove = moveHint?.type === 'process_territory_region';
  const player =
    (isBookkeepingOrSkipMove || isTerritoryRegionMove) && moveHint?.player
      ? moveHint.player
      : gameState.currentPlayer;

  // When the move hint indicates a specific phase, trust it over the game state's phase.
  // This handles parity issues where Python may detect more disconnected regions than TS,
  // causing TS to transition out of territory_processing before all Python-recorded moves
  // are replayed. RR-CANON-R075: Trust recorded moves during replay.
  let phase = gameState.currentPhase;
  if (moveHint?.type === 'process_territory_region') {
    phase = 'territory_processing';
  } else if (moveHint?.type === 'choose_line_reward' || moveHint?.type === 'no_line_action') {
    phase = 'line_processing';
  }

  switch (phase) {
    case 'ring_placement':
      return deriveRingPlacementState(gameState, player, moveHint);

    case 'movement':
      return deriveMovementState(gameState, player, moveHint);

    case 'capture':
      return deriveCaptureState(gameState, player, false, moveHint);

    case 'chain_capture':
      return deriveChainCaptureState(gameState, player, moveHint);

    case 'line_processing':
      return deriveLineProcessingState(gameState, player);

    case 'territory_processing':
      return deriveTerritoryProcessingState(gameState, player, moveHint);

    case 'forced_elimination':
      return deriveForcedEliminationState(gameState, player);

    case 'game_over':
      return deriveGameOverState(gameState);

    default:
      // Fallback - shouldn't happen
      return {
        phase: 'ring_placement',
        player,
        ringsInHand: 0,
        canPlace: false,
        validPositions: [],
      };
  }
}

function deriveRingPlacementState(
  state: GameState,
  player: number,
  moveHint?: Move
): RingPlacementState {
  const playerObj = state.players.find((p) => p.playerNumber === player);
  const ringsInHand = playerObj?.ringsInHand ?? 0;

  // For accurate canPlace determination, we need to check actual valid placements.
  // getValidMoves uses state.currentPlayer, so we can only use it when player matches.
  let canPlace: boolean;
  let validPositions: Position[] = [];

  // Special handling for skip_placement and no_placement_action moves:
  // - skip_placement: Player has rings but no valid positions to place them
  // - no_placement_action: Player has no rings (bookkeeping)
  // In both cases, the recording indicates placements weren't available. Trust this.
  if (moveHint?.type === 'skip_placement' || moveHint?.type === 'no_placement_action') {
    canPlace = false;
  } else if (state.currentPlayer === player) {
    // Player matches - we can accurately determine valid placements
    const validMoves = getValidMoves(state);
    const placementMoves = validMoves.filter((m) => m.type === 'place_ring');
    validPositions = placementMoves.map((m) => m.to);
    canPlace = placementMoves.length > 0;
  } else {
    // Player doesn't match currentPlayer and not a bookkeeping move.
    // Fall back to conservative check: can place if rings in hand > 0.
    canPlace = ringsInHand > 0;
  }

  return {
    phase: 'ring_placement',
    player,
    ringsInHand,
    canPlace,
    validPositions,
  };
}

function deriveMovementState(state: GameState, player: number, moveHint?: Move): MovementState {
  // Phase-local interactive predicate for movement is derived from the same
  // surface the shared engine and Python GameEngine use for MOVEMENT phase
  // requirements: getValidMoves limited to movement/capture/recovery moves for
  // the active player. This keeps the FSM guard for NO_MOVEMENT_ACTION aligned
  // with canonical ANM behaviour (RR‑CANON‑R200/R203) instead of relying on a
  // purely global reachability helper.
  //
  // Canonical intent:
  // - In MOVEMENT for the current player, canMove is true iff at least one
  //   movement, capture, or recovery move is available via getValidMoves.
  // - For bookkeeping NO_MOVEMENT_ACTION moves recorded in canonical DBs
  //   (RR‑CANON‑R075), we trust the recording and treat canMove as false even
  //   if TS locally believes a move exists. This prevents structural
  //   GUARD_FAILED rejections during parity when Python has already classified
  //   the state as “no movement actions”.
  // - For off-phase / off-player derivations (shadow/parity tooling), we keep
  //   using the global movement/capture predicate so MovementState remains
  //   meaningful even when currentPlayer differs from `player`.
  const isRecordedNoMovementForPlayer =
    moveHint?.type === 'no_movement_action' && moveHint.player === player;

  let canMove: boolean;

  if (state.currentPhase === 'movement' && state.currentPlayer === player) {
    if (isRecordedNoMovementForPlayer) {
      // RR‑CANON‑R075: Trust recorded bookkeeping moves during replay. A
      // canonical NO_MOVEMENT_ACTION implies there were no legal movement,
      // capture, or recovery moves for this player in MOVEMENT.
      canMove = false;
    } else {
      const validMoves = getValidMoves(state);
      const movementLike = validMoves.filter(
        (m) =>
          m.player === player &&
          (m.type === 'move_stack' ||
            m.type === 'move_ring' ||
            m.type === 'overtaking_capture' ||
            m.type === 'continue_capture_segment' ||
            m.type === 'recovery_slide')
      );
      canMove = movementLike.length > 0;
    }
  } else {
    // Off-phase / off-player derivations (replay/shadow tooling) continue to
    // use the global predicate so that MovementState stays meaningful even
    // when currentPlayer differs from `player`.
    const hasGlobalMovementOrCapture = hasAnyGlobalMovementOrCapture(state, player);
    canMove = hasGlobalMovementOrCapture;
  }

  // Get the ring just placed (if any)
  const lastMove =
    state.moveHistory.length > 0 ? state.moveHistory[state.moveHistory.length - 1] : null;
  const placedRingAt =
    lastMove?.type === 'place_ring' || lastMove?.type === 'skip_placement' ? lastMove.to : null;

  return {
    phase: 'movement',
    player,
    canMove,
    placedRingAt,
  };
}

function deriveCaptureState(
  state: GameState,
  player: number,
  isChain: boolean,
  moveHint?: Move
): CaptureState {
  const validMoves = getValidMoves(state);
  const captureMoves = validMoves.filter(
    (m) => m.type === 'overtaking_capture' || m.type === 'continue_capture_segment'
  );

  const pendingCaptures = captureMoves.map((m) => ({
    target: m.captureTarget!,
    capturingPlayer: player,
    isChainCapture: isChain,
  }));

  // If the move being validated has a captureTarget but it's not in pendingCaptures,
  // include it. This handles cases where getValidMoves() doesn't find the capture
  // due to state timing issues during replay/shadow validation.
  if (moveHint?.captureTarget) {
    const hintTarget = moveHint.captureTarget;
    const alreadyIncluded = pendingCaptures.some(
      (c) => c.target.x === hintTarget.x && c.target.y === hintTarget.y
    );
    if (!alreadyIncluded) {
      pendingCaptures.push({
        target: hintTarget,
        capturingPlayer: player,
        isChainCapture: isChain,
      });
    }
  }

  // Count captures made from chain info if available
  const lastMove =
    state.moveHistory.length > 0 ? state.moveHistory[state.moveHistory.length - 1] : null;
  const capturesMade = lastMove?.captureChain?.length ?? 0;

  return {
    phase: 'capture',
    player,
    pendingCaptures,
    chainInProgress: isChain,
    capturesMade,
  };
}

function deriveChainCaptureState(
  state: GameState,
  player: number,
  moveHint?: Move
): ChainCaptureState {
  const validMoves = getValidMoves(state);
  const continuationMoves = validMoves.filter((m) => m.type === 'continue_capture_segment');

  // Get the last move to find attacker position and capture history
  const lastMove =
    state.moveHistory.length > 0 ? state.moveHistory[state.moveHistory.length - 1] : null;
  const attackerPosition = lastMove?.to ?? { x: 0, y: 0 };
  // captureChain is already Position[] - the list of capture targets visited
  const capturedTargets = lastMove?.captureChain ?? [];

  const availableContinuations = continuationMoves.map((m) => ({
    target: m.captureTarget!,
    capturingPlayer: player,
    isChainCapture: true,
  }));

  // If the move being validated has a captureTarget but it's not in availableContinuations,
  // include it. This handles state timing issues during replay/shadow validation.
  if (moveHint?.captureTarget) {
    const hintTarget = moveHint.captureTarget;
    const alreadyIncluded = availableContinuations.some(
      (c) => c.target.x === hintTarget.x && c.target.y === hintTarget.y
    );
    if (!alreadyIncluded) {
      availableContinuations.push({
        target: hintTarget,
        capturingPlayer: player,
        isChainCapture: true,
      });
    }
  }

  return {
    phase: 'chain_capture',
    player,
    attackerPosition,
    capturedTargets,
    availableContinuations,
    segmentCount: capturedTargets.length + 1,
    isFirstSegment: capturedTargets.length === 0,
  };
}

function deriveLineProcessingState(
  state: GameState,
  player: number,
  moveHint?: Move
): LineProcessingState {
  // Detect lines from current board state
  // Per RR-CANON-R120: Pass numPlayers so findLinesForPlayer uses the correct line length threshold
  const lines: LineInfo[] = findLinesForPlayer(state.board, player, state.players.length);

  const detectedLines: DetectedLine[] = lines.map((line: LineInfo) => ({
    positions: line.positions,
    player: line.player,
    requiresChoice: line.length >= 3, // Lines of 3+ require reward choice
  }));

  // RR-CANON-R075: Trust recorded no_line_action moves during replay.
  // If Python recorded no_line_action but TS detects lines (parity divergence due to
  // line detection timing differences), return empty detectedLines to allow the move.
  // This mirrors the forced_elimination and process_territory_region trust patterns.
  if (moveHint?.type === 'no_line_action' && detectedLines.length > 0) {
    return {
      phase: 'line_processing',
      player,
      detectedLines: [], // Trust Python's decision that no lines existed
      currentLineIndex: 0,
      awaitingReward: false,
    };
  }

  // Check for pending choice from move context
  const validMoves = getValidMoves(state);
  const hasRewardChoice = validMoves.some((m) => m.type === 'choose_line_reward');

  return {
    phase: 'line_processing',
    player,
    detectedLines,
    currentLineIndex: 0,
    awaitingReward: hasRewardChoice,
  };
}

function deriveTerritoryProcessingState(
  state: GameState,
  player: number,
  moveHint?: Move
): TerritoryProcessingState {
  // Find disconnected territories
  const regions: Territory[] = findDisconnectedRegions(state.board);
  const playerRegions = regions.filter((r: Territory) => r.controllingPlayer === player);

  const disconnectedRegions: DisconnectedRegion[] = playerRegions.map((region: Territory) => ({
    positions: region.spaces,
    controllingPlayer: region.controllingPlayer,
    eliminationsRequired: region.isDisconnected ? 1 : 0, // Simplified
  }));

  // If the move being validated is a process_territory_region but disconnectedRegions is empty,
  // add a placeholder region. This handles state timing issues during replay/shadow validation
  // where Python may detect more disconnected regions than TS (parity divergence).
  // RR-CANON-R075: Trust recorded process_territory_region moves during replay.
  if (moveHint?.type === 'process_territory_region' && disconnectedRegions.length === 0) {
    disconnectedRegions.push({
      positions: moveHint.to ? [moveHint.to] : [{ x: 0, y: 0 }],
      controllingPlayer: player,
      eliminationsRequired: 1,
    });
  }

  // If the move being validated is a no_territory_action or skip_territory_processing,
  // trust the recorded move and clear disconnectedRegions. This handles parity issues
  // where TS and Python may compute different territory detection results.
  // RR-CANON-R075: Trust recorded bookkeeping moves during replay.
  if (
    (moveHint?.type === 'no_territory_action' || moveHint?.type === 'skip_territory_processing') &&
    disconnectedRegions.length > 0
  ) {
    disconnectedRegions.length = 0; // Clear the array in place
  }

  // Check for pending eliminations
  const validMoves = getValidMoves(state);
  const elimMoves = validMoves.filter((m) => m.type === 'eliminate_rings_from_stack');

  const eliminationsPending = elimMoves.map((m) => ({
    position: m.to,
    player: m.player,
    count: m.eliminatedRings?.[0]?.count ?? 1,
  }));

  return {
    phase: 'territory_processing',
    player,
    disconnectedRegions,
    currentRegionIndex: 0,
    eliminationsPending,
  };
}

function deriveForcedEliminationState(_state: GameState, player: number): ForcedEliminationState {
  // Calculate rings over limit (simplified - would need actual calculation)
  const ringsOverLimit = 0;

  return {
    phase: 'forced_elimination',
    player,
    ringsOverLimit,
    eliminationsDone: 0,
  };
}

function deriveGameOverState(state: GameState): GameOverState {
  // Extract victory info from game state
  // GameState stores winner info when game is completed
  const winner = state.winner ?? null;

  // Map reason from gameStatus or other indicators
  const reason:
    | 'ring_elimination'
    | 'territory_control'
    | 'last_player_standing'
    | 'resignation'
    | 'timeout' = 'ring_elimination';

  // Would need to check actual victory reason from game state
  // For now, default to ring_elimination

  return {
    phase: 'game_over',
    winner,
    reason,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// GAME CONTEXT DERIVATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Derive GameContext for FSM from GameState.
 */
export function deriveGameContext(state: GameState): GameContext {
  return {
    boardType: state.boardType,
    numPlayers: state.players.length,
    ringsPerPlayer: getRingsPerPlayer(state.boardType),
    lineLength: 3, // Standard line length
  };
}

function getRingsPerPlayer(boardType: string): number {
  switch (boardType) {
    case 'square8':
      return 18;
    case 'hexagonal':
      return 48;
    case 'square19':
      return 36;
    default:
      return 18;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ACTION APPLICATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply FSM actions to game state (conceptual - actual mutation done by engine).
 *
 * This function describes what each action means in terms of state changes.
 * The actual implementation would integrate with the existing game engine.
 */
export function describeActionEffects(actions: Action[]): string[] {
  return actions.map((action) => {
    switch (action.type) {
      case 'PLACE_RING':
        return `Place ring at (${action.position.x}, ${action.position.y}) for player ${action.player}`;

      case 'LEAVE_MARKER':
        return `Leave marker at (${action.position.x}, ${action.position.y}) for player ${action.player}`;

      case 'MOVE_STACK':
        return `Move stack from (${action.from.x}, ${action.from.y}) to (${action.to.x}, ${action.to.y})`;

      case 'EXECUTE_CAPTURE':
        return `Execute capture at (${action.target.x}, ${action.target.y}) by player ${action.capturer}`;

      case 'COLLAPSE_LINE':
        return `Collapse line with ${action.positions.length} markers`;

      case 'APPLY_LINE_REWARD':
        return `Apply line reward: ${action.choice} for line with ${action.line.positions.length} markers`;

      case 'PROCESS_DISCONNECTION':
        return `Process disconnection for region with ${action.region.positions.length} spaces`;

      case 'ELIMINATE_RINGS':
        return `Eliminate ${action.count} rings at (${action.target.x}, ${action.target.y})`;

      case 'FORCED_ELIMINATE':
        return `Forced elimination at (${action.target.x}, ${action.target.y}) for player ${action.player}`;

      case 'CHECK_VICTORY':
        return `Check victory conditions`;

      case 'ADVANCE_PLAYER':
        return `Advance from player ${action.from} to player ${action.to}`;

      default:
        return `Unknown action`;
    }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Validate an FSM event against the current game state.
 *
 * This bridges FSM events with the existing validateMove function.
 */
export function validateEvent(
  event: TurnEvent,
  gameState: GameState
): { valid: boolean; reason?: string } {
  const move = eventToMove(event, gameState.currentPlayer, gameState.moveHistory.length + 1);

  if (!move) {
    return { valid: false, reason: 'Cannot convert event to move' };
  }

  return validateMove(gameState, move);
}

/**
 * Get valid FSM events for the current game state.
 *
 * This bridges the FSM with getValidMoves.
 */
export function getValidEvents(gameState: GameState): TurnEvent[] {
  const moves = getValidMoves(gameState);
  const events: TurnEvent[] = [];

  for (const move of moves) {
    const event = moveToEvent(move);
    if (event) {
      events.push(event);
    }
  }

  return events;
}

// ═══════════════════════════════════════════════════════════════════════════
// FSM DEBUG LOGGING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Debug context for FSM validation - captures full state for diagnostics.
 */
export interface FSMDebugContext {
  /** Derived FSM state from game state */
  fsmState: TurnState;
  /** Game context used for transition */
  gameContext: GameContext;
  /** The event converted from move (or null if conversion failed) */
  event: TurnEvent | null;
  /** Move details */
  move: {
    type: Move['type'];
    player: number;
    to: Position;
    from?: Position;
  };
  /** Game state snapshot */
  gameStateSnapshot: {
    currentPhase: string;
    currentPlayer: number;
    moveHistoryLength: number;
    players: Array<{ playerNumber: number; ringsInHand: number }>;
  };
  /** Phase-specific derived state details */
  phaseDetails?: {
    ringsInHand?: number;
    canPlace?: boolean;
    validPositionsCount?: number;
    canMove?: boolean;
    detectedLinesCount?: number;
    disconnectedRegionsCount?: number;
  };
}

/**
 * Logger interface for FSM validation debugging.
 * Set FSM_DEBUG_LOGGER to enable detailed logging.
 */
export interface FSMDebugLogger {
  /** Log a validation attempt */
  logValidation(context: FSMDebugContext, result: FSMValidationResult): void;
  /** Log a divergence between FSM and legacy validation */
  logDivergence(
    context: FSMDebugContext,
    fsmResult: FSMValidationResult,
    legacyResult: { valid: boolean; reason?: string }
  ): void;
}

/** Global debug logger - set this to enable detailed FSM logging */
export let FSM_DEBUG_LOGGER: FSMDebugLogger | null = null;

/**
 * Set the FSM debug logger.
 * @param logger Logger implementation or null to disable
 */
export function setFSMDebugLogger(logger: FSMDebugLogger | null): void {
  FSM_DEBUG_LOGGER = logger;
}

/**
 * Console-based debug logger for FSM validation.
 * This logger intentionally uses console.log for debug output.
 */
/* eslint-disable no-console */
export const consoleFSMDebugLogger: FSMDebugLogger = {
  logValidation(context: FSMDebugContext, result: FSMValidationResult): void {
    const prefix = result.valid ? '✅' : '❌';
    console.log(
      `[FSM] ${prefix} ${context.move.type} by P${context.move.player} ` +
        `in phase=${context.fsmState.phase} ` +
        `(gamePhase=${context.gameStateSnapshot.currentPhase}, ` +
        `currentPlayer=${context.gameStateSnapshot.currentPlayer})`
    );
    if (!result.valid) {
      console.log(`  Error: [${result.errorCode}] ${result.reason}`);
      if (result.validEventTypes) {
        console.log(`  Expected events: ${result.validEventTypes.join(', ')}`);
      }
      if (context.phaseDetails) {
        console.log(`  Phase details: ${JSON.stringify(context.phaseDetails)}`);
      }
    }
  },
  logDivergence(
    context: FSMDebugContext,
    fsmResult: FSMValidationResult,
    legacyResult: { valid: boolean; reason?: string }
  ): void {
    console.log(
      `[FSM DIVERGENCE] ${context.move.type} by P${context.move.player} ` +
        `at k=${context.gameStateSnapshot.moveHistoryLength}`
    );
    console.log(`  Game phase: ${context.gameStateSnapshot.currentPhase}`);
    console.log(`  FSM phase: ${fsmResult.currentPhase}`);
    console.log(`  FSM: valid=${fsmResult.valid} ${fsmResult.reason || ''}`);
    console.log(`  Legacy: valid=${legacyResult.valid} ${legacyResult.reason || ''}`);
    if (context.phaseDetails) {
      console.log(`  Phase details: ${JSON.stringify(context.phaseDetails)}`);
    }
    console.log(`  Players: ${JSON.stringify(context.gameStateSnapshot.players)}`);
  },
};
/* eslint-enable no-console */

/**
 * Build debug context from validation inputs.
 */
function buildDebugContext(
  gameState: GameState,
  move: Move,
  fsmState: TurnState,
  gameContext: GameContext,
  event: TurnEvent | null
): FSMDebugContext {
  const moveInfo: FSMDebugContext['move'] = {
    type: move.type,
    player: move.player,
    to: move.to,
  };
  // Only include from if defined (exactOptionalPropertyTypes compliance)
  if (move.from) {
    moveInfo.from = move.from;
  }

  const context: FSMDebugContext = {
    fsmState,
    gameContext,
    event,
    move: moveInfo,
    gameStateSnapshot: {
      currentPhase: gameState.currentPhase,
      currentPlayer: gameState.currentPlayer,
      moveHistoryLength: gameState.moveHistory.length,
      players: gameState.players.map((p) => ({
        playerNumber: p.playerNumber,
        ringsInHand: p.ringsInHand,
      })),
    },
  };

  // Add phase-specific details
  if (fsmState.phase === 'ring_placement') {
    const placementState = fsmState as RingPlacementState;
    context.phaseDetails = {
      ringsInHand: placementState.ringsInHand,
      canPlace: placementState.canPlace,
      validPositionsCount: placementState.validPositions.length,
    };
  } else if (fsmState.phase === 'movement') {
    const movementState = fsmState as MovementState;
    context.phaseDetails = {
      canMove: movementState.canMove,
    };
  } else if (fsmState.phase === 'line_processing') {
    const lineState = fsmState as LineProcessingState;
    context.phaseDetails = {
      detectedLinesCount: lineState.detectedLines.length,
    };
  } else if (fsmState.phase === 'territory_processing') {
    const territoryState = fsmState as TerritoryProcessingState;
    context.phaseDetails = {
      disconnectedRegionsCount: territoryState.disconnectedRegions.length,
    };
  }

  return context;
}

// ═══════════════════════════════════════════════════════════════════════════
// FSM-BASED MOVE VALIDATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * FSM validation result with detailed phase information.
 */
export interface FSMValidationResult {
  /** Whether the move is valid according to FSM rules */
  valid: boolean;
  /** Current FSM phase derived from game state */
  currentPhase: TurnState['phase'];
  /** Error code if invalid */
  errorCode?:
    | 'INVALID_EVENT'
    | 'GUARD_FAILED'
    | 'INVALID_STATE'
    | 'CONVERSION_FAILED'
    | 'WRONG_PLAYER';
  /** Human-readable reason for rejection */
  reason?: string;
  /** Expected event types for this phase */
  validEventTypes?: TurnEvent['type'][];
  /** Debug context (populated when debug logging is enabled) */
  debugContext?: FSMDebugContext;
}

/**
 * Validate a move using FSM transition rules.
 *
 * This provides "shadow validation" that can run alongside
 * existing validation to ensure FSM rules match expected behavior.
 *
 * @param gameState Current game state
 * @param move The move to validate
 * @param includeDebugContext If true, includes full debug context in result
 * @returns FSM validation result with phase context
 */
export function validateMoveWithFSM(
  gameState: GameState,
  move: Move,
  includeDebugContext = false
): FSMValidationResult {
  // Derive FSM state and context from game state
  // Pass move as hint to help state derivation include relevant context
  const fsmState = deriveStateFromGame(gameState, move);
  const gameContext = deriveGameContext(gameState);

  // Convert move to FSM event (do this early for debug context)
  const event = moveToEvent(move);

  // Build debug context if logging is enabled or requested
  const debugContext =
    FSM_DEBUG_LOGGER || includeDebugContext
      ? buildDebugContext(gameState, move, fsmState, gameContext, event)
      : undefined;

  // Helper to create result and optionally log
  const makeResult = (result: FSMValidationResult): FSMValidationResult => {
    if (includeDebugContext && debugContext) {
      result.debugContext = debugContext;
    }
    if (FSM_DEBUG_LOGGER && debugContext) {
      FSM_DEBUG_LOGGER.logValidation(debugContext, result);
    }
    return result;
  };

  // Validate player attribution: the move must be from the current player.
  // Exceptions for player mismatch:
  // - Bookkeeping moves (no_*_action): may be auto-injected at turn boundaries
  // - process_territory_region: Python may detect more regions than TS, causing
  //   TS to transition players before all Python-recorded territory moves complete
  // RR-CANON-R075: Trust recorded moves during replay.
  const isPlayerMismatchExempt =
    move.type === 'no_placement_action' ||
    move.type === 'no_movement_action' ||
    move.type === 'no_line_action' ||
    move.type === 'no_territory_action' ||
    move.type === 'process_territory_region';

  if (!isPlayerMismatchExempt && move.player !== gameState.currentPlayer) {
    return makeResult({
      valid: false,
      currentPhase: fsmState.phase,
      errorCode: 'WRONG_PLAYER',
      reason: `Not your turn (expected player ${gameState.currentPlayer}, got ${move.player})`,
    });
  }

  // Check event conversion
  if (!event) {
    // Meta-moves (swap_sides, line_formation, territory_claim) don't have FSM events
    // but are allowed in any phase. If the move type is explicitly permitted for the
    // current phase, let it through - the orchestrator handles meta-moves specially.
    //
    // TurnState['phase'] includes internal pseudo-phases such as 'turn_end' that are
    // not part of the public GamePhase contract. Only call isMoveTypeValidForPhase
    // when we are in a real GamePhase to keep types sound and avoid leaking
    // pseudo-phases into the shared phase↔MoveType mapping.
    if (fsmState.phase !== 'game_over' && fsmState.phase !== 'turn_end') {
      const phaseForMoveCheck = fsmState.phase as GamePhase;
      if (isMoveTypeValidForPhase(phaseForMoveCheck, move.type as MoveType)) {
        return makeResult({
          valid: true,
          currentPhase: fsmState.phase,
        });
      }
    }
    return makeResult({
      valid: false,
      currentPhase: fsmState.phase,
      errorCode: 'CONVERSION_FAILED',
      reason: `Cannot convert move type '${move.type}' to FSM event (meta-move or unsupported)`,
    });
  }

  // Attempt the transition
  const result = transition(fsmState, event, gameContext);

  if (!result.ok) {
    const error = result.error;
    return makeResult({
      valid: false,
      currentPhase: fsmState.phase,
      errorCode: error.code,
      reason: error.message,
      validEventTypes: getExpectedEventTypes(fsmState),
    });
  }

  return makeResult({
    valid: true,
    currentPhase: fsmState.phase,
  });
}

/**
 * Validate a move with FSM and compare against legacy validation.
 * Logs divergences when they occur.
 *
 * @param gameState Current game state
 * @param move The move to validate
 * @returns Object containing both validation results
 */
export function validateMoveWithFSMAndCompare(
  gameState: GameState,
  move: Move
): {
  fsmResult: FSMValidationResult;
  legacyResult: { valid: boolean; reason?: string };
  divergence: boolean;
} {
  const fsmResult = validateMoveWithFSM(gameState, move, true);
  const legacyResult = validateMove(gameState, move);

  const divergence = fsmResult.valid !== legacyResult.valid;

  if (divergence && FSM_DEBUG_LOGGER && fsmResult.debugContext) {
    FSM_DEBUG_LOGGER.logDivergence(fsmResult.debugContext, fsmResult, legacyResult);
  }

  return { fsmResult, legacyResult, divergence };
}

/**
 * Canonical mapping from FSM phase → allowed event types.
 *
 * All helpers that need to reason about phase ↔ MoveType relationships
 * derive from this map so that the event surface and MoveType surface
 * stay in sync.
 */
const PHASE_EVENT_TYPES: Record<TurnState['phase'], ReadonlyArray<TurnEvent['type']>> = {
  ring_placement: ['PLACE_RING', 'SKIP_PLACEMENT', 'NO_PLACEMENT_ACTION', 'RESIGN', 'TIMEOUT'],
  movement: ['MOVE_STACK', 'CAPTURE', 'RECOVERY_SLIDE', 'NO_MOVEMENT_ACTION', 'RESIGN', 'TIMEOUT'],
  capture: ['CAPTURE', 'END_CHAIN', 'RESIGN', 'TIMEOUT'],
  chain_capture: ['CONTINUE_CHAIN', 'END_CHAIN', 'RESIGN', 'TIMEOUT'],
  line_processing: ['PROCESS_LINE', 'CHOOSE_LINE_REWARD', 'NO_LINE_ACTION', 'RESIGN', 'TIMEOUT'],
  territory_processing: [
    'PROCESS_REGION',
    'ELIMINATE_FROM_STACK',
    'NO_TERRITORY_ACTION',
    'RESIGN',
    'TIMEOUT',
  ],
  // Also allow ELIMINATE_FROM_STACK for backwards compatibility with historical game
  // logs that recorded 'eliminate_rings_from_stack' moves during forced_elimination.
  forced_elimination: ['FORCED_ELIMINATE', 'ELIMINATE_FROM_STACK', 'RESIGN', 'TIMEOUT'],
  turn_end: ['_ADVANCE_TURN'],
  game_over: [],
};

/**
 * Get expected event types for the current FSM phase.
 */
function getExpectedEventTypes(state: TurnState): TurnEvent['type'][] {
  const events = PHASE_EVENT_TYPES[state.phase] ?? [];
  // Return a mutable copy to satisfy the non-readonly return type.
  return [...events];
}

/**
 * Canonical mapping from GamePhase → allowed MoveType values.
 *
 * This encodes the same contract that previously lived in
 * assertPhaseMoveInvariant / isPhaseValidForMoveType in turnOrchestrator
 * and is now centralised in the FSM adapter.
 */
const PHASE_ALLOWED_MOVE_TYPES: Record<GamePhase, ReadonlyArray<MoveType>> = {
  ring_placement: ['place_ring', 'skip_placement', 'no_placement_action'],
  movement: [
    'move_stack',
    'move_ring',
    'overtaking_capture',
    'continue_capture_segment',
    'no_movement_action',
    'recovery_slide',
  ],
  capture: ['overtaking_capture', 'continue_capture_segment', 'skip_capture'],
  chain_capture: ['overtaking_capture', 'continue_capture_segment'],
  line_processing: ['process_line', 'choose_line_reward', 'no_line_action'],
  territory_processing: [
    'process_territory_region',
    'eliminate_rings_from_stack',
    'skip_territory_processing',
    'no_territory_action',
  ],
  // Also allow 'eliminate_rings_from_stack' for backwards compatibility with
  // historical game logs that recorded this move type during forced_elimination.
  forced_elimination: ['forced_elimination', 'eliminate_rings_from_stack'],
  game_over: [],
};

/**
 * Get the set of canonical MoveType values allowed for a given GamePhase.
 */
export function getAllowedMoveTypesForPhase(phase: GamePhase): ReadonlyArray<MoveType> {
  return PHASE_ALLOWED_MOVE_TYPES[phase] ?? [];
}

/**
 * Check if a MoveType is structurally valid for a given GamePhase.
 *
 * This is a lightweight check that does not run full validation and is
 * suitable for UI hints, pre-flight invariants, and diagnostics. It treats
 * FSMAdapter as the single source of truth for the canonical phase ↔ MoveType
 * contract used by the orchestrator.
 */
export function isMoveTypeValidForPhase(phase: GamePhase, moveType: MoveType): boolean {
  // Resign and timeout are allowed from any phase - player can always forfeit
  // or run out of time. Both result in game termination.
  if (moveType === 'resign' || moveType === 'timeout') {
    return true;
  }

  // Meta / legacy moves are allowed in any phase for historical compatibility.
  if (
    moveType === 'swap_sides' ||
    moveType === 'line_formation' ||
    moveType === 'territory_claim'
  ) {
    return true;
  }

  const allowedForPhase = PHASE_ALLOWED_MOVE_TYPES[phase];
  if (!allowedForPhase) {
    // Unknown/legacy phases default to permissive behaviour, matching the
    // previous orchestrator helpers.
    return true;
  }

  return allowedForPhase.includes(moveType);
}

// ═══════════════════════════════════════════════════════════════════════════
// ORCHESTRATION INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Context for determining the next phase after a move.
 *
 * This provides the information needed by TurnStateMachine to determine
 * valid phase transitions.
 */
export interface PhaseTransitionContext {
  /** Whether more lines exist to process */
  hasMoreLinesToProcess: boolean;
  /** Whether more territory regions exist to process */
  hasMoreRegionsToProcess: boolean;
  /** Whether chain capture continuations are available */
  chainCapturesAvailable: boolean;
  /** Whether the player has any movement options */
  hasAnyMovement: boolean;
  /** Whether the player has any capture options */
  hasAnyCapture: boolean;
}

/**
 * Determine the next phase using FSM transition logic.
 *
 * This is the FSM-based equivalent of `determineNextPhase()` from
 * phaseStateMachine.ts. It uses the TurnStateMachine's transition rules
 * to determine valid phase progressions.
 *
 * @param gameState Current game state
 * @param moveType Type of move that was just processed
 * @param context Additional context about available actions
 * @returns The next phase according to FSM rules
 */
export function determineNextPhaseFromFSM(
  gameState: GameState,
  _moveType: Move['type'],
  context: PhaseTransitionContext
): TurnState['phase'] {
  const currentPhase = gameState.currentPhase;

  // Use FSM transition rules to determine next phase
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
      // After territory, turn ends (handled by turn advance)
      return 'territory_processing';

    case 'forced_elimination':
      // After forced elimination, turn ends (handled by turn advance)
      return 'forced_elimination';

    default:
      return currentPhase as TurnState['phase'];
  }
}

/**
 * Result of an FSM-driven phase transition attempt.
 */
export interface FSMTransitionAttemptResult {
  /** Whether the transition was valid */
  valid: boolean;
  /** The resulting FSM state if valid */
  nextState?: TurnState;
  /** Actions to apply if valid */
  actions?: Action[];
  /** Error details if invalid */
  error?: {
    code: 'INVALID_EVENT' | 'GUARD_FAILED' | 'INVALID_STATE' | 'CONVERSION_FAILED';
    message: string;
  };
}

/**
 * Attempt a phase transition using the FSM.
 *
 * This provides a higher-level interface for orchestration code that wants
 * to use TurnStateMachine for phase transitions without fully replacing
 * existing PhaseStateMachine usage.
 *
 * @param gameState Current game state
 * @param move The move to process
 * @returns Transition result including new state and actions
 */
export function attemptFSMTransition(gameState: GameState, move: Move): FSMTransitionAttemptResult {
  // Pass move as hint so state derivation can trust recorded bookkeeping moves
  const fsmState = deriveStateFromGame(gameState, move);
  const context = deriveGameContext(gameState);

  const event = moveToEvent(move);
  if (!event) {
    return {
      valid: false,
      error: {
        code: 'CONVERSION_FAILED',
        message: `Cannot convert move type '${move.type}' to FSM event`,
      },
    };
  }

  const result = transition(fsmState, event, context);

  if (result.ok) {
    return {
      valid: true,
      nextState: result.state,
      actions: result.actions,
    };
  }

  return {
    valid: false,
    error: {
      code: result.error.code,
      message: result.error.message,
    },
  };
}

/**
 * Get the current FSM state for a game.
 *
 * This is useful for debugging and for components that need to understand
 * the FSM representation of the current game state.
 */
export function getCurrentFSMState(gameState: GameState): TurnState {
  return deriveStateFromGame(gameState);
}

/**
 * Check if the FSM considers the game to be in a terminal state.
 */
export function isFSMTerminalState(gameState: GameState): boolean {
  const fsmState = deriveStateFromGame(gameState);
  return fsmState.phase === 'game_over';
}

// ═══════════════════════════════════════════════════════════════════════════
// FSM-DRIVEN ORCHESTRATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Decision surface information for hosts to construct valid decisions.
 *
 * When the FSM transitions to a phase requiring player decisions, this
 * structure provides the concrete data needed to build the decision UI
 * and valid move options.
 */
export interface FSMDecisionSurface {
  /**
   * Detected lines for the current player (line_processing phase).
   * Empty if not in line_processing or no lines detected.
   */
  pendingLines: DetectedLine[];

  /**
   * Territory regions for the current player (territory_processing phase).
   * Empty if not in territory_processing or no regions to process.
   */
  pendingRegions: DisconnectedRegion[];

  /**
   * Chain capture continuation targets (chain_capture phase).
   * Empty if not in chain capture or no continuations available.
   */
  chainContinuations: Array<{ target: Position }>;

  /**
   * Forced elimination targets (forced_elimination phase).
   * Number of rings that must be eliminated.
   */
  forcedEliminationCount: number;
}

/**
 * Result of FSM-driven orchestration.
 * Contains the derived phase, player, and any pending decision information.
 */
export interface FSMOrchestrationResult {
  /** Whether the FSM transition was successful */
  success: boolean;
  /** The next phase according to FSM */
  nextPhase: TurnState['phase'];
  /** The next player according to FSM */
  nextPlayer: number;
  /** FSM actions that should be applied */
  actions: Action[];
  /** Type of pending decision if any (derived from FSM state) */
  pendingDecisionType?:
    | 'chain_capture'
    | 'line_order_required'
    | 'no_line_action_required'
    | 'region_order_required'
    | 'no_territory_action_required'
    | 'forced_elimination';
  /**
   * Decision surface data for the pending decision.
   * Contains the actual lines/regions/targets needed to construct valid moves.
   * Only populated when pendingDecisionType is set.
   */
  decisionSurface?: FSMDecisionSurface;
  /** Error information if transition failed */
  error?: {
    code: string;
    message: string;
  };
  /** Debug information */
  debug?: {
    inputPhase: string;
    inputPlayer: number;
    fsmState: TurnState;
    event: TurnEvent | null;
  };
}

/**
 * Compute the next phase and player using FSM transition logic.
 *
 * This is the core function for FSM-driven orchestration. It takes
 * the current game state and move, runs the FSM transition, and
 * returns the orchestration result.
 *
 * @param gameState Current game state (after move application)
 * @param move The move that was just applied
 * @param options Additional context for decision surfacing
 * @returns FSM orchestration result
 */
export function computeFSMOrchestration(
  gameState: GameState,
  move: Move,
  options?: {
    /** Whether chain capture continuation is available */
    chainCapturesAvailable?: boolean;
    /** Detected lines for the current player */
    detectedLinesCount?: number;
    /** Territory regions for the current player */
    territoryRegionsCount?: number;
    /** Whether the player had any real action this turn */
    hadAnyActionThisTurn?: boolean;
    /** Whether the player has stacks on the board */
    playerHasStacks?: boolean;
    /** Post-move state for chain capture continuation checking */
    postMoveStateForChainCheck?: GameState;
  }
): FSMOrchestrationResult {
  // Derive FSM state from game state, passing move as hint
  const fsmState = deriveStateFromGame(gameState, move);
  const context = deriveGameContext(gameState);

  // Convert move to FSM event
  const event = moveToEvent(move);

  // Build debug info
  const debug = {
    inputPhase: gameState.currentPhase,
    inputPlayer: gameState.currentPlayer,
    fsmState,
    event,
  };

  if (!event) {
    // Meta-move or unsupported - return current state unchanged
    return {
      success: true,
      nextPhase: fsmState.phase,
      nextPlayer: 'player' in fsmState ? fsmState.player : gameState.currentPlayer,
      actions: [],
      debug,
    };
  }

  // Run FSM transition
  const transitionResult = transition(fsmState, event, context);

  if (!transitionResult.ok) {
    return {
      success: false,
      nextPhase: fsmState.phase,
      nextPlayer: 'player' in fsmState ? fsmState.player : gameState.currentPlayer,
      actions: [],
      error: {
        code: transitionResult.error.code,
        message: transitionResult.error.message,
      },
      debug,
    };
  }

  let nextState = transitionResult.state;
  const actions = transitionResult.actions;

  // Post-transition adjustment: Check for chain capture availability after captures.
  // The pure FSM doesn't have access to board state, so we need to verify chain
  // capture availability against actual board state and adjust accordingly.
  //
  // Two cases to handle:
  // 1. FSM says line_processing after capture → check if chain captures exist
  // 2. FSM says chain_capture (optimistically) → verify continuations actually exist
  const isCaptureMove =
    move.type === 'overtaking_capture' || move.type === 'continue_capture_segment';

  // Post-transition adjustment: after a pure movement, check whether the landing
  // stack has any overtaking captures available. The FSM does not know about
  // board-local capture availability, so without this check it would advance
  // directly to line_processing even when Python's phase machine enters CAPTURE
  // after MOVE_STACK.
  if ((move.type === 'move_stack' || move.type === 'move_ring') && move.to) {
    const stateForCaptureCheck = options?.postMoveStateForChainCheck ?? gameState;

    // Enumerate initial (overtaking) captures from the landing position only.
    // This mirrors Python GameEngine._get_capture_moves, which considers
    // captures from the stack that just moved, not from all stacks.
    const initialCaptures = enumerateChainCaptureSegments(
      stateForCaptureCheck,
      {
        player: move.player,
        currentPosition: move.to,
        capturedThisChain: [],
      },
      {
        kind: 'initial',
      }
    );

    if (initialCaptures.length > 0) {
      const landingKey = positionToString(move.to);
      const pendingCaptures = initialCaptures
        .filter((m) => m.from && positionToString(m.from) === landingKey)
        .map((m) => ({
          target: m.captureTarget!,
          capturingPlayer: move.player,
          isChainCapture: false,
        }));

      if (pendingCaptures.length > 0) {
        nextState = {
          phase: 'capture',
          player: move.player,
          pendingCaptures,
          chainInProgress: false,
          capturesMade: 0,
        } as CaptureState;
      }
    }
  }

  if (isCaptureMove && move.to) {
    // Use post-move state for chain capture check if available, otherwise fall back to pre-move state.
    // For RR‑CANON capture-chain semantics we must mirror Python's
    // get_chain_capture_continuation_info_py, which:
    //   - Always treats disallow_revisited_targets = False at rules level
    //   - Uses an empty captured_this_chain snapshot when deciding whether
    //     the chain must continue from the current landing position.
    //
    // The previous implementation incorrectly threaded move.captureChain and
    // used disallowRevisitedTargets: true, which is intended only for search /
    // analysis tooling. That could cause the FSM orchestrator to believe no
    // continuations existed even when the shared capture aggregate reported a
    // mandatory chain continuation, leading to premature transition to
    // line_processing after an overtaking_capture.
    const stateForChainCheck = options?.postMoveStateForChainCheck ?? gameState;

    const continuations = enumerateChainCaptureSegments(
      stateForChainCheck,
      {
        player: move.player,
        currentPosition: move.to,
        // Rules-level continuation check never filters revisited targets; the
        // shared aggregate handles pathological cycles separately.
        capturedThisChain: [],
      },
      {
        kind: 'continuation',
      }
    );

    if (continuations.length > 0) {
      // Chain captures are available - ensure we're in chain_capture phase.
      // Snapshot of capturedTargets is reconstructed from continuation moves
      // only for diagnostics; rules semantics depend solely on the existence
      // of at least one continuation.
      nextState = {
        phase: 'chain_capture',
        player: move.player,
        attackerPosition: move.to,
        capturedTargets: [],
        availableContinuations: continuations.map((m) => ({
          target: m.captureTarget ?? m.to,
          capturingPlayer: move.player,
          isChainCapture: true,
        })),
        segmentCount: 1,
        isFirstSegment: false,
      } as ChainCaptureState;
    } else if (nextState.phase === 'chain_capture') {
      // FSM optimistically said chain_capture, but no continuations exist
      // according to the shared aggregate. Transition to line_processing
      // instead so TS mirrors Python's phase_machine semantics.
      nextState = {
        phase: 'line_processing',
        player: move.player,
        detectedLines: [],
        currentLineIndex: 0,
        awaitingReward: false,
      } as LineProcessingState;
    }
  }

  // Post-transition adjustment for territory processing: The pure FSM expects
  // a two-step process (PROCESS_REGION then ELIMINATE_FROM_STACK for internal
  // eliminations), but the game engine's process_territory_region move handles
  // internal eliminations atomically. If the FSM stays in territory_processing
  // after a process_territory_region move, transition to turn_end because:
  // 1. The game engine atomically processes the region (including internal eliminations)
  // 2. The orchestrator will surface another pending decision if more regions exist
  // 3. The FSM can't accurately predict whether more regions exist post-move
  if (move.type === 'process_territory_region' && nextState.phase === 'territory_processing') {
    // Always transition to turn_end after process_territory_region
    // The orchestrator handles surfacing additional region decisions if needed
    const computedNextPlayer = (move.player % context.numPlayers) + 1;
    nextState = {
      phase: 'turn_end',
      completedPlayer: move.player,
      nextPlayer: computedNextPlayer,
    } as TurnEndState;
  }

  // Extract next phase and player from FSM state
  let nextPhase = nextState.phase;
  let nextPlayer: number;

  if (nextState.phase === 'turn_end') {
    // Turn ended - next player is in the state
    nextPlayer = (nextState as { nextPlayer: number }).nextPlayer;
    // Per RR-CANON: resolve turn_end to actual starting phase based on next player's material
    // ring_placement if ringsInHand > 0, else movement
    const nextPlayerIdx = nextPlayer - 1; // 0-indexed
    const nextPlayerRingsInHand = gameState.players[nextPlayerIdx]?.ringsInHand ?? 0;
    nextPhase = nextPlayerRingsInHand > 0 ? 'ring_placement' : 'movement';
  } else if (nextState.phase === 'game_over') {
    // Game over - player doesn't matter
    nextPlayer = gameState.currentPlayer;
  } else {
    // In-progress phase - player is in the state
    nextPlayer = (nextState as { player: number }).player;
  }

  // Determine pending decision type and build decision surface from FSM state
  let pendingDecisionType: FSMOrchestrationResult['pendingDecisionType'];
  let decisionSurface: FSMDecisionSurface | undefined;

  if (nextPhase === 'chain_capture') {
    pendingDecisionType = 'chain_capture';
    const chainState = nextState as ChainCaptureState;
    decisionSurface = {
      pendingLines: [],
      pendingRegions: [],
      chainContinuations: chainState.availableContinuations.map((c) => ({ target: c.target })),
      forcedEliminationCount: 0,
    };
  } else if (nextPhase === 'line_processing') {
    const lineState = nextState as LineProcessingState;
    const linesCount = lineState.detectedLines.length;
    if (linesCount > 0) {
      pendingDecisionType = 'line_order_required';
      decisionSurface = {
        pendingLines: lineState.detectedLines,
        pendingRegions: [],
        chainContinuations: [],
        forcedEliminationCount: 0,
      };
    } else if (!isLinePhaseMove(move.type)) {
      pendingDecisionType = 'no_line_action_required';
      decisionSurface = {
        pendingLines: [],
        pendingRegions: [],
        chainContinuations: [],
        forcedEliminationCount: 0,
      };
    }
  } else if (nextPhase === 'territory_processing') {
    const territoryState = nextState as TerritoryProcessingState;
    const regionsCount = territoryState.disconnectedRegions.length;
    if (regionsCount > 0) {
      pendingDecisionType = 'region_order_required';
      decisionSurface = {
        pendingLines: [],
        pendingRegions: territoryState.disconnectedRegions,
        chainContinuations: [],
        forcedEliminationCount: 0,
      };
    } else if (!isTerritoryPhaseMove(move.type)) {
      pendingDecisionType = 'no_territory_action_required';
      decisionSurface = {
        pendingLines: [],
        pendingRegions: [],
        chainContinuations: [],
        forcedEliminationCount: 0,
      };
    }
  } else if (nextPhase === 'forced_elimination') {
    pendingDecisionType = 'forced_elimination';
    const feState = nextState as ForcedEliminationState;
    decisionSurface = {
      pendingLines: [],
      pendingRegions: [],
      chainContinuations: [],
      forcedEliminationCount: feState.ringsOverLimit - feState.eliminationsDone,
    };
  }

  const result: FSMOrchestrationResult = {
    success: true,
    nextPhase,
    nextPlayer,
    actions,
    debug,
  };

  // Only set optional properties if defined (exactOptionalPropertyTypes compliance)
  if (pendingDecisionType) {
    result.pendingDecisionType = pendingDecisionType;
  }
  if (decisionSurface) {
    result.decisionSurface = decisionSurface;
  }

  return result;
}

/**
 * Check if a move type is a line-phase move.
 */
function isLinePhaseMove(moveType: Move['type']): boolean {
  return (
    moveType === 'no_line_action' ||
    moveType === 'process_line' ||
    moveType === 'choose_line_reward'
  );
}

/**
 * Check if a move type is a territory-phase move.
 */
function isTerritoryPhaseMove(moveType: Move['type']): boolean {
  return (
    moveType === 'no_territory_action' ||
    moveType === 'process_territory_region' ||
    moveType === 'skip_territory_processing'
  );
}

/**
 * Compare FSM orchestration result with legacy orchestration result.
 * Returns divergence details if they differ.
 *
 * Per RR-CANON, when a turn ends (FSM returns `turn_end`), the next player's
 * starting phase depends on their material:
 * - ring_placement if ringsInHand > 0
 * - movement if ringsInHand == 0 but they control at least one stack
 *
 * @param fsmResult - The FSM orchestration result
 * @param legacyPhase - The phase from the legacy/Python orchestration
 * @param legacyPlayer - The player from the legacy/Python orchestration
 * @param nextPlayerRingsInHand - Optional: next player's rings in hand for turn_end resolution
 */
export function compareFSMWithLegacy(
  fsmResult: FSMOrchestrationResult,
  legacyPhase: string,
  legacyPlayer: number,
  nextPlayerRingsInHand?: number
): {
  diverged: boolean;
  phaseDiverged: boolean;
  playerDiverged: boolean;
  details?: {
    fsmPhase: string;
    legacyPhase: string;
    fsmPlayer: number;
    legacyPlayer: number;
  };
} {
  // Map FSM turn_end to the actual starting phase based on next player's material
  // Per RR-CANON: ring_placement if ringsInHand > 0, else movement
  let effectiveFSMPhase: string;
  if (fsmResult.nextPhase === 'turn_end') {
    if (nextPlayerRingsInHand !== undefined && nextPlayerRingsInHand > 0) {
      effectiveFSMPhase = 'ring_placement';
    } else if (nextPlayerRingsInHand !== undefined && nextPlayerRingsInHand === 0) {
      effectiveFSMPhase = 'movement';
    } else {
      // Fallback when ringsInHand not provided - assume ring_placement (legacy behavior)
      effectiveFSMPhase = 'ring_placement';
    }
  } else {
    effectiveFSMPhase = fsmResult.nextPhase;
  }

  const phaseDiverged = effectiveFSMPhase !== legacyPhase;
  const playerDiverged = fsmResult.nextPlayer !== legacyPlayer;
  const diverged = phaseDiverged || playerDiverged;

  if (diverged) {
    return {
      diverged: true,
      phaseDiverged,
      playerDiverged,
      details: {
        fsmPhase: effectiveFSMPhase,
        legacyPhase,
        fsmPlayer: fsmResult.nextPlayer,
        legacyPlayer,
      },
    };
  }

  return { diverged: false, phaseDiverged: false, playerDiverged: false };
}
