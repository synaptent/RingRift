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
  BoardType,
} from '../../types/game';
import { positionToString, BOARD_CONFIGS } from '../../types/game';
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
import { hasAnyGlobalMovementOrCapture, playerHasAnyRings } from '../globalActions';
import { isEligibleForRecovery } from '../playerStateHelpers';
import { VALID_MOVES_BY_PHASE, isMoveValidInPhase } from '../phaseValidation';
import { isLegacyMoveValidInPhase } from '../legacy/legacyPhaseValidation';
import { isLegacyMoveType, normalizeLegacyMove } from '../legacy/legacyMoveTypes';
import { getEffectiveLineLengthThreshold, getEffectiveRingsPerPlayer } from '../rulesConfig';
import { isValidPosition } from '../validators/utils';

// ═══════════════════════════════════════════════════════════════════════════
// TURN ROTATION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute the next player after the given player, skipping permanently eliminated
 * players (RR-CANON-R201).
 *
 * A player is permanently eliminated if they have no rings anywhere:
 * - No controlled stacks (top ring)
 * - No buried rings (their rings inside stacks controlled by others)
 * - No rings in hand
 *
 * Such players are removed from turn rotation entirely.
 *
 * @param gameState - The current game state (used to check player elimination status)
 * @param currentPlayer - The player whose turn just ended
 * @param numPlayers - Total number of players in the game
 * @returns The next non-eliminated player number
 */
function computeNextNonEliminatedPlayer(
  gameState: GameState,
  currentPlayer: number,
  numPlayers: number
): number {
  let nextPlayer = (currentPlayer % numPlayers) + 1;
  let skips = 0;

  // Skip up to numPlayers times to find a non-eliminated player
  while (skips < numPlayers) {
    if (playerHasAnyRings(gameState, nextPlayer)) {
      return nextPlayer;
    }
    // Player has no rings anywhere - permanently eliminated, skip
    nextPlayer = (nextPlayer % numPlayers) + 1;
    skips += 1;
  }

  // All players eliminated - return the simple rotation (shouldn't happen in valid games)
  return (currentPlayer % numPlayers) + 1;
}

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
  if (isLegacyMoveType(move.type)) {
    return null;
  }

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

    case 'skip_recovery':
      return { type: 'SKIP_RECOVERY' };

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

    case 'choose_line_option': {
      // Determine choice from move data
      const choice = extractLineRewardChoice(move);
      return { type: 'CHOOSE_LINE_REWARD', choice };
    }

    case 'no_line_action':
      return { type: 'NO_LINE_ACTION' };

    // Territory Processing
    case 'choose_territory_option':
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
      // Meta-move not handled by turn FSM
      return null;

    default:
      // Exhaustive check - should never reach here
      return null;
  }
}

/**
 * Extract line reward choice from move data.
 *
 * RR-CANON-R123: The choice is determined by comparing collapsed markers to line length:
 * - Option 1 (eliminate): Collapse ALL markers → ring elimination reward
 * - Option 2 (territory): Collapse MINIMUM markers (cap-height from each end) → no reward
 *
 * If collapsedMarkers.length == line.length: Option 1 (eliminate)
 * If collapsedMarkers.length < line.length: Option 2 (territory)
 */
export function extractLineRewardChoice(move: Move): LineRewardChoice {
  // Get the line length from formedLines if available
  const line = move.formedLines?.[0];
  const lineLength = line?.length ?? line?.positions?.length ?? 0;
  const collapsedCount = move.collapsedMarkers?.length ?? 0;

  // If we have both line length and collapsed markers, compare them
  if (lineLength > 0 && collapsedCount > 0) {
    // Option 2 (territory) = collapse MINIMUM (fewer than all markers)
    // Option 1 (eliminate) = collapse ALL markers
    if (collapsedCount < lineLength) {
      return 'territory';
    }
    // collapsedCount >= lineLength means collapse all (Option 1)
    return 'eliminate';
  }

  // Fallback: if no collapsed markers at all, assume eliminate (Option 1)
  // This handles old recordings that didn't track collapsedMarkers
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

    case 'SKIP_RECOVERY':
      return { ...baseMove, type: 'skip_recovery', to: { x: 0, y: 0 } };

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
      return { ...baseMove, type: 'choose_line_option', to: { x: 0, y: 0 } };

    case 'NO_LINE_ACTION':
      return { ...baseMove, type: 'no_line_action', to: { x: 0, y: 0 } };

    case 'PROCESS_REGION':
      return { ...baseMove, type: 'choose_territory_option', to: { x: 0, y: 0 } };

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
  const hint = moveHint ? normalizeLegacyMove(moveHint) : undefined;
  // For bookkeeping moves and skip_placement, use the move's player instead of
  // currentPlayer because these moves may be recorded at turn boundaries where
  // the state's currentPlayer hasn't been updated yet.
  const isBookkeepingOrSkipMove =
    hint?.type === 'skip_placement' ||
    hint?.type === 'no_placement_action' ||
    hint?.type === 'no_movement_action' ||
    hint?.type === 'no_line_action' ||
    hint?.type === 'no_territory_action';

  // Trust the move hint's player for moves where parity divergence may cause
  // TS and Python to have different current players. This includes:
  // - Bookkeeping moves (existing behavior)
  // - Territory region processing (Python may detect more regions than TS)
  // - Territory elimination (eliminate_rings_from_stack after choose_territory_option)
  // - Forced elimination (Python records the player whose rings are eliminated,
  //   which may differ from the current turn's player in multiplayer games)
  // RR-CANON-R075: Trust recorded moves during replay.
  const isTerritoryRegionMove =
    hint?.type === 'choose_territory_option' || hint?.type === 'eliminate_rings_from_stack';
  const isForcedEliminationMove = hint?.type === 'forced_elimination';
  const player =
    (isBookkeepingOrSkipMove || isTerritoryRegionMove || isForcedEliminationMove) && hint?.player
      ? hint.player
      : gameState.currentPlayer;

  // When the move hint indicates a specific phase, trust it over the game state's phase.
  // This handles parity issues where Python may detect more disconnected regions than TS,
  // causing TS to transition out of territory_processing before all Python-recorded moves
  // are replayed. RR-CANON-R075: Trust recorded moves during replay.
  let phase = gameState.currentPhase;
  if (
    hint?.type === 'place_ring' ||
    hint?.type === 'no_placement_action' ||
    hint?.type === 'skip_placement'
  ) {
    // RR-CANON-R073/R075: Placement moves are always in ring_placement phase.
    // This handles replay scenarios where the previous player's turn ended at a different phase
    // (e.g., line_processing or territory_processing) and the state wasn't properly transitioned
    // before the next player's placement move is validated.
    phase = 'ring_placement';
  } else if (hint?.type === 'choose_territory_option') {
    phase = 'territory_processing';
  } else if (hint?.type === 'eliminate_rings_from_stack') {
    // RR-CANON-R123: Check eliminationContext to determine correct phase.
    // Line-context eliminations belong to line_processing, territory-context to territory_processing.
    // TypeScript doesn't narrow the Move type to include eliminationContext after type check,
    // so we access it via type assertion on the object with eliminationContext property.
    const elimContext = (
      hint as { eliminationContext?: 'line' | 'territory' | 'forced' | 'recovery' }
    ).eliminationContext;
    if (elimContext === 'line') {
      phase = 'line_processing';
    } else {
      // Default to territory_processing for backwards compatibility with historical games
      // that didn't record eliminationContext or used it for territory claims.
      phase = 'territory_processing';
    }
  } else if (hint?.type === 'choose_line_option' || hint?.type === 'no_line_action') {
    phase = 'line_processing';
  } else if (hint?.type === 'forced_elimination') {
    phase = 'forced_elimination';
  }

  switch (phase) {
    case 'ring_placement':
      return deriveRingPlacementState(gameState, player, hint);

    case 'movement':
      return deriveMovementState(gameState, player, hint);

    case 'capture':
      return deriveCaptureState(gameState, player, false, hint);

    case 'chain_capture':
      return deriveChainCaptureState(gameState, player, hint);

    case 'line_processing':
      return deriveLineProcessingState(gameState, player, hint);

    case 'territory_processing':
      return deriveTerritoryProcessingState(gameState, player, hint);

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

  // RR-CANON-R075: Trust recorded placement moves during replay.
  // If Python recorded place_ring at a specific position, TS should trust it even
  // if local enumeration doesn't find the position as valid (parity divergence).
  // IMPORTANT: Only trust positions that are within board bounds - out-of-bounds
  // positions are definitively invalid regardless of parity considerations.
  const boardType = state.board.type as BoardType;
  const boardSize = BOARD_CONFIGS[boardType]?.size ?? 8;
  const isPositionInBounds = (pos: Position | undefined): boolean => {
    if (!pos) return false;
    return isValidPosition(pos, boardType, boardSize);
  };
  const isRecordedPlacementMoveForPlayer =
    moveHint?.type === 'place_ring' &&
    moveHint.player === player &&
    isPositionInBounds(moveHint.to);

  // Special handling for skip_placement and no_placement_action moves:
  // - skip_placement: Player CHOOSES not to place (has rings, has stacks with legal moves)
  // - no_placement_action: Player CANNOT place (no valid positions or no rings)
  //
  // RR-FIX-2026-01-19: During LIVE validation (currentPlayer === player), we must
  // strictly verify no_placement_action - it should ONLY be valid when no placements
  // exist. The bug was that the AI could pass (no_placement_action) when valid
  // placements existed because the hint was blindly trusted.
  //
  // skip_placement is different: the PlacementAggregate's evaluateSkipPlacementEligibility
  // allows it when the player has stacks with legal moves, giving players the CHOICE
  // not to place. We continue to trust skip_placement hints to maintain this behavior.
  if (moveHint?.type === 'no_placement_action') {
    if (state.currentPlayer === player) {
      // Live validation: verify no valid placements exist
      const validMoves = getValidMoves(state);
      const placementMoves = validMoves.filter((m) => m.type === 'place_ring');
      validPositions = placementMoves.map((m) => m.to);
      canPlace = placementMoves.length > 0;
    } else {
      // Replay mode: trust the recorded move (parity compatibility)
      canPlace = false;
    }
  } else if (moveHint?.type === 'skip_placement') {
    // skip_placement is a player CHOICE - trust the aggregate's eligibility check
    // which allows it when the player has stacks with legal moves/captures.
    // Set canPlace = false so the FSM guard passes (allowing the skip).
    canPlace = false;
  } else if (isRecordedPlacementMoveForPlayer) {
    // Trust the recorded placement move - position was valid when Python recorded it
    canPlace = true;
    if (moveHint.to) {
      validPositions = [moveHint.to];
    }
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

  // RR-CANON-R075: Trust recorded movement moves during replay.
  // If Python recorded movement/capture moves (including legacy aliases that have been normalized),
  // TS should trust them even if local enumeration doesn't find the move (parity divergence timing).
  const isRecordedMovementMoveForPlayer =
    moveHint &&
    moveHint.player === player &&
    (moveHint.type === 'move_stack' ||
      moveHint.type === 'overtaking_capture' ||
      moveHint.type === 'continue_capture_segment' ||
      moveHint.type === 'recovery_slide' ||
      moveHint.type === 'skip_recovery');

  let canMove: boolean;
  const recoveryEligible = isEligibleForRecovery(state, player);
  const isRecordedRecoverySlideForPlayer =
    moveHint?.type === 'recovery_slide' && moveHint.player === player;
  let recoveryMovesAvailable: boolean = false;

  if (state.currentPhase === 'movement' && state.currentPlayer === player) {
    if (isRecordedNoMovementForPlayer) {
      // RR‑CANON‑R075: Trust recorded bookkeeping moves during replay. A
      // canonical NO_MOVEMENT_ACTION implies there were no legal movement,
      // capture, or recovery moves for this player in MOVEMENT.
      canMove = false;
      recoveryMovesAvailable = false;
    } else if (isRecordedMovementMoveForPlayer) {
      // RR-CANON-R075: Trust recorded movement moves during replay.
      // If Python recorded a movement move, there must have been valid moves.
      canMove = true;
      // Trust recorded recovery_slide even if local enumeration doesn't find it.
      recoveryMovesAvailable = isRecordedRecoverySlideForPlayer;
    } else {
      const validMoves = getValidMoves(state);
      const movementLike = validMoves.filter(
        (m) =>
          m.player === player &&
          (m.type === 'move_stack' ||
            m.type === 'overtaking_capture' ||
            m.type === 'continue_capture_segment' ||
            m.type === 'recovery_slide' ||
            m.type === 'skip_recovery')
      );
      canMove = movementLike.length > 0;
      recoveryMovesAvailable = validMoves.some(
        (m) => m.player === player && m.type === 'recovery_slide'
      );
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
    recoveryEligible,
    recoveryMovesAvailable,
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

  const pendingCaptures = captureMoves
    .filter((m): m is typeof m & { captureTarget: Position } => m.captureTarget != null)
    .map((m) => ({
      target: m.captureTarget,
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

  const availableContinuations = continuationMoves
    .filter((m): m is typeof m & { captureTarget: Position } => m.captureTarget != null)
    .map((m) => ({
      target: m.captureTarget,
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
  // This mirrors the forced_elimination and territory-processing trust patterns.
  if (moveHint?.type === 'no_line_action' && detectedLines.length > 0) {
    return {
      phase: 'line_processing',
      player,
      detectedLines: [], // Trust Python's decision that no lines existed
      currentLineIndex: 0,
      awaitingReward: false,
    };
  }

  // RR-CANON-R075: Trust recorded process_line moves during replay.
  // If Python recorded process_line but TS doesn't detect lines (parity divergence due to
  // line detection timing or state differences), create a placeholder line from moveHint.
  // This mirrors the territory-processing trust pattern at lines 632-638.
  if (moveHint?.type === 'process_line' && detectedLines.length === 0) {
    // Use formedLines from moveHint if available, otherwise create placeholder
    if (moveHint.formedLines && moveHint.formedLines.length > 0) {
      const hintLine = moveHint.formedLines[0];
      detectedLines.push({
        positions: hintLine.positions,
        player: hintLine.player ?? player,
        requiresChoice: (hintLine.length ?? hintLine.positions.length) >= 3,
      });
    } else {
      // Fallback placeholder if formedLines not available
      detectedLines.push({
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
          { x: 2, y: 2 },
        ],
        player,
        requiresChoice: true,
      });
    }
  }

  // Check for pending choice from move context
  const validMoves = getValidMoves(state);
  const hasRewardChoice = validMoves.some((m) => m.type === 'choose_line_option');

  // RR-CANON-R123: Detect pending line elimination from moveHint.
  // If the incoming move is eliminate_rings_from_stack with eliminationContext='line',
  // set pendingLineRewardElimination so the FSM allows this move.
  // TypeScript doesn't narrow the Move type to include eliminationContext after type check,
  // so we access it via type assertion on the object with eliminationContext property.
  const pendingLineRewardElimination =
    moveHint?.type === 'eliminate_rings_from_stack' &&
    (moveHint as { eliminationContext?: 'line' | 'territory' | 'forced' | 'recovery' })
      .eliminationContext === 'line';

  // RR-CANON-R075: Trust recorded choose_line_option moves during replay.
  // If Python recorded these moves but TS doesn't detect lines, create placeholder line.
  if (moveHint?.type === 'choose_line_option' && detectedLines.length === 0) {
    if (moveHint.formedLines && moveHint.formedLines.length > 0) {
      const hintLine = moveHint.formedLines[0];
      detectedLines.push({
        positions: hintLine.positions,
        player: hintLine.player ?? player,
        requiresChoice: true,
      });
    } else {
      // Fallback placeholder
      detectedLines.push({
        positions: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
          { x: 2, y: 2 },
        ],
        player,
        requiresChoice: true,
      });
    }
  }

  return {
    phase: 'line_processing',
    player,
    detectedLines,
    currentLineIndex: 0,
    awaitingReward: hasRewardChoice || moveHint?.type === 'choose_line_option',
    pendingLineRewardElimination,
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

  // If the move being validated is a choose_territory_option but disconnectedRegions is empty,
  // add a placeholder region. This handles state timing issues during replay/shadow validation
  // where Python may detect more disconnected regions than TS (parity divergence).
  // RR-CANON-R075: Trust recorded territory-processing moves during replay.
  if (moveHint?.type === 'choose_territory_option' && disconnectedRegions.length === 0) {
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

  let eliminationsPending = elimMoves.map((m) => ({
    position: m.to,
    player: m.player,
    count: m.eliminatedRings?.[0]?.count ?? 1,
  }));

  // RR-FIX-2026-01-10: Trust eliminate_rings_from_stack moves during validation.
  // When the moveHint is an eliminate_rings_from_stack move with territory/recovery context,
  // include it in eliminationsPending even if getValidMoves doesn't return it.
  //
  // Background: The decision loop in processTurnAsync resolves decisions inline before
  // moveHistory is updated. When FSM validates the elimination move, getValidMoves relies
  // on moveHistory to find the processed region (via getPendingTerritorySelfEliminationRegion),
  // but the choose_territory_option move isn't in history yet. This causes getValidMoves to
  // return empty, making eliminationsPending empty and FSM to reject the move.
  //
  // Fix: When the moveHint is an eliminate_rings_from_stack move, trust it as valid.
  // This mirrors RR-CANON-R075 trust patterns for replay compatibility.
  if (moveHint?.type === 'eliminate_rings_from_stack') {
    const elimContext = (
      moveHint as { eliminationContext?: 'line' | 'territory' | 'forced' | 'recovery' }
    ).eliminationContext;
    // Only trust territory/recovery context eliminations in territory_processing phase.
    // Line-context eliminations should be in line_processing phase (handled by phase derivation).
    if (elimContext === 'territory' || elimContext === 'recovery' || elimContext === undefined) {
      const hintInEliminations = eliminationsPending.some(
        (e) => e.position?.x === moveHint.to?.x && e.position?.y === moveHint.to?.y
      );
      if (!hintInEliminations) {
        // Add the move hint as a pending elimination
        eliminationsPending = [
          ...eliminationsPending,
          {
            position: moveHint.to,
            player: moveHint.player,
            count: moveHint.eliminatedRings?.[0]?.count ?? 1,
          },
        ];
      }
    }
  }

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
    ringsPerPlayer: getEffectiveRingsPerPlayer(state.boardType, state.rulesOptions),
    lineLength: getEffectiveLineLengthThreshold(
      state.boardType,
      state.players.length,
      state.rulesOptions
    ),
  };
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
 * @param options Optional flags (e.g., replayCompatibility for legacy replays)
 * @returns FSM validation result with phase context
 */
export interface FSMValidationOptions {
  /**
   * Allow legacy replay move/phase coercions to pass validation.
   * Use only for replay/parity tooling (RR-CANON-R075).
   */
  replayCompatibility?: boolean;
}

export function validateMoveWithFSM(
  gameState: GameState,
  move: Move,
  includeDebugContext = false,
  options?: FSMValidationOptions
): FSMValidationResult {
  const replayCompatibility = options?.replayCompatibility ?? false;
  const moveForValidation = replayCompatibility ? normalizeLegacyMove(move) : move;

  // Derive FSM state and context from game state
  // Pass move as hint to help state derivation include relevant context
  const fsmState = deriveStateFromGame(gameState, moveForValidation);
  const gameContext = deriveGameContext(gameState);

  // Convert move to FSM event (do this early for debug context)
  const event = moveToEvent(moveForValidation);

  // Build debug context if logging is enabled or requested
  const debugContext =
    FSM_DEBUG_LOGGER || includeDebugContext
      ? buildDebugContext(gameState, moveForValidation, fsmState, gameContext, event)
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

  // Validate player attribution: in canonical mode the move must be from the
  // current player. Replay compatibility trusts recorded moves (RR-CANON-R075),
  // including legacy turn-transition mismatches handled below.
  const isPlayerMismatchFromDifferentPlayer = move.player !== gameState.currentPlayer;

  // IMPORTANT: For phase validation, we must distinguish between:
  // 1. Normal validation: check against the ORIGINAL gameState.currentPhase
  // 2. Replay compatibility: check against fsmState.phase (which may be overridden)
  //
  // The deriveStateFromGame function overrides the phase for replay compatibility
  // (e.g., place_ring forces ring_placement phase). In normal mode, we should
  // reject moves that don't match the actual game state phase.
  const originalPhaseForCheck =
    gameState.currentPhase !== 'game_over' ? (gameState.currentPhase as GamePhase) : null;
  const phaseForMoveCheck =
    fsmState.phase !== 'turn_end' && fsmState.phase !== 'game_over'
      ? (fsmState.phase as GamePhase)
      : null;

  // In normal (non-replay) mode, validate against the original game state phase
  const isCanonicalMoveForOriginalPhase = originalPhaseForCheck
    ? isMoveValidInPhase(moveForValidation.type as MoveType, originalPhaseForCheck)
    : false;
  const isCanonicalMoveForPhase = phaseForMoveCheck
    ? isMoveValidInPhase(moveForValidation.type as MoveType, phaseForMoveCheck)
    : false;
  const isLegacyMoveForPhase = phaseForMoveCheck
    ? isLegacyMoveValidInPhase(move.type as MoveType, phaseForMoveCheck)
    : false;

  // Legacy replay: next player's move arriving in various phases
  // Any move from a different player when FSM is in a post-move phase indicates
  // a turn transition that the FSM didn't process
  const isLegacyTurnTransition =
    replayCompatibility &&
    isPlayerMismatchFromDifferentPlayer &&
    !isCanonicalMoveForPhase &&
    isLegacyMoveForPhase &&
    (fsmState.phase === 'forced_elimination' ||
      fsmState.phase === 'movement' ||
      fsmState.phase === 'capture' ||
      fsmState.phase === 'chain_capture' ||
      fsmState.phase === 'line_processing' ||
      fsmState.phase === 'territory_processing');

  // For legacy turn transitions, skip FSM validation entirely and trust the recorded move.
  // Legacy phase validation covers these patterns; canonical validation does not.
  if (isLegacyTurnTransition) {
    return makeResult({
      valid: true,
      currentPhase: fsmState.phase,
      // Note: This is a legacy replay pattern - the FSM state is stale
    });
  }

  // For legacy replay, trust the recorded player attribution (RR-CANON-R075)
  if (!replayCompatibility && move.player !== gameState.currentPlayer) {
    return makeResult({
      valid: false,
      currentPhase: fsmState.phase,
      errorCode: 'WRONG_PLAYER',
      reason: `Not your turn (expected player ${gameState.currentPlayer}, got ${move.player})`,
    });
  }

  // In non-replay mode, reject moves that are invalid for the ORIGINAL game state phase.
  // This prevents the phase override logic (intended for replay compatibility) from
  // allowing moves that shouldn't be valid in the current phase.
  if (!replayCompatibility && originalPhaseForCheck && !isCanonicalMoveForOriginalPhase) {
    return makeResult({
      valid: false,
      currentPhase: fsmState.phase,
      errorCode: 'INVALID_EVENT',
      reason: `Move type '${move.type}' is not valid for phase '${originalPhaseForCheck}'`,
      validEventTypes: getExpectedEventTypes(fsmState),
    });
  }

  if (phaseForMoveCheck && !isCanonicalMoveForPhase) {
    if (!replayCompatibility || !isLegacyMoveForPhase) {
      return makeResult({
        valid: false,
        currentPhase: fsmState.phase,
        errorCode: 'INVALID_EVENT',
        reason: `Move type '${move.type}' is not valid for phase '${phaseForMoveCheck}'`,
        validEventTypes: getExpectedEventTypes(fsmState),
      });
    }
  }

  // Check event conversion
  if (!event) {
    // Meta-moves (swap_sides) don't have FSM events.
    // If the move type is explicitly permitted for the current phase, let it through -
    // the orchestrator handles meta-moves specially.
    //
    // TurnState['phase'] includes internal pseudo-phases such as 'turn_end' that are
    // not part of the public GamePhase contract. Only call isMoveTypeValidForPhase
    // when we are in a real GamePhase to keep types sound and avoid leaking
    // pseudo-phases into the shared phase↔MoveType mapping.
    if (fsmState.phase !== 'game_over' && fsmState.phase !== 'turn_end') {
      const phaseForMoveCheck = fsmState.phase as GamePhase;
      if (isMoveTypeValidForPhase(phaseForMoveCheck, moveForValidation.type as MoveType)) {
        return makeResult({
          valid: true,
          currentPhase: fsmState.phase,
        });
      }
      if (replayCompatibility && isLegacyMoveForPhase) {
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
    // For legacy replay, trust recorded moves even if FSM guards fail.
    // The move type validation above already ensures the move is valid for the phase.
    if (replayCompatibility && (isCanonicalMoveForPhase || isLegacyMoveForPhase)) {
      return makeResult({
        valid: true,
        currentPhase: fsmState.phase,
        // Note: FSM guards failed but move is trusted for legacy replay
      });
    }

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
  movement: [
    'MOVE_STACK',
    'CAPTURE',
    'RECOVERY_SLIDE',
    'SKIP_RECOVERY',
    'NO_MOVEMENT_ACTION',
    'RESIGN',
    'TIMEOUT',
  ],
  capture: ['CAPTURE', 'END_CHAIN', 'RESIGN', 'TIMEOUT'],
  chain_capture: ['CONTINUE_CHAIN', 'END_CHAIN', 'RESIGN', 'TIMEOUT'],
  // RR-CANON-R123: ELIMINATE_FROM_STACK is valid in line_processing for line-reward eliminations.
  line_processing: [
    'PROCESS_LINE',
    'CHOOSE_LINE_REWARD',
    'ELIMINATE_FROM_STACK',
    'NO_LINE_ACTION',
    'RESIGN',
    'TIMEOUT',
  ],
  territory_processing: [
    'PROCESS_REGION',
    'ELIMINATE_FROM_STACK',
    'NO_TERRITORY_ACTION',
    'RESIGN',
    'TIMEOUT',
  ],
  // Also allow ELIMINATE_FROM_STACK and NO_TERRITORY_ACTION for backwards compatibility
  // with historical game logs that recorded these moves during forced_elimination.
  forced_elimination: [
    'FORCED_ELIMINATE',
    'ELIMINATE_FROM_STACK',
    'NO_TERRITORY_ACTION',
    'RESIGN',
    'TIMEOUT',
  ],
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
 * Get the set of canonical MoveType values allowed for a given GamePhase.
 */
export function getAllowedMoveTypesForPhase(phase: GamePhase): ReadonlyArray<MoveType> {
  return VALID_MOVES_BY_PHASE[phase] ?? [];
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
  return isMoveValidInPhase(moveType, phase);
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
 * This is the FSM-based function for determining phase progressions
 * using the TurnStateMachine's transition rules.
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

  /**
   * RR-CANON-R123: Pending line elimination after choose_line_option with 'eliminate' choice.
   * When true, the player must execute eliminate_rings_from_stack before territory_processing.
   */
  pendingLineElimination?: boolean;
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
    | 'line_elimination_required'
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
  if (move.type === 'move_stack' && move.to) {
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
        .filter(
          (m): m is typeof m & { captureTarget: Position } =>
            m.from != null && positionToString(m.from) === landingKey && m.captureTarget != null
        )
        .map((m) => ({
          target: m.captureTarget,
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

  // NOTE: We no longer force turn_end after choose_territory_option.
  // Per RR-CANON-R145, the orchestrator now properly handles pending self-elimination
  // decisions by returning a pendingDecision when the player must eliminate from a
  // stack OUTSIDE the processed region. The FSM should remain in territory_processing
  // until all eliminations are complete. Removed the automatic turn_end transition
  // that was incorrectly assuming internal eliminations were the only eliminations.

  // Handle no_territory_action: this also ends the turn and advances to next player
  // The FSM may return ring_placement directly, but we need to ensure correct next player
  // computation and phase resolution based on that player's ringsInHand.
  if (move.type === 'no_territory_action' && nextState.phase !== 'turn_end') {
    // no_territory_action ends the turn - compute next player and transition to turn_end
    // so the ringsInHand check below can properly resolve to ring_placement or movement
    // Use computeNextNonEliminatedPlayer to skip permanently eliminated players (RR-CANON-R201)
    const computedNextPlayer = computeNextNonEliminatedPlayer(
      gameState,
      move.player,
      context.numPlayers
    );
    nextState = {
      phase: 'turn_end',
      completedPlayer: move.player,
      nextPlayer: computedNextPlayer,
    } as TurnEndState;
  }

  // Handle forced_elimination: when FSM transitions to turn_end, the pure FSM uses
  // simple modular rotation which doesn't skip eliminated players. Recompute the
  // next player using computeNextNonEliminatedPlayer to match Python's FSM behavior.
  // RR-PARITY-FIX-2025-12-13: Fixes 4-player games where forced_elimination would
  // incorrectly rotate to eliminated players instead of skipping them.
  if (move.type === 'forced_elimination' && nextState.phase === 'turn_end') {
    const computedNextPlayer = computeNextNonEliminatedPlayer(
      gameState,
      move.player,
      context.numPlayers
    );
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
    // Per RR-CANON-R073: ALL players start in ring_placement without exception.
    // NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action.
    nextPhase = 'ring_placement';
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
    } else if (lineState.pendingLineRewardElimination) {
      // RR-CANON-R123: After choose_line_option with 'eliminate' choice,
      // the player must execute a separate eliminate_rings_from_stack move.
      // Surface a decision so the orchestrator knows not to transition to territory_processing.
      pendingDecisionType = 'line_elimination_required';
      decisionSurface = {
        pendingLines: [],
        pendingRegions: [],
        chainContinuations: [],
        forcedEliminationCount: 0,
        // Include flag for orchestrator to surface the elimination decision
        pendingLineElimination: true,
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
    moveType === 'choose_line_option'
  );
}

/**
 * Check if a move type is a territory-phase move.
 */
function isTerritoryPhaseMove(moveType: Move['type']): boolean {
  return (
    moveType === 'no_territory_action' ||
    moveType === 'choose_territory_option' ||
    moveType === 'skip_territory_processing'
  );
}

/**
 * Compare FSM orchestration result with legacy orchestration result.
 * Returns divergence details if they differ.
 *
 * Per RR-CANON-R073: ALL players start in ring_placement without exception.
 * NO PHASE SKIPPING - players with ringsInHand == 0 will emit no_placement_action
 * which triggers the transition to movement. The engine MUST NOT skip directly
 * to movement phase based on ringsInHand count.
 *
 * When FSM returns `turn_end`, the next player's starting phase is ALWAYS
 * ring_placement (no phase skipping allowed).
 *
 * @param fsmResult - The FSM orchestration result
 * @param legacyPhase - The phase from the legacy/Python orchestration
 * @param legacyPlayer - The player from the legacy/Python orchestration
 * @param nextPlayerRingsInHand - Optional: next player's rings in hand (retained for API compatibility)
 */
export function compareFSMWithLegacy(
  fsmResult: FSMOrchestrationResult,
  legacyPhase: string,
  legacyPlayer: number,
  _nextPlayerRingsInHand?: number
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
  // Per RR-CANON-R073: ALL players start in ring_placement without exception.
  // NO PHASE SKIPPING - turn_end always maps to ring_placement.
  let effectiveFSMPhase: string;
  if (fsmResult.nextPhase === 'turn_end') {
    // NO PHASE SKIPPING - always ring_placement regardless of ringsInHand
    effectiveFSMPhase = 'ring_placement';
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
