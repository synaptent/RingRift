/**
 * ---------------------------------------------------------------------------
 * Legacy Replay Helper
 * ---------------------------------------------------------------------------
 *
 * DEPRECATED: Phase coercion logic for legacy/non-canonical recordings.
 *
 * This module contains phase coercion patterns that were previously inline in
 * turnOrchestrator.ts. These patterns violate:
 * - RR-CANON-R073: NO phase skipping
 * - RR-CANON-R075: Every phase transition must be recorded as a move
 *
 * This code is quarantined here for:
 * - Historical analysis of legacy databases
 * - One-time migration scripts
 * - TS<->Python parity debugging of non-canonical recordings
 *
 * DO NOT use this code for:
 * - Training data generation
 * - Production gameplay
 * - New canonical recordings
 *
 * @deprecated Use check_canonical_phase_history.py to validate records.
 * Non-canonical records should be quarantined, not silently coerced.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 */

import type { GameState, GamePhase, Move, MoveType } from '../../types/game';

/**
 * Result of phase coercion detection.
 */
export interface PhaseCoercionResult {
  /** Whether coercion is needed */
  needsCoercion: boolean;
  /** The phase to coerce to (if coercion is needed) */
  targetPhase?: GamePhase;
  /** The player to coerce to (if player mismatch) */
  targetPlayer?: number;
  /** Description of the coercion for logging */
  reason?: string;
}

/**
 * Coercion pattern identifier for logging and analysis.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 */
export type CoercionPatternId =
  | 'FORCED_ELIM_TO_TERRITORY'
  | 'FORCED_ELIM_TO_LINE'
  | 'FORCED_ELIM_TO_PLACEMENT'
  | 'PLACEMENT_PLAYER_SKIP'
  | 'PLACEMENT_PHASE_TO_FORCED_ELIM'
  | 'LINE_TO_CAPTURE'
  | 'RING_PLACEMENT_TO_TERRITORY'
  | 'TERRITORY_TO_MOVEMENT'
  | 'MOVEMENT_TO_LINE';

/**
 * Detect if phase coercion would be needed for a legacy recording.
 *
 * This function identifies the 9 phase coercion patterns that were previously
 * inline in turnOrchestrator.ts. It does NOT apply the coercion - it only
 * detects whether coercion would be needed.
 *
 * @deprecated Use check_canonical_phase_history.py to validate records instead.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 *
 * @param state - Current game state
 * @param move - The move being applied
 * @returns Detection result with coercion details if needed
 */
export function detectPhaseCoercion(state: GameState, move: Move): PhaseCoercionResult {
  if (state.gameStatus !== 'active') {
    return { needsCoercion: false };
  }

  const currentPhase = state.currentPhase;
  const currentPlayer = state.currentPlayer;
  const moveType = move.type as MoveType;
  const movePlayer = move.player;

  // Pattern 1: forced_elimination -> territory_processing
  // When in forced_elimination but a territory move comes in
  if (
    currentPhase === 'forced_elimination' &&
    (moveType === 'process_territory_region' ||
      moveType === 'choose_territory_option' ||
      moveType === 'eliminate_rings_from_stack' ||
      moveType === 'skip_territory_processing' ||
      moveType === 'no_territory_action')
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'territory_processing',
      reason:
        '[COERCION-PATTERN-1] forced_elimination -> territory_processing for territory-type move',
    };
  }

  // Pattern 2: forced_elimination -> line_processing
  // When in forced_elimination but a line move comes in
  if (
    currentPhase === 'forced_elimination' &&
    (moveType === 'process_line' ||
      moveType === 'choose_line_option' ||
      moveType === 'choose_line_reward' ||
      moveType === 'no_line_action')
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'line_processing',
      reason: '[COERCION-PATTERN-2] forced_elimination -> line_processing for line-type move',
    };
  }

  // Pattern 3: forced_elimination -> ring_placement
  // When in forced_elimination but a placement move comes in
  if (
    currentPhase === 'forced_elimination' &&
    (moveType === 'place_ring' ||
      moveType === 'skip_placement' ||
      moveType === 'no_placement_action')
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'ring_placement',
      reason: '[COERCION-PATTERN-3] forced_elimination -> ring_placement for placement-type move',
    };
  }

  // Pattern 4: Player skip with phase coercion
  // When in territory_processing/movement/line_processing/ring_placement and a placement
  // move comes in for a different player
  if (
    (currentPhase === 'territory_processing' ||
      currentPhase === 'movement' ||
      currentPhase === 'line_processing' ||
      currentPhase === 'ring_placement') &&
    (moveType === 'place_ring' ||
      moveType === 'skip_placement' ||
      moveType === 'no_placement_action') &&
    movePlayer !== currentPlayer
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'ring_placement',
      targetPlayer: movePlayer,
      reason:
        '[COERCION-PATTERN-4] Player skip: Python advanced to next player, coercing phase and player',
    };
  }

  // Pattern 5: ring_placement/movement -> forced_elimination
  // When a forced_elimination move comes in while in placement/movement phase
  if (
    (currentPhase === 'ring_placement' || currentPhase === 'movement') &&
    moveType === 'forced_elimination'
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'forced_elimination',
      reason:
        '[COERCION-PATTERN-5] ring_placement/movement -> forced_elimination for forced_elimination move',
    };
  }

  // Pattern 6: line_processing -> capture/chain_capture
  // When a capture move comes in while in line_processing
  if (
    currentPhase === 'line_processing' &&
    (moveType === 'overtaking_capture' || moveType === 'continue_capture_segment')
  ) {
    const targetPhase = moveType === 'continue_capture_segment' ? 'chain_capture' : 'movement';
    return {
      needsCoercion: true,
      targetPhase,
      reason: `[COERCION-PATTERN-6] line_processing -> ${targetPhase} for capture move`,
    };
  }

  // Pattern 7: ring_placement/movement/line_processing -> territory_processing
  // When a territory move comes in while in an earlier phase
  if (
    (currentPhase === 'ring_placement' ||
      currentPhase === 'movement' ||
      currentPhase === 'line_processing') &&
    (moveType === 'choose_territory_option' ||
      moveType === 'process_territory_region' ||
      moveType === 'no_territory_action' ||
      moveType === 'eliminate_rings_from_stack' ||
      moveType === 'skip_territory_processing')
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'territory_processing',
      reason:
        '[COERCION-PATTERN-7] ring_placement/movement/line_processing -> territory_processing for territory move',
    };
  }

  // Pattern 8: territory_processing -> movement/chain_capture
  // When a movement/capture move comes in while in territory_processing
  if (
    currentPhase === 'territory_processing' &&
    (moveType === 'overtaking_capture' ||
      moveType === 'continue_capture_segment' ||
      moveType === 'move_stack' ||
      moveType === 'move_ring' ||
      moveType === 'no_movement_action')
  ) {
    const targetPhase =
      moveType === 'continue_capture_segment'
        ? 'chain_capture'
        : moveType === 'overtaking_capture' ||
            moveType === 'move_stack' ||
            moveType === 'move_ring' ||
            moveType === 'no_movement_action'
          ? 'movement'
          : currentPhase;
    const targetPlayer = movePlayer !== currentPlayer ? movePlayer : currentPlayer;
    return {
      needsCoercion: true,
      targetPhase,
      targetPlayer,
      reason: `[COERCION-PATTERN-8] territory_processing -> ${targetPhase} for movement/capture move`,
    };
  }

  // Pattern 9: movement -> line_processing
  // When a line move comes in while in movement phase
  if (
    currentPhase === 'movement' &&
    (moveType === 'no_line_action' ||
      moveType === 'process_line' ||
      moveType === 'choose_line_option' ||
      moveType === 'choose_line_reward')
  ) {
    return {
      needsCoercion: true,
      targetPhase: 'line_processing',
      reason: '[COERCION-PATTERN-9] movement -> line_processing for line move',
    };
  }

  return { needsCoercion: false };
}

/**
 * Apply phase coercion to game state for legacy replay.
 *
 * WARNING: This function mutates the logical phase/player state to allow
 * non-canonical moves to be replayed. This should ONLY be used for:
 * - Historical analysis of legacy databases
 * - One-time migration scripts
 * - TS<->Python parity debugging of non-canonical recordings
 *
 * @deprecated Use check_canonical_phase_history.py to validate records instead.
 * Non-canonical records should be quarantined, not silently coerced.
 *
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R073 (NO phase skipping)
 * @see RULES_CANONICAL_SPEC.md RR-CANON-R075 (Every phase transition recorded)
 *
 * @param state - Current game state (will be copied, not mutated)
 * @param coercion - The coercion to apply
 * @returns New state with coerced phase/player
 */
export function applyPhaseCoercion(state: GameState, coercion: PhaseCoercionResult): GameState {
  if (!coercion.needsCoercion) {
    return state;
  }

  const newState: GameState = { ...state };

  if (coercion.targetPhase) {
    newState.currentPhase = coercion.targetPhase;
  }

  if (coercion.targetPlayer !== undefined) {
    newState.currentPlayer = coercion.targetPlayer;
  }

  return newState;
}

/**
 * Log a phase coercion event for analysis.
 *
 * This function outputs structured JSON logs that can be collected for
 * analysis of non-canonical recording patterns in legacy databases.
 *
 * @deprecated
 */
export function logPhaseCoercion(
  gameId: string | undefined,
  moveNumber: number,
  coercion: PhaseCoercionResult,
  originalPhase: GamePhase,
  originalPlayer: number
): void {
  // Only log if coercion is needed
  if (!coercion.needsCoercion) {
    return;
  }

  console.warn(
    JSON.stringify({
      kind: 'legacy_phase_coercion',
      gameId,
      moveNumber,
      originalPhase,
      originalPlayer,
      targetPhase: coercion.targetPhase,
      targetPlayer: coercion.targetPlayer,
      reason: coercion.reason,
      timestamp: new Date().toISOString(),
    })
  );
}
