/**
 * Legacy Phase Validation for Replay Compatibility
 *
 * This module extends the canonical phase-move validation to support replaying
 * games recorded before RR-PARITY-FIX-2025-12-20. These games may have phase
 * sequences that diverge from the canonical spec due to orchestrator changes.
 *
 * IMPORTANT: This module should ONLY be used for replay compatibility.
 * New games and validation should use the canonical phaseValidation module.
 *
 * @see phaseValidation.ts for the canonical phase-move matrix
 * @see legacyReplayHelper.ts for the coercion patterns used during replay
 */

import type { GamePhase, MoveType as GameMoveType } from '../types/game';

export type MoveType = GameMoveType;

/**
 * Extended phase-move matrix for legacy replay compatibility.
 *
 * These extensions allow moves that were valid in pre-RR-PARITY-FIX-2025-12-20
 * recordings but are no longer canonical.
 *
 * Coercion patterns from legacyReplayHelper.ts:
 * - Pattern 1: FORCED_ELIM_TO_TERRITORY - territory moves in forced_elimination
 * - Pattern 2: FORCED_ELIM_TO_LINE - line moves in forced_elimination
 * - Pattern 3: FORCED_ELIM_TO_PLACEMENT - placement moves in forced_elimination
 */
export const LEGACY_VALID_MOVES_BY_PHASE: Readonly<Record<GamePhase, readonly MoveType[]>> = {
  ring_placement: ['place_ring', 'skip_placement', 'no_placement_action', 'swap_sides'],
  movement: [
    'move_stack',
    'move_ring',
    'build_stack',
    'no_movement_action',
    'overtaking_capture',
    'continue_capture_segment',
    'recovery_slide',
    'skip_recovery',
    'swap_sides',
  ],
  capture: ['overtaking_capture', 'continue_capture_segment', 'skip_capture', 'swap_sides'],
  chain_capture: ['overtaking_capture', 'continue_capture_segment', 'swap_sides'],
  line_processing: [
    'process_line',
    'choose_line_option',
    'choose_line_reward',
    'eliminate_rings_from_stack',
    'no_line_action',
    'line_formation',
    // Legacy compatibility: capture/movement moves from phase divergence
    'skip_capture',
    'overtaking_capture',
    'continue_capture_segment',
    'move_stack',
    'move_ring',
    'build_stack',
    'no_movement_action',
  ],
  territory_processing: [
    'choose_territory_option',
    'process_territory_region',
    'eliminate_rings_from_stack',
    'no_territory_action',
    'skip_territory_processing',
    'territory_claim',
  ],
  forced_elimination: [
    'forced_elimination',
    // Pattern 1: FORCED_ELIM_TO_TERRITORY
    'no_territory_action',
    'process_territory_region',
    'choose_territory_option',
    'eliminate_rings_from_stack',
    'skip_territory_processing',
    // Pattern 2: FORCED_ELIM_TO_LINE
    'no_line_action',
    'process_line',
    'choose_line_option',
    'choose_line_reward',
    // Pattern 3: FORCED_ELIM_TO_PLACEMENT
    'place_ring',
    'skip_placement',
    'no_placement_action',
  ],
  game_over: [],
} as const;

/**
 * Check if a move type is valid in the given phase for legacy replay.
 *
 * This uses the extended legacy validation matrix instead of canonical validation.
 *
 * @param moveType The move type to check
 * @param phase The current game phase
 * @returns true if the move type is valid in this phase (legacy compatible)
 */
export function isLegacyMoveValidInPhase(moveType: MoveType, phase: GamePhase): boolean {
  const validMoves = LEGACY_VALID_MOVES_BY_PHASE[phase];
  if (!validMoves) {
    return false;
  }
  return (validMoves as readonly string[]).includes(moveType);
}
