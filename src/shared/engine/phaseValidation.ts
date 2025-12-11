/**
 * Phase-Move Validation Matrix
 *
 * This module defines a declarative mapping of which move types are valid
 * in which game phases. This serves as:
 *
 * 1. Single source of truth for phase/move compatibility
 * 2. Quick lookup without FSM machinery
 * 3. Documentation of the phase/move relationship
 *
 * The FSM (TurnStateMachine) still handles the detailed transition logic,
 * but this matrix provides a fast, declarative check.
 *
 * @see TurnStateMachine for the canonical FSM implementation
 * @see RULES_CANONICAL_SPEC.md for the underlying rules
 */

import type { GamePhase } from '../types/game';

// ═══════════════════════════════════════════════════════════════════════════
// MOVE TYPES - Canonical list of all move types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * All valid move types in the game.
 *
 * These correspond to the `Move.type` field and the FSM events.
 */
export type MoveType =
  // Ring Placement Phase
  | 'place_ring'
  | 'skip_placement'
  | 'no_placement_action'
  // Movement Phase
  | 'move_stack'
  | 'move_ring'
  | 'no_movement_action'
  // Capture (initiated from movement)
  | 'overtaking_capture'
  | 'continue_capture_segment'
  | 'end_capture_chain'
  // Recovery (RR-CANON-R110–R115)
  | 'recovery_slide'
  | 'skip_recovery'
  // Line Processing Phase
  | 'process_line'
  | 'choose_line_reward'
  | 'no_line_action'
  // Territory Processing Phase
  | 'process_territory_region'
  | 'eliminate_rings_from_stack'
  | 'no_territory_action'
  | 'skip_territory_processing'
  // Forced Elimination Phase
  | 'forced_elimination'
  // Game Meta
  | 'resign'
  | 'timeout';

// ═══════════════════════════════════════════════════════════════════════════
// PHASE-MOVE MATRIX
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Declarative matrix mapping phases to valid move types.
 *
 * Rules derivation:
 * - ring_placement: place_ring, skip_placement, no_placement_action, recovery_slide, skip_recovery
 *   (recovery can happen during placement phase when player has buried rings)
 * - movement: move_stack, move_ring, no_movement_action, overtaking_capture
 *   (capture is initiated from movement)
 * - capture: continue_capture_segment, end_capture_chain
 * - chain_capture: continue_capture_segment, end_capture_chain
 * - line_processing: process_line, choose_line_reward, no_line_action, eliminate_rings_from_stack
 *   (elimination is part of line reward)
 * - territory_processing: process_territory_region, eliminate_rings_from_stack, no_territory_action, skip_territory_processing
 * - forced_elimination: forced_elimination, eliminate_rings_from_stack
 * - game_over: none (game ended)
 *
 * Note: Some moves can span multiple phases due to phase coercion in turnOrchestrator.
 * This matrix represents the PRIMARY valid phase for each move type.
 */
export const VALID_MOVES_BY_PHASE: Readonly<Record<GamePhase, readonly MoveType[]>> = {
  ring_placement: [
    'place_ring',
    'skip_placement',
    'no_placement_action',
    'recovery_slide',
    'skip_recovery',
    'forced_elimination', // Can trigger from ring_placement when ANM detected
  ],
  movement: [
    'move_stack',
    'move_ring',
    'no_movement_action',
    'overtaking_capture', // Capture is initiated from movement phase
  ],
  capture: ['overtaking_capture', 'continue_capture_segment', 'end_capture_chain'],
  chain_capture: ['continue_capture_segment', 'end_capture_chain'],
  line_processing: [
    'process_line',
    'choose_line_reward',
    'no_line_action',
    'eliminate_rings_from_stack', // Line reward option 1
  ],
  territory_processing: [
    'process_territory_region',
    'eliminate_rings_from_stack', // Self-elimination for territory
    'no_territory_action',
    'skip_territory_processing',
  ],
  forced_elimination: ['forced_elimination', 'eliminate_rings_from_stack'],
  game_over: [],
} as const;

/**
 * Moves that are always valid regardless of phase (meta moves).
 */
export const ALWAYS_VALID_MOVES: readonly MoveType[] = ['resign', 'timeout'] as const;

// ═══════════════════════════════════════════════════════════════════════════
// VALIDATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a move type is valid in the given phase.
 *
 * This is a fast, O(1) lookup that doesn't require FSM state construction.
 * Use this for quick validation before invoking the full FSM transition.
 *
 * @param moveType The move type to check
 * @param phase The current game phase
 * @returns true if the move type is valid in this phase
 */
export function isMoveValidInPhase(moveType: MoveType, phase: GamePhase): boolean {
  // Meta moves are always valid
  if ((ALWAYS_VALID_MOVES as readonly string[]).includes(moveType)) {
    return true;
  }

  const validMoves = VALID_MOVES_BY_PHASE[phase];
  if (!validMoves) {
    return false;
  }

  return (validMoves as readonly string[]).includes(moveType);
}

/**
 * Get all valid move types for a given phase.
 *
 * @param phase The game phase
 * @returns Array of valid move types (including always-valid moves)
 */
export function getValidMoveTypesForPhase(phase: GamePhase): readonly MoveType[] {
  const phaseMoves = VALID_MOVES_BY_PHASE[phase] ?? [];
  return [...phaseMoves, ...ALWAYS_VALID_MOVES];
}

/**
 * Get the primary phase(s) where a move type is valid.
 *
 * @param moveType The move type to look up
 * @returns Array of phases where this move is valid
 */
export function getPhasesForMoveType(moveType: MoveType): GamePhase[] {
  if ((ALWAYS_VALID_MOVES as readonly string[]).includes(moveType)) {
    // Meta moves are valid in all phases except game_over
    return [
      'ring_placement',
      'movement',
      'capture',
      'chain_capture',
      'line_processing',
      'territory_processing',
      'forced_elimination',
    ];
  }

  const phases: GamePhase[] = [];
  for (const [phase, moves] of Object.entries(VALID_MOVES_BY_PHASE)) {
    if ((moves as readonly string[]).includes(moveType)) {
      phases.push(phase as GamePhase);
    }
  }
  return phases;
}

// ═══════════════════════════════════════════════════════════════════════════
// ELIMINATION CONTEXT HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Infer the elimination context from the current phase.
 *
 * This maps the game phase to the appropriate EliminationContext
 * for the EliminationAggregate functions.
 *
 * @param phase The current game phase
 * @returns The appropriate elimination context, or null if elimination not applicable
 */
export function getEliminationContextForPhase(
  phase: GamePhase
): 'line' | 'territory' | 'forced' | 'recovery' | null {
  switch (phase) {
    case 'line_processing':
      return 'line';
    case 'territory_processing':
      return 'territory';
    case 'forced_elimination':
      return 'forced';
    case 'ring_placement':
      // Recovery can happen during placement phase
      return 'recovery';
    default:
      return null;
  }
}

/**
 * Determine if elimination moves are valid in the current phase.
 *
 * @param phase The current game phase
 * @returns true if elimination moves are valid
 */
export function canEliminateInPhase(phase: GamePhase): boolean {
  return (
    phase === 'line_processing' ||
    phase === 'territory_processing' ||
    phase === 'forced_elimination'
  );
}
