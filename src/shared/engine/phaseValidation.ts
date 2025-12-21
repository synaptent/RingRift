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

import type { GamePhase, MoveType as GameMoveType } from '../types/game';

// ═══════════════════════════════════════════════════════════════════════════
// MOVE TYPES - Canonical list of all move types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * MoveType surface used by phase validation.
 *
 * This is intentionally derived from the canonical shared type
 * (`src/shared/types/game.ts`) to avoid drift between validators and the
 * executable engine SSoT.
 */
export type MoveType = GameMoveType;

// ═══════════════════════════════════════════════════════════════════════════
// PHASE-MOVE MATRIX
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Declarative matrix mapping phases to valid move types.
 *
 * Rules derivation (canonical):
 * - ring_placement: place_ring, skip_placement, no_placement_action, swap_sides
 * - movement: move_stack, overtaking_capture, recovery_slide, skip_recovery, no_movement_action, swap_sides
 * - capture: overtaking_capture, skip_capture, swap_sides
 * - chain_capture: continue_capture_segment, swap_sides
 * - line_processing: process_line, choose_line_option, eliminate_rings_from_stack, no_line_action
 * - territory_processing: choose_territory_option, eliminate_rings_from_stack,
 *   no_territory_action, skip_territory_processing
 * - forced_elimination: forced_elimination
 * - game_over: none (game ended)
 *
 * Legacy replay extensions live in `legacy/legacyPhaseValidation.ts`.
 */
export const VALID_MOVES_BY_PHASE: Readonly<Record<GamePhase, readonly MoveType[]>> = {
  ring_placement: ['place_ring', 'skip_placement', 'no_placement_action', 'swap_sides'],
  movement: [
    'move_stack',
    'overtaking_capture',
    'recovery_slide',
    'skip_recovery',
    'no_movement_action',
    'swap_sides',
  ],
  capture: ['overtaking_capture', 'skip_capture', 'swap_sides'],
  chain_capture: ['continue_capture_segment', 'swap_sides'],
  line_processing: [
    'process_line',
    'choose_line_option',
    // RR-CANON-R123: line reward elimination uses eliminate_rings_from_stack
    'eliminate_rings_from_stack',
    'no_line_action',
  ],
  territory_processing: [
    'choose_territory_option',
    'eliminate_rings_from_stack', // Self-elimination for territory
    'no_territory_action',
    'skip_territory_processing',
  ],
  forced_elimination: ['forced_elimination'],
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
    case 'movement':
      // Recovery (recovery_slide) is a movement-phase action.
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
