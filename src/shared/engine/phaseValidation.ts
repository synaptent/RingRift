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

export type MoveType = GameMoveType;

export const VALID_MOVES_BY_PHASE: Readonly<Record<GamePhase, readonly MoveType[]>> = {
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
  capture: [
    'overtaking_capture',
    'continue_capture_segment',
    'skip_capture',
    'swap_sides',
  ],
  chain_capture: [
    'overtaking_capture',
    'continue_capture_segment',
    'swap_sides',
  ],
  line_processing: [
    'process_line',
    'choose_line_option',
    'choose_line_reward',
    'eliminate_rings_from_stack',
    'no_line_action',
    'line_formation',
  ],
  territory_processing: [
    'choose_territory_option',
    'process_territory_region',
    'eliminate_rings_from_stack',
    'no_territory_action',
    'skip_territory_processing',
    'territory_claim',
  ],
  forced_elimination: ['forced_elimination'],
  game_over: [],
} as const;

export const CANONICAL_VALID_MOVES_BY_PHASE: Readonly<Record<GamePhase, readonly MoveType[]>> = {
  ring_placement: ['place_ring', 'skip_placement', 'no_placement_action', 'swap_sides'],
  movement: [
    'move_stack',
    'move_ring',
    'build_stack',
    'overtaking_capture',
    'continue_capture_segment',
    'recovery_slide',
    'skip_recovery',
    'no_movement_action',
  ],
  capture: ['overtaking_capture', 'continue_capture_segment', 'skip_capture'],
  chain_capture: ['continue_capture_segment'],
  line_processing: [
    'process_line',
    'choose_line_option',
    'eliminate_rings_from_stack',
    'no_line_action',
  ],
  territory_processing: [
    'choose_territory_option',
    'eliminate_rings_from_stack',
    'skip_territory_processing',
    'no_territory_action',
  ],
  forced_elimination: ['forced_elimination'],
  game_over: [],
} as const;

export const ALWAYS_VALID_MOVES: readonly MoveType[] = ['resign', 'timeout'] as const;

export function isMoveValidInPhase(moveType: MoveType, phase: GamePhase): boolean {
  if ((ALWAYS_VALID_MOVES as readonly string[]).includes(moveType)) {
    return true;
  }
  const validMoves = VALID_MOVES_BY_PHASE[phase];
  if (!validMoves) {
    return false;
  }
  return (validMoves as readonly string[]).includes(moveType);
}

export function getValidMoveTypesForPhase(phase: GamePhase): readonly MoveType[] {
  const phaseMoves = VALID_MOVES_BY_PHASE[phase] ?? [];
  return [...phaseMoves, ...ALWAYS_VALID_MOVES];
}

export function getPhasesForMoveType(moveType: MoveType): GamePhase[] {
  if ((ALWAYS_VALID_MOVES as readonly string[]).includes(moveType)) {
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
      return 'recovery';
    default:
      return null;
  }
}

export function canEliminateInPhase(phase: GamePhase): boolean {
  return (
    phase === 'line_processing' ||
    phase === 'territory_processing' ||
    phase === 'forced_elimination'
  );
}
