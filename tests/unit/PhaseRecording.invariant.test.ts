/**
 * INV-PHASE-RECORDING: Phase recording invariant tests per RR-CANON-R074/R075.
 *
 * These tests verify that:
 * - Every player has a recorded action for every turn phase for every turn
 * - This holds regardless of whether the player has any material (rings/stacks)
 * - Valid recorded actions include:
 *   - Actual actions (place_ring, move_stack, overtaking_capture, etc.)
 *   - Voluntary skips (skip_placement, skip_capture, skip_territory_processing)
 *   - No-action markers (no_placement_action, no_movement_action, no_line_action, no_territory_action)
 *   - Forced elimination (forced_elimination)
 *
 * Per RR-CANON-R074: Every phase must be visited
 * Per RR-CANON-R075: Every phase transition must produce a recorded action
 */

import type { Move, MoveType, GamePhase, GameState } from '../../src/shared/types/game';
import type { GameRecord } from '../../src/shared/types/gameRecord';

/**
 * All canonical turn phases per RR-CANON-R070.
 * Note: capture and chain_capture are optional phases that may be skipped.
 */
const CANONICAL_PHASES: GamePhase[] = [
  'ring_placement',
  'movement',
  'capture',
  'chain_capture',
  'line_processing',
  'territory_processing',
  'forced_elimination',
];

/**
 * Phases that must always have a recorded action when entered.
 * Capture phases are optional and may be skipped in phase transitions.
 */
const MANDATORY_RECORD_PHASES: GamePhase[] = [
  'ring_placement',
  'movement',
  'line_processing',
  'territory_processing',
  'forced_elimination',
];

/**
 * Map of phases to valid move types that can be recorded in that phase.
 *
 * Note: 'game_over' is a terminal pseudo-phase; no moves are recorded there,
 * so it is intentionally omitted from this mapping and treated as having
 * no valid move types.
 */
const PHASE_TO_VALID_MOVE_TYPES: Partial<Record<GamePhase, MoveType[]>> = {
  ring_placement: ['place_ring', 'skip_placement', 'no_placement_action', 'swap_sides'],
  movement: [
    'move_stack',
    'overtaking_capture',
    'no_movement_action',
    'recovery_slide',
    'skip_recovery',
  ],
  capture: ['overtaking_capture', 'skip_capture'],
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
};

/**
 * Interface for tracking phase records per player per turn.
 */
interface TurnPhaseRecords {
  turnNumber: number;
  playerNumber: number;
  phases: Map<GamePhase, Move[]>;
}

/**
 * Interface for phase recording violations.
 */
interface PhaseRecordingViolation {
  turnNumber: number;
  playerNumber: number;
  phase: GamePhase;
  reason: string;
}

/**
 * Local helper type for tests that annotate moves with an explicit phase.
 * This mirrors the optional .phase field sometimes present in historical
 * records while remaining assignable to the core Move type.
 */
type PhaseAnnotatedMove = Move & { phase?: GamePhase };

/**
 * Extract phase records from move history, organized by turn and player.
 */
function extractPhaseRecordsByTurn(moves: Move[]): TurnPhaseRecords[] {
  const records: TurnPhaseRecords[] = [];
  let currentTurn = 1;
  let currentPlayer = 1;
  let currentTurnRecord: TurnPhaseRecords | null = null;

  for (const move of moves) {
    const movePlayer = move.player ?? (move as any).playerNumber;
    const movePhase = (move as any).phase ?? inferPhaseFromMoveType(move.type);

    // Detect turn boundary (player change back to first player or explicit turn marker)
    if (movePlayer !== currentPlayer) {
      // If we have a current record, save it
      if (currentTurnRecord) {
        records.push(currentTurnRecord);
      }

      // Check if we're cycling back to player 1 (new turn)
      if (movePlayer < currentPlayer) {
        currentTurn++;
      }
      currentPlayer = movePlayer;

      // Start new turn record
      currentTurnRecord = {
        turnNumber: currentTurn,
        playerNumber: movePlayer,
        phases: new Map(),
      };
    }

    if (!currentTurnRecord) {
      currentTurnRecord = {
        turnNumber: currentTurn,
        playerNumber: movePlayer,
        phases: new Map(),
      };
    }

    // Record the phase
    if (movePhase) {
      const phaseRecords = currentTurnRecord.phases.get(movePhase) ?? [];
      phaseRecords.push(move);
      currentTurnRecord.phases.set(movePhase, phaseRecords);
    }
  }

  // Don't forget the last record
  if (currentTurnRecord) {
    records.push(currentTurnRecord);
  }

  return records;
}

/**
 * Infer phase from move type when phase field is not explicitly set.
 */
function inferPhaseFromMoveType(moveType: MoveType): GamePhase | null {
  switch (moveType) {
    case 'place_ring':
    case 'skip_placement':
    case 'no_placement_action':
    case 'swap_sides':
      return 'ring_placement';
    case 'move_stack':
    case 'no_movement_action':
    case 'recovery_slide':
    case 'skip_recovery':
      return 'movement';
    case 'overtaking_capture':
    case 'skip_capture':
      return 'capture';
    case 'continue_capture_segment':
      return 'chain_capture';
    case 'process_line':
    case 'choose_line_option':
    case 'no_line_action':
      return 'line_processing';
    case 'eliminate_rings_from_stack':
      // Appears in line_processing and territory_processing; default to earliest phase.
      return 'line_processing';
    case 'choose_territory_option':
    case 'skip_territory_processing':
    case 'no_territory_action':
      return 'territory_processing';
    case 'forced_elimination':
      return 'forced_elimination';
    default:
      return null;
  }
}

/**
 * Check that a move is valid for its recorded phase.
 */
function isMoveValidForPhase(move: Move, phase: GamePhase): boolean {
  const validTypes = PHASE_TO_VALID_MOVE_TYPES[phase];
  if (!validTypes) return false;
  return validTypes.includes(move.type);
}

/**
 * INV-PHASE-RECORDING: Check that all mandatory phases have recorded actions.
 *
 * This is a strict check per RR-CANON-R074/R075:
 * - Every player must have a record for ring_placement, movement, line_processing,
 *   territory_processing when they enter that phase
 * - If player has no actions available, they must record a no_*_action move
 */
export function checkPhaseRecordingInvariant(
  moves: Move[],
  numPlayers: number
): PhaseRecordingViolation[] {
  const violations: PhaseRecordingViolation[] = [];
  const records = extractPhaseRecordsByTurn(moves);

  // Group records by turn
  const turnMap = new Map<number, TurnPhaseRecords[]>();
  for (const record of records) {
    const turnRecords = turnMap.get(record.turnNumber) ?? [];
    turnRecords.push(record);
    turnMap.set(record.turnNumber, turnRecords);
  }

  // For each turn, check that each player has mandatory phase records
  for (const [turnNumber, turnRecords] of turnMap) {
    // Find which players participated this turn
    const playersThisTurn = new Set(turnRecords.map((r) => r.playerNumber));

    for (const record of turnRecords) {
      // Check mandatory phases
      for (const phase of MANDATORY_RECORD_PHASES) {
        // forced_elimination is only required when entered (conditional)
        if (phase === 'forced_elimination') {
          continue; // This phase is conditional, not mandatory for every turn
        }

        const phaseRecords = record.phases.get(phase);
        if (!phaseRecords || phaseRecords.length === 0) {
          violations.push({
            turnNumber,
            playerNumber: record.playerNumber,
            phase,
            reason: `No recorded action for ${phase} phase`,
          });
        } else {
          // Verify recorded moves are valid for this phase
          for (const move of phaseRecords) {
            if (!isMoveValidForPhase(move, phase)) {
              violations.push({
                turnNumber,
                playerNumber: record.playerNumber,
                phase,
                reason: `Invalid move type '${move.type}' for ${phase} phase`,
              });
            }
          }
        }
      }
    }
  }

  return violations;
}

/**
 * Validate that a game record adheres to phase recording invariants.
 */
export function validateGameRecordPhaseRecording(gameRecord: GameRecord): {
  valid: boolean;
  violations: PhaseRecordingViolation[];
} {
  const moves = (gameRecord.moves ?? []) as unknown as Move[];
  const numPlayers = gameRecord.players?.length ?? 2;

  const violations = checkPhaseRecordingInvariant(moves, numPlayers);

  return {
    valid: violations.length === 0,
    violations,
  };
}

// =============================================================================
// TESTS
// =============================================================================

describe('INV-PHASE-RECORDING invariant (RR-CANON-R074/R075)', () => {
  describe('inferPhaseFromMoveType', () => {
    it('correctly infers ring_placement phase', () => {
      expect(inferPhaseFromMoveType('place_ring')).toBe('ring_placement');
      expect(inferPhaseFromMoveType('skip_placement')).toBe('ring_placement');
      expect(inferPhaseFromMoveType('no_placement_action')).toBe('ring_placement');
      expect(inferPhaseFromMoveType('swap_sides')).toBe('ring_placement');
    });

    it('correctly infers movement phase', () => {
      expect(inferPhaseFromMoveType('move_stack')).toBe('movement');
      expect(inferPhaseFromMoveType('no_movement_action')).toBe('movement');
      expect(inferPhaseFromMoveType('recovery_slide')).toBe('movement');
      expect(inferPhaseFromMoveType('skip_recovery')).toBe('movement');
    });

    it('correctly infers capture phases', () => {
      expect(inferPhaseFromMoveType('overtaking_capture')).toBe('capture');
      expect(inferPhaseFromMoveType('skip_capture')).toBe('capture');
      expect(inferPhaseFromMoveType('continue_capture_segment')).toBe('chain_capture');
    });

    it('correctly infers processing phases', () => {
      expect(inferPhaseFromMoveType('process_line')).toBe('line_processing');
      expect(inferPhaseFromMoveType('choose_line_option')).toBe('line_processing');
      expect(inferPhaseFromMoveType('no_line_action')).toBe('line_processing');
      expect(inferPhaseFromMoveType('eliminate_rings_from_stack')).toBe('line_processing');
      expect(inferPhaseFromMoveType('choose_territory_option')).toBe('territory_processing');
      expect(inferPhaseFromMoveType('no_territory_action')).toBe('territory_processing');
    });

    it('correctly infers forced_elimination phase', () => {
      expect(inferPhaseFromMoveType('forced_elimination')).toBe('forced_elimination');
    });
  });
  describe('isMoveValidForPhase', () => {
    it('validates ring_placement moves', () => {
      const move: Move = {
        id: 'm1',
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };
      expect(isMoveValidForPhase(move, 'ring_placement')).toBe(true);
      expect(isMoveValidForPhase(move, 'movement')).toBe(false);
    });

    it('validates no_*_action moves', () => {
      const noPlacementMove: Move = {
        id: 'm1',
        type: 'no_placement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };
      expect(isMoveValidForPhase(noPlacementMove, 'ring_placement')).toBe(true);

      const noMovementMove: Move = {
        id: 'm2',
        type: 'no_movement_action',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 2,
      };
      expect(isMoveValidForPhase(noMovementMove, 'movement')).toBe(true);
    });

    it('validates forced_elimination moves', () => {
      const forcedElimMove: Move = {
        id: 'm1',
        type: 'forced_elimination',
        player: 1,
        to: { x: 0, y: 0 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };
      expect(isMoveValidForPhase(forcedElimMove, 'forced_elimination')).toBe(true);
      expect(isMoveValidForPhase(forcedElimMove, 'movement')).toBe(false);
    });
  });
  describe('checkPhaseRecordingInvariant', () => {
    it('detects missing ring_placement phase record', () => {
      const moves: PhaseAnnotatedMove[] = [
        // Player 1 only has movement, no placement
        {
          id: 'm1',
          type: 'move_stack',
          player: 1,
          phase: 'movement',
          from: { x: 0, y: 0 },
          to: { x: 1, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];

      const violations = checkPhaseRecordingInvariant(moves, 2);
      expect(violations.length).toBeGreaterThan(0);
      expect(violations.some((v) => v.phase === 'ring_placement')).toBe(true);
    });

    it('accepts valid complete turn with all phases', () => {
      const moves: PhaseAnnotatedMove[] = [
        // Player 1 complete turn
        {
          id: 'm1',
          type: 'place_ring',
          player: 1,
          phase: 'ring_placement',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
        {
          id: 'm2',
          type: 'move_stack',
          player: 1,
          phase: 'movement',
          from: { x: 0, y: 0 },
          to: { x: 1, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 2,
        },
        {
          id: 'm3',
          type: 'no_line_action',
          player: 1,
          phase: 'line_processing',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 3,
        },
        {
          id: 'm4',
          type: 'no_territory_action',
          player: 1,
          phase: 'territory_processing',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 4,
        },
      ];

      const violations = checkPhaseRecordingInvariant(moves, 2);
      expect(violations.length).toBe(0);
    });

    it('accepts no_*_action markers for players with no material', () => {
      const moves: PhaseAnnotatedMove[] = [
        // Player 1 has no material - records all no-action markers
        {
          id: 'm1',
          type: 'no_placement_action',
          player: 1,
          phase: 'ring_placement',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
        {
          id: 'm2',
          type: 'no_movement_action',
          player: 1,
          phase: 'movement',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 2,
        },
        {
          id: 'm3',
          type: 'no_line_action',
          player: 1,
          phase: 'line_processing',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 3,
        },
        {
          id: 'm4',
          type: 'no_territory_action',
          player: 1,
          phase: 'territory_processing',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 4,
        },
      ];

      const violations = checkPhaseRecordingInvariant(moves, 2);
      expect(violations.length).toBe(0);
    });

    it('accepts forced_elimination when player has stacks but no actions', () => {
      const moves: PhaseAnnotatedMove[] = [
        // Player 1 blocked in all phases, forced elimination
        {
          id: 'm1',
          type: 'no_placement_action',
          player: 1,
          phase: 'ring_placement',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
        {
          id: 'm2',
          type: 'no_movement_action',
          player: 1,
          phase: 'movement',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 2,
        },
        {
          id: 'm3',
          type: 'no_line_action',
          player: 1,
          phase: 'line_processing',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 3,
        },
        {
          id: 'm4',
          type: 'no_territory_action',
          player: 1,
          phase: 'territory_processing',
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 4,
        },
        {
          id: 'm5',
          type: 'forced_elimination',
          player: 1,
          phase: 'forced_elimination',
          to: { x: 2, y: 2 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 5,
        },
      ];

      const violations = checkPhaseRecordingInvariant(moves, 2);
      expect(violations.length).toBe(0);
    });
  });

  describe('PHASE_TO_VALID_MOVE_TYPES', () => {
    it('includes all canonical no-action move types', () => {
      expect(PHASE_TO_VALID_MOVE_TYPES.ring_placement).toContain('no_placement_action');
      expect(PHASE_TO_VALID_MOVE_TYPES.movement).toContain('no_movement_action');
      expect(PHASE_TO_VALID_MOVE_TYPES.line_processing).toContain('no_line_action');
      expect(PHASE_TO_VALID_MOVE_TYPES.territory_processing).toContain('no_territory_action');
    });

    it('includes forced_elimination in its phase', () => {
      expect(PHASE_TO_VALID_MOVE_TYPES.forced_elimination).toContain('forced_elimination');
    });
  });
});
