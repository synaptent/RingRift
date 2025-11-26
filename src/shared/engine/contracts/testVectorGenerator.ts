/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Test Vector Generator for Contract-Based Parity Testing
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Generates test vectors from existing game states and moves in the canonical
 * format defined by the contract schemas. These vectors are used for:
 * 1. TypeScript ↔ Python parity validation
 * 2. Regression testing across engine refactors
 * 3. Contract compliance verification
 */

import type { GameState, Move } from '../../types/game';
import { serializeGameState, computeStateDiff, SerializedGameState } from './serialization';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Test vector category based on move type.
 */
export type TestVectorCategory =
  | 'placement'
  | 'movement'
  | 'capture'
  | 'chain_capture'
  | 'line_detection'
  | 'line_processing'
  | 'territory'
  | 'territory_processing'
  | 'victory'
  | 'edge_case';

/**
 * Test vector following the contract schema format.
 */
export interface ContractTestVector {
  id: string;
  version: string;
  category: TestVectorCategory;
  description: string;
  tags: string[];
  source: 'manual' | 'recorded' | 'generated' | 'regression';
  createdAt: string;

  input: {
    state: SerializedGameState;
    move: Move;
  };

  expectedOutput: {
    status: 'complete' | 'awaiting_decision';
    assertions: TestVectorAssertions;
  };
}

/**
 * Assertions to validate against the result state.
 */
export interface TestVectorAssertions {
  // Phase/turn state
  currentPlayer?: number;
  currentPhase?: string;
  gameStatus?: string;

  // Board counts
  stackCount?: number;
  markerCount?: number;
  collapsedCount?: number;

  // Player state
  player1RingsInHand?: number;
  player1EliminatedRings?: number;
  player1TerritorySpaces?: number;
  player2RingsInHand?: number;
  player2EliminatedRings?: number;
  player2TerritorySpaces?: number;

  // Invariants
  sInvariantDelta?: number;
  sInvariantValue?: number;

  // Victory
  victoryWinner?: number | null;
  victoryReason?: string;

  // Hash for exact match (optional)
  stateHash?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// Generator Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Infer the test vector category from a move type.
 */
export function inferCategory(moveType: Move['type']): TestVectorCategory {
  switch (moveType) {
    case 'place_ring':
    case 'skip_placement':
      return 'placement';
    case 'move_stack':
    case 'move_ring':
    case 'build_stack':
      return 'movement';
    case 'overtaking_capture':
      return 'capture';
    case 'continue_capture_segment':
      return 'chain_capture';
    case 'process_line':
    case 'choose_line_reward':
      return 'line_processing';
    case 'process_territory_region':
    case 'eliminate_rings_from_stack':
      return 'territory_processing';
    default:
      return 'edge_case';
  }
}

/**
 * Generate a unique test vector ID.
 */
export function generateVectorId(category: TestVectorCategory, suffix?: string): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 6);
  const suffixPart = suffix ? `.${suffix}` : '';
  return `${category}.${timestamp}.${random}${suffixPart}`;
}

/**
 * Create a test vector from a before state, move, and after state.
 */
export function createContractTestVector(
  beforeState: GameState,
  move: Move,
  afterState: GameState,
  options: {
    description?: string | undefined;
    tags?: string[] | undefined;
    source?: ContractTestVector['source'] | undefined;
    includeHash?: boolean | undefined;
  } = {}
): ContractTestVector {
  const category = inferCategory(move.type);
  const diff = computeStateDiff(beforeState, afterState);

  const assertions: TestVectorAssertions = {
    currentPlayer: afterState.currentPlayer,
    currentPhase: afterState.currentPhase,
    gameStatus: afterState.gameStatus,
    stackCount: afterState.board.stacks.size,
    markerCount: afterState.board.markers.size,
    collapsedCount: afterState.board.collapsedSpaces.size,
    sInvariantDelta: diff.sInvariantDelta as number,
  };

  // Add player state
  if (afterState.players[0]) {
    assertions.player1RingsInHand = afterState.players[0].ringsInHand;
    assertions.player1EliminatedRings = afterState.players[0].eliminatedRings;
  }
  if (afterState.players[1]) {
    assertions.player2RingsInHand = afterState.players[1].ringsInHand;
    assertions.player2EliminatedRings = afterState.players[1].eliminatedRings;
  }

  // Add victory info if game is over
  if (afterState.gameStatus === 'completed' || afterState.gameStatus === 'finished') {
    assertions.victoryWinner = afterState.winner ?? null;
  }

  return {
    id: generateVectorId(category),
    version: 'v2',
    category,
    description: options.description || `${move.type} by player ${move.player}`,
    tags: options.tags || [],
    source: options.source || 'generated',
    createdAt: new Date().toISOString(),
    input: {
      state: serializeGameState(beforeState),
      move: { ...move },
    },
    expectedOutput: {
      status: 'complete',
      assertions,
    },
  };
}

/**
 * Create multiple test vectors from a game trace.
 */
export function createTestVectorsFromTrace(
  trace: { initialState: GameState; entries: Array<{ action: Move; stateAfter?: GameState }> },
  options: {
    tags?: string[];
    source?: ContractTestVector['source'];
  } = {}
): ContractTestVector[] {
  const vectors: ContractTestVector[] = [];
  let currentState = trace.initialState;

  for (const entry of trace.entries) {
    if (entry.stateAfter) {
      vectors.push(
        createContractTestVector(currentState, entry.action, entry.stateAfter, {
          tags: options.tags,
          source: options.source,
        })
      );
      currentState = entry.stateAfter;
    }
  }

  return vectors;
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Utilities
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Validate a result state against test vector assertions.
 */
export function validateAgainstAssertions(
  resultState: GameState,
  assertions: TestVectorAssertions
): { valid: boolean; failures: string[] } {
  const failures: string[] = [];

  if (
    assertions.currentPlayer !== undefined &&
    resultState.currentPlayer !== assertions.currentPlayer
  ) {
    failures.push(
      `currentPlayer: expected ${assertions.currentPlayer}, got ${resultState.currentPlayer}`
    );
  }

  if (
    assertions.currentPhase !== undefined &&
    resultState.currentPhase !== assertions.currentPhase
  ) {
    failures.push(
      `currentPhase: expected ${assertions.currentPhase}, got ${resultState.currentPhase}`
    );
  }

  if (assertions.gameStatus !== undefined && resultState.gameStatus !== assertions.gameStatus) {
    failures.push(`gameStatus: expected ${assertions.gameStatus}, got ${resultState.gameStatus}`);
  }

  if (
    assertions.stackCount !== undefined &&
    resultState.board.stacks.size !== assertions.stackCount
  ) {
    failures.push(
      `stackCount: expected ${assertions.stackCount}, got ${resultState.board.stacks.size}`
    );
  }

  if (
    assertions.markerCount !== undefined &&
    resultState.board.markers.size !== assertions.markerCount
  ) {
    failures.push(
      `markerCount: expected ${assertions.markerCount}, got ${resultState.board.markers.size}`
    );
  }

  if (
    assertions.collapsedCount !== undefined &&
    resultState.board.collapsedSpaces.size !== assertions.collapsedCount
  ) {
    failures.push(
      `collapsedCount: expected ${assertions.collapsedCount}, got ${resultState.board.collapsedSpaces.size}`
    );
  }

  // Player state validations
  if (
    assertions.player1RingsInHand !== undefined &&
    resultState.players[0] &&
    resultState.players[0].ringsInHand !== assertions.player1RingsInHand
  ) {
    failures.push(
      `player1RingsInHand: expected ${assertions.player1RingsInHand}, got ${resultState.players[0].ringsInHand}`
    );
  }

  if (
    assertions.player1EliminatedRings !== undefined &&
    resultState.players[0] &&
    resultState.players[0].eliminatedRings !== assertions.player1EliminatedRings
  ) {
    failures.push(
      `player1EliminatedRings: expected ${assertions.player1EliminatedRings}, got ${resultState.players[0].eliminatedRings}`
    );
  }

  if (
    assertions.player1TerritorySpaces !== undefined &&
    resultState.players[0] &&
    resultState.players[0].territorySpaces !== assertions.player1TerritorySpaces
  ) {
    failures.push(
      `player1TerritorySpaces: expected ${assertions.player1TerritorySpaces}, got ${resultState.players[0].territorySpaces}`
    );
  }

  if (
    assertions.player2RingsInHand !== undefined &&
    resultState.players[1] &&
    resultState.players[1].ringsInHand !== assertions.player2RingsInHand
  ) {
    failures.push(
      `player2RingsInHand: expected ${assertions.player2RingsInHand}, got ${resultState.players[1].ringsInHand}`
    );
  }

  if (
    assertions.player2EliminatedRings !== undefined &&
    resultState.players[1] &&
    resultState.players[1].eliminatedRings !== assertions.player2EliminatedRings
  ) {
    failures.push(
      `player2EliminatedRings: expected ${assertions.player2EliminatedRings}, got ${resultState.players[1].eliminatedRings}`
    );
  }

  if (
    assertions.player2TerritorySpaces !== undefined &&
    resultState.players[1] &&
    resultState.players[1].territorySpaces !== assertions.player2TerritorySpaces
  ) {
    failures.push(
      `player2TerritorySpaces: expected ${assertions.player2TerritorySpaces}, got ${resultState.players[1].territorySpaces}`
    );
  }

  if (assertions.victoryWinner !== undefined) {
    const actualWinner = resultState.winner ?? null;
    if (actualWinner !== assertions.victoryWinner) {
      failures.push(`victoryWinner: expected ${assertions.victoryWinner}, got ${actualWinner}`);
    }
  }

  return {
    valid: failures.length === 0,
    failures,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Export Bundle
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Export test vectors to JSON format for Python consumption.
 */
export function exportVectorBundle(vectors: ContractTestVector[]): string {
  return JSON.stringify(
    {
      version: 'v2',
      generated: new Date().toISOString(),
      count: vectors.length,
      categories: [...new Set(vectors.map((v) => v.category))],
      vectors,
    },
    null,
    2
  );
}

/**
 * Import test vectors from JSON format.
 */
export function importVectorBundle(json: string): ContractTestVector[] {
  const bundle = JSON.parse(json);
  return bundle.vectors || [];
}
