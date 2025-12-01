/**
 * Chain Capture Edge Case Fixtures
 *
 * These fixtures were discovered using the chain capture exhaustive generator
 * (scripts/run-chain-capture-exhaustive.ts) and represent interesting edge cases
 * that test the boundaries of the chain capture rules.
 *
 * Key edge cases covered:
 * - Multi-capture from same target (Section 4.3: "Recapture from the same target multiple times")
 * - Long chains (10+ segments on 19x19)
 * - Direction changes within chains
 * - Hex coordinate chains
 */

import type { BoardState, GameState, Position } from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  addStack as addStackHelper,
} from '../utils/fixtures';

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

function createEmptyBoard(
  boardType: 'square8' | 'square19' | 'hexagonal',
  _size: number
): BoardState {
  return createTestBoard(boardType);
}

function addStack(board: BoardState, pos: Position, player: number, height: number): void {
  addStackHelper(board, pos, player, height);
}

function createGameState(board: BoardState, currentPlayer: number = 1): GameState {
  return createTestGameState({
    board: { ...board, type: board.type as any },
    boardType: board.type as any,
    currentPlayer,
    currentPhase: 'capture',
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 1: Multi-capture from same target
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests the rule that allows capturing from the same target multiple times.
 *
 * Setup (square8):
 * - Attacker at (6,4) height 3
 * - Target at (6,3) height 2 - can be captured twice
 *
 * Expected behavior:
 * 1. First capture: (6,4) -> (6,3) -> (6,1)  [target now height 1]
 * 2. Second capture: (6,1) -> (6,3) -> (6,7) [target now empty]
 *
 * Reference: Section 4.3 "Recapture from the same target multiple times
 * (as long as it still has rings and path constraints remain satisfied)"
 */
export interface MultiCaptureFixture {
  name: string;
  board: BoardState;
  attackerPos: Position;
  targetPos: Position;
  expectedFirstLanding: Position;
  expectedSecondLanding: Position;
  maxChainLength: number;
}

export function createMultiCaptureSameTargetFixture(): MultiCaptureFixture {
  const board = createEmptyBoard('square8', 8);

  // Attacker at (6,4) with height 3
  addStack(board, { x: 6, y: 4 }, 1, 3);

  // Target at (6,3) with height 2 - can be captured twice
  addStack(board, { x: 6, y: 3 }, 2, 2);

  return {
    name: 'Multi-capture from same target (square8)',
    board,
    attackerPos: { x: 6, y: 4 },
    targetPos: { x: 6, y: 3 },
    expectedFirstLanding: { x: 6, y: 1 },
    expectedSecondLanding: { x: 6, y: 7 },
    maxChainLength: 2,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 2: 5-segment chain on square8
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests a 5-segment chain capture sequence on 8x8 board.
 *
 * Discovered seed: 4271921684
 * Setup:
 * - Attacker at (6,4) height 3
 * - Targets at (6,3), (6,0), (2,3), (0,3) heights 2-3
 *
 * Example chain:
 * 1. (6,4) -> (6,3) -> (6,1)
 * 2. (6,1) -> (6,3) -> (6,7)  [recapture same target]
 * 3. (6,7) -> (2,3) -> (0,1)
 * 4. (0,1) -> (0,3) -> (0,7)
 * 5. (0,7) -> (0,3) -> (0,0)  [recapture same target]
 */
export interface LongChainFixture {
  name: string;
  board: BoardState;
  attackerPos: Position;
  gameState: GameState;
  maxChainLength: number;
  expectedSequences: number;
}

export function createSquare8LongChainFixture(): LongChainFixture {
  const board = createEmptyBoard('square8', 8);

  // Attacker at (6,4) height 3
  addStack(board, { x: 6, y: 4 }, 1, 3);

  // Targets
  addStack(board, { x: 6, y: 3 }, 2, 2); // height 2
  addStack(board, { x: 6, y: 0 }, 2, 2); // height 2
  addStack(board, { x: 2, y: 3 }, 2, 3); // height 3
  addStack(board, { x: 0, y: 3 }, 2, 2); // height 2

  return {
    name: '5-segment chain with double recapture (square8)',
    board,
    attackerPos: { x: 6, y: 4 },
    gameState: createGameState(board),
    maxChainLength: 5,
    expectedSequences: 5,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 3: 10-segment chain on square19
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests the maximum discovered chain length on 19x19 board.
 *
 * Discovered seed: 2307637706
 * Setup:
 * - Attacker at (14,4) height 3
 * - Targets strategically placed for 10-segment chain
 *
 * This demonstrates the combinatorial explosion of chain possibilities
 * on larger boards - over 1000 distinct sequences possible.
 */
export function createSquare19LongChainFixture(): LongChainFixture {
  const board = createEmptyBoard('square19', 19);

  // Attacker at (14,4) height 3
  addStack(board, { x: 14, y: 4 }, 1, 3);

  // Targets (from discovered seed 2307637706)
  addStack(board, { x: 13, y: 4 }, 2, 3);
  addStack(board, { x: 13, y: 0 }, 2, 2);
  addStack(board, { x: 9, y: 0 }, 1, 3); // Own stack - legal to capture
  addStack(board, { x: 13, y: 8 }, 2, 2);

  return {
    name: '10-segment chain on 19x19',
    board,
    attackerPos: { x: 14, y: 4 },
    gameState: createGameState(board),
    maxChainLength: 10,
    expectedSequences: 1050,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 4: Hex coordinate chain capture
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests chain captures on hexagonal board with cube coordinates.
 *
 * Discovered seed: 555593847
 * Setup:
 * - Attacker at (-9,2,7) height 3
 * - Targets using hex geometry
 *
 * This tests that hex cube coordinate capture logic works correctly
 * for chains that span the board.
 */
export function createHexLongChainFixture(): LongChainFixture {
  const board = createEmptyBoard('hexagonal', 11);

  // Attacker at (-9,2,7) height 3
  addStack(board, { x: -9, y: 2, z: 7 }, 1, 3);

  // Targets (from discovered seed 555593847)
  addStack(board, { x: -8, y: 2, z: 6 }, 2, 2);
  addStack(board, { x: -8, y: -1, z: 9 }, 2, 2);
  addStack(board, { x: -4, y: -2, z: 6 }, 2, 3);
  addStack(board, { x: -6, y: 0, z: 6 }, 2, 2);

  return {
    name: '9-segment chain on hex board',
    board,
    attackerPos: { x: -9, y: 2, z: 7 },
    gameState: createGameState(board),
    maxChainLength: 9,
    expectedSequences: 1486,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 5: Self-capture in chain
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests capturing your own stack as part of a chain.
 *
 * The rules explicitly allow capturing your own stacks:
 * Section 4.2: "Capturing from your own stack is allowed; the ring changes
 * vertical position but remains in play."
 *
 * Key constraints:
 * - Landing distance >= attacker stack height
 * - Landing must be empty (no stack, no collapsed space)
 */
export interface SelfCaptureFixture {
  name: string;
  board: BoardState;
  attackerPos: Position;
  ownStackPos: Position;
  expectedCanCapture: boolean;
}

export function createSelfCaptureInChainFixture(): SelfCaptureFixture {
  const board = createEmptyBoard('square8', 8);

  // Attacker at (1,4) height 2 (not 3, to make landing easier)
  addStack(board, { x: 1, y: 4 }, 1, 2);

  // Own stack at (2,4) height 1 - can be captured
  // Landing must be at distance >= 2 from origin, so (3,4) or beyond
  addStack(board, { x: 2, y: 4 }, 1, 1);

  // Enemy stack further away for potential chain
  addStack(board, { x: 6, y: 4 }, 2, 2);

  return {
    name: 'Self-capture in chain (square8)',
    board,
    attackerPos: { x: 1, y: 4 },
    ownStackPos: { x: 2, y: 4 },
    expectedCanCapture: true,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 6: Direction change in chain
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests that chains can change direction between segments.
 *
 * Section 4.3: "Chains may: Change direction between segments."
 *
 * Setup creates a scenario where the optimal chain involves
 * capturing East, then South, then West.
 */
export interface DirectionChangeFixture {
  name: string;
  board: BoardState;
  attackerPos: Position;
  expectedDirections: Array<{ dx: number; dy: number }>;
}

export function createDirectionChangeChainFixture(): DirectionChangeFixture {
  const board = createEmptyBoard('square8', 8);

  // Attacker at (3,3) height 3
  addStack(board, { x: 3, y: 3 }, 1, 3);

  // Target East at (4,3)
  addStack(board, { x: 4, y: 3 }, 2, 2);

  // After landing at (6,3), target South at (6,4)
  addStack(board, { x: 6, y: 4 }, 2, 2);

  // After landing at (6,6), target West at (5,6)
  addStack(board, { x: 5, y: 6 }, 2, 2);

  return {
    name: 'Direction change in chain (E -> S -> W)',
    board,
    attackerPos: { x: 3, y: 3 },
    expectedDirections: [
      { dx: 1, dy: 0 }, // East
      { dx: 0, dy: 1 }, // South
      { dx: -1, dy: 0 }, // West
    ],
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Edge Case 7: 180° reversal in chain
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Tests 180° reversal over previously captured stacks.
 *
 * Section 4.3: "Chains may: 180° reverse over previously captured stacks."
 */
export interface ReversalFixture {
  name: string;
  board: BoardState;
  attackerPos: Position;
  targetPos: Position;
  expectedReversalPossible: boolean;
}

export function createReversalChainFixture(): ReversalFixture {
  const board = createEmptyBoard('square8', 8);

  // Attacker at center (4,4) height 3
  addStack(board, { x: 4, y: 4 }, 1, 3);

  // Target at (5,4) height 2
  addStack(board, { x: 5, y: 4 }, 2, 2);

  // After capturing East and landing at (7,4), can reverse West
  // through the now-reduced (5,4) if it still has rings

  return {
    name: '180° reversal chain (square8)',
    board,
    attackerPos: { x: 4, y: 4 },
    targetPos: { x: 5, y: 4 },
    expectedReversalPossible: true,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Export all fixtures
// ═══════════════════════════════════════════════════════════════════════════

export const chainCaptureEdgeCaseFixtures = {
  multiCaptureSameTarget: createMultiCaptureSameTargetFixture,
  square8LongChain: createSquare8LongChainFixture,
  square19LongChain: createSquare19LongChainFixture,
  hexLongChain: createHexLongChainFixture,
  selfCaptureInChain: createSelfCaptureInChainFixture,
  directionChangeChain: createDirectionChangeChainFixture,
  reversalChain: createReversalChainFixture,
};
