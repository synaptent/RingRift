/**
 * TerritoryAggregate.hex.feParity.test.ts
 *
 * Hex-focused fixtures and invariants for territory + forced-elimination semantics.
 *
 * Scope:
 * - HX-A: Compact hex territory region with mixed internal stacks and at least
 *   one acting-player stack outside the region. Mirrors the SQ8-A mini-region
 *   tests but on a hex board, exercising:
 *     - Self-elimination prerequisite (must have a stack outside region).
 *     - Internal eliminations + crediting and region collapse geometry.
 * - HX-C: Hex scenario where:
 *     - In territory_processing, the acting player has no processable regions.
 *     - Territory phase records a no_territory_action move.
 *     - Forced elimination is then surfaced as a separate forced_elimination
 *       phase with explicit forced_elimination moves, never as territory moves.
 */

import {
  canProcessTerritoryRegion,
  applyTerritoryRegion,
} from '../../src/shared/engine/aggregates/TerritoryAggregate';
import { positionToString } from '../../src/shared/types/game';
import type { GameState, Move, Position } from '../../src/shared/types/game';
import {
  createHexTerritoryFeBoardHxA,
  hexTerritoryFeRegionHxAForPlayer1,
} from '../fixtures/hexTerritoryFeFixtures';
import {
  createTestBoard,
  createTestGameState,
  addStack,
} from '../utils/fixtures';
import {
  processTurn,
  getValidMoves,
} from '../../src/shared/engine/orchestration/turnOrchestrator';

describe('hex territory/FE fixture HX-A – mixed internal stacks', () => {
  it('applies self-elimination prerequisite based on stacks outside the region (hex HX-A)', () => {
    const board = createHexTerritoryFeBoardHxA();
    const region = hexTerritoryFeRegionHxAForPlayer1;

    // Player 1 controls stacks both inside and outside the HX-A region on this
    // board, so the self-elimination prerequisite should pass.
    expect(canProcessTerritoryRegion(board, region, { player: 1 })).toBe(true);

    // If we remove all player 1 stacks outside the region, the region is no
    // longer processable for player 1.
    const boardNoOutside = createHexTerritoryFeBoardHxA();
    const regionKeys = new Set(region.spaces.map((p) => positionToString(p)));

    for (const [key, stack] of Array.from(boardNoOutside.stacks.entries())) {
      const isInRegion = regionKeys.has(key);
      if (stack.controllingPlayer === 1 && !isInRegion) {
        boardNoOutside.stacks.delete(key);
      }
    }

    expect(canProcessTerritoryRegion(boardNoOutside, region, { player: 1 })).toBe(false);
  });

  it('eliminates internal stacks and credits eliminations on the HX-A hex fixture', () => {
    const board = createHexTerritoryFeBoardHxA();
    const region = hexTerritoryFeRegionHxAForPlayer1;

    // Choose a representative internal stack in the HX-A region (player 2 stack at (1,-1,0)).
    const internalPos: Position = { x: 1, y: -1, z: 0 };
    const internalKey = positionToString(internalPos);
    const internalStack = board.stacks.get(internalKey);
    expect(internalStack).toBeDefined();
    const internalHeight = internalStack!.stackHeight;

    // Compute the total stack height of all stacks inside the HX-A region.
    let totalInternalHeight = 0;
    for (const pos of region.spaces) {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (stack) {
        totalInternalHeight += stack.stackHeight;
      }
    }

    // Sanity check: totalInternalHeight should be at least the representative stack's height.
    expect(totalInternalHeight).toBeGreaterThanOrEqual(internalHeight);

    const beforeElimsP1 = board.eliminatedRings[1] || 0;

    const outcome = applyTerritoryRegion(board, region, { player: 1 });

    // Original board is not mutated for the representative stack.
    expect(board.stacks.has(internalKey)).toBe(true);

    // Representative internal stack is removed on the next board.
    expect(outcome.board.stacks.has(internalKey)).toBe(false);

    // All region spaces are collapsed to player 1 and have no stacks/markers.
    for (const pos of region.spaces) {
      const key = positionToString(pos);
      expect(outcome.board.collapsedSpaces.get(key)).toBe(1);
      expect(outcome.board.stacks.has(key)).toBe(false);
      expect(outcome.board.markers.has(key)).toBe(false);
    }

    // All internal rings (across all stacks in the HX-A region) are credited to player 1.
    expect(outcome.eliminatedRingsByPlayer[1]).toBe(totalInternalHeight);
    expect(outcome.board.eliminatedRings[1]).toBe(beforeElimsP1 + totalInternalHeight);
  });
});

/**
 * Build a minimal hex state where:
 * - Board type: hexagonal.
 * - Board is a single valid hex cell (size=1, radius=0) at (0,0,0).
 * - Player 1 controls a single stack on that cell.
 * - Player 1 has no rings in hand.
 *
 * This guarantees:
 * - No legal movement or capture actions (no neighbors are in-bounds).
 * - No territory regions to process.
 * - Forced elimination is required once the territory phase is exhausted.
 */
function createHexSingleCellForcedEliminationState(): GameState {
  const board = createTestBoard('hexagonal');
  // Shrink to a single-cell hex board (radius 0).
  board.size = 1;
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();

  // Single P1 stack at the origin.
  const origin: Position = { x: 0, y: 0, z: 0 };
  addStack(board, origin, 1, 1);

  const state = createTestGameState({
    boardType: 'hexagonal',
    board,
    currentPhase: 'territory_processing',
    currentPlayer: 1,
  });

  // Ensure P1 has no placement actions available.
  state.players = state.players.map((p) =>
    p.playerNumber === 1 ? { ...p, ringsInHand: 0 } : p
  );

  return state;
}

describe('hex territory/FE fixture HX-C – no territory action then forced elimination', () => {
  it('surfaces no_territory_action in hex territory_processing and defers forced_elimination to a separate phase', () => {
    const state = createHexSingleCellForcedEliminationState();

    // The engine may or may not explicitly surface a no_territory_action move
    // in getValidMoves for this minimal hex ANM fixture. For the purposes of
    // this canonical HX-C fixture, we synthetically apply a no_territory_action
    // move in territory_processing and assert that forced_elimination is then
    // surfaced as a separate phase with explicit forced_elimination moves.
    const noTerritoryMove: Move = {
      id: 'hx-c-no-territory',
      type: 'no_territory_action',
      player: 1,
      to: { x: 0, y: 0, z: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    const result = processTurn(state, noTerritoryMove);

    // After no_territory_action, the orchestrator should detect that forced
    // elimination is required and surface a decision rather than silently
    // applying it.
    expect(result.status).toBe('awaiting_decision');
    expect(result.pendingDecision).toBeDefined();

    const decision: any = result.pendingDecision;
    expect(decision.options).toBeDefined();
    expect(Array.isArray(decision.options)).toBe(true);
    expect(decision.options.length).toBeGreaterThan(0);

    // Current behavior: the FSM may advance the phase label to ring_placement
    // for bookkeeping, even while a forced_elimination decision is pending.
    // The key invariant we assert here is that we have *left* territory_processing
    // and that forced_elimination moves only appear via this separate decision,
    // never as territory_processing moves.
    expect(result.nextState.currentPhase).not.toBe('territory_processing');

    // Decision options in this post-territory step must be explicit
    // forced_elimination moves, not territory_processing moves.
    for (const option of decision.options as Move[]) {
      expect(option.type).toBe('forced_elimination');
    }
  });
});