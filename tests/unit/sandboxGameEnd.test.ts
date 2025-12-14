/**
 * Unit tests for sandboxGameEnd.ts
 *
 * Tests for resolveGlobalStalemateIfNeededSandbox and checkAndApplyVictorySandbox
 * functions, focusing on edge cases and branch coverage.
 */

import {
  resolveGlobalStalemateIfNeededSandbox,
  checkAndApplyVictorySandbox,
  SandboxGameEndHooks,
} from '../../src/client/sandbox/sandboxGameEnd';
import { createTestBoard, createTestGameState, createTestPlayer, pos } from '../utils/fixtures';
import type { GameState } from '../../src/shared/types/game';

describe('sandboxGameEnd', () => {
  // Helper to create hooks
  function createHooks(placementsByPlayer: { [player: number]: number }): SandboxGameEndHooks {
    return {
      enumerateLegalRingPlacements: (playerNumber: number) => {
        const count = placementsByPlayer[playerNumber] ?? 0;
        // Return array of positions with the specified count
        return Array.from({ length: count }, (_, i) => pos(i, 0));
      },
    };
  }

  describe('resolveGlobalStalemateIfNeededSandbox', () => {
    it('returns state unchanged when gameStatus is not active', () => {
      const state = createTestGameState({
        gameStatus: 'completed',
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 5 }), createTestPlayer(2, { ringsInHand: 5 })],
      });
      state.board.stacks.clear();

      const hooks = createHooks({ 1: 0, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      expect(result).toBe(state);
    });

    it('returns state unchanged when board has stacks', () => {
      const board = createTestBoard('square8');
      board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [createTestPlayer(1, { ringsInHand: 5 }), createTestPlayer(2, { ringsInHand: 5 })],
      });

      const hooks = createHooks({ 1: 0, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      expect(result).toBe(state);
    });

    it('returns state unchanged when no players have rings in hand', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
      });

      const hooks = createHooks({ 1: 0, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      expect(result).toBe(state);
    });

    it('returns state unchanged when any player has legal placements', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [createTestPlayer(1, { ringsInHand: 5 }), createTestPlayer(2, { ringsInHand: 5 })],
      });

      // Player 1 has legal placements
      const hooks = createHooks({ 1: 3, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      expect(result).toBe(state);
    });

    it('skips players with no rings in hand when checking legal placements', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0 }), // No rings in hand
          createTestPlayer(2, { ringsInHand: 5 }),
        ],
      });

      // Player 1 would have placements if they had rings, but they don't
      // Player 2 has rings but no placements
      const hooks = createHooks({ 1: 10, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      // Should convert player 2's rings to eliminated (stalemate)
      expect(result).not.toBe(state);
      expect(result.players.find((p) => p.playerNumber === 2)?.ringsInHand).toBe(0);
      expect(result.players.find((p) => p.playerNumber === 2)?.eliminatedRings).toBe(5);
    });

    it('converts rings in hand to eliminated for global stalemate', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();
      board.eliminatedRings = { 1: 2, 2: 3 };

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 4, eliminatedRings: 2 }),
          createTestPlayer(2, { ringsInHand: 6, eliminatedRings: 3 }),
        ],
        totalRingsEliminated: 5,
      });

      // No legal placements for anyone
      const hooks = createHooks({ 1: 0, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      // Rings in hand should be converted to eliminated
      expect(result).not.toBe(state);

      const p1 = result.players.find((p) => p.playerNumber === 1);
      const p2 = result.players.find((p) => p.playerNumber === 2);

      expect(p1?.ringsInHand).toBe(0);
      expect(p1?.eliminatedRings).toBe(6); // 2 + 4

      expect(p2?.ringsInHand).toBe(0);
      expect(p2?.eliminatedRings).toBe(9); // 3 + 6

      expect(result.totalRingsEliminated).toBe(15); // 5 + 4 + 6
      expect(result.board.eliminatedRings[1]).toBe(6); // 2 + 4
      expect(result.board.eliminatedRings[2]).toBe(9); // 3 + 6
    });

    it('preserves player with no rings in hand during stalemate conversion', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0, eliminatedRings: 10 }),
          createTestPlayer(2, { ringsInHand: 5, eliminatedRings: 0 }),
        ],
        totalRingsEliminated: 10,
      });

      const hooks = createHooks({ 1: 0, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      const p1 = result.players.find((p) => p.playerNumber === 1);
      expect(p1?.eliminatedRings).toBe(10); // Unchanged
      expect(p1?.ringsInHand).toBe(0); // Still 0
    });

    it('initializes eliminatedRings for player if not present in board', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();
      board.eliminatedRings = {}; // Empty

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 3, eliminatedRings: 0 }),
          createTestPlayer(2, { ringsInHand: 0, eliminatedRings: 0 }),
        ],
        totalRingsEliminated: 0,
      });

      const hooks = createHooks({ 1: 0, 2: 0 });
      const result = resolveGlobalStalemateIfNeededSandbox(state, hooks);

      expect(result.board.eliminatedRings[1]).toBe(3);
    });
  });

  describe('checkAndApplyVictorySandbox', () => {
    it('returns null result when gameStatus is not active', () => {
      const state = createTestGameState({
        gameStatus: 'completed',
        board: createTestBoard('square8'),
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      const hooks = createHooks({ 1: 0, 2: 0 });
      const { state: resultState, result } = checkAndApplyVictorySandbox(state, hooks);

      expect(result).toBeNull();
      expect(resultState).toBe(state);
    });

    it('returns null result when no victory condition met', () => {
      const board = createTestBoard('square8');
      board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      });

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 16, eliminatedRings: 0 }),
          createTestPlayer(2, { ringsInHand: 18, eliminatedRings: 0 }),
        ],
        victoryThreshold: 18,
        territoryVictoryThreshold: 64,
      });

      const hooks = createHooks({ 1: 10, 2: 10 });
      const { state: resultState, result } = checkAndApplyVictorySandbox(state, hooks);

      expect(result).toBeNull();
    });

    it('returns victory result and marks game completed for ring elimination', () => {
      const board = createTestBoard('square8');

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 5, eliminatedRings: 18 }),
          createTestPlayer(2, { ringsInHand: 13, eliminatedRings: 0 }),
        ],
        victoryThreshold: 18,
      });

      const hooks = createHooks({ 1: 0, 2: 0 });
      const { state: resultState, result } = checkAndApplyVictorySandbox(state, hooks);

      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('ring_elimination');
      expect(resultState.gameStatus).toBe('completed');
      expect(resultState.winner).toBe(1);
    });

    it('returns victory result after stalemate resolution', () => {
      const board = createTestBoard('square8');
      board.stacks.clear();

      // Add collapsed spaces to board for territory count (victory check uses board.collapsedSpaces)
      // Player 1 controls 5 spaces, Player 2 controls 3 spaces
      for (let i = 0; i < 5; i++) {
        board.collapsedSpaces.set(`${i},0`, 1);
      }
      for (let i = 0; i < 3; i++) {
        board.collapsedSpaces.set(`${i},1`, 2);
      }

      const state = createTestGameState({
        gameStatus: 'active',
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 10, eliminatedRings: 0, territorySpaces: 5 }),
          createTestPlayer(2, { ringsInHand: 8, eliminatedRings: 0, territorySpaces: 3 }),
        ],
        victoryThreshold: 100,
        territoryVictoryThreshold: 100,
      });

      // No legal placements - triggers stalemate
      const hooks = createHooks({ 1: 0, 2: 0 });
      const { state: resultState, result } = checkAndApplyVictorySandbox(state, hooks);

      // After stalemate resolution, territory tie-breaker should apply
      expect(result).not.toBeNull();
      expect(result!.winner).toBe(1);
      expect(result!.reason).toBe('territory_control');
      expect(resultState.gameStatus).toBe('completed');
    });
  });
});
