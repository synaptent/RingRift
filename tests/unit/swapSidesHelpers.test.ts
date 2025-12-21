/**
 * Unit tests for swapSidesHelpers module
 *
 * Direct tests for pure swap-sides (pie rule) helper functions to ensure
 * complete branch coverage of all guard conditions and edge cases.
 */

import {
  shouldOfferSwapSides,
  validateSwapSidesMove,
  applySwapSidesIdentitySwap,
} from '../../src/shared/engine/swapSidesHelpers';
import type { GameState, GamePhase, Move, Player } from '../../src/shared/types/game';

describe('swapSidesHelpers', () => {
  const createBaseState = (overrides: Partial<GameState> = {}): GameState =>
    ({
      gameStatus: 'active',
      currentPlayer: 2,
      currentPhase: 'ring_placement' as GamePhase,
      players: [
        {
          id: 'p1',
          username: 'Alice',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 5,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Bob',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 5,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      board: {
        type: 'square8',
        width: 8,
        height: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: [],
      },
      turnNumber: 2,
      moveHistory: [
        { type: 'place_ring', player: 1, to: { x: 3, y: 3 }, placementCount: 1 } as Move,
        { type: 'move_stack', player: 1, from: { x: 3, y: 3 }, to: { x: 4, y: 3 } } as Move,
      ],
      rulesOptions: { swapRuleEnabled: true },
      ...overrides,
    }) as unknown as GameState;

  describe('shouldOfferSwapSides', () => {
    it('should return true when all conditions are met', () => {
      const state = createBaseState();
      expect(shouldOfferSwapSides(state)).toBe(true);
    });

    it('should return false when swapRuleEnabled is false', () => {
      const state = createBaseState({ rulesOptions: { swapRuleEnabled: false } });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when rulesOptions is undefined', () => {
      const state = createBaseState({ rulesOptions: undefined });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when game is not active', () => {
      const state = createBaseState({ gameStatus: 'completed' });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when game status is waiting', () => {
      const state = createBaseState({ gameStatus: 'waiting' });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when more than 2 players', () => {
      const state = createBaseState();
      state.players.push({
        id: 'p3',
        username: 'Charlie',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 5,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as Player);
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when current player is Player 1', () => {
      const state = createBaseState({ currentPlayer: 1 });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when current player is Player 3', () => {
      const state = createBaseState({ currentPlayer: 3 });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false in line_formed phase', () => {
      const state = createBaseState({ currentPhase: 'line_formed' as GamePhase });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false in territory_formed phase', () => {
      const state = createBaseState({ currentPhase: 'territory_formed' as GamePhase });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return true in movement phase', () => {
      const state = createBaseState({ currentPhase: 'movement' as GamePhase });
      expect(shouldOfferSwapSides(state)).toBe(true);
    });

    it('should return true in capture phase', () => {
      const state = createBaseState({ currentPhase: 'capture' as GamePhase });
      expect(shouldOfferSwapSides(state)).toBe(true);
    });

    it('should return true in chain_capture phase', () => {
      const state = createBaseState({ currentPhase: 'chain_capture' as GamePhase });
      expect(shouldOfferSwapSides(state)).toBe(true);
    });

    it('should return false when moveHistory is empty', () => {
      const state = createBaseState({ moveHistory: [] });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when swap_sides move already exists', () => {
      const state = createBaseState({
        moveHistory: [
          { type: 'place_ring', player: 1, to: { x: 3, y: 3 }, placementCount: 1 } as Move,
          { type: 'move_stack', player: 1, from: { x: 3, y: 3 }, to: { x: 4, y: 3 } } as Move,
          { type: 'swap_sides', player: 2, to: { x: 0, y: 0 } } as unknown as Move,
        ],
      });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when no Player 1 moves exist', () => {
      const state = createBaseState({
        moveHistory: [
          { type: 'place_ring', player: 2, to: { x: 3, y: 3 }, placementCount: 1 } as Move,
        ],
      });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });

    it('should return false when Player 2 has already made a non-swap move', () => {
      const state = createBaseState({
        moveHistory: [
          { type: 'place_ring', player: 1, to: { x: 3, y: 3 }, placementCount: 1 } as Move,
          { type: 'move_stack', player: 1, from: { x: 3, y: 3 }, to: { x: 4, y: 3 } } as Move,
          { type: 'place_ring', player: 2, to: { x: 5, y: 5 }, placementCount: 1 } as Move,
        ],
      });
      expect(shouldOfferSwapSides(state)).toBe(false);
    });
  });

  describe('validateSwapSidesMove', () => {
    it('should return valid when all conditions are met', () => {
      const state = createBaseState();
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({ valid: true });
    });

    it('should reject when swapRuleEnabled is false', () => {
      const state = createBaseState({ rulesOptions: { swapRuleEnabled: false } });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: 'swap_sides is disabled for this game',
      });
    });

    it('should reject when game is not active', () => {
      const state = createBaseState({ gameStatus: 'completed' });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: 'swap_sides is only available in active games',
      });
    });

    it('should reject in 3-player games', () => {
      const state = createBaseState();
      state.players.push({
        id: 'p3',
        username: 'Charlie',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: 600000,
        ringsInHand: 5,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as Player);
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: 'swap_sides is only defined for 2-player games',
      });
    });

    it('should reject when not the current player', () => {
      const state = createBaseState({ currentPlayer: 1 });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: 'Only the active player may request swap_sides',
      });
    });

    it('should reject when Player 1 tries to swap', () => {
      const state = createBaseState({ currentPlayer: 1 });
      const result = validateSwapSidesMove(state, 1);
      expect(result).toEqual({
        valid: false,
        reason: 'Only Player 2 may request swap_sides',
      });
    });

    it('should reject in line_formed phase', () => {
      const state = createBaseState({ currentPhase: 'line_formed' as GamePhase });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: 'swap_sides is only available at the start of ring placement',
      });
    });

    it('should reject in territory_formed phase', () => {
      const state = createBaseState({ currentPhase: 'territory_formed' as GamePhase });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: 'swap_sides is only available at the start of ring placement',
      });
    });

    it('should reject when no P1 move exists', () => {
      const state = createBaseState({ moveHistory: [] });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: "swap_sides is only available immediately after Player 1's first turn",
      });
    });

    it('should reject when P2 has already moved', () => {
      const state = createBaseState({
        moveHistory: [
          { type: 'place_ring', player: 1, to: { x: 3, y: 3 }, placementCount: 1 } as Move,
          { type: 'move_stack', player: 1, from: { x: 3, y: 3 }, to: { x: 4, y: 3 } } as Move,
          { type: 'place_ring', player: 2, to: { x: 5, y: 5 }, placementCount: 1 } as Move,
        ],
      });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: "swap_sides is only available immediately after Player 1's first turn",
      });
    });

    it('should reject when swap_sides already used', () => {
      const state = createBaseState({
        moveHistory: [
          { type: 'place_ring', player: 1, to: { x: 3, y: 3 }, placementCount: 1 } as Move,
          { type: 'move_stack', player: 1, from: { x: 3, y: 3 }, to: { x: 4, y: 3 } } as Move,
          { type: 'swap_sides', player: 2, to: { x: 0, y: 0 } } as unknown as Move,
        ],
      });
      const result = validateSwapSidesMove(state, 2);
      expect(result).toEqual({
        valid: false,
        reason: "swap_sides is only available immediately after Player 1's first turn",
      });
    });
  });

  describe('applySwapSidesIdentitySwap', () => {
    const createPlayers = () => [
      {
        playerNumber: 1,
        id: 'p1-id',
        username: 'Alice',
        type: 'human' as const,
        rating: 1500,
        aiDifficulty: undefined,
        aiProfile: undefined,
      },
      {
        playerNumber: 2,
        id: 'p2-id',
        username: 'Bob',
        type: 'ai' as const,
        rating: 1600,
        aiDifficulty: 3,
        aiProfile: { style: 'aggressive' },
      },
    ];

    it('should swap user identities between Player 1 and Player 2', () => {
      const players = createPlayers();
      const result = applySwapSidesIdentitySwap(players);

      const p1 = result.find((p) => p.playerNumber === 1)!;
      const p2 = result.find((p) => p.playerNumber === 2)!;

      expect(p1.id).toBe('p2-id');
      expect(p1.username).toBe('Bob');
      expect(p1.type).toBe('ai');
      expect(p1.rating).toBe(1600);
      expect(p1.aiDifficulty).toBe(3);
      expect(p1.aiProfile).toEqual({ style: 'aggressive' });

      expect(p2.id).toBe('p1-id');
      expect(p2.username).toBe('Alice');
      expect(p2.type).toBe('human');
      expect(p2.rating).toBe(1500);
      expect(p2.aiDifficulty).toBeUndefined();
      expect(p2.aiProfile).toBeUndefined();
    });

    it('should preserve player numbers (seats)', () => {
      const players = createPlayers();
      const result = applySwapSidesIdentitySwap(players);

      expect(result.find((p) => p.playerNumber === 1)).toBeDefined();
      expect(result.find((p) => p.playerNumber === 2)).toBeDefined();
    });

    it('should return unchanged array if Player 1 is missing', () => {
      const players = [
        {
          playerNumber: 2,
          id: 'p2-id',
          username: 'Bob',
          type: 'human' as const,
          rating: 1600,
        },
      ];
      const result = applySwapSidesIdentitySwap(players);
      expect(result).toEqual(players);
    });

    it('should return unchanged array if Player 2 is missing', () => {
      const players = [
        {
          playerNumber: 1,
          id: 'p1-id',
          username: 'Alice',
          type: 'human' as const,
          rating: 1500,
        },
      ];
      const result = applySwapSidesIdentitySwap(players);
      expect(result).toEqual(players);
    });

    it('should preserve other players (Player 3, 4) unchanged', () => {
      const players = [
        ...createPlayers(),
        {
          playerNumber: 3,
          id: 'p3-id',
          username: 'Charlie',
          type: 'human' as const,
          rating: 1400,
        },
        {
          playerNumber: 4,
          id: 'p4-id',
          username: 'Diana',
          type: 'ai' as const,
          rating: 1700,
          aiDifficulty: 5,
        },
      ];
      const result = applySwapSidesIdentitySwap(players);

      const p3 = result.find((p) => p.playerNumber === 3)!;
      const p4 = result.find((p) => p.playerNumber === 4)!;

      expect(p3.id).toBe('p3-id');
      expect(p3.username).toBe('Charlie');
      expect(p4.id).toBe('p4-id');
      expect(p4.username).toBe('Diana');
    });

    it('should not mutate the original array', () => {
      const players = createPlayers();
      const originalP1Id = players[0].id;
      applySwapSidesIdentitySwap(players);
      expect(players[0].id).toBe(originalP1Id);
    });

    it('should return a new array reference', () => {
      const players = createPlayers();
      const result = applySwapSidesIdentitySwap(players);
      expect(result).not.toBe(players);
    });

    it('should handle empty array', () => {
      const result = applySwapSidesIdentitySwap([]);
      expect(result).toEqual([]);
    });
  });
});
