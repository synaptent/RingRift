/**
 * PlayerStateHelpers branch coverage tests
 * Tests for src/shared/engine/playerStateHelpers.ts
 */

import {
  playerHasMaterial,
  playerControlsAnyStack,
  playerHasActiveMaterial,
  hasAnyRealAction,
  ActionAvailabilityDelegates,
} from '../../src/shared/engine/playerStateHelpers';
import type { GameState, BoardState, RingStack } from '../../src/shared/types/game';

function createMinimalBoard(
  overrides: Partial<{
    stacks: Map<string, RingStack>;
  }>
): BoardState {
  return {
    type: 'square8',
    size: 8,
    stacks: overrides.stacks ?? new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    rings: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    geometry: { type: 'square8', size: 8 },
  } as BoardState;
}

function createMinimalState(
  overrides: Partial<{
    gameStatus: string;
    currentPlayer: number;
    stacks: Map<string, RingStack>;
    players: Array<{ playerNumber: number; ringsInHand: number; eliminated: boolean }>;
  }>
): GameState {
  const players = overrides.players ?? [
    { playerNumber: 1, ringsInHand: 10, eliminated: false },
    { playerNumber: 2, ringsInHand: 10, eliminated: false },
  ];

  return {
    board: createMinimalBoard({ stacks: overrides.stacks }),
    currentPhase: 'movement',
    currentPlayer: overrides.currentPlayer ?? 1,
    players,
    turnNumber: 1,
    gameStatus: overrides.gameStatus ?? 'active',
    moveHistory: [],
    pendingDecision: null,
    victoryCondition: null,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
  } as unknown as GameState;
}

describe('playerStateHelpers', () => {
  describe('playerHasMaterial', () => {
    it('should return true when player has rings in hand', () => {
      const state = createMinimalState({
        players: [
          { playerNumber: 1, ringsInHand: 5, eliminated: false },
          { playerNumber: 2, ringsInHand: 0, eliminated: false },
        ],
      });

      expect(playerHasMaterial(state, 1)).toBe(true);
    });

    it('should return true when player has rings on board', () => {
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [1, 2],
            stackHeight: 2,
            capHeight: 1,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        stacks,
        players: [
          { playerNumber: 1, ringsInHand: 0, eliminated: false },
          { playerNumber: 2, ringsInHand: 0, eliminated: false },
        ],
      });

      expect(playerHasMaterial(state, 1)).toBe(true);
    });

    it('should return false when player has no material', () => {
      const state = createMinimalState({
        players: [
          { playerNumber: 1, ringsInHand: 0, eliminated: false },
          { playerNumber: 2, ringsInHand: 5, eliminated: false },
        ],
      });

      expect(playerHasMaterial(state, 1)).toBe(false);
    });
  });

  describe('playerControlsAnyStack', () => {
    it('should return true when player controls at least one stack', () => {
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
        ],
        [
          '1,0',
          {
            position: { x: 1, y: 0 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
        ],
      ]);

      const board = createMinimalBoard({ stacks });

      expect(playerControlsAnyStack(board, 1)).toBe(true);
    });

    it('should return false when player controls no stacks', () => {
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
        ],
      ]);

      const board = createMinimalBoard({ stacks });

      expect(playerControlsAnyStack(board, 1)).toBe(false);
    });

    it('should return false when board has no stacks', () => {
      const board = createMinimalBoard({ stacks: new Map() });

      expect(playerControlsAnyStack(board, 1)).toBe(false);
    });
  });

  describe('playerHasActiveMaterial', () => {
    it('should return false when player not found', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 5, eliminated: false }],
      });

      expect(playerHasActiveMaterial(state, 99)).toBe(false);
    });

    it('should return true when player has rings in hand', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 5, eliminated: false }],
      });

      expect(playerHasActiveMaterial(state, 1)).toBe(true);
    });

    it('should return true when player controls a stack but has no rings in hand', () => {
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [1],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 1,
          },
        ],
      ]);

      const state = createMinimalState({
        stacks,
        players: [{ playerNumber: 1, ringsInHand: 0, eliminated: false }],
      });

      expect(playerHasActiveMaterial(state, 1)).toBe(true);
    });

    it('should return false when player has no rings in hand and controls no stacks', () => {
      const stacks = new Map<string, RingStack>([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
        ],
      ]);

      const state = createMinimalState({
        stacks,
        players: [
          { playerNumber: 1, ringsInHand: 0, eliminated: false },
          { playerNumber: 2, ringsInHand: 5, eliminated: false },
        ],
      });

      expect(playerHasActiveMaterial(state, 1)).toBe(false);
    });
  });

  describe('hasAnyRealAction', () => {
    const createDelegates = (options: {
      hasPlacement?: boolean;
      hasMovement?: boolean;
      hasCapture?: boolean;
      hasRecovery?: boolean;
    }): ActionAvailabilityDelegates => ({
      hasPlacement: () => options.hasPlacement ?? false,
      hasMovement: () => options.hasMovement ?? false,
      hasCapture: () => options.hasCapture ?? false,
      hasRecovery: () => options.hasRecovery ?? false,
    });

    it('should return false when game is not active', () => {
      const state = createMinimalState({ gameStatus: 'finished' });
      const delegates = createDelegates({
        hasPlacement: true,
        hasMovement: true,
        hasCapture: true,
      });

      expect(hasAnyRealAction(state, 1, delegates)).toBe(false);
    });

    it('should return false when player not found', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 5, eliminated: false }],
      });
      const delegates = createDelegates({ hasPlacement: true });

      expect(hasAnyRealAction(state, 99, delegates)).toBe(false);
    });

    it('should return true when player has placement available', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 5, eliminated: false }],
      });
      const delegates = createDelegates({ hasPlacement: true });

      expect(hasAnyRealAction(state, 1, delegates)).toBe(true);
    });

    it('should not check placement when player has no rings in hand', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 0, eliminated: false }],
      });
      const delegates = createDelegates({ hasPlacement: true });

      // Should return false because ringsInHand is 0, even though hasPlacement returns true
      expect(hasAnyRealAction(state, 1, delegates)).toBe(false);
    });

    it('should return true when player has movement available', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 0, eliminated: false }],
      });
      const delegates = createDelegates({ hasMovement: true });

      expect(hasAnyRealAction(state, 1, delegates)).toBe(true);
    });

    it('should return true when player has capture available', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 0, eliminated: false }],
      });
      const delegates = createDelegates({ hasCapture: true });

      expect(hasAnyRealAction(state, 1, delegates)).toBe(true);
    });

    it('should return false when player only has recovery available', () => {
      // NOTE: Recovery is intentionally NOT counted as a "real action" for LPS purposes.
      // Per the implementation comment in playerStateHelpers.ts:150-154, recovery moves
      // don't count toward preventing LPS loss - players must place/move/capture.
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 0, eliminated: false }],
      });
      const delegates = createDelegates({ hasRecovery: true });

      // Recovery alone doesn't count as a real action
      expect(hasAnyRealAction(state, 1, delegates)).toBe(false);
    });

    it('should return false when player has no actions available', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 0, eliminated: false }],
      });
      const delegates = createDelegates({
        hasPlacement: false,
        hasMovement: false,
        hasCapture: false,
        hasRecovery: false,
      });

      expect(hasAnyRealAction(state, 1, delegates)).toBe(false);
    });

    it('should prioritize placement check when player has rings in hand', () => {
      const state = createMinimalState({
        players: [{ playerNumber: 1, ringsInHand: 5, eliminated: false }],
      });

      let placementChecked = false;
      let movementChecked = false;

      const delegates: ActionAvailabilityDelegates = {
        hasPlacement: () => {
          placementChecked = true;
          return true;
        },
        hasMovement: () => {
          movementChecked = true;
          return true;
        },
        hasCapture: () => true,
      };

      const result = hasAnyRealAction(state, 1, delegates);

      expect(result).toBe(true);
      expect(placementChecked).toBe(true);
      // Movement should not be checked because placement returned true
      expect(movementChecked).toBe(false);
    });
  });
});
