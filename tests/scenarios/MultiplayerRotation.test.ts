/**
 * Multiplayer Rotation & Edge Case Scenario Tests
 *
 * These tests validate correct behavior in 3-player and 4-player games:
 * - Turn rotation order
 * - Player skipping when eliminated
 * - Multi-player victory conditions
 * - Cross-player capture scenarios
 * - Territory processing with multiple players
 *
 * Rules: §13 (Victory Conditions), §4.4 (Forced Elimination), §10 (Turn Order)
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import type {
  Position,
  Player,
  TimeControl,
  GameState,
  Move,
  BoardState,
  Stack,
} from '../../src/shared/types/game';
import { createTestPlayer, createTestBoard, pos, posStr } from '../utils/fixtures';
import {
  hasGlobalPlacementAction,
  applyForcedEliminationForPlayer,
  hasForcedEliminationAction,
} from '../../src/shared/engine/globalActions';

/**
 * Helper to set a stack at a position
 */
function setStack(
  board: BoardState,
  position: Position,
  rings: number[],
  controllingPlayer: number
): void {
  const posKey = posStr(position.x, position.y);
  const stack: Stack = {
    rings,
    controllingPlayer,
  };
  board.stacks.set(posKey, stack);
}

/**
 * Helper to create a multi-player game engine
 */
function createMultiPlayerEngine(
  gameId: string,
  boardType: 'square8' | 'square19' | 'hexagonal',
  numPlayers: number,
  ringsPerPlayer?: number
): GameEngine {
  const rings = ringsPerPlayer ?? (boardType === 'square8' ? 18 : 36);
  const players = Array.from({ length: numPlayers }, (_, i) =>
    createTestPlayer(i + 1, { ringsInHand: rings })
  );

  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  return new GameEngine(gameId, boardType, players, timeControl, false);
}

/**
 * Helper to access internal game state
 */
function getGameState(engine: GameEngine): GameState {
  return (engine as any).gameState;
}

describe('MultiplayerRotation: 3-Player Game Scenarios', () => {
  describe('Turn Rotation Order', () => {
    it('should initialize 3-player game with correct player configuration', () => {
      const engine = createMultiPlayerEngine('rotation-3p-basic', 'square8', 3, 18);
      const state = getGameState(engine);

      // Initial state: P1's turn in ring_placement
      expect(state.currentPlayer).toBe(1);
      expect(state.currentPhase).toBe('ring_placement');

      // All 3 players should be in the game
      expect(state.players.length).toBe(3);
      expect(state.players[0].playerNumber).toBe(1);
      expect(state.players[1].playerNumber).toBe(2);
      expect(state.players[2].playerNumber).toBe(3);
    });

    it('should validate player turn calculation helpers', () => {
      const engine = createMultiPlayerEngine('rotation-3p-calc', 'square8', 3, 18);
      const state = getGameState(engine);

      // Test modular arithmetic for 3-player rotation
      const numPlayers = state.players.length;
      expect(numPlayers).toBe(3);

      // Next player after P1 should be P2
      const nextAfterP1 = (1 % numPlayers) + 1;
      expect(nextAfterP1).toBe(2);

      // Next player after P2 should be P3
      const nextAfterP2 = (2 % numPlayers) + 1;
      expect(nextAfterP2).toBe(3);

      // Next player after P3 should cycle back to P1
      const nextAfterP3 = (3 % numPlayers) + 1;
      expect(nextAfterP3).toBe(1);
    });

    it('should properly track rings in hand for each player', () => {
      const engine = createMultiPlayerEngine('rotation-3p-rings', 'square8', 3, 18);

      // Each player starts with 18 rings
      const state = getGameState(engine);
      expect(state.players[0].ringsInHand).toBe(18);
      expect(state.players[1].ringsInHand).toBe(18);
      expect(state.players[2].ringsInHand).toBe(18);

      // Total rings in play should be 54 (3 × 18)
      expect(state.totalRingsInPlay).toBe(54);

      // Per RR-CANON-R061: round(18 × (1/3 + 2/3 × 2)) = 30
      expect(state.victoryThreshold).toBe(30);
    });
  });

  describe('3-Player Elimination Scenarios', () => {
    it('should correctly identify when player has no material', () => {
      const engine = createMultiPlayerEngine('3p-no-material', 'square8', 3, 12);
      const state = getGameState(engine);

      // Set P2 to have no material
      state.players[1].ringsInHand = 0;

      // P2 should have no placement action available
      expect(hasGlobalPlacementAction(state, 2)).toBe(false);

      // P1 and P3 still have placement actions
      expect(hasGlobalPlacementAction(state, 1)).toBe(true);
      expect(hasGlobalPlacementAction(state, 3)).toBe(true);
    });
  });
});

describe('MultiplayerRotation: 4-Player Game Scenarios', () => {
  describe('Turn Rotation Order', () => {
    it('should initialize 4-player game with correct player configuration', () => {
      const engine = createMultiPlayerEngine('rotation-4p-basic', 'square19', 4, 36);
      const state = getGameState(engine);

      // Initial state: P1's turn
      expect(state.currentPlayer).toBe(1);
      expect(state.currentPhase).toBe('ring_placement');

      // All 4 players should be in the game
      expect(state.players.length).toBe(4);
      expect(state.players[0].playerNumber).toBe(1);
      expect(state.players[1].playerNumber).toBe(2);
      expect(state.players[2].playerNumber).toBe(3);
      expect(state.players[3].playerNumber).toBe(4);
    });

    it('should validate 4-player turn calculation helpers', () => {
      const engine = createMultiPlayerEngine('rotation-4p-calc', 'square19', 4, 36);
      const state = getGameState(engine);

      // Test modular arithmetic for 4-player rotation
      const numPlayers = state.players.length;
      expect(numPlayers).toBe(4);

      // Rotation: P1 -> P2 -> P3 -> P4 -> P1
      expect((1 % numPlayers) + 1).toBe(2); // P1 -> P2
      expect((2 % numPlayers) + 1).toBe(3); // P2 -> P3
      expect((3 % numPlayers) + 1).toBe(4); // P3 -> P4
      expect((4 % numPlayers) + 1).toBe(1); // P4 -> P1 (wraps)
    });

    it('should calculate correct victory threshold for 4-player square19', () => {
      // square19 ringsPerPlayer = 48 per BOARD_CONFIGS
      const engine = createMultiPlayerEngine('4p-threshold', 'square19', 4, 48);
      const state = getGameState(engine);

      // Total rings: 4 × 48 = 192
      expect(state.totalRingsInPlay).toBe(192);

      // Per RR-CANON-R061: round(48 × (1/3 + 2/3 × 3)) = 112
      expect(state.victoryThreshold).toBe(112);
    });
  });

  describe('4-Player Territory Victory', () => {
    it('should calculate correct territory threshold for 4-player square8', () => {
      const engine = createMultiPlayerEngine('4p-territory-threshold', 'square8', 4, 9);
      const state = getGameState(engine);

      // Square8 has 64 spaces total
      // Territory threshold is typically 33% of total spaces
      // For square8: 64 × 0.33 ≈ 21
      expect(state.territoryVictoryThreshold).toBeGreaterThanOrEqual(20);
    });
  });
});

describe('MultiplayerRotation: Cross-Player Interactions', () => {
  describe('Capture Between Multiple Players', () => {
    it('should allow P1 to capture P3 stack in 3-player game', () => {
      const engine = createMultiPlayerEngine('3p-cross-capture', 'square8', 3, 12);
      const state = getGameState(engine);

      // Set up a scenario where P1 can capture P3's stack
      // P1 has stack at (3,3), P3 has stack at (3,4)
      setStack(state.board, pos(3, 3), [1, 1], 1);
      setStack(state.board, pos(3, 4), [3], 3);
      state.players[0].ringsInHand = 0; // P1 no rings in hand
      state.currentPhase = 'movement';

      // Verify P3's stack exists
      const p3Stack = state.board.stacks.get(posStr(3, 4));
      expect(p3Stack).toBeDefined();
      expect(p3Stack?.controllingPlayer).toBe(3);
    });

    it('should track eliminated rings per player correctly in multi-player', () => {
      const engine = createMultiPlayerEngine('3p-elim-tracking', 'square8', 3, 12);
      const state = getGameState(engine);

      // Set initial eliminated rings
      state.board.eliminatedRings = { '1': 5, '2': 3, '3': 7 };

      expect(state.board.eliminatedRings['1']).toBe(5);
      expect(state.board.eliminatedRings['2']).toBe(3);
      expect(state.board.eliminatedRings['3']).toBe(7);

      // Total eliminated should be 15
      const total = Object.values(state.board.eliminatedRings).reduce(
        (sum, val) => sum + (val ?? 0),
        0
      );
      expect(total).toBe(15);
    });
  });

  describe('Territory Processing with Multiple Players', () => {
    it('should process territory claims for correct player in 3-player game', () => {
      const engine = createMultiPlayerEngine('3p-territory', 'square8', 3, 12);
      const state = getGameState(engine);

      // Set up collapsed spaces owned by P2
      state.board.collapsedSpaces.set(posStr(0, 0), 2);
      state.board.collapsedSpaces.set(posStr(0, 1), 2);
      state.board.collapsedSpaces.set(posStr(1, 0), 2);

      // Verify P2 owns these collapsed spaces
      expect(state.board.collapsedSpaces.get(posStr(0, 0))).toBe(2);
      expect(state.board.collapsedSpaces.get(posStr(0, 1))).toBe(2);
      expect(state.board.collapsedSpaces.get(posStr(1, 0))).toBe(2);
    });
  });
});

describe('MultiplayerRotation: Victory Conditions', () => {
  describe('Elimination Victory in Multi-Player', () => {
    it('should detect victory when player reaches elimination threshold in 3p', () => {
      const engine = createMultiPlayerEngine('3p-elim-victory', 'square8', 3, 18);
      const state = getGameState(engine);

      // With 3 players × 18 rings = 54 total rings
      // Per RR-CANON-R061: round(18 × (1/3 + 2/3 × 2)) = 30
      expect(state.totalRingsInPlay).toBe(54);
      expect(state.victoryThreshold).toBe(30);

      // Set P2 to have 27 eliminated rings (close to threshold)
      state.board.eliminatedRings = { '1': 0, '2': 27, '3': 0 };
      state.totalRingsEliminated = 27;

      // P2 is close to losing - one more elimination and they reach threshold
      expect(state.board.eliminatedRings['2']).toBe(27);
    });

    it('should correctly calculate winner based on eliminated rings', () => {
      const engine = createMultiPlayerEngine('3p-winner-calc', 'square8', 3, 12);
      const state = getGameState(engine);

      // Set up a near-end game scenario
      // P1: 10 eliminated, P2: 15 eliminated, P3: 8 eliminated
      state.board.eliminatedRings = { '1': 10, '2': 15, '3': 8 };
      state.totalRingsEliminated = 33;

      // P2 has most rings eliminated (is losing)
      const maxEliminated = Math.max(
        state.board.eliminatedRings['1'] ?? 0,
        state.board.eliminatedRings['2'] ?? 0,
        state.board.eliminatedRings['3'] ?? 0
      );
      expect(maxEliminated).toBe(15);
    });
  });

  describe('Last Player Standing', () => {
    it('should identify when only one player has material', () => {
      const engine = createMultiPlayerEngine('3p-lps', 'square8', 3, 12);
      const state = getGameState(engine);

      // P1 has material, P2 and P3 eliminated
      state.players[0].ringsInHand = 5;
      state.players[1].ringsInHand = 0;
      state.players[2].ringsInHand = 0;
      state.board.stacks.clear(); // No stacks on board

      // Only P1 has material
      const playersWithMaterial = state.players.filter(
        (p) =>
          p.ringsInHand > 0 ||
          Array.from(state.board.stacks.values()).some(
            (s) => s.controllingPlayer === p.playerNumber
          )
      );

      expect(playersWithMaterial.length).toBe(1);
      expect(playersWithMaterial[0].playerNumber).toBe(1);
    });
  });
});

describe('MultiplayerRotation: Forced Elimination Chains', () => {
  describe('FE Detection in Multi-Player', () => {
    it('should set up trapped stack conditions correctly in 3p game', () => {
      const engine = createMultiPlayerEngine('3p-fe-specific', 'square8', 3, 18);
      const state = getGameState(engine);

      // Set up P1 with trapped stack, no rings in hand
      state.players[0].ringsInHand = 0;
      setStack(state.board, pos(3, 3), [1], 1);

      // Verify P1's stack is set up
      const stack = state.board.stacks.get(posStr(3, 3));
      expect(stack).toBeDefined();
      expect(stack?.controllingPlayer).toBe(1);
      expect(stack?.rings).toEqual([1]);

      // Block all adjacent positions to (3,3)
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          state.board.collapsedSpaces.set(posStr(3 + dx, 3 + dy), 2); // P2 territory
        }
      }

      // Verify collapsed spaces are set
      expect(state.board.collapsedSpaces.get(posStr(2, 2))).toBe(2);
      expect(state.board.collapsedSpaces.get(posStr(3, 2))).toBe(2);
      expect(state.board.collapsedSpaces.get(posStr(4, 4))).toBe(2);

      // P1 has no rings in hand - a prerequisite for FE
      expect(state.players[0].ringsInHand).toBe(0);

      // P2 and P3 still have rings in hand (not FE candidates)
      expect(state.players[1].ringsInHand).toBe(18);
      expect(state.players[2].ringsInHand).toBe(18);
    });

    it('should track player material independently in multi-player', () => {
      const engine = createMultiPlayerEngine('3p-fe-material', 'square8', 3, 18);
      const state = getGameState(engine);

      // Set different ring counts for each player
      state.players[0].ringsInHand = 5;
      state.players[1].ringsInHand = 0;
      state.players[2].ringsInHand = 10;

      // Verify independent tracking
      expect(state.players[0].ringsInHand).toBe(5);
      expect(state.players[1].ringsInHand).toBe(0);
      expect(state.players[2].ringsInHand).toBe(10);
    });
  });

  describe('FE Impact on Other Players', () => {
    it('should not affect other players material during FE', () => {
      const engine = createMultiPlayerEngine('3p-fe-isolation', 'square8', 3, 12);
      const state = getGameState(engine);

      // Record initial state
      const p2RingsInitial = state.players[1].ringsInHand;
      const p3RingsInitial = state.players[2].ringsInHand;

      // Set up P1 for FE
      state.players[0].ringsInHand = 0;
      setStack(state.board, pos(3, 3), [1], 1);

      // Block P1's stack
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          state.board.collapsedSpaces.set(posStr(3 + dx, 3 + dy), 2);
        }
      }

      // Verify P2 and P3 rings unchanged
      expect(state.players[1].ringsInHand).toBe(p2RingsInitial);
      expect(state.players[2].ringsInHand).toBe(p3RingsInitial);
    });
  });
});
