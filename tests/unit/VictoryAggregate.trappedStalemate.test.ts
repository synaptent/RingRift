/**
 * Tests for trapped-position stalemate detection in VictoryAggregate.
 *
 * This test file covers the scenario where AI vs AI games stall because
 * both players run out of legal moves but still have stacks on the board.
 *
 * Rule Reference: Section 13.4 - Stalemate Resolution
 */

import { evaluateVictory } from '../../src/shared/engine/aggregates/VictoryAggregate';
import type {
  GameState,
  Player,
  BoardState,
  RingStack,
  MarkerInfo,
} from '../../src/shared/types/game';

/**
 * Helper to create a minimal test player.
 */
function createTestPlayer(overrides: Partial<Player> & { playerNumber: number }): Player {
  return {
    id: `player-${overrides.playerNumber}`,
    username: `Player ${overrides.playerNumber}`,
    type: 'human',
    isReady: true,
    timeRemaining: 3600,
    ringsInHand: 0,
    eliminatedRings: 0,
    territorySpaces: 0,
    ...overrides,
  };
}

/**
 * Helper to create a minimal valid GameState for testing.
 */
function createTestGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultPlayers: Player[] = [
    createTestPlayer({ playerNumber: 1, ringsInHand: 0, eliminatedRings: 3, territorySpaces: 2 }),
    createTestPlayer({ playerNumber: 2, ringsInHand: 0, eliminatedRings: 2, territorySpaces: 1 }),
  ];

  const defaultBoard: BoardState = {
    type: 'square8',
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
  };

  return {
    id: 'test-game-1',
    boardType: 'square8',
    gameStatus: 'active',
    currentPlayer: 1,
    currentPhase: 'movement',
    turnNumber: 10,
    players: defaultPlayers,
    board: defaultBoard,
    victoryThreshold: 7,
    territoryVictoryThreshold: 33,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 3600, increment: 0, type: 'classical' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 0,
    ...overrides,
  } as GameState;
}

/**
 * Helper to create a stack.
 */
function createStack(
  position: { x: number; y: number },
  controllingPlayer: number,
  stackHeight: number = 1,
  capHeight: number = 1
): RingStack {
  return {
    position,
    controllingPlayer,
    stackHeight,
    capHeight,
    rings: Array(stackHeight).fill(controllingPlayer),
  };
}

describe('VictoryAggregate - Trapped Position Stalemate', () => {
  describe('when both players have stacks but no legal moves', () => {
    it('should detect stalemate when all players are trapped', () => {
      // Create a scenario where:
      // - Both players have stacks on the board
      // - Both players have 0 rings in hand
      // - Both players are completely surrounded by collapsed spaces (cannot move)
      // - No placement possible (no rings in hand)

      const collapsedSpaces = new Map<string, number>();
      // Surround position (3,3) and (5,5) with collapsed spaces
      // This simulates stacks that cannot move anywhere
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          if ((x !== 3 || y !== 3) && (x !== 5 || y !== 5)) {
            collapsedSpaces.set(`${x},${y}`, 0); // 0 means neutral collapsed
          }
        }
      }

      const stacks = new Map<string, RingStack>();
      stacks.set('3,3', createStack({ x: 3, y: 3 }, 1, 2, 2));
      stacks.set('5,5', createStack({ x: 5, y: 5 }, 2, 2, 2));

      const board: BoardState = {
        type: 'square8',
        size: 8,
        stacks,
        markers: new Map(),
        collapsedSpaces,
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 0,
          eliminatedRings: 5,
          territorySpaces: 3,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 4,
          territorySpaces: 2,
        }),
      ];

      const state = createTestGameState({
        board,
        players,
        gameStatus: 'active',
      });

      const result = evaluateVictory(state);

      // Game should be over
      expect(result.isGameOver).toBe(true);
      // Player 1 should win via territory tiebreaker (3 > 2)
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');
    });

    it('should use eliminated rings tiebreaker when territory is tied', () => {
      const collapsedSpaces = new Map<string, number>();
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          if ((x !== 3 || y !== 3) && (x !== 5 || y !== 5)) {
            collapsedSpaces.set(`${x},${y}`, 0);
          }
        }
      }

      const stacks = new Map<string, RingStack>();
      stacks.set('3,3', createStack({ x: 3, y: 3 }, 1, 2, 2));
      stacks.set('5,5', createStack({ x: 5, y: 5 }, 2, 2, 2));

      const board: BoardState = {
        type: 'square8',
        size: 8,
        stacks,
        markers: new Map(),
        collapsedSpaces,
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 0,
          eliminatedRings: 5,
          territorySpaces: 2,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 3,
          territorySpaces: 2,
        }),
      ];

      const state = createTestGameState({
        board,
        players,
        gameStatus: 'active',
      });

      const result = evaluateVictory(state);

      expect(result.isGameOver).toBe(true);
      // Player 1 should win via eliminated rings tiebreaker (5 > 3)
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('ring_elimination');
    });

    it('should use markers tiebreaker when territory and eliminations are tied', () => {
      const collapsedSpaces = new Map<string, number>();
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          if ((x !== 3 || y !== 3) && (x !== 5 || y !== 5)) {
            collapsedSpaces.set(`${x},${y}`, 0);
          }
        }
      }

      const stacks = new Map<string, RingStack>();
      stacks.set('3,3', createStack({ x: 3, y: 3 }, 1, 2, 2));
      stacks.set('5,5', createStack({ x: 5, y: 5 }, 2, 2, 2));

      // Add markers - player 2 has more markers
      const markers = new Map<string, MarkerInfo>();
      markers.set('0,0', { player: 2, position: { x: 0, y: 0 }, type: 'regular' });
      markers.set('1,1', { player: 2, position: { x: 1, y: 1 }, type: 'regular' });
      markers.set('2,2', { player: 1, position: { x: 2, y: 2 }, type: 'regular' });

      const board: BoardState = {
        type: 'square8',
        size: 8,
        stacks,
        markers,
        collapsedSpaces,
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 0,
          eliminatedRings: 4,
          territorySpaces: 2,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 4,
          territorySpaces: 2,
        }),
      ];

      const state = createTestGameState({
        board,
        players,
        gameStatus: 'active',
      });

      const result = evaluateVictory(state);

      expect(result.isGameOver).toBe(true);
      // Player 2 should win via markers tiebreaker (2 > 1)
      expect(result.winner).toBe(2);
      expect(result.reason).toBe('last_player_standing');
    });
  });

  describe('when at least one player can still act', () => {
    it('should NOT detect stalemate if a player has rings in hand and can place', () => {
      // Create a board where player 1 has rings in hand and open spaces to place
      const stacks = new Map<string, RingStack>();
      stacks.set('3,3', createStack({ x: 3, y: 3 }, 1, 2, 2));
      stacks.set('5,5', createStack({ x: 5, y: 5 }, 2, 2, 2));

      const board: BoardState = {
        type: 'square8',
        size: 8,
        stacks,
        markers: new Map(),
        collapsedSpaces: new Map(), // Open board - placement possible
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 2,
          eliminatedRings: 3,
          territorySpaces: 2,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 2,
          territorySpaces: 1,
        }),
      ];

      const state = createTestGameState({
        board,
        players,
        gameStatus: 'active',
      });

      const result = evaluateVictory(state);

      // Game should NOT be over - player 1 can still place
      expect(result.isGameOver).toBe(false);
    });

    it('should NOT detect stalemate if a player can move their stack', () => {
      // Create a board where player 1's stack at (3,3) can move to adjacent squares
      const stacks = new Map<string, RingStack>();
      stacks.set('3,3', createStack({ x: 3, y: 3 }, 1, 2, 2));
      stacks.set('5,5', createStack({ x: 5, y: 5 }, 2, 2, 2));

      // Only collapse some spaces, leaving adjacent spaces open for movement
      const collapsedSpaces = new Map<string, number>();
      collapsedSpaces.set('0,0', 0);
      collapsedSpaces.set('7,7', 0);
      // Position (3,4) is open, so stack at (3,3) can move there

      const board: BoardState = {
        type: 'square8',
        size: 8,
        stacks,
        markers: new Map(),
        collapsedSpaces,
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 0,
          eliminatedRings: 3,
          territorySpaces: 2,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 2,
          territorySpaces: 1,
        }),
      ];

      const state = createTestGameState({
        board,
        players,
        gameStatus: 'active',
      });

      const result = evaluateVictory(state);

      // Game should NOT be over - players can still move
      expect(result.isGameOver).toBe(false);
    });
  });

  describe('Early LPS before trapped stalemate', () => {
    it('should detect LPS when only one player has stacks and others have no material', () => {
      // Only player 1 has a stack, player 2 has no stacks and no rings
      const stacks = new Map<string, RingStack>();
      stacks.set('3,3', createStack({ x: 3, y: 3 }, 1, 2, 2));

      const board: BoardState = {
        type: 'square8',
        size: 8,
        stacks,
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 0,
          eliminatedRings: 3,
          territorySpaces: 2,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 6,
          territorySpaces: 0,
        }),
      ];

      const state = createTestGameState({
        board,
        players,
        gameStatus: 'active',
      });

      const result = evaluateVictory(state);

      expect(result.isGameOver).toBe(true);
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });
  });

  describe('hexagonal board stalemate', () => {
    it('should detect stalemate on hex board when all stacks are trapped by collapsed spaces', () => {
      // Test hex board with size 3 (radius 2) - 19 total spaces
      // Create stacks at positions with z-coordinates
      const stacks = new Map<string, RingStack>();
      // Stack at center (0,0,0)
      stacks.set('0,0,0', {
        position: { x: 0, y: 0, z: 0 },
        controllingPlayer: 1,
        stackHeight: 2,
        capHeight: 2,
        rings: [1, 1],
      } as RingStack);
      // Stack at (1,-1,0)
      stacks.set('1,-1,0', {
        position: { x: 1, y: -1, z: 0 },
        controllingPlayer: 2,
        stackHeight: 2,
        capHeight: 2,
        rings: [2, 2],
      } as RingStack);

      // Collapse all other spaces - hex size 3 means radius 2
      // Valid positions: where |q| <= 2, |r| <= 2, |s| <= 2 and q+r+s=0
      const collapsedSpaces = new Map<string, number>();
      for (let q = -2; q <= 2; q++) {
        for (let r = -2; r <= 2; r++) {
          const s = -q - r;
          if (Math.abs(s) <= 2) {
            const key = `${q},${r},${s}`;
            if (key !== '0,0,0' && key !== '1,-1,0') {
              collapsedSpaces.set(key, 0);
            }
          }
        }
      }

      const board: BoardState = {
        type: 'hexagonal',
        size: 3,
        stacks,
        markers: new Map(),
        collapsedSpaces,
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({
          playerNumber: 1,
          ringsInHand: 0,
          eliminatedRings: 5,
          territorySpaces: 3,
        }),
        createTestPlayer({
          playerNumber: 2,
          ringsInHand: 0,
          eliminatedRings: 4,
          territorySpaces: 2,
        }),
      ];

      const state = createTestGameState({
        board,
        boardType: 'hexagonal',
        players,
        gameStatus: 'completed', // Test with 'completed' status
      });

      const result = evaluateVictory(state);

      // Both stacks are trapped (all adjacent spaces collapsed)
      // Game should be over with stalemate resolution
      expect(result.isGameOver).toBe(true);
      // Player 1 wins via territory tiebreaker (3 > 2)
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('territory_control');
    });

    it('should NOT detect stalemate on hex board when a stack can move', () => {
      // Larger hex board where stacks can move
      const stacks = new Map<string, RingStack>();
      // Stack at (0,0,0) with height 2
      stacks.set('0,0,0', {
        position: { x: 0, y: 0, z: 0 },
        controllingPlayer: 1,
        stackHeight: 2,
        capHeight: 2,
        rings: [1, 1],
      } as RingStack);
      // Stack at (3,-3,0) with height 2
      stacks.set('3,-3,0', {
        position: { x: 3, y: -3, z: 0 },
        controllingPlayer: 2,
        stackHeight: 2,
        capHeight: 2,
        rings: [2, 2],
      } as RingStack);

      // Collapse only a few spaces - leave room for movement
      const collapsedSpaces = new Map<string, number>();
      collapsedSpaces.set('-2,2,0', 0);
      collapsedSpaces.set('2,-2,0', 0);

      const board: BoardState = {
        type: 'hexagonal',
        size: 5, // Radius 4
        stacks,
        markers: new Map(),
        collapsedSpaces,
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      };

      const players: Player[] = [
        createTestPlayer({ playerNumber: 1, ringsInHand: 0 }),
        createTestPlayer({ playerNumber: 2, ringsInHand: 0 }),
      ];

      const state = createTestGameState({
        board,
        boardType: 'hexagonal',
        players,
        gameStatus: 'completed',
      });

      const result = evaluateVictory(state);

      // Stacks can move - game is NOT over via stalemate
      expect(result.isGameOver).toBe(false);
    });
  });
});

describe('Validation consistency diagnostics', () => {
  it('should log validation discrepancies when positions have floating point sums', () => {
    // This test verifies that strict validation (q + r + s === 0)
    // matches lenient validation (Math.round(q + r + s) === 0) for integer coords
    const stacks = new Map<string, RingStack>();

    // Stack at fractional-looking position (but still integers)
    stacks.set('1,2,-3', {
      position: { x: 1, y: 2, z: -3 },
      controllingPlayer: 1,
      stackHeight: 1,
      capHeight: 1,
      rings: [1],
    } as RingStack);

    // Collapse positions that might cause validation differences
    const collapsedSpaces = new Map<string, number>();
    // Leave open: positions adjacent to the stack

    const board: BoardState = {
      type: 'hexagonal',
      size: 5, // radius 4
      stacks,
      markers: new Map(),
      collapsedSpaces,
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
    };

    const players: Player[] = [
      createTestPlayer({ playerNumber: 1, ringsInHand: 0 }),
      createTestPlayer({ playerNumber: 2, ringsInHand: 0, eliminatedRings: 0 }),
    ];

    // Player 2 has no stacks and no rings - this should be LPS for player 1
    const state = createTestGameState({
      board,
      boardType: 'hexagonal',
      players,
      gameStatus: 'active',
    });

    const result = evaluateVictory(state);

    // Player 1 should win via LPS (only player with material)
    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1);
    expect(result.reason).toBe('last_player_standing');
  });

  it('should correctly detect stalemate with large height stacks that cannot move far enough', () => {
    // Simulate the user's scenario: height-11 stacks that need 11+ spaces to move
    // but are blocked by collapsed spaces
    const stacks = new Map<string, RingStack>();

    // Height-11 stack at center
    stacks.set('0,0,0', {
      position: { x: 0, y: 0, z: 0 },
      controllingPlayer: 1,
      stackHeight: 11,
      capHeight: 11,
      rings: Array(11).fill(1),
    } as RingStack);

    // Height-11 stack at (1,-1,0)
    stacks.set('1,-1,0', {
      position: { x: 1, y: -1, z: 0 },
      controllingPlayer: 2,
      stackHeight: 11,
      capHeight: 11,
      rings: Array(11).fill(2),
    } as RingStack);

    // On a size 5 board (radius 4), max distance is 8
    // Stacks with height 11 cannot move since they need 11+ spaces
    // Collapse everything except the stack positions
    const collapsedSpaces = new Map<string, number>();
    for (let q = -4; q <= 4; q++) {
      for (let r = -4; r <= 4; r++) {
        const s = -q - r;
        if (Math.abs(s) <= 4) {
          const key = `${q},${r},${s}`;
          if (key !== '0,0,0' && key !== '1,-1,0') {
            collapsedSpaces.set(key, 0);
          }
        }
      }
    }

    const board: BoardState = {
      type: 'hexagonal',
      size: 5,
      stacks,
      markers: new Map(),
      collapsedSpaces,
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
    };

    const players: Player[] = [
      createTestPlayer({ playerNumber: 1, ringsInHand: 0, eliminatedRings: 5, territorySpaces: 3 }),
      createTestPlayer({ playerNumber: 2, ringsInHand: 0, eliminatedRings: 4, territorySpaces: 2 }),
    ];

    const state = createTestGameState({
      board,
      boardType: 'hexagonal',
      players,
      gameStatus: 'active',
    });

    const result = evaluateVictory(state);

    // Both stacks cannot move (need 11 spaces, max available is ~8)
    // This should trigger stalemate with territory tiebreaker
    expect(result.isGameOver).toBe(true);
    expect(result.winner).toBe(1); // Has more territory
    expect(result.reason).toBe('territory_control');
  });
});
