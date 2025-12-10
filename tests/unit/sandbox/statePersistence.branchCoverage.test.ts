/**
 * statePersistence.branchCoverage.test.ts
 *
 * Branch coverage tests for statePersistence.ts
 */

import {
  saveCurrentGameState,
  exportScenarioToFile,
  exportGameStateToFile,
  buildTestFixtureFromGameState,
  importScenarioFromFile,
  importAndSaveScenarioFromFile,
} from '../../../src/client/sandbox/statePersistence';
import type { GameState, BoardState, Player } from '../../../src/shared/types/game';
import type { LoadableScenario } from '../../../src/client/sandbox/scenarioTypes';

// Mock dependencies
jest.mock('../../../src/client/sandbox/scenarioLoader', () => ({
  saveCustomScenario: jest.fn(),
}));

jest.mock('../../../src/shared/engine/contracts/serialization', () => ({
  serializeGameState: jest.fn((state) => ({
    id: state.id,
    boardType: state.boardType,
    board: {},
    players: state.players,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
  })),
}));

// Helper to create game states for testing
function createGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultPlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  const defaultBoard: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  return {
    id: 'test-game',
    boardType: 'square8',
    board: defaultBoard,
    players: defaultPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 0,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
    ...overrides,
  } as GameState;
}

describe('statePersistence', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('saveCurrentGameState', () => {
    it('saves game state with provided metadata', () => {
      const gameState = createGameState();
      const result = saveCurrentGameState(gameState, {
        name: 'My Test Game',
        description: 'Test description',
        category: 'custom',
        tags: ['test', 'custom'],
      });

      expect(result.name).toBe('My Test Game');
      expect(result.description).toBe('Test description');
      expect(result.category).toBe('custom');
      expect(result.tags).toEqual(['test', 'custom']);
      expect(result.id).toMatch(/^custom_\d+_[a-z0-9]+$/);
    });

    it('uses default name when not provided', () => {
      const gameState = createGameState();
      const result = saveCurrentGameState(gameState, { name: '' });

      expect(result.name).toContain('Saved Game');
    });

    it('generates default description with turn number and phase', () => {
      const gameState = createGameState({
        moveHistory: [
          {
            id: '1',
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          },
        ],
        currentPhase: 'movement',
      });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.description).toContain('turn 2');
      expect(result.description).toContain('movement phase');
    });

    it('infers category from ring_placement phase', () => {
      const gameState = createGameState({ currentPhase: 'ring_placement' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('placement');
    });

    it('infers category from movement phase', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('movement');
    });

    it('infers category from chain_capture phase', () => {
      const gameState = createGameState({ currentPhase: 'chain_capture' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('chain_capture');
    });

    it('infers category from process_line phase', () => {
      const gameState = createGameState({ currentPhase: 'process_line' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('line_processing');
    });

    it('infers category from choose_line_reward phase', () => {
      const gameState = createGameState({ currentPhase: 'choose_line_reward' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('line_processing');
    });

    it('infers category from process_territory_region phase', () => {
      const gameState = createGameState({ currentPhase: 'process_territory_region' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('territory_processing');
    });

    it('defaults to custom category for unknown phases', () => {
      const gameState = createGameState({
        currentPhase: 'unknown_phase' as GameState['currentPhase'],
      });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.category).toBe('custom');
    });

    it('generates default tags from phase', () => {
      const gameState = createGameState({ currentPhase: 'movement' });
      const result = saveCurrentGameState(gameState, { name: 'Test' });

      expect(result.tags).toContain('saved');
      expect(result.tags).toContain('movement');
    });
  });

  // Skip browser-specific export tests in Node environment
  // These functions rely on DOM APIs (createElement, URL.createObjectURL, etc.)
  // that don't exist or work differently in Node.js
  describe('exportScenarioToFile', () => {
    it.skip('creates download link with sanitized filename (browser-only)', () => {
      // This test requires browser DOM APIs
    });
  });

  describe('exportGameStateToFile', () => {
    it.skip('exports game state with given name (browser-only)', () => {
      // This test requires browser DOM APIs
    });
  });

  describe('buildTestFixtureFromGameState', () => {
    it('creates fixture with all required fields', () => {
      const gameState = createGameState({
        currentPhase: 'movement',
        currentPlayer: 2,
        moveHistory: [
          {
            id: '1',
            type: 'place_ring',
            player: 1,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          },
        ],
        history: [{ action: {}, before: {}, after: {} }] as GameState['history'],
      });
      (gameState as GameState & { rngSeed: number }).rngSeed = 12345;

      const fixture = buildTestFixtureFromGameState(gameState) as {
        kind: string;
        boardType: string;
        currentPhase: string;
        currentPlayer: number;
        rngSeed: number;
        moveHistory: unknown[];
        historyLength: number;
        debug: {
          gameStatus: string;
          perPlayerActions: Array<{
            playerNumber: number;
            summary: {
              hasTurnMaterial: boolean;
              hasGlobalPlacementAction: boolean;
              hasPhaseLocalInteractiveMove: boolean;
              hasForcedEliminationAction: boolean;
            };
          }>;
          isANMState: boolean;
          victoryProbe: { isGameOver: boolean; winner?: number; reason?: string | undefined };
        };
      };

      expect(fixture.kind).toBe('ringrift_sandbox_fixture_v1');
      expect(fixture.boardType).toBe('square8');
      expect(fixture.currentPhase).toBe('movement');
      expect(fixture.currentPlayer).toBe(2);
      expect(fixture.rngSeed).toBe(12345);
      expect(fixture.moveHistory).toHaveLength(1);
      expect(fixture.historyLength).toBe(1);
      expect(fixture.debug.gameStatus).toBe(gameState.gameStatus);
      expect(fixture.debug.perPlayerActions).toHaveLength(gameState.players.length);
    });

    it('handles null rngSeed', () => {
      const gameState = createGameState();
      // Ensure rngSeed is not set
      delete (gameState as GameState & { rngSeed?: number }).rngSeed;

      const fixture = buildTestFixtureFromGameState(gameState) as { rngSeed: number | null };

      expect(fixture.rngSeed).toBeNull();
    });

    it('handles undefined moveHistory', () => {
      const gameState = createGameState();
      // @ts-expect-error - Testing undefined case
      gameState.moveHistory = undefined;

      const fixture = buildTestFixtureFromGameState(gameState) as { moveHistory: unknown[] };

      expect(fixture.moveHistory).toEqual([]);
    });

    it('handles undefined history', () => {
      const gameState = createGameState();
      // @ts-expect-error - Testing undefined case
      gameState.history = undefined;

      const fixture = buildTestFixtureFromGameState(gameState) as { historyLength: number };

      expect(fixture.historyLength).toBe(0);
    });
  });

  // Skip File-based import tests in Node environment
  // File.text() is a browser API that doesn't work the same in Node.js
  describe('importScenarioFromFile', () => {
    it.skip('throws on invalid JSON (browser-only)', () => {});
    it.skip('throws on non-object data (browser-only)', () => {});
    it.skip('throws on null data (browser-only)', () => {});
    it.skip('throws on missing id field (browser-only)', () => {});
    it.skip('throws on missing name field (browser-only)', () => {});
    it.skip('throws on missing state field (browser-only)', () => {});
    it.skip('throws on missing boardType field (browser-only)', () => {});
    it.skip('throws on missing state.board field (browser-only)', () => {});
    it.skip('throws on missing state.players field (browser-only)', () => {});
    it.skip('throws on missing state.currentPlayer field (browser-only)', () => {});
    it.skip('throws on missing state.currentPhase field (browser-only)', () => {});
    it.skip('imports valid scenario (browser-only)', () => {});
    it.skip('handles missing optional fields (browser-only)', () => {});
  });

  describe('importAndSaveScenarioFromFile', () => {
    it.skip('imports and saves valid scenario (browser-only)', () => {});
  });
});
