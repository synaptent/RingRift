/**
 * Serialization Utilities Unit Tests
 *
 * Tests for contracts/serialization.ts covering:
 * - serializeBoardState / deserializeBoardState
 * - serializeGameState / deserializeGameState
 * - gameStateToJson / jsonToGameState
 * - createTestVector
 * - computeStateDiff
 */

import {
  serializeBoardState,
  deserializeBoardState,
  serializeGameState,
  deserializeGameState,
  gameStateToJson,
  jsonToGameState,
  createTestVector,
  computeStateDiff,
} from '../../../src/shared/engine/contracts/serialization';
import type {
  BoardState,
  GameState,
  Move,
  RingStack,
  MarkerInfo,
} from '../../../src/shared/types/game';

describe('serialization', () => {
  // Helper to create a minimal board state
  const createMinimalBoardState = (): BoardState =>
    ({
      type: 'square8',
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: { 1: 0, 2: 0 },
    }) as BoardState;

  // Helper to create a populated board state
  const createPopulatedBoardState = (): BoardState => {
    const stacks = new Map<string, RingStack>();
    stacks.set('3,3', {
      position: { x: 3, y: 3 },
      rings: [1, 2, 1],
      stackHeight: 3,
      capHeight: 1,
      controllingPlayer: 1,
    });
    stacks.set('4,4', {
      position: { x: 4, y: 4 },
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    });

    const markers = new Map<string, MarkerInfo>();
    markers.set('5,5', {
      position: { x: 5, y: 5 },
      player: 1,
      type: 'regular',
    });
    markers.set('6,6', {
      position: { x: 6, y: 6 },
      player: 2,
      type: 'collapsed',
    });

    const collapsedSpaces = new Map<string, number>();
    collapsedSpaces.set('7,7', 1);

    return {
      type: 'square8',
      size: 8,
      stacks,
      markers,
      collapsedSpaces,
      territories: new Map(),
      formedLines: [
        {
          player: 1,
          length: 4,
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          direction: { x: 1, y: 0 },
        },
      ],
      eliminatedRings: { 1: 2, 2: 3 },
    } as BoardState;
  };

  // Helper to create a minimal game state
  const createMinimalGameState = (): GameState =>
    ({
      id: 'game-123',
      boardType: 'square8',
      board: createMinimalBoardState(),
      players: [
        {
          id: 'p1',
          username: 'Player1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'Player2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ],
      currentPlayer: 1,
      currentPhase: 'ring_placement',
      moveHistory: [],
      history: [],
      gameStatus: 'active',
      winner: undefined,
      timeControl: { initialTime: 600, increment: 5, type: 'rapid' },
      spectators: [],
      createdAt: new Date('2024-01-01'),
      lastMoveAt: new Date('2024-01-01'),
      isRated: true,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
      territoryVictoryThreshold: 33,
    }) as GameState;

  // Helper to create a game state with moves
  const createGameStateWithMoves = (): GameState => {
    const state = createMinimalGameState();
    state.board = createPopulatedBoardState();
    state.moveHistory = [
      {
        id: 'move-1',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date('2024-01-01T10:00:00Z'),
        thinkTime: 1500,
        moveNumber: 1,
      } as Move,
      {
        id: 'move-2',
        type: 'place_ring',
        player: 2,
        to: { x: 4, y: 4 },
        timestamp: new Date('2024-01-01T10:00:30Z'),
        thinkTime: 2000,
        moveNumber: 2,
      } as Move,
    ];
    state.totalRingsEliminated = 5;
    return state;
  };

  describe('serializeBoardState', () => {
    it('should serialize an empty board state', () => {
      const board = createMinimalBoardState();
      const serialized = serializeBoardState(board);

      expect(serialized.type).toBe('square8');
      expect(serialized.size).toBe(8);
      expect(Object.keys(serialized.stacks)).toHaveLength(0);
      expect(Object.keys(serialized.markers)).toHaveLength(0);
      expect(Object.keys(serialized.collapsedSpaces)).toHaveLength(0);
    });

    it('should serialize stacks correctly', () => {
      const board = createPopulatedBoardState();
      const serialized = serializeBoardState(board);

      expect(Object.keys(serialized.stacks)).toHaveLength(2);
      expect(serialized.stacks['3,3']).toEqual({
        position: { x: 3, y: 3 },
        rings: [1, 2, 1],
        stackHeight: 3,
        capHeight: 1,
        controllingPlayer: 1,
      });
      expect(serialized.stacks['4,4'].controllingPlayer).toBe(2);
    });

    it('should serialize markers correctly', () => {
      const board = createPopulatedBoardState();
      const serialized = serializeBoardState(board);

      expect(Object.keys(serialized.markers)).toHaveLength(2);
      expect(serialized.markers['5,5']).toEqual({
        position: { x: 5, y: 5 },
        player: 1,
        type: 'regular',
      });
      expect(serialized.markers['6,6'].type).toBe('collapsed');
    });

    it('should serialize collapsedSpaces correctly', () => {
      const board = createPopulatedBoardState();
      const serialized = serializeBoardState(board);

      expect(Object.keys(serialized.collapsedSpaces)).toHaveLength(1);
      expect(serialized.collapsedSpaces['7,7']).toBe(1);
    });

    it('should serialize eliminatedRings correctly', () => {
      const board = createPopulatedBoardState();
      const serialized = serializeBoardState(board);

      expect(serialized.eliminatedRings).toEqual({ 1: 2, 2: 3 });
    });

    it('should serialize formedLines correctly', () => {
      const board = createPopulatedBoardState();
      const serialized = serializeBoardState(board);

      expect(serialized.formedLines).toHaveLength(1);
    });

    it('should handle board with no formedLines', () => {
      const board = createMinimalBoardState();
      (board as { formedLines?: unknown[] }).formedLines = undefined;
      const serialized = serializeBoardState(board);

      expect(serialized.formedLines).toEqual([]);
    });
  });

  describe('deserializeBoardState', () => {
    it('should deserialize an empty board state', () => {
      const serialized = {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: { 1: 0, 2: 0 },
      };

      const board = deserializeBoardState(serialized);

      expect(board.type).toBe('square8');
      expect(board.size).toBe(8);
      expect(board.stacks.size).toBe(0);
      expect(board.markers.size).toBe(0);
      expect(board.collapsedSpaces.size).toBe(0);
      expect(board.territories.size).toBe(0);
    });

    it('should deserialize stacks correctly', () => {
      const serialized = {
        type: 'square8',
        size: 8,
        stacks: {
          '3,3': {
            position: { x: 3, y: 3 },
            rings: [1, 2, 1],
            stackHeight: 3,
            capHeight: 1,
            controllingPlayer: 1,
          },
        },
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: {},
      };

      const board = deserializeBoardState(serialized);

      expect(board.stacks.size).toBe(1);
      const stack = board.stacks.get('3,3');
      expect(stack?.rings).toEqual([1, 2, 1]);
      expect(stack?.controllingPlayer).toBe(1);
    });

    it('should deserialize markers correctly', () => {
      const serialized = {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {
          '5,5': {
            position: { x: 5, y: 5 },
            player: 1,
            type: 'regular',
          },
        },
        collapsedSpaces: {},
        eliminatedRings: {},
      };

      const board = deserializeBoardState(serialized);

      expect(board.markers.size).toBe(1);
      const marker = board.markers.get('5,5');
      expect(marker?.player).toBe(1);
      expect(marker?.type).toBe('regular');
    });

    it('should deserialize collapsedSpaces correctly', () => {
      const serialized = {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: { '7,7': 1 },
        eliminatedRings: {},
      };

      const board = deserializeBoardState(serialized);

      expect(board.collapsedSpaces.size).toBe(1);
      expect(board.collapsedSpaces.get('7,7')).toBe(1);
    });

    it('should handle missing formedLines', () => {
      const serialized = {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: {},
        // formedLines is missing
      };

      const board = deserializeBoardState(serialized);

      expect(board.formedLines).toEqual([]);
    });

    it('should deserialize formedLines when present', () => {
      const serialized = {
        type: 'square8',
        size: 8,
        stacks: {},
        markers: {},
        collapsedSpaces: {},
        eliminatedRings: {},
        formedLines: [{ player: 1, length: 4 }],
      };

      const board = deserializeBoardState(serialized);

      expect(board.formedLines).toHaveLength(1);
    });
  });

  describe('serializeGameState', () => {
    it('should serialize a minimal game state', () => {
      const state = createMinimalGameState();
      const serialized = serializeGameState(state);

      expect(serialized.gameId).toBe('game-123');
      expect(serialized.currentPlayer).toBe(1);
      expect(serialized.currentPhase).toBe('ring_placement');
      expect(serialized.gameStatus).toBe('active');
      expect(serialized.victoryThreshold).toBe(19);
      expect(serialized.territoryVictoryThreshold).toBe(33);
    });

    it('should calculate turnNumber from moveHistory length', () => {
      const state = createMinimalGameState();
      state.moveHistory = [];
      let serialized = serializeGameState(state);
      expect(serialized.turnNumber).toBe(1);

      state.moveHistory = [{ id: 'm1' } as Move, { id: 'm2' } as Move];
      serialized = serializeGameState(state);
      expect(serialized.turnNumber).toBe(3);
    });

    it('should serialize players correctly', () => {
      const state = createMinimalGameState();
      const serialized = serializeGameState(state);

      expect(serialized.players).toHaveLength(2);
      expect(serialized.players[0]).toEqual({
        playerNumber: 1,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
        isActive: true,
      });
    });

    it('should serialize moveHistory with content', () => {
      const state = createGameStateWithMoves();
      const serialized = serializeGameState(state);

      expect(serialized.moveHistory).toHaveLength(2);
      expect(serialized.moveHistory[0].id).toBe('move-1');
      expect(serialized.moveHistory[1].id).toBe('move-2');
    });

    it('should include totalRingsEliminated', () => {
      const state = createGameStateWithMoves();
      const serialized = serializeGameState(state);

      expect(serialized.totalRingsEliminated).toBe(5);
    });

    it('should include chainCapturePosition when present', () => {
      const state = createMinimalGameState();
      state.chainCapturePosition = { x: 5, y: 5 };
      const serialized = serializeGameState(state);

      expect(serialized.chainCapturePosition).toEqual({ x: 5, y: 5 });
    });

    it('should not include chainCapturePosition when absent', () => {
      const state = createMinimalGameState();
      state.chainCapturePosition = undefined;
      const serialized = serializeGameState(state);

      expect(serialized.chainCapturePosition).toBeUndefined();
    });

    it('should handle empty moveHistory', () => {
      const state = createMinimalGameState();
      state.moveHistory = [];
      const serialized = serializeGameState(state);

      expect(serialized.turnNumber).toBe(1);
      expect(serialized.moveHistory).toEqual([]);
    });
  });

  describe('deserializeGameState', () => {
    it('should deserialize a minimal serialized state', () => {
      const serialized = {
        gameId: 'game-456',
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
      };

      const state = deserializeGameState(serialized);

      expect(state.id).toBe('game-456');
      expect(state.boardType).toBe('square8');
      expect(state.currentPlayer).toBe(1);
      expect(state.currentPhase).toBe('ring_placement');
      expect(state.gameStatus).toBe('active');
    });

    it('should handle missing gameId', () => {
      const serialized = {
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
      };

      const state = deserializeGameState(serialized);

      expect(state.id).toBe('');
    });

    it('should deserialize players with isActive flag', () => {
      const serialized = {
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          {
            playerNumber: 1,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
            isActive: true,
          },
          {
            playerNumber: 2,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
            isActive: false,
          },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
      };

      const state = deserializeGameState(serialized);

      expect(state.players[0].type).toBe('human'); // isActive: true -> 'human'
      expect(state.players[1].type).toBe('ai'); // isActive: false -> 'ai'
    });

    it('should deserialize moveHistory correctly', () => {
      const serialized = {
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 17, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 2,
        currentPhase: 'ring_placement',
        turnNumber: 2,
        moveHistory: [
          {
            id: 'move-1',
            type: 'place_ring',
            player: 1,
            to: { x: 3, y: 3 },
            timestamp: '2024-01-01T10:00:00Z',
            thinkTime: 1500,
            moveNumber: 1,
          },
        ],
        gameStatus: 'active',
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
      };

      const state = deserializeGameState(serialized);

      expect(state.moveHistory).toHaveLength(1);
      expect(state.moveHistory[0].id).toBe('move-1');
    });

    it('should handle missing totalRingsEliminated', () => {
      const serialized = {
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        turnNumber: 1,
        moveHistory: [],
        gameStatus: 'active',
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
      };

      const state = deserializeGameState(serialized);

      expect(state.totalRingsEliminated).toBe(0);
    });

    it('should include chainCapturePosition when present', () => {
      const serialized = {
        board: {
          type: 'square8',
          size: 8,
          stacks: {},
          markers: {},
          collapsedSpaces: {},
          eliminatedRings: {},
        },
        players: [
          { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
          { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
        ],
        currentPlayer: 1,
        currentPhase: 'chain_capture',
        chainCapturePosition: { x: 4, y: 4 },
        turnNumber: 5,
        moveHistory: [],
        gameStatus: 'active',
        victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
        territoryVictoryThreshold: 33,
      };

      const state = deserializeGameState(serialized);

      expect(state.chainCapturePosition).toEqual({ x: 4, y: 4 });
    });
  });

  describe('gameStateToJson', () => {
    it('should convert game state to JSON string', () => {
      const state = createMinimalGameState();
      const json = gameStateToJson(state);

      expect(typeof json).toBe('string');
      const parsed = JSON.parse(json);
      expect(parsed.gameId).toBe('game-123');
      expect(parsed.currentPlayer).toBe(1);
    });

    it('should produce formatted JSON', () => {
      const state = createMinimalGameState();
      const json = gameStateToJson(state);

      // Should have newlines (formatted with indent 2)
      expect(json).toContain('\n');
    });

    it('should handle complex game states', () => {
      const state = createGameStateWithMoves();
      const json = gameStateToJson(state);

      const parsed = JSON.parse(json);
      expect(parsed.moveHistory).toHaveLength(2);
      expect(Object.keys(parsed.board.stacks)).toHaveLength(2);
    });
  });

  describe('jsonToGameState', () => {
    it('should convert JSON string to game state', () => {
      const json = `{
        "gameId": "game-789",
        "board": {
          "type": "square8",
          "size": 8,
          "stacks": {},
          "markers": {},
          "collapsedSpaces": {},
          "eliminatedRings": {}
        },
        "players": [
          { "playerNumber": 1, "ringsInHand": 18, "eliminatedRings": 0, "territorySpaces": 0 },
          { "playerNumber": 2, "ringsInHand": 18, "eliminatedRings": 0, "territorySpaces": 0 }
        ],
        "currentPlayer": 1,
        "currentPhase": "ring_placement",
        "turnNumber": 1,
        "moveHistory": [],
        "gameStatus": "active",
        "victoryThreshold": 18,
        "territoryVictoryThreshold": 33
      }`;

      const state = jsonToGameState(json);

      expect(state.id).toBe('game-789');
      expect(state.currentPlayer).toBe(1);
      expect(state.board.stacks.size).toBe(0);
    });

    it('should roundtrip correctly', () => {
      const originalState = createGameStateWithMoves();
      const json = gameStateToJson(originalState);
      const restoredState = jsonToGameState(json);

      expect(restoredState.id).toBe(originalState.id);
      expect(restoredState.currentPlayer).toBe(originalState.currentPlayer);
      expect(restoredState.moveHistory.length).toBe(originalState.moveHistory.length);
    });
  });

  describe('createTestVector', () => {
    it('should create a test vector with all required fields', () => {
      const state = createMinimalGameState();
      const move: Move = {
        id: 'test-move',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 1000,
        moveNumber: 1,
      };
      const assertions = {
        currentPlayerChanged: true,
        newCurrentPlayer: 2,
      };

      const vector = createTestVector(
        'test-001',
        'placement',
        'Test ring placement',
        state,
        move,
        assertions
      );

      expect(vector).toMatchObject({
        id: 'test-001',
        category: 'placement',
        description: 'Test ring placement',
        input: {
          move: move,
        },
        expectedOutput: {
          status: 'complete',
          assertions: assertions,
        },
        tags: [],
        source: 'generated',
      });
      expect((vector as { createdAt: string }).createdAt).toBeDefined();
    });

    it('should serialize the input state', () => {
      const state = createMinimalGameState();
      const move: Move = {
        id: 'test-move',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 1000,
        moveNumber: 1,
      };

      const vector = createTestVector('test-002', 'test', 'Test', state, move, {});

      const input = (vector as { input: { state: { gameId: string } } }).input;
      expect(input.state.gameId).toBe('game-123');
    });
  });

  describe('computeStateDiff', () => {
    it('should detect currentPlayer change', () => {
      const before = createMinimalGameState();
      const after = { ...createMinimalGameState(), currentPlayer: 2 };

      const diff = computeStateDiff(before, after);

      expect(diff.currentPlayerChanged).toBe(true);
      expect(diff.newCurrentPlayer).toBe(2);
    });

    it('should not include currentPlayerChanged when unchanged', () => {
      const before = createMinimalGameState();
      const after = createMinimalGameState();

      const diff = computeStateDiff(before, after);

      expect(diff.currentPlayerChanged).toBeUndefined();
    });

    it('should detect currentPhase change', () => {
      const before = createMinimalGameState();
      const after = { ...createMinimalGameState(), currentPhase: 'movement' as const };

      const diff = computeStateDiff(before, after);

      expect(diff.currentPhaseChanged).toBe(true);
      expect(diff.newCurrentPhase).toBe('movement');
    });

    it('should detect gameStatus change', () => {
      const before = createMinimalGameState();
      const after = { ...createMinimalGameState(), gameStatus: 'finished' as const };

      const diff = computeStateDiff(before, after);

      expect(diff.gameStatusChanged).toBe(true);
      expect(diff.newGameStatus).toBe('finished');
    });

    it('should calculate stack count delta', () => {
      const before = createMinimalGameState();
      const after = createMinimalGameState();
      after.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      const diff = computeStateDiff(before, after);

      expect(diff.stackCountDelta).toBe(1);
    });

    it('should calculate marker count delta', () => {
      const before = createMinimalGameState();
      before.board.markers.set('1,1', {
        position: { x: 1, y: 1 },
        player: 1,
        type: 'regular',
      });
      const after = createMinimalGameState();

      const diff = computeStateDiff(before, after);

      expect(diff.markerCountDelta).toBe(-1);
    });

    it('should calculate collapsed count delta', () => {
      const before = createMinimalGameState();
      const after = createMinimalGameState();
      after.board.collapsedSpaces.set('5,5', 1);
      after.board.collapsedSpaces.set('6,6', 2);

      const diff = computeStateDiff(before, after);

      expect(diff.collapsedCountDelta).toBe(2);
    });

    it('should calculate S-invariant delta', () => {
      const before = createMinimalGameState();
      const after = createMinimalGameState();
      // Add a stack to change S-invariant
      after.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      const diff = computeStateDiff(before, after);

      expect(typeof diff.sInvariantDelta).toBe('number');
    });

    it('should handle multiple changes at once', () => {
      const before = createMinimalGameState();
      const after = createGameStateWithMoves();
      after.currentPlayer = 2;
      after.currentPhase = 'movement';
      after.gameStatus = 'active';

      const diff = computeStateDiff(before, after);

      expect(diff.currentPlayerChanged).toBe(true);
      expect(diff.currentPhaseChanged).toBe(true);
      expect(diff.stackCountDelta).toBe(2); // createPopulatedBoardState adds 2 stacks
    });
  });

  describe('roundtrip integrity', () => {
    it('should preserve board state through serialize/deserialize', () => {
      const original = createPopulatedBoardState();
      const serialized = serializeBoardState(original);
      const deserialized = deserializeBoardState(serialized);

      expect(deserialized.type).toBe(original.type);
      expect(deserialized.size).toBe(original.size);
      expect(deserialized.stacks.size).toBe(original.stacks.size);
      expect(deserialized.markers.size).toBe(original.markers.size);
      expect(deserialized.collapsedSpaces.size).toBe(original.collapsedSpaces.size);
    });

    it('should preserve game state through serialize/deserialize', () => {
      const original = createGameStateWithMoves();
      const serialized = serializeGameState(original);
      const deserialized = deserializeGameState(serialized);

      expect(deserialized.id).toBe(original.id);
      expect(deserialized.currentPlayer).toBe(original.currentPlayer);
      expect(deserialized.currentPhase).toBe(original.currentPhase);
      expect(deserialized.gameStatus).toBe(original.gameStatus);
      expect(deserialized.moveHistory.length).toBe(original.moveHistory.length);
    });
  });
});
