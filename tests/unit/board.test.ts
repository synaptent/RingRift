/**
 * Sample Unit Test - BoardState and Fixtures
 * Verifies test framework configuration is working correctly
 */

import {
  createTestBoard,
  createTestPlayer,
  createTestGameState,
  addStack,
  addMarker,
  addCollapsedSpace,
  pos,
  BOARD_CONFIGS,
} from '../utils/fixtures';

describe('BoardState Test Fixtures', () => {
  describe('createTestBoard', () => {
    it('should create a square8 board with correct properties', () => {
      const board = createTestBoard('square8');

      expect(board.type).toBe('square8');
      expect(board.size).toBe(8);
      expect(board.stacks).toBeInstanceOf(Map);
      expect(board.markers).toBeInstanceOf(Map);
      expect(board.collapsedSpaces).toBeInstanceOf(Map);
      expect(board.stacks.size).toBe(0);
    });

    it('should create a square19 board with correct properties', () => {
      const board = createTestBoard('square19');

      expect(board.type).toBe('square19');
      expect(board.size).toBe(19);
    });

    it('should create a hexagonal board with correct properties', () => {
      const board = createTestBoard('hexagonal');

      expect(board.type).toBe('hexagonal');
      expect(board.size).toBe(13); // radius=12
    });
  });

  describe('createTestPlayer', () => {
    it('should create a player with default values', () => {
      const player = createTestPlayer(1);

      expect(player.playerNumber).toBe(1);
      expect(player.id).toBe('player-1');
      expect(player.username).toBe('TestPlayer1');
      expect(player.type).toBe('human');
      expect(player.ringsInHand).toBe(18);
      expect(player.eliminatedRings).toBe(0);
      expect(player.territorySpaces).toBe(0);
    });

    it('should allow overriding player properties', () => {
      const player = createTestPlayer(2, {
        ringsInHand: 10,
        eliminatedRings: 5,
      });

      expect(player.playerNumber).toBe(2);
      expect(player.ringsInHand).toBe(10);
      expect(player.eliminatedRings).toBe(5);
    });
  });

  describe('createTestGameState', () => {
    it('should create a game state with default values', () => {
      const gameState = createTestGameState();

      expect(gameState.id).toBe('test-game-123');
      expect(gameState.boardType).toBe('square8');
      expect(gameState.board).toBeDefined();
      expect(gameState.players).toHaveLength(2);
      expect(gameState.currentPhase).toBe('ring_placement');
      expect(gameState.gameStatus).toBe('active');
    });

    it('should allow custom board types', () => {
      const gameState = createTestGameState({ boardType: 'hexagonal' });

      expect(gameState.boardType).toBe('hexagonal');
      expect(gameState.board.type).toBe('hexagonal');
      expect(gameState.board.size).toBe(13); // radius=12
    });
  });
});

describe('Board Manipulation Helpers', () => {
  let board: ReturnType<typeof createTestBoard>;

  beforeEach(() => {
    board = createTestBoard('square8');
  });

  describe('addStack', () => {
    it('should add a stack to the board', () => {
      const position = pos(3, 3);
      addStack(board, position, 1, 2);

      const stack = board.stacks.get('3,3');
      expect(stack).toBeDefined();
      expect(stack?.controllingPlayer).toBe(1);
      expect(stack?.stackHeight).toBe(2);
      expect(stack?.capHeight).toBe(2);
      expect(stack?.rings).toEqual([1, 1]);
    });

    it('should handle hexagonal positions', () => {
      const hexBoard = createTestBoard('hexagonal');
      const position = pos(0, 0, 0);
      addStack(hexBoard, position, 2);

      const stack = hexBoard.stacks.get('0,0,0');
      expect(stack).toBeDefined();
      expect(stack?.controllingPlayer).toBe(2);
    });
  });

  describe('addMarker', () => {
    it('should add a marker to the board', () => {
      const position = pos(2, 2);
      addMarker(board, position, 1);

      const marker = board.markers.get('2,2');
      expect(marker).toBeDefined();
      expect(marker?.player).toBe(1);
      expect(marker?.type).toBe('regular');
    });

    it('should add a collapsed marker', () => {
      const position = pos(4, 4);
      addMarker(board, position, 2, 'collapsed');

      const marker = board.markers.get('4,4');
      expect(marker?.type).toBe('collapsed');
    });
  });

  describe('addCollapsedSpace', () => {
    it('should add a collapsed space to the board', () => {
      const position = pos(5, 5);
      addCollapsedSpace(board, position, 1);

      const player = board.collapsedSpaces.get('5,5');
      expect(player).toBe(1);
    });
  });
});

describe('Board Configurations', () => {
  it('should provide correct configuration for square8', () => {
    const config = BOARD_CONFIGS.square8;

    expect(config.type).toBe('square8');
    expect(config.size).toBe(8);
    expect(config.ringsPerPlayer).toBe(18);
    expect(config.minLineLength).toBe(4);
    expect(config.adjacencyType).toBe('moore');
  });

  it('should provide correct configuration for square19', () => {
    const config = BOARD_CONFIGS.square19;

    expect(config.type).toBe('square19');
    expect(config.size).toBe(19);
    expect(config.ringsPerPlayer).toBe(60);
    expect(config.minLineLength).toBe(4);
    expect(config.adjacencyType).toBe('moore');
  });

  it('should provide correct configuration for hexagonal', () => {
    const config = BOARD_CONFIGS.hexagonal;

    expect(config.type).toBe('hexagonal');
    expect(config.size).toBe(13); // radius=12
    expect(config.ringsPerPlayer).toBe(72);
    expect(config.minLineLength).toBe(4);
    expect(config.adjacencyType).toBe('hexagonal');
  });
});
