/**
 * Test Fixtures and Utilities
 * Common test data and helper functions for RingRift tests
 */

import {
  BoardType,
  BoardState,
  GameState,
  Position,
  Player,
  GamePhase,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Position helper - creates a position object
 */
export function pos(x: number, y: number, z?: number): Position {
  return z !== undefined ? { x, y, z } : { x, y };
}

/**
 * Position string converter
 */
export function posStr(x: number, y: number, z?: number): string {
  return z !== undefined ? `${x},${y},${z}` : `${x},${y}`;
}

/**
 * Creates a minimal BoardState for testing
 */
export function createTestBoard(boardType: BoardType = 'square8'): BoardState {
  return {
    type: boardType,
    size: boardType === 'square8' ? 8 : boardType === 'square19' ? 19 : 13, // hex: size=13, radius=12
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
  };
}

/**
 * Creates a minimal Player for testing
 */
export function createTestPlayer(playerNumber: number, overrides: Partial<Player> = {}): Player {
  return {
    id: `player-${playerNumber}`,
    username: `TestPlayer${playerNumber}`,
    type: 'human',
    playerNumber,
    isReady: true,
    timeRemaining: 600,
    ringsInHand: 18, // Default for square8
    eliminatedRings: 0,
    territorySpaces: 0,
    ...overrides,
  };
}

/**
 * Creates a minimal GameState for testing
 */
export function createTestGameState(overrides: Partial<GameState> = {}): GameState {
  const boardType = (overrides.boardType || 'square8') as BoardType;
  const board = overrides.board || createTestBoard(boardType);

  return {
    id: 'test-game-123',
    boardType,
    board,
    players: overrides.players || [createTestPlayer(1), createTestPlayer(2)],
    currentPlayer: 0,
    currentPhase: 'ring_placement',
    moveHistory: [],
    // Initialise structured history so helpers that rely on it (e.g. for
    // moveNumber computation) behave like real engine states.
    history: overrides.history || [],
    timeControl: { initialTime: 600, increment: 0, type: 'blitz' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 32,
    ...overrides,
  };
}

/**
 * Board type configurations
 */
export const BOARD_CONFIGS = {
  square8: {
    type: 'square8' as BoardType,
    size: 8,
    ringsPerPlayer: 18,
    minLineLength: 4,
    adjacencyType: 'moore' as const,
  },
  square19: {
    type: 'square19' as BoardType,
    size: 19,
    ringsPerPlayer: 36,
    minLineLength: 5,
    adjacencyType: 'von_neumann' as const,
  },
  hexagonal: {
    type: 'hexagonal' as BoardType,
    size: 13, // radius=12
    ringsPerPlayer: 48,
    minLineLength: 4, // Same as other boards
    adjacencyType: 'hexagonal' as const,
  },
} as const;

/**
 * Common test positions for square boards
 */
export const SQUARE_POSITIONS = {
  center8: pos(3, 3),
  corner8: pos(0, 0),
  edge8: pos(0, 3),
  center19: pos(9, 9),
  corner19: pos(0, 0),
  edge19: pos(0, 9),
};

/**
 * Common test positions for hexagonal board
 */
export const HEX_POSITIONS = {
  center: pos(0, 0, 0),
  edge: pos(10, 0, -10),
  corner: pos(10, -10, 0),
};

/**
 * Common game phases for testing
 */
export const GAME_PHASES: readonly GamePhase[] = [
  'ring_placement',
  'movement',
  'capture',
  'line_processing',
  'territory_processing',
] as const;

/**
 * Helper to create a stack at a position
 */
export function addStack(
  board: BoardState,
  position: Position,
  player: number,
  height: number = 1
): void {
  const key = positionToString(position);

  board.stacks.set(key, {
    position,
    rings: Array(height).fill(player),
    stackHeight: height,
    capHeight: height,
    controllingPlayer: player,
  });
}

/**
 * Helper to add a marker at a position
 */
export function addMarker(
  board: BoardState,
  position: Position,
  player: number,
  type: 'regular' | 'collapsed' = 'regular'
): void {
  const key = positionToString(position);

  board.markers.set(key, {
    position,
    player,
    type,
  });
}

/**
 * Helper to add a collapsed space
 */
export function addCollapsedSpace(board: BoardState, position: Position, player: number): void {
  const key = positionToString(position);

  board.collapsedSpaces.set(key, player);
}

/**
 * Helper to create a line of markers
 */
export function createMarkerLine(
  board: BoardState,
  start: Position,
  direction: { dx: number; dy: number; dz?: number },
  length: number,
  player: number
): void {
  for (let i = 0; i < length; i++) {
    const position: Position =
      direction.dz !== undefined
        ? {
            x: start.x + direction.dx * i,
            y: start.y + direction.dy * i,
            z: (start.z || 0) + direction.dz * i,
          }
        : {
            x: start.x + direction.dx * i,
            y: start.y + direction.dy * i,
          };

    addMarker(board, position, player);
  }
}

/**
 * Assertion helper - checks if position exists in board
 */
export function assertPositionHasStack(
  board: BoardState,
  position: Position,
  expectedPlayer?: number
): void {
  const key =
    position.z !== undefined
      ? `${position.x},${position.y},${position.z}`
      : `${position.x},${position.y}`;

  const stack = board.stacks.get(key);
  if (!stack) {
    throw new Error(`Expected stack at position ${key}, but found none`);
  }

  if (expectedPlayer !== undefined && stack.controllingPlayer !== expectedPlayer) {
    throw new Error(
      `Expected stack at ${key} to be owned by player ${expectedPlayer}, but found player ${stack.controllingPlayer}`
    );
  }
}

/**
 * Assertion helper - checks if position has marker
 */
export function assertPositionHasMarker(
  board: BoardState,
  position: Position,
  expectedPlayer?: number
): void {
  const key =
    position.z !== undefined
      ? `${position.x},${position.y},${position.z}`
      : `${position.x},${position.y}`;

  const marker = board.markers.get(key);
  if (!marker) {
    throw new Error(`Expected marker at position ${key}, but found none`);
  }

  if (expectedPlayer !== undefined && marker.player !== expectedPlayer) {
    throw new Error(
      `Expected marker at ${key} to be owned by player ${expectedPlayer}, but found player ${marker.player}`
    );
  }
}

/**
 * Assertion helper - checks if position is collapsed
 */
export function assertPositionCollapsed(
  board: BoardState,
  position: Position,
  expectedPlayer?: number
): void {
  const key =
    position.z !== undefined
      ? `${position.x},${position.y},${position.z}`
      : `${position.x},${position.y}`;

  const player = board.collapsedSpaces.get(key);
  if (player === undefined) {
    throw new Error(`Expected collapsed space at position ${key}, but found none`);
  }

  if (expectedPlayer !== undefined && player !== expectedPlayer) {
    throw new Error(
      `Expected collapsed space at ${key} to be owned by player ${expectedPlayer}, but found player ${player}`
    );
  }
}

/**
 * Helper to get all positions within a board
 */
export function getAllBoardPositions(boardType: BoardType, size: number): Position[] {
  const positions: Position[] = [];

  if (boardType === 'hexagonal') {
    const radius = size - 1;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        positions.push({ x: q, y: r, z: s });
      }
    }
  } else {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        positions.push({ x, y });
      }
    }
  }

  return positions;
}

/**
 * Deep clone a game state for testing
 */
export function cloneGameState(gameState: GameState): GameState {
  return JSON.parse(
    JSON.stringify({
      ...gameState,
      board: {
        ...gameState.board,
        stacks: Array.from(gameState.board.stacks.entries()),
        markers: Array.from(gameState.board.markers.entries()),
        collapsedSpaces: Array.from(gameState.board.collapsedSpaces.entries()),
      },
    })
  );
}
