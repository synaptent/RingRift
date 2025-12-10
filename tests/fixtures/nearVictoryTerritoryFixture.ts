/**
 * Near-Victory Territory Fixture
 *
 * Provides programmatic access to game states where Player 1 is one
 * territory region resolution away from winning by territory control.
 *
 * Territory victory threshold: > 50% of board spaces (33 for square8).
 * These fixtures set up Player 1 with 32 territory spaces and a single
 * pending region that, when processed, pushes them to 33+ spaces.
 */

import type {
  GameState,
  BoardState,
  Player,
  Position,
  Territory,
  Move,
  RingStack,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

/**
 * Result of creating a near-victory territory fixture.
 */
export interface NearVictoryTerritoryFixture {
  /** Complete game state ready for engine use */
  gameState: GameState;
  /** Move that triggers territory victory when applied */
  winningMove: Move;
  /** Expected winner (should be 1) */
  expectedWinner: number;
  /** Victory type (always 'territory') */
  victoryType: 'territory';
  /** Number of territory spaces Player 1 has before the winning move */
  initialTerritorySpaces: number;
  /** Territory victory threshold (33 for square8) */
  territoryVictoryThreshold: number;
}

/**
 * Configuration options for creating near-victory territory fixtures.
 */
export interface NearVictoryTerritoryOptions {
  /** Board type (default: 'square8') */
  boardType?: 'square8' | 'square19' | 'hexagonal';
  /** Custom game ID (default: auto-generated) */
  gameId?: string;
  /** How many spaces below threshold to start (default: 1) */
  spacesbelowThreshold?: number;
  /** Number of spaces in the pending region (default: 1) */
  pendingRegionSize?: number;
}

/**
 * Creates a near-victory territory fixture for square8 board.
 *
 * Sets up:
 * - Player 1 with 32 collapsed territory spaces (rows 0-3 fully collapsed)
 * - A pending territory region at (4,4) with 1 space
 * - Player 1 has a stack at (7,7) to maintain game validity
 * - Player 2 has a stack at (6,7) to have an active opponent
 * - Game in 'territory_processing' phase with pending decision
 *
 * When the territory region is processed, Player 1 reaches 33+ spaces
 * and triggers territory_control victory.
 */
export function createNearVictoryTerritoryFixture(
  options: NearVictoryTerritoryOptions = {}
): NearVictoryTerritoryFixture {
  const boardType = options.boardType ?? 'square8';
  const gameId = options.gameId ?? `near-victory-territory-${Date.now()}`;
  const spacesbelowThreshold = options.spacesbelowThreshold ?? 1;
  const pendingRegionSize = options.pendingRegionSize ?? 1;

  // Board dimensions
  const boardSize = boardType === 'square8' ? 8 : boardType === 'square19' ? 19 : 13;
  const totalSpaces = boardType === 'hexagonal' ? 469 : boardSize * boardSize;

  // Territory victory threshold is > 50% of board spaces
  const territoryVictoryThreshold = Math.floor(totalSpaces / 2) + 1;

  // Initial territory spaces (just below threshold)
  const initialTerritorySpaces = territoryVictoryThreshold - spacesbelowThreshold;

  // Create collapsed spaces (Player 1 territory)
  const collapsedSpaces = new Map<string, number>();
  let placedSpaces = 0;
  const skipPositions = new Set<string>();

  // Reserve the pending region position
  const pendingRegionSpaces: Position[] = [];
  const centerX = Math.floor(boardSize / 2);
  const centerY = Math.floor(boardSize / 2);

  for (let i = 0; i < pendingRegionSize; i++) {
    const pos: Position = { x: centerX, y: centerY + i };
    pendingRegionSpaces.push(pos);
    skipPositions.add(positionToString(pos));
  }

  // Also reserve stack positions
  const p1StackPos: Position = { x: boardSize - 1, y: boardSize - 1 };
  const p2StackPos: Position = { x: boardSize - 2, y: boardSize - 1 };
  skipPositions.add(positionToString(p1StackPos));
  skipPositions.add(positionToString(p2StackPos));

  // Fill collapsed spaces up to initialTerritorySpaces
  for (let y = 0; y < boardSize && placedSpaces < initialTerritorySpaces; y++) {
    for (let x = 0; x < boardSize && placedSpaces < initialTerritorySpaces; x++) {
      const key = positionToString({ x, y });
      if (!skipPositions.has(key)) {
        collapsedSpaces.set(key, 1);
        placedSpaces++;
      }
    }
  }

  // Create stacks map
  const stacks = new Map<string, RingStack>();

  // Player 1 stack at corner
  stacks.set(positionToString(p1StackPos), {
    position: p1StackPos,
    rings: [1, 1],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: 1,
  });

  // Player 2 stack adjacent
  stacks.set(positionToString(p2StackPos), {
    position: p2StackPos,
    rings: [2],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 2,
  });

  // Create pending territory
  const territoryId = 'pending_victory_region';
  const territories = new Map<string, Territory>();
  territories.set(territoryId, {
    spaces: pendingRegionSpaces,
    controllingPlayer: 1,
    isDisconnected: false,
  });

  // Create board state
  const board: BoardState = {
    type: boardType,
    size: boardSize,
    stacks,
    markers: new Map(),
    collapsedSpaces,
    territories,
    formedLines: [],
    eliminatedRings: { 1: 0, 2: 0 },
  };

  // Create players
  const players: Player[] = [
    {
      id: 'player-1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: initialTerritorySpaces,
    },
    {
      id: 'player-2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  // Create game state
  const gameState: GameState = {
    id: gameId,
    boardType,
    board,
    players,
    currentPhase: 'territory_processing',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: {
      initialTime: 600,
      increment: 0,
      type: 'blitz',
    },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 3,
    totalRingsEliminated: 0,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold,
  };

  // Add pending territory decision to state
  (gameState as unknown as Record<string, unknown>).pendingTerritoryDecision = {
    territories: [territoryId],
    currentIndex: 0,
  };

  // Create the winning move
  const winningMove: Move = {
    id: `process-victory-region-${Date.now()}`,
    type: 'process_territory_region',
    player: 1,
    to: pendingRegionSpaces[0],
    disconnectedRegions: [
      {
        spaces: pendingRegionSpaces,
        controllingPlayer: 1,
        isDisconnected: false,
      },
    ],
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    winningMove,
    expectedWinner: 1,
    victoryType: 'territory',
    initialTerritorySpaces,
    territoryVictoryThreshold,
  };
}

/**
 * Creates a variant fixture where the pending region has multiple cells,
 * ensuring the final territory count exceeds the threshold by more than 1.
 */
export function createNearVictoryTerritoryFixtureMultiRegion(): NearVictoryTerritoryFixture {
  return createNearVictoryTerritoryFixture({
    spacesbelowThreshold: 2,
    pendingRegionSize: 3, // 3 cells ensures we pass the threshold
    gameId: `near-victory-territory-multi-${Date.now()}`,
  });
}

/**
 * Serializes the fixture's game state to a JSON-compatible format
 * suitable for contract vectors or test snapshots.
 */
export function serializeNearVictoryTerritoryFixture(fixture: NearVictoryTerritoryFixture): {
  gameState: Record<string, unknown>;
  winningMove: Record<string, unknown>;
} {
  const state = fixture.gameState;

  // Convert Maps to plain objects for JSON serialization
  const stacksObj: Record<string, unknown> = {};
  for (const [key, stack] of state.board.stacks) {
    stacksObj[key] = {
      position: stack.position,
      rings: stack.rings,
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  }

  const markersObj: Record<string, unknown> = {};
  for (const [key, marker] of state.board.markers) {
    markersObj[key] = {
      player: marker.player,
      position: marker.position,
      type: marker.type,
    };
  }

  const collapsedObj: Record<string, number> = {};
  for (const [key, owner] of state.board.collapsedSpaces) {
    collapsedObj[key] = owner;
  }

  const territoriesObj: Record<string, unknown> = {};
  for (const [key, territory] of state.board.territories) {
    territoriesObj[key] = {
      spaces: territory.spaces,
      controllingPlayer: territory.controllingPlayer,
      isDisconnected: territory.isDisconnected,
    };
  }

  return {
    gameState: {
      gameId: state.id,
      boardType: state.boardType,
      board: {
        type: state.board.type,
        size: state.board.size,
        stacks: stacksObj,
        markers: markersObj,
        collapsedSpaces: collapsedObj,
        territories: territoriesObj,
        eliminatedRings: state.board.eliminatedRings,
        formedLines: state.board.formedLines,
      },
      players: state.players.map((p) => ({
        playerNumber: p.playerNumber,
        ringsInHand: p.ringsInHand,
        eliminatedRings: p.eliminatedRings,
        territorySpaces: p.territorySpaces,
        isActive: true,
      })),
      currentPlayer: state.currentPlayer,
      currentPhase: state.currentPhase,
      turnNumber: 20,
      moveHistory: [],
      gameStatus: state.gameStatus,
      victoryThreshold: state.victoryThreshold,
      territoryVictoryThreshold: state.territoryVictoryThreshold,
      totalRingsEliminated: state.totalRingsEliminated,
    },
    winningMove: {
      id: fixture.winningMove.id,
      type: fixture.winningMove.type,
      player: fixture.winningMove.player,
      to: fixture.winningMove.to,
      disconnectedRegions: fixture.winningMove.disconnectedRegions,
      timestamp: fixture.winningMove.timestamp.toISOString(),
      thinkTime: fixture.winningMove.thinkTime,
      moveNumber: fixture.winningMove.moveNumber,
    },
  };
}
