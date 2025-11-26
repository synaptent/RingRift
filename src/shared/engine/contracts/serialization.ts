/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Serialization Utilities for Engine Contracts
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This file provides utilities for serializing GameState and related objects
 * to/from JSON format for cross-engine communication.
 */

import type {
  GameState,
  BoardState,
  Position,
  Move,
  RingStack,
  MarkerInfo,
} from '../../types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Serialized board state with Maps converted to objects.
 */
export interface SerializedBoardState {
  type: string;
  size: number;
  stacks: { [key: string]: SerializedStack };
  markers: { [key: string]: SerializedMarker };
  collapsedSpaces: { [key: string]: number };
  eliminatedRings: { [player: number]: number };
  formedLines?: unknown[];
}

/**
 * Serialized stack representation.
 */
export interface SerializedStack {
  position: Position;
  rings: number[];
  stackHeight: number;
  capHeight: number;
  controllingPlayer: number;
}

/**
 * Serialized marker representation.
 */
export interface SerializedMarker {
  position: Position;
  player: number;
  type: string;
}

/**
 * Serialized game state.
 */
export interface SerializedGameState {
  gameId?: string;
  board: SerializedBoardState;
  players: Array<{
    playerNumber: number;
    ringsInHand: number;
    eliminatedRings: number;
    territorySpaces: number;
    isActive?: boolean;
  }>;
  currentPlayer: number;
  currentPhase: string;
  turnNumber: number;
  moveHistory: Move[];
  gameStatus: string;
  victoryThreshold: number;
  territoryVictoryThreshold: number;
  totalRingsEliminated?: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// Serialization Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Serialize a BoardState to a plain object.
 */
export function serializeBoardState(board: BoardState): SerializedBoardState {
  const stacks: { [key: string]: SerializedStack } = {};
  for (const [key, stack] of board.stacks) {
    stacks[key] = {
      position: stack.position,
      rings: [...stack.rings],
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  }

  const markers: { [key: string]: SerializedMarker } = {};
  for (const [key, marker] of board.markers) {
    markers[key] = {
      position: marker.position,
      player: marker.player,
      type: marker.type,
    };
  }

  const collapsedSpaces: { [key: string]: number } = {};
  for (const [key, player] of board.collapsedSpaces) {
    collapsedSpaces[key] = player;
  }

  return {
    type: board.type,
    size: board.size,
    stacks,
    markers,
    collapsedSpaces,
    eliminatedRings: { ...board.eliminatedRings },
    formedLines: board.formedLines ? [...board.formedLines] : [],
  };
}

/**
 * Deserialize a plain object to a BoardState.
 */
export function deserializeBoardState(data: SerializedBoardState): BoardState {
  const stacks = new Map<string, RingStack>();
  for (const [key, stack] of Object.entries(data.stacks)) {
    stacks.set(key, {
      position: stack.position,
      rings: [...stack.rings],
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    });
  }

  const markers = new Map<string, MarkerInfo>();
  for (const [key, marker] of Object.entries(data.markers)) {
    markers.set(key, {
      position: marker.position,
      player: marker.player,
      type: marker.type as 'regular' | 'collapsed',
    });
  }

  const collapsedSpaces = new Map<string, number>();
  for (const [key, player] of Object.entries(data.collapsedSpaces)) {
    collapsedSpaces.set(key, player);
  }

  return {
    type: data.type as BoardState['type'],
    size: data.size,
    stacks,
    markers,
    collapsedSpaces,
    eliminatedRings: { ...data.eliminatedRings },
    territories: new Map(),
    formedLines: (data.formedLines as BoardState['formedLines']) || [],
  };
}

/**
 * Serialize a GameState to a plain object.
 */
export function serializeGameState(state: GameState): SerializedGameState {
  // Compute turnNumber from moveHistory length (1-based)
  const turnNumber = (state.moveHistory?.length ?? 0) + 1;

  return {
    gameId: state.id,
    board: serializeBoardState(state.board),
    players: state.players.map((p) => ({
      playerNumber: p.playerNumber,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
      isActive: true,
    })),
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    turnNumber: turnNumber,
    moveHistory: state.moveHistory.map((m) => ({ ...m })),
    gameStatus: state.gameStatus,
    victoryThreshold: state.victoryThreshold,
    territoryVictoryThreshold: state.territoryVictoryThreshold,
    totalRingsEliminated: state.totalRingsEliminated,
  };
}

/**
 * Deserialize a plain object to a GameState.
 * Note: Creates a minimal GameState suitable for engine processing.
 */
export function deserializeGameState(data: SerializedGameState): GameState {
  return {
    id: data.gameId || '',
    boardType: data.board.type as GameState['boardType'],
    board: deserializeBoardState(data.board),
    players: data.players.map((p) => ({
      // Create minimal Player with required fields
      id: `player-${p.playerNumber}`,
      username: `Player ${p.playerNumber}`,
      type: p.isActive ? ('human' as const) : ('ai' as const),
      playerNumber: p.playerNumber,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
    })),
    currentPlayer: data.currentPlayer,
    currentPhase: data.currentPhase as GameState['currentPhase'],
    moveHistory: data.moveHistory.map((m) => ({ ...m })),
    history: [],
    gameStatus: data.gameStatus as GameState['gameStatus'],
    winner: undefined,
    timeControl: { initialTime: 600, increment: 5, type: 'rapid' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: data.players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: data.totalRingsEliminated || 0,
    victoryThreshold: data.victoryThreshold,
    territoryVictoryThreshold: data.territoryVictoryThreshold,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON Import/Export
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Export a GameState to a JSON string.
 */
export function gameStateToJson(state: GameState): string {
  return JSON.stringify(serializeGameState(state), null, 2);
}

/**
 * Import a GameState from a JSON string.
 */
export function jsonToGameState(json: string): GameState {
  const data = JSON.parse(json) as SerializedGameState;
  return deserializeGameState(data);
}

// ═══════════════════════════════════════════════════════════════════════════
// Test Vector Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a test vector from a state and move.
 */
export function createTestVector(
  id: string,
  category: string,
  description: string,
  inputState: GameState,
  move: Move,
  expectedAssertions: Record<string, unknown>
): unknown {
  return {
    id,
    category,
    description,
    input: {
      state: serializeGameState(inputState),
      move,
    },
    expectedOutput: {
      status: 'complete',
      assertions: expectedAssertions,
    },
    tags: [],
    source: 'generated',
    createdAt: new Date().toISOString(),
  };
}

/**
 * Compute state diff for test vector assertions.
 */
export function computeStateDiff(before: GameState, after: GameState): Record<string, unknown> {
  const diff: Record<string, unknown> = {};

  if (before.currentPlayer !== after.currentPlayer) {
    diff.currentPlayerChanged = true;
    diff.newCurrentPlayer = after.currentPlayer;
  }

  if (before.currentPhase !== after.currentPhase) {
    diff.currentPhaseChanged = true;
    diff.newCurrentPhase = after.currentPhase;
  }

  if (before.gameStatus !== after.gameStatus) {
    diff.gameStatusChanged = true;
    diff.newGameStatus = after.gameStatus;
  }

  diff.stackCountDelta = after.board.stacks.size - before.board.stacks.size;
  diff.markerCountDelta = after.board.markers.size - before.board.markers.size;
  diff.collapsedCountDelta = after.board.collapsedSpaces.size - before.board.collapsedSpaces.size;

  // S-invariant
  const sInvariantBefore =
    before.board.markers.size +
    before.board.collapsedSpaces.size +
    before.players.reduce((sum, p) => sum + p.eliminatedRings, 0);
  const sInvariantAfter =
    after.board.markers.size +
    after.board.collapsedSpaces.size +
    after.players.reduce((sum, p) => sum + p.eliminatedRings, 0);
  diff.sInvariantDelta = sInvariantAfter - sInvariantBefore;

  return diff;
}
