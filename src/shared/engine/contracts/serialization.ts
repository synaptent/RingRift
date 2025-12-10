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
import { computeProgressSnapshot } from '../core';

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
  /** Chain capture position during chain_capture phase */
  chainCapturePosition?: Position;
  /**
   * When present, restricts movement/capture/chain_capture so that only actions
   * originating from the keyed stack are legal for the remainder of the current
   * turn. This is set after place_ring and cleared at turn end.
   */
  mustMoveFromStackKey?: string;
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
    // Handle different ring formats: array, object with numeric keys, or missing
    let rings: number[];
    if (Array.isArray(stack.rings) && stack.rings.length > 0) {
      rings = [...stack.rings];
    } else if (stack.rings && typeof stack.rings === 'object' && !Array.isArray(stack.rings)) {
      // Convert object with numeric keys to array (e.g., {0: 1, 1: 2} -> [1, 2])
      rings = Object.values(stack.rings) as number[];
    } else if (stack.stackHeight > 0 && stack.controllingPlayer) {
      // Generate synthetic rings array from stackHeight and controllingPlayer
      // This handles curated scenarios that omit the rings array
      rings = Array(stack.stackHeight).fill(stack.controllingPlayer);
    } else {
      rings = [];
    }

    // Handle position: use stack.position if present, otherwise reconstruct from key
    let position: Position;
    if (stack.position && typeof stack.position.x === 'number') {
      position = { ...stack.position };
    } else {
      // Reconstruct position from key (e.g., "2,4" -> {x: 2, y: 4})
      const parts = key.split(',');
      position = {
        x: Number(parts[0]),
        y: Number(parts[1]),
        ...(parts[2] !== undefined ? { z: Number(parts[2]) } : {}),
      };
    }

    stacks.set(key, {
      position,
      rings,
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    });
  }

  const markers = new Map<string, MarkerInfo>();
  for (const [key, marker] of Object.entries(data.markers)) {
    // Handle position: use marker.position if present, otherwise reconstruct from key
    let markerPosition: Position;
    if (marker.position && typeof marker.position.x === 'number') {
      markerPosition = { ...marker.position };
    } else {
      const parts = key.split(',');
      markerPosition = {
        x: Number(parts[0]),
        y: Number(parts[1]),
        ...(parts[2] !== undefined ? { z: Number(parts[2]) } : {}),
      };
    }

    markers.set(key, {
      position: markerPosition,
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

  const serialized: SerializedGameState = {
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

  // Include chainCapturePosition if present (during chain_capture phase)
  if (state.chainCapturePosition) {
    serialized.chainCapturePosition = { ...state.chainCapturePosition };
  }

  // Include mustMoveFromStackKey if present (constrains movement/capture to one stack)
  if (state.mustMoveFromStackKey) {
    serialized.mustMoveFromStackKey = state.mustMoveFromStackKey;
  }

  return serialized;
}

/**
 * Deserialize a plain object to a GameState.
 * Note: Creates a minimal GameState suitable for engine processing.
 *
 * Handles both SerializedGameState format (from curated scenarios) and
 * full GameState format (from self-play database) by checking for
 * alternative field names.
 */
export function deserializeGameState(data: SerializedGameState): GameState {
  // Handle both formats: SerializedGameState uses gameId, full GameState uses id
  const dataAny = data as unknown as Record<string, unknown>;
  const gameId = data.gameId || (dataAny.id as string) || '';

  // Handle both formats for board type
  const boardType = (data.board?.type ||
    (dataAny.boardType as string) ||
    'square8') as GameState['boardType'];

  // Handle players - self-play format may have full Player objects
  const players = data.players.map((p) => {
    const pAny = p as Record<string, unknown>;
    return {
      // Use existing id/username if present (self-play format), otherwise generate
      id: (pAny.id as string) || `player-${p.playerNumber}`,
      username: (pAny.username as string) || `Player ${p.playerNumber}`,
      type: (pAny.type as 'human' | 'ai') || (p.isActive ? ('human' as const) : ('ai' as const)),
      playerNumber: p.playerNumber,
      isReady: (pAny.isReady as boolean) ?? true,
      timeRemaining: (pAny.timeRemaining as number) ?? 600,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
    };
  });

  return {
    id: gameId,
    boardType,
    board: deserializeBoardState(data.board),
    players,
    currentPlayer: data.currentPlayer,
    currentPhase: data.currentPhase as GameState['currentPhase'],
    chainCapturePosition: data.chainCapturePosition,
    mustMoveFromStackKey: data.mustMoveFromStackKey,
    moveHistory: data.moveHistory?.map((m) => ({ ...m })) || [],
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
  const sInvariantBefore = computeProgressSnapshot(before).S;
  const sInvariantAfter = computeProgressSnapshot(after).S;
  diff.sInvariantDelta = sInvariantAfter - sInvariantBefore;

  return diff;
}
