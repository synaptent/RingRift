import { GameState, ProcessTerritoryAction, EliminateStackAction } from '../types';
import { positionToString } from '../../types/game';
import { calculateCapHeight } from '../core';

export function mutateProcessTerritory(
  state: GameState,
  action: ProcessTerritoryAction
): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      territories: new Map(state.board.territories),
      collapsedSpaces: new Map(state.board.collapsedSpaces),
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  // Processing a territory region means collapsing all spaces in that region
  // into territory (collapsed spaces) controlled by the acting player.
  // Any stacks inside the region are eliminated and credited to the processing player.

  const region = newState.board.territories.get(action.regionId);
  if (!region) throw new Error('TerritoryMutator: Region not found');

  const player = action.playerId;
  let internalEliminations = 0;
  let territoryGain = 0;

  // 1. Process each space in the region
  for (const pos of region.spaces) {
    const key = positionToString(pos);

    // Eliminate any stack inside the region
    const stack = newState.board.stacks.get(key);
    if (stack) {
      internalEliminations += stack.stackHeight;
      newState.board.stacks.delete(key);
    }

    // Remove any marker at this position
    newState.board.markers.delete(key);

    // Collapse the space to the processing player's territory
    newState.board.collapsedSpaces.set(key, player);
    territoryGain++;
  }

  // 2. Remove the territory entry since it's now collapsed
  newState.board.territories.delete(action.regionId);

  // 3. Update player's territorySpaces
  if (territoryGain > 0) {
    const playerObj = newState.players.find((p) => p.playerNumber === player);
    if (playerObj) {
      playerObj.territorySpaces += territoryGain;
    }
  }

  // 4. Credit internal eliminations to the processing player
  if (internalEliminations > 0) {
    newState.board.eliminatedRings[player] =
      (newState.board.eliminatedRings[player] || 0) + internalEliminations;
    newState.totalRingsEliminated = (newState.totalRingsEliminated || 0) + internalEliminations;

    const playerObj = newState.players.find((p) => p.playerNumber === player);
    if (playerObj) {
      playerObj.eliminatedRings += internalEliminations;
    }
  }

  newState.lastMoveAt = new Date();
  return newState;
}

export function mutateEliminateStack(state: GameState, action: EliminateStackAction): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
  };

  const key = positionToString(action.stackPosition);
  const stack = newState.board.stacks.get(key);

  if (!stack) {
    throw new Error('TerritoryMutator: Stack to eliminate not found');
  }

  // "Eliminate rings from stack"
  // Per RR-CANON: For line/territory processing, we must eliminate the entire CAP
  // (all consecutive top rings of the controlling color). For mixed-colour stacks,
  // this exposes buried rings of other colours; for single-colour stacks with
  // height > 1, this eliminates all rings.

  // Rings are [top, ..., bottom].
  // We must eliminate the entire CAP (all consecutive top rings of controlling color).

  const capHeight = calculateCapHeight(stack.rings);
  const topRingOwner = stack.rings[0];

  // Remove cap (slice from start)
  const remainingRings = stack.rings.slice(capHeight);

  if (remainingRings.length === 0) {
    newState.board.stacks.delete(key);
  } else {
    stack.rings = remainingRings;
    stack.stackHeight = remainingRings.length;
    stack.capHeight = calculateCapHeight(remainingRings);
    stack.controllingPlayer = remainingRings[0]; // New top ring
  }

  // Update elimination counts
  if (capHeight > 0) {
    newState.totalRingsEliminated += capHeight;
    newState.board.eliminatedRings[topRingOwner] =
      (newState.board.eliminatedRings[topRingOwner] || 0) + capHeight;

    const player = newState.players.find((p) => p.playerNumber === topRingOwner);
    if (player) {
      player.eliminatedRings += capHeight;
    }
  }

  newState.lastMoveAt = new Date();
  return newState;
}
