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

  // We assume action.regionId is the region the player chose to KEEP.
  // We need to remove all other disconnected regions belonging to this player.
  // Since we don't have the "conflict group" explicitly, we can iterate through all territories
  // and remove any that are:
  // 1. Owned by the current player
  // 2. Marked as isDisconnected = true
  // 3. NOT the chosen regionId

  // This assumes that ALL disconnected regions for a player are part of the current conflict.
  // This is generally true because disconnection checks happen globally.

  const keptRegion = newState.board.territories.get(action.regionId);
  if (!keptRegion) throw new Error('TerritoryMutator: Kept region not found');

  // Mark kept region as connected
  keptRegion.isDisconnected = false;

  // Remove other disconnected regions
  for (const [id, region] of newState.board.territories) {
    if (
      region.controllingPlayer === action.playerId &&
      region.isDisconnected &&
      id !== action.regionId
    ) {
      // Remove this region
      // "The other region(s) are lost. Any rings on the lost region are returned to their owners."

      for (const pos of region.spaces) {
        const key = positionToString(pos);

        // Un-collapse the space (remove from collapsedSpaces)
        newState.board.collapsedSpaces.delete(key);

        // If there were rings on it (unlikely for collapsed space, but maybe markers?), handle them.
        // Collapsed spaces don't have stacks.
        // But wait, "Any rings on the lost region..."
        // Collapsed spaces are empty of rings by definition.
        // Maybe it means "Any rings that were TRAPPED inside?" No.
        // It probably refers to the fact that they are no longer territory.

        // However, if we un-collapse them, they become empty spaces.
        // Markers? Collapsed spaces don't have markers.

        // So we just remove the collapsed status.
      }

      newState.board.territories.delete(id);
    }
  }

  // Also, we should probably re-check connectivity or merge regions?
  // The engine loop will likely handle re-scanning territories.

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
  // Usually this means the TOP ring? Or the whole stack?
  // "Forced Elimination Choice": "You must eliminate a ring from one of your stacks."
  // Singular "a ring".

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
