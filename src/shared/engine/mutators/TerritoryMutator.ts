import { GameState, ProcessTerritoryAction, EliminateStackAction } from '../types';
import { positionToString } from '../../types/game';
import {
  eliminateFromStack,
  getRingsToEliminate,
  type EliminationContext,
} from '../aggregates/EliminationAggregate';

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
  const key = positionToString(action.stackPosition);
  const stack = state.board.stacks.get(key);

  if (!stack) {
    throw new Error('TerritoryMutator: Stack to eliminate not found');
  }

  const topRingOwner = stack.rings[0];

  // Determine elimination context - defaults to 'forced' for backwards compatibility
  // The mutator is typically used for internal territory eliminations where any stack
  // inside the region is eliminated unconditionally (no eligibility check needed)
  const eliminationContext: EliminationContext =
    (action.eliminationContext as EliminationContext) ?? 'forced';

  // DELEGATE TO EliminationAggregate for canonical elimination semantics
  // Per RR-CANON-R022, R122, R145, R100:
  // - 'line': Eliminate exactly ONE ring from the top (any controlled stack is eligible)
  // - 'territory': Eliminate entire cap (only eligible stacks: multicolor or height > 1)
  // - 'forced': Eliminate entire cap (any controlled stack is eligible)
  const eliminationResult = eliminateFromStack({
    context: eliminationContext,
    player: topRingOwner,
    stackPosition: action.stackPosition,
    board: state.board,
  });

  if (!eliminationResult.success) {
    // If elimination failed, return unchanged state
    return state;
  }

  // Create new state with the updated board from EliminationAggregate
  const newState = {
    ...state,
    board: eliminationResult.updatedBoard,
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
  };

  // Update player's eliminatedRings count
  const player = newState.players.find((p) => p.playerNumber === topRingOwner);
  if (player) {
    player.eliminatedRings += eliminationResult.ringsEliminated;
  }

  // Update total rings eliminated
  newState.totalRingsEliminated =
    (state as GameState & { totalRingsEliminated: number }).totalRingsEliminated +
    eliminationResult.ringsEliminated;

  newState.lastMoveAt = new Date();
  return newState;
}
