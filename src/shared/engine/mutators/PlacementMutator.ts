import { GameState, PlaceRingAction } from '../types';
import { positionToString } from '../../types/game';
import { calculateCapHeight } from '../core';

export function mutatePlacement(state: GameState, action: PlaceRingAction): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsInPlay: number;
    lastMoveAt: Date;
  };

  const posKey = positionToString(action.position);
  const existingStack = newState.board.stacks.get(posKey);
  const player = newState.players.find((p) => p.playerNumber === action.playerId);

  if (!player) {
    throw new Error('PlacementMutator: Player not found');
  }

  // 1. Decrement rings in hand (clamped). The legacy backend treats
  // totalRingsInPlay as a fixed pool initialised from BOARD_CONFIGS and
  // does not modify it on placement, so we mirror that behaviour here.
  const toSpend = Math.min(action.count, player.ringsInHand);
  player.ringsInHand -= toSpend;

  // 2. Update board stack
  if (existingStack) {
    // Add to existing stack
    const newRings = [...existingStack.rings];
    for (let i = 0; i < action.count; i++) {
      newRings.push(action.playerId);
    }

    newState.board.stacks.set(posKey, {
      ...existingStack,
      rings: newRings,
      stackHeight: newRings.length,
      capHeight: calculateCapHeight(newRings),
      controllingPlayer: action.playerId, // Top ring is now placed by current player
    });
  } else {
    // Create new stack
    const newRings = Array(action.count).fill(action.playerId);

    newState.board.stacks.set(posKey, {
      position: action.position,
      rings: newRings,
      stackHeight: action.count,
      capHeight: action.count,
      controllingPlayer: action.playerId,
    });
  }

  // 3. Update timestamps
  newState.lastMoveAt = new Date();

  return newState;
}
