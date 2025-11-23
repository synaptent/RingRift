import { GameState, PlaceRingAction } from '../types';
import { BoardState, Position, positionToString } from '../../types/game';
import { calculateCapHeight } from '../core';

/**
 * Canonical board-level placement mutator used by both the shared GameEngine
 * and client/server hosts. This applies a placement for `playerId` at
 * `position` and returns a new BoardState with:
 *
 * - stack/marker exclusivity enforced (any marker at the destination is
 *   cleared before writing the stack),
 * - new rings added on top of the stack (front of the `rings` array),
 * - capHeight / stackHeight / controllingPlayer recomputed from the
 *   resulting ring sequence.
 */
export function applyPlacementOnBoard(
  board: BoardState,
  position: Position,
  playerId: number,
  count: number
): BoardState {
  const effectiveCount = Math.max(1, count);

  const newBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
  };

  const posKey = positionToString(position);

  // Maintain stack/marker exclusivity: any marker at this position is
  // removed before writing the stack, mirroring BoardManager.setStack
  // semantics on the backend and sandbox marker behaviour.
  newBoard.markers.delete(posKey);

  const existingStack = newBoard.stacks.get(posKey);
  const placementRings = new Array(effectiveCount).fill(playerId);

  if (existingStack && existingStack.rings.length > 0) {
    const rings = [...placementRings, ...existingStack.rings];
    newBoard.stacks.set(posKey, {
      ...existingStack,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: rings[0],
    });
  } else {
    const rings = placementRings;
    newBoard.stacks.set(posKey, {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: rings[0],
    });
  }

  return newBoard;
}

/**
 * GameEngine-level placement mutator. This is a thin wrapper around
 * {@link applyPlacementOnBoard} that is responsible for updating player
 * ringsInHand and bookkeeping fields on GameState, while leaving the
 * totalRingsInPlay counter unchanged to mirror the legacy backend
 * semantics (the total pool is initialised from BOARD_CONFIGS).
 */
export function mutatePlacement(state: GameState, action: PlaceRingAction): GameState {
  const players = state.players.map((p) => ({ ...p }));
  const player = players.find((p) => p.playerNumber === action.playerId);

  if (!player) {
    throw new Error('PlacementMutator: Player not found');
  }

  // Decrement rings in hand (clamped) and apply the same number of rings
  // to the board. validatePlacement ensures action.count never exceeds
  // the player's supply, so this is primarily a defensive guard.
  const toSpend = Math.min(action.count, player.ringsInHand);
  if (toSpend <= 0) {
    // No rings to place – return state unchanged.
    return state;
  }
  player.ringsInHand -= toSpend;

  const updatedBoard = applyPlacementOnBoard(
    state.board as BoardState,
    action.position,
    action.playerId,
    toSpend
  );

  const newState: GameState & { totalRingsInPlay: number; lastMoveAt: Date } = {
    ...(state as GameState & { totalRingsInPlay: number; lastMoveAt: Date }),
    board: updatedBoard,
    players,
    moveHistory: [...state.moveHistory],
    lastMoveAt: new Date(),
  };

  return newState;
}
