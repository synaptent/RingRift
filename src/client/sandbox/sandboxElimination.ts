import {
  BoardState,
  Player,
  Position,
  RingStack,
  positionToString
} from '../../shared/types/game';
import { calculateCapHeight } from '../../shared/engine/core';

export interface ForcedEliminationResult {
  board: BoardState;
  players: Player[];
  totalRingsEliminatedDelta: number;
}

/**
 * Core cap-elimination helper operating directly on the board and players.
 * This mirrors the logic in ClientSandboxEngine.forceEliminateCap but is
 * pure with respect to GameState, returning updated structures and the
 * number of rings eliminated.
 */
export function forceEliminateCapOnBoard(
  board: BoardState,
  players: Player[],
  playerNumber: number,
  stacks: RingStack[]
): ForcedEliminationResult {
  const player = players.find(p => p.playerNumber === playerNumber);
  if (!player) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  if (stacks.length === 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  const stack = stacks.find(s => s.capHeight > 0) ?? stacks[0];
  const capHeight = calculateCapHeight(stack.rings);
  if (capHeight <= 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  const remainingRings = stack.rings.slice(capHeight);

  const updatedEliminatedRings = { ...board.eliminatedRings };
  updatedEliminatedRings[playerNumber] =
    (updatedEliminatedRings[playerNumber] || 0) + capHeight;

  const updatedPlayers = players.map(p =>
    p.playerNumber === playerNumber
      ? { ...p, eliminatedRings: p.eliminatedRings + capHeight }
      : p
  );

  const nextBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: updatedEliminatedRings
  };

  if (remainingRings.length > 0) {
    const newStack: RingStack = {
      ...stack,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0]
    };
    const key = positionToString(stack.position);
    nextBoard.stacks.set(key, newStack);
  } else {
    const key = positionToString(stack.position);
    nextBoard.stacks.delete(key);
  }

  return {
    board: nextBoard,
    players: updatedPlayers,
    totalRingsEliminatedDelta: capHeight
  };
}
