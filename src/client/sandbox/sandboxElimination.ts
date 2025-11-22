import { BoardState, Player, Position, RingStack, positionToString } from '../../shared/types/game';
import { calculateCapHeight } from '../../shared/engine/core';

const TERRITORY_TRACE_DEBUG =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

export interface ForcedEliminationResult {
  board: BoardState;
  players: Player[];
  totalRingsEliminatedDelta: number;
}

function assertForcedEliminationConsistency(
  context: string,
  before: { board: BoardState; players: Player[] },
  after: { board: BoardState; players: Player[]; delta: number },
  playerNumber: number
): void {
  const isTestEnv =
    typeof process !== 'undefined' &&
    !!(process as any).env &&
    (process as any).env.NODE_ENV === 'test';

  const sumEliminated = (players: Player[]): number =>
    players.reduce((acc, p) => acc + p.eliminatedRings, 0);

  const sumBoardEliminated = (board: BoardState): number =>
    Object.values(board.eliminatedRings ?? {}).reduce((acc, v) => acc + v, 0);

  const beforePlayerTotal = sumEliminated(before.players);
  const beforeBoardTotal = sumBoardEliminated(before.board);
  const afterPlayerTotal = sumEliminated(after.players);
  const afterBoardTotal = sumBoardEliminated(after.board);

  const deltaPlayers = afterPlayerTotal - beforePlayerTotal;
  const deltaBoard = afterBoardTotal - beforeBoardTotal;

  const errors: string[] = [];

  if (deltaPlayers !== after.delta) {
    errors.push(
      `forced elimination (${context}) player delta mismatch: expected ${after.delta}, actual ${deltaPlayers}`
    );
  }

  if (deltaBoard !== after.delta) {
    errors.push(
      `forced elimination (${context}) board delta mismatch: expected ${after.delta}, actual ${deltaBoard}`
    );
  }

  if (after.delta < 0) {
    errors.push(
      `forced elimination (${context}) produced negative delta=${after.delta} for player ${playerNumber}`
    );
  }

  if (errors.length === 0) {
    return;
  }

  const message = `sandboxElimination invariant violation (${context}):` + '\n' + errors.join('\n');

  // eslint-disable-next-line no-console
  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
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
  const player = players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  if (stacks.length === 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  const stack = stacks.find((s) => s.capHeight > 0) ?? stacks[0];
  const capHeight = calculateCapHeight(stack.rings);
  if (capHeight <= 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  if (TERRITORY_TRACE_DEBUG) {
    // eslint-disable-next-line no-console
    console.log('[sandboxElimination.forceEliminateCapOnBoard]', {
      playerNumber,
      stackPosition: stack.position,
      capHeight,
      stackHeight: stack.stackHeight,
    });
  }

  const remainingRings = stack.rings.slice(capHeight);

  const updatedEliminatedRings = { ...board.eliminatedRings };
  updatedEliminatedRings[playerNumber] = (updatedEliminatedRings[playerNumber] || 0) + capHeight;

  const updatedPlayers = players.map((p) =>
    p.playerNumber === playerNumber ? { ...p, eliminatedRings: p.eliminatedRings + capHeight } : p
  );

  const nextBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: updatedEliminatedRings,
  };

  if (remainingRings.length > 0) {
    const newStack: RingStack = {
      ...stack,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0],
    };
    const key = positionToString(stack.position);
    nextBoard.stacks.set(key, newStack);
  } else {
    const key = positionToString(stack.position);
    nextBoard.stacks.delete(key);
  }

  const result: ForcedEliminationResult = {
    board: nextBoard,
    players: updatedPlayers,
    totalRingsEliminatedDelta: capHeight,
  };

  assertForcedEliminationConsistency(
    'forceEliminateCapOnBoard',
    { board, players },
    { board: result.board, players: result.players, delta: result.totalRingsEliminatedDelta },
    playerNumber
  );

  return result;
}
