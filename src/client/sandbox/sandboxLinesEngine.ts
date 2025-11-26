import type {
  BoardState,
  BoardType,
  GameState,
  LineInfo,
  Player,
  Position,
  RingStack,
  Move,
  LineDecisionApplicationOutcome,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  calculateCapHeight,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  findLinesForPlayer,
} from '../../shared/engine';
import { findAllLinesOnBoard } from './sandboxLines';
import { forceEliminateCapOnBoard } from './sandboxElimination';

function assertLineEngineMonotonicity(
  context: string,
  before: { collapsedSpaces: number; totalRingsEliminated: number },
  after: { collapsedSpaces: number; totalRingsEliminated: number }
): void {
  const isTestEnv =
    typeof process !== 'undefined' &&
    !!(process as any).env &&
    (process as any).env.NODE_ENV === 'test';

  const errors: string[] = [];

  if (after.collapsedSpaces < before.collapsedSpaces) {
    errors.push(
      `collapsedSpaces decreased in line engine (${context}): before=${before.collapsedSpaces}, after=${after.collapsedSpaces}`
    );
  }

  if (after.totalRingsEliminated < before.totalRingsEliminated) {
    errors.push(
      `totalRingsEliminated decreased in line engine (${context}): before=${before.totalRingsEliminated}, after=${after.totalRingsEliminated}`
    );
  }

  if (errors.length === 0) {
    return;
  }

  const message = `sandboxLinesEngine invariant violation (${context}):` + '\n' + errors.join('\n');

  // eslint-disable-next-line no-console
  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
}

/**
 * Pure line-processing helpers for the sandbox engine.
 *
 * These functions mirror the behaviour of ClientSandboxEngine
 * processLinesForCurrentPlayer + collapseLineMarkers + forceEliminateCap,
 * but operate directly on GameState and return an updated copy. This keeps
 * the line-processing logic modular and testable.
 */

function stringToPositionLocal(posStr: string): Position {
  const parts = posStr.split(',').map(Number);
  if (parts.length === 2) {
    const [x, y] = parts;
    return { x, y };
  }
  if (parts.length === 3) {
    const [x, y, z] = parts;
    return { x, y, z };
  }
  return { x: 0, y: 0 };
}

function isValidPosition(boardType: BoardType, board: BoardState, pos: Position): boolean {
  const config = BOARD_CONFIGS[boardType];
  if (boardType === 'hexagonal') {
    const radius = config.size - 1;
    const x = pos.x;
    const y = pos.y;
    const z = pos.z !== undefined ? pos.z : -x - y;
    const distance = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    return distance <= radius;
  }
  return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
}

function getPlayerStacks(board: BoardState, playerNumber: number): RingStack[] {
  const stacks: RingStack[] = [];
  for (const stack of board.stacks.values()) {
    if (stack.controllingPlayer === playerNumber) {
      stacks.push(stack);
    }
  }
  return stacks;
}

export function collapseLineMarkersOnBoard(
  board: BoardState,
  players: Player[],
  positions: Position[],
  playerNumber: number
): { board: BoardState; players: Player[] } {
  const nextBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };

  const collapsedKeys = new Set<string>();

  for (const pos of positions) {
    const key = positionToString(pos);
    collapsedKeys.add(key);
    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, playerNumber);
  }

  const territoryGain = collapsedKeys.size;
  const nextPlayers = players.map((p) =>
    p.playerNumber === playerNumber
      ? { ...p, territorySpaces: p.territorySpaces + territoryGain }
      : p
  );

  return {
    board: nextBoard,
    players: nextPlayers,
  };
}

/**
 * Enumerate canonical line-processing decision moves for the current
 * player. This mirrors GameEngine.getValidLineProcessingMoves.
 */
export function getValidLineProcessingMoves(gameState: GameState): Move[] {
  const currentPlayer = gameState.currentPlayer;

  // Base process_line decisions are enumerated via the shared helper so that
  // sandbox, backend GameEngine, and shared GameEngine all see the same line
  // geometry and Move payloads.
  const processMoves = enumerateProcessLineMoves(gameState, currentPlayer, {
    detectionMode: 'detect_now',
  });

  // Reward moves are driven per line index using the shared geometry helper.
  // This surfaces:
  // - For exact-length lines: a single collapse-all choose_line_reward (optional).
  // - For overlength lines: one collapse-all + all contiguous minimum-collapse
  //   segments of length L.
  const playerLines = findLinesForPlayer(gameState.board, currentPlayer);
  const rewardMoves: Move[] = [];

  playerLines.forEach((_line, index) => {
    rewardMoves.push(...enumerateChooseLineRewardMoves(gameState, currentPlayer, index));
  });

  return [...processMoves, ...rewardMoves];
}

/**
 * Apply a single line decision move to the game state using the shared
 * lineDecisionHelpers. This is the sandbox counterpart to the backend
 * GameEngine.applyDecisionMove line branch and returns both the next
 * GameState and a flag indicating whether a mandatory self-elimination
 * reward is now owed by the acting player.
 *
 * Elimination itself is orchestrated at the ClientSandboxEngine level so
 * that automatic sandbox flows can continue to apply elimination
 * immediately, while move-driven decision phases can surface explicit
 * eliminate_rings_from_stack Moves for parity with the backend.
 */
export function applyLineDecisionMove(
  gameState: GameState,
  move: Move
): LineDecisionApplicationOutcome {
  if (move.type !== 'process_line' && move.type !== 'choose_line_reward') {
    return {
      nextState: gameState,
      pendingLineRewardElimination: false,
    };
  }

  return move.type === 'process_line'
    ? applyProcessLineDecision(gameState, move)
    : applyChooseLineRewardDecision(gameState, move);
}

/**
 * Process all lines for the current player in the given GameState,
 * mirroring the default sandbox behaviour (no line_order or
 * line_reward_option choices):
 *
 * - Only lines for gameState.currentPlayer are considered.
 * - Exact-length lines: collapse all markers and eliminate a cap.
 * - Longer lines: collapse only the minimum required markers; no elimination.
 */
export function processLinesForCurrentPlayer(gameState: GameState): GameState {
  const boardType = gameState.boardType;
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;

  let board = gameState.board;
  let players = gameState.players;
  let totalRingsEliminated = gameState.totalRingsEliminated;
  const currentPlayer = gameState.currentPlayer;

  const beforeSnapshot = {
    collapsedSpaces: board.collapsedSpaces.size,
    totalRingsEliminated,
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const allLines: LineInfo[] = findAllLinesOnBoard(
      boardType,
      board,
      (pos: Position) => isValidPosition(boardType, board, pos),
      (posStr: string) => stringToPositionLocal(posStr)
    );

    const playerLines = allLines.filter((line) => line.player === currentPlayer);
    if (playerLines.length === 0) {
      break;
    }

    const line = playerLines[0];

    if (line.length === requiredLength) {
      // Exact length: collapse all markers and eliminate a cap.
      const collapsed = collapseLineMarkersOnBoard(board, players, line.positions, currentPlayer);
      board = collapsed.board;
      players = collapsed.players;

      // Eliminate one cap from the current player's stacks.
      const stacks = getPlayerStacks(board, currentPlayer);
      const elimResult = forceEliminateCapOnBoard(board, players, currentPlayer, stacks);
      board = elimResult.board;
      players = elimResult.players;
      totalRingsEliminated += elimResult.totalRingsEliminatedDelta;
    } else if (line.length > requiredLength) {
      // Longer: collapse only requiredLength markers, no elimination.
      const markersToCollapse = line.positions.slice(0, requiredLength);
      const collapsed = collapseLineMarkersOnBoard(
        board,
        players,
        markersToCollapse,
        currentPlayer
      );
      board = collapsed.board;
      players = collapsed.players;
    } else {
      // Defensive: should not happen if line detection respects minimum length.
      break;
    }
  }

  const afterSnapshot = {
    collapsedSpaces: board.collapsedSpaces.size,
    totalRingsEliminated,
  };

  assertLineEngineMonotonicity('processLinesForCurrentPlayer', beforeSnapshot, afterSnapshot);

  return {
    ...gameState,
    board,
    players,
    totalRingsEliminated,
  };
}
