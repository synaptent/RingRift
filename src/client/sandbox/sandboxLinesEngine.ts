import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  GameState,
  LineInfo,
  Player,
  Position,
  RingStack,
  Move,
  positionToString,
} from '../../shared/types/game';
import { calculateCapHeight } from '../../shared/engine/core';
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
  const moves: Move[] = [];
  const boardType = gameState.boardType;
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;
  const currentPlayer = gameState.currentPlayer;

  const allLines: LineInfo[] = findAllLinesOnBoard(
    boardType,
    gameState.board,
    (pos: Position) => isValidPosition(boardType, gameState.board, pos),
    (posStr: string) => stringToPositionLocal(posStr)
  );

  const playerLines = allLines.filter((line) => line.player === currentPlayer);

  if (playerLines.length === 0) {
    return moves;
  }

  // One process_line move per player-owned line
  playerLines.forEach((line, index) => {
    const lineKey = line.positions.map((p) => positionToString(p)).join('|');
    moves.push({
      id: `process-line-${index}-${lineKey}`,
      type: 'process_line',
      player: currentPlayer,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.moveHistory.length + 1,
    } as Move);
  });

  // For overlength lines, also surface choose_line_reward decisions
  const overlengthLines = playerLines.filter((line) => line.positions.length > requiredLength);

  overlengthLines.forEach((line, index) => {
    const lineKey = line.positions.map((p) => positionToString(p)).join('|');

    // Option 1: Collapse All (default/implicit)
    moves.push({
      id: `choose-line-reward-${index}-${lineKey}-all`,
      type: 'choose_line_reward',
      player: currentPlayer,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.moveHistory.length + 1,
    } as Move);

    // Option 2: Minimum Collapse
    const minMarkers = line.positions.slice(0, requiredLength);
    moves.push({
      id: `choose-line-reward-${index}-${lineKey}-min`,
      type: 'choose_line_reward',
      player: currentPlayer,
      formedLines: [line],
      collapsedMarkers: minMarkers,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: gameState.moveHistory.length + 1,
    } as Move);
  });

  return moves;
}

/**
 * Apply a single line decision move to the game state.
 */
export function applyLineDecisionMove(gameState: GameState, move: Move): GameState {
  if (move.type !== 'process_line' && move.type !== 'choose_line_reward') {
    return gameState;
  }

  const boardType = gameState.boardType;
  const requiredLength = BOARD_CONFIGS[boardType].lineLength;

  let board = gameState.board;
  let players = gameState.players;
  let totalRingsEliminated = gameState.totalRingsEliminated;

  let targetLine: LineInfo | undefined;

  // Prefer the canonical line payload carried on the Move itself so that
  // callers (including tests that stub findAllLinesOnBoard) can control
  // exactly which line is being processed.
  if (move.formedLines && move.formedLines.length > 0) {
    targetLine = move.formedLines[0];
  } else {
    // Fallback: recompute lines when move.formedLines is absent. This keeps
    // the helper usable with legacy callers that don't populate formedLines.
    const allLines: LineInfo[] = findAllLinesOnBoard(
      boardType,
      board,
      (pos: Position) => isValidPosition(boardType, board, pos),
      (posStr: string) => stringToPositionLocal(posStr)
    );

    const playerLines = allLines.filter((line) => line.player === move.player);
    if (playerLines.length === 0) {
      return gameState;
    }

    targetLine = playerLines[0];
  }

  if (!targetLine) {
    return gameState;
  }

  const lineLength = targetLine.positions.length;

  if (lineLength === requiredLength) {
    // Exact length: collapse all and eliminate
    const collapsed = collapseLineMarkersOnBoard(board, players, targetLine.positions, move.player);
    board = collapsed.board;
    players = collapsed.players;

    const stacks = getPlayerStacks(board, move.player);
    const elimResult = forceEliminateCapOnBoard(board, players, move.player, stacks);
    board = elimResult.board;
    players = elimResult.players;
    totalRingsEliminated += elimResult.totalRingsEliminatedDelta;
  } else if (lineLength > requiredLength) {
    // Overlength: check move for choice
    if (
      move.type === 'choose_line_reward' &&
      move.collapsedMarkers &&
      move.collapsedMarkers.length === requiredLength
    ) {
      // Option 2: Minimum collapse, no elimination
      const collapsed = collapseLineMarkersOnBoard(
        board,
        players,
        move.collapsedMarkers,
        move.player
      );
      board = collapsed.board;
      players = collapsed.players;
    } else if (move.type === 'choose_line_reward') {
      // Option 1 (explicit): Collapse all and eliminate
      const collapsed = collapseLineMarkersOnBoard(
        board,
        players,
        targetLine.positions,
        move.player
      );
      board = collapsed.board;
      players = collapsed.players;

      const stacks = getPlayerStacks(board, move.player);
      const elimResult = forceEliminateCapOnBoard(board, players, move.player, stacks);
      board = elimResult.board;
      players = elimResult.players;
      totalRingsEliminated += elimResult.totalRingsEliminatedDelta;
    } else {
      // process_line default for overlength: Option 2 (Minimum collapse, no elimination)
      // This preserves legacy sandbox behavior where overlength lines don't trigger elimination.
      const markersToCollapse = targetLine.positions.slice(0, requiredLength);
      const collapsed = collapseLineMarkersOnBoard(board, players, markersToCollapse, move.player);
      board = collapsed.board;
      players = collapsed.players;
    }
  }

  return {
    ...gameState,
    board,
    players,
    totalRingsEliminated,
  };
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
