/**
 * Golden Replay Test Helpers
 *
 * Provides invariant assertions and replay utilities for golden game tests.
 * Golden games are curated test fixtures that exercise all major rules axes
 * and serve as high-confidence parity tests between TS and Python engines.
 */

import type {
  GameState,
  Position,
  BoardType,
  GamePhase,
  GameStatus,
} from '../../src/shared/types/game';
import type { GameRecord } from '../../src/shared/types/gameRecord';
// Use legacy replay helper for golden games recorded before canonical spec was finalized.
// These games may have phase transitions that differ from the current TS engine.
// TODO: Regenerate golden games with canonical Python engine, then switch back to
// reconstructStateAtMove from '../../src/shared/engine'.
import { reconstructStateAtMoveLegacy as reconstructStateAtMove } from '../../src/shared/engine/legacy/legacyReplayHelpers';
import { isValidPosition } from '../../src/shared/engine/validators/utils';
import { BOARD_CONFIGS as BoardConfigs } from '../../src/shared/types/game';

// =============================================================================
// STRUCTURAL INVARIANTS
// =============================================================================

export interface InvariantViolation {
  invariant: string;
  moveIndex: number;
  details: string;
  state?: Partial<GameState>;
}

export interface InvariantCheckResult {
  passed: boolean;
  violations: InvariantViolation[];
}

/**
 * Get board size from board type
 */
function getBoardSize(boardType: BoardType): number {
  return BoardConfigs[boardType].size;
}

/**
 * Get rings per player from board type
 */
function getRingsPerPlayer(boardType: BoardType): number {
  return BoardConfigs[boardType].ringsPerPlayer;
}

/**
 * Count rings on board for a specific player
 */
function countRingsOnBoardForPlayer(state: GameState, playerNumber: number): number {
  let count = 0;
  for (const [, stack] of state.board.stacks.entries()) {
    for (const ringOwner of stack.rings) {
      if (ringOwner === playerNumber) {
        count++;
      }
    }
  }
  return count;
}

/**
 * INV-BOARD-CONSISTENCY: All positions in board state are valid for the board type
 */
export function checkBoardConsistency(state: GameState, moveIndex: number): InvariantViolation[] {
  const violations: InvariantViolation[] = [];
  const { board, boardType } = state;
  const boardSize = getBoardSize(boardType);

  // Check all stacks have valid positions
  for (const [posKey, stack] of board.stacks.entries()) {
    const parts = posKey.split(',').map(Number);
    const position: Position =
      parts.length === 3 ? { x: parts[0], y: parts[1], z: parts[2] } : { x: parts[0], y: parts[1] };

    if (!isValidPosition(position, boardType, boardSize)) {
      violations.push({
        invariant: 'INV-BOARD-CONSISTENCY',
        moveIndex,
        details: `Stack at invalid position: ${posKey}`,
        state: { boardType },
      });
    }

    // Check stack has valid height
    if (stack.stackHeight < 1) {
      violations.push({
        invariant: 'INV-BOARD-CONSISTENCY',
        moveIndex,
        details: `Stack at ${posKey} has invalid height: ${stack.stackHeight}`,
      });
    }
  }

  // Check all markers have valid positions
  for (const [posKey] of board.markers.entries()) {
    const parts = posKey.split(',').map(Number);
    const position: Position =
      parts.length === 3 ? { x: parts[0], y: parts[1], z: parts[2] } : { x: parts[0], y: parts[1] };

    if (!isValidPosition(position, boardType, boardSize)) {
      violations.push({
        invariant: 'INV-BOARD-CONSISTENCY',
        moveIndex,
        details: `Marker at invalid position: ${posKey}`,
        state: { boardType },
      });
    }
  }

  return violations;
}

/**
 * INV-TURN-SEQUENCE: Move history length increases with each step
 */
export function checkTurnSequence(
  prevState: GameState | null,
  currentState: GameState,
  moveIndex: number
): InvariantViolation[] {
  const violations: InvariantViolation[] = [];

  if (prevState) {
    // Move history should not decrease
    const prevMoveCount = prevState.moveHistory.length;
    const currentMoveCount = currentState.moveHistory.length;

    if (currentMoveCount < prevMoveCount) {
      violations.push({
        invariant: 'INV-TURN-SEQUENCE',
        moveIndex,
        details: `Move history decreased from ${prevMoveCount} to ${currentMoveCount}`,
      });
    }
  }

  return violations;
}

/**
 * INV-PLAYER-RINGS: Ring counts are non-negative and consistent
 */
export function checkPlayerRings(state: GameState, moveIndex: number): InvariantViolation[] {
  const violations: InvariantViolation[] = [];
  const ringsPerPlayer = getRingsPerPlayer(state.boardType);

  for (const player of state.players) {
    // Check eliminated rings is non-negative
    if (player.eliminatedRings < 0) {
      violations.push({
        invariant: 'INV-PLAYER-RINGS',
        moveIndex,
        details: `Player ${player.playerNumber} has negative eliminatedRings: ${player.eliminatedRings}`,
      });
    }

    // Check rings in hand is non-negative
    if (player.ringsInHand < 0) {
      violations.push({
        invariant: 'INV-PLAYER-RINGS',
        moveIndex,
        details: `Player ${player.playerNumber} has negative ringsInHand: ${player.ringsInHand}`,
      });
    }

    // Check rings on board count
    const ringsOnBoard = countRingsOnBoardForPlayer(state, player.playerNumber);
    if (ringsOnBoard < 0) {
      violations.push({
        invariant: 'INV-PLAYER-RINGS',
        moveIndex,
        details: `Player ${player.playerNumber} has negative rings on board: ${ringsOnBoard}`,
      });
    }

    // Total rings should not exceed starting rings (for standard games)
    const totalRings = ringsOnBoard + player.eliminatedRings + player.ringsInHand;
    if (totalRings > ringsPerPlayer) {
      violations.push({
        invariant: 'INV-PLAYER-RINGS',
        moveIndex,
        details: `Player ${player.playerNumber} has more total rings (${totalRings}) than starting (${ringsPerPlayer})`,
      });
    }
  }

  return violations;
}

/**
 * INV-PHASE-VALID: Game phase is a valid phase value
 */
export function checkPhaseValid(state: GameState, moveIndex: number): InvariantViolation[] {
  const violations: InvariantViolation[] = [];

  const validPhases: GamePhase[] = [
    'ring_placement',
    'movement',
    'capture',
    'chain_capture',
    'line_processing',
    'territory_processing',
    'forced_elimination', // 7th phase per RR-CANON-R070
    // Terminal phase is used only after the final recorded move to mark completion
    'game_over',
  ];

  if (!validPhases.includes(state.currentPhase)) {
    violations.push({
      invariant: 'INV-PHASE-VALID',
      moveIndex,
      details: `Invalid phase: ${state.currentPhase}`,
    });
  }

  return violations;
}

/**
 * INV-ACTIVE-PLAYER: Active player index is valid
 */
export function checkActivePlayer(state: GameState, moveIndex: number): InvariantViolation[] {
  const violations: InvariantViolation[] = [];

  if (state.gameStatus === 'active') {
    if (state.currentPlayer < 0 || state.currentPlayer >= state.players.length) {
      violations.push({
        invariant: 'INV-ACTIVE-PLAYER',
        moveIndex,
        details: `Invalid currentPlayer: ${state.currentPlayer} (players: ${state.players.length})`,
      });
    }
  }

  return violations;
}

/**
 * INV-GAME-STATUS: Game status is consistent with winner field
 */
export function checkGameStatus(state: GameState, moveIndex: number): InvariantViolation[] {
  const violations: InvariantViolation[] = [];

  // If there's a winner, game should be finished/completed
  if (state.winner !== undefined) {
    const terminalStatuses: GameStatus[] = ['finished', 'completed'];
    if (!terminalStatuses.includes(state.gameStatus)) {
      violations.push({
        invariant: 'INV-GAME-STATUS',
        moveIndex,
        details: `Winner is set (${state.winner}) but status is ${state.gameStatus}`,
      });
    }
  }

  // If game is finished, there should usually be a winner (unless draw/abandonment)
  const finishedStatuses: GameStatus[] = ['finished', 'completed'];
  if (finishedStatuses.includes(state.gameStatus) && state.winner === undefined) {
    // This isn't always a violation (could be draw), so we'll just note it
    // Not adding a violation for this case
  }

  return violations;
}

/**
 * Run all structural invariants on a game state
 */
export function checkAllInvariants(
  state: GameState,
  moveIndex: number,
  prevState: GameState | null = null
): InvariantViolation[] {
  return [
    ...checkBoardConsistency(state, moveIndex),
    ...checkTurnSequence(prevState, state, moveIndex),
    ...checkPlayerRings(state, moveIndex),
    ...checkPhaseValid(state, moveIndex),
    ...checkActivePlayer(state, moveIndex),
    ...checkGameStatus(state, moveIndex),
  ];
}

// =============================================================================
// REPLAY UTILITIES
// =============================================================================

export interface ReplayResult {
  success: boolean;
  finalState: GameState | null;
  stateAtMove: Map<number, GameState>;
  invariantViolations: InvariantViolation[];
  error?: string;
}

/**
 * Replay a game record and check invariants at each move
 */
export function replayAndAssertInvariants(gameRecord: GameRecord): ReplayResult {
  const stateAtMove = new Map<number, GameState>();
  const invariantViolations: InvariantViolation[] = [];
  let finalState: GameState | null = null;

  try {
    // Reconstruct initial state (move 0)
    let prevState: GameState | null = null;
    const totalMoves = gameRecord.moves.length;

    for (let moveIndex = 0; moveIndex <= totalMoves; moveIndex++) {
      const state = reconstructStateAtMove(gameRecord, moveIndex);
      stateAtMove.set(moveIndex, state);

      // Check invariants
      const violations = checkAllInvariants(state, moveIndex, prevState);
      invariantViolations.push(...violations);

      prevState = state;

      if (moveIndex === totalMoves) {
        finalState = state;
      }
    }

    return {
      success: invariantViolations.length === 0,
      finalState,
      stateAtMove,
      invariantViolations,
    };
  } catch (error) {
    return {
      success: false,
      finalState: null,
      stateAtMove,
      invariantViolations,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Assert that final state matches recorded outcome
 */
export function assertFinalStateMatchesOutcome(
  finalState: GameState,
  gameRecord: GameRecord
): InvariantViolation[] {
  const violations: InvariantViolation[] = [];
  const moveIndex = gameRecord.moves.length;

  // Check game is over
  const terminalStatuses: GameStatus[] = ['finished', 'completed'];
  if (!terminalStatuses.includes(finalState.gameStatus)) {
    violations.push({
      invariant: 'INV-FINAL-STATE',
      moveIndex,
      details: `Game status should be finished/completed but is '${finalState.gameStatus}'`,
    });
  }

  // Check winner matches (if recorded)
  if (gameRecord.winner !== undefined && finalState.winner !== undefined) {
    // Winner in gameRecord is player index (0-based)
    if (gameRecord.winner !== finalState.winner) {
      violations.push({
        invariant: 'INV-FINAL-STATE',
        moveIndex,
        details: `Winner mismatch: recorded player ${gameRecord.winner}, actual player ${finalState.winner}`,
      });
    }
  }

  return violations;
}

// =============================================================================
// FIXTURE LOADING
// =============================================================================

import * as fs from 'fs';
import * as path from 'path';

export interface GoldenGameInfo {
  filename: string;
  category: string;
  description: string;
  boardType: BoardType;
  numPlayers: number;
  expectedOutcome?: string;
}

/**
 * Load all golden game fixtures from a directory
 */
export function loadGoldenGames(
  fixturesDir: string
): Array<{ info: GoldenGameInfo; record: GameRecord }> {
  const games: Array<{ info: GoldenGameInfo; record: GameRecord }> = [];

  if (!fs.existsSync(fixturesDir)) {
    return games;
  }

  const files = fs
    .readdirSync(fixturesDir)
    .filter((f) => f.endsWith('.jsonl') || f.endsWith('.json'));

  for (const file of files) {
    const filePath = path.join(fixturesDir, file);
    const content = fs.readFileSync(filePath, 'utf-8');

    try {
      // Try parsing as single JSON first
      const record = JSON.parse(content) as GameRecord;

      // Extract metadata from filename: category_description_boardtype_numplayers.json
      const parts = file.replace(/\.(jsonl?|json)$/, '').split('_');
      const info: GoldenGameInfo = {
        filename: file,
        category: parts[0] || 'uncategorized',
        description: parts.slice(1, -2).join('_') || file,
        boardType: (record.boardType as BoardType) || 'square8',
        numPlayers: record.players?.length || 2,
        expectedOutcome: record.outcome,
      };

      games.push({ info, record });
    } catch {
      // Try parsing as JSONL (multiple lines)
      const lines = content.trim().split('\n');
      for (let i = 0; i < lines.length; i++) {
        try {
          const record = JSON.parse(lines[i]) as GameRecord;
          const info: GoldenGameInfo = {
            filename: `${file}:${i}`,
            category: 'jsonl',
            description: `Line ${i} of ${file}`,
            boardType: (record.boardType as BoardType) || 'square8',
            numPlayers: record.players?.length || 2,
            expectedOutcome: record.outcome,
          };
          games.push({ info, record });
        } catch {
          // Skip invalid lines
        }
      }
    }
  }

  return games;
}

/**
 * Get the path to the golden games fixtures directory
 */
export function getGoldenGamesDir(): string {
  return path.join(__dirname, '..', 'fixtures', 'golden-games');
}
