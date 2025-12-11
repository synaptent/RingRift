/**
 * ═══════════════════════════════════════════════════════════════════════════
 * LineAggregate - Consolidated Marker Line Domain
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This aggregate consolidates all marker line detection, validation, and
 * collapse logic from:
 *
 * - lineDetection.ts → line finding algorithms
 * - lineDecisionHelpers.ts → line decision enumeration
 * - validators/LineValidator.ts → validation
 * - mutators/LineMutator.ts → mutation
 *
 * Rule Reference: Section 11 - Line Formation & Collapse
 *
 * Key Rules:
 * - RR-CANON-R100: Line definition (3+ consecutive markers in straight line)
 * - RR-CANON-R101: Line length thresholds (square8: 3, square19/hex: 4)
 * - RR-CANON-R102: Line collapse process
 * - RR-CANON-R103: Player chooses which markers to remove
 * - RR-CANON-R104: Removed markers return to supply
 * - Process all lines for current player before advancing turn
 *
 * Design principles:
 * - Pure functions: No side effects, return new state
 * - Type safety: Full TypeScript typing
 * - Backward compatibility: Source files continue to export their functions
 */

import type {
  GameState,
  BoardState,
  Position,
  Move,
  BoardType,
  LineInfo,
  RingStack,
} from '../../types/game';
import { BOARD_CONFIGS, positionToString, stringToPosition } from '../../types/game';

import type { ProcessLineAction, ChooseLineRewardAction } from '../types';
import { getEffectiveLineLengthThreshold } from '../rulesConfig';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Detected line information - alias for LineInfo to maintain backward
 * compatibility while providing a more descriptive name in the aggregate.
 */
export type DetectedLine = LineInfo;

/**
 * Decision made about how to collapse a line.
 *
 * Players have two options for lines longer than the minimum:
 * - COLLAPSE_ALL: Collapse entire line, gain territory + potential elimination reward
 * - MINIMUM_COLLAPSE: Collapse only minimum required markers, no reward
 */
export interface LineCollapseDecision {
  /** Index of the line being processed (into board.formedLines). */
  lineIndex: number;
  /** Which option the player chose. */
  selection: 'COLLAPSE_ALL' | 'MINIMUM_COLLAPSE';
  /**
   * For MINIMUM_COLLAPSE, the specific consecutive positions to collapse.
   * Must be exactly `lineLength` positions that form a contiguous subset.
   */
  collapsedPositions?: Position[];
  /** The line being processed. */
  line: DetectedLine;
  /** Player making the decision. */
  player: number;
}

/**
 * Result of validating a line-related action.
 */
export type LineValidationResult = {
  valid: boolean;
  reason?: string;
  code?: string;
};

/**
 * Result of applying a line mutation.
 */
export type LineMutationResult =
  | { success: true; newState: GameState }
  | { success: false; reason: string };

/**
 * Options that control how line-processing moves are enumerated.
 */
export interface LineEnumerationOptions {
  /**
   * Whether to re-run line detection from scratch over the current board
   * state (`detect_now`) or to trust `state.board.formedLines`
   * (`use_board_cache`).
   *
   * - In the backend GameEngine, lines are typically detected immediately
   *   after a movement/capture step and cached on the board.
   * - In some sandbox flows, line detection may be re-run on demand.
   *
   * Default: 'use_board_cache'.
   */
  detectionMode?: 'use_board_cache' | 'detect_now';

  /**
   * When `detectionMode === 'detect_now'`, controls which board type /
   * configuration should be used for line-adjacency rules. In almost all
   * cases this is simply `state.board.type`.
   */
  boardTypeOverride?: BoardType;
}

/**
 * Result of applying a line-processing decision.
 */
export interface LineDecisionApplicationOutcome {
  /**
   * Next GameState after applying the chosen decision, including:
   *
   * - collapse of markers to territory where appropriate;
   * - updates to `board.collapsedSpaces`, `board.markers`, and
   *   `players[n].territorySpaces`;
   * - any rings returned to hand when stacks are removed from collapsed
   *   spaces; and
   * - updates to `board.formedLines` when processed or broken lines are
   *   removed.
   */
  nextState: GameState;

  /**
   * When true, this decision granted the acting player a mandatory
   * self-elimination reward that must be paid via a follow-up
   * `eliminate_rings_from_stack` move. Under current backend/sandbox
   * semantics this is the case for:
   *
   * - exact-length lines processed via `process_line`, and
   * - overlength lines when using the collapse-all reward option.
   *
   * LINE ELIMINATION COST (RR-CANON-R022): The player must eliminate ONE ring
   * from the top of any controlled stack. Any controlled stack is an eligible
   * target for line processing, including height-1 standalone rings.
   *
   * The choice of *where* to eliminate from and how that elimination is
   * surfaced (explicit move vs automatic from hand) remains a host concern;
   * this flag exists purely to keep bookkeeping consistent across engines.
   */
  pendingLineRewardElimination: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal Helpers - Line Geometry
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get the canonical line directions for a board type.
 * For square boards: 4 directions (E, SE, S, NE) to cover all lines.
 * For hex boards: 3 directions (E, NE, NW) in cube coordinates.
 */
function getLineDirections(boardType: string): Position[] {
  if (boardType === 'hexagonal') {
    // 3 directions for hexagonal (in cube coordinates)
    return [
      { x: 1, y: 0, z: -1 }, // East
      { x: 1, y: -1, z: 0 }, // Northeast
      { x: 0, y: -1, z: 1 }, // Northwest
    ];
  } else {
    // 4 directions for square (Moore adjacency for lines)
    return [
      { x: 1, y: 0 }, // East
      { x: 1, y: 1 }, // Southeast
      { x: 0, y: 1 }, // South
      { x: 1, y: -1 }, // Northeast
    ];
  }
}

/**
 * Check if a position is valid on the board.
 */
function isValidBoardPosition(position: Position, board: BoardState): boolean {
  const size = board.size;
  if (board.type === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z || -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  } else {
    return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
  }
}

/**
 * Get the marker owner at a position.
 */
function getMarkerOwner(position: Position, board: BoardState): number | undefined {
  const posKey = positionToString(position);
  const marker = board.markers.get(posKey);
  return marker?.player;
}

/**
 * Check if a position is a collapsed space.
 */
function isCollapsedSpace(position: Position, board: BoardState): boolean {
  const posKey = positionToString(position);
  return board.collapsedSpaces.has(posKey);
}

/**
 * Get stack at a position.
 */
function getStackAt(position: Position, board: BoardState): RingStack | undefined {
  const posKey = positionToString(position);
  return board.stacks.get(posKey);
}

/**
 * Find all positions in a line from a starting position in a given direction.
 */
function findLineInDirection(
  startPosition: Position,
  direction: Position,
  playerId: number,
  board: BoardState
): Position[] {
  const line: Position[] = [startPosition];
  const isHex = board.type === 'hexagonal';

  // Helper to step one cell in the given direction
  const step = (current: Position, sign: 1 | -1): Position => {
    if (isHex) {
      return {
        x: current.x + sign * direction.x,
        y: current.y + sign * direction.y,
        z: (current.z || 0) + sign * (direction.z || 0),
      };
    }
    return {
      x: current.x + sign * direction.x,
      y: current.y + sign * direction.y,
    };
  };

  // Check forward direction
  let current = startPosition;
  while (true) {
    const next = step(current, 1);

    if (!isValidBoardPosition(next, board)) break;

    const marker = getMarkerOwner(next, board);
    if (marker !== playerId) break;

    if (isCollapsedSpace(next, board) || getStackAt(next, board)) break;

    line.push(next);
    current = next;
  }

  // Check backward direction
  current = startPosition;
  while (true) {
    const prev = step(current, -1);

    if (!isValidBoardPosition(prev, board)) break;

    const marker = getMarkerOwner(prev, board);
    if (marker !== playerId) break;

    if (isCollapsedSpace(prev, board) || getStackAt(prev, board)) break;

    line.unshift(prev);
    current = prev;
  }

  return line;
}

/**
 * Canonical, order-independent key for a line based on its marker positions.
 */
function canonicalLineKey(line: LineInfo): string {
  return line.positions
    .map((p) => positionToString(p))
    .sort()
    .join('|');
}

/**
 * Compute the next canonical moveNumber for decision moves.
 */
function computeNextMoveNumber(state: GameState): number {
  if (state.history && state.history.length > 0) {
    const last = state.history[state.history.length - 1];
    if (typeof last.moveNumber === 'number' && last.moveNumber > 0) {
      return last.moveNumber + 1;
    }
  }

  if (state.moveHistory && state.moveHistory.length > 0) {
    const lastLegacy = state.moveHistory[state.moveHistory.length - 1];
    if (typeof lastLegacy.moveNumber === 'number' && lastLegacy.moveNumber > 0) {
      return lastLegacy.moveNumber + 1;
    }
  }

  return 1;
}

/**
 * Detect all lines for a given player.
 */
function detectPlayerLines(
  state: GameState,
  player: number,
  options?: LineEnumerationOptions
): LineInfo[] {
  const board = state.board;
  const mode = options?.detectionMode ?? 'use_board_cache';

  if (mode === 'use_board_cache' && board.formedLines && board.formedLines.length > 0) {
    return board.formedLines.filter((line) => line.player === player);
  }

  // Fresh detection; filter by owner.
  return findAllLines(board).filter((line) => line.player === player);
}

/**
 * Resolve the concrete LineInfo on the current board that a decision Move is
 * referring to.
 */
function resolveLineForMove(
  state: GameState,
  move: Move,
  options?: LineEnumerationOptions
): LineInfo | undefined {
  // When replaying canonical games or applying moves constructed from Python's
  // GameEngine, prefer the explicit line geometry carried on the Move itself.
  // This mirrors Python _apply_line_formation, which trusts move.formed_lines
  // rather than re-detecting from the current board. Re-detection can fail in
  // replay contexts where board.formedLines was never populated, leading to
  // missing collapses and TS↔Python structural mismatches.
  if (move.formedLines && move.formedLines.length > 0) {
    const target = move.formedLines[0] as LineInfo;

    // Normalise optional fields in case the source omitted them.
    const length = (target as any).length ?? target.positions.length;
    const direction =
      (target as any).direction ??
      (target.positions.length >= 2
        ? {
            x: target.positions[1].x - target.positions[0].x,
            y: target.positions[1].y - target.positions[0].y,
          }
        : { x: 1, y: 0 });

    return {
      ...target,
      length,
      direction,
    } as LineInfo;
  }

  const player = move.player;
  const playerLines = detectPlayerLines(state, player, options);
  if (playerLines.length === 0) {
    return undefined;
  }

  // Fallback: first detected line for the player when no explicit geometry was
  // supplied on the Move.
  return playerLines[0];
}

/**
 * Collapse the given marker positions into territory for player, returning
 * rings from any stacks on those spaces to the appropriate players' hands.
 */
function collapseLinePositions(
  state: GameState,
  positions: Position[],
  player: number
): { nextState: GameState; collapsedCount: number } {
  const board = state.board;

  // DEBUG: Trace input stacks
  if (process.env.NODE_ENV === 'test') {
    // eslint-disable-next-line no-console
    console.log('[collapseLinePositions] INPUT stacks:', {
      stackCount: board.stacks.size,
      stackKeys: Array.from(board.stacks.keys()),
      positionsToCollapse: positions.map((p) => positionToString(p)),
      player,
    });
  }

  const nextBoard = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };

  const nextPlayers = state.players.map((p) => ({ ...p }));
  const collapsedKeys = new Set<string>();

  for (const pos of positions) {
    const key = positionToString(pos);
    if (collapsedKeys.has(key)) {
      continue;
    }
    collapsedKeys.add(key);

    // Return any rings on this space to their owners' hands, then remove the
    // stack entirely.
    const stack = nextBoard.stacks.get(key);
    if (stack && Array.isArray(stack.rings) && stack.rings.length > 0) {
      for (const ringOwner of stack.rings as number[]) {
        const idx = nextPlayers.findIndex((p) => p.playerNumber === ringOwner);
        if (idx >= 0) {
          const current = nextPlayers[idx];
          nextPlayers[idx] = {
            ...current,
            ringsInHand: (current.ringsInHand ?? 0) + 1,
          };
        }
      }
      nextBoard.stacks.delete(key);
    }

    // Remove any marker at this position.
    if (nextBoard.markers.has(key)) {
      nextBoard.markers.delete(key);
    }

    // Mark as collapsed territory for the acting player.
    nextBoard.collapsedSpaces.set(key, player);
  }

  if (collapsedKeys.size > 0) {
    const idx = nextPlayers.findIndex((p) => p.playerNumber === player);
    if (idx >= 0) {
      const current = nextPlayers[idx];
      nextPlayers[idx] = {
        ...current,
        territorySpaces: current.territorySpaces + collapsedKeys.size,
      };
    }
  }

  // Drop any cached formedLines that intersect collapsed spaces
  if (nextBoard.formedLines && nextBoard.formedLines.length > 0) {
    nextBoard.formedLines = nextBoard.formedLines.filter((line) => {
      return !line.positions.some((p) => collapsedKeys.has(positionToString(p)));
    });
  }

  const nextState: GameState = {
    ...state,
    board: nextBoard,
    players: nextPlayers,
  };

  // DEBUG: Trace output stacks
  if (process.env.NODE_ENV === 'test') {
    // eslint-disable-next-line no-console
    console.log('[collapseLinePositions] OUTPUT stacks:', {
      stackCount: nextBoard.stacks.size,
      stackKeys: Array.from(nextBoard.stacks.keys()),
    });
  }

  return { nextState, collapsedCount: collapsedKeys.size };
}

// ═══════════════════════════════════════════════════════════════════════════
// Detection Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Detect all marker lines on the board (4+ for 8x8, 4+ for 19x19/hex)
 * according to the canonical RingRift line rules (Section 11.1).
 *
 * This helper is the single source of truth for line geometry used by:
 * - the shared GameEngine (advanced phases),
 * - the backend BoardManager / RuleEngine, and
 * - the client sandbox line engines.
 *
 * CRITICAL: Lines are formed by MARKERS, not stacks.
 *
 * Rule Reference: RR-CANON-R100, RR-CANON-R101
 */
export function findAllLines(board: BoardState): DetectedLine[] {
  const lines: DetectedLine[] = [];
  const processedLines = new Set<string>();
  const config = BOARD_CONFIGS[board.type];

  // Iterate through all MARKERS (not stacks!)
  for (const [posStr, marker] of board.markers) {
    const position = stringToPosition(posStr);

    // Treat stacks and collapsed spaces as hard blockers
    if (isCollapsedSpace(position, board) || getStackAt(position, board)) {
      continue;
    }

    const directions = getLineDirections(board.type);

    for (const direction of directions) {
      const line = findLineInDirection(position, direction, marker.player, board);

      if (line.length >= config.lineLength) {
        // Create a unique key for this line (sorted positions to avoid duplicates)
        const lineKey = line
          .map((p) => positionToString(p))
          .sort()
          .join('|');

        if (!processedLines.has(lineKey)) {
          processedLines.add(lineKey);
          lines.push({
            positions: line,
            player: marker.player,
            length: line.length,
            direction: direction,
          });
        }
      }
    }
  }

  return lines;
}

/**
 * Detect all marker lines on the board that belong to a specific player.
 * Thin convenience wrapper over findAllLines.
 */
export function findLinesForPlayer(board: BoardState, playerNumber: number): DetectedLine[] {
  return findAllLines(board).filter((line) => line.player === playerNumber);
}

/**
 * Find all lines that contain a specific position.
 *
 * This is useful for determining which lines are affected when a marker
 * at a specific position changes (placed, removed, or collapsed).
 */
export function findLinesContainingPosition(state: GameState, position: Position): DetectedLine[] {
  const posKey = positionToString(position);
  const allLines = findAllLines(state.board);

  return allLines.filter((line) => line.positions.some((p) => positionToString(p) === posKey));
}

// ═══════════════════════════════════════════════════════════════════════════
// Validation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Validate a PROCESS_LINE action against full GameState.
 *
 * Checks:
 * - Phase must be 'line_processing'
 * - Must be the player's turn
 * - Line index must be valid
 * - Line must belong to the acting player
 */
export function validateProcessLine(
  state: GameState,
  action: ProcessLineAction
): LineValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'line_processing') {
    return { valid: false, reason: 'Not in line processing phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Line Existence Check
  if (action.lineIndex < 0 || action.lineIndex >= state.board.formedLines.length) {
    return { valid: false, reason: 'Invalid line index', code: 'INVALID_LINE_INDEX' };
  }

  const line = state.board.formedLines[action.lineIndex];

  // 4. Line Ownership Check
  if (line.player !== action.playerId) {
    return { valid: false, reason: 'Cannot process opponent line', code: 'NOT_YOUR_LINE' };
  }

  return { valid: true };
}

/**
 * Validate a CHOOSE_LINE_REWARD action against full GameState.
 *
 * Checks:
 * - Phase must be 'line_processing'
 * - Must be the player's turn
 * - Line index must be valid
 * - Line must belong to the acting player
 * - Selection must be valid for the line length
 * - For MINIMUM_COLLAPSE, positions must be valid and consecutive
 */
export function validateChooseLineReward(
  state: GameState,
  action: ChooseLineRewardAction
): LineValidationResult {
  // 1. Phase Check
  if (state.currentPhase !== 'line_processing') {
    return { valid: false, reason: 'Not in line processing phase', code: 'INVALID_PHASE' };
  }

  // 2. Turn Check
  if (action.playerId !== state.currentPlayer) {
    return { valid: false, reason: 'Not your turn', code: 'NOT_YOUR_TURN' };
  }

  // 3. Line Existence Check
  if (action.lineIndex < 0 || action.lineIndex >= state.board.formedLines.length) {
    return { valid: false, reason: 'Invalid line index', code: 'INVALID_LINE_INDEX' };
  }

  const line = state.board.formedLines[action.lineIndex];

  // 4. Line Ownership Check
  if (line.player !== action.playerId) {
    return { valid: false, reason: 'Cannot process opponent line', code: 'NOT_YOUR_LINE' };
  }

  const requiredLength = getEffectiveLineLengthThreshold(
    state.board.type as BoardType,
    state.players.length,
    state.rulesOptions
  );

  // 5. Option Validity Check
  if (line.length === requiredLength) {
    // Exact length lines MUST be fully collapsed
    if (action.selection === 'MINIMUM_COLLAPSE') {
      return {
        valid: false,
        reason: 'Cannot choose minimum collapse for exact length line',
        code: 'INVALID_SELECTION',
      };
    }
  }

  if (action.selection === 'MINIMUM_COLLAPSE') {
    if (!action.collapsedPositions) {
      return {
        valid: false,
        reason: 'Must provide collapsed positions for minimum collapse',
        code: 'MISSING_POSITIONS',
      };
    }

    if (action.collapsedPositions.length !== requiredLength) {
      return {
        valid: false,
        reason: `Must select exactly ${requiredLength} positions`,
        code: 'INVALID_POSITION_COUNT',
      };
    }

    // Verify all selected positions are actually part of the line
    const linePosKeys = new Set(line.positions.map((p) => positionToString(p)));
    for (const pos of action.collapsedPositions) {
      if (!linePosKeys.has(positionToString(pos))) {
        return {
          valid: false,
          reason: 'Selected position is not part of the line',
          code: 'INVALID_POSITION',
        };
      }
    }

    // Verify selected positions are consecutive
    const indices = action.collapsedPositions
      .map((pos) => {
        const key = positionToString(pos);
        return line.positions.findIndex((p) => positionToString(p) === key);
      })
      .sort((a, b) => a - b);

    for (let i = 0; i < indices.length - 1; i++) {
      if (indices[i + 1] !== indices[i] + 1) {
        return {
          valid: false,
          reason: 'Selected positions must be consecutive',
          code: 'NON_CONSECUTIVE',
        };
      }
    }
  }

  return { valid: true };
}

/**
 * Unified validation for line decisions (convenience wrapper).
 *
 * Validates either a LineCollapseDecision or a standard action.
 */
export function validateLineDecision(
  state: GameState,
  decision: LineCollapseDecision
): LineValidationResult {
  const action: ChooseLineRewardAction = {
    type: 'CHOOSE_LINE_REWARD',
    playerId: decision.player,
    lineIndex: decision.lineIndex,
    selection: decision.selection,
    ...(decision.collapsedPositions && { collapsedPositions: decision.collapsedPositions }),
  };

  return validateChooseLineReward(state, action);
}

// ═══════════════════════════════════════════════════════════════════════════
// Enumeration Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enumerate `process_line` decision moves for the specified player in the
 * current `GameState`.
 *
 * Semantics:
 *
 * - Only lines owned by `player` are considered.
 * - Line geometry is derived either from `state.board.formedLines` or by
 *   invoking the shared `findAllLines` helper, depending on
 *   LineEnumerationOptions.detectionMode.
 * - Each returned Move has:
 *   - `type: 'process_line'`,
 *   - `player: player`,
 *   - `formedLines[0]` containing the selected line, and
 *   - `to` set to a representative position on that line (the first position).
 */
export function enumerateProcessLineMoves(
  state: GameState,
  player: number,
  options?: LineEnumerationOptions
): Move[] {
  const playerLines = detectPlayerLines(state, player, options);

  if (playerLines.length === 0) {
    return [];
  }

  const boardType = state.board.type as BoardType;
  // Use the effective line length threshold which accounts for player count
  // (e.g., square8 2-player requires 4, while 3-4 player requires 3).
  // This matches Python's BoardManager.find_all_lines and ensures ANM parity.
  const minLineLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );

  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  playerLines.forEach((line, index) => {
    // Filter out lines that do not meet the effective threshold for this
    // board type and player count. Per RR-CANON-R120, square8 2-player uses
    // line length 4, while 3-4 player uses 3.
    if (line.length < minLineLength) {
      return;
    }

    const representative = line.positions[0] ?? { x: 0, y: 0 };
    const lineKey = line.positions.map((p) => positionToString(p)).join('|');

    moves.push({
      id: `process-line-${index}-${lineKey}`,
      type: 'process_line',
      player,
      to: representative,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  });

  return moves;
}
/**
 * Enumerate `choose_line_reward` decision moves for a specific line that has
 * already been selected for processing.
 *
 * For lines longer than the minimum threshold:
 * - Option 1: collapse the entire line (with elimination reward)
 * - Option 2: collapse minimum contiguous subset (no reward)
 *
 * For exact-length lines:
 * - Only a single collapse-all variant is available
 */
export function enumerateChooseLineRewardMoves(
  state: GameState,
  player: number,
  lineIndex: number
): Move[] {
  if (lineIndex < 0) {
    return [];
  }

  const playerLines = detectPlayerLines(state, player);
  if (playerLines.length === 0 || lineIndex >= playerLines.length) {
    return [];
  }

  const line = playerLines[lineIndex];
  const boardType = state.board.type as BoardType;
  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );

  if (!line.positions || line.positions.length === 0) {
    return [];
  }

  if (line.length < requiredLength) {
    // Not a complete, collapsible line – no reward moves.
    return [];
  }

  const representative = line.positions[0] ?? { x: 0, y: 0 };
  const lineKey = line.positions.map((p) => positionToString(p)).join('|');
  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  // Exact-length line: a single collapse-all variant
  if (line.length === requiredLength) {
    moves.push({
      id: `choose-line-reward-${lineIndex}-${lineKey}-all`,
      type: 'choose_line_reward',
      player,
      to: representative,
      formedLines: [line],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);

    return moves;
  }

  // Overlength line (> requiredLength).
  // Option 1: collapse the entire line.
  moves.push({
    id: `choose-line-reward-${lineIndex}-${lineKey}-all`,
    type: 'choose_line_reward',
    player,
    to: representative,
    formedLines: [line],
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: nextMoveNumber,
  } as Move);

  // Option 2: all legal minimum-collapse segments of length L along the line.
  const maxStart = line.length - requiredLength;
  for (let start = 0; start <= maxStart; start++) {
    const segment = line.positions.slice(start, start + requiredLength);

    moves.push({
      id: `choose-line-reward-${lineIndex}-${lineKey}-min-${start}`,
      type: 'choose_line_reward',
      player,
      to: representative,
      formedLines: [line],
      collapsedMarkers: segment,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  }

  return moves;
}

/**
 * Enumerate all line collapse options for a specific line.
 *
 * This is the aggregate-level function that returns LineCollapseDecision
 * objects representing all valid collapse options.
 */
export function enumerateLineCollapseOptions(
  state: GameState,
  line: DetectedLine
): LineCollapseDecision[] {
  const boardType = state.board.type as BoardType;
  const requiredLength = getEffectiveLineLengthThreshold(
    boardType,
    state.players.length,
    state.rulesOptions
  );
  const decisions: LineCollapseDecision[] = [];

  // Find the line index
  const lineIndex = state.board.formedLines.findIndex(
    (l) => canonicalLineKey(l) === canonicalLineKey(line)
  );

  if (lineIndex < 0 || line.length < requiredLength) {
    return decisions;
  }

  // Option 1: Collapse all (always available)
  decisions.push({
    lineIndex,
    selection: 'COLLAPSE_ALL',
    line,
    player: line.player,
  });

  // Option 2: Minimum collapse segments (only for overlength lines)
  if (line.length > requiredLength) {
    const maxStart = line.length - requiredLength;
    for (let start = 0; start <= maxStart; start++) {
      const segment = line.positions.slice(start, start + requiredLength);
      decisions.push({
        lineIndex,
        selection: 'MINIMUM_COLLAPSE',
        collapsedPositions: segment,
        line,
        player: line.player,
      });
    }
  }

  return decisions;
}

// ═══════════════════════════════════════════════════════════════════════════
// Mutation Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply line processing mutation via action.
 *
 * For exact-length lines, this immediately collapses the line.
 * For overlength lines, this is a no-op (requires choose_line_reward).
 */
export function mutateProcessLine(state: GameState, action: ProcessLineAction): GameState {
  const line = state.board.formedLines[action.lineIndex];
  const requiredLength = getEffectiveLineLengthThreshold(
    state.board.type as BoardType,
    state.players.length,
    state.rulesOptions
  );

  if (line.length > requiredLength) {
    throw new Error('LineMutator: Line length > minimum requires ChooseLineRewardAction');
  }

  // Execute Option 1: Collapse All
  return executeLineCollapse(state, line.positions, action.lineIndex);
}

/**
 * Apply line reward choice mutation.
 */
export function mutateChooseLineReward(
  state: GameState,
  action: ChooseLineRewardAction
): GameState {
  const line = state.board.formedLines[action.lineIndex];

  if (action.selection === 'COLLAPSE_ALL') {
    return executeLineCollapse(state, line.positions, action.lineIndex);
  } else {
    // MINIMUM_COLLAPSE
    if (!action.collapsedPositions) {
      throw new Error('LineMutator: Missing collapsedPositions for MINIMUM_COLLAPSE');
    }
    return executeLineCollapse(state, action.collapsedPositions, action.lineIndex);
  }
}

/**
 * Execute the actual line collapse operation.
 *
 * Internal helper that:
 * - Removes stacks/markers at collapsed positions
 * - Marks those spaces as collapsed territory
 * - Returns rings to owners' hands
 * - Removes the processed line and any broken lines
 */
function executeLineCollapse(
  state: GameState,
  positionsToCollapse: Position[],
  lineIndex: number
): GameState {
  const newState = {
    ...state,
    board: {
      ...state.board,
      stacks: new Map(state.board.stacks),
      markers: new Map(state.board.markers),
      collapsedSpaces: new Map(state.board.collapsedSpaces),
      formedLines: [...state.board.formedLines],
      eliminatedRings: { ...state.board.eliminatedRings },
    },
    players: state.players.map((p) => ({ ...p })),
    moveHistory: [...state.moveHistory],
  } as GameState & {
    totalRingsEliminated: number;
    lastMoveAt: Date;
    totalRingsInPlay: number;
  };

  const player = newState.players.find((p) => p.playerNumber === newState.currentPlayer);
  if (!player) throw new Error('LineMutator: Player not found');

  // 1. Remove stacks/markers at collapsed positions and mark as collapsed
  for (const pos of positionsToCollapse) {
    const key = positionToString(pos);

    // Remove stack if any - rings returned to hand
    const stack = newState.board.stacks.get(key);
    if (stack) {
      for (const ringOwner of stack.rings) {
        const p = newState.players.find((pl) => pl.playerNumber === ringOwner);
        if (p) {
          p.ringsInHand++;
          newState.totalRingsInPlay--;
        }
      }
      newState.board.stacks.delete(key);
    }

    // Remove marker if any
    if (newState.board.markers.has(key)) {
      newState.board.markers.delete(key);
    }

    // Mark as collapsed territory
    newState.board.collapsedSpaces.set(key, newState.currentPlayer);
  }

  // Update territory count
  const idx = newState.players.findIndex((p) => p.playerNumber === newState.currentPlayer);
  if (idx >= 0) {
    newState.players[idx].territorySpaces += positionsToCollapse.length;
  }

  // 2. Remove the processed line from formedLines
  newState.board.formedLines.splice(lineIndex, 1);

  // 3. Check if other lines are broken by this collapse
  const collapsedKeys = new Set(positionsToCollapse.map((p) => positionToString(p)));

  newState.board.formedLines = newState.board.formedLines.filter((l) => {
    for (const pos of l.positions) {
      if (collapsedKeys.has(positionToString(pos))) {
        return false; // Line broken
      }
    }
    return true;
  });

  newState.lastMoveAt = new Date();
  return newState;
}

/**
 * Apply a `process_line` move produced by enumerateProcessLineMoves.
 *
 * For exact-length lines: collapses the line and sets pendingLineRewardElimination.
 * For overlength lines: no-op (requires choose_line_reward).
 */
export function applyProcessLineDecision(
  state: GameState,
  move: Move
): LineDecisionApplicationOutcome {
  if (move.type !== 'process_line') {
    throw new Error(
      `applyProcessLineDecision expected move.type === 'process_line', got '${move.type}'`
    );
  }

  const line = resolveLineForMove(state, move);
  if (!line) {
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  const boardType = state.board.type as BoardType;
  const numPlayers = state.players.length;
  const requiredLength = getEffectiveLineLengthThreshold(boardType, numPlayers, state.rulesOptions);
  const minLineLength = BOARD_CONFIGS[boardType].lineLength;

  // Python's GameEngine treats any detected line (>= base lineLength) as a
  // valid PROCESS_LINE target, even when it is shorter than the effective
  // reward threshold for the current player count (for example, 3-in-a-row on
  // 2p square8 where requiredLength === 4). Such "mini" lines collapse to
  // territory but do **not** grant a mandatory elimination reward.
  if (line.length < minLineLength) {
    // Not actually a complete line for this board; treat as no-op.
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  // Overlength lines (longer than the effective threshold) require a
  // choose_line_reward decision per RR-CANON-R122. Treating process_line
  // on an overlength line as a no-op avoids silently defaulting to either
  // option (collapse-all vs min-collapse).
  if (line.length > requiredLength) {
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  const { nextState } = collapseLinePositions(state, line.positions, move.player);

  return {
    nextState,
    // Elimination reward is only granted when the collapsed line meets or
    // exceeds the effective threshold for this board/player-count combination.
    // Shorter (mini) lines still grant territory but no mandatory
    // self-elimination.
    pendingLineRewardElimination: line.length >= requiredLength,
  };
}

/**
 * Apply a `choose_line_reward` move produced by enumerateChooseLineRewardMoves.
 *
 * Semantics (Python-aligned, including legacy fixtures):
 *
 * - When `move.collapsedMarkers` is provided and non-empty:
 *   - Collapse exactly those marker positions, even if the line length equals
 *     the effective threshold or the subset length is less than the threshold.
 *   - This mirrors Python GameEngine._apply_line_formation, which always trusts
 *     `collapsed_markers` when present (including historical DBs that carry
 *     partial segments for exact-length lines).
 * - When `move.collapsedMarkers` is absent or empty:
 *   - Collapse the entire line.
 *
 * Elimination reward semantics:
 *
 * - A mandatory self-elimination reward is only granted when:
 *   - the line length is at least the effective threshold for this
 *     board/player-count combination, and
 *   - the collapse covers the entire line (all markers in the line).
 * - Minimum-collapse variants (proper subsets of the line) never set
 *   `pendingLineRewardElimination`, regardless of line length.
 */
export function applyChooseLineRewardDecision(
  state: GameState,
  move: Move
): LineDecisionApplicationOutcome {
  if (move.type !== 'choose_line_reward') {
    throw new Error(
      `applyChooseLineRewardDecision expected move.type === 'choose_line_reward', got '${move.type}'`
    );
  }

  const line = resolveLineForMove(state, move);
  if (!line) {
    return {
      nextState: state,
      pendingLineRewardElimination: false,
    };
  }

  const boardType = state.board.type as BoardType;
  const numPlayers = state.players.length;
  const requiredLength = getEffectiveLineLengthThreshold(boardType, numPlayers, state.rulesOptions);
  const length = line.length;

  // Determine which positions to collapse, preferring explicit geometry from the move
  // when present. This matches Python's _apply_line_formation, which always uses
  // collapsed_markers when provided.
  const collapsed = move.collapsedMarkers;
  let positionsToCollapse: Position[];

  if (collapsed && collapsed.length > 0) {
    positionsToCollapse = collapsed;
  } else {
    positionsToCollapse = line.positions;
  }

  // Decide whether this collapse should grant a mandatory elimination reward.
  // Reward is only pending when:
  // - the line is at least the effective threshold length, and
  // - the collapse covers the entire line (no markers from this line remain).
  const collapsedKeys = new Set(positionsToCollapse.map((p) => positionToString(p)));
  const lineKeys = new Set(line.positions.map((p) => positionToString(p)));
  const isFullCollapse =
    collapsedKeys.size === lineKeys.size && Array.from(collapsedKeys).every((key) => lineKeys.has(key));

  const pendingReward = length >= requiredLength && isFullCollapse;

  const { nextState } = collapseLinePositions(state, positionsToCollapse, move.player);

  return {
    nextState,
    pendingLineRewardElimination: pendingReward,
  };
}

/**
 * Apply a line collapse decision and return a result type for easier error handling.
 *
 * This is a unified wrapper that handles both process_line and choose_line_reward moves.
 */
export function applyLineCollapse(
  state: GameState,
  decision: LineCollapseDecision
): LineMutationResult {
  try {
    const action: ChooseLineRewardAction = {
      type: 'CHOOSE_LINE_REWARD',
      playerId: decision.player,
      lineIndex: decision.lineIndex,
      selection: decision.selection,
      ...(decision.collapsedPositions && { collapsedPositions: decision.collapsedPositions }),
    };

    const validation = validateChooseLineReward(state, action);
    if (!validation.valid) {
      return {
        success: false,
        reason: validation.reason ?? 'Invalid line collapse decision',
      };
    }

    const newState = mutateChooseLineReward(state, action);
    return {
      success: true,
      newState,
    };
  } catch (error) {
    return {
      success: false,
      reason: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}
