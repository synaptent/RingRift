import { BoardType, BOARD_CONFIGS, Move, Position } from '../types/game';

/**
 * Shared move-notation helpers.
 *
 * These utilities provide a lightweight, human-readable notation for
 * debugging, logging, and UI display built on top of the canonical Move
 * model. The notation is intentionally simple and does not attempt to be
 * a full algebraic system; it is designed to make traces and histories
 * easy to scan in tests and tools.
 */

export interface MoveNotationOptions {
  /**
   * Board type used to format positions. When omitted, square8-style
   * coordinates (a1, b3, etc.) are used for non-negative coordinates and
   * raw x,y,z tuples are used otherwise.
   */
  boardType?: BoardType;
  /**
   * Optional override for board size when formatting square coordinates.
   * Defaults to BOARD_CONFIGS[boardType].size when available.
   */
  boardSizeOverride?: number;
  /**
   * When true, square board ranks are computed from the bottom (y = size-1)
   * rather than from the top (y = 0). This aligns with traditional chess-style
   * display where the bottom row is rank 1.
   *
   * - Default (false): rank = y + 1 (y=0 → rank 1, y=7 → rank 8)
   * - With squareRankFromBottom (true): rank = boardSize - y (y=7 → rank 1, y=0 → rank 8)
   *
   * Use this option in sandbox/teaching contexts where the board labels show
   * the bottom row as rank 1, to keep MoveHistory coordinates aligned with
   * the visual board edge labels.
   */
  squareRankFromBottom?: boolean;
}

/**
 * Format a board position into a compact string.
 *
 * For square boards (BOARD_CONFIGS[boardType].type === 'square') this
 * returns chess-like coordinates using 0-based x,y:
 *   (0,0) -> a1, (1,0) -> b1, ..., (0,1) -> a2, etc.
 *
 * For hex boards or positions with negative coordinates, a raw tuple
 * "(x,y,z)" is used to avoid ambiguity.
 */
export function formatPosition(pos: Position, options: MoveNotationOptions = {}): string {
  const boardType = options.boardType ?? 'square8';
  const config = BOARD_CONFIGS[boardType];

  const isSquareBoard = config.type === 'square';

  // Use algebraic-style coordinates only for square boards with
  // non-negative coordinates; otherwise fall back to raw tuple.
  if (isSquareBoard && pos.x >= 0 && pos.y >= 0) {
    const fileCode = 'a'.charCodeAt(0) + pos.x;
    const file = String.fromCharCode(fileCode);

    // Determine the rank based on the squareRankFromBottom option.
    // Default: rank = y + 1 (y=0 → rank 1)
    // With squareRankFromBottom: rank = boardSize - y (y=size-1 → rank 1)
    let rankNum: number;
    if (options.squareRankFromBottom) {
      const boardSize = options.boardSizeOverride ?? config.size;
      rankNum = boardSize - pos.y;
    } else {
      rankNum = pos.y + 1;
    }

    return `${file}${rankNum}`;
  }

  if (config.type === 'hexagonal') {
    // Map hex coordinates (q, r, s) to algebraic-like (File+Rank).
    // We treat 'q' (vertical) as Rank and 'r' (oblique) as File.
    // Radius is size - 1.
    const radius = config.size - 1;

    // Rank: Map q from [-radius, radius] to [2*radius + 1, 1]
    // Top row (q = -radius) -> Rank 2*radius + 1
    // Bottom row (q = radius) -> Rank 1
    const rankNum = radius - pos.x + 1;

    // File: Map r from [-radius, radius] to ['a', ...]
    // We shift r by radius to get a 0-based index.
    const fileCode = 'a'.charCodeAt(0) + (pos.y + radius);
    const file = String.fromCharCode(fileCode);

    return `${file}${rankNum}`;
  }

  const z = pos.z !== undefined ? pos.z : -pos.x - pos.y;
  return `(${pos.x},${pos.y},${z})`;
}

function formatMoveType(move: Move): string {
  switch (move.type) {
    case 'place_ring':
      return 'R';
    case 'move_ring':
    case 'move_stack':
      return 'M';
    case 'overtaking_capture':
      return 'C';
    case 'line_formation':
      return 'L';
    case 'territory_claim':
      return 'T';
    case 'skip_placement':
      return 'S';
    case 'recovery_slide':
      return 'Rv'; // Recovery slide (RR-CANON-R110–R115)
    case 'skip_recovery':
      return 'RvS'; // Skip recovery (RR-CANON-R115)
    default:
      return move.type;
  }
}

/**
 * Format a canonical Move into a one-line debug notation string.
 *
 * Examples (for square boards):
 *   P1: R a3
 *   P2: M c3→c7
 *   P3: C d5×e6→f7
 */
export function formatMove(move: Move, options: MoveNotationOptions = {}): string {
  const prefix = `P${move.player}:`;
  const kind = formatMoveType(move);

  const toPos = move.to ? formatPosition(move.to, options) : undefined;
  const fromPos = move.from ? formatPosition(move.from, options) : undefined;
  const targetPos = move.captureTarget ? formatPosition(move.captureTarget, options) : undefined;

  if (move.type === 'place_ring') {
    const count = move.placementCount && move.placementCount > 1 ? ` x${move.placementCount}` : '';
    return `${prefix} ${kind} ${toPos ?? '?'}${count}`;
  }

  if (move.type === 'move_ring' || move.type === 'move_stack') {
    if (fromPos && toPos) {
      return `${prefix} ${kind} ${fromPos}→${toPos}`;
    }
    if (toPos) {
      return `${prefix} ${kind} ${toPos}`;
    }
    return `${prefix} ${kind}`;
  }

  if (move.type === 'overtaking_capture') {
    if (fromPos && targetPos && toPos) {
      return `${prefix} ${kind} ${fromPos}×${targetPos}→${toPos}`;
    }
    if (fromPos && toPos) {
      return `${prefix} ${kind} ${fromPos}→${toPos}`;
    }
    return `${prefix} ${kind}`;
  }

  // Recovery slide (RR-CANON-R110–R115): marker slide that completes a line
  if (move.type === 'recovery_slide') {
    const optSuffix = move.recoveryOption ? ` [opt${move.recoveryOption}]` : '';
    if (fromPos && toPos) {
      return `${prefix} ${kind} ${fromPos}→${toPos}${optSuffix}`;
    }
    if (toPos) {
      return `${prefix} ${kind} ${toPos}${optSuffix}`;
    }
    return `${prefix} ${kind}${optSuffix}`;
  }

  // Fallback for other/legacy move types.
  const parts: string[] = [`${prefix} ${kind}`];
  if (fromPos) parts.push(fromPos);
  if (toPos) parts.push(`→${toPos}`);
  return parts.join(' ');
}

/**
 * Very small helper to render a list of moves as numbered notation lines.
 * Primarily used by tests, logs, and debug tools.
 */
export function formatMoveList(moves: Move[], options: MoveNotationOptions = {}): string[] {
  return moves.map((m, idx) => `${idx + 1}. ${formatMove(m, options)}`);
}
