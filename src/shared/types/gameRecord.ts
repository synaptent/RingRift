/**
 * TypeScript types for RingRift Game Records
 *
 * Game records are the canonical format for storing completed games, supporting:
 * - Training data pipelines (JSONL export)
 * - Replay systems
 * - Analysis tooling
 * - Historical game storage
 *
 * Mirrors Python types from ai-service/app/models/game_record.py
 */

import {
  BoardType,
  LegacyMoveType,
  MoveType,
  Position,
  LineInfo,
  Territory,
  Move,
  ProgressSnapshot,
  positionToString,
  stringToPosition,
} from './game';

const LEGACY_MOVE_TYPES: readonly LegacyMoveType[] = [
  'move_ring',
  'build_stack',
  'choose_line_reward',
  'process_territory_region',
  'line_formation',
  'territory_claim',
];

function isLegacyMoveType(moveType: MoveType): moveType is LegacyMoveType {
  return (LEGACY_MOVE_TYPES as readonly string[]).includes(moveType);
}

// ────────────────────────────────────────────────────────────────────────────
// Enums and Type Unions
// ────────────────────────────────────────────────────────────────────────────

/** How the game ended. */
export type GameOutcome =
  | 'ring_elimination'
  | 'territory_control'
  | 'last_player_standing'
  | 'timeout'
  | 'resignation'
  | 'draw'
  | 'abandonment';

/** Where the game record originated. */
export type RecordSource =
  | 'online_game'
  | 'self_play'
  | 'cmaes_optimization'
  | 'tournament'
  | 'soak_test'
  | 'manual_import';

// ────────────────────────────────────────────────────────────────────────────
// Player Record Info
// ────────────────────────────────────────────────────────────────────────────

/** Player information as recorded in a game record. */
export interface PlayerRecordInfo {
  playerNumber: number;
  username: string;
  playerType: 'human' | 'ai';
  ratingBefore?: number;
  ratingAfter?: number;
  aiDifficulty?: number;
  aiType?: string;
}

// ────────────────────────────────────────────────────────────────────────────
// Move Record
// ────────────────────────────────────────────────────────────────────────────

/**
 * Lightweight move record for storage and training data.
 *
 * This is a simplified version of Move that retains only the fields
 * needed for replay, training, and analysis. Full diagnostic fields
 * (stackMoved, capturedStacks, etc.) are omitted for space efficiency.
 */
export interface MoveRecord {
  moveNumber: number;
  player: number;
  type: MoveType;
  from?: Position;
  to?: Position;

  // Capture metadata (when applicable)
  captureTarget?: Position;

  // Placement metadata
  placementCount?: number;
  placedOnStack?: boolean;

  // Line/territory processing metadata
  formedLines?: LineInfo[];
  collapsedMarkers?: Position[];
  disconnectedRegions?: Territory[];
  eliminatedRings?: { player: number; count: number }[];

  // Timing
  thinkTimeMs: number;

  // Optional RingRift Notation representation
  rrn?: string;
}

/**
 * Create a MoveRecord from a full Move object.
 *
 * Uses conditional spreading to avoid assigning `undefined` to optional properties,
 * which is required by TypeScript's `exactOptionalPropertyTypes` setting.
 */
export function moveToMoveRecord(move: Move): MoveRecord {
  return {
    moveNumber: move.moveNumber,
    player: move.player,
    type: move.type,
    thinkTimeMs: move.thinkTime,
    ...(move.from !== undefined && { from: move.from }),
    ...(move.to !== undefined && { to: move.to }),
    ...(move.captureTarget !== undefined && { captureTarget: move.captureTarget }),
    ...(move.placementCount !== undefined && { placementCount: move.placementCount }),
    ...(move.placedOnStack !== undefined && { placedOnStack: move.placedOnStack }),
    ...(move.formedLines !== undefined && { formedLines: move.formedLines }),
    ...(move.collapsedMarkers !== undefined && { collapsedMarkers: move.collapsedMarkers }),
    ...(move.disconnectedRegions !== undefined && {
      disconnectedRegions: move.disconnectedRegions,
    }),
    ...(move.eliminatedRings !== undefined && { eliminatedRings: move.eliminatedRings }),
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Game Record Metadata
// ────────────────────────────────────────────────────────────────────────────

/** Metadata about the game record itself (not the game). */
export interface GameRecordMetadata {
  recordVersion: string;
  createdAt: Date | string;
  source: RecordSource;
  sourceId?: string; // e.g., CMA-ES run ID
  generation?: number; // For evolutionary algorithms
  candidateId?: number;
  tags: string[];
}

// ────────────────────────────────────────────────────────────────────────────
// Final Score
// ────────────────────────────────────────────────────────────────────────────

/** Final score breakdown at game end. */
export interface FinalScore {
  ringsEliminated: Record<number, number>;
  territorySpaces: Record<number, number>;
  ringsRemaining: Record<number, number>;
}

// ────────────────────────────────────────────────────────────────────────────
// Game Record
// ────────────────────────────────────────────────────────────────────────────

/**
 * Complete record of a finished RingRift game.
 *
 * This is the canonical format for storing games for:
 * - Training data generation (JSONL export)
 * - Replay viewing
 * - Statistical analysis
 * - Historical record keeping
 *
 * The record is designed to be:
 * - Space-efficient (no redundant board snapshots)
 * - Self-contained (all info needed to replay without external state)
 * - Forward-compatible (versioned schema with optional fields)
 */
export interface GameRecord {
  // Unique identifier
  id: string;

  // Game configuration
  boardType: BoardType;
  numPlayers: number;
  rngSeed?: number;
  isRated: boolean;

  // Players
  players: PlayerRecordInfo[];

  // Game result
  winner?: number;
  outcome: GameOutcome;
  finalScore: FinalScore;

  // Timing
  startedAt: Date | string;
  endedAt: Date | string;
  totalMoves: number;
  totalDurationMs: number;

  // Move history
  moves: MoveRecord[];

  // Record metadata
  metadata: GameRecordMetadata;

  // Optional extended data
  initialStateHash?: string;
  finalStateHash?: string;
  progressSnapshots?: ProgressSnapshot[];
}

/**
 * Serialize a GameRecord to a single JSONL line for training data pipelines.
 */
export function gameRecordToJsonlLine(record: GameRecord): string {
  return JSON.stringify(record);
}

/**
 * Deserialize a GameRecord from a JSONL line.
 */
export function jsonlLineToGameRecord(line: string): GameRecord {
  return JSON.parse(line) as GameRecord;
}

// ────────────────────────────────────────────────────────────────────────────
// RingRift Notation (RRN)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Coordinate representation for RingRift Notation.
 *
 * Square boards use algebraic notation: a1-h8 (8x8) or a1-s19 (19x19)
 * Hexagonal boards use axial notation: (x,y,z) or simplified (x,y)
 */
export interface RRNCoordinate {
  notation: string;
  position: Position;
}

/**
 * Convert a Position to RRN coordinate notation.
 */
export function positionToRRN(pos: Position, boardType: BoardType): string {
  if (boardType === 'hexagonal') {
    if (pos.z !== undefined) {
      return `(${pos.x},${pos.y},${pos.z})`;
    }
    return `(${pos.x},${pos.y})`;
  }
  // Square boards: column letter + row number (1-indexed)
  const colLetter = String.fromCharCode('a'.charCodeAt(0) + pos.x);
  const rowNumber = pos.y + 1;
  return `${colLetter}${rowNumber}`;
}

/**
 * Parse RRN coordinate notation to Position.
 */
export function rrnToPosition(notation: string, boardType: BoardType): Position {
  const trimmed = notation.trim();

  if (boardType === 'hexagonal') {
    // Hex notation: (x,y) or (x,y,z)
    if (trimmed.startsWith('(') && trimmed.endsWith(')')) {
      const parts = trimmed.slice(1, -1).split(',');
      if (parts.length === 2) {
        return { x: parseInt(parts[0]), y: parseInt(parts[1]) };
      } else if (parts.length === 3) {
        return { x: parseInt(parts[0]), y: parseInt(parts[1]), z: parseInt(parts[2]) };
      }
    }
    throw new Error(`Invalid hex coordinate: ${notation}`);
  }

  // Square notation: letter + number (e.g., a1, h8, s19)
  if (trimmed.length < 2) {
    throw new Error(`Invalid square coordinate: ${notation}`);
  }
  const col = trimmed.charCodeAt(0) - 'a'.charCodeAt(0);
  const row = parseInt(trimmed.slice(1)) - 1;
  return { x: col, y: row };
}

/**
 * RingRift Notation representation of a single move.
 *
 * Notation format:
 * - Placement: P{coord} or P{coord}x{count} for multi-ring
 * - Movement: {from}-{to}
 * - Capture: {from}x{target}-{to}
 * - Chain capture: {from}x{target}-{to}+
 * - Line processing: L{coord} (first marker in line)
 * - Territory processing: T{coord} (representative position)
 * - Skip: -
 * - Swap sides: S
 *
 * Examples:
 * - "Pa1"       - Place ring at a1
 * - "e4-e6"     - Move stack from e4 to e6
 * - "d4xd5-d6"  - Capture at d5, land at d6
 * - "d4xd5-d6+" - Capture with continuation available
 * - "La3"       - Process line starting at a3
 * - "Tb2"       - Process territory region at b2
 * - "-"         - Skip (placement/capture)
 * - "S"         - Swap sides (pie rule)
 */
export interface RRNMove {
  notation: string;
  moveType: MoveType;
  player: number;
}

/**
 * Generate RingRift Notation string from a MoveRecord.
 *
 * Canonical-only. Legacy move types must be handled via
 * `src/shared/engine/legacy/legacyGameRecord.ts`.
 */
export function moveRecordToRRN(record: MoveRecord, boardType: BoardType): string {
  if (isLegacyMoveType(record.type)) {
    throw new Error(
      `Legacy move type '${record.type}' is not supported by canonical RRN. ` +
        'Use legacy helpers in src/shared/engine/legacy for legacy records.'
    );
  }

  const posToStr = (pos: Position | undefined): string => {
    if (!pos) return '?';
    return positionToRRN(pos, boardType);
  };

  const t = record.type;

  if (t === 'place_ring') {
    let base = `P${posToStr(record.to)}`;
    if (record.placementCount && record.placementCount > 1) {
      base += `x${record.placementCount}`;
    }
    return base;
  }

  if (t === 'skip_placement') {
    return '-';
  }

  if (t === 'swap_sides') {
    return 'S';
  }

  if (t === 'move_stack') {
    return `${posToStr(record.from)}-${posToStr(record.to)}`;
  }

  if (t === 'overtaking_capture') {
    return `${posToStr(record.from)}x${posToStr(record.captureTarget)}-${posToStr(record.to)}`;
  }

  if (t === 'continue_capture_segment') {
    return `${posToStr(record.from)}x${posToStr(record.captureTarget)}-${posToStr(record.to)}+`;
  }

  if (t === 'process_line') {
    if (record.formedLines && record.formedLines.length > 0) {
      const firstPos = record.formedLines[0].positions[0];
      return `L${posToStr(firstPos)}`;
    }
    return 'L?';
  }

  if (t === 'choose_line_option') {
    // O1 for option 1 (collapse all), O2 for option 2 (minimum collapse)
    if (record.formedLines && record.collapsedMarkers) {
      const lineLen = record.formedLines[0].positions.length;
      const collapsedLen = record.collapsedMarkers.length;
      return collapsedLen === lineLen ? 'O1' : 'O2';
    }
    return 'O?';
  }

  if (t === 'choose_territory_option') {
    if (record.disconnectedRegions && record.disconnectedRegions.length > 0) {
      const repPos = record.disconnectedRegions[0].spaces[0];
      return `T${posToStr(repPos)}`;
    }
    return 'T?';
  }

  if (t === 'eliminate_rings_from_stack') {
    return `E${posToStr(record.to)}`;
  }

  // Fallback for legacy/unknown move types
  return `?${t}`;
}

/**
 * Parsed result from an RRN move notation.
 */
export interface ParsedRRNMove {
  moveType: MoveType;
  from?: Position;
  to?: Position;
}

/**
 * Parse a RingRift Notation move string.
 *
 * Returns parsed move type and positions.
 * For moves without spatial coordinates, positions may be undefined.
 *
 * Canonical-only. Legacy notation must be handled via
 * `src/shared/engine/legacy/legacyGameRecord.ts`.
 */
export function parseRRNMove(notation: string, boardType: BoardType): ParsedRRNMove {
  const trimmed = notation.trim();

  if (trimmed === '-') {
    return { moveType: 'skip_placement' };
  }

  if (trimmed === 'S') {
    return { moveType: 'swap_sides' };
  }

  if (trimmed.startsWith('P')) {
    const rest = trimmed.slice(1);
    const coordPart = rest.includes('x') ? rest.split('x')[0] : rest;
    const pos = rrnToPosition(coordPart, boardType);
    return { moveType: 'place_ring', to: pos };
  }

  if (trimmed.startsWith('L')) {
    const pos = rrnToPosition(trimmed.slice(1), boardType);
    return { moveType: 'process_line', to: pos };
  }

  if (trimmed.startsWith('T')) {
    const pos = rrnToPosition(trimmed.slice(1), boardType);
    return { moveType: 'choose_territory_option', to: pos };
  }

  if (trimmed.startsWith('E')) {
    const pos = rrnToPosition(trimmed.slice(1), boardType);
    return { moveType: 'eliminate_rings_from_stack', to: pos };
  }

  if (trimmed === 'O1' || trimmed === 'O2') {
    return { moveType: 'choose_line_option' };
  }

  // Movement or capture: {from}-{to} or {from}x{target}-{to}
  if (trimmed.includes('x')) {
    const cleaned = trimmed.replace('+', '');
    const [fromStr, rest] = cleaned.split('x');
    const targetTo = rest.split('-');
    const toStr = targetTo.length > 1 ? targetTo[1] : targetTo[0];

    const fromPos = rrnToPosition(fromStr, boardType);
    const toPos = rrnToPosition(toStr, boardType);

    const moveType: MoveType = trimmed.endsWith('+')
      ? 'continue_capture_segment'
      : 'overtaking_capture';

    return { moveType, from: fromPos, to: toPos };
  }

  if (trimmed.includes('-')) {
    const [fromStr, toStr] = trimmed.split('-');
    const fromPos = rrnToPosition(fromStr, boardType);
    const toPos = rrnToPosition(toStr, boardType);
    return { moveType: 'move_stack', from: fromPos, to: toPos };
  }

  if (trimmed.includes('>')) {
    throw new Error(
      `Legacy RRN notation '${trimmed}' is not supported by canonical parsing. ` +
        'Use legacy helpers in src/shared/engine/legacy for legacy records.'
    );
  }

  throw new Error(`Unable to parse RRN: ${notation}`);
}

/**
 * Convert a complete GameRecord to RRN notation string.
 *
 * Format: {board_type}:{num_players}:{seed?}:{moves...}
 * Example: "square8:2:12345:Pa1 Pa8 d4-d6 d8-d4 d6xd5-d4"
 */
export function gameRecordToRRN(record: GameRecord): string {
  const headerParts = [record.boardType, String(record.numPlayers)];

  if (record.rngSeed !== undefined) {
    headerParts.push(String(record.rngSeed));
  } else {
    headerParts.push('_');
  }

  const moveStrs = record.moves.map((move) => moveRecordToRRN(move, record.boardType));

  const header = headerParts.join(':');
  const moves = moveStrs.join(' ');
  return `${header}:${moves}`;
}

/**
 * Parsed result from a complete RRN string.
 */
export interface ParsedRRN {
  boardType: BoardType;
  numPlayers: number;
  rngSeed?: number;
  moves: ParsedRRNMove[];
}

/**
 * Parse an RRN string to extract board config and move list.
 *
 * Uses conditional spreading to avoid assigning `undefined` to optional properties,
 * which is required by TypeScript's `exactOptionalPropertyTypes` setting.
 */
export function rrnToMoves(rrnString: string): ParsedRRN {
  const parts = rrnString.split(':');
  if (parts.length < 4) {
    throw new Error(`Invalid RRN format: ${rrnString}`);
  }

  const boardType = parts[0] as BoardType;
  const numPlayers = parseInt(parts[1]);
  const rngSeedStr = parts[2];
  const movesStr = parts.slice(3).join(':'); // Rejoin in case moves contain colons

  const moves = movesStr
    .split(/\s+/)
    .filter((s) => s.length > 0)
    .map((moveNotation) => parseRRNMove(moveNotation, boardType));

  return {
    boardType,
    numPlayers,
    moves,
    ...(rngSeedStr !== '_' && { rngSeed: parseInt(rngSeedStr) }),
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Coordinate Conversion Utilities
// ────────────────────────────────────────────────────────────────────────────

/**
 * Convert between different coordinate systems for all board types.
 */
export const CoordinateUtils = {
  /**
   * Convert a position to a human-readable algebraic notation.
   * For square boards: a1-h8 (8x8) or a1-s19 (19x19)
   * For hex boards: (x,y) or (x,y,z)
   */
  toAlgebraic: positionToRRN,

  /**
   * Parse algebraic notation back to a Position.
   */
  fromAlgebraic: rrnToPosition,

  /**
   * Convert position to key string (e.g., "3,4" or "3,4,2").
   */
  toKey: positionToString,

  /**
   * Parse key string back to Position.
   */
  fromKey: stringToPosition,

  /**
   * Get all valid positions for a board type.
   */
  getAllPositions(boardType: BoardType): Position[] {
    const positions: Position[] = [];

    if (boardType === 'square8') {
      for (let x = 0; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          positions.push({ x, y });
        }
      }
    } else if (boardType === 'square19') {
      for (let x = 0; x < 19; x++) {
        for (let y = 0; y < 19; y++) {
          positions.push({ x, y });
        }
      }
    } else if (boardType === 'hexagonal') {
      // Hexagonal board with radius 11 using axial coordinates
      const radius = 11;
      for (let q = -radius; q <= radius; q++) {
        for (let r = Math.max(-radius, -q - radius); r <= Math.min(radius, -q + radius); r++) {
          const s = -q - r;
          positions.push({ x: q, y: r, z: s });
        }
      }
    }

    return positions;
  },

  /**
   * Check if a position is valid for a given board type.
   */
  isValid(pos: Position, boardType: BoardType): boolean {
    if (boardType === 'square8') {
      return pos.x >= 0 && pos.x < 8 && pos.y >= 0 && pos.y < 8;
    } else if (boardType === 'square19') {
      return pos.x >= 0 && pos.x < 19 && pos.y >= 0 && pos.y < 19;
    } else if (boardType === 'hexagonal') {
      // Hexagonal validation using cube coordinates
      const radius = 11;
      const q = pos.x;
      const r = pos.y;
      const s = pos.z ?? -q - r;
      return (
        Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
      );
    }
    return false;
  },

  /**
   * Calculate Manhattan distance for square boards,
   * or hex distance for hexagonal boards.
   */
  distance(a: Position, b: Position, boardType: BoardType): number {
    if (boardType === 'hexagonal') {
      // Hex distance using cube coordinates
      const aq = a.x,
        ar = a.y,
        as = a.z ?? -a.x - a.y;
      const bq = b.x,
        br = b.y,
        bs = b.z ?? -b.x - b.y;
      return Math.max(Math.abs(aq - bq), Math.abs(ar - br), Math.abs(as - bs));
    }
    // Chebyshev distance for square boards (8-direction movement)
    return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
  },

  /**
   * Get adjacent positions for a given position.
   */
  getAdjacent(pos: Position, boardType: BoardType): Position[] {
    const adjacent: Position[] = [];

    if (boardType === 'hexagonal') {
      // 6 hex directions
      const hexDirs = [
        { dq: 1, dr: 0 },
        { dq: 1, dr: -1 },
        { dq: 0, dr: -1 },
        { dq: -1, dr: 0 },
        { dq: -1, dr: 1 },
        { dq: 0, dr: 1 },
      ];
      for (const dir of hexDirs) {
        const newPos: Position = {
          x: pos.x + dir.dq,
          y: pos.y + dir.dr,
          z: (pos.z ?? -pos.x - pos.y) - dir.dq - dir.dr,
        };
        if (CoordinateUtils.isValid(newPos, boardType)) {
          adjacent.push(newPos);
        }
      }
    } else {
      // 8 directions for square boards (Moore neighborhood)
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          const newPos: Position = { x: pos.x + dx, y: pos.y + dy };
          if (CoordinateUtils.isValid(newPos, boardType)) {
            adjacent.push(newPos);
          }
        }
      }
    }

    return adjacent;
  },
};
