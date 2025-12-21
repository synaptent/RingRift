/**
 * Legacy RRN parsing/formatting helpers.
 *
 * These functions preserve older move-type aliases (move_ring, build_stack,
 * choose_line_reward, process_territory_region, line_formation, territory_claim)
 * for replay tooling and historical data migration only.
 */

import type { BoardType } from '../../types/game';
import type { MoveRecord, ParsedRRNMove, ParsedRRN, GameRecord } from '../../types/gameRecord';
import {
  moveRecordToRRN as canonicalMoveRecordToRRN,
  parseRRNMove as canonicalParseRRNMove,
  rrnToMoves as canonicalRrnToMoves,
  positionToRRN,
  rrnToPosition,
} from '../../types/gameRecord';
import { isLegacyMoveType } from './legacyMoveTypes';

export function legacyMoveRecordToRRN(record: MoveRecord, boardType: BoardType): string {
  if (!isLegacyMoveType(record.type)) {
    return canonicalMoveRecordToRRN(record, boardType);
  }

  const posToStr = (pos: MoveRecord['to'] | undefined): string => {
    if (!pos) return '?';
    return positionToRRN(pos, boardType);
  };

  switch (record.type) {
    case 'move_ring':
      return `${posToStr(record.from)}-${posToStr(record.to)}`;
    case 'build_stack':
      return `${posToStr(record.from)}>${posToStr(record.to)}`;
    case 'choose_line_reward':
    case 'line_formation': {
      if (record.formedLines && record.formedLines.length > 0) {
        const firstPos = record.formedLines[0].positions[0];
        return `L${posToStr(firstPos)}`;
      }
      return 'L?';
    }
    case 'process_territory_region':
    case 'territory_claim': {
      if (record.disconnectedRegions && record.disconnectedRegions.length > 0) {
        const repPos = record.disconnectedRegions[0].spaces[0];
        return `T${posToStr(repPos)}`;
      }
      return 'T?';
    }
    default:
      return `?${record.type}`;
  }
}

export function legacyParseRRNMove(notation: string, boardType: BoardType): ParsedRRNMove {
  const trimmed = notation.trim();

  if (trimmed === '-' || trimmed === 'S' || trimmed.startsWith('P')) {
    return canonicalParseRRNMove(trimmed, boardType);
  }

  if (trimmed.startsWith('L')) {
    const pos = rrnToPosition(trimmed.slice(1), boardType);
    return { moveType: 'line_formation', to: pos };
  }

  if (trimmed.startsWith('T')) {
    const pos = rrnToPosition(trimmed.slice(1), boardType);
    return { moveType: 'process_territory_region', to: pos };
  }

  if (trimmed.startsWith('E')) {
    const pos = rrnToPosition(trimmed.slice(1), boardType);
    return { moveType: 'eliminate_rings_from_stack', to: pos };
  }

  if (trimmed === 'O1' || trimmed === 'O2') {
    return { moveType: 'choose_line_reward' };
  }

  if (trimmed.includes('x')) {
    return canonicalParseRRNMove(trimmed, boardType);
  }

  if (trimmed.includes('>')) {
    const [fromStr, toStr] = trimmed.split('>');
    const fromPos = rrnToPosition(fromStr, boardType);
    const toPos = rrnToPosition(toStr, boardType);
    return { moveType: 'build_stack', from: fromPos, to: toPos };
  }

  if (trimmed.includes('-')) {
    const [fromStr, toStr] = trimmed.split('-');
    const fromPos = rrnToPosition(fromStr, boardType);
    const toPos = rrnToPosition(toStr, boardType);
    return { moveType: 'move_ring', from: fromPos, to: toPos };
  }

  throw new Error(`Unable to parse legacy RRN: ${notation}`);
}

export function legacyRrnToMoves(rrnString: string): ParsedRRN {
  const parts = rrnString.split(':');
  if (parts.length < 4) {
    throw new Error(`Invalid RRN format: ${rrnString}`);
  }

  const boardType = parts[0] as BoardType;
  const numPlayers = parseInt(parts[1], 10);
  const rngSeedStr = parts[2];
  const movesStr = parts.slice(3).join(':');

  const rngSeed = rngSeedStr === '_' ? undefined : parseInt(rngSeedStr, 10);
  const moves = movesStr
    .split(' ')
    .filter((token) => token.trim().length > 0)
    .map((moveNotation) => legacyParseRRNMove(moveNotation, boardType));

  return {
    boardType,
    numPlayers,
    ...(rngSeed !== undefined && { rngSeed }),
    moves,
  };
}

export function legacyGameRecordToRRN(record: GameRecord): string {
  const headerParts = [record.boardType, String(record.numPlayers)];
  if (record.rngSeed !== undefined) {
    headerParts.push(String(record.rngSeed));
  } else {
    headerParts.push('_');
  }

  const moves = record.moves.map((move) => legacyMoveRecordToRRN(move, record.boardType));
  return `${headerParts.join(':')}:${moves.join(' ')}`;
}

export function legacyRrnToCanonicalMoves(rrnString: string): ParsedRRN {
  return canonicalRrnToMoves(rrnString);
}
