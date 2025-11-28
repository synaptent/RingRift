import type { BoardType, Position, Move } from '../types/game';

import type { CaptureBoardAdapters as AggregateCaptureBoardAdapters } from './aggregates/CaptureAggregate';
import { enumerateCaptureMoves as enumerateCaptureMovesAggregate } from './aggregates/CaptureAggregate';

/**
 * Legacy wrapper for capture enumeration that delegates to the canonical
 * implementation in {@link CaptureAggregate.enumerateCaptureMoves}.
 *
 * This module preserves the original public API surface
 * (`CaptureBoardAdapters`, `enumerateCaptureMoves`) while ensuring that
 * all callers share a single algorithmic source of truth for capture
 * segment enumeration.
 */
export type CaptureBoardAdapters = AggregateCaptureBoardAdapters;

/**
 * Enumerate all legal overtaking capture segments for the given player from
 * the specified stack position. This is a thin adapter that forwards to
 * {@link CaptureAggregate.enumerateCaptureMoves}.
 */
export function enumerateCaptureMoves(
  boardType: BoardType,
  from: Position,
  playerNumber: number,
  adapters: CaptureBoardAdapters,
  moveNumber: number
): Move[] {
  return enumerateCaptureMovesAggregate(boardType, from, playerNumber, adapters, moveNumber);
}
