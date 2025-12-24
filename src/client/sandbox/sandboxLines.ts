/**
 * @fileoverview Sandbox Line Detection Helpers - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It provides line detection for sandbox/offline games by delegating to shared helpers.
 *
 * Canonical SSoT:
 * - Line detection: `src/shared/engine/aggregates/LineAggregate.ts`
 * - Line geometry: `src/shared/engine/lineDetection.ts`
 *
 * This adapter:
 * - Wraps `findAllLines()` for backwards-compatible `findAllLinesOnBoard()` interface
 * - Extra parameters retained for test stubs, but unused - all geometry from shared engine
 *
 * DO NOT add line-detection semantics here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 * @module client/sandbox/sandboxLines
 */
import type { BoardState, BoardType, LineInfo, Position } from '../../shared/engine';
import { findAllLines as findAllLinesShared } from '../../shared/engine';

/**
 * Find all marker lines on the board for all players.
 *
 * This is a thin adapter over the canonical shared helper
 * {@link findAllLinesShared} so that sandbox code and rules/parity
 * tests can continue to depend on the historical
 * findAllLinesOnBoard(...) export without re-implementing geometry.
 *
 * The extra parameters (boardType, isValidPosition, stringToPosition)
 * are retained for backwards compatibility and for tests that stub
 * this function, but they are no longer used to derive line geometry.
 * All hosts (backend, sandbox, shared GameEngine) now share a single
 * source of truth in src/shared/engine/lineDetection.ts.
 */
export function findAllLinesOnBoard(
  boardType: BoardType,
  board: BoardState,
  isValidPosition: (pos: Position) => boolean,
  stringToPosition: (posStr: string) => Position
): LineInfo[] {
  void boardType;
  void isValidPosition;
  void stringToPosition;
  return findAllLinesShared(board);
}
