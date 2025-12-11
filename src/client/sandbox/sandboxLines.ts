/**
 * Sandbox line-detection helpers.
 *
 * This module is a thin adapter between the client-local sandbox engine
 * (ClientSandboxEngine) and the canonical shared line-detection helpers.
 * Line geometry and victory semantics live in the shared engine; sandbox
 * code should not re-introduce bespoke line logic here.
 *
 * - Geometry and detection: src/shared/engine/aggregates/LineAggregate.ts
 * - Public adapter: findAllLinesOnBoard -> findAllLines(board)
 *
 * Do NOT add new line-detection semantics in this file. Extend the shared
 * helpers instead so backend, sandbox, and rules-layer tests remain aligned.
 *
 * Line geometry functions (getLineDirections, findLineInDirection) are now
 * exported from LineAggregate for testing purposes. Production code should
 * use findAllLines() which uses these internally.
 *
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
