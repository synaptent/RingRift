import { Move, Position, positionToString } from '../../src/shared/types/game';

/**
 * Shared helpers for comparing and matching canonical Move objects
 * across engines/tests.
 */

export function positionsEqual(a?: Position, b?: Position): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.x === b.x && a.y === b.y && (a.z ?? 0) === (b.z ?? 0);
}

/**
 * Loosely compare two Moves for equivalence in parity/debug contexts.
 *
 * The goal is to treat semantically identical actions as equal even
 * when minor metadata (e.g. placementCount, move_ring vs move_stack)
 * differs between engines.
 */
export function movesLooselyMatch(a: Move, b: Move): boolean {
  if (a.player !== b.player) return false;

  // Treat simple non-capture movements as equivalent whether they are
  // labelled move_ring (legacy) or move_stack (canonical), as long as
  // from/to match.
  const isSimpleMovementPair =
    (a.type === 'move_ring' && b.type === 'move_stack') ||
    (a.type === 'move_stack' && b.type === 'move_ring') ||
    (a.type === 'move_ring' && b.type === 'move_ring') ||
    (a.type === 'move_stack' && b.type === 'move_stack');

  if (isSimpleMovementPair) {
    return positionsEqual(a.from, b.from) && positionsEqual(a.to, b.to);
  }

  if (a.type !== b.type) return false;

  // For placement moves, require same destination and the same
  // placementCount. Earlier we ignored placementCount, but for
  // trace-parity we need backend placements to mirror the sandbox
  // multi-ring counts so hashes and ring inventories stay aligned.
  if (a.type === 'place_ring') {
    const aCount = a.placementCount ?? 1;
    const bCount = b.placementCount ?? 1;
    return positionsEqual(a.to, b.to) && aCount === bCount;
  }

  // For overtaking captures, require from, captureTarget, and landing
  // to match.
  if (a.type === 'overtaking_capture') {
    return (
      positionsEqual(a.from, b.from) &&
      positionsEqual(a.captureTarget, b.captureTarget) &&
      positionsEqual(a.to, b.to)
    );
  }

  // For other move types (build_stack, etc.) we require exact type
  // match and strict position equality when applicable.
  if (a.from || b.from) {
    if (!positionsEqual(a.from, b.from)) return false;
  }
  if (a.to || b.to) {
    if (!positionsEqual(a.to, b.to)) return false;
  }

  return true;
}

/**
 * Find a Move in `candidates` that loosely matches the `reference`
 * move according to movesLooselyMatch.
 */
export function findMatchingBackendMove(reference: Move, candidates: Move[]): Move | null {
  for (const candidate of candidates) {
    if (movesLooselyMatch(reference, candidate)) {
      return candidate;
    }
  }
  return null;
}

/**
 * Human-friendly one-line description of a Move for debug logs.
 */
export function describeMoveForLog(move: Move): string {
  const parts: string[] = [];
  parts.push(`type=${move.type}`);
  parts.push(`player=${move.player}`);
  if (move.from) {
    parts.push(`from=${positionToString(move.from)}`);
  }
  if (move.to) {
    parts.push(`to=${positionToString(move.to)}`);
  }
  if (move.captureTarget) {
    parts.push(`captureTarget=${positionToString(move.captureTarget)}`);
  }
  if (typeof move.placementCount === 'number') {
    parts.push(`placementCount=${move.placementCount}`);
  }
  return parts.join(',');
}

export function describeMovesListForLog(moves: Move[]): string {
  if (!moves.length) return '(none)';
  return moves.map(describeMoveForLog).join(' | ');
}
