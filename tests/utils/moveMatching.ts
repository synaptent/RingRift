import { Move, Position, positionToString } from '../../src/shared/types/game';

/**
 * Shared helpers for comparing and matching canonical Move objects
 * across engines/tests.
 */

const TRACE_DEBUG_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

export function positionsEqual(a?: Position, b?: Position): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return a.x === b.x && a.y === b.y && (a.z ?? 0) === (b.z ?? 0);
}

/**
 * Loosely compare two Moves for equivalence in parity/debug contexts.
 *
 * The goal is to treat semantically identical actions as equal even
 * when minor metadata (e.g. placementCount, move_stack vs move_stack)
 * differs between engines.
 */
export function movesLooselyMatch(a: Move, b: Move): boolean {
  if (a.player !== b.player) return false;

  // Treat simple non-capture movements as equivalent whether they are
  // labelled move_stack (legacy) or move_stack (canonical), as long as
  // from/to match.
  const isSimpleMovementPair =
    (a.type === 'move_stack' && b.type === 'move_stack') ||
    (a.type === 'move_stack' && b.type === 'move_stack') ||
    (a.type === 'move_stack' && b.type === 'move_stack') ||
    (a.type === 'move_stack' && b.type === 'move_stack');

  if (isSimpleMovementPair) {
    return positionsEqual(a.from, b.from) && positionsEqual(a.to, b.to);
  }

  // When running under the seed/trace debug harness, treat a sandbox
  // overtaking_capture as loosely equivalent to a backend move_stack
  // with the same from/to landing geometry. This allows replay
  // harnesses to progress far enough to surface **geometric** parity
  // mismatches (stacks/markers/collapsedSpaces) even when the backend
  // fails to classify a move as a capture. Outside of trace-debug
  // runs, we continue to require strict type equality so capture vs
  // non-capture divergences remain visible to CI.
  if (TRACE_DEBUG_ENABLED) {
    const isCaptureVsMoveStackPair =
      (a.type === 'overtaking_capture' && b.type === 'move_stack') ||
      (a.type === 'move_stack' && b.type === 'overtaking_capture');

    if (isCaptureVsMoveStackPair) {
      return positionsEqual(a.from, b.from) && positionsEqual(a.to, b.to);
    }
  }

  if (a.type !== b.type) return false;

  // For line-processing decisions, require that the underlying line
  // geometry (and, when present, collapsedMarkers for reward choices)
  // matches across engines. This prevents trace replays from pairing a
  // sandbox process_line / choose_line_option Move with a backend
  // candidate that targets a different line, which would desynchronise
  // stack heights and phases even when types/players match.
  if (a.type === 'process_line' || a.type === 'choose_line_option') {
    const aLine = a.formedLines && a.formedLines[0];
    const bLine = b.formedLines && b.formedLines[0];

    const aSpaces = aLine?.positions ?? [];
    const bSpaces = bLine?.positions ?? [];

    if (aSpaces.length && bSpaces.length) {
      if (aSpaces.length !== bSpaces.length) {
        return false;
      }

      const aKeys = new Set(aSpaces.map(positionToString));
      const bKeys = new Set(bSpaces.map(positionToString));

      if (aKeys.size !== bKeys.size) {
        return false;
      }

      for (const key of aKeys) {
        if (!bKeys.has(key)) {
          return false;
        }
      }

      if (a.type === 'choose_line_option') {
        const aCollapsed = a.collapsedMarkers ?? [];
        const bCollapsed = b.collapsedMarkers ?? [];

        if (aCollapsed.length || bCollapsed.length) {
          if (aCollapsed.length !== bCollapsed.length) {
            return false;
          }

          const aCollapsedKeys = new Set(aCollapsed.map(positionToString));
          const bCollapsedKeys = new Set(bCollapsed.map(positionToString));

          if (aCollapsedKeys.size !== bCollapsedKeys.size) {
            return false;
          }

          for (const key of aCollapsedKeys) {
            if (!bCollapsedKeys.has(key)) {
              return false;
            }
          }
        }
      }

      return true;
    }
    // If either side lacks line metadata, fall through to the generic
    // positional checks below so legacy traces that do not encode
    // formedLines continue to use the original loose type+player
    // matching semantics.
  }

  // For territory-processing decisions, require that the disconnected
  // region being processed matches exactly (up to set equality of
  // spaces) when both moves carry region metadata. This prevents the
  // trace replay harness from treating *any* choose_territory_option
  // Move for the same player as equivalent and instead ensures we
  // select the backend candidate whose region geometry matches the
  // sandbox trace.
  if (a.type === 'choose_territory_option') {
    const aRegion = a.disconnectedRegions && a.disconnectedRegions[0];
    const bRegion = b.disconnectedRegions && b.disconnectedRegions[0];

    const aSpaces = aRegion?.spaces ?? [];
    const bSpaces = bRegion?.spaces ?? [];

    if (aSpaces.length && bSpaces.length) {
      if (aSpaces.length !== bSpaces.length) {
        return false;
      }

      const aKeys = new Set(aSpaces.map(positionToString));
      const bKeys = new Set(bSpaces.map(positionToString));

      if (aKeys.size !== bKeys.size) {
        return false;
      }

      for (const key of aKeys) {
        if (!bKeys.has(key)) {
          return false;
        }
      }

      return true;
    }
    // If either side lacks region-space metadata, fall through to the
    // generic positional checks below so legacy traces that do not
    // encode disconnectedRegions continue to use the original loose
    // type+player matching semantics.
  }

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

  // For other move types (including legacy aliases), require exact type
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
