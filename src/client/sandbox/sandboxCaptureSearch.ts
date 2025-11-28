import type { BoardState, BoardType, Position, RingStack } from '../../shared/engine';
import { positionToString } from '../../shared/engine';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters,
  CaptureSegment,
} from './sandboxCaptures';

/**
 * A single segment in an overtaking capture chain.
 *
 * This mirrors both the Rust `CaptureRecord` triple (from/target/landing)
 * and the existing `CaptureSegment` used by sandboxCaptures.
 */
export interface ChainSegment extends CaptureSegment {}

/**
 * Result of a full chain search from a given starting stack.
 */
export interface ChainSearchResult {
  segments: ChainSegment[];
  finalPosition: Position;
  finalHeight: number;
}

export interface ChainSearchOptions {
  /** Safety cap on the number of segments in a single chain. */
  maxDepth?: number;
  /**
   * If true, avoid revisiting attacker landing positions within a single chain
   * (simple cycle control). This mirrors the visited-position heuristics used
   * in the Rust engine and the Node cyclic-capture search scripts.
   */
  pruneVisitedPositions?: boolean;
}

interface SearchState {
  board: BoardState;
  currentPos: Position;
  segments: ChainSegment[];
  visited: Set<string>; // attacker landing squares (positionToString)
}

/**
 * Shallow-ish clone of a BoardState suitable for sandbox search.
 *
 * This follows the same pattern used on the server side (RuleEngine/GameEngine):
 * we keep the arrays/Maps structurally distinct while sharing inner objects
 * where safe. For search/diagnostic use this is sufficient and much cheaper
 * than a deep structural copy of every stack/space.
 */
function cloneBoard(board: BoardState): BoardState {
  return {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    territories: new Map(board.territories),
    collapsedSpaces: new Map(board.collapsedSpaces),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings },
  };
}

/**
 * Enumerate all legal overtaking capture chains starting from the given
 * stack position and return those with maximal segment count.
 *
 * This mirrors the structure of the Rust-side chain search logic
 * (e.g. `simulate_best_post_move_chain`) but is built entirely on the
 * sandbox capture helpers and the shared, pure validator
 * `validateCaptureSegmentOnBoard`.
 *
 * IMPORTANT:
 * - This search is **analysis / debug only** and is not a rules SSOT.
 * - It MUST NOT be used to drive user-facing legality surfaces such as
 *   getValidMoves, ClientSandboxEngine.handleMovementClick, or AI move
 *   validation.
 * - All rules-level capture enumeration and application must instead go
 *   through the shared capture aggregate
 *   (CaptureAggregate.enumerateCaptureMoves / applyCaptureSegment /
 *   applyCapture) or its shims.
 *
 * Callers should treat this helper as a tooling primitive for exploring
 * potential chains under additional heuristics (e.g. pruneVisitedPositions),
 * not as an authority on what the current player is required or allowed
 * to play.
 */
export function findMaxCaptureChains(
  boardType: BoardType,
  initialBoard: BoardState,
  startPos: Position,
  playerNumber: number,
  adapters: CaptureBoardAdapters & CaptureApplyAdapters,
  options: ChainSearchOptions = {}
): ChainSearchResult[] {
  const maxDepth = options.maxDepth ?? 32; // defensive default

  const results: ChainSearchResult[] = [];
  let bestLength = 0;

  const startKey = positionToString(startPos);

  const initialStack = initialBoard.stacks.get(startKey) as RingStack | undefined;
  if (!initialStack || initialStack.stackHeight <= 0) {
    // No attacker at the requested start position; no chains exist.
    return [];
  }

  function dfs(state: SearchState): void {
    const length = state.segments.length;

    if (length > bestLength) {
      bestLength = length;
      results.length = 0; // reset, we found a new maximum
    }

    if (length === bestLength && length > 0) {
      const finalKey = positionToString(state.currentPos);
      const finalStack = state.board.stacks.get(finalKey) as RingStack | undefined;
      if (finalStack && finalStack.stackHeight > 0) {
        results.push({
          segments: [...state.segments],
          finalPosition: state.currentPos,
          finalHeight: finalStack.stackHeight,
        });
      }
    }

    if (length >= maxDepth) {
      return; // reached safety cap
    }

    const segments = enumerateCaptureSegmentsFromBoard(
      boardType,
      state.board,
      state.currentPos,
      playerNumber,
      adapters
    );

    if (segments.length === 0) {
      // Chain terminates naturally at this node.
      return;
    }

    for (const seg of segments) {
      const landingKey = positionToString(seg.landing);

      if (options.pruneVisitedPositions && state.visited.has(landingKey)) {
        // Simple cycle-avoidance: do not revisit prior landing squares.
        continue;
      }

      const nextBoard = cloneBoard(state.board);
      applyCaptureSegmentOnBoard(
        nextBoard,
        seg.from,
        seg.target,
        seg.landing,
        playerNumber,
        adapters
      );

      const nextVisited = new Set(state.visited);
      nextVisited.add(landingKey);

      dfs({
        board: nextBoard,
        currentPos: seg.landing,
        segments: [...state.segments, seg],
        visited: nextVisited,
      });
    }
  }

  dfs({
    board: cloneBoard(initialBoard),
    currentPos: startPos,
    segments: [],
    visited: new Set([startKey]),
  });

  return results;
}
