import type { GameState, Move, Position } from '../types/game';
import { positionToString, stringToPosition } from '../types/game';
import { validateCaptureSegmentOnBoard, type CaptureSegmentBoardView } from './core';
import { isValidPosition } from './validators/utils';
import type {
  ChainCaptureStateSnapshot as AggregateChainSnapshot,
  ChainCaptureEnumerationOptions as AggregateChainOptions,
  CaptureBoardAdapters as AggregateCaptureBoardAdapters,
} from './aggregates/CaptureAggregate';
import {
  enumerateChainCaptureSegments as enumerateChainCaptureSegmentsAggregate,
  enumerateCaptureMoves as enumerateCaptureMovesAggregate,
} from './aggregates/CaptureAggregate';

/**
 * Shared primitives for capture-chain orchestration (legacy shim).
 *
 * This module now delegates all capture-segment geometry and chain
 * enumeration to the canonical implementation in
 * {@link aggregates/CaptureAggregate.ts} while preserving the older helper
 * surface used by tests and tooling:
 *
 * - {@link enumerateChainCaptureSegments}
 * - {@link getChainCaptureContinuationInfo}
 * - {@link canCapture}
 * - {@link getValidCaptureTargets}
 * - {@link processChainCapture}
 *
 * The goal for P1.2 is to ensure there is a single algorithmic source of
 * truth for capture logic while keeping these helpers available as a thin,
 * well-documented compatibility layer.
 */

/**
 * Lightweight snapshot of chain-capture state used by legacy helpers.
 *
 * NOTE: The canonical aggregate snapshot uses {@code capturedThisChain:
 * Position[]}. This shim maps {@link visitedTargets} to that field and does
 * not own any independent enumeration logic.
 */
export interface ChainCaptureStateSnapshot {
  /** Player performing the capture chain. */
  player: number;

  /**
   * Position of the capturing stack at the start of the **next** segment.
   * For the first segment this is the origin of the initial capture; for
   * later segments it is the previous landing position.
   */
  currentPosition: Position;

  /**
   * Optional set of capture-target positions (stringified via
   * {@link positionToString}) that have already been used in this chain.
   * When combined with {@link ChainCaptureEnumerationOptions.disallowRevisitedTargets}
   * this allows callers to request "no repeated target" enumeration without
   * re-encoding that policy locally.
   */
  visitedTargets?: string[];
}

/**
 * Configuration for capture-chain segment enumeration.
 *
 * This mirrors the canonical {@link AggregateChainOptions} type but keeps
 * the older name for compatibility.
 */
export interface ChainCaptureEnumerationOptions {
  /** See {@link AggregateChainOptions.disallowRevisitedTargets}. */
  disallowRevisitedTargets?: boolean;
  /** See {@link AggregateChainOptions.moveNumber}. */
  moveNumber?: number;
  /** See {@link AggregateChainOptions.kind}. */
  kind?: 'initial' | 'continuation';
}

/**
 * Result of asking whether a capture chain may or must continue from the
 * current position.
 *
 * Canonical aggregate naming uses {@code mustContinue} +
 * {@code availableContinuations}; this shim preserves the older
 * {@code hasFurtherCaptures} / {@code segments} naming used by tests.
 */
export interface ChainCaptureContinuationInfo {
  /** True when at least one legal capture segment exists from currentPosition. */
  hasFurtherCaptures: boolean;
  /** Concrete {@link Move} values describing each legal next segment. */
  segments: Move[];
}

// ---------------------------------------------------------------------------
// Internal mapping helpers
// ---------------------------------------------------------------------------

function toAggregateSnapshot(snapshot: ChainCaptureStateSnapshot): AggregateChainSnapshot {
  const capturedThisChain = snapshot.visitedTargets?.map((s) => stringToPosition(s)) ?? [];

  return {
    player: snapshot.player,
    currentPosition: snapshot.currentPosition,
    capturedThisChain,
  };
}

function toAggregateOptions(
  options?: ChainCaptureEnumerationOptions
): AggregateChainOptions | undefined {
  if (!options) return undefined;

  const result: AggregateChainOptions = {};

  if (options.disallowRevisitedTargets !== undefined) {
    result.disallowRevisitedTargets = options.disallowRevisitedTargets;
  }
  if (options.moveNumber !== undefined) {
    result.moveNumber = options.moveNumber;
  }
  if (options.kind !== undefined) {
    result.kind = options.kind;
  }

  return result;
}

function createCaptureBoardAdaptersFromState(state: GameState): AggregateCaptureBoardAdapters {
  const board = state.board;
  const boardType = state.board.type;
  const size = board.size;

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

function createCaptureSegmentViewFromState(state: GameState): CaptureSegmentBoardView {
  const board = state.board;
  const boardType = state.board.type;
  const size = board.size;

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const marker = board.markers.get(positionToString(pos));
      return marker?.player;
    },
  };
}

// ---------------------------------------------------------------------------
// Canonical delegation helpers
// ---------------------------------------------------------------------------

/**
 * Enumerate all legal capture-chain segments for the specified player and
 * starting position on the given {@link GameState}.
 *
 * This is now a thin wrapper over
 * {@link aggregates/CaptureAggregate.enumerateChainCaptureSegments}, with
 * a small adapter from {@link ChainCaptureStateSnapshot} to the canonical
 * aggregate snapshot.
 */
export function enumerateChainCaptureSegments(
  state: GameState,
  snapshot: ChainCaptureStateSnapshot,
  options?: ChainCaptureEnumerationOptions
): Move[] {
  const aggregateSnapshot = toAggregateSnapshot(snapshot);
  const aggregateOptions = toAggregateOptions(options);

  return enumerateChainCaptureSegmentsAggregate(state, aggregateSnapshot, aggregateOptions);
}

/**
 * Convenience wrapper that answers "must this chain continue?" in a single
 * call, returning both the boolean and the concrete segment list.
 *
 * Hosts are expected to:
 * - call this helper after applying each capture segment, and
 * - transition to the `chain_capture` phase when
 *   {@link ChainCaptureContinuationInfo.hasFurtherCaptures} is true, or
 *   proceed to line processing otherwise.
 */
export function getChainCaptureContinuationInfo(
  state: GameState,
  snapshot: ChainCaptureStateSnapshot,
  options?: ChainCaptureEnumerationOptions
): ChainCaptureContinuationInfo {
  const aggregateSnapshot = toAggregateSnapshot(snapshot);
  const aggregateOptions = toAggregateOptions(options);

  const segments = enumerateChainCaptureSegmentsAggregate(
    state,
    aggregateSnapshot,
    aggregateOptions
  );

  return {
    hasFurtherCaptures: segments.length > 0,
    segments,
  };
}

/**
 * Check if a capture is valid from a given position to a target and landing.
 *
 * This is a thin wrapper around {@link validateCaptureSegmentOnBoard} for
 * convenience. It does not own any capture geometry logic.
 */
export function canCapture(
  state: GameState,
  from: Position,
  target: Position,
  landing: Position,
  player: number
): boolean {
  const view = createCaptureSegmentViewFromState(state);

  return validateCaptureSegmentOnBoard(state.boardType, from, target, landing, player, view);
}

/**
 * Get all valid capture targets from a given position for a player.
 *
 * Returns an array of objects containing the target position and all valid
 * landing positions for each target. Enumeration is delegated entirely to
 * {@link CaptureAggregate.enumerateCaptureMoves}.
 */
export function getValidCaptureTargets(
  state: GameState,
  from: Position,
  player: number
): Array<{ target: Position; landings: Position[] }> {
  const adapters = createCaptureBoardAdaptersFromState(state);
  const moveNumber = state.moveHistory.length + 1;

  const moves = enumerateCaptureMovesAggregate(state.boardType, from, player, adapters, moveNumber);

  // Group moves by captureTarget
  const byTarget = new Map<string, { target: Position; landings: Position[] }>();

  for (const move of moves) {
    if (!move.captureTarget || !move.to) continue;

    const key = positionToString(move.captureTarget);
    let entry = byTarget.get(key);
    if (!entry) {
      entry = { target: move.captureTarget, landings: [] };
      byTarget.set(key, entry);
    }
    entry.landings.push(move.to);
  }

  return Array.from(byTarget.values());
}

/**
 * Process a full capture chain starting from an initial capture move in a
 * purely analytical way (no mutation). This helper is primarily useful for
 * UI visualisation and AI decision making.
 *
 * NOTE:
 * - This helper re-validates the initial segment via {@link canCapture}.
 * - Continuation options are computed using the canonical chain enumerator
 *   on the *unmutated* state, so
 *   {@link ChainCaptureContinuationInfo.hasFurtherCaptures} is an
 *   approximation. Hosts that need exact continuation sets should apply the
 *   capture to a cloned state and then call the aggregate helpers directly.
 */
export function processChainCapture(
  state: GameState,
  initialFrom: Position,
  initialTarget: Position,
  initialLanding: Position,
  player: number
): {
  isValid: boolean;
  hasContinuation: boolean;
  continuationOptions: Move[];
} {
  if (!canCapture(state, initialFrom, initialTarget, initialLanding, player)) {
    return {
      isValid: false,
      hasContinuation: false,
      continuationOptions: [],
    };
  }

  const snapshot: ChainCaptureStateSnapshot = {
    player,
    currentPosition: initialLanding,
    visitedTargets: [positionToString(initialTarget)],
  };

  const info = getChainCaptureContinuationInfo(state, snapshot, {
    disallowRevisitedTargets: true,
    kind: 'continuation',
  });

  return {
    isValid: true,
    hasContinuation: info.hasFurtherCaptures,
    continuationOptions: info.segments,
  };
}
