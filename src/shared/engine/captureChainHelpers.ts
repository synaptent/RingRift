import type { GameState, Move, Position } from '../types/game';

/**
 * Shared primitives for capture-chain orchestration.
 *
 * These helpers provide a host-agnostic surface that both the backend
 * [`captureChainEngine`](src/server/game/rules/captureChainEngine.ts:1) and the
 * client sandbox chain-capture logic can target instead of re-encoding the
 * rules around:
 *
 * - which follow-up capture segments are legal from a given landing position, and
 * - when a chain must continue vs may terminate.
 *
 * The functions here intentionally do **not** own any long-lived chain state;
 * they operate on a lightweight snapshot that each host is free to extend with
 * UI / history metadata. This mirrors the existing separation between:
 *
 * - backend: `TsChainCaptureState` + `getCaptureOptionsFromPosition`,
 * - sandbox: `performCaptureChainSandbox` +
 *   `enumerateCaptureSegmentsFromSandbox`.
 */
export interface ChainCaptureStateSnapshot {
  /**
   * Player performing the capture chain.
   *
   * This must match {@link GameState.currentPlayer} in normal engine flows;
   * the helpers do not mutate turn ownership.
   */
  player: number;

  /**
   * Position of the capturing stack at the start of the **next** segment.
   * For the first segment this is the origin of the initial overtaking
   * capture; for later segments it is the landing position of the most
   * recent segment.
   */
  currentPosition: Position;

  /**
   * Optional set of capture-target positions (stringified via
   * {@link positionToString}) that have already been used in this chain.
   * When combined with {@link ChainCaptureEnumerationOptions.disallowRevisitedTargets}
   * this allows hosts to enforce "no immediate backtracking" or
   * "no repeated target" semantics without re-deriving them locally.
   */
  visitedTargets?: string[];
}

/**
 * Configuration for capture-chain segment enumeration.
 */
export interface ChainCaptureEnumerationOptions {
  /**
   * When true, filter out any candidate segment whose captureTarget is
   * present in {@link ChainCaptureStateSnapshot.visitedTargets}. This
   * matches the semantics of the backend
   * [`getCaptureOptionsFromPosition`](src/server/game/rules/captureChainEngine.ts:1)
   * helper.
   *
   * Default: false.
   */
  disallowRevisitedTargets?: boolean;

  /**
   * Move number to embed in the generated {@link Move} instances. When not
   * provided, callers are expected to patch `moveNumber` themselves before
   * appending the moves to history.
   */
  moveNumber?: number;

  /**
   * How generated moves should be typed:
   *
   * - 'initial'      – `type: 'overtaking_capture'`
   * - 'continuation' – `type: 'continue_capture_segment'`
   *
   * Hosts typically use 'initial' for the first segment of a chain and
   * 'continuation' for all subsequent segments.
   *
   * Default: 'continuation'.
   */
  kind?: 'initial' | 'continuation';
}

/**
 * Result of asking the shared engine whether a capture chain may or must
 * continue from the current position.
 */
export interface ChainCaptureContinuationInfo {
  /**
   * True when at least one legal capture segment exists from the current
   * position. Under the standard rules this implies that the chain **must**
   * continue; when false, the chain terminates and bookkeeping proceeds to
   * line processing.
   */
  hasFurtherCaptures: boolean;

  /**
   * Concrete `Move` instances describing each legal next segment, suitable
   * for inclusion in `getValidMoves` during the `chain_capture` phase.
   *
   * These moves are not applied by this helper; hosts call
   * [`applyCaptureSegment`](src/shared/engine/movementApplication.ts:1) (or
   * their local equivalent during refactor) to apply the chosen segment.
   */
  segments: Move[];
}

/**
 * Enumerate all legal capture-chain segments for the specified player and
 * starting position on the given GameState.
 *
 * Responsibilities:
 *
 * - Delegate the geometric legality checks to the shared
 *   [`enumerateCaptureMoves`](src/shared/engine/captureLogic.ts:1) helper.
 * - Optionally filter out segments that revisit an already-captured target
 *   according to {@link ChainCaptureEnumerationOptions.disallowRevisitedTargets}.
 * - Normalise the `Move.type` field to either:
 *   - 'overtaking_capture'   (kind === 'initial'), or
 *   - 'continue_capture_segment' (kind === 'continuation').
 *
 * Error handling:
 *
 * - This helper assumes the provided {@link GameState} satisfies the usual
 *   board invariants (stack/marker/territory exclusivity).
 * - It is allowed to throw if structural invariants are violated, but is
 *   expected to be total for states produced by the canonical engines.
 *
 * NOTE: For P0 Task #21 this function is specified and documented but left
 * as a design-time stub; backend and sandbox hosts are **not** yet wired to
 * call it. Later tasks will port existing host logic into its body.
 */
export function enumerateChainCaptureSegments(
  state: GameState,
  snapshot: ChainCaptureStateSnapshot,
  options?: ChainCaptureEnumerationOptions
): Move[] {
  throw new Error(
    'TODO(P0-HELPERS): enumerateChainCaptureSegments is a design-time stub. ' +
      'See P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md for the intended semantics.'
  );
}

/**
 * Convenience wrapper that answers "must this chain continue?" in a single
 * call, returning both the boolean and the concrete segment list.
 *
 * This mirrors the combined usage pattern of:
 *
 * - backend: `updateChainCaptureStateAfterCapture` +
 *   `getCaptureOptionsFromPosition`, and
 * - sandbox: `enumerateCaptureSegmentsFromSandbox` +
 *   `performCaptureChainSandbox`.
 *
 * Hosts are expected to:
 *
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
  const segments = enumerateChainCaptureSegments(state, snapshot, options);
  return {
    hasFurtherCaptures: segments.length > 0,
    segments,
  };
}
