import type { GameState, Position } from '../types/game';

import type {
  SimpleMovementParams as AggregateSimpleMovementParams,
  MovementApplicationOutcome as AggregateMovementApplicationOutcome,
} from './aggregates/MovementAggregate';
import { applySimpleMovement as applySimpleMovementAggregate } from './aggregates/MovementAggregate';

import type {
  CaptureSegmentParams as AggregateCaptureSegmentParams,
  CaptureApplicationOutcome as AggregateCaptureApplicationOutcome,
} from './aggregates/CaptureAggregate';
import { applyCaptureSegment as applyCaptureSegmentAggregate } from './aggregates/CaptureAggregate';

/**
 * Compatibility wrappers for applying already-validated movement and capture
 * steps to a GameState.
 *
 * Historically this module contained its own movement/capture mutation logic
 * that duplicated the backend GameEngine and sandbox engines. As part of the
 * shared-engine consolidation (P2.2), those semantics now live exclusively in
 * the MovementAggregate / CaptureAggregate modules. The helpers exported here
 * are thin adapters over those aggregates so existing tests and callers that
 * reference movementApplication continue to compile while all rules behaviour
 * is single-sourced.
 *
 * New code should prefer the aggregate APIs directly:
 * - Movement: MovementAggregate.applySimpleMovement / applyMovement
 * - Capture: CaptureAggregate.applyCaptureSegment / applyCapture
 */

/**
 * Parameters for applying a simple (non-capturing) movement.
 *
 * This mirrors the historic movementApplication.SimpleMovementParams type
 * but is implemented via MovementAggregate.SimpleMovementParams under the hood.
 */
export interface SimpleMovementParams {
  /** Origin of the moving stack. Must currently contain a stack. */
  from: Position;
  /** Landing position for the non-capturing move. */
  to: Position;
  /** Numeric player index expected to control the moving stack. */
  player: number;
  /**
   * When false, suppresses the default behaviour of leaving a departure
   * marker at `from`. This is mainly useful for hypothetical simulations
   * that want to reuse marker semantics but defer marker placement.
   *
   * Default: true.
   */
  leaveDepartureMarker?: boolean;
}

/**
 * Parameters for applying a single overtaking capture segment.
 *
 * This mirrors the historic movementApplication.CaptureSegmentParams type but
 * delegates to the canonical CaptureAggregate.applyCaptureSegment helper.
 */
export interface CaptureSegmentParams {
  /** Origin of the capturing stack before the segment. */
  from: Position;
  /** Position of the overtaken stack for this segment. */
  target: Position;
  /** Landing position after the segment. */
  landing: Position;
  /** Numeric player index expected to control the attacking stack. */
  player: number;
}

/**
 * Result of applying a movement or capture mutation.
 *
 * This is a narrowed view of the aggregate outcomes that preserves the
 * original movementApplication API surface used by existing tests.
 */
export interface MovementApplicationOutcome {
  /**
   * Next GameState after applying the movement or capture, including all stack,
   * marker, collapsed-space, and elimination bookkeeping.
   */
  nextState: GameState;

  /**
   * Optional per-player elimination deltas caused directly by this operation.
   * For simple non-capturing movement this is non-empty only when the move
   * lands on the mover's own marker and triggers top-ring elimination.
   */
  eliminatedRingsByPlayer?: { [player: number]: number };
}

/**
 * Apply a non-capturing stack movement (canonical `move_stack`)
 * to a GameState by delegating to MovementAggregate.applySimpleMovement.
 *
 * Callers are expected to have validated legality separately using the shared
 * validators; this helper focuses on board mutation and elimination accounting.
 */
export function applySimpleMovement(
  state: GameState,
  params: SimpleMovementParams
): MovementApplicationOutcome {
  const aggregateParams: AggregateSimpleMovementParams =
    params.leaveDepartureMarker === undefined
      ? {
          from: params.from,
          to: params.to,
          player: params.player,
        }
      : {
          from: params.from,
          to: params.to,
          player: params.player,
          leaveDepartureMarker: params.leaveDepartureMarker,
        };

  const outcome: AggregateMovementApplicationOutcome = applySimpleMovementAggregate(
    state,
    aggregateParams
  );

  return {
    nextState: outcome.nextState,
    ...(outcome.eliminatedRingsByPlayer && {
      eliminatedRingsByPlayer: outcome.eliminatedRingsByPlayer,
    }),
  };
}

/**
 * Apply a single overtaking capture segment to a GameState by delegating to
 * CaptureAggregate.applyCaptureSegment.
 *
 * Hosts that need full chain-capture semantics should prefer the higher-level
 * CaptureAggregate helpers; this function is a one-segment convenience used by
 * legacy tests.
 */
export function applyCaptureSegment(
  state: GameState,
  params: CaptureSegmentParams
): MovementApplicationOutcome {
  const aggregateParams: AggregateCaptureSegmentParams = {
    from: params.from,
    target: params.target,
    landing: params.landing,
    player: params.player,
  };

  // Snapshot elimination counts before applying the capture so we can
  // reconstruct per-player deltas for the compatibility return type.
  const beforeEliminated = { ...state.board.eliminatedRings };

  const outcome: AggregateCaptureApplicationOutcome = applyCaptureSegmentAggregate(
    state,
    aggregateParams
  );

  const afterEliminated = outcome.nextState.board.eliminatedRings;
  const eliminatedRingsByPlayer: { [player: number]: number } = {};

  for (const key of new Set([...Object.keys(beforeEliminated), ...Object.keys(afterEliminated)])) {
    const player = Number(key);
    const before = beforeEliminated[player] ?? 0;
    const after = afterEliminated[player] ?? 0;
    const delta = after - before;
    if (delta > 0) {
      eliminatedRingsByPlayer[player] = delta;
    }
  }

  return Object.keys(eliminatedRingsByPlayer).length > 0
    ? { nextState: outcome.nextState, eliminatedRingsByPlayer }
    : { nextState: outcome.nextState };
}
