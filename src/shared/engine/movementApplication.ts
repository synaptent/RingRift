import type { GameState, Position } from '../types/game';

/**
 * Shared helpers for applying already-validated movement and capture steps
 * to a GameState.
 *
 * These helpers are deliberately higher-level than the geometry/validation
 * layer in core.ts / validators and encode the board-mutation semantics that
 * are currently duplicated between:
 *
 * - Backend GameEngine.applyMove (movement/capture branches).
 * - sandboxMovementEngine.handleMovementClickSandbox and related helpers.
 * - sandboxCaptures.applyCaptureSegmentOnBoard.
 *
 * The functions exported here are not yet wired into either host; they exist
 * as the canonical target APIs for later refactors in P0 Tasks #7–#9.
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

export interface MovementApplicationOutcome {
  /**
   * Next GameState after applying the movement or capture, including all stack,
   * marker, collapsed-space, and elimination bookkeeping.
   *
   * Implementations must treat the input state as immutable and return a
   * shallow clone with cloned board/players/maps, matching the pattern used
   * by the existing shared mutators.
   */
  nextState: GameState;

  /**
   * Optional per-player elimination deltas caused directly by this operation.
   * For simple non-capturing movement this is non-empty only when the move
   * lands on the mover's own marker and triggers bottom-ring elimination.
   */
  eliminatedRingsByPlayer?: { [player: number]: number };
}

/**
 * Apply a non-capturing stack movement (canonical `move_stack` / `move_ring`)
 * to a GameState, including:
 *
 * - leaving a departure marker at `from` (unless `leaveDepartureMarker === false`);
 * - processing intermediate markers along the path using the shared
 *   marker path semantics from core.ts;
 * - moving or merging the stack at `from` into `to`;
 * - handling the "landing on own marker eliminates the bottom ring" rule; and
 * - updating per-player and global elimination counters.
 *
 * Preconditions / invariants:
 *
 * - The move has already been validated using the shared movement/capture
 *   validators or equivalent host logic. This helper does not re-check legality.
 * - The provided GameState must obey the standard board invariants enforced by
 *   BoardManager on the backend and the sandbox engines:
 *   - at most one of stack / marker / collapsed space per position;
 *   - stackHeight / capHeight / controllingPlayer consistent with rings.
 * - Hosts are responsible for updating currentPhase, currentPlayer, any
 *   decision-phase flags, and history; this helper focuses purely on board
 *   and elimination effects.
 *
 * Error handling:
 *
 * - Implementations are allowed to throw when structural invariants are
 *   violated (for example, no stack at `from`) but should never throw for a
 *   move that passed validation on a consistent state.
 */
export function applySimpleMovement(
  state: GameState,
  params: SimpleMovementParams
): MovementApplicationOutcome {
  // The concrete implementation will be ported from the non-capturing
  // movement branches in the backend GameEngine and sandbox movement engine
  // and will delegate marker handling to the shared marker path helper
  // in core.ts. For P0 Task #21 this helper is intentionally left as a
  // stub so that hosts are not yet rewired to it.
  throw new Error(
    'TODO(P0-HELPERS): applySimpleMovement is a design-time stub. ' +
      'See P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md for the intended semantics.'
  );
}

/**
 * Apply a single overtaking capture segment to a GameState.
 *
 * Responsibilities:
 *
 * - Leave a departure marker at `from` using the same semantics as simple
 *   movement.
 * - Process markers along the path `from → target → landing` using the shared
 *   marker path rules from core.ts.
 * - Apply the overtaking capture itself:
 *   - remove the captured stack's top ring and append it to the bottom of
 *     the attacker stack (preserving attacker on top),
 *   - update both stacks' heights, cap heights, and controlling players,
 *     removing empty stacks entirely.
 * - Place the (possibly enlarged) attacker stack at `landing`.
 * - If `landing` contains an own-colour marker for `params.player`, remove
 *   that marker and eliminate the bottom ring of the combined stack, updating
 *   both per-player and board-level elimination accounting.
 *
 * Chain semantics:
 *
 * - This helper applies a single capture segment in isolation; it does not
 *   track or enforce "must continue chain" rules. Hosts implement the chain
 *   state machine on top by repeatedly calling the shared capture-chain
 *   enumeration helpers and this mutator until no further segments are
 *   available.
 */
export function applyCaptureSegment(
  state: GameState,
  params: CaptureSegmentParams
): MovementApplicationOutcome {
  throw new Error(
    'TODO(P0-HELPERS): applyCaptureSegment is a design-time stub. ' +
      'See P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md for the intended semantics.'
  );
}
