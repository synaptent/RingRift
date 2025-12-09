/**
 * DecisionPhaseState - State machine for player decision timeouts
 *
 * This module provides a shared state machine for managing player decisions
 * that require timeouts. It's used by both the server (GameSession) and
 * client (sandbox) to handle:
 *
 * - Line reward decisions (eliminate vs territory)
 * - Territory region ordering
 * - Chain capture continuation
 * - Other phase-specific player choices
 *
 * The state machine tracks:
 * - Current decision state (idle, pending, warning, expired, etc.)
 * - Which player needs to decide
 * - Which game phase triggered the decision
 * - Deadline and remaining time
 *
 * Usage:
 * ```typescript
 * // Start a decision
 * let state = initializeDecision({
 *   phase: 'line_processing',
 *   player: 1,
 *   choiceType: 'line_reward',
 *   timeoutMs: 30000,
 * });
 *
 * // Issue warning at 5 seconds remaining
 * state = issueWarning(state, 5000);
 *
 * // Decision resolved by player
 * state = resolveDecision(state, 'player_action', moveId);
 *
 * // Or decision expired
 * state = expireDecision(state);
 * ```
 *
 * @module DecisionPhaseState
 */

import type { GamePhase } from '../types/game';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Types of player choices that may have timeouts.
 * Aligns with PlayerChoiceType from the choice state machine.
 */
export type DecisionChoiceType =
  | 'line_reward'
  | 'line_order'
  | 'region_order'
  | 'elimination_target'
  | 'capture_direction'
  | 'chain_capture'
  | 'no_action';

/**
 * More specific choice kind for telemetry/debugging.
 */
export type DecisionChoiceKind =
  | 'line_reward_choice'
  | 'line_order_choice'
  | 'region_order_choice'
  | 'elimination_target_choice'
  | 'chain_capture_choice'
  | 'no_line_action'
  | 'no_territory_action'
  | 'no_movement_action'
  | 'no_placement_action';

/**
 * How a decision was resolved.
 */
export type DecisionResolutionType = 'player_action' | 'auto_resolved' | 'timeout' | 'cancelled';

// ═══════════════════════════════════════════════════════════════════════════
// STATES - Discriminated union
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Discriminated union of all decision phase states.
 */
export type DecisionPhaseState =
  | DecisionIdleState
  | DecisionPendingState
  | DecisionWarningState
  | DecisionExpiredState
  | DecisionResolvedState
  | DecisionCancelledState;

/**
 * No active decision. Default state.
 */
export interface DecisionIdleState {
  readonly kind: 'idle';
}

/**
 * A decision is pending - waiting for player input.
 */
export interface DecisionPendingState {
  readonly kind: 'pending';
  /** Game phase that triggered this decision */
  readonly phase: GamePhase;
  /** Player who must decide */
  readonly player: number;
  /** Type of choice */
  readonly choiceType: DecisionChoiceType;
  /** Specific kind for telemetry */
  readonly choiceKind: DecisionChoiceKind;
  /** Timestamp when decision started (ms since epoch) */
  readonly startedAt: number;
  /** Deadline timestamp (ms since epoch) */
  readonly deadlineMs: number;
  /** Total timeout duration in ms */
  readonly timeoutMs: number;
}

/**
 * Warning has been issued - deadline approaching.
 */
export interface DecisionWarningState {
  readonly kind: 'warning_issued';
  /** Game phase that triggered this decision */
  readonly phase: GamePhase;
  /** Player who must decide */
  readonly player: number;
  /** Type of choice */
  readonly choiceType: DecisionChoiceType;
  /** Specific kind for telemetry */
  readonly choiceKind: DecisionChoiceKind;
  /** Timestamp when decision started */
  readonly startedAt: number;
  /** Deadline timestamp */
  readonly deadlineMs: number;
  /** Remaining time when warning was issued */
  readonly remainingMsAtWarning: number;
  /** Timestamp when warning was issued */
  readonly warningIssuedAt: number;
}

/**
 * Decision expired - timeout reached without player action.
 */
export interface DecisionExpiredState {
  readonly kind: 'expired';
  /** Game phase that triggered this decision */
  readonly phase: GamePhase;
  /** Player who failed to decide */
  readonly player: number;
  /** Type of choice that expired */
  readonly choiceType: DecisionChoiceType;
  /** Specific kind for telemetry */
  readonly choiceKind: DecisionChoiceKind;
  /** Timestamp when decision expired */
  readonly expiredAt: number;
  /** Original deadline */
  readonly deadlineMs: number;
}

/**
 * Decision was resolved (by player, auto, or timeout).
 */
export interface DecisionResolvedState {
  readonly kind: 'resolved';
  /** How the decision was resolved */
  readonly resolution: DecisionResolutionType;
  /** Game phase that triggered this decision */
  readonly phase: GamePhase;
  /** Player who was deciding */
  readonly player: number;
  /** Type of choice */
  readonly choiceType: DecisionChoiceType;
  /** Move ID that resolved the decision (if applicable) */
  readonly resolvedMoveId?: string;
  /** Timestamp when resolved */
  readonly resolvedAt: number;
  /** How long the decision took (ms) */
  readonly durationMs: number;
}

/**
 * Decision was cancelled before resolution.
 */
export interface DecisionCancelledState {
  readonly kind: 'cancelled';
  /** Reason for cancellation */
  readonly reason: string;
  /** Game phase at cancellation */
  readonly phase?: GamePhase;
  /** Player who was deciding */
  readonly player?: number;
  /** Timestamp when cancelled */
  readonly cancelledAt: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Parameters for initializing a new decision.
 */
export interface InitializeDecisionParams {
  phase: GamePhase;
  player: number;
  choiceType: DecisionChoiceType;
  choiceKind: DecisionChoiceKind;
  timeoutMs: number;
  /** Optional timestamp override (for testing) */
  nowMs?: number;
}

/**
 * Create the initial idle state.
 */
export function createIdleState(): DecisionIdleState {
  return { kind: 'idle' };
}

/**
 * Initialize a new pending decision.
 *
 * @param params Decision parameters
 * @returns New pending state
 */
export function initializeDecision(params: InitializeDecisionParams): DecisionPendingState {
  const nowMs = params.nowMs ?? Date.now();
  return {
    kind: 'pending',
    phase: params.phase,
    player: params.player,
    choiceType: params.choiceType,
    choiceKind: params.choiceKind,
    startedAt: nowMs,
    deadlineMs: nowMs + params.timeoutMs,
    timeoutMs: params.timeoutMs,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Issue a warning that the deadline is approaching.
 *
 * @param state Current pending state
 * @param remainingMs Remaining time when warning issued
 * @param nowMs Optional timestamp override
 * @returns Warning state
 */
export function issueWarning(
  state: DecisionPendingState,
  remainingMs: number,
  nowMs?: number
): DecisionWarningState {
  return {
    kind: 'warning_issued',
    phase: state.phase,
    player: state.player,
    choiceType: state.choiceType,
    choiceKind: state.choiceKind,
    startedAt: state.startedAt,
    deadlineMs: state.deadlineMs,
    remainingMsAtWarning: remainingMs,
    warningIssuedAt: nowMs ?? Date.now(),
  };
}

/**
 * Mark the decision as expired due to timeout.
 *
 * @param state Current pending or warning state
 * @param nowMs Optional timestamp override
 * @returns Expired state
 */
export function expireDecision(
  state: DecisionPendingState | DecisionWarningState,
  nowMs?: number
): DecisionExpiredState {
  return {
    kind: 'expired',
    phase: state.phase,
    player: state.player,
    choiceType: state.choiceType,
    choiceKind: state.choiceKind,
    expiredAt: nowMs ?? Date.now(),
    deadlineMs: state.deadlineMs,
  };
}

/**
 * Resolve the decision (player action, auto, or timeout resolution).
 *
 * @param state Current pending, warning, or expired state
 * @param resolution How it was resolved
 * @param moveId Optional move ID that resolved it
 * @param nowMs Optional timestamp override
 * @returns Resolved state
 */
export function resolveDecision(
  state: DecisionPendingState | DecisionWarningState | DecisionExpiredState,
  resolution: DecisionResolutionType,
  moveId?: string,
  nowMs?: number
): DecisionResolvedState {
  const resolvedAt = nowMs ?? Date.now();
  const startedAt =
    state.kind === 'expired'
      ? state.deadlineMs - (state.deadlineMs - state.expiredAt)
      : state.startedAt;

  return {
    kind: 'resolved',
    resolution,
    phase: state.phase,
    player: state.player,
    choiceType: state.choiceType,
    ...(moveId !== undefined ? { resolvedMoveId: moveId } : {}),
    resolvedAt,
    durationMs: resolvedAt - startedAt,
  };
}

/**
 * Cancel the decision.
 *
 * @param state Current state (any active state)
 * @param reason Reason for cancellation
 * @param nowMs Optional timestamp override
 * @returns Cancelled state
 */
export function cancelDecision(
  state: DecisionPhaseState,
  reason: string,
  nowMs?: number
): DecisionCancelledState {
  const hasPhaseAndPlayer = state.kind !== 'idle' && state.kind !== 'cancelled';

  return {
    kind: 'cancelled',
    reason,
    ...(hasPhaseAndPlayer ? { phase: state.phase, player: state.player } : {}),
    cancelledAt: nowMs ?? Date.now(),
  };
}

/**
 * Clear the decision state back to idle.
 */
export function clearDecision(): DecisionIdleState {
  return createIdleState();
}

// ═══════════════════════════════════════════════════════════════════════════
// QUERIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a decision is currently active (pending or warning).
 */
export function isDecisionActive(
  state: DecisionPhaseState
): state is DecisionPendingState | DecisionWarningState {
  return state.kind === 'pending' || state.kind === 'warning_issued';
}

/**
 * Check if the decision has expired or been resolved.
 */
export function isDecisionTerminal(
  state: DecisionPhaseState
): state is DecisionExpiredState | DecisionResolvedState | DecisionCancelledState {
  return state.kind === 'expired' || state.kind === 'resolved' || state.kind === 'cancelled';
}

/**
 * Get remaining time for an active decision.
 *
 * @param state Current state
 * @param nowMs Optional timestamp override
 * @returns Remaining milliseconds, or null if not active
 */
export function getRemainingTime(state: DecisionPhaseState, nowMs?: number): number | null {
  if (!isDecisionActive(state)) {
    return null;
  }
  const now = nowMs ?? Date.now();
  return Math.max(0, state.deadlineMs - now);
}

/**
 * Get decision metadata for telemetry/logging.
 */
export function getDecisionMetadata(state: DecisionPhaseState): {
  kind: string;
  phase?: GamePhase;
  player?: number;
  choiceType?: DecisionChoiceType;
  remainingMs?: number;
} {
  if (state.kind === 'idle') {
    return { kind: 'idle' };
  }
  if (state.kind === 'cancelled') {
    return {
      kind: 'cancelled',
      ...(state.phase !== undefined ? { phase: state.phase } : {}),
      ...(state.player !== undefined ? { player: state.player } : {}),
    };
  }
  if (state.kind === 'resolved') {
    return {
      kind: 'resolved',
      phase: state.phase,
      player: state.player,
      choiceType: state.choiceType,
    };
  }

  const remaining = getRemainingTime(state);
  return {
    kind: state.kind,
    phase: state.phase,
    player: state.player,
    choiceType: state.choiceType,
    ...(remaining !== null ? { remainingMs: remaining } : {}),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// TIMEOUT SCHEDULING HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Configuration for decision timeout scheduling.
 */
export interface DecisionTimeoutConfig {
  /** Base timeout for decisions (ms) */
  baseTimeoutMs: number;
  /** Time before deadline to issue warning (ms) */
  warningBeforeMs: number;
  /** Override timeout for specific choice types */
  choiceTypeOverrides?: Partial<Record<DecisionChoiceType, number>>;
}

/**
 * Default timeout configuration.
 */
export const DEFAULT_TIMEOUT_CONFIG: DecisionTimeoutConfig = {
  baseTimeoutMs: 30000, // 30 seconds
  warningBeforeMs: 5000, // 5 second warning
  choiceTypeOverrides: {
    chain_capture: 15000, // Shorter timeout for chain capture
    no_action: 10000, // Shorter for no-action decisions
  },
};

/**
 * Get timeout for a specific choice type.
 */
export function getTimeoutForChoice(
  choiceType: DecisionChoiceType,
  config: DecisionTimeoutConfig = DEFAULT_TIMEOUT_CONFIG
): number {
  return config.choiceTypeOverrides?.[choiceType] ?? config.baseTimeoutMs;
}

/**
 * Calculate when to schedule warning callback.
 */
export function getWarningScheduleTime(
  state: DecisionPendingState,
  config: DecisionTimeoutConfig = DEFAULT_TIMEOUT_CONFIG
): number {
  const warningAt = state.deadlineMs - config.warningBeforeMs;
  const now = Date.now();
  return Math.max(0, warningAt - now);
}

/**
 * Calculate when to schedule expiry callback.
 */
export function getExpiryScheduleTime(state: DecisionPendingState): number {
  const now = Date.now();
  return Math.max(0, state.deadlineMs - now);
}
