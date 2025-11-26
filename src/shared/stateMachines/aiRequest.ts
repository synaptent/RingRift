import type { AIServiceErrorCode } from '../../server/services/AIServiceClient';

/**
 * Minimal explicit state model for a single AI move request lifecycle.
 * Initially used for diagnostics and tests; can later drive orchestration
 * (fallbacks, retries, cancellation) in GameSession / AIEngine.
 *
 * State transitions:
 *   idle → queued → in_flight → completed
 *                 → in_flight → timed_out
 *                 → in_flight → fallback_local → completed
 *                 → in_flight → failed
 *                 → canceled (any non-terminal state)
 */

export type AIRequestTerminalCode = AIServiceErrorCode | 'AI_SERVICE_OVERLOADED';

/**
 * Reason codes for AI request cancellation.
 */
export type AIRequestCancelReason =
  | 'game_terminated'
  | 'player_disconnected'
  | 'session_cleanup'
  | 'manual';

export type AIRequestState =
  | { kind: 'idle' }
  | { kind: 'queued'; requestedAt: number; timeoutMs?: number }
  | {
      kind: 'in_flight';
      requestedAt: number;
      lastAttemptAt: number;
      attempt: number;
      /** Optional timeout deadline (epoch ms) for this request */
      deadlineAt?: number;
    }
  | { kind: 'fallback_local'; requestedAt: number; lastAttemptAt: number }
  | { kind: 'completed'; completedAt: number; latencyMs?: number }
  | {
      kind: 'timed_out';
      requestedAt: number;
      completedAt: number;
      /** Duration from request start to timeout */
      durationMs: number;
      /** The attempt that timed out */
      attempt: number;
    }
  | {
      kind: 'failed';
      completedAt: number;
      code: AIRequestTerminalCode;
      aiErrorType?: string | undefined;
      /** Duration from request start to failure */
      durationMs?: number;
    }
  | {
      kind: 'canceled';
      completedAt: number;
      reason: AIRequestCancelReason | string;
      /** Duration from request start to cancellation */
      durationMs?: number;
    };

export const idleAIRequest: AIRequestState = { kind: 'idle' };

/**
 * Check if a state is terminal (no further transitions possible)
 */
export function isTerminalState(state: AIRequestState): boolean {
  return (
    state.kind === 'completed' ||
    state.kind === 'failed' ||
    state.kind === 'canceled' ||
    state.kind === 'timed_out'
  );
}

/**
 * Check if the state represents an in-progress request that can be canceled
 */
export function isCancelable(state: AIRequestState): boolean {
  return state.kind === 'queued' || state.kind === 'in_flight' || state.kind === 'fallback_local';
}

export function markQueued(now: number = Date.now(), timeoutMs?: number): AIRequestState {
  return { kind: 'queued', requestedAt: now, ...(timeoutMs !== undefined && { timeoutMs }) };
}

export function markInFlight(
  previous: AIRequestState,
  now: number = Date.now(),
  timeoutMs?: number
): AIRequestState {
  const requestedAt =
    previous.kind === 'queued' || previous.kind === 'in_flight' ? previous.requestedAt : now;
  const attempt = previous.kind === 'in_flight' ? previous.attempt + 1 : 1;

  // Calculate deadline from timeoutMs if provided, or inherit from queued state
  let deadlineAt: number | undefined;
  if (timeoutMs !== undefined) {
    deadlineAt = now + timeoutMs;
  } else if (previous.kind === 'queued' && previous.timeoutMs !== undefined) {
    deadlineAt = now + previous.timeoutMs;
  }

  return {
    kind: 'in_flight',
    requestedAt,
    lastAttemptAt: now,
    attempt,
    ...(deadlineAt !== undefined && { deadlineAt }),
  };
}

export function markFallbackLocal(
  previous: AIRequestState,
  now: number = Date.now()
): AIRequestState {
  const requestedAt =
    previous.kind === 'in_flight' || previous.kind === 'queued' ? previous.requestedAt : now;
  return { kind: 'fallback_local', requestedAt, lastAttemptAt: now };
}

export function markCompleted(
  previous: AIRequestState | undefined,
  now: number = Date.now()
): AIRequestState {
  const requestedAt =
    previous &&
    (previous.kind === 'in_flight' ||
      previous.kind === 'queued' ||
      previous.kind === 'fallback_local')
      ? previous.requestedAt
      : now;
  const latencyMs = now - requestedAt;
  return { kind: 'completed', completedAt: now, latencyMs };
}

/**
 * Mark a request as timed out. This creates an explicit timeout terminal state
 * distinct from a generic failure, enabling better observability.
 */
export function markTimedOut(previous: AIRequestState, now: number = Date.now()): AIRequestState {
  const requestedAt =
    previous.kind === 'in_flight' || previous.kind === 'queued' ? previous.requestedAt : now;
  const durationMs = now - requestedAt;
  const attempt = previous.kind === 'in_flight' ? previous.attempt : 1;

  return {
    kind: 'timed_out',
    requestedAt,
    completedAt: now,
    durationMs,
    attempt,
  };
}

export function markFailed(
  code: AIRequestTerminalCode,
  aiErrorType?: string,
  previous?: AIRequestState,
  now: number = Date.now()
): AIRequestState {
  const requestedAt =
    previous &&
    (previous.kind === 'in_flight' ||
      previous.kind === 'queued' ||
      previous.kind === 'fallback_local')
      ? previous.requestedAt
      : now;
  const durationMs = now - requestedAt;

  return { kind: 'failed', completedAt: now, code, aiErrorType, durationMs };
}

/**
 * Mark a request as canceled. Use this when:
 * - Game terminates during AI request
 * - Player disconnects
 * - Session cleanup
 */
export function markCanceled(
  reason: AIRequestCancelReason | string,
  previous?: AIRequestState,
  now: number = Date.now()
): AIRequestState {
  const requestedAt =
    previous &&
    (previous.kind === 'in_flight' ||
      previous.kind === 'queued' ||
      previous.kind === 'fallback_local')
      ? previous.requestedAt
      : undefined;

  if (requestedAt !== undefined) {
    const durationMs = now - requestedAt;
    return { kind: 'canceled', completedAt: now, reason, durationMs };
  }
  return { kind: 'canceled', completedAt: now, reason };
}

/**
 * Get the terminal kind for metrics/logging purposes
 */
export function getTerminalKind(state: AIRequestState): string | null {
  if (isTerminalState(state)) {
    return state.kind;
  }
  return null;
}

/**
 * Check if a deadline has passed for an in-flight request
 */
export function isDeadlineExceeded(state: AIRequestState, now: number = Date.now()): boolean {
  if (state.kind !== 'in_flight') {
    return false;
  }
  return state.deadlineAt !== undefined && now >= state.deadlineAt;
}
