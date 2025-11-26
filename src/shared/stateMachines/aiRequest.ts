import type { AIServiceErrorCode } from '../../server/services/AIServiceClient';

/**
 * Minimal explicit state model for a single AI move request lifecycle.
 * Initially used for diagnostics and tests; can later drive orchestration
 * (fallbacks, retries, cancellation) in GameSession / AIEngine.
 */

export type AIRequestTerminalCode = AIServiceErrorCode | 'AI_SERVICE_OVERLOADED';

export type AIRequestState =
  | { kind: 'idle' }
  | { kind: 'queued'; requestedAt: number }
  | { kind: 'in_flight'; requestedAt: number; lastAttemptAt: number; attempt: number }
  | { kind: 'fallback_local'; requestedAt: number; lastAttemptAt: number }
  | { kind: 'completed'; completedAt: number }
  | {
      kind: 'failed';
      completedAt: number;
      code: AIRequestTerminalCode;
      aiErrorType?: string | undefined;
    }
  | { kind: 'canceled'; completedAt: number; reason: string };

export const idleAIRequest: AIRequestState = { kind: 'idle' };

export function markQueued(now: number = Date.now()): AIRequestState {
  return { kind: 'queued', requestedAt: now };
}

export function markInFlight(previous: AIRequestState, now: number = Date.now()): AIRequestState {
  const requestedAt =
    previous.kind === 'queued' || previous.kind === 'in_flight' ? previous.requestedAt : now;
  const attempt = previous.kind === 'in_flight' ? previous.attempt + 1 : 1;
  return { kind: 'in_flight', requestedAt, lastAttemptAt: now, attempt };
}

export function markFallbackLocal(
  previous: AIRequestState,
  now: number = Date.now()
): AIRequestState {
  const requestedAt =
    previous.kind === 'in_flight' || previous.kind === 'queued' ? previous.requestedAt : now;
  return { kind: 'fallback_local', requestedAt, lastAttemptAt: now };
}

export function markCompleted(now: number = Date.now()): AIRequestState {
  return { kind: 'completed', completedAt: now };
}

export function markFailed(
  code: AIRequestTerminalCode,
  aiErrorType?: string,
  now: number = Date.now()
): AIRequestState {
  return { kind: 'failed', completedAt: now, code, aiErrorType };
}

export function markCanceled(reason: string, now: number = Date.now()): AIRequestState {
  return { kind: 'canceled', completedAt: now, reason };
}
