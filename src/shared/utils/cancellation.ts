// Shared cancellation token primitives for async operations in TS hosts.
//
// These utilities are intentionally small, host-agnostic, and dependency-light.
// They are designed to support the Tier 3 "AI/WebSocket/Gameflow state machine
// hardening" work described in ARCHITECTURE_REMEDIATION_PLAN.md:
//   - Explicit cancellation tokens for async operations (AI requests,
//     WebSocket choices, reconnection timeouts, etc.).
//   - Integration with shared state machines in src/shared/stateMachines/**.
//
// At this stage they are utilities only; callers in server/client code should
// be migrated gradually and covered by targeted unit + integration tests.

export type CancellationReason = unknown;

/**
 * Read-only view of a cancellation token.
 */
export interface CancellationToken {
  /** True once cancel() has been invoked on the associated source. */
  readonly isCanceled: boolean;
  /** Optional reason supplied by the canceller (for logging/diagnostics). */
  readonly reason?: CancellationReason;

  /**
   * Throws an Error if the token has been canceled.
   *
   * Useful for early-exit checks inside async workflows:
   *
   *   token.throwIfCanceled('before starting network request');
   */
  throwIfCanceled(contextMessage?: string): void;
}

/**
 * Mutable source for a {@link CancellationToken}.
 *
 * The typical pattern is:
 *   const source = createCancellationSource();
 *   doAsyncWork(source.token).catch(handleError);
 *   // later, perhaps from a timeout or WebSocket disconnect:
 *   source.cancel(new Error('request timed out'));
 */
export interface CancellationSource {
  readonly token: CancellationToken;
  /**
   * Marks the token as canceled. Subsequent calls are no-ops.
   *
   * The optional `reason` is preserved on the token for diagnostics.
   */
  cancel(reason?: CancellationReason): void;
}

export function createCancellationSource(): CancellationSource {
  let canceled = false;
  let reason: CancellationReason | undefined;

  const token: CancellationToken = {
    get isCanceled() {
      return canceled;
    },
    get reason() {
      return reason;
    },
    throwIfCanceled(contextMessage?: string): void {
      if (!canceled) return;
      const detail = contextMessage ? ` (${contextMessage})` : '';
      const message = `Operation canceled${detail}`;
      const error = new Error(message);
      // Preserve the original reason on the error object for richer logging.
      (error as any).cancellationReason = reason;
      throw error;
    },
  };

  const source: CancellationSource = {
    token,
    cancel(nextReason?: CancellationReason): void {
      if (canceled) return;
      canceled = true;
      reason = nextReason;
    },
  };

  return source;
}

/**
 * Helper to derive a child token from a parent.
 *
 * The child starts non-canceled and becomes canceled whenever the parent is
 * canceled, but canceling the child does **not** affect the parent.
 *
 * This is useful when a high-level operation (for example a game session) has
 * a long-lived parent token and we need shorter-lived child tokens for
 * individual AI requests or WebSocket choices.
 */
export function createLinkedCancellationSource(parent: CancellationToken): CancellationSource {
  const child = createCancellationSource();

  // If the parent is already canceled, propagate immediately.
  if (parent.isCanceled) {
    child.cancel(parent.reason);
    return child;
  }

  // Best-effort polling hook – callers should invoke this periodically from
  // their own loops (for example inside a setInterval or request lifecycle)
  // to observe parent cancellation. We intentionally avoid timers here so the
  // utility stays framework-agnostic.
  function syncFromParent() {
    if (parent.isCanceled) {
      child.cancel(parent.reason);
    }
  }

  // Expose the sync function for explicit use by hosts.
  (child as any).syncFromParent = syncFromParent;

  return child;
}
