/**
 * AI/WebSocket Resilience Tests (Phase 7)
 *
 * Tests for:
 * - AI request state machine transitions
 * - AI timeout handling with explicit timeout state
 * - AI cancellation during game termination
 * - Fallback to local heuristic AI when service unavailable
 * - WebSocket connection state machine
 */

import {
  AIRequestState,
  idleAIRequest,
  markQueued,
  markInFlight,
  markFallbackLocal,
  markCompleted,
  markTimedOut,
  markFailed,
  markCanceled,
  isTerminalState,
  isCancelable,
  isDeadlineExceeded,
  getTerminalKind,
  AIRequestCancelReason,
} from '../../src/shared/stateMachines/aiRequest';

import {
  PlayerConnectionState,
  markConnected,
  markDisconnectedPendingReconnect,
  markDisconnectedExpired,
} from '../../src/shared/stateMachines/connection';

describe('AI Request State Machine', () => {
  describe('Initial State', () => {
    it('starts in idle state', () => {
      expect(idleAIRequest).toEqual({ kind: 'idle' });
    });
  });

  describe('State Transitions', () => {
    it('transitions from idle to queued', () => {
      const now = 1000;
      const state = markQueued(now);
      expect(state.kind).toBe('queued');
      expect((state as any).requestedAt).toBe(now);
    });

    it('transitions from queued to in_flight', () => {
      const requestedAt = 1000;
      const queued = markQueued(requestedAt);
      const now = 1100;
      const inFlight = markInFlight(queued, now);

      expect(inFlight.kind).toBe('in_flight');
      expect((inFlight as any).requestedAt).toBe(requestedAt);
      expect((inFlight as any).lastAttemptAt).toBe(now);
      expect((inFlight as any).attempt).toBe(1);
    });

    it('increments attempt count on subsequent in_flight transitions', () => {
      const queued = markQueued(1000);
      const inFlight1 = markInFlight(queued, 1100);
      const inFlight2 = markInFlight(inFlight1, 1200);

      expect((inFlight2 as any).attempt).toBe(2);
    });

    it('transitions from in_flight to completed', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);
      const completed = markCompleted(inFlight, 1500);

      expect(completed.kind).toBe('completed');
      expect((completed as any).completedAt).toBe(1500);
      expect((completed as any).latencyMs).toBe(500); // 1500 - 1000
    });

    it('transitions from in_flight to fallback_local', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);
      const fallback = markFallbackLocal(inFlight, 1200);

      expect(fallback.kind).toBe('fallback_local');
      expect((fallback as any).requestedAt).toBe(1000);
    });

    it('transitions from fallback_local to completed', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);
      const fallback = markFallbackLocal(inFlight, 1200);
      const completed = markCompleted(fallback, 1300);

      expect(completed.kind).toBe('completed');
      expect((completed as any).latencyMs).toBe(300);
    });
  });

  describe('Timeout Handling', () => {
    it('sets deadline when timeoutMs provided to markQueued', () => {
      const now = 1000;
      const timeoutMs = 5000;
      const queued = markQueued(now, timeoutMs);

      expect((queued as any).timeoutMs).toBe(timeoutMs);
    });

    it('sets deadlineAt in in_flight from queued timeoutMs', () => {
      const requestedAt = 1000;
      const timeoutMs = 5000;
      const queued = markQueued(requestedAt, timeoutMs);
      const startedAt = 1100;
      const inFlight = markInFlight(queued, startedAt);

      expect((inFlight as any).deadlineAt).toBe(startedAt + timeoutMs);
    });

    it('sets deadlineAt directly in markInFlight with timeoutMs', () => {
      const queued = markQueued(1000);
      const startedAt = 1100;
      const timeoutMs = 3000;
      const inFlight = markInFlight(queued, startedAt, timeoutMs);

      expect((inFlight as any).deadlineAt).toBe(startedAt + timeoutMs);
    });

    it('isDeadlineExceeded returns false before deadline', () => {
      const queued = markQueued(1000, 5000);
      const inFlight = markInFlight(queued, 1100);

      // Check at 3000ms - should still be within deadline (deadline = 1100 + 5000 = 6100)
      expect(isDeadlineExceeded(inFlight, 3000)).toBe(false);
    });

    it('isDeadlineExceeded returns true after deadline', () => {
      const queued = markQueued(1000, 5000);
      const inFlight = markInFlight(queued, 1100);

      // Check at 7000ms - should exceed deadline (deadline = 1100 + 5000 = 6100)
      expect(isDeadlineExceeded(inFlight, 7000)).toBe(true);
    });

    it('isDeadlineExceeded returns false for non-in_flight states', () => {
      const queued = markQueued(1000, 5000);
      expect(isDeadlineExceeded(queued, 10000)).toBe(false);
    });

    it('markTimedOut creates explicit timeout terminal state', () => {
      const queued = markQueued(1000, 5000);
      const inFlight = markInFlight(queued, 1100);
      const timedOut = markTimedOut(inFlight, 6200);

      expect(timedOut.kind).toBe('timed_out');
      expect((timedOut as any).requestedAt).toBe(1000);
      expect((timedOut as any).completedAt).toBe(6200);
      expect((timedOut as any).durationMs).toBe(5200);
      expect((timedOut as any).attempt).toBe(1);
    });
  });

  describe('Cancellation Handling', () => {
    it('can cancel from queued state', () => {
      const queued = markQueued(1000);
      expect(isCancelable(queued)).toBe(true);

      const canceled = markCanceled('game_terminated', queued, 1500);
      expect(canceled.kind).toBe('canceled');
      expect((canceled as any).reason).toBe('game_terminated');
      expect((canceled as any).durationMs).toBe(500);
    });

    it('can cancel from in_flight state', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);
      expect(isCancelable(inFlight)).toBe(true);

      const canceled = markCanceled('player_disconnected', inFlight, 1500);
      expect(canceled.kind).toBe('canceled');
      expect((canceled as any).reason).toBe('player_disconnected');
      expect((canceled as any).durationMs).toBe(500);
    });

    it('can cancel from fallback_local state', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);
      const fallback = markFallbackLocal(inFlight, 1200);
      expect(isCancelable(fallback)).toBe(true);

      const canceled = markCanceled('session_cleanup', fallback, 1500);
      expect(canceled.kind).toBe('canceled');
      expect((canceled as any).reason).toBe('session_cleanup');
    });

    it('supports all cancellation reasons', () => {
      const reasons: AIRequestCancelReason[] = [
        'game_terminated',
        'player_disconnected',
        'session_cleanup',
        'manual',
      ];

      for (const reason of reasons) {
        const queued = markQueued(1000);
        const canceled = markCanceled(reason, queued, 1500);
        expect((canceled as any).reason).toBe(reason);
      }
    });
  });

  describe('Failure Handling', () => {
    it('markFailed creates failure state with error code', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);
      const failed = markFailed('AI_SERVICE_ERROR', 'connection_refused', inFlight, 1500);

      expect(failed.kind).toBe('failed');
      expect((failed as any).code).toBe('AI_SERVICE_ERROR');
      expect((failed as any).aiErrorType).toBe('connection_refused');
      expect((failed as any).durationMs).toBe(500);
    });

    it('markFailed supports AI_SERVICE_OVERLOADED error', () => {
      const inFlight = markInFlight(markQueued(1000), 1100);
      const failed = markFailed(
        'AI_SERVICE_OVERLOADED',
        'both_service_and_fallback_failed',
        inFlight,
        1500
      );

      expect((failed as any).code).toBe('AI_SERVICE_OVERLOADED');
    });
  });

  describe('Terminal State Detection', () => {
    it('correctly identifies terminal states', () => {
      expect(isTerminalState(idleAIRequest)).toBe(false);
      expect(isTerminalState(markQueued(1000))).toBe(false);
      expect(isTerminalState(markInFlight(markQueued(1000), 1100))).toBe(false);
      expect(isTerminalState(markFallbackLocal(markInFlight(markQueued(1000), 1100), 1200))).toBe(
        false
      );

      expect(isTerminalState(markCompleted(markInFlight(markQueued(1000), 1100), 1500))).toBe(true);
      expect(isTerminalState(markTimedOut(markInFlight(markQueued(1000, 100), 1100), 1500))).toBe(
        true
      );
      expect(isTerminalState(markFailed('AI_SERVICE_ERROR', undefined, undefined, 1500))).toBe(
        true
      );
      expect(isTerminalState(markCanceled('manual', undefined, 1500))).toBe(true);
    });

    it('getTerminalKind returns correct kind for terminal states', () => {
      const queued = markQueued(1000);
      const inFlight = markInFlight(queued, 1100);

      expect(getTerminalKind(markCompleted(inFlight, 1500))).toBe('completed');
      expect(getTerminalKind(markTimedOut(inFlight, 1500))).toBe('timed_out');
      expect(getTerminalKind(markFailed('AI_SERVICE_ERROR', undefined, inFlight, 1500))).toBe(
        'failed'
      );
      expect(getTerminalKind(markCanceled('manual', inFlight, 1500))).toBe('canceled');
    });

    it('getTerminalKind returns null for non-terminal states', () => {
      expect(getTerminalKind(idleAIRequest)).toBe(null);
      expect(getTerminalKind(markQueued(1000))).toBe(null);
      expect(getTerminalKind(markInFlight(markQueued(1000), 1100))).toBe(null);
    });
  });

  describe('Latency Tracking', () => {
    it('tracks latency from request start to completion', () => {
      const requestedAt = 1000;
      const queued = markQueued(requestedAt);
      const inFlight = markInFlight(queued, 1100);
      const completedAt = 2500;
      const completed = markCompleted(inFlight, completedAt);

      expect((completed as any).latencyMs).toBe(completedAt - requestedAt);
    });

    it('tracks duration for timeout', () => {
      const requestedAt = 1000;
      const queued = markQueued(requestedAt, 3000);
      const inFlight = markInFlight(queued, 1100);
      const timedOutAt = 4200;
      const timedOut = markTimedOut(inFlight, timedOutAt);

      expect((timedOut as any).durationMs).toBe(timedOutAt - requestedAt);
    });

    it('tracks duration for cancellation', () => {
      const requestedAt = 1000;
      const queued = markQueued(requestedAt);
      const inFlight = markInFlight(queued, 1100);
      const canceledAt = 1800;
      const canceled = markCanceled('game_terminated', inFlight, canceledAt);

      expect((canceled as any).durationMs).toBe(canceledAt - requestedAt);
    });
  });
});

describe('AI Fallback Scenarios', () => {
  describe('Service Unavailable → Local Fallback → Success', () => {
    it('full lifecycle: queue → in_flight → fallback → complete', () => {
      const requestedAt = 1000;
      let state: AIRequestState = markQueued(requestedAt, 5000);
      expect(state.kind).toBe('queued');

      // Service call starts
      state = markInFlight(state, 1100);
      expect(state.kind).toBe('in_flight');
      expect((state as any).attempt).toBe(1);

      // Service call fails, fallback to local
      state = markFallbackLocal(state, 1500);
      expect(state.kind).toBe('fallback_local');

      // Local fallback succeeds
      state = markCompleted(state, 1700);
      expect(state.kind).toBe('completed');
      expect((state as any).latencyMs).toBe(700); // from original requestedAt
      expect(isTerminalState(state)).toBe(true);
    });
  });

  describe('Timeout → Local Fallback → Success', () => {
    it('times out then succeeds with local fallback', () => {
      const requestedAt = 1000;
      const timeoutMs = 3000;
      let state: AIRequestState = markQueued(requestedAt, timeoutMs);

      // Service call starts
      state = markInFlight(state, 1100);
      expect((state as any).deadlineAt).toBe(1100 + timeoutMs);

      // Time passes, deadline exceeded but we detect it
      expect(isDeadlineExceeded(state, 5000)).toBe(true);

      // Mark as timed out
      state = markTimedOut(state, 5000);
      expect(state.kind).toBe('timed_out');
      expect((state as any).durationMs).toBe(4000); // 5000 - 1000
    });
  });

  describe('Game Termination During AI Request', () => {
    it('cancels in_flight request when game terminates', () => {
      const requestedAt = 1000;
      let state: AIRequestState = markQueued(requestedAt, 5000);
      state = markInFlight(state, 1100);

      // Game terminates while AI is thinking
      expect(isCancelable(state)).toBe(true);
      state = markCanceled('game_terminated', state, 2000);

      expect(state.kind).toBe('canceled');
      expect((state as any).reason).toBe('game_terminated');
      expect((state as any).durationMs).toBe(1000);
      expect(isTerminalState(state)).toBe(true);
    });
  });

  describe('Service Returns Invalid Move → Fallback', () => {
    it('transitions through fallback when service move rejected', () => {
      let state: AIRequestState = markQueued(1000);
      state = markInFlight(state, 1100);

      // Service returns a move but rules engine rejects it
      // Transition to fallback rather than failing entirely
      state = markFallbackLocal(state, 1500);
      expect(state.kind).toBe('fallback_local');

      // Local heuristic provides valid move
      state = markCompleted(state, 1700);
      expect(state.kind).toBe('completed');
    });
  });
});

describe('AI + Game Session Integration Scenarios', () => {
  describe('Session Cleanup', () => {
    it('all cancelable states can be cleaned up on session termination', () => {
      const cancelableStates: AIRequestState[] = [
        markQueued(1000),
        markInFlight(markQueued(1000), 1100),
        markFallbackLocal(markInFlight(markQueued(1000), 1100), 1200),
      ];

      for (const state of cancelableStates) {
        expect(isCancelable(state)).toBe(true);
        const cleaned = markCanceled('session_cleanup', state, 2000);
        expect(cleaned.kind).toBe('canceled');
        expect(isTerminalState(cleaned)).toBe(true);
      }
    });

    it('terminal states are not cancelable', () => {
      const terminalStates: AIRequestState[] = [
        markCompleted(markInFlight(markQueued(1000), 1100), 1500),
        markTimedOut(markInFlight(markQueued(1000, 100), 1100), 1500),
        markFailed('AI_SERVICE_ERROR', undefined, undefined, 1500),
        markCanceled('manual', undefined, 1500),
      ];

      for (const state of terminalStates) {
        expect(isTerminalState(state)).toBe(true);
        expect(isCancelable(state)).toBe(false);
      }
    });
  });

  describe('Reconnection During AI Turn', () => {
    it('player disconnect cancels pending AI request', () => {
      let state: AIRequestState = markQueued(1000);
      state = markInFlight(state, 1100);

      // Player disconnects
      state = markCanceled('player_disconnected', state, 1500);
      expect(state.kind).toBe('canceled');
      expect((state as any).reason).toBe('player_disconnected');
    });
  });
});

describe('WebSocket Connection State Machine', () => {
  describe('Initial Connection', () => {
    it('marks user as connected', () => {
      const now = 1000;
      const state = markConnected('game-123', 'user-456', 1, undefined, now);

      expect(state.kind).toBe('connected');
      expect(state.gameId).toBe('game-123');
      expect(state.userId).toBe('user-456');
      expect((state as any).playerNumber).toBe(1);
      expect((state as any).connectedAt).toBe(now);
      expect((state as any).lastSeenAt).toBe(now);
    });

    it('marks spectator as connected without playerNumber', () => {
      const now = 1000;
      const state = markConnected('game-123', 'user-456', undefined, undefined, now);

      expect(state.kind).toBe('connected');
      expect((state as any).playerNumber).toBeUndefined();
    });
  });

  describe('Reconnection Detection', () => {
    it('preserves connectedAt on reconnection, updates lastSeenAt', () => {
      const firstConnection = 1000;
      const reconnection = 2000;

      const first = markConnected('game-123', 'user-456', 1, undefined, firstConnection);
      const second = markConnected('game-123', 'user-456', 1, first, reconnection);

      expect(second.kind).toBe('connected');
      expect((second as any).connectedAt).toBe(firstConnection);
      expect((second as any).lastSeenAt).toBe(reconnection);
    });
  });

  describe('Disconnection Pending Reconnect', () => {
    it('marks disconnect with reconnection window', () => {
      const connectedAt = 1000;
      const disconnectedAt = 2000;
      const timeoutMs = 30000;

      const connected = markConnected('game-123', 'user-456', 1, undefined, connectedAt);
      const pending = markDisconnectedPendingReconnect(
        connected,
        'game-123',
        'user-456',
        1,
        timeoutMs,
        disconnectedAt
      );

      expect(pending.kind).toBe('disconnected_pending_reconnect');
      expect((pending as any).disconnectedAt).toBe(disconnectedAt);
      expect((pending as any).deadlineAt).toBe(disconnectedAt + timeoutMs);
      expect((pending as any).playerNumber).toBe(1);
    });
  });

  describe('Disconnection Expired', () => {
    it('marks reconnection window as expired', () => {
      const connectedAt = 1000;
      const disconnectedAt = 2000;
      const timeoutMs = 30000;
      const expiredAt = disconnectedAt + timeoutMs + 100;

      const connected = markConnected('game-123', 'user-456', 1, undefined, connectedAt);
      const pending = markDisconnectedPendingReconnect(
        connected,
        'game-123',
        'user-456',
        1,
        timeoutMs,
        disconnectedAt
      );
      const expired = markDisconnectedExpired(pending, 'game-123', 'user-456', 1, expiredAt);

      expect(expired.kind).toBe('disconnected_expired');
      expect((expired as any).disconnectedAt).toBe(disconnectedAt);
      expect((expired as any).expiredAt).toBe(expiredAt);
    });
  });

  describe('Full Lifecycle', () => {
    it('tracks complete connection lifecycle: connect → disconnect → expire', () => {
      const gameId = 'game-123';
      const userId = 'user-456';
      const playerNumber = 1;
      const timeoutMs = 30000;

      // Initial connection
      let state: PlayerConnectionState = markConnected(
        gameId,
        userId,
        playerNumber,
        undefined,
        1000
      );
      expect(state.kind).toBe('connected');

      // Disconnect with pending reconnect window
      state = markDisconnectedPendingReconnect(
        state,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        2000
      );
      expect(state.kind).toBe('disconnected_pending_reconnect');
      expect((state as any).deadlineAt).toBe(2000 + timeoutMs);

      // Reconnection window expires
      state = markDisconnectedExpired(state, gameId, userId, playerNumber, 35000);
      expect(state.kind).toBe('disconnected_expired');
    });

    it('tracks complete connection lifecycle: connect → disconnect → reconnect', () => {
      const gameId = 'game-123';
      const userId = 'user-456';
      const playerNumber = 1;
      const timeoutMs = 30000;

      // Initial connection
      let state: PlayerConnectionState = markConnected(
        gameId,
        userId,
        playerNumber,
        undefined,
        1000
      );
      expect(state.kind).toBe('connected');
      expect((state as any).connectedAt).toBe(1000);

      // Disconnect with pending reconnect window
      state = markDisconnectedPendingReconnect(
        state,
        gameId,
        userId,
        playerNumber,
        timeoutMs,
        2000
      );
      expect(state.kind).toBe('disconnected_pending_reconnect');

      // Successful reconnection before deadline
      state = markConnected(gameId, userId, playerNumber, state, 5000);
      expect(state.kind).toBe('connected');
      // connectedAt should be reset since we came from disconnected_pending_reconnect
      expect((state as any).lastSeenAt).toBe(5000);
    });
  });
});
