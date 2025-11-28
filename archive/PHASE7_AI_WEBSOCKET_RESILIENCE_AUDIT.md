# Phase 7: AI/WebSocket Resilience Audit

## Overview

This document details the Phase 7 implementation addressing Weakness #6 from the architecture assessment:

> **Error handling and resilience in async/gameflow paths**
>
> - Scope: WebSocket lifecycle, AI service calls, Python rules integration
> - The most complex async paths are where transient errors and race conditions most likely occur
> - Game sessions, AI calls, and WebSocket events form a distributed state machine
> - If transitions aren't modeled explicitly, inconsistent states can occur

## 1. State Machine Audit (7.1)

### 1.1 AI Request State Machine (`src/shared/stateMachines/aiRequest.ts`)

**States:**

- `idle` - No active request
- `queued` - Request queued but not started (includes `requestedAt`, optional `timeoutMs`)
- `in_flight` - Request sent to AI service (includes `requestedAt`, `lastAttemptAt`, `attempt`, optional `deadlineAt`)
- `fallback_local` - Fallen back to local heuristic AI
- `completed` - Request completed successfully (includes `completedAt`, `latencyMs`)
- `timed_out` - Request timed out (includes `requestedAt`, `completedAt`, `durationMs`, `attempt`)
- `failed` - Request failed (includes `completedAt`, `code`, optional `aiErrorType`, `durationMs`)
- `canceled` - Request canceled (includes `completedAt`, `reason`, optional `durationMs`)

**Transitions:**

```
idle → queued → in_flight → completed
                         → timed_out
                         → fallback_local → completed
                         → failed
     (any non-terminal) → canceled
```

**Key Functions:**

- `markQueued(now, timeoutMs?)` - Start a new request
- `markInFlight(previous, now, timeoutMs?)` - Mark request as sent
- `markCompleted(previous, now)` - Mark successful completion
- `markTimedOut(previous, now)` - Mark explicit timeout
- `markFailed(code, aiErrorType, previous, now)` - Mark failure
- `markCanceled(reason, previous, now)` - Cancel request
- `isTerminalState(state)` - Check if state is terminal
- `isCancelable(state)` - Check if state can be canceled
- `isDeadlineExceeded(state, now)` - Check if timeout deadline passed

### 1.2 Connection State Machine (`src/shared/stateMachines/connection.ts`)

**States:**

- `connected` - Player actively connected (includes `connectedAt`, `lastSeenAt`, optional `playerNumber`)
- `disconnected_pending_reconnect` - Disconnected within reconnection window (includes `disconnectedAt`, `deadlineAt`)
- `disconnected_expired` - Reconnection window expired (includes `disconnectedAt`, `expiredAt`)

**Transitions:**

```
connected → disconnected_pending_reconnect → connected (reconnection)
                                          → disconnected_expired
```

**Integration:** Used by `WebSocketServer` to track per-player connection states for diagnostics.

### 1.3 Game Session Status Machine (`src/shared/stateMachines/gameSession.ts`)

**Status Kinds:**

- `waiting_for_players`
- `active_turn`
- `active_ai_thinking`
- `active_decision`
- `completed`
- `abandoned`

**Function:** `deriveGameSessionStatus(gameState, result?)` - Projects high-level status from GameState.

### 1.4 Choice State Machine (`src/shared/stateMachines/choice.ts`)

**States:** `pending`, `resolved`, `timed_out`, `canceled`

**Purpose:** Tracks player choice lifecycle for interactive decisions (line processing, territory processing).

## 2. AI Request Lifecycle Hardening (7.2)

### 2.1 Exactly-Once/At-Most-Once AI Decisions

**Implementation in `GameSession.ts`:**

```typescript
// Start of AI turn - cancel any previous request
this.cancelInFlightAIRequest('manual');

// Create new request with timeout
this.aiRequestState = markQueued(Date.now(), this.aiRequestTimeoutMs);
this.aiAbortController = new AbortController();
this.aiRequestState = markInFlight(this.aiRequestState, Date.now());
```

**Guarantees:**

- Only one AI request active per session at a time
- Previous request canceled before new one starts
- AbortController propagates cancellation to async operations

### 2.2 Explicit Timeout Modeling

```typescript
// Timeout configuration (default 30s, configurable via config)
private readonly aiRequestTimeoutMs: number;

// Timeout wrapper
private async getAIMoveWithTimeout(playerNumber, state, timeoutMs): Promise<Move | null> {
  const timeoutPromise = new Promise<null>((_, reject) => {
    setTimeout(() => reject(new Error('AI request timeout')), timeoutMs);
  });
  const movePromise = globalAIEngine.getAIMove(playerNumber, state);
  return Promise.race([movePromise, timeoutPromise]);
}

// Timeout detection in state machine
if (isDeadlineExceeded(this.aiRequestState)) {
  this.aiRequestState = markTimedOut(this.aiRequestState);
  getMetricsService().recordAITurnRequestTerminal('timed_out');
}
```

### 2.3 Cancellation on Game Termination

```typescript
// GameSession.terminate()
terminate(reason: AIRequestCancelReason = 'session_cleanup'): void {
  this.cancelInFlightAIRequest(reason);
  // ... cleanup
}

// Cancel in-flight request
cancelInFlightAIRequest(reason: AIRequestCancelReason): void {
  if (isCancelable(this.aiRequestState)) {
    this.aiRequestState = markCanceled(reason, this.aiRequestState);
    if (this.aiAbortController) {
      this.aiAbortController.abort();
      this.aiAbortController = null;
    }
  }
}
```

### 2.4 Fallback Handling

**Three-level fallback chain:**

1. AI Service (Python via HTTP)
2. Local Heuristic AI (`globalAIEngine.getLocalFallbackMove()`)
3. Random valid move selection

```typescript
// Service failure → local fallback
private async handleNoMoveFromService(playerNumber, state): Promise<void> {
  this.aiRequestState = markFallbackLocal(this.aiRequestState);
  getMetricsService().recordAIFallback('no_move');

  const fallbackMove = globalAIEngine.getLocalFallbackMove(playerNumber, state);
  if (fallbackMove) {
    const result = await this.rulesFacade.applyMove(fallbackMove);
    if (result.success) {
      this.aiRequestState = markCompleted(this.aiRequestState);
      // ... persist and broadcast
    }
  }
}
```

## 3. WebSocket Session Hardening (7.3)

### 3.1 Connection State Tracking

```typescript
// In WebSocketServer
private readonly playerConnectionStates = new Map<string, PlayerConnectionState>();

// On join game
const nextState = markConnected(gameId, socket.userId, playerNumber, previous);
this.playerConnectionStates.set(key, nextState);

// On disconnect
const nextState = markDisconnectedPendingReconnect(
  previous, gameId, userId, playerNumber, RECONNECTION_TIMEOUT_MS
);
```

### 3.2 Reconnection Handling

**30-second reconnection window:**

```typescript
const RECONNECTION_TIMEOUT_MS = 30_000;

// On disconnect
this.pendingReconnections.set(reconnectionKey, {
  timeout: setTimeout(() => this.handleReconnectionTimeout(...), RECONNECTION_TIMEOUT_MS),
  playerNumber,
  gameId,
  userId,
});

// On reconnection
if (pendingReconnection) {
  clearTimeout(pendingReconnection.timeout);
  this.pendingReconnections.delete(reconnectionKey);
  getMetricsService().recordWebsocketReconnection('success');
}
```

### 3.3 Stale Choice Cleanup

**On reconnection timeout:**

```typescript
private handleReconnectionTimeout(gameId, userId, playerNumber): void {
  getMetricsService().recordWebsocketReconnection('timeout');
  session.getInteractionHandler().cancelAllChoicesForPlayer(playerNumber);
}
```

### 3.4 Diagnostics Endpoint

```typescript
public getGameDiagnosticsForGame(gameId: string): {
  sessionStatus: any | null;
  lastAIRequestState: any | null;
  aiDiagnostics: any | null;
  connections: Record<string, PlayerConnectionState>;
  hasInMemorySession: boolean;
}
```

## 4. Observability Improvements (7.4)

### 4.1 New Metrics

| Metric                                             | Type      | Labels                          | Description                             |
| -------------------------------------------------- | --------- | ------------------------------- | --------------------------------------- |
| `ringrift_websocket_reconnection_total`            | Counter   | `result`                        | Reconnection attempts (success/timeout) |
| `ringrift_game_session_abnormal_termination_total` | Counter   | `reason`                        | Abnormal session terminations           |
| `ringrift_ai_request_latency_ms`                   | Histogram | `outcome`                       | AI request latency                      |
| `ringrift_ai_request_timeout_total`                | Counter   | -                               | AI request timeouts                     |
| `ringrift_ai_turn_request_terminal_total`          | Counter   | `kind`, `code`, `ai_error_type` | AI turn terminal states                 |
| `ringrift_game_session_status_transitions_total`   | Counter   | `from`, `to`                    | Session status changes                  |
| `ringrift_game_session_status_current`             | Gauge     | `status`                        | Current sessions by status              |

### 4.2 Structured Logging

```typescript
// State transitions
logger.info('AI request canceled', { gameId, reason });
logger.info('Game session status changed', { gameId, from, to });
logger.info('Player reconnected within window', { userId, gameId, playerNumber });

// Error recovery
logger.warn('No move from AI service, trying fallback', { gameId, playerNumber });
logger.error('AI fatal failure - both service and fallback failed', { gameId, reason });
```

### 4.3 Session Diagnostics

```typescript
interface SessionAIDiagnostics {
  rulesServiceFailureCount: number;
  rulesShadowErrorCount: number;
  aiServiceFailureCount: number;
  aiFallbackMoveCount: number;
  aiQualityMode: 'normal' | 'fallbackLocalAI' | 'rulesServiceDegraded';
}
```

## 5. Test Coverage (7.5)

### 5.1 New Test File

`tests/unit/AIWebSocketResilience.test.ts` - 40 tests covering:

**AI Request State Machine:**

- Initial state
- State transitions (idle → queued → in_flight → completed)
- Timeout handling with deadlines
- Cancellation from all cancelable states
- Failure handling with error codes
- Terminal state detection
- Latency tracking

**AI Fallback Scenarios:**

- Service unavailable → local fallback → success
- Timeout → fallback
- Game termination during AI request
- Invalid service move → fallback

**WebSocket Connection State Machine:**

- Initial connection
- Reconnection detection (preserves connectedAt)
- Disconnection with pending reconnect window
- Disconnection expired
- Full lifecycle tests

### 5.2 Test Results

```
Test Suites: 1 passed, 1 total
Tests:       40 passed, 40 total
```

## 6. Files Modified

| File                                                  | Changes                                                                       |
| ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| `src/server/game/GameSession.ts`                      | AI request state machine integration, timeout/cancellation, fallback handling |
| `src/server/websocket/server.ts`                      | Reconnection metrics recording                                                |
| `src/server/services/MetricsService.ts`               | New Phase 7 metrics                                                           |
| `tests/unit/AIWebSocketResilience.test.ts`            | New comprehensive test suite                                                  |
| `docs/drafts/PHASE7_AI_WEBSOCKET_RESILIENCE_AUDIT.md` | This documentation                                                            |

## 7. Success Criteria Verification

| Criterion                                                       | Status        |
| --------------------------------------------------------------- | ------------- |
| State machines audited and documented                           | ✅            |
| AI request lifecycle has explicit timeout/cancellation handling | ✅            |
| WebSocket session handles disconnection/reconnection gracefully | ✅            |
| Observability metrics added for async flows                     | ✅            |
| Test coverage for key failure scenarios                         | ✅            |
| No regressions in existing tests                                | ✅ (verified) |

## 8. Future Enhancements

1. **Circuit breaker for AI service** - Automatically fail fast when service is overloaded
2. **Exponential backoff for retries** - Currently single-attempt with fallback
3. **Player notification on AI timeout** - Inform human players when AI is struggling
4. **Distributed session recovery** - Handle server restarts with active games
