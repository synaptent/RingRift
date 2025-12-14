# AI Timeout/Fallback SLO Budgets

> **Doc Status (2025-12-13): Active Reference**
>
> This document consolidates AI timeout, fallback, and latency SLO budgets to ensure AI-controlled seats never stall production games.

## Overview

AI move generation has multiple layers of timeout/budget protection:

1. **Per-request timeout**: Hard limit on how long the Node backend waits for the Python AI service
2. **Per-tier think time budgets**: Soft search budgets per AI difficulty level
3. **Fallback mechanism**: Local heuristic fallback when remote AI is unavailable/slow
4. **SLO targets**: Production reliability targets for AI latency and fallback rates

## Configuration Summary

| Parameter       | Default Value | Env Variable                    | Description                          |
| --------------- | ------------- | ------------------------------- | ------------------------------------ |
| Request Timeout | 5000ms        | `AI_SERVICE_REQUEST_TIMEOUT_MS` | Max wait for AI service response     |
| Rules Timeout   | 5000ms        | `AI_RULES_REQUEST_TIMEOUT_MS`   | Max wait for Python rules validation |
| Max Concurrent  | 16            | `AI_MAX_CONCURRENT_REQUESTS`    | Concurrent AI request limit          |
| Circuit Breaker | 60000ms       | (hardcoded)                     | Cooldown after repeated failures     |

## Per-Tier Think Time Budgets

Think times are search budgets passed to the AI service, not artificial delays. The AI service attempts to complete search within this budget.

| Difficulty | AI Type      | Think Time | Avg Budget (p50) | P95 Budget |
| ---------- | ------------ | ---------- | ---------------- | ---------- |
| D1         | Random       | 150ms      | ~100ms           | 165ms      |
| D2         | Heuristic    | 200ms      | ~150ms           | 220ms      |
| D3         | Minimax      | 1800ms     | ~1200ms          | 1980ms     |
| D4         | Minimax+NNUE | 2800ms     | ~2000ms          | 3080ms     |
| D5         | MCTS         | 4000ms     | ~3000ms          | 4400ms     |
| D6         | MCTS+Neural  | 5500ms     | ~4000ms          | 6050ms     |
| D7         | MCTS+Neural  | 7500ms     | ~5500ms          | 8250ms     |
| D8         | MCTS+Neural  | 9600ms     | ~7000ms          | 10560ms    |
| D9         | Descent      | 12600ms    | ~9000ms          | 13860ms    |
| D10        | Descent      | 16000ms    | ~12000ms         | 17600ms    |

Budget formulas (from `ai-service/app/config/perf_budgets.py`):

- `max_avg_move_ms = think_time_ms * 1.10`
- `max_p95_move_ms = think_time_ms * 1.25`

## SLO Targets

From `docs/operations/SLO_VERIFICATION.md` and `tests/load/config/thresholds.json`:

| Metric           | Target  | Priority |
| ---------------- | ------- | -------- |
| AI Response p95  | <1000ms | High     |
| AI Response p99  | <2000ms | Medium   |
| AI Fallback Rate | ≤1%     | Medium   |
| Move Stall Rate  | ≤0.5%   | High     |

## Fallback Behavior

When the AI service fails (timeout, error, overload), the system falls back to local heuristics:

1. **Service unavailable**: Immediate fallback to local move selection
2. **Timeout**: After `requestTimeoutMs` (5s default), fallback triggered
3. **Overload**: If concurrent limit reached, proactive fallback
4. **Circuit breaker**: After repeated failures, 60s cooldown before retrying

**Fallback guarantees:**

- Games continue without interruption
- Legal moves always generated (via shared TS rules engine)
- AI strength degraded but functional
- Metrics track `ai_fallback_total` by reason

## Alerting Thresholds

From `monitoring/prometheus/alerts.yml`:

| Alert                    | Condition                     | Severity |
| ------------------------ | ----------------------------- | -------- |
| `AIFallbackRateHigh`     | Fallback rate > threshold     | Warning  |
| `AIFallbackRateCritical` | Majority of requests fallback | Critical |
| `AIRequestHighLatency`   | p95 latency exceeds budget    | Warning  |
| `AIServiceDown`          | AI service health check fails | Critical |

## Never-Stall Guarantee

The combination of:

1. **Hard timeout** (5s) ensures no request blocks indefinitely
2. **Local fallback** ensures a move is always available
3. **Circuit breaker** prevents cascade failures during outages
4. **Concurrent limits** prevent overload of AI service

This means AI-controlled seats will **never stall** a game, even if:

- The AI service is completely down
- Network connectivity is lost
- AI service is overloaded

In degraded mode, AI strength is reduced but games proceed normally.

## Monitoring Queries

```promql
# Current AI fallback rate (10m window)
sum(rate(ringrift_ai_fallback_total[10m])) / sum(rate(ringrift_ai_requests_total[10m]))

# AI latency p95
histogram_quantile(0.95, sum(rate(ringrift_ai_request_duration_seconds_bucket[5m])) by (le))

# AI service status
ringrift_service_status{service="ai_service"}
```

## Related Documentation

- **Performance budgets**: `docs/ai/AI_TIER_PERF_BUDGETS.md`
- **Fallback runbook**: `docs/runbooks/AI_FALLBACK.md`
- **SLO verification**: `docs/operations/SLO_VERIFICATION.md`
- **AI architecture**: `docs/architecture/AI_ARCHITECTURE.md`
- **Alert configuration**: `monitoring/prometheus/alerts.yml`

## Configuration Reference

**Environment variables** (see `src/server/config/env.ts`):

```bash
AI_SERVICE_URL=http://localhost:8765
AI_SERVICE_REQUEST_TIMEOUT_MS=5000
AI_RULES_REQUEST_TIMEOUT_MS=5000
AI_MAX_CONCURRENT_REQUESTS=16
```

**TypeScript config** (see `src/server/game/ai/AIEngine.ts`):

```typescript
export const AI_DIFFICULTY_PRESETS = {
  1: { aiType: 'random', thinkTime: 150, profileId: 'v1-random-1' },
  // ... see source for full table
  10: { aiType: 'descent', thinkTime: 16000, profileId: 'v1-descent-10' },
};
```
