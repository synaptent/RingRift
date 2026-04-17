import client from 'prom-client';
import { logger } from './logger';
import { getMetricsService } from '../services/MetricsService';

/**
 * Prometheus counters for TS <-> Python rules-engine parity.
 *
 * These are incremented by the RulesBackendFacade whenever it observes
 * a discrepancy between the TS GameEngine and the Python rules service
 * during python-authoritative diagnostics or explicit parity jobs.
 *
 * The concrete counters are now provided by MetricsService so that:
 * - Metric names are RingRift-prefixed (`ringrift_rules_parity_*`) and
 *   aligned with alert rules in monitoring/prometheus/alerts.yml.
 * - All rules-parity metrics share the same default prom-client registry
 *   and /metrics surface as the rest of the application.
 */
const metricsService = getMetricsService();

export const rulesParityMetrics = {
  validMismatch: metricsService.rulesParityValidMismatch,
  hashMismatch: metricsService.rulesParityHashMismatch,
  sMismatch: metricsService.rulesParitySMismatch,
  gameStatusMismatch: metricsService.rulesParityGameStatusMismatch,
};

/**
 * Core application-wide Prometheus metrics for AI, move processing, and WebSockets.
 * These share the default Node.js registry and are exported individually so
 * callers can import only what they need without re-configuring collectors.
 */
export const aiMoveLatencyHistogram = new client.Histogram({
  name: 'ai_move_latency_ms',
  help: 'Latency of AI move selection calls in milliseconds',
  labelNames: ['aiType', 'difficulty'] as const,
  buckets: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400],
});

export const aiFallbackCounter = new client.Counter({
  name: 'ai_fallback_total',
  help: 'Total number of AI fallbacks by reason',
  labelNames: ['reason'] as const,
});

/**
 * Richer fallback counter (D5 / issue #82). Emitted alongside the existing
 * `ai_fallback_total` so downstream dashboards are not disturbed; new
 * alerting paths key off this metric because it includes the difficulty
 * tier and intended AI type.
 *
 * The `ai_type` label carries the tier's CONFIGURED AI type (what the
 * request would have used if the service call had succeeded) — not the
 * eventual fallback AI type. This makes it straightforward to alert on
 * "D10 should be Gumbel MCTS, not local heuristic."
 */
export const aiFallbackMovesCounter = new client.Counter({
  name: 'ai_fallback_moves_total',
  help:
    'Total /ai/move requests that fell back to local heuristic or random, ' +
    'labeled by the classified reason, configured AI type, and difficulty tier.',
  labelNames: ['reason', 'ai_type', 'difficulty'] as const,
});

/**
 * Circuit-breaker state observability. 0 = closed, 1 = open, 0.5 = half-open.
 * Alertmanager can key off `max_over_time(ai_circuit_breaker_state[5m]) == 1`
 * to detect a sustained open state.
 */
export const aiCircuitBreakerStateGauge = new client.Gauge({
  name: 'ai_circuit_breaker_state',
  help:
    'Current state of the AI service circuit breaker: 0=closed, ' +
    '0.5=half-open, 1=open. Derived from AIServiceClient.CircuitBreaker.',
});

/**
 * Count transitions between circuit-breaker states so we can distinguish a
 * flapping breaker from a stably-open one.
 */
export const aiCircuitBreakerTransitionsCounter = new client.Counter({
  name: 'ai_circuit_breaker_transitions_total',
  help: 'Total AI circuit-breaker state transitions, labeled by from/to state.',
  labelNames: ['from_state', 'to_state'] as const,
});

export const gameMoveLatencyHistogram = new client.Histogram({
  name: 'game_move_latency_ms',
  help: 'Latency of game move processing in milliseconds',
  labelNames: ['boardType', 'phase'] as const,
  buckets: [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560],
});

export const webSocketConnectionsGauge = new client.Gauge({
  name: 'websocket_connections_current',
  help: 'Current number of active WebSocket connections',
});

/**
 * Structured logging helper for rules parity discrepancies.
 *
 * All parity-related logs are emitted with a common message key so that
 * they can be discovered and aggregated easily in log pipelines.
 */
export function logRulesMismatch(
  kind: 'valid' | 'hash' | 'S' | 'gameStatus' | 'backend_fallback' | 'shadow_error',
  details: Record<string, unknown>
): void {
  logger.warn('rules_parity_mismatch', {
    kind,
    ...details,
  });
}

/**
 * Unified helper for recording TS <-> Python rules mismatches on the
 * ringrift_rules_parity_mismatches_total counter. This provides a single
 * metric surface keyed by mismatch_type and suite/parity bucket while
 * preserving the existing per-dimension counters for backwards-compatibility.
 *
 * Typical mismatchType values:
 * - 'validation'  – verdict mismatch (valid vs invalid)
 * - 'hash'        – post-move state hash mismatch
 * - 's_invariant' – S-invariant mismatch
 * - 'game_status' – gameStatus / victory mismatch
 *
 * Typical suite values:
 * - 'runtime_python_mode'   – python-authoritative runtime diagnostics
 * - 'runtime_ts'            – TS-authoritative sanity checks
 * - 'contract_vectors_v2'   – contract-vector based parity jobs (future use)
 * - 'runtime_shadow'        – legacy label kept for historical fixtures only
 */
export function recordRulesParityMismatch(params: {
  mismatchType: 'validation' | 'hash' | 's_invariant' | 'game_status';
  suite: string;
}): void {
  metricsService.recordRulesParityMismatch(params.mismatchType, params.suite);
}
