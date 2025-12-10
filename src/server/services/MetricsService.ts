/**
 * MetricsService - Centralized Prometheus metrics collection for RingRift.
 *
 * This service provides:
 * - HTTP request metrics (duration, total, request/response sizes)
 * - Business metrics (games, moves, users)
 * - Service health metrics (service status, degradation level)
 * - AI service metrics (requests, duration, fallbacks)
 * - Rate limiting metrics
 * - WebSocket connection metrics
 *
 * All metrics are registered with the default prom-client registry and exposed
 * via the /metrics endpoint for Prometheus scraping.
 */

import client, { Registry, Counter, Histogram, Gauge } from 'prom-client';
import { logger } from '../utils/logger';
import { DegradationLevel, ServiceName, ServiceHealthStatus } from './ServiceStatusManager';
import type { RulesUxEventPayload } from '../../shared/telemetry/rulesUxEvents';
import type { DifficultyCalibrationEventPayload } from '../../shared/telemetry/difficultyCalibrationEvents';

/**
 * HTTP request method type for labels.
 */
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'OPTIONS' | 'HEAD' | string;

/**
 * Game type for business metrics.
 */
type GameType = 'pvp' | 'ai';

/**
 * Move type for business metrics.
 */
type MoveType = 'placement' | 'movement';

/**
 * AI request outcome for AI metrics.
 */
type AIOutcome = 'success' | 'fallback' | 'error';
export type AIChoiceOutcome = 'success' | 'fallback' | 'timeout' | 'error';

/**
 * Singleton MetricsService class that manages all Prometheus metrics.
 */
export class MetricsService {
  private static instance: MetricsService | null = null;
  private readonly registry: Registry;
  private initialized = false;

  // ===================
  // HTTP Request Metrics
  // ===================

  /** Histogram: HTTP request duration in seconds */
  public readonly httpRequestDuration: Histogram<'method' | 'path' | 'status'>;

  /** Counter: Total HTTP requests */
  public readonly httpRequestsTotal: Counter<'method' | 'path' | 'status'>;

  /** Histogram: HTTP request body size in bytes */
  public readonly httpRequestSizeBytes: Histogram<'method' | 'path'>;

  /** Histogram: HTTP response body size in bytes */
  public readonly httpResponseSizeBytes: Histogram<'method' | 'path' | 'status'>;

  // ===================
  // Business Metrics
  // ===================

  /** Counter: Total games created by type */
  public readonly gamesTotal: Counter<'type'>;

  /** Gauge: Currently active games */
  public readonly gamesActive: Gauge<string>;

  /** Counter: Total moves made by type */
  public readonly movesTotal: Counter<'type'>;

  /** Histogram: Game duration in seconds */
  public readonly gameDurationSeconds: Histogram<'type'>;

  /** Gauge: Currently connected users */
  public readonly usersActive: Gauge<string>;

  /** Gauge: Active WebSocket connections */
  public readonly websocketConnections: Gauge<string>;

  // ===================
  // Service Health Metrics
  // ===================

  /** Gauge: Per-service health status (0=down, 1=up) */
  public readonly serviceStatus: Gauge<'service'>;

  /** Gauge: Current degradation level (0=FULL, 1=DEGRADED, 2=MINIMAL, 3=OFFLINE) */
  public readonly degradationLevel: Gauge<string>;

  /** Histogram: Service call response times in seconds */
  public readonly serviceResponseTime: Histogram<'service'>;

  // ===================
  // AI Service Metrics
  // ===================

  /** Counter: AI service requests by outcome */
  public readonly aiRequestsTotal: Counter<'outcome'>;

  /** Histogram: AI request duration in seconds */
  public readonly aiRequestDuration: Histogram<'ai_type' | 'difficulty'>;

  /** Counter: AI fallback occurrences */
  public readonly aiFallbackTotal: Counter<'reason'>;

  // ===================
  // Rate Limiting Metrics
  // ===================

  /** Counter: Rate limit hits by endpoint and limiter type */
  public readonly rateLimitHitsTotal: Counter<'endpoint' | 'limiter'>;

  // ===================
  // Game Move Metrics
  // ===================

  /** Histogram: Game move processing latency */
  public readonly gameMoveLatency: Histogram<'board_type' | 'phase'>;

  // ===================
  // Game Session / AI Turn Metrics
  // ===================

  /** Gauge: Current in-memory game sessions by derived status kind */
  public readonly gameSessionStatusCurrent: Gauge<'status'>;

  /** Counter: Game session status transitions (from -> to) */
  public readonly gameSessionStatusTransitions: Counter<'from' | 'to'>;

  /** Counter: AI turn request terminal outcomes keyed by final state */
  public readonly aiTurnRequestTerminals: Counter<'kind' | 'code' | 'ai_error_type'>;

  /** Counter: WebSocket reconnection attempts */
  public readonly websocketReconnectionTotal: Counter<'result'>;

  /** Counter: Game session abnormal terminations by reason */
  public readonly gameSessionAbnormalTerminationTotal: Counter<'reason'>;

  /** Histogram: AI request latency in milliseconds */
  public readonly aiRequestLatencyMs: Histogram<'outcome'>;

  /** Counter: AI request timeout occurrences */
  public readonly aiRequestTimeoutTotal: Counter<string>;
  /** Counter: AI choice requests by choice type and outcome */
  public readonly aiChoiceRequestsTotal: Counter<'choice_type' | 'outcome'>;
  /** Histogram: AI choice latency in milliseconds by choice type and outcome */
  public readonly aiChoiceLatencyMs: Histogram<'choice_type' | 'outcome'>;

  // ===================
  // Move Rejection Metrics
  // ===================

  /** Counter: Total moves rejected by the server, labelled by reason */
  public readonly movesRejectedTotal: Counter<'reason'>;

  // ===================
  // Rules UX Telemetry Metrics
  // ===================

  /**
   * Counter: Lightweight rules-UX telemetry events emitted by the client.
   *
   * Labels are intentionally low-cardinality and derived from bounded enums:
   * - event_type: coarse event discriminant (help_open, weird_state_banner_impression, teaching_step_started, ...)
   * - rules_context: semantic rules concept (anm_forced_elimination, structural_stalemate, ...)
   * - source: emitting surface (hud, victory_modal, teaching_overlay, sandbox, ...)
   * - board_type: coarse board topology (square8, square19, hexagonal, ...)
   * - num_players: number of players/seats (1–4 or "unknown")
   * - difficulty: coarse difficulty bucket or AI level (tutorial, casual, ranked_low, 1–10, or "unknown")
   * - is_ranked: "true" / "false" / "unknown"
   * - is_sandbox: "true" / "false" / "unknown"
   */
  public readonly rulesUxEventsTotal: Counter<
    | 'event_type'
    | 'rules_context'
    | 'source'
    | 'board_type'
    | 'num_players'
    | 'difficulty'
    | 'is_ranked'
    | 'is_sandbox'
  >;

  // ===================
  // Difficulty Calibration Telemetry Metrics
  // ===================

  /**
   * Counter: AI difficulty calibration telemetry events emitted by the client.
   *
   * Labels are intentionally low-cardinality:
   * - type: event type (game_started, game_completed)
   * - board_type: coarse board topology (square8, square19, hexagonal, ...)
   * - num_players: number of players/seats (1–4)
   * - difficulty: primary AI difficulty bucket (1–10) or "unknown"
   * - result: coarse human result ('win' | 'loss' | 'draw' | 'abandoned' | 'none')
   */
  public readonly difficultyCalibrationEventsTotal: Counter<
    'type' | 'board_type' | 'num_players' | 'difficulty' | 'result'
  >;

  // ===================
  // Rules Parity Metrics (already exist, re-exported for convenience)
  // ===================

  /** Counter: TS vs Python rules validation verdict mismatch */
  public readonly rulesParityValidMismatch: Counter<string>;

  /** Counter: TS vs Python rules post-move state hash mismatch */
  public readonly rulesParityHashMismatch: Counter<string>;

  /** Counter: TS vs Python rules S-invariant mismatch */
  public readonly rulesParitySMismatch: Counter<string>;

  /** Counter: TS vs Python rules gameStatus mismatch */
  public readonly rulesParityGameStatusMismatch: Counter<string>;

  /**
   * Unified counter: TS vs Python rules mismatches by mismatch type and
   * suite/parity bucket. This provides a single surface for alerts and
   * dashboards while keeping the legacy per-dimension counters for
   * backwards-compatibility.
   *
   * Labels:
   * - mismatch_type: 'validation' | 'hash' | 's_invariant' | 'game_status' | ...
   * - suite: high-level parity bucket / PARITY-* ID (e.g. 'runtime_shadow',
   *   'runtime_python_mode', 'contract_vectors_v2').
   */
  public readonly rulesParityMismatchesTotal: Counter<'mismatch_type' | 'suite'>;

  // ===================
  // Orchestrator Rollout Metrics
  // ===================

  /** Counter: Total game sessions by engine selection and reason */
  public readonly orchestratorSessionsTotal: Counter<'engine' | 'selection_reason'>;

  /** Counter: Total moves processed by engine and outcome */
  public readonly orchestratorMovesTotal: Counter<'engine' | 'outcome'>;

  /** Gauge: Orchestrator circuit breaker state (0=closed, 1=open) */
  public readonly orchestratorCircuitBreakerState: Gauge<string>;

  /** Gauge: Current orchestrator error rate (0.0–1.0) */
  public readonly orchestratorErrorRate: Gauge<string>;

  /** Gauge: Configured orchestrator rollout percentage (0–100) */
  public readonly orchestratorRolloutPercentage: Gauge<string>;

  /** Counter: Total orchestrator-related invariant violations by type */
  public readonly orchestratorInvariantViolationsTotal: Counter<'type' | 'invariant_id'>;

  // ===================
  // Rules Correctness Dashboard Metrics
  // ===================

  /** Counter: Parity checks between TS and Python implementations */
  private parityChecks: Counter<'result'>;

  /** Gauge: Number of contract test vectors currently passing */
  private contractTestsPassing: Gauge<string>;

  /** Gauge: Total number of contract test vectors */
  private contractTestsTotal: Gauge<string>;

  /** Counter: Rules engine validation/mutation errors by type */
  private rulesErrors: Counter<'error_type'>;

  /** Histogram: Line detection processing time in milliseconds */
  private lineDetectionDuration: Histogram<string>;

  /** Histogram: Capture chain depth (segments per chain) */
  private captureChainDepth: Histogram<string>;

  // ===================
  // System Health Dashboard Metrics
  // ===================

  /** Counter: Redis cache hits */
  private cacheHits: Counter<string>;

  /** Counter: Redis cache misses */
  private cacheMisses: Counter<string>;

  /**
   * Private constructor - use getInstance() instead.
   */
  private constructor() {
    // Use the default registry
    this.registry = client.register;

    // ===================
    // HTTP Request Metrics
    // ===================

    this.httpRequestDuration = new Histogram({
      name: 'http_request_duration_seconds',
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['method', 'path', 'status'] as const,
      buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    });

    this.httpRequestsTotal = new Counter({
      name: 'http_requests_total',
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'path', 'status'] as const,
    });

    this.httpRequestSizeBytes = new Histogram({
      name: 'http_request_size_bytes',
      help: 'Size of HTTP request bodies in bytes',
      labelNames: ['method', 'path'] as const,
      buckets: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    });

    this.httpResponseSizeBytes = new Histogram({
      name: 'http_response_size_bytes',
      help: 'Size of HTTP response bodies in bytes',
      labelNames: ['method', 'path', 'status'] as const,
      buckets: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    });

    // ===================
    // Business Metrics
    // ===================

    this.gamesTotal = new Counter({
      name: 'ringrift_games_total',
      help: 'Total number of games created by type',
      labelNames: ['type'] as const,
    });

    this.gamesActive = new Gauge({
      name: 'ringrift_games_active',
      help: 'Number of currently active games',
    });

    this.movesTotal = new Counter({
      name: 'ringrift_moves_total',
      help: 'Total number of moves made by type',
      labelNames: ['type'] as const,
    });

    this.gameDurationSeconds = new Histogram({
      name: 'ringrift_game_duration_seconds',
      help: 'Duration of completed games in seconds',
      labelNames: ['type'] as const,
      buckets: [60, 120, 300, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200],
    });

    this.usersActive = new Gauge({
      name: 'ringrift_users_active',
      help: 'Number of currently connected users',
    });

    this.websocketConnections = new Gauge({
      name: 'ringrift_websocket_connections',
      help: 'Number of active WebSocket connections',
    });

    // ===================
    // Service Health Metrics
    // ===================

    this.serviceStatus = new Gauge({
      name: 'ringrift_service_status',
      help: 'Service availability status (0=down, 1=up)',
      labelNames: ['service'] as const,
    });

    this.degradationLevel = new Gauge({
      name: 'ringrift_degradation_level',
      help: 'Current system degradation level (0=FULL, 1=DEGRADED, 2=MINIMAL, 3=OFFLINE)',
    });

    this.serviceResponseTime = new Histogram({
      name: 'ringrift_service_response_time_seconds',
      help: 'Service call response times in seconds',
      labelNames: ['service'] as const,
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
    });

    // ===================
    // AI Service Metrics
    // ===================

    this.aiRequestsTotal = new Counter({
      name: 'ringrift_ai_requests_total',
      help: 'Total AI service requests by outcome',
      labelNames: ['outcome'] as const,
    });

    this.aiRequestDuration = new Histogram({
      name: 'ringrift_ai_request_duration_seconds',
      help: 'Duration of AI service requests in seconds',
      labelNames: ['ai_type', 'difficulty'] as const,
      buckets: [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8],
    });

    this.aiFallbackTotal = new Counter({
      name: 'ringrift_ai_fallback_total',
      help: 'Total number of AI fallbacks by reason',
      labelNames: ['reason'] as const,
    });

    // ===================
    // Rate Limiting Metrics
    // ===================

    this.rateLimitHitsTotal = new Counter({
      name: 'ringrift_rate_limit_hits_total',
      help: 'Total number of rate limit violations',
      labelNames: ['endpoint', 'limiter'] as const,
    });

    // ===================
    // Game Move Metrics
    // ===================

    this.gameMoveLatency = new Histogram({
      name: 'ringrift_game_move_latency_seconds',
      help: 'Game move processing latency in seconds',
      labelNames: ['board_type', 'phase'] as const,
      buckets: [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56],
    });

    // ===================
    // Game Session / AI Turn Metrics
    // ===================

    this.gameSessionStatusCurrent = new Gauge({
      name: 'ringrift_game_session_status_current',
      help: 'Number of in-memory game sessions by derived status kind',
      labelNames: ['status'] as const,
    });

    this.gameSessionStatusTransitions = new Counter({
      name: 'ringrift_game_session_status_transitions_total',
      help: 'Total number of game session status transitions (from -> to)',
      labelNames: ['from', 'to'] as const,
    });

    this.aiTurnRequestTerminals = new Counter({
      name: 'ringrift_ai_turn_request_terminal_total',
      help: 'Total number of AI turn requests reaching a terminal state',
      labelNames: ['kind', 'code', 'ai_error_type'] as const,
    });

    this.websocketReconnectionTotal = new Counter({
      name: 'ringrift_websocket_reconnection_total',
      help: 'Total number of WebSocket reconnection attempts by result',
      labelNames: ['result'] as const,
    });

    this.gameSessionAbnormalTerminationTotal = new Counter({
      name: 'ringrift_game_session_abnormal_termination_total',
      help: 'Total number of game sessions terminated abnormally by reason',
      labelNames: ['reason'] as const,
    });

    this.aiRequestLatencyMs = new Histogram({
      name: 'ringrift_ai_request_latency_ms',
      help: 'AI request latency in milliseconds',
      labelNames: ['outcome'] as const,
      buckets: [50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
    });

    this.aiRequestTimeoutTotal = new Counter({
      name: 'ringrift_ai_request_timeout_total',
      help: 'Total number of AI request timeouts',
    });
    this.aiChoiceRequestsTotal = new Counter({
      name: 'ringrift_ai_choice_requests_total',
      help: 'Total number of AI choice requests by outcome and choice type',
      labelNames: ['choice_type', 'outcome'] as const,
    });
    this.aiChoiceLatencyMs = new Histogram({
      name: 'ringrift_ai_choice_latency_ms',
      help: 'AI choice latency in milliseconds by choice type and outcome',
      labelNames: ['choice_type', 'outcome'] as const,
      buckets: [50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
    });

    // ===================
    // Move Rejection Metrics
    // ===================

    this.movesRejectedTotal = new Counter({
      name: 'ringrift_moves_rejected_total',
      help: 'Total number of moves rejected by the server, labelled by reason',
      labelNames: ['reason'] as const,
    });

    // ===================
    // Rules UX Telemetry Metrics
    // ===================

    this.rulesUxEventsTotal = new Counter({
      name: 'ringrift_rules_ux_events_total',
      help: 'Total number of rules-UX telemetry events by type and coarse context',
      labelNames: [
        'event_type',
        'rules_context',
        'source',
        'board_type',
        'num_players',
        'difficulty',
        'is_ranked',
        'is_sandbox',
      ] as const,
    });

    // ===================
    // Difficulty Calibration Telemetry Metrics
    // ===================

    this.difficultyCalibrationEventsTotal = new Counter({
      name: 'ringrift_difficulty_calibration_events_total',
      help: 'Total number of AI difficulty calibration telemetry events by type and coarse context',
      labelNames: ['type', 'board_type', 'num_players', 'difficulty', 'result'] as const,
    });

    // ===================
    // Rules Parity Metrics
    // ===================

    this.rulesParityValidMismatch = new Counter({
      name: 'ringrift_rules_parity_valid_mismatch_total',
      help: 'TS vs Python rules: validation verdict mismatch count',
    });

    this.rulesParityHashMismatch = new Counter({
      name: 'ringrift_rules_parity_hash_mismatch_total',
      help: 'TS vs Python rules: post-move state hash mismatch count',
    });

    this.rulesParitySMismatch = new Counter({
      name: 'ringrift_rules_parity_s_mismatch_total',
      help: 'TS vs Python rules: S-invariant mismatch count',
    });

    this.rulesParityGameStatusMismatch = new Counter({
      name: 'ringrift_rules_parity_game_status_mismatch_total',
      help: 'TS vs Python rules: gameStatus mismatch count',
    });

    this.rulesParityMismatchesTotal = new Counter({
      name: 'ringrift_rules_parity_mismatches_total',
      help: 'Unified TS vs Python rules mismatches by mismatch type and suite',
      labelNames: ['mismatch_type', 'suite'] as const,
    });

    // ===================
    // Orchestrator Shadow Mode Metrics
    // ===================

    this.orchestratorShadowComparisonsCurrent = new Gauge({
      name: 'ringrift_orchestrator_shadow_comparisons_current',
      help: 'Current number of shadow mode comparisons held in memory',
    });

    this.orchestratorShadowMismatchesCurrent = new Gauge({
      name: 'ringrift_orchestrator_shadow_mismatches_current',
      help: 'Current number of shadow mode mismatches held in memory',
    });

    this.orchestratorShadowMismatchRate = new Gauge({
      name: 'ringrift_orchestrator_shadow_mismatch_rate',
      help: 'Current mismatch rate for orchestrator shadow comparisons (0.0–1.0)',
    });

    this.orchestratorShadowOrchestratorErrorsCurrent = new Gauge({
      name: 'ringrift_orchestrator_shadow_orchestrator_errors_current',
      help: 'Current number of orchestrator errors in shadow comparisons',
    });

    this.orchestratorShadowOrchestratorErrorRate = new Gauge({
      name: 'ringrift_orchestrator_shadow_orchestrator_error_rate',
      help: 'Current orchestrator error rate in shadow comparisons (0.0–1.0)',
    });

    this.orchestratorShadowAvgLegacyLatencyMs = new Gauge({
      name: 'ringrift_orchestrator_shadow_avg_legacy_latency_ms',
      help: 'Average legacy engine latency across shadow comparisons (milliseconds, rolling window)',
    });

    this.orchestratorShadowAvgOrchestratorLatencyMs = new Gauge({
      name: 'ringrift_orchestrator_shadow_avg_orchestrator_latency_ms',
      help: 'Average orchestrator latency across shadow comparisons (milliseconds, rolling window)',
    });

    // ===================
    // Orchestrator Rollout Metrics
    // ===================

    this.orchestratorSessionsTotal = new Counter({
      name: 'ringrift_orchestrator_sessions_total',
      help: 'Total number of game sessions by engine selection and reason',
      labelNames: ['engine', 'selection_reason'] as const,
    });

    this.orchestratorMovesTotal = new Counter({
      name: 'ringrift_orchestrator_moves_total',
      help: 'Total number of moves processed by engine and outcome',
      labelNames: ['engine', 'outcome'] as const,
    });

    this.orchestratorCircuitBreakerState = new Gauge({
      name: 'ringrift_orchestrator_circuit_breaker_state',
      help: 'Orchestrator circuit breaker state (0=closed, 1=open)',
    });

    this.orchestratorErrorRate = new Gauge({
      name: 'ringrift_orchestrator_error_rate',
      help: 'Current orchestrator error rate (0.0–1.0)',
    });

    this.orchestratorRolloutPercentage = new Gauge({
      name: 'ringrift_orchestrator_rollout_percentage',
      help: 'Configured orchestrator rollout percentage (0–100)',
    });

    this.orchestratorInvariantViolationsTotal = new Counter({
      name: 'ringrift_orchestrator_invariant_violations_total',
      help: 'Total number of orchestrator-related invariant violations observed by backend hosts',
      // NOTE: `type` is a low-level violation identifier (e.g. S_INVARIANT_DECREASED),
      // while `invariant_id` is a high-level invariant from
      // docs/INVARIANTS_AND_PARITY_FRAMEWORK.md (e.g. INV-S-MONOTONIC).
      labelNames: ['type', 'invariant_id'] as const,
    });

    // ===================
    // Rules Correctness Dashboard Metrics
    // ===================

    this.parityChecks = new Counter({
      name: 'ringrift_parity_checks_total',
      help: 'Total parity checks between TS and Python implementations',
      labelNames: ['result'] as const,
    });

    this.contractTestsPassing = new Gauge({
      name: 'ringrift_contract_tests_passing',
      help: 'Number of contract test vectors currently passing',
    });

    this.contractTestsTotal = new Gauge({
      name: 'ringrift_contract_tests_total',
      help: 'Total number of contract test vectors',
    });

    this.rulesErrors = new Counter({
      name: 'ringrift_rules_errors_total',
      help: 'Total rules engine validation/mutation errors',
      labelNames: ['error_type'] as const,
    });

    this.lineDetectionDuration = new Histogram({
      name: 'ringrift_line_detection_duration_ms',
      help: 'Line detection processing time in milliseconds',
      buckets: [10, 25, 50, 100, 250, 500, 1000],
    });

    this.captureChainDepth = new Histogram({
      name: 'ringrift_capture_chain_depth',
      help: 'Distribution of capture chain depths (number of segments in a chain)',
      buckets: [1, 2, 3, 4, 5, 6, 8, 10],
    });

    // ===================
    // System Health Dashboard Metrics
    // ===================

    this.cacheHits = new Counter({
      name: 'ringrift_cache_hits_total',
      help: 'Total Redis cache hits',
    });

    this.cacheMisses = new Counter({
      name: 'ringrift_cache_misses_total',
      help: 'Total Redis cache misses',
    });

    this.initialized = true;
    logger.info('MetricsService initialized');
  }

  /**
   * Get the singleton MetricsService instance.
   */
  public static getInstance(): MetricsService {
    if (!MetricsService.instance) {
      MetricsService.instance = new MetricsService();
    }
    return MetricsService.instance;
  }

  /**
   * Reset the singleton instance (for testing only).
   */
  public static resetInstance(): void {
    if (MetricsService.instance) {
      // Clear all metrics from the registry
      client.register.clear();
      MetricsService.instance = null;
    }
  }

  /**
   * Get metrics in Prometheus text format.
   */
  public async getMetrics(): Promise<string> {
    // Refresh orchestrator shadow-mode gauges before snapshotting the registry
    // so that /metrics reflects the latest comparison statistics.
    this.refreshOrchestratorShadowMetrics();
    return this.registry.metrics();
  }

  /**
   * Get the content type for metrics response.
   */
  public getContentType(): string {
    return this.registry.contentType;
  }

  /**
   * Check if the service is initialized.
   */
  public isInitialized(): boolean {
    return this.initialized;
  }

  // ===================
  // HTTP Request Helpers
  // ===================

  /**
   * Record an HTTP request with all metrics.
   */
  public recordHttpRequest(
    method: HttpMethod,
    path: string,
    statusCode: number,
    durationSeconds: number,
    requestSizeBytes?: number,
    responseSizeBytes?: number
  ): void {
    const normalizedPath = this.normalizePath(path);
    const status = String(statusCode);

    this.httpRequestsTotal.labels(method, normalizedPath, status).inc();
    this.httpRequestDuration.labels(method, normalizedPath, status).observe(durationSeconds);

    if (requestSizeBytes !== undefined) {
      this.httpRequestSizeBytes.labels(method, normalizedPath).observe(requestSizeBytes);
    }

    if (responseSizeBytes !== undefined) {
      this.httpResponseSizeBytes.labels(method, normalizedPath, status).observe(responseSizeBytes);
    }
  }

  /**
   * Normalize URL paths to prevent high cardinality.
   * Replaces dynamic segments with placeholders.
   */
  private normalizePath(path: string): string {
    // Remove query strings
    const pathWithoutQuery = path.split('?')[0];

    // Normalize common dynamic path patterns
    return (
      pathWithoutQuery
        // UUID patterns (with or without hyphens)
        .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi, ':id')
        .replace(/[0-9a-f]{32}/gi, ':id')
        // Generic ID patterns (numeric IDs, MongoDB ObjectIds, etc.)
        .replace(/\/[0-9]+(?=\/|$)/g, '/:id')
        .replace(/\/[0-9a-f]{24}(?=\/|$)/gi, '/:id')
        // General alphanumeric IDs (e.g., short game IDs)
        .replace(/\/[a-zA-Z0-9]{8,}(?=\/|$)/g, '/:id')
    );
  }

  // ===================
  // Business Metric Helpers
  // ===================

  /**
   * Record a new game creation.
   */
  public recordGameCreated(type: GameType): void {
    this.gamesTotal.labels(type).inc();
    this.gamesActive.inc();
  }

  /**
   * Record a game ending.
   */
  public recordGameEnded(type: GameType, durationSeconds: number): void {
    this.gamesActive.dec();
    this.gameDurationSeconds.labels(type).observe(durationSeconds);
  }

  /**
   * Set the number of active games (for initialization or sync).
   */
  public setActiveGames(count: number): void {
    this.gamesActive.set(count);
  }

  /**
   * Record a move made.
   */
  public recordMove(type: MoveType): void {
    this.movesTotal.labels(type).inc();
  }

  /**
   * Set the number of active users.
   */
  public setActiveUsers(count: number): void {
    this.usersActive.set(count);
  }

  /**
   * Set the number of WebSocket connections.
   */
  public setWebSocketConnections(count: number): void {
    this.websocketConnections.set(count);
  }

  /**
   * Increment WebSocket connections.
   */
  public incWebSocketConnections(): void {
    this.websocketConnections.inc();
  }

  /**
   * Decrement WebSocket connections.
   */
  public decWebSocketConnections(): void {
    this.websocketConnections.dec();
  }

  // ===================
  // Service Health Helpers
  // ===================

  /**
   * Update service status metric.
   */
  public updateServiceStatus(service: ServiceName, status: ServiceHealthStatus): void {
    const value = status === 'healthy' ? 1 : 0;
    this.serviceStatus.labels(service).set(value);
  }

  /**
   * Update degradation level metric.
   */
  public updateDegradationLevel(level: DegradationLevel): void {
    const levelMap: Record<DegradationLevel, number> = {
      [DegradationLevel.FULL]: 0,
      [DegradationLevel.DEGRADED]: 1,
      [DegradationLevel.MINIMAL]: 2,
      [DegradationLevel.OFFLINE]: 3,
    };
    this.degradationLevel.set(levelMap[level] ?? 0);
  }

  /**
   * Record service response time.
   */
  public recordServiceResponseTime(service: ServiceName, durationSeconds: number): void {
    this.serviceResponseTime.labels(service).observe(durationSeconds);
  }

  // ===================
  // AI Service Helpers
  // ===================

  /**
   * Record an AI request outcome.
   */
  public recordAIRequest(outcome: AIOutcome): void {
    this.aiRequestsTotal.labels(outcome).inc();
  }

  /**
   * Record AI request duration.
   */
  public recordAIRequestDuration(
    aiType: string,
    difficulty: string | number,
    durationSeconds: number
  ): void {
    this.aiRequestDuration.labels(aiType, String(difficulty)).observe(durationSeconds);
  }

  /**
   * Record an AI fallback.
   */
  public recordAIFallback(reason: string): void {
    this.aiFallbackTotal.labels(reason).inc();
  }

  // ===================
  // Rate Limiting Helpers
  // ===================

  /**
   * Record a rate limit hit.
   */
  public recordRateLimitHit(endpoint: string, limiter: string): void {
    this.rateLimitHitsTotal.labels(endpoint, limiter).inc();
  }

  // ===================
  // Game Move Helpers
  // ===================

  /**
   * Record game move processing latency.
   */
  public recordGameMoveLatency(boardType: string, phase: string, durationSeconds: number): void {
    this.gameMoveLatency.labels(boardType, phase).observe(durationSeconds);
  }

  // ===================
  // Game Session / AI Turn Helpers
  // ===================

  /**
   * Record a transition between two high-level GameSessionStatus kinds.
   *
   * The `from` label uses a sentinel value of `none` for initial projections
   * when no prior status was available.
   */
  public recordGameSessionStatusTransition(from: string, to: string): void {
    this.gameSessionStatusTransitions.labels(from, to).inc();
  }

  /**
   * Update the gauge that tracks the current number of in-memory game
   * sessions by derived status kind. Callers should pass `null` for
   * `from` when initializing a session and `null` for `to` when a
   * session is being torn down.
   */
  public updateGameSessionStatusCurrent(from: string | null, to: string | null): void {
    if (from) {
      this.gameSessionStatusCurrent.labels(from).dec();
    }
    if (to) {
      this.gameSessionStatusCurrent.labels(to).inc();
    }
  }

  /**
   * Record the terminal outcome for an AI turn request as modeled by
   * AIRequestState. For non-failed terminals, `code` and `ai_error_type`
   * should typically be omitted.
   */
  public recordAITurnRequestTerminal(kind: string, code?: string, aiErrorType?: string): void {
    this.aiTurnRequestTerminals.labels(kind, code ?? 'none', aiErrorType ?? 'none').inc();
  }

  // ===================
  // WebSocket Session Helpers
  // ===================

  /**
   * Record a WebSocket reconnection attempt.
   */
  public recordWebsocketReconnection(result: 'success' | 'failed' | 'timeout'): void {
    this.websocketReconnectionTotal.labels(result).inc();
  }

  // ===================
  // Abnormal Termination Helpers
  // ===================

  /**
   * Record an abnormal game session termination.
   */
  public recordAbnormalTermination(reason: string): void {
    this.gameSessionAbnormalTerminationTotal.labels(reason).inc();
  }

  // ===================
  // AI Request Latency Helpers
  // ===================

  /**
   * Record AI request latency in milliseconds.
   */
  public recordAIRequestLatencyMs(
    latencyMs: number,
    outcome: 'success' | 'fallback' | 'timeout' | 'error'
  ): void {
    this.aiRequestLatencyMs.labels(outcome).observe(latencyMs);
  }

  /**
   * Record an AI request timeout.
   */
  public recordAIRequestTimeout(): void {
    this.aiRequestTimeoutTotal.inc();
  }

  /**
   * Record an AI choice request outcome for a specific choice type.
   */
  public recordAIChoiceRequest(choiceType: string, outcome: AIChoiceOutcome): void {
    this.aiChoiceRequestsTotal.labels(choiceType, outcome).inc();
  }

  /**
   * Record AI choice latency in milliseconds for a specific choice type.
   */
  public recordAIChoiceLatencyMs(
    choiceType: string,
    latencyMs: number,
    outcome: AIChoiceOutcome
  ): void {
    this.aiChoiceLatencyMs.labels(choiceType, outcome).observe(latencyMs);
  }

  // ===================
  // Orchestrator Shadow Mode Helpers
  // ===================

  /**
   * Refresh the orchestrator shadow-mode gauges from the latest
   * ShadowModeComparator snapshot. This is called from getMetrics()
   * so that /metrics always exposes up-to-date comparison statistics.
   */
  public refreshOrchestratorShadowMetrics(): void {
    let snapshot: ShadowMetrics;
    try {
      snapshot = shadowComparator.getMetrics();
    } catch (err) {
      logger.warn('Failed to refresh orchestrator shadow metrics', {
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }

    this.orchestratorShadowComparisonsCurrent.set(snapshot.totalComparisons);
    this.orchestratorShadowMismatchesCurrent.set(snapshot.mismatches);
    this.orchestratorShadowMismatchRate.set(snapshot.mismatchRate);
    this.orchestratorShadowOrchestratorErrorsCurrent.set(snapshot.orchestratorErrors);
    this.orchestratorShadowOrchestratorErrorRate.set(snapshot.orchestratorErrorRate);
    this.orchestratorShadowAvgLegacyLatencyMs.set(snapshot.avgLegacyLatencyMs);
    this.orchestratorShadowAvgOrchestratorLatencyMs.set(snapshot.avgOrchestratorLatencyMs);
  }

  // ===================
  // Orchestrator Rollout Helpers
  // ===================

  /**
   * Record a new game session selection for orchestrator rollout metrics.
   */
  public recordOrchestratorSession(engine: string, selectionReason: string): void {
    this.orchestratorSessionsTotal.labels(engine, selectionReason).inc();
  }

  /**
   * Record a move processed by a given engine and outcome.
   */
  public recordOrchestratorMove(
    engine: 'legacy' | 'orchestrator',
    outcome: 'success' | 'error'
  ): void {
    this.orchestratorMovesTotal.labels(engine, outcome).inc();
  }

  /**
   * Update the circuit breaker state gauge (0=closed, 1=open).
   */
  public setOrchestratorCircuitBreakerState(isOpen: boolean): void {
    this.orchestratorCircuitBreakerState.set(isOpen ? 1 : 0);
  }

  /**
   * Update the orchestrator error-rate gauge (0.0–1.0).
   */
  public setOrchestratorErrorRate(rate: number): void {
    this.orchestratorErrorRate.set(rate);
  }

  /**
   * Set the configured orchestrator rollout percentage (0–100).
   */
  public setOrchestratorRolloutPercentage(percent: number): void {
    this.orchestratorRolloutPercentage.set(percent);
  }

  /**
   * Record a rules parity mismatch using the unified counter. This is the
   * preferred surface for alerts and dashboards; the legacy
   * ringrift_rules_parity_*_mismatch_total counters remain available for
   * backwards-compatibility and coarse-grained dashboards.
   *
   * @param mismatchType - High-level mismatch category (e.g. 'validation',
   *   'hash', 's_invariant', 'game_status').
   * @param suite - Parity bucket / PARITY-* ID or runtime context
   *   (e.g. 'runtime_shadow', 'runtime_python_mode', 'contract_vectors_v2').
   */
  public recordRulesParityMismatch(mismatchType: string, suite: string): void {
    this.rulesParityMismatchesTotal.labels(mismatchType, suite).inc();
  }

  /**
   * Map a low-level violation type string to a high-level invariant ID from
   * docs/INVARIANTS_AND_PARITY_FRAMEWORK.md. This keeps metric labels stable
   * even if internal violation IDs evolve.
   */
  private mapInvariantTypeToId(type: string): string {
    switch (type) {
      // S-invariant and elimination monotonicity
      case 'S_INVARIANT_DECREASED':
        return 'INV-S-MONOTONIC';
      case 'TOTAL_RINGS_ELIMINATED_DECREASED':
        return 'INV-ELIMINATION-MONOTONIC';

      // Active player has at least one legal action
      case 'ACTIVE_NO_MOVES':
      case 'ACTIVE_NO_CANDIDATE_MOVES':
        return 'INV-ACTIVE-NO-MOVES';

      // Structural board/state invariants
      case 'NEGATIVE_STACK_HEIGHT':
      case 'STACK_HEIGHT_MISMATCH':
      case 'INVALID_CAP_HEIGHT':
      case 'NEGATIVE_ELIMINATED_RINGS':
        return 'INV-STATE-STRUCTURAL';

      // Orchestrator vs host move validation
      case 'ORCHESTRATOR_VALIDATE_MOVE_FAILED':
      case 'HOST_REJECTED_MOVE':
        return 'INV-ORCH-VALIDATION';

      // GameStatus / termination anomalies
      case 'UNEXPECTED_GAME_STATUS':
      case 'UNHANDLED_EXCEPTION':
        return 'INV-TERMINATION';

      default:
        // Fallback: treat unknown types as termination-related until they are
        // explicitly mapped. This preserves metric emission while keeping the
        // label space small and stable.
        return 'INV-TERMINATION';
    }
  }

  /**
   * Record an orchestrator-related invariant violation of a given type.
   * The environment dimension is provided via Prometheus external_labels.
   *
   * `type` is a low-level violation identifier (e.g. S_INVARIANT_DECREASED).
   * `invariantId` is an optional high-level invariant ID (e.g. INV-S-MONOTONIC);
   * when omitted, it is derived from `type` using mapInvariantTypeToId.
   */
  public recordOrchestratorInvariantViolation(type: string, invariantId?: string): void {
    const resolvedInvariantId = invariantId ?? this.mapInvariantTypeToId(type);
    this.orchestratorInvariantViolationsTotal.labels(type, resolvedInvariantId).inc();
  }

  // ===================
  // Rules Correctness Metrics Helpers
  // ===================

  /**
   * Record a parity check between TS and Python implementations.
   * Used to track consistency validation between the two rule engines.
   *
   * @param success - Whether the parity check passed (true) or failed (false)
   */
  public recordParityCheck(success: boolean): void {
    this.parityChecks.inc({ result: success ? 'success' : 'failure' });
  }

  /**
   * Update contract test metrics with current pass/total counts.
   * Provides visibility into contract vector validation coverage.
   *
   * @param passing - Number of contract test vectors currently passing
   * @param total - Total number of contract test vectors
   */
  public updateContractTestMetrics(passing: number, total: number): void {
    this.contractTestsPassing.set(passing);
    this.contractTestsTotal.set(total);
  }

  /**
   * Record a rules engine error during validation or mutation.
   * Tracks different types of errors for troubleshooting and alerts.
   *
   * @param errorType - Type of error: 'validation', 'mutation', or 'internal'
   */
  public recordRulesError(errorType: 'validation' | 'mutation' | 'internal'): void {
    this.rulesErrors.inc({ error_type: errorType });
  }

  /**
   * Record line detection processing duration.
   * Tracks performance of line detection algorithms in the rules engine.
   *
   * @param durationMs - Processing time in milliseconds
   */
  public recordLineDetection(durationMs: number): void {
    this.lineDetectionDuration.observe(durationMs);
  }

  /**
   * Record a capture chain depth sample for analytics.
   * Tracks how many segments typical chain captures contain.
   *
   * @param depth - Number of segments in a single capture chain (>= 1)
   */
  public recordCaptureChainDepth(depth: number): void {
    if (depth > 0) {
      this.captureChainDepth.observe(depth);
    }
  }

  /**
   * Record a rejected move and its high-level reason.
   *
   * @param reason - Short, stable reason identifier (e.g. 'rules_invalid', 'invalid_payload',
   *                 'authz', 'game_not_active', 'db_unavailable', 'decision_timeout_auto_rejected').
   */
  public recordMoveRejected(reason: string): void {
    this.movesRejectedTotal.inc({ reason });
  }

  // ===================
  // Rules UX Telemetry Helpers
  // ===================

  /**
   * Record a single rules-UX telemetry event.
   *
   * The payload is intentionally low-cardinality and omits any user-identifying
   * information; labels are normalised and bucketed to keep the metrics
   * surface stable over time.
   */
  public recordRulesUxEvent(event: RulesUxEventPayload): void {
    const eventType = event.type || 'unknown';

    const boardType = (event.boardType as string) || 'unknown';

    const numPlayersRaw =
      typeof event.numPlayers === 'number' && Number.isFinite(event.numPlayers)
        ? event.numPlayers
        : NaN;
    const numPlayers = numPlayersRaw >= 1 && numPlayersRaw <= 4 ? String(numPlayersRaw) : 'unknown';

    // Normalised rules_context: semantic rules concept or "none".
    const rulesContextLabel =
      typeof event.rulesContext === 'string' && event.rulesContext.length > 0
        ? event.rulesContext.slice(0, 64)
        : 'none';

    // Normalised source: emitting surface or "unknown".
    const sourceLabel =
      typeof event.source === 'string' && event.source.length > 0
        ? event.source.slice(0, 64)
        : 'unknown';

    // Difficulty: prefer explicit difficulty bucket, fall back to aiDifficulty (1–10),
    // otherwise "unknown".
    let difficultyLabel = 'unknown';
    if (typeof event.difficulty === 'string' && event.difficulty.length > 0) {
      difficultyLabel = event.difficulty.slice(0, 64);
    } else if (typeof event.aiDifficulty === 'number' && Number.isFinite(event.aiDifficulty)) {
      const clamped = Math.min(10, Math.max(1, Math.round(event.aiDifficulty)));
      difficultyLabel = String(clamped);
    }

    const isRankedLabel =
      typeof event.isRanked === 'boolean' ? (event.isRanked ? 'true' : 'false') : 'unknown';

    const isSandboxLabel =
      typeof event.isSandbox === 'boolean' ? (event.isSandbox ? 'true' : 'false') : 'unknown';

    this.rulesUxEventsTotal
      .labels(
        eventType,
        rulesContextLabel,
        sourceLabel,
        boardType,
        numPlayers,
        difficultyLabel,
        isRankedLabel,
        isSandboxLabel
      )
      .inc();
  }

  /**
   * Record a single AI difficulty calibration telemetry event.
   *
   * The payload is intentionally low-cardinality and omits any user-identifying
   * information; labels are normalised and bucketed to keep the metrics
   * surface stable over time. Only events where isCalibrationOptIn=true are
   * recorded so that organic games do not pollute calibration series.
   */
  public recordDifficultyCalibrationEvent(event: DifficultyCalibrationEventPayload): void {
    if (!event.isCalibrationOptIn) {
      // Only record explicit calibration opt-in games. Clients may still choose
      // to send events for non-opt-in games, but these are ignored at the
      // metrics layer to keep semantics tight.
      return;
    }

    const typeLabel = event.type || 'unknown';

    const boardType = (event.boardType as string) || 'unknown';

    const numPlayersRaw =
      typeof event.numPlayers === 'number' && Number.isFinite(event.numPlayers)
        ? event.numPlayers
        : NaN;
    const numPlayers = numPlayersRaw >= 1 && numPlayersRaw <= 4 ? String(numPlayersRaw) : 'unknown';

    const difficultyRaw =
      typeof event.difficulty === 'number' && Number.isFinite(event.difficulty)
        ? event.difficulty
        : NaN;
    const difficulty =
      difficultyRaw >= 1 && difficultyRaw <= 10 ? String(Math.round(difficultyRaw)) : 'unknown';

    let result = 'none';
    if (
      event.result === 'win' ||
      event.result === 'loss' ||
      event.result === 'draw' ||
      event.result === 'abandoned'
    ) {
      result = event.result;
    }

    this.difficultyCalibrationEventsTotal
      .labels(typeLabel, boardType, numPlayers, difficulty, result)
      .inc();
  }

  // ===================
  // System Health Metrics Helpers
  // ===================

  /**
   * Record a Redis cache hit.
   * Tracks successful cache retrievals for monitoring cache effectiveness.
   */
  public recordCacheHit(): void {
    this.cacheHits.inc();
  }

  /**
   * Record a Redis cache miss.
   * Tracks cache misses to monitor cache effectiveness and identify patterns.
   */
  public recordCacheMiss(): void {
    this.cacheMisses.inc();
  }
}

// Singleton getter for convenience
export const getMetricsService = (): MetricsService => MetricsService.getInstance();

// Default export for simpler imports
export default MetricsService;
