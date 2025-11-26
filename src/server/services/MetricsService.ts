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

import client, { Registry, Counter, Histogram, Gauge, Summary } from 'prom-client';
import { logger } from '../utils/logger';
import { DegradationLevel, ServiceName, ServiceHealthStatus } from './ServiceStatusManager';

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
}

// Singleton getter for convenience
export const getMetricsService = (): MetricsService => MetricsService.getInstance();

// Default export for simpler imports
export default MetricsService;
