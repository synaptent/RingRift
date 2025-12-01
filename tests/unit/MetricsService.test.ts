/**
 * Unit tests for MetricsService.
 *
 * Tests the Prometheus metrics collection service including:
 * - Singleton pattern
 * - HTTP request metrics
 * - Business metrics (games, moves, users)
 * - Service health metrics
 * - AI service metrics
 * - Rate limiting metrics
 * - Game move metrics
 * - Game session / AI turn metrics
 */

import client from 'prom-client';
import { MetricsService, getMetricsService } from '../../src/server/services/MetricsService';
import { DegradationLevel } from '../../src/server/services/ServiceStatusManager';

describe('MetricsService', () => {
  beforeEach(() => {
    // Reset the singleton and clear all metrics before each test
    MetricsService.resetInstance();
    client.register.clear();
  });

  afterAll(() => {
    // Clean up after all tests
    MetricsService.resetInstance();
    client.register.clear();
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance on multiple calls', () => {
      const instance1 = getMetricsService();
      const instance2 = getMetricsService();
      expect(instance1).toBe(instance2);
    });

    it('should be initialized after getInstance', () => {
      const metrics = getMetricsService();
      expect(metrics.isInitialized()).toBe(true);
    });

    it('should create new instance after reset', () => {
      const instance1 = getMetricsService();
      MetricsService.resetInstance();
      client.register.clear();
      const instance2 = getMetricsService();
      expect(instance1).not.toBe(instance2);
    });
  });

  describe('getMetrics', () => {
    it('should return valid Prometheus format metrics', async () => {
      const metrics = getMetricsService();
      const output = await metrics.getMetrics();

      // Should contain metric definitions
      expect(output).toContain('# HELP');
      expect(output).toContain('# TYPE');

      // Should contain our custom metrics
      expect(output).toContain('http_request_duration_seconds');
      expect(output).toContain('ringrift_games_total');
      expect(output).toContain('ringrift_service_status');
    });

    it('should return correct content type', () => {
      const metrics = getMetricsService();
      const contentType = metrics.getContentType();
      expect(contentType).toContain('text/plain');
    });
  });

  describe('HTTP Request Metrics', () => {
    it('should record HTTP requests', async () => {
      const metrics = getMetricsService();

      metrics.recordHttpRequest('GET', '/api/games', 200, 0.05, 100, 500);
      metrics.recordHttpRequest('POST', '/api/games', 201, 0.1, 1000, 200);
      metrics.recordHttpRequest('GET', '/api/games', 500, 0.2);

      const output = await metrics.getMetrics();

      // Should have recorded the requests
      expect(output).toContain('http_requests_total');
      expect(output).toContain('method="GET"');
      expect(output).toContain('method="POST"');
      expect(output).toContain('status="200"');
      expect(output).toContain('status="201"');
      expect(output).toContain('status="500"');
    });

    it('should normalize dynamic path segments', async () => {
      const metrics = getMetricsService();

      // Should normalize UUIDs
      metrics.recordHttpRequest(
        'GET',
        '/api/games/550e8400-e29b-41d4-a716-446655440000',
        200,
        0.05
      );
      metrics.recordHttpRequest(
        'GET',
        '/api/games/123e4567-e89b-12d3-a456-426614174000',
        200,
        0.05
      );

      // Should normalize numeric IDs
      metrics.recordHttpRequest('GET', '/api/users/12345', 200, 0.05);

      const output = await metrics.getMetrics();

      // Paths should be normalized to :id
      expect(output).toContain('path="/api/games/:id"');
      expect(output).toContain('path="/api/users/:id"');
    });

    it('should record request and response sizes', async () => {
      const metrics = getMetricsService();

      metrics.recordHttpRequest('POST', '/api/games', 201, 0.1, 5000, 1000);

      const output = await metrics.getMetrics();

      expect(output).toContain('http_request_size_bytes');
      expect(output).toContain('http_response_size_bytes');
    });
  });

  describe('Business Metrics', () => {
    it('should track game creation', async () => {
      const metrics = getMetricsService();

      metrics.recordGameCreated('pvp');
      metrics.recordGameCreated('ai');
      metrics.recordGameCreated('ai');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_games_total');
      expect(output).toContain('type="pvp"');
      expect(output).toContain('type="ai"');
    });

    it('should track active games', async () => {
      const metrics = getMetricsService();

      metrics.recordGameCreated('pvp'); // +1
      metrics.recordGameCreated('ai'); // +1
      metrics.recordGameEnded('pvp', 300); // -1

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_games_active 1');
    });

    it('should track game duration', async () => {
      const metrics = getMetricsService();

      metrics.recordGameEnded('pvp', 600); // 10 minutes

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_game_duration_seconds');
    });

    it('should track moves', async () => {
      const metrics = getMetricsService();

      metrics.recordMove('placement');
      metrics.recordMove('movement');
      metrics.recordMove('placement');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_moves_total');
      expect(output).toContain('type="placement"');
      expect(output).toContain('type="movement"');
    });

    it('should track active users', async () => {
      const metrics = getMetricsService();

      metrics.setActiveUsers(42);

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_users_active 42');
    });

    it('should track WebSocket connections', async () => {
      const metrics = getMetricsService();

      metrics.incWebSocketConnections();
      metrics.incWebSocketConnections();
      metrics.decWebSocketConnections();

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_websocket_connections 1');
    });
  });

  describe('Service Health Metrics', () => {
    it('should track service status', async () => {
      const metrics = getMetricsService();

      metrics.updateServiceStatus('database', 'healthy');
      metrics.updateServiceStatus('redis', 'unhealthy');
      metrics.updateServiceStatus('aiService', 'degraded');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_service_status');
      expect(output).toContain('service="database"');
      expect(output).toContain('service="redis"');
      expect(output).toContain('service="aiService"');
    });

    it('should set 1 for healthy, 0 for unhealthy', async () => {
      const metrics = getMetricsService();

      metrics.updateServiceStatus('database', 'healthy');
      metrics.updateServiceStatus('redis', 'unhealthy');

      const output = await metrics.getMetrics();

      // Database should be 1 (healthy)
      expect(output).toMatch(/ringrift_service_status\{service="database"\} 1/);
      // Redis should be 0 (unhealthy)
      expect(output).toMatch(/ringrift_service_status\{service="redis"\} 0/);
    });

    it('should track degradation level', async () => {
      const metrics = getMetricsService();

      metrics.updateDegradationLevel(DegradationLevel.FULL);
      let output = await metrics.getMetrics();
      expect(output).toContain('ringrift_degradation_level 0');

      metrics.updateDegradationLevel(DegradationLevel.DEGRADED);
      output = await metrics.getMetrics();
      expect(output).toContain('ringrift_degradation_level 1');

      metrics.updateDegradationLevel(DegradationLevel.MINIMAL);
      output = await metrics.getMetrics();
      expect(output).toContain('ringrift_degradation_level 2');

      metrics.updateDegradationLevel(DegradationLevel.OFFLINE);
      output = await metrics.getMetrics();
      expect(output).toContain('ringrift_degradation_level 3');
    });

    it('should track service response times', async () => {
      const metrics = getMetricsService();

      metrics.recordServiceResponseTime('database', 0.01);
      metrics.recordServiceResponseTime('redis', 0.005);
      metrics.recordServiceResponseTime('aiService', 0.5);

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_service_response_time_seconds');
    });
  });

  describe('AI Service Metrics', () => {
    it('should track AI request outcomes', async () => {
      const metrics = getMetricsService();

      metrics.recordAIRequest('success');
      metrics.recordAIRequest('fallback');
      metrics.recordAIRequest('error');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_ai_requests_total');
      expect(output).toContain('outcome="success"');
      expect(output).toContain('outcome="fallback"');
      expect(output).toContain('outcome="error"');
    });

    it('should track AI request duration', async () => {
      const metrics = getMetricsService();

      metrics.recordAIRequestDuration('heuristic', 5, 0.2);
      metrics.recordAIRequestDuration('mcts', '7', 1.5);

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_ai_request_duration_seconds');
      expect(output).toContain('ai_type="heuristic"');
      expect(output).toContain('ai_type="mcts"');
    });

    it('should track AI fallbacks', async () => {
      const metrics = getMetricsService();

      metrics.recordAIFallback('timeout');
      metrics.recordAIFallback('circuit_breaker');
      metrics.recordAIFallback('overloaded');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_ai_fallback_total');
      expect(output).toContain('reason="timeout"');
      expect(output).toContain('reason="circuit_breaker"');
      expect(output).toContain('reason="overloaded"');
    });
  });

  describe('Rate Limiting Metrics', () => {
    it('should track rate limit hits', async () => {
      const metrics = getMetricsService();

      metrics.recordRateLimitHit('/api/games', 'api');
      metrics.recordRateLimitHit('/api/auth/login', 'authLogin');
      metrics.recordRateLimitHit('/api/games', 'api');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_rate_limit_hits_total');
      expect(output).toContain('endpoint="/api/games"');
      expect(output).toContain('limiter="api"');
      expect(output).toContain('limiter="authLogin"');
    });
  });

  describe('Game Move Metrics', () => {
    it('should track game move latency', async () => {
      const metrics = getMetricsService();

      metrics.recordGameMoveLatency('square', 'placement', 0.05);
      metrics.recordGameMoveLatency('hexagonal', 'movement', 0.1);

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_game_move_latency_seconds');
      expect(output).toContain('board_type="square"');
      expect(output).toContain('board_type="hexagonal"');
      expect(output).toContain('phase="placement"');
      expect(output).toContain('phase="movement"');
    });
  });

  describe('Game Session / AI Turn Metrics', () => {
    it('should update game session status current gauge correctly', async () => {
      const metrics = getMetricsService();

      // Initialize a session in the "active" state
      metrics.updateGameSessionStatusCurrent(null, 'active');
      let output = await metrics.getMetrics();
      expect(output).toMatch(/ringrift_game_session_status_current\{status="active"\} 1/);

      // Transition from "active" -> "completed"
      metrics.updateGameSessionStatusCurrent('active', 'completed');
      output = await metrics.getMetrics();
      expect(output).toMatch(/ringrift_game_session_status_current\{status="active"\} 0/);
      expect(output).toMatch(/ringrift_game_session_status_current\{status="completed"\} 1/);

      // Tear down the "completed" session
      metrics.updateGameSessionStatusCurrent('completed', null);
      output = await metrics.getMetrics();
      expect(output).toMatch(/ringrift_game_session_status_current\{status="completed"\} 0/);
    });

    it('should record game session status transitions', async () => {
      const metrics = getMetricsService();

      metrics.recordGameSessionStatusTransition('none', 'active');
      metrics.recordGameSessionStatusTransition('active', 'abandoned');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_game_session_status_transitions_total');
      expect(output).toMatch(
        /ringrift_game_session_status_transitions_total\{from="none",to="active"\} 1/
      );
      expect(output).toMatch(
        /ringrift_game_session_status_transitions_total\{from="active",to="abandoned"\} 1/
      );
    });

    it('should record AI turn request terminal outcomes with defaulted labels', async () => {
      const metrics = getMetricsService();

      // Completed with no error details
      metrics.recordAITurnRequestTerminal('completed');
      // Failed with explicit code and error type
      metrics.recordAITurnRequestTerminal('failed', 'AI_TIMEOUT', 'rules');
      // Canceled with only an error-type style reason
      metrics.recordAITurnRequestTerminal('canceled', undefined, 'client_abort');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_ai_turn_request_terminal_total');

      // Completed -> default code/ai_error_type to "none"
      expect(output).toMatch(
        /ringrift_ai_turn_request_terminal_total\{kind="completed",code="none",ai_error_type="none"\} 1/
      );

      // Failed -> preserve provided code and error type
      expect(output).toMatch(
        /ringrift_ai_turn_request_terminal_total\{kind="failed",code="AI_TIMEOUT",ai_error_type="rules"\} 1/
      );

      // Canceled -> default code to "none" but preserve provided ai_error_type
      expect(output).toMatch(
        /ringrift_ai_turn_request_terminal_total\{kind="canceled",code="none",ai_error_type="client_abort"\} 1/
      );
    });
  });

  describe('Orchestrator rollout metrics', () => {
    it('should expose orchestrator rollout gauges and counters', async () => {
      const metrics = getMetricsService();

      metrics.recordOrchestratorSession('orchestrator', 'percentage_rollout');
      metrics.recordOrchestratorMove('orchestrator', 'success');
      metrics.setOrchestratorCircuitBreakerState(false);
      metrics.setOrchestratorErrorRate(0.05);
      metrics.setOrchestratorRolloutPercentage(25);

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_orchestrator_sessions_total');
      expect(output).toContain('engine="orchestrator"');
      expect(output).toContain('selection_reason="percentage_rollout"');

      expect(output).toContain('ringrift_orchestrator_moves_total');
      expect(output).toContain('outcome="success"');

      expect(output).toContain('ringrift_orchestrator_circuit_breaker_state');
      expect(output).toContain('ringrift_orchestrator_error_rate');
      expect(output).toContain('ringrift_orchestrator_rollout_percentage');
    });

    it('should record orchestrator invariant violations with invariant_id labels', async () => {
      const metrics = getMetricsService();

      // S-invariant decrease and ACTIVE-no-moves should be mapped to their
      // corresponding high-level invariant IDs.
      metrics.recordOrchestratorInvariantViolation('S_INVARIANT_DECREASED');
      metrics.recordOrchestratorInvariantViolation('ACTIVE_NO_MOVES');

      const output = await metrics.getMetrics();

      expect(output).toContain('ringrift_orchestrator_invariant_violations_total');
      expect(output).toContain(
        'ringrift_orchestrator_invariant_violations_total{type="S_INVARIANT_DECREASED",invariant_id="INV-S-MONOTONIC"}'
      );
      expect(output).toContain(
        'ringrift_orchestrator_invariant_violations_total{type="ACTIVE_NO_MOVES",invariant_id="INV-ACTIVE-NO-MOVES"}'
      );
    });
  });

  describe('Default Node.js Metrics', () => {
    it('should include default Node.js metrics when collected', async () => {
      // Collect default metrics (this is done in server startup)
      client.collectDefaultMetrics();

      const metrics = getMetricsService();
      const output = await metrics.getMetrics();

      // Should include process metrics
      expect(output).toContain('process_cpu');
      expect(output).toContain('nodejs_heap');
    });
  });
});
