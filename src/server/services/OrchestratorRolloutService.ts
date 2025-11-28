/**
 * OrchestratorRolloutService
 *
 * Centralized service for making rollout decisions about which rules engine to use.
 * Implements a priority-based decision system for safe, gradual rollout of the
 * orchestrator adapter.
 *
 * Decision Priority (highest to lowest):
 * 1. Kill switch (adapterEnabled=false) → always legacy
 * 2. Denylist → user in denylist → force legacy
 * 3. Allowlist → user in allowlist → force orchestrator
 * 4. Circuit breaker → if tripped → force legacy
 * 5. Percentage rollout → hash(sessionId) determines engine
 *
 * @see docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md
 */

import { config } from '../config';
import { logger } from '../utils/logger';
import { getMetricsService } from './MetricsService';

/**
 * Engine selection result.
 */
export enum EngineSelection {
  /** Use the legacy TypeScript rules engine */
  LEGACY = 'legacy',
  /** Use the new orchestrator adapter */
  ORCHESTRATOR = 'orchestrator',
  /** Shadow mode: run both engines, compare results, use legacy */
  SHADOW = 'shadow',
}

/**
 * Result of an engine selection decision.
 */
export interface RolloutDecision {
  /** The engine to use */
  engine: EngineSelection;
  /** Human-readable reason for the decision (for metrics/logging) */
  reason: string;
}

/**
 * Circuit breaker state for monitoring and debugging.
 */
export interface CircuitBreakerState {
  /** Whether the circuit breaker is currently open (tripped) */
  isOpen: boolean;
  /** Number of errors in the current window */
  errorCount: number;
  /** Number of requests in the current window */
  requestCount: number;
  /** Timestamp when the current window started */
  windowStart: number;
}

/**
 * Minimum number of requests required before circuit breaker can trip.
 * Prevents premature tripping on small sample sizes.
 */
const MIN_REQUESTS_FOR_CIRCUIT_BREAKER = 10;

/**
 * Service for making orchestrator rollout decisions.
 *
 * This service is responsible for determining which rules engine should
 * handle game operations based on feature flags, user targeting, and
 * circuit breaker state.
 */
export class OrchestratorRolloutService {
  private circuitBreakerState: CircuitBreakerState;

  constructor() {
    this.circuitBreakerState = {
      isOpen: false,
      errorCount: 0,
      requestCount: 0,
      windowStart: Date.now(),
    };
  }

  /**
   * Determine which engine to use for a session.
   *
   * Decision priority:
   * 1. Kill switch - if disabled, always use legacy
   * 2. Denylist - if user is denylisted, use legacy
   * 3. Allowlist - if user is allowlisted, use orchestrator (or shadow)
   * 4. Circuit breaker - if open, use legacy
   * 5. Percentage rollout - use consistent hash to determine
   *
   * @param sessionId - The game session ID (used for consistent hashing)
   * @param userId - Optional user ID for targeting (allowlist/denylist)
   * @returns The engine selection decision with reason
   */
  selectEngine(sessionId: string, userId?: string): RolloutDecision {
    const orchestratorConfig = config.featureFlags.orchestrator;

    // 1. Kill switch check
    if (!orchestratorConfig.adapterEnabled) {
      const decision = { engine: EngineSelection.LEGACY, reason: 'kill_switch' };
      logger.debug('Engine selection: kill switch disabled', {
        sessionId,
        userId,
        decision: decision.reason,
      });
      return decision;
    }

    // 2. Denylist check (denylist takes precedence over allowlist)
    if (userId && orchestratorConfig.denylistUsers.includes(userId)) {
      const decision = { engine: EngineSelection.LEGACY, reason: 'denylist' };
      logger.debug('Engine selection: user in denylist', {
        sessionId,
        userId,
        decision: decision.reason,
      });
      return decision;
    }

    // 3. Allowlist check
    if (userId && orchestratorConfig.allowlistUsers.includes(userId)) {
      const decision = this.shadowOrOrchestrator('allowlist');
      logger.debug('Engine selection: user in allowlist', {
        sessionId,
        userId,
        decision: decision.reason,
        engine: decision.engine,
      });
      return decision;
    }

    // 4. Circuit breaker check
    if (this.isCircuitBreakerOpen()) {
      const decision = { engine: EngineSelection.LEGACY, reason: 'circuit_breaker' };
      logger.debug('Engine selection: circuit breaker open', {
        sessionId,
        userId,
        decision: decision.reason,
        circuitBreakerState: this.circuitBreakerState,
      });
      return decision;
    }

    // 5. Percentage rollout using consistent hash
    const percentage = orchestratorConfig.rolloutPercentage;
    const sessionBucket = this.hashToPercentage(sessionId);

    if (percentage >= 100 || sessionBucket < percentage) {
      const decision = this.shadowOrOrchestrator('percentage_rollout');
      logger.debug('Engine selection: percentage rollout included', {
        sessionId,
        userId,
        sessionBucket,
        rolloutPercentage: percentage,
        decision: decision.reason,
        engine: decision.engine,
      });
      return decision;
    }

    // Session not selected by percentage
    const decision = { engine: EngineSelection.LEGACY, reason: 'percentage_excluded' };
    logger.debug('Engine selection: percentage rollout excluded', {
      sessionId,
      userId,
      sessionBucket,
      rolloutPercentage: percentage,
      decision: decision.reason,
    });
    return decision;
  }

  /**
   * Returns SHADOW if shadow mode is enabled, otherwise ORCHESTRATOR.
   *
   * @param baseReason - The base reason for the decision (will be suffixed with _shadow if applicable)
   * @returns The appropriate engine selection
   */
  private shadowOrOrchestrator(baseReason: string): RolloutDecision {
    if (config.featureFlags.orchestrator.shadowModeEnabled) {
      return { engine: EngineSelection.SHADOW, reason: `${baseReason}_shadow` };
    }
    return { engine: EngineSelection.ORCHESTRATOR, reason: baseReason };
  }

  /**
   * Consistent hash of session ID to 0-99 percentage bucket.
   *
   * Uses a simple djb2-like hash function to deterministically map session IDs
   * to percentage buckets. The same session ID will always map to the same
   * bucket, ensuring consistent engine selection for a given session.
   *
   * @param sessionId - The session ID to hash
   * @returns A number from 0 to 99
   */
  private hashToPercentage(sessionId: string): number {
    let hash = 0;
    for (let i = 0; i < sessionId.length; i++) {
      // djb2-like hash: hash * 33 + charCode
      hash = (hash << 5) - hash + sessionId.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash % 100);
  }

  /**
   * Record a successful orchestrator request.
   *
   * Called after each successful operation through the orchestrator.
   * Used for circuit breaker error rate calculation.
   */
  recordSuccess(): void {
    this.maybeResetWindow();
    this.circuitBreakerState.requestCount++;
    this.updateMetrics();
  }

  /**
   * Record a failed orchestrator request.
   *
   * Called after each failed operation through the orchestrator.
   * May trip the circuit breaker if error rate exceeds threshold.
   */
  recordError(): void {
    this.maybeResetWindow();
    this.circuitBreakerState.requestCount++;
    this.circuitBreakerState.errorCount++;

    // Check if we should trip the circuit breaker
    this.checkCircuitBreaker();
    this.updateMetrics();
  }

  /**
   * Check if the circuit breaker is currently open (tripped).
   *
   * The circuit breaker opens when:
   * - Circuit breaker feature is enabled
   * - Error rate exceeds the configured threshold
   * - Minimum request count has been met
   *
   * @returns true if the circuit breaker is open (orchestrator should be bypassed)
   */
  isCircuitBreakerOpen(): boolean {
    // Reset window if needed (may close circuit breaker due to window expiry)
    this.maybeResetWindow();
    return this.circuitBreakerState.isOpen;
  }

  /**
   * Reset the circuit breaker state.
   *
   * This is a manual recovery action that should be called after
   * investigating and fixing the underlying issue. The circuit breaker
   * does not auto-recover to prevent flapping.
   */
  resetCircuitBreaker(): void {
    const wasOpen = this.circuitBreakerState.isOpen;

    this.circuitBreakerState = {
      isOpen: false,
      errorCount: 0,
      requestCount: 0,
      windowStart: Date.now(),
    };

    if (wasOpen) {
      logger.info('Circuit breaker manually reset', {
        service: 'OrchestratorRolloutService',
        action: 'circuit_breaker_reset',
      });
    }

    this.updateMetrics();
  }

  /**
   * Get the current circuit breaker state for monitoring.
   *
   * @returns A copy of the current circuit breaker state
   */
  getCircuitBreakerState(): CircuitBreakerState {
    this.maybeResetWindow();
    return { ...this.circuitBreakerState };
  }

  /**
   * Get the current error rate as a percentage.
   *
   * @returns Error rate from 0-100, or 0 if no requests recorded
   */
  getErrorRate(): number {
    if (this.circuitBreakerState.requestCount === 0) {
      return 0;
    }
    return (this.circuitBreakerState.errorCount / this.circuitBreakerState.requestCount) * 100;
  }

  /**
   * Reset the error window if it has expired.
   *
   * Called before any circuit breaker state reads or writes to ensure
   * the window is current.
   */
  private maybeResetWindow(): void {
    const now = Date.now();
    const windowMs = config.featureFlags.orchestrator.circuitBreaker.errorWindowSeconds * 1000;
    const windowAge = now - this.circuitBreakerState.windowStart;

    if (windowAge >= windowMs) {
      const wasOpen = this.circuitBreakerState.isOpen;
      const previousErrorCount = this.circuitBreakerState.errorCount;
      const previousRequestCount = this.circuitBreakerState.requestCount;

      // Window has elapsed, reset counts
      this.circuitBreakerState = {
        isOpen: false, // Close the breaker on window reset
        errorCount: 0,
        requestCount: 0,
        windowStart: now,
      };

      if (wasOpen) {
        logger.info('Circuit breaker closed due to window expiry', {
          service: 'OrchestratorRolloutService',
          action: 'circuit_breaker_closed',
          previousErrorCount,
          previousRequestCount,
          windowMs,
        });
      }

      this.updateMetrics();
    }
  }

  /**
   * Check if the circuit breaker should be tripped based on current error rate.
   */
  private checkCircuitBreaker(): void {
    const cbConfig = config.featureFlags.orchestrator.circuitBreaker;

    // Circuit breaker must be enabled
    if (!cbConfig.enabled) {
      return;
    }

    // Need minimum requests before we can trip
    if (this.circuitBreakerState.requestCount < MIN_REQUESTS_FOR_CIRCUIT_BREAKER) {
      return;
    }

    // Calculate error rate
    const errorRate = this.getErrorRate();

    // Check if we should trip
    if (errorRate > cbConfig.errorThresholdPercent && !this.circuitBreakerState.isOpen) {
      this.circuitBreakerState.isOpen = true;

      logger.error('Circuit breaker OPENED - orchestrator disabled', {
        service: 'OrchestratorRolloutService',
        action: 'circuit_breaker_opened',
        errorRate: errorRate.toFixed(2),
        errorCount: this.circuitBreakerState.errorCount,
        requestCount: this.circuitBreakerState.requestCount,
        thresholdPercent: cbConfig.errorThresholdPercent,
        windowSeconds: cbConfig.errorWindowSeconds,
      });

      this.updateMetrics();
    }
  }

  /**
   * Update orchestrator rollout-related metrics (circuit breaker state and
   * error rate) for observability.
   */
  private updateMetrics(): void {
    const metrics = getMetricsService();
    // Circuit breaker state: 0=closed, 1=open
    metrics.setOrchestratorCircuitBreakerState(this.circuitBreakerState.isOpen);
    // Error rate gauge exposed as 0.0–1.0 fraction
    const errorRatePercent = this.getErrorRate();
    metrics.setOrchestratorErrorRate(errorRatePercent / 100);
  }
}

/**
 * Singleton instance of the OrchestratorRolloutService.
 *
 * Use this instance throughout the application for consistent rollout decisions.
 */
export const orchestratorRollout = new OrchestratorRolloutService();
