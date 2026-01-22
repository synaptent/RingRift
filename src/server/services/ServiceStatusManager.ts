/**
 * Service Status Manager - Tracks external service health and computes degradation levels.
 *
 * This module provides a centralized way to:
 * - Track the health status of all external dependencies (Database, Redis, AI Service)
 * - Compute the overall degradation level of the application
 * - Notify subscribers when service status changes
 * - Provide graceful degradation information to API responses
 *
 * Degradation Levels:
 * - FULL: All services available, full functionality
 * - DEGRADED: Non-critical services unavailable (AI service, Redis)
 * - MINIMAL: Only core functionality (local AI, no matchmaking, in-memory rate limiting)
 * - OFFLINE: Database unavailable, maintenance mode
 */

import { EventEmitter } from 'events';
import { logger } from '../utils/logger';

/**
 * External service identifiers tracked by the status manager.
 */
export type ServiceName = 'database' | 'redis' | 'aiService';

/**
 * Health status for individual services.
 */
export type ServiceHealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

/**
 * Overall application degradation level.
 */
export enum DegradationLevel {
  /** All services available, full functionality */
  FULL = 'FULL',
  /** Non-critical services unavailable (AI service down, Redis unavailable) */
  DEGRADED = 'DEGRADED',
  /** Only core functionality available (local AI, in-memory rate limiting) */
  MINIMAL = 'MINIMAL',
  /** Critical services unavailable (database down), maintenance mode */
  OFFLINE = 'OFFLINE',
}

/**
 * Detailed status for a single service.
 */
export interface ServiceStatus {
  name: ServiceName;
  status: ServiceHealthStatus;
  lastChecked: Date;
  lastHealthy?: Date | undefined;
  error?: string | undefined;
  latencyMs?: number | undefined;
  /** Number of consecutive failures */
  failureCount: number;
  /** Whether fallback is currently active */
  fallbackActive: boolean;
}

/**
 * Snapshot of overall system status.
 */
export interface SystemStatus {
  degradationLevel: DegradationLevel;
  services: Record<ServiceName, ServiceStatus>;
  degradedServices: ServiceName[];
  timestamp: Date;
}

/**
 * Event types emitted by the ServiceStatusManager.
 */
export interface ServiceStatusEvents {
  /** Emitted when any service status changes */
  statusChange: (
    service: ServiceName,
    oldStatus: ServiceHealthStatus,
    newStatus: ServiceHealthStatus
  ) => void;
  /** Emitted when the overall degradation level changes */
  degradationLevelChange: (oldLevel: DegradationLevel, newLevel: DegradationLevel) => void;
  /** Emitted when a service recovers from unhealthy/degraded to healthy */
  serviceRecovered: (service: ServiceName) => void;
  /** Emitted when a service becomes unavailable */
  serviceDown: (service: ServiceName, error?: string) => void;
}

/**
 * Configuration for automatic health check polling.
 */
export interface HealthCheckConfig {
  /** Interval in milliseconds between automatic health checks */
  pollingIntervalMs: number;
  /** Whether to enable automatic polling */
  enablePolling: boolean;
  /** Services to poll (subset of all services) */
  servicesToPoll: ServiceName[];
}

/**
 * Default configuration for health check polling.
 */
const DEFAULT_CONFIG: HealthCheckConfig = {
  pollingIntervalMs: 30000, // 30 seconds
  enablePolling: false, // Disabled by default, enabled explicitly
  servicesToPoll: ['database', 'redis', 'aiService'],
};

/**
 * Centralized service status manager.
 *
 * Tracks external service health, computes degradation levels, and provides
 * graceful degradation information for API responses and WebSocket broadcasts.
 *
 * Usage:
 * ```typescript
 * const statusManager = getServiceStatusManager();
 * statusManager.updateServiceStatus('aiService', 'unhealthy', 'Connection refused');
 *
 * const status = statusManager.getSystemStatus();
 * console.log(status.degradationLevel); // DegradationLevel.DEGRADED
 *
 * statusManager.on('degradationLevelChange', (oldLevel, newLevel) => {
 *   logger.warn('Degradation level changed', { oldLevel, newLevel });
 * });
 * ```
 */
/**
 * Type for health check callback results.
 */
export interface HealthCheckResult {
  status: ServiceHealthStatus;
  error?: string | undefined;
  latencyMs?: number | undefined;
}

export class ServiceStatusManager extends EventEmitter {
  private services: Map<ServiceName, ServiceStatus> = new Map();
  private currentDegradationLevel: DegradationLevel = DegradationLevel.FULL;
  private config: HealthCheckConfig;
  private pollingInterval: NodeJS.Timeout | null = null;
  private healthCheckCallbacks: Map<ServiceName, () => Promise<HealthCheckResult>> = new Map();

  constructor(config: Partial<HealthCheckConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.initializeServices();
  }

  /**
   * Initialize all tracked services with unknown status.
   */
  private initializeServices(): void {
    const serviceNames: ServiceName[] = ['database', 'redis', 'aiService'];
    const now = new Date();

    for (const name of serviceNames) {
      this.services.set(name, {
        name,
        status: 'unknown',
        lastChecked: now,
        failureCount: 0,
        fallbackActive: false,
      });
    }
  }

  /**
   * Register a health check callback for a service.
   * The callback will be invoked during polling to check service health.
   */
  registerHealthCheck(service: ServiceName, callback: () => Promise<HealthCheckResult>): void {
    this.healthCheckCallbacks.set(service, callback);
    logger.debug('Health check callback registered', { service });
  }

  /**
   * Update the status of a service.
   *
   * @param service - The service to update
   * @param status - New health status
   * @param error - Optional error message if unhealthy
   * @param latencyMs - Optional latency measurement
   */
  updateServiceStatus(
    service: ServiceName,
    status: ServiceHealthStatus,
    error?: string,
    latencyMs?: number
  ): void {
    const current = this.services.get(service);
    if (!current) {
      logger.warn('Attempted to update unknown service', { service });
      return;
    }

    const oldStatus = current.status;
    const now = new Date();

    // Update service status
    const updated: ServiceStatus = {
      ...current,
      status,
      lastChecked: now,
      error: status === 'healthy' ? undefined : error,
      latencyMs,
      failureCount:
        status === 'healthy' ? 0 : current.failureCount + (oldStatus !== status ? 1 : 0),
      fallbackActive: status !== 'healthy',
    };

    if (status === 'healthy') {
      updated.lastHealthy = now;
    }

    this.services.set(service, updated);

    // Emit events if status changed
    if (oldStatus !== status) {
      this.emit('statusChange', service, oldStatus, status);

      if (status === 'healthy' && (oldStatus === 'unhealthy' || oldStatus === 'degraded')) {
        this.emit('serviceRecovered', service);
        logger.info('Service recovered', { service, previousStatus: oldStatus });
      } else if (status === 'unhealthy' || (status === 'degraded' && oldStatus === 'healthy')) {
        this.emit('serviceDown', service, error);
        logger.warn('Service became unavailable', { service, status, error });
      }

      // Recompute degradation level
      this.updateDegradationLevel();
    }
  }

  /**
   * Mark a service as having active fallback (e.g., using local AI instead of remote).
   */
  setFallbackActive(service: ServiceName, active: boolean): void {
    const current = this.services.get(service);
    if (current) {
      current.fallbackActive = active;
      this.services.set(service, current);
    }
  }

  /**
   * Get the current status of a specific service.
   */
  getServiceStatus(service: ServiceName): ServiceStatus | undefined {
    return this.services.get(service);
  }

  /**
   * Get comprehensive system status including all services and degradation level.
   */
  getSystemStatus(): SystemStatus {
    const degradedServices: ServiceName[] = [];

    for (const [name, status] of this.services) {
      if (status.status !== 'healthy') {
        degradedServices.push(name);
      }
    }

    return {
      degradationLevel: this.currentDegradationLevel,
      services: Object.fromEntries(this.services) as Record<ServiceName, ServiceStatus>,
      degradedServices,
      timestamp: new Date(),
    };
  }

  /**
   * Get the current degradation level.
   */
  getDegradationLevel(): DegradationLevel {
    return this.currentDegradationLevel;
  }

  /**
   * Check if the system is operating in degraded mode.
   */
  isDegraded(): boolean {
    return this.currentDegradationLevel !== DegradationLevel.FULL;
  }

  /**
   * Check if a specific service is healthy.
   */
  isServiceHealthy(service: ServiceName): boolean {
    const status = this.services.get(service);
    return status?.status === 'healthy';
  }

  /**
   * Get list of services currently in degraded/unhealthy state.
   */
  getDegradedServices(): ServiceName[] {
    const degraded: ServiceName[] = [];
    for (const [name, status] of this.services) {
      if (status.status !== 'healthy') {
        degraded.push(name);
      }
    }
    return degraded;
  }

  /**
   * Compute and update the overall degradation level based on service statuses.
   */
  private updateDegradationLevel(): void {
    const oldLevel = this.currentDegradationLevel;
    const newLevel = this.computeDegradationLevel();

    if (oldLevel !== newLevel) {
      this.currentDegradationLevel = newLevel;
      this.emit('degradationLevelChange', oldLevel, newLevel);

      logger.info('System degradation level changed', {
        oldLevel,
        newLevel,
        services: Object.fromEntries(
          Array.from(this.services.entries()).map(([name, status]) => [name, status.status])
        ),
      });
    }
  }

  /**
   * Compute degradation level from current service statuses.
   *
   * Logic:
   * - OFFLINE: Database is unhealthy
   * - MINIMAL: Multiple non-critical services down or database degraded
   * - DEGRADED: Any non-critical service (AI, Redis) is unhealthy or degraded
   * - FULL: All services healthy
   */
  private computeDegradationLevel(): DegradationLevel {
    const database = this.services.get('database');
    const redis = this.services.get('redis');
    const aiService = this.services.get('aiService');

    // Database is critical - if it's unhealthy, we're offline
    if (database?.status === 'unhealthy') {
      return DegradationLevel.OFFLINE;
    }

    // Count non-critical service issues
    const nonCriticalIssues: ServiceName[] = [];

    if (redis?.status === 'unhealthy' || redis?.status === 'degraded') {
      nonCriticalIssues.push('redis');
    }
    if (aiService?.status === 'unhealthy' || aiService?.status === 'degraded') {
      nonCriticalIssues.push('aiService');
    }

    // If database is degraded or multiple non-critical services are down
    if (database?.status === 'degraded' && nonCriticalIssues.length > 0) {
      return DegradationLevel.MINIMAL;
    }

    // If all non-critical services are down
    if (nonCriticalIssues.length >= 2) {
      return DegradationLevel.MINIMAL;
    }

    // Any degraded non-critical service
    if (nonCriticalIssues.length > 0 || database?.status === 'degraded') {
      return DegradationLevel.DEGRADED;
    }

    return DegradationLevel.FULL;
  }

  /**
   * Get HTTP headers to indicate service degradation status.
   */
  getDegradationHeaders(): Record<string, string> {
    const status = this.getSystemStatus();

    if (status.degradationLevel === DegradationLevel.FULL) {
      return {};
    }

    const headers: Record<string, string> = {
      'X-Service-Status': status.degradationLevel.toLowerCase(),
    };

    if (status.degradedServices.length > 0) {
      headers['X-Degraded-Services'] = status.degradedServices.join(',');
    }

    return headers;
  }

  /**
   * Start automatic health check polling.
   *
   * Polling is enabled when:
   * - ENABLE_HEALTH_POLLING=true (explicit enable), OR
   * - NODE_ENV=production (auto-enable in prod)
   *
   * Polling is disabled when:
   * - ENABLE_HEALTH_POLLING=false (explicit disable), OR
   * - Not production AND not explicitly enabled
   */
  startPolling(): void {
    // Check environment for auto-enable in production
    const isProduction = process.env.NODE_ENV === 'production';
    const explicitlyEnabled = process.env.ENABLE_HEALTH_POLLING === 'true';
    const explicitlyDisabled = process.env.ENABLE_HEALTH_POLLING === 'false';

    // Determine if polling should be enabled
    const shouldPoll = explicitlyDisabled
      ? false
      : explicitlyEnabled || isProduction || this.config.enablePolling;

    if (!shouldPoll || this.pollingInterval) {
      return;
    }

    this.pollingInterval = setInterval(async () => {
      await this.runHealthChecks();
    }, this.config.pollingIntervalMs);

    logger.info('Service status polling started', {
      intervalMs: this.config.pollingIntervalMs,
      services: this.config.servicesToPoll,
    });
  }

  /**
   * Stop automatic health check polling.
   */
  stopPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
      logger.info('Service status polling stopped');
    }
  }

  /**
   * Run health checks for all configured services.
   */
  async runHealthChecks(): Promise<void> {
    const promises = this.config.servicesToPoll.map(async (service) => {
      const callback = this.healthCheckCallbacks.get(service);
      if (!callback) {
        return;
      }

      try {
        const result = await callback();
        this.updateServiceStatus(service, result.status, result.error, result.latencyMs);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.updateServiceStatus(service, 'unhealthy', errorMessage);
      }
    });

    await Promise.all(promises);
  }

  /**
   * Force a refresh of all service statuses.
   */
  async refresh(): Promise<SystemStatus> {
    await this.runHealthChecks();
    return this.getSystemStatus();
  }

  /**
   * Reset all service statuses to unknown.
   * Useful for testing or after a system restart.
   */
  reset(): void {
    this.initializeServices();
    this.currentDegradationLevel = DegradationLevel.FULL;
    this.healthCheckCallbacks.clear();
    logger.debug('Service status manager reset');
  }

  /**
   * Clean up resources.
   */
  destroy(): void {
    this.stopPolling();
    this.removeAllListeners();
    this.healthCheckCallbacks.clear();
    logger.debug('Service status manager destroyed');
  }
}

// Singleton instance
let serviceStatusManager: ServiceStatusManager | null = null;

/**
 * Get the singleton ServiceStatusManager instance.
 */
export function getServiceStatusManager(): ServiceStatusManager {
  if (!serviceStatusManager) {
    serviceStatusManager = new ServiceStatusManager();
  }
  return serviceStatusManager;
}

/**
 * Initialize the ServiceStatusManager with custom configuration.
 */
export function initServiceStatusManager(
  config?: Partial<HealthCheckConfig>
): ServiceStatusManager {
  if (serviceStatusManager) {
    serviceStatusManager.destroy();
  }
  serviceStatusManager = new ServiceStatusManager(config);
  return serviceStatusManager;
}

/**
 * Reset the singleton instance (primarily for testing).
 */
export function resetServiceStatusManager(): void {
  if (serviceStatusManager) {
    serviceStatusManager.destroy();
    serviceStatusManager = null;
  }
}

export const ServiceStatusService = {
  getServiceStatusManager,
  initServiceStatusManager,
  resetServiceStatusManager,
};
