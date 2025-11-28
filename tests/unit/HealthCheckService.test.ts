/**
 * Unit tests for HealthCheckService
 *
 * Tests cover:
 * - Liveness checks (always healthy)
 * - Readiness checks (healthy, degraded, unhealthy states)
 * - Individual service checks (database, Redis, AI service)
 * - Timeout handling
 * - Error handling
 * - ServiceStatusManager integration
 */

import {
  HealthCheckService,
  getLivenessStatus,
  getReadinessStatus,
  isServiceReady,
  registerHealthChecksWithStatusManager,
  HealthCheckResponse,
} from '../../src/server/services/HealthCheckService';

// Mock dependencies
const mockCheckDatabaseHealth = jest.fn();
const mockGetDatabaseClient = jest.fn();
const mockGetRedisClient = jest.fn();
const mockGetServiceStatusManager = jest.fn();
const mockUpdateServiceStatus = jest.fn();
const mockRegisterHealthCheck = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  checkDatabaseHealth: () => mockCheckDatabaseHealth(),
  getDatabaseClient: () => mockGetDatabaseClient(),
}));

jest.mock('../../src/server/cache/redis', () => ({
  getRedisClient: () => mockGetRedisClient(),
}));

jest.mock('../../src/server/config', () => ({
  config: {
    aiService: {
      url: 'http://ai-service:8000',
    },
    app: {
      version: '1.0.0-test',
    },
  },
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

jest.mock('../../src/server/services/ServiceStatusManager', () => ({
  getServiceStatusManager: () => mockGetServiceStatusManager(),
  ServiceHealthStatus: {
    healthy: 'healthy',
    degraded: 'degraded',
    unhealthy: 'unhealthy',
    unknown: 'unknown',
  },
}));

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('HealthCheckService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Default mock implementations
    mockGetServiceStatusManager.mockReturnValue({
      updateServiceStatus: mockUpdateServiceStatus,
      registerHealthCheck: mockRegisterHealthCheck,
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  // ==========================================================================
  // getLivenessStatus Tests
  // ==========================================================================

  describe('getLivenessStatus', () => {
    it('returns healthy status immediately', () => {
      const result = getLivenessStatus();

      expect(result.status).toBe('healthy');
      expect(result.version).toBe('1.0.0-test');
      expect(result.uptime).toBeGreaterThanOrEqual(0);
      expect(result.timestamp).toBeDefined();
    });

    it('returns valid ISO timestamp', () => {
      const result = getLivenessStatus();

      const timestamp = new Date(result.timestamp);
      expect(timestamp.toISOString()).toBe(result.timestamp);
    });

    it('does not include checks in response', () => {
      const result = getLivenessStatus();

      expect(result.checks).toBeUndefined();
    });
  });

  // ==========================================================================
  // getReadinessStatus Tests - Healthy Scenarios
  // ==========================================================================

  describe('getReadinessStatus - healthy scenarios', () => {
    it('returns healthy when all services are up', async () => {
      // Mock all services as healthy
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('healthy');
      expect(result.checks?.database?.status).toBe('healthy');
      expect(result.checks?.redis?.status).toBe('healthy');
      expect(result.checks?.aiService?.status).toBe('healthy');
    });

    it('includes latency measurements for healthy services', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.checks?.database?.latency).toBeDefined();
      expect(result.checks?.redis?.latency).toBeDefined();
      expect(result.checks?.aiService?.latency).toBeDefined();
    });
  });

  // ==========================================================================
  // getReadinessStatus Tests - Degraded Scenarios
  // ==========================================================================

  describe('getReadinessStatus - degraded scenarios', () => {
    it('returns degraded when Redis is down', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue(null); // Redis not connected
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('degraded');
      expect(result.checks?.database?.status).toBe('healthy');
      expect(result.checks?.redis?.status).toBe('degraded');
      expect(result.checks?.redis?.error).toBe('Redis client not connected');
      expect(result.checks?.aiService?.status).toBe('healthy');
    });

    it('returns degraded when AI service is unavailable', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockRejectedValue(new Error('Connection refused'));

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('degraded');
      expect(result.checks?.database?.status).toBe('healthy');
      expect(result.checks?.redis?.status).toBe('healthy');
      expect(result.checks?.aiService?.status).toBe('degraded');
      expect(result.checks?.aiService?.error).toContain('Connection refused');
    });

    it('returns degraded when AI service returns non-ok response', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('degraded');
      expect(result.checks?.aiService?.status).toBe('degraded');
      expect(result.checks?.aiService?.error).toContain('503');
    });

    it('returns degraded when Redis ping fails', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockRejectedValue(new Error('Redis timeout')),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('degraded');
      expect(result.checks?.redis?.status).toBe('degraded');
      expect(result.checks?.redis?.error).toContain('Redis timeout');
    });

    it('returns degraded when Redis returns unexpected response', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('UNEXPECTED'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('degraded');
      expect(result.checks?.redis?.status).toBe('degraded');
      expect(result.checks?.redis?.error).toContain('Unexpected PING response');
    });
  });

  // ==========================================================================
  // getReadinessStatus Tests - Unhealthy Scenarios
  // ==========================================================================

  describe('getReadinessStatus - unhealthy scenarios', () => {
    it('returns unhealthy when database is down', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(false); // Database query failed
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('unhealthy');
      expect(result.checks?.database?.status).toBe('unhealthy');
      expect(result.checks?.database?.error).toBe('Database query failed');
    });

    it('returns unhealthy when database client is not initialized', async () => {
      mockGetDatabaseClient.mockReturnValue(null); // Database not initialized
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('unhealthy');
      expect(result.checks?.database?.status).toBe('unhealthy');
      expect(result.checks?.database?.error).toBe('Database client not initialized');
    });

    it('returns unhealthy when database throws exception', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockRejectedValue(new Error('Connection lost'));
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      expect(result.status).toBe('unhealthy');
      expect(result.checks?.database?.status).toBe('unhealthy');
      expect(result.checks?.database?.error).toContain('Connection lost');
    });
  });

  // ==========================================================================
  // Timeout Handling Tests
  // ==========================================================================

  describe('timeout handling', () => {
    it('handles database timeout', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      // Never-resolving promise to simulate timeout
      mockCheckDatabaseHealth.mockImplementation(() => new Promise(() => {}));
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const resultPromise = getReadinessStatus({ timeoutMs: 100 });

      // Fast-forward past the timeout
      jest.advanceTimersByTime(150);

      const result = await resultPromise;

      expect(result.status).toBe('unhealthy');
      expect(result.checks?.database?.status).toBe('unhealthy');
      expect(result.checks?.database?.error).toContain('timed out');
    });

    it('handles Redis timeout', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockImplementation(() => new Promise(() => {})),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const resultPromise = getReadinessStatus({ timeoutMs: 100 });

      jest.advanceTimersByTime(150);

      const result = await resultPromise;

      expect(result.status).toBe('degraded');
      expect(result.checks?.redis?.status).toBe('degraded');
      expect(result.checks?.redis?.error).toContain('timed out');
    });
  });

  // ==========================================================================
  // isServiceReady Tests
  // ==========================================================================

  describe('isServiceReady', () => {
    it('returns true for healthy status', () => {
      const response: HealthCheckResponse = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
      };

      expect(isServiceReady(response)).toBe(true);
    });

    it('returns true for degraded status', () => {
      const response: HealthCheckResponse = {
        status: 'degraded',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
      };

      expect(isServiceReady(response)).toBe(true);
    });

    it('returns false for unhealthy status', () => {
      const response: HealthCheckResponse = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        uptime: 100,
      };

      expect(isServiceReady(response)).toBe(false);
    });
  });

  // ==========================================================================
  // registerHealthChecksWithStatusManager Tests
  // ==========================================================================

  describe('registerHealthChecksWithStatusManager', () => {
    it('registers health checks for all services', () => {
      registerHealthChecksWithStatusManager(5000);

      expect(mockRegisterHealthCheck).toHaveBeenCalledTimes(3);
      expect(mockRegisterHealthCheck).toHaveBeenCalledWith('database', expect.any(Function));
      expect(mockRegisterHealthCheck).toHaveBeenCalledWith('redis', expect.any(Function));
      expect(mockRegisterHealthCheck).toHaveBeenCalledWith('aiService', expect.any(Function));
    });

    it('uses provided timeout for health checks', async () => {
      registerHealthChecksWithStatusManager(1000);

      // Get the database health check callback
      const databaseCallback = mockRegisterHealthCheck.mock.calls.find(
        (call: [string, () => Promise<unknown>]) => call[0] === 'database'
      )?.[1];

      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);

      const result = await databaseCallback();

      expect(result.status).toBe('healthy');
    });

    it('handles registration failure gracefully', () => {
      mockGetServiceStatusManager.mockImplementation(() => {
        throw new Error('Status manager not initialized');
      });

      // Should not throw
      expect(() => registerHealthChecksWithStatusManager(5000)).not.toThrow();
    });
  });

  // ==========================================================================
  // Options Tests
  // ==========================================================================

  describe('options handling', () => {
    it('uses default timeout when not specified', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus();

      expect(result.status).toBe('healthy');
    });

    it('excludes AI service check when includeAIService is false', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });

      const result = await getReadinessStatus({ includeAIService: false });

      expect(result.status).toBe('healthy');
      expect(result.checks?.aiService).toBeUndefined();
      expect(mockFetch).not.toHaveBeenCalled();
    });
  });

  // ==========================================================================
  // ServiceStatusManager Integration Tests
  // ==========================================================================

  describe('ServiceStatusManager integration', () => {
    it('updates service status after health check', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      await getReadinessStatus({ timeoutMs: 1000 });

      expect(mockUpdateServiceStatus).toHaveBeenCalledTimes(3);
      expect(mockUpdateServiceStatus).toHaveBeenCalledWith(
        'database',
        'healthy',
        undefined,
        expect.any(Number)
      );
      expect(mockUpdateServiceStatus).toHaveBeenCalledWith(
        'redis',
        'healthy',
        undefined,
        expect.any(Number)
      );
      expect(mockUpdateServiceStatus).toHaveBeenCalledWith(
        'aiService',
        'healthy',
        undefined,
        expect.any(Number)
      );
    });

    it('continues health check even if status manager update fails', async () => {
      mockUpdateServiceStatus.mockImplementation(() => {
        throw new Error('Update failed');
      });

      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const result = await getReadinessStatus({ timeoutMs: 1000 });

      // Should still complete successfully
      expect(result.status).toBe('healthy');
    });
  });

  // ==========================================================================
  // HealthCheckService Object Tests
  // ==========================================================================

  describe('HealthCheckService object', () => {
    it('exports all required methods', () => {
      expect(HealthCheckService.getLivenessStatus).toBeDefined();
      expect(HealthCheckService.getReadinessStatus).toBeDefined();
      expect(HealthCheckService.isServiceReady).toBeDefined();
      expect(HealthCheckService.registerHealthChecksWithStatusManager).toBeDefined();
    });

    it('methods work through service object', async () => {
      mockGetDatabaseClient.mockReturnValue({});
      mockCheckDatabaseHealth.mockResolvedValue(true);
      mockGetRedisClient.mockReturnValue({
        ping: jest.fn().mockResolvedValue('PONG'),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
      });

      const livenessResult = HealthCheckService.getLivenessStatus();
      expect(livenessResult.status).toBe('healthy');

      const readinessResult = await HealthCheckService.getReadinessStatus({ timeoutMs: 1000 });
      expect(readinessResult.status).toBe('healthy');

      expect(HealthCheckService.isServiceReady(readinessResult)).toBe(true);
    });
  });
});
