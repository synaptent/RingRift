/**
 * Server bootstrap failure tests
 *
 * These tests exercise startup failure paths and ensure that:
 * - Database connection failures cause a logged error and process exit
 * - Redis connection failures are logged as warnings but do not prevent
 *   the HTTP server from starting (as per local-dev runbook)
 *
 * We intentionally mock the Node http server and backend services so that
 * importing src/server/index.ts does not open real network listeners.
 */

const mockConnectDatabase = jest.fn();
const mockConnectRedis = jest.fn();

const mockEnforceAppTopology = jest.fn();

// NOTE: src/server/index.ts reads many config fields at module-import time.
// Keep this mock in sync with the canonical config shape.
const mockConfig = {
  nodeEnv: 'development',
  isProduction: false,
  isDevelopment: true,
  isTest: false,
  app: {
    version: 'test',
    topology: 'single',
  },
  server: {
    port: 3000,
    host: '127.0.0.1',
  },
  metrics: {
    enabled: false,
    exposeOnMain: false,
    port: 9090,
    apiKey: undefined,
  },
  healthChecks: {
    enabled: false,
  },
  featureFlags: {
    orchestrator: {
      adapterEnabled: true,
      allowlistUsers: [],
      denylistUsers: [],
      circuitBreaker: {
        enabled: true,
        errorThresholdPercent: 5,
        errorWindowSeconds: 300,
      },
      latencyThresholdMs: 500,
    },
    analysisMode: {
      enabled: false,
    },
    sandboxAi: {
      enabled: false,
    },
    httpMoveHarness: {
      enabled: false,
      timeoutMs: 30_000,
    },
  },
  rules: {
    mode: 'ts',
  },
} as any;

jest.mock('../../src/server/config', () => ({
  config: mockConfig,
  enforceAppTopology: mockEnforceAppTopology,
}));

jest.mock('http', () => {
  const actual = jest.requireActual('http');
  const listen = jest.fn((_port: number, cb?: () => void) => {
    if (typeof cb === 'function') cb();
  });
  const close = jest.fn((cb?: () => void) => {
    if (typeof cb === 'function') cb();
  });

  return {
    ...actual,
    createServer: jest.fn(() => ({
      listen,
      close,
    })),
  };
});

jest.mock('../../src/server/database/connection', () => ({
  connectDatabase: mockConnectDatabase,
  getDatabaseClient: jest.fn(() => null),
}));

jest.mock('../../src/server/cache/redis', () => ({
  connectRedis: mockConnectRedis,
}));

const mockLogger = {
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  debug: jest.fn(),
};

jest.mock('../../src/server/utils/logger', () => ({
  logger: mockLogger,
  httpLogger: jest.fn(),
}));

jest.mock('../../src/server/services/MetricsService', () => ({
  getMetricsService: () => ({
    getContentType: jest.fn(() => 'text/plain'),
    getMetrics: jest.fn(async () => ''),
    updateServiceStatus: jest.fn(),
    updateDegradationLevel: jest.fn(),
    setOrchestratorRolloutPercentage: jest.fn(),
  }),
}));

jest.mock('../../src/server/services/ServiceStatusManager', () => ({
  getServiceStatusManager: () => ({
    on: jest.fn(),
  }),
}));

jest.mock('../../src/server/websocket/server', () => ({
  WebSocketServer: jest.fn().mockImplementation(() => ({})),
}));

describe('server bootstrap failures', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('exits with code 1 when database connection fails', async () => {
    mockConnectDatabase.mockRejectedValueOnce(new Error('Database unavailable'));

    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(((_code?: number) => {
      // prevent Jest from exiting the process
    }) as any);

    await new Promise<void>((resolve) => {
      jest.isolateModules(() => {
        // Importing index.ts will invoke startServer() once.
        require('../../src/server/index');
      });
      // Allow the async startServer() rejection path to run.
      setImmediate(resolve);
    });

    expect(mockConnectDatabase).toHaveBeenCalledTimes(1);
    expect(mockLogger.error).toHaveBeenCalledWith('Failed to start server:', expect.any(Error));
    expect(exitSpy).toHaveBeenCalledWith(1);

    exitSpy.mockRestore();
  });

  it('logs a warning but continues startup when Redis connection fails', async () => {
    mockConnectDatabase.mockResolvedValueOnce(undefined);
    mockConnectRedis.mockRejectedValueOnce(new Error('Redis unavailable'));

    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(((_code?: number) => {
      // prevent Jest from exiting the process
    }) as any);

    await new Promise<void>((resolve) => {
      jest.isolateModules(() => {
        require('../../src/server/index');
      });
      setImmediate(resolve);
    });

    expect(mockConnectDatabase).toHaveBeenCalledTimes(1);
    expect(mockConnectRedis).toHaveBeenCalledTimes(1);
    expect(mockLogger.warn).toHaveBeenCalledWith(
      'Redis connection failed; continuing without Redis',
      expect.objectContaining({
        error: expect.any(String),
      })
    );
    expect(exitSpy).not.toHaveBeenCalled();

    exitSpy.mockRestore();
  });

  it('logs error and exits when enforceAppTopology throws', async () => {
    mockEnforceAppTopology.mockImplementationOnce(() => {
      throw new Error('Unsupported topology');
    });

    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(((_code?: number) => {
      // prevent Jest from exiting the process
    }) as any);

    await new Promise<void>((resolve) => {
      jest.isolateModules(() => {
        require('../../src/server/index');
      });
      setImmediate(resolve);
    });

    expect(mockEnforceAppTopology).toHaveBeenCalledWith(mockConfig);
    expect(mockConnectDatabase).not.toHaveBeenCalled();
    expect(mockConnectRedis).not.toHaveBeenCalled();
    expect(mockLogger.error).toHaveBeenCalledWith('Failed to start server:', expect.any(Error));
    expect(exitSpy).toHaveBeenCalledWith(1);

    exitSpy.mockRestore();
  });
});
