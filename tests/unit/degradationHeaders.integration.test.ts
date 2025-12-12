/**
 * Degradation Headers Integration Tests
 *
 * Integration-level tests for degradation headers middleware.
 *
 * Note: Unit tests for middleware internals are in tests/unit/middleware/degradationHeaders.test.ts
 */

import { Request, Response, NextFunction } from 'express';
import {
  degradationHeadersMiddleware,
  offlineModeMiddleware,
  wrapResponseWithDegradationInfo,
  getDegradationStatus,
} from '../../src/server/middleware/degradationHeaders';
import {
  ServiceStatusManager,
  DegradationLevel,
  getServiceStatusManager,
  resetServiceStatusManager,
} from '../../src/server/services/ServiceStatusManager';

// Mock the ServiceStatusManager
jest.mock('../../src/server/services/ServiceStatusManager', () => {
  const actual = jest.requireActual('../../src/server/services/ServiceStatusManager');
  return {
    ...actual,
    getServiceStatusManager: jest.fn(),
  };
});

describe('degradationHeadersMiddleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;
  let mockStatusManager: Partial<ServiceStatusManager>;

  beforeEach(() => {
    mockReq = {
      path: '/api/test',
      method: 'GET',
    };

    mockRes = {
      setHeader: jest.fn(),
      set: jest.fn(),
    };

    mockNext = jest.fn();

    mockStatusManager = {
      getDegradationLevel: jest.fn(),
      getDegradationHeaders: jest.fn(),
      isDegraded: jest.fn(),
    };

    (getServiceStatusManager as jest.Mock).mockReturnValue(mockStatusManager);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('when system is healthy', () => {
    beforeEach(() => {
      (mockStatusManager.getDegradationLevel as jest.Mock).mockReturnValue(DegradationLevel.FULL);
      (mockStatusManager.getDegradationHeaders as jest.Mock).mockReturnValue({});
      (mockStatusManager.isDegraded as jest.Mock).mockReturnValue(false);
    });

    it('should not add any degradation headers', () => {
      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).not.toHaveBeenCalled();
      expect(mockNext).toHaveBeenCalled();
    });
  });

  describe('when system is degraded', () => {
    beforeEach(() => {
      (mockStatusManager.getDegradationLevel as jest.Mock).mockReturnValue(
        DegradationLevel.DEGRADED
      );
      (mockStatusManager.getDegradationHeaders as jest.Mock).mockReturnValue({
        'X-Service-Status': 'degraded',
        'X-Degraded-Services': 'aiService',
      });
      (mockStatusManager.isDegraded as jest.Mock).mockReturnValue(true);
    });

    it('should add degradation headers', () => {
      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Service-Status', 'degraded');
      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Degraded-Services', 'aiService');
      expect(mockNext).toHaveBeenCalled();
    });
  });

  describe('when system is in MINIMAL mode', () => {
    beforeEach(() => {
      (mockStatusManager.getDegradationLevel as jest.Mock).mockReturnValue(
        DegradationLevel.MINIMAL
      );
      (mockStatusManager.getDegradationHeaders as jest.Mock).mockReturnValue({
        'X-Service-Status': 'minimal',
        'X-Degraded-Services': 'aiService,redis',
      });
      (mockStatusManager.isDegraded as jest.Mock).mockReturnValue(true);
    });

    it('should add minimal mode headers', () => {
      degradationHeadersMiddleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Service-Status', 'minimal');
      expect(mockRes.setHeader).toHaveBeenCalledWith('X-Degraded-Services', 'aiService,redis');
      expect(mockNext).toHaveBeenCalled();
    });
  });
});

describe('offlineModeMiddleware', () => {
  let mockReq: Partial<Request>;
  let mockRes: Partial<Response>;
  let mockNext: NextFunction;
  let mockStatusManager: Partial<ServiceStatusManager>;
  let mockJsonFn: jest.Mock;

  beforeEach(() => {
    mockReq = {
      path: '/api/game/moves',
      method: 'POST',
      ip: '127.0.0.1',
    };

    mockJsonFn = jest.fn().mockReturnThis();
    mockRes = {
      status: jest.fn().mockReturnThis(),
      json: mockJsonFn,
      setHeader: jest.fn().mockReturnThis(),
    };

    mockNext = jest.fn();

    mockStatusManager = {
      getDegradationLevel: jest.fn(),
    };

    (getServiceStatusManager as jest.Mock).mockReturnValue(mockStatusManager);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('when system is online', () => {
    beforeEach(() => {
      (mockStatusManager.getDegradationLevel as jest.Mock).mockReturnValue(DegradationLevel.FULL);
    });

    it('should call next without blocking', () => {
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRes.status).not.toHaveBeenCalled();
    });
  });

  describe('when system is degraded but not offline', () => {
    beforeEach(() => {
      (mockStatusManager.getDegradationLevel as jest.Mock).mockReturnValue(
        DegradationLevel.DEGRADED
      );
    });

    it('should call next without blocking', () => {
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockNext).toHaveBeenCalled();
      expect(mockRes.status).not.toHaveBeenCalled();
    });
  });

  describe('when system is offline', () => {
    beforeEach(() => {
      (mockStatusManager.getDegradationLevel as jest.Mock).mockReturnValue(
        DegradationLevel.OFFLINE
      );
    });

    it('should return 503 Service Unavailable', () => {
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.status).toHaveBeenCalledWith(503);
      expect(mockJsonFn).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            code: 'SERVICE_UNAVAILABLE',
            message: expect.stringContaining('maintenance'),
          }),
        })
      );
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should add Retry-After header', () => {
      const middleware = offlineModeMiddleware();
      middleware(mockReq as Request, mockRes as Response, mockNext);

      expect(mockRes.setHeader).toHaveBeenCalledWith('Retry-After', expect.any(String));
    });

    describe('with allowed paths', () => {
      it('should allow health check requests through', () => {
        const healthReq = { ...mockReq, path: '/health' } as Request;
        const middleware = offlineModeMiddleware(['/health', '/api/health']);
        middleware(healthReq, mockRes as Response, mockNext);

        expect(mockNext).toHaveBeenCalled();
        expect(mockRes.status).not.toHaveBeenCalled();
      });

      it('should allow prefixed paths through', () => {
        const healthReq = { ...mockReq, path: '/health/detailed' } as Request;
        const middleware = offlineModeMiddleware(['/health']);
        middleware(healthReq, mockRes as Response, mockNext);

        expect(mockNext).toHaveBeenCalled();
        expect(mockRes.status).not.toHaveBeenCalled();
      });
    });
  });
});

describe('wrapResponseWithDegradationInfo', () => {
  let mockRes: Partial<Response>;
  let mockStatusManager: Partial<ServiceStatusManager>;
  let originalJson: jest.Mock;

  beforeEach(() => {
    originalJson = jest.fn().mockReturnValue({} as Response);

    mockRes = {
      json: originalJson,
      statusCode: 200,
    } as Partial<Response>;

    mockStatusManager = {
      getSystemStatus: jest.fn(),
    };

    (getServiceStatusManager as jest.Mock).mockReturnValue(mockStatusManager);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('when system is healthy', () => {
    beforeEach(() => {
      (mockStatusManager.getSystemStatus as jest.Mock).mockReturnValue({
        degradationLevel: DegradationLevel.FULL,
        degradedServices: [],
      });
    });

    it('should not modify response body', () => {
      wrapResponseWithDegradationInfo({} as Request, mockRes as Response, jest.fn());

      // Call the wrapped json function
      mockRes.json!({ data: 'test' });

      expect(originalJson).toHaveBeenCalledWith({ data: 'test' });
    });
  });

  describe('when system is degraded', () => {
    beforeEach(() => {
      (mockStatusManager.getSystemStatus as jest.Mock).mockReturnValue({
        degradationLevel: DegradationLevel.DEGRADED,
        degradedServices: ['aiService'],
      });
    });

    it('should add _serviceStatus to response body', () => {
      wrapResponseWithDegradationInfo({} as Request, mockRes as Response, jest.fn());

      // Trigger the overridden json function
      mockRes.json!({ data: 'test' });

      expect(originalJson).toHaveBeenCalledWith(
        expect.objectContaining({
          data: 'test',
          _serviceStatus: expect.objectContaining({
            degradationLevel: DegradationLevel.DEGRADED,
            degradedServices: ['aiService'],
          }),
        })
      );
    });
  });
});

describe('getDegradationStatus', () => {
  let mockStatusManager: Partial<ServiceStatusManager>;

  beforeEach(() => {
    mockStatusManager = {
      getSystemStatus: jest.fn(),
    };

    (getServiceStatusManager as jest.Mock).mockReturnValue(mockStatusManager);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should return null when system is at full capacity', () => {
    (mockStatusManager.getSystemStatus as jest.Mock).mockReturnValue({
      degradationLevel: DegradationLevel.FULL,
      degradedServices: [],
    });

    const result = getDegradationStatus();
    expect(result).toBeNull();
  });

  it('should return degradation info when system is degraded', () => {
    (mockStatusManager.getSystemStatus as jest.Mock).mockReturnValue({
      degradationLevel: DegradationLevel.DEGRADED,
      degradedServices: ['aiService', 'redis'],
    });

    const result = getDegradationStatus();
    expect(result).not.toBeNull();
    expect(result?.isDegraded).toBe(true);
    expect(result?.level).toBe(DegradationLevel.DEGRADED);
    expect(result?.services).toEqual(['aiService', 'redis']);
  });
});

describe('Integration: Degradation Headers with Real ServiceStatusManager', () => {
  let realStatusManager: ServiceStatusManager;

  beforeEach(() => {
    resetServiceStatusManager();
    realStatusManager = new ServiceStatusManager();
    (getServiceStatusManager as jest.Mock).mockReturnValue(realStatusManager);
  });

  afterEach(() => {
    realStatusManager.destroy();
    jest.clearAllMocks();
  });

  it('should reflect actual service status in headers', () => {
    // Set up a degraded state
    realStatusManager.updateServiceStatus('database', 'healthy');
    realStatusManager.updateServiceStatus('redis', 'healthy');
    realStatusManager.updateServiceStatus('aiService', 'unhealthy', 'Service down');

    const mockReq = { path: '/api/test', method: 'GET' } as Request;
    const mockRes = {
      setHeader: jest.fn(),
    } as unknown as Response;
    const mockNext = jest.fn();

    degradationHeadersMiddleware(mockReq, mockRes, mockNext);

    expect(mockRes.setHeader).toHaveBeenCalledWith('X-Service-Status', 'degraded');
    expect(mockRes.setHeader).toHaveBeenCalledWith(
      'X-Degraded-Services',
      expect.stringContaining('aiService')
    );
  });

  it('should block requests when database is offline', () => {
    // Set database as unhealthy
    realStatusManager.updateServiceStatus('database', 'unhealthy', 'Connection refused');
    realStatusManager.updateServiceStatus('redis', 'healthy');
    realStatusManager.updateServiceStatus('aiService', 'healthy');

    const mockReq = { path: '/api/game/moves', method: 'POST', ip: '127.0.0.1' } as Request;
    const mockJsonFn = jest.fn().mockReturnThis();
    const mockRes = {
      status: jest.fn().mockReturnThis(),
      json: mockJsonFn,
      setHeader: jest.fn().mockReturnThis(),
    } as unknown as Response;
    const mockNext = jest.fn();

    const middleware = offlineModeMiddleware();
    middleware(mockReq, mockRes, mockNext);

    expect(mockRes.status).toHaveBeenCalledWith(503);
    expect(mockNext).not.toHaveBeenCalled();
  });
});
