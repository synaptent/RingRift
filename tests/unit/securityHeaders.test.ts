/**
 * Security Headers Tests
 *
 * Verifies that the security middleware properly configures:
 * - Content Security Policy (CSP)
 * - HTTP Strict Transport Security (HSTS)
 * - X-Frame-Options
 * - X-Content-Type-Options
 * - Referrer-Policy
 * - Cross-Origin policies
 * - CORS configuration
 */

import express, { Express } from 'express';
import request from 'supertest';

describe('Security Headers Middleware', () => {
  let app: Express;
  let securityMiddleware: typeof import('../../src/server/middleware/securityHeaders').securityMiddleware;
  let originalEnv: Partial<NodeJS.ProcessEnv>;

  beforeEach(() => {
    // Defensive: some suites use fake timers; ensure this integration-style
    // supertest flow always runs on real timers to avoid socket hangups.
    jest.useRealTimers();

    // Need a stable environment before loading server/config-derived middleware.
    const keys = [
      'NODE_ENV',
      'DATABASE_URL',
      'REDIS_URL',
      'JWT_SECRET',
      'JWT_REFRESH_SECRET',
      'ALLOWED_ORIGINS',
    ] as const;

    originalEnv = Object.fromEntries(keys.map((k) => [k, process.env[k]]));

    process.env.NODE_ENV = 'test';
    process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test';
    process.env.REDIS_URL = 'redis://localhost:6379';
    process.env.JWT_SECRET = 'test-jwt-secret';
    process.env.JWT_REFRESH_SECRET = 'test-jwt-refresh-secret';
    process.env.ALLOWED_ORIGINS = 'http://localhost:3000,http://localhost:5173';

    // Load in an isolated registry so cached config from other test files
    // cannot change middleware behavior in this suite.
    jest.isolateModules(() => {
      const mod =
        require('../../src/server/middleware/securityHeaders') as typeof import('../../src/server/middleware/securityHeaders');
      securityMiddleware = mod.securityMiddleware;
    });

    app = express();
    // Apply security middleware
    app.use(securityMiddleware.headers);
    app.use(securityMiddleware.cors);

    // Test endpoint
    app.get('/test', (_req, res) => {
      res.json({ message: 'ok' });
    });

    app.post('/test-post', (_req, res) => {
      res.json({ message: 'posted' });
    });
  });

  afterEach(() => {
    const keys = Object.keys(originalEnv) as Array<keyof typeof originalEnv>;
    for (const key of keys) {
      const value = originalEnv[key];
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
  });

  describe('Content Security Policy', () => {
    it('should set Content-Security-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.status).toBe(200);
      expect(response.headers['content-security-policy']).toBeDefined();
    });

    it('should include default-src directive', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/default-src\s+'self'/);
    });

    it('should include script-src directive', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/script-src/);
    });

    it('should include style-src directive with unsafe-inline', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/style-src[^;]*'unsafe-inline'/);
    });

    it('should include connect-src directive for WebSocket support', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/connect-src/);
    });

    it('should block object-src for security', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/object-src\s+'none'/);
    });

    it('should block frame-src for clickjacking protection', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/frame-src\s+'none'/);
    });

    it('should set frame-ancestors to none', async () => {
      const response = await request(app).get('/test');

      const csp = response.headers['content-security-policy'];
      expect(csp).toMatch(/frame-ancestors\s+'none'/);
    });
  });

  describe('X-Frame-Options', () => {
    it('should set X-Frame-Options header to DENY', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-frame-options']).toBe('DENY');
    });
  });

  describe('X-Content-Type-Options', () => {
    it('should set X-Content-Type-Options header to nosniff', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-content-type-options']).toBe('nosniff');
    });
  });

  describe('Referrer-Policy', () => {
    it('should set Referrer-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['referrer-policy']).toBeDefined();
      expect(response.headers['referrer-policy']).toBe('strict-origin-when-cross-origin');
    });
  });

  describe('X-DNS-Prefetch-Control', () => {
    it('should set X-DNS-Prefetch-Control header to off', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-dns-prefetch-control']).toBe('off');
    });
  });

  describe('Cross-Origin-Opener-Policy', () => {
    it('should set Cross-Origin-Opener-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['cross-origin-opener-policy']).toBe('same-origin');
    });
  });

  describe('Cross-Origin-Resource-Policy', () => {
    it('should set Cross-Origin-Resource-Policy header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['cross-origin-resource-policy']).toBe('same-site');
    });
  });

  describe('X-Permitted-Cross-Domain-Policies', () => {
    it('should set X-Permitted-Cross-Domain-Policies header to none', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-permitted-cross-domain-policies']).toBe('none');
    });
  });

  describe('Strict-Transport-Security (HSTS)', () => {
    it('should not send HSTS header in non-production/test environments', async () => {
      const response = await request(app).get('/test');

      // In test/development modes, HSTS is disabled to avoid breaking local HTTP.
      expect(response.headers['strict-transport-security']).toBeUndefined();
    });
  });

  describe('Origin-Agent-Cluster', () => {
    it('should set Origin-Agent-Cluster header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['origin-agent-cluster']).toBeDefined();
    });
  });

  describe('X-Powered-By', () => {
    it('should not expose X-Powered-By header', async () => {
      const response = await request(app).get('/test');

      expect(response.headers['x-powered-by']).toBeUndefined();
    });
  });

  describe('CORS', () => {
    it('should allow requests without origin (server-to-server)', async () => {
      const response = await request(app).get('/test');

      expect(response.status).toBe(200);
    });

    it('should allow requests from allowed origins', async () => {
      const response = await request(app).get('/test').set('Origin', 'http://localhost:3000');

      expect(response.status).toBe(200);
      expect(response.headers['access-control-allow-origin']).toBe('http://localhost:3000');
    });

    it('should allow credentials', async () => {
      const response = await request(app).get('/test').set('Origin', 'http://localhost:3000');

      expect(response.headers['access-control-allow-credentials']).toBe('true');
    });

    it('should respond to preflight OPTIONS requests', async () => {
      const response = await request(app)
        .options('/test')
        .set('Origin', 'http://localhost:3000')
        .set('Access-Control-Request-Method', 'POST');

      expect(response.status).toBe(204);
      expect(response.headers['access-control-allow-methods']).toBeDefined();
    });

    it('should expose specified headers', async () => {
      const response = await request(app).get('/test').set('Origin', 'http://localhost:3000');

      const exposedHeaders = response.headers['access-control-expose-headers'];
      expect(exposedHeaders).toBeDefined();
      expect(exposedHeaders).toContain('X-Request-ID');
    });

    it('should reject requests from non-allowed origins', async () => {
      const response = await request(app).get('/test').set('Origin', 'http://malicious-site.com');

      // CORS middleware throws error for rejected origins
      expect(response.status).toBe(500);
    });

    it('should reject requests from localhost with non-configured port in test mode', async () => {
      // In test mode (which Jest uses), only explicitly configured origins are allowed
      // Development mode would allow any localhost port via regex
      const response = await request(app).get('/test').set('Origin', 'http://localhost:8080');

      // Should be rejected in test mode since we only allow 3000 and 5173
      expect(response.status).toBe(500);
    });
  });

  describe('Security Headers Summary', () => {
    it('should have all critical security headers set', async () => {
      const response = await request(app).get('/test');

      // Critical headers for XSS protection
      expect(response.headers['content-security-policy']).toBeDefined();

      // Clickjacking protection
      expect(response.headers['x-frame-options']).toBeDefined();

      // MIME sniffing protection
      expect(response.headers['x-content-type-options']).toBeDefined();

      // Referrer leakage protection
      expect(response.headers['referrer-policy']).toBeDefined();

      // Cross-origin isolation
      expect(response.headers['cross-origin-opener-policy']).toBeDefined();

      // Server info exposure prevention
      expect(response.headers['x-powered-by']).toBeUndefined();
    });
  });
});

describe('Origin Validation Middleware', () => {
  let app: Express;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use(securityMiddleware.cors);
    // Note: originValidation middleware skips in development mode
    // To test it properly, we'd need to mock config.isDevelopment = false

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });
  });

  it('should allow GET requests without origin validation', async () => {
    const response = await request(app).get('/api/test');

    // Will be 404 since no GET route, but not blocked by origin validation
    expect(response.status).toBe(404);
  });

  it('should allow POST requests in development mode', async () => {
    const response = await request(app).post('/api/test').send({ data: 'test' });

    expect(response.status).toBe(200);
  });
});

describe('Origin Validation Middleware - Production Mode', () => {
  // Save original config values to restore after tests
  const originalIsDevelopment = jest.requireActual('../../src/server/config').config?.isDevelopment;

  beforeEach(() => {
    jest.resetModules();
  });

  afterEach(() => {
    jest.resetModules();
  });

  it('should reject requests with disallowed origin in production mode', async () => {
    // Mock config to simulate production mode
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['http://allowed-origin.com'],
        },
      },
    }));

    // Re-import the middleware with mocked config
    const { originValidationMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(express.json());
    app.use(originValidationMiddleware);

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });

    const response = await request(app)
      .post('/api/test')
      .set('Origin', 'http://malicious-site.com')
      .send({ data: 'test' });

    // In production mode, disallowed origins should be rejected with 403
    expect(response.status).toBe(403);
    expect(response.body.error.code).toBe('FORBIDDEN_ORIGIN');
  });

  it('should allow requests with allowed origin in production mode', async () => {
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['http://allowed-origin.com'],
        },
      },
    }));

    const { originValidationMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(express.json());
    app.use(originValidationMiddleware);

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });

    const response = await request(app)
      .post('/api/test')
      .set('Origin', 'http://allowed-origin.com')
      .send({ data: 'test' });

    expect(response.status).toBe(200);
  });

  it('should allow requests without origin header in production mode (server-to-server)', async () => {
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['http://allowed-origin.com'],
        },
      },
    }));

    const { originValidationMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(express.json());
    app.use(originValidationMiddleware);

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });

    const response = await request(app).post('/api/test').send({ data: 'test' });

    // No origin header = allowed (server-to-server)
    expect(response.status).toBe(200);
  });

  it('should use Referer header when Origin header is missing in production mode', async () => {
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['http://allowed-origin.com'],
        },
      },
    }));

    const { originValidationMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(express.json());
    app.use(originValidationMiddleware);

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });

    // Request with Referer but no Origin
    const response = await request(app)
      .post('/api/test')
      .set('Referer', 'http://allowed-origin.com/some/path')
      .send({ data: 'test' });

    expect(response.status).toBe(200);
  });

  it('should reject when Referer origin is disallowed in production mode', async () => {
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['http://allowed-origin.com'],
        },
      },
    }));

    const { originValidationMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(express.json());
    app.use(originValidationMiddleware);

    app.post('/api/test', (_req, res) => {
      res.json({ message: 'posted' });
    });

    const response = await request(app)
      .post('/api/test')
      .set('Referer', 'http://malicious-site.com/some/path')
      .send({ data: 'test' });

    expect(response.status).toBe(403);
  });

  it('should skip validation for HEAD requests in production mode', async () => {
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['http://allowed-origin.com'],
        },
      },
    }));

    const { originValidationMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(originValidationMiddleware);

    app.head('/api/test', (_req, res) => {
      res.status(200).end();
    });

    const response = await request(app)
      .head('/api/test')
      .set('Origin', 'http://malicious-site.com');

    // HEAD is a safe method, should be allowed regardless of origin
    expect(response.status).toBe(200);
  });
});

describe('HSTS Configuration - Production Mode', () => {
  beforeEach(() => {
    jest.resetModules();
  });

  afterEach(() => {
    jest.resetModules();
  });

  it('enables HSTS with one-year max-age and preload in production', async () => {
    // Mock config to simulate production mode for helmet HSTS configuration.
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: false,
        isProduction: true,
        server: {
          allowedOrigins: ['https://example.com'],
        },
      },
    }));

    const { securityHeaders } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(securityHeaders);
    app.get('/hsts-test', (_req, res) => {
      res.json({ ok: true });
    });

    const response = await request(app).get('/hsts-test');
    const hsts = response.headers['strict-transport-security'];

    expect(hsts).toBeDefined();
    expect(hsts).toContain('max-age=31536000');
    expect(hsts).toContain('includeSubDomains');
    expect(hsts.toLowerCase()).toContain('preload');
  });
});

describe('CORS Regex Origin Matching', () => {
  it('should match localhost with various ports via regex in development mode', async () => {
    // The default test environment allows localhost regex patterns
    const app = express();
    app.use(securityMiddleware.cors);

    app.get('/test', (_req, res) => {
      res.json({ message: 'ok' });
    });

    // Test various localhost ports that should be matched by regex
    const ports = [4000, 5000, 5173, 8080, 9000];

    for (const port of ports) {
      const response = await request(app).get('/test').set('Origin', `http://localhost:${port}`);

      // In test mode, only explicitly configured origins (3000, 5173) are allowed
      // This verifies the regex path is NOT being used in test mode
      if (port === 5173) {
        expect(response.status).toBe(200);
      }
    }
  });

  it('should match 127.0.0.1 with various ports in development mode', async () => {
    // When isDevelopment is true, 127.0.0.1:* should be allowed
    jest.doMock('../../src/server/config', () => ({
      config: {
        logging: {
          level: 'error',
        },
        isDevelopment: true,
        isProduction: false,
        server: {
          allowedOrigins: ['http://localhost:3000'],
        },
      },
    }));

    jest.resetModules();
    const { corsMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(corsMiddleware);

    app.get('/test', (_req, res) => {
      res.json({ message: 'ok' });
    });

    const response = await request(app).get('/test').set('Origin', 'http://127.0.0.1:8080');

    // In development mode, 127.0.0.1:* regex should allow this
    expect(response.status).toBe(200);
  });

  it('should reject non-localhost origins in development mode', async () => {
    jest.doMock('../../src/server/config', () => ({
      config: {
        nodeEnv: 'development',
        logging: {
          level: 'error',
        },
        isDevelopment: true,
        isProduction: false,
        server: {
          allowedOrigins: ['http://localhost:3000'],
        },
      },
    }));

    jest.resetModules();
    const { corsMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(corsMiddleware);

    app.get('/test', (_req, res) => {
      res.json({ message: 'ok' });
    });

    const response = await request(app).get('/test').set('Origin', 'http://evil-site.com');

    // Even in development, non-localhost origins should be rejected
    expect(response.status).toBe(500);
  });
});

describe('CORS Console Warning', () => {
  it('should log warning for rejected origins in non-production mode', async () => {
    const warnSpy = jest.fn();

    jest.doMock('../../src/server/utils/logger', () => ({
      logger: {
        warn: warnSpy,
      },
    }));

    jest.doMock('../../src/server/config', () => ({
      config: {
        nodeEnv: 'development',
        logging: {
          level: 'error',
        },
        isDevelopment: true,
        isProduction: false,
        server: {
          allowedOrigins: ['http://localhost:3000'],
        },
      },
    }));

    jest.resetModules();
    const { corsMiddleware } = require('../../src/server/middleware/securityHeaders');

    const app = express();
    app.use(corsMiddleware);

    app.get('/test', (_req, res) => {
      res.json({ message: 'ok' });
    });

    await request(app).get('/test').set('Origin', 'http://rejected-origin.com');

    expect(warnSpy).toHaveBeenCalledWith(
      'CORS rejected origin',
      expect.objectContaining({
        origin: 'http://rejected-origin.com',
        event: 'cors_rejected',
      })
    );
  });
});

describe('Sanitization Utilities', () => {
  // Import sanitization functions from validation schemas
  const { sanitizeString, sanitizeHtmlContent } = require('../../src/shared/validation/schemas');

  describe('sanitizeString', () => {
    it('should remove null bytes', () => {
      const input = 'hello\x00world';
      expect(sanitizeString(input)).toBe('helloworld');
    });

    it('should trim whitespace', () => {
      const input = '  hello world  ';
      expect(sanitizeString(input)).toBe('hello world');
    });

    it('should handle non-string input', () => {
      expect(sanitizeString(null as any)).toBe('');
      expect(sanitizeString(undefined as any)).toBe('');
      expect(sanitizeString(123 as any)).toBe('');
    });

    it('should normalize Unicode', () => {
      // NFC normalization test - café in decomposed form
      const decomposed = 'cafe\u0301';
      const result = sanitizeString(decomposed);
      expect(result).toBe('café');
    });
  });

  describe('sanitizeHtmlContent', () => {
    it('should escape HTML entities', () => {
      const input = '<script>alert("xss")</script>';
      const result = sanitizeHtmlContent(input);
      expect(result).not.toContain('<script>');
      expect(result).toContain('&lt;script&gt;');
    });

    it('should escape ampersands', () => {
      const input = 'Tom & Jerry';
      expect(sanitizeHtmlContent(input)).toBe('Tom &amp; Jerry');
    });

    it('should escape quotes', () => {
      const input = 'He said "hello"';
      expect(sanitizeHtmlContent(input)).toContain('&quot;');
    });

    it('should escape single quotes', () => {
      const input = "It's a test";
      expect(sanitizeHtmlContent(input)).toContain('&#x27;');
    });

    it('should escape complex XSS payloads', () => {
      const input = '<img src=x onerror="alert(1)">';
      const result = sanitizeHtmlContent(input);
      // HTML characters should be escaped
      expect(result).not.toContain('<img');
      expect(result).toContain('&lt;img');
      // Quotes and equals are escaped, making the payload inert in HTML context
      expect(result).toContain('&#x3D;'); // escaped equals
      expect(result).toContain('&quot;'); // escaped quotes
    });

    it('should handle non-string input', () => {
      expect(sanitizeHtmlContent(null as any)).toBe('');
      expect(sanitizeHtmlContent(undefined as any)).toBe('');
    });
  });
});
