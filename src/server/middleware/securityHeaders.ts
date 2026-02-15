/**
 * Security Headers Middleware Configuration
 *
 * Centralizes security-related middleware configuration for:
 * - Content Security Policy (CSP)
 * - HTTP Strict Transport Security (HSTS)
 * - X-Frame-Options
 * - X-Content-Type-Options
 * - Referrer-Policy
 * - Cross-Origin policies
 * - CORS configuration
 *
 * Reference: OWASP Security Headers
 * https://owasp.org/www-project-secure-headers/
 */

import helmet from 'helmet';
import cors from 'cors';
import { RequestHandler } from 'express';
import { config } from '../config';
import { logger } from '../utils/logger';

// ====================================================================
// CSP (Content Security Policy) Configuration
// ====================================================================

// No separate function needed - CSP config is inlined in helmet() call below

// ====================================================================
// Helmet Configuration
// ====================================================================

/**
 * Helmet middleware configuration with comprehensive security headers.
 */
export const securityHeaders: RequestHandler = helmet({
  // Content Security Policy - prevents XSS and data injection
  contentSecurityPolicy: {
    directives: {
      // Default fallback for any unspecified directive
      defaultSrc: ["'self'"],

      // Script sources - no inline scripts in production for XSS protection
      // In development, we may need 'unsafe-eval' for HMR/Vite tooling
      // Production includes Cloudflare Insights for analytics (injected by Cloudflare proxy)
      scriptSrc: config.isDevelopment
        ? ["'self'", "'unsafe-inline'", "'unsafe-eval'"]
        : ["'self'", 'https://static.cloudflareinsights.com', 'https://www.googletagmanager.com'],

      // Styles - allow inline for CSS-in-JS libraries (React Hot Toast, etc.)
      styleSrc: ["'self'", "'unsafe-inline'"],

      // Images - self, data URIs (for base64), and HTTPS sources
      imgSrc: ["'self'", 'data:', 'https:'],

      // Fonts - self only
      fontSrc: ["'self'"],

      // Connect (XHR, WebSocket, fetch) - self plus WebSocket protocols
      connectSrc: config.isDevelopment
        ? ["'self'", 'ws:', 'wss:', 'http://localhost:*', 'ws://localhost:*']
        : ["'self'", 'wss:', 'https://www.google-analytics.com'],

      // Media (audio/video) - none by default
      mediaSrc: ["'none'"],

      // Object/embed/applet - disabled for security
      objectSrc: ["'none'"],

      // Frames - none (prevents clickjacking via iframes)
      frameSrc: ["'none'"],

      // Child contexts (iframes, workers)
      childSrc: ["'self'"],

      // Workers (Web Workers, Service Workers)
      workerSrc: ["'self'", 'blob:'],

      // Form action targets
      formAction: ["'self'"],

      // Frame ancestors (who can embed this site)
      frameAncestors: ["'none'"],

      // Manifest files for PWA
      manifestSrc: ["'self'"],

      // Base URI for relative URLs
      baseUri: ["'self'"],

      // Upgrade insecure requests in production
      ...(config.isProduction && { upgradeInsecureRequests: [] }),
    },
  },

  // HTTP Strict Transport Security - enforces HTTPS
  hsts: config.isProduction
    ? {
        maxAge: 31536000, // 1 year in seconds
        includeSubDomains: true,
        preload: true, // Allow HSTS preload list submission
      }
    : false, // Disable HSTS in development (local dev uses HTTP)

  // X-Frame-Options - prevents clickjacking
  // helmet v7 uses frameguard for this
  frameguard: {
    action: 'deny', // DENY is more restrictive than SAMEORIGIN
  },

  // X-Content-Type-Options - prevents MIME-type sniffing
  // Enabled by default in helmet, explicitly confirming
  noSniff: true,

  // X-DNS-Prefetch-Control - controls browser DNS prefetching
  dnsPrefetchControl: {
    allow: false, // Disable to prevent privacy leaks
  },

  // Referrer-Policy - controls referrer information sent with requests
  referrerPolicy: {
    policy: 'strict-origin-when-cross-origin',
  },

  // X-Permitted-Cross-Domain-Policies - restricts Adobe Flash/PDF cross-domain
  permittedCrossDomainPolicies: {
    permittedPolicies: 'none',
  },

  // Cross-Origin-Embedder-Policy - controls cross-origin resource loading
  // Disabled to allow loading cross-origin resources (fonts, CDN assets)
  crossOriginEmbedderPolicy: false,

  // Cross-Origin-Opener-Policy - isolates browser context
  crossOriginOpenerPolicy: {
    policy: 'same-origin',
  },

  // Cross-Origin-Resource-Policy - controls cross-origin resource sharing
  crossOriginResourcePolicy: {
    policy: 'same-site',
  },

  // Origin-Agent-Cluster - request dedicated agent cluster
  originAgentCluster: true,

  // Hide X-Powered-By header (removed automatically by helmet)
  hidePoweredBy: true,

  // IE-specific header - disable since IE is deprecated
  ieNoOpen: true,

  // Legacy XSS filter header (deprecated but still set by some policies)
  // Modern browsers ignore this, but it doesn't hurt
  xssFilter: true,
});

// ====================================================================
// CORS Configuration
// ====================================================================

/**
 * Parse and validate allowed origins from configuration.
 * Supports both string arrays and regex patterns.
 */
const getAllowedOrigins = (): (string | RegExp)[] => {
  const origins = config.server.allowedOrigins;

  // In development, also allow common local development URLs
  if (config.isDevelopment) {
    return [...origins, /^http:\/\/localhost:\d+$/, /^http:\/\/127\.0\.0\.1:\d+$/];
  }

  return origins;
};

/**
 * CORS configuration with proper origin whitelisting.
 *
 * NOTE: Since the application uses JWT tokens in headers (not cookies for
 * API auth), CSRF is less of a concern for API calls. However, we still
 * configure CORS properly for defense-in-depth.
 *
 * Cookie-based auth (refresh tokens) uses sameSite: 'strict' which provides
 * CSRF protection for cookie-carrying requests.
 */
export const corsMiddleware: RequestHandler = cors({
  // Origin validation - uses the allowedOrigins from config
  origin: (origin, callback) => {
    // Allow requests with no origin (server-to-server, curl, Postman)
    if (!origin) {
      callback(null, true);
      return;
    }

    const allowedOrigins = getAllowedOrigins();

    // Check if origin matches any allowed origin (string or regex)
    const isAllowed = allowedOrigins.some((allowed) => {
      if (typeof allowed === 'string') {
        return allowed === origin;
      }
      // RegExp pattern
      return allowed.test(origin);
    });

    if (isAllowed) {
      callback(null, true);
    } else {
      // Log rejected origins in non-production for debugging
      if (!config.isProduction) {
        logger.warn('CORS rejected origin', { origin, event: 'cors_rejected' });
      }
      callback(new Error(`Origin ${origin} not allowed by CORS`));
    }
  },

  // Allow credentials (cookies) for refresh token requests
  credentials: true,

  // Allowed HTTP methods
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],

  // Allowed request headers
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Request-ID',
    'X-Client-Version',
    'Accept',
    'Accept-Language',
    'Accept-Encoding',
  ],

  // Headers exposed to the client
  exposedHeaders: [
    'X-Request-ID',
    'X-RateLimit-Limit',
    'X-RateLimit-Remaining',
    'X-RateLimit-Reset',
  ],

  // Preflight request cache duration (in seconds)
  maxAge: 600, // 10 minutes

  // Pass preflight requests to the next handler
  preflightContinue: false,

  // Provide 204 for preflight OPTIONS requests
  optionsSuccessStatus: 204,
});

// ====================================================================
// Origin Validation Middleware
// ====================================================================

/**
 * Additional origin validation for state-changing requests.
 * This provides defense-in-depth by validating Origin/Referer headers
 * on mutating requests.
 *
 * NOTE: This is complementary to CORS. CORS handles browser preflight and
 * response headers, while this validates the actual Origin header on the
 * request for defense-in-depth.
 */
export const originValidationMiddleware: RequestHandler = (req, res, next) => {
  // Skip for safe HTTP methods (GET, HEAD, OPTIONS)
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    return next();
  }

  // Skip origin validation in development for easier testing
  if (config.isDevelopment) {
    return next();
  }

  const origin = req.get('Origin');
  const referer = req.get('Referer');

  // If no Origin header, check Referer (some older browsers)
  const requestOrigin = origin || (referer ? new URL(referer).origin : null);

  // Allow requests without origin (server-to-server, mobile apps)
  if (!requestOrigin) {
    return next();
  }

  const allowedOrigins = getAllowedOrigins();

  const isAllowed = allowedOrigins.some((allowed) => {
    if (typeof allowed === 'string') {
      return allowed === requestOrigin;
    }
    return allowed.test(requestOrigin);
  });

  if (!isAllowed) {
    res.status(403).json({
      success: false,
      error: {
        message: 'Origin not allowed',
        code: 'FORBIDDEN_ORIGIN',
        timestamp: new Date(),
      },
    });
    return;
  }

  next();
};

// ====================================================================
// Export all security middleware
// ====================================================================

export const securityMiddleware = {
  headers: securityHeaders,
  cors: corsMiddleware,
  originValidation: originValidationMiddleware,
};
