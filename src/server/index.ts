import * as Sentry from '@sentry/node';
import express, { type Request, type Response, type NextFunction } from 'express';
import { createServer } from 'http';
import path from 'path';
import compression from 'compression';
import { WebSocketServer } from './websocket/server';
import { setupRoutes } from './routes';
import { errorHandler } from './middleware/errorHandler';
import { rateLimiter } from './middleware/rateLimiter';
import { requestContext } from './middleware/requestContext';
import { apiRequestLogger } from './middleware/requestLogger';
import { securityMiddleware } from './middleware/securityHeaders';
import { metricsMiddleware } from './middleware/metricsMiddleware';
import { logger } from './utils/logger';
import { connectDatabase, disconnectDatabase, getDatabaseClient } from './database/connection';
import { connectRedis, disconnectRedis } from './cache/redis';
import {
  createDataRetentionService,
  type DataRetentionService,
} from './services/DataRetentionService';
import client from 'prom-client';
import { config, enforceAppTopology } from './config';
import { HealthCheckService, isServiceReady } from './services/HealthCheckService';
import { getMetricsService } from './services/MetricsService';
import { getServiceStatusManager } from './services/ServiceStatusManager';

// Initialize Sentry for server-side error tracking (only when DSN is configured)
const sentryDsn = process.env.SENTRY_DSN;
if (sentryDsn) {
  Sentry.init({
    dsn: sentryDsn,
    environment: config.isDevelopment ? 'development' : 'production',
    tracesSampleRate: 0.1,
  });
}

const app = express();
const server = createServer(app);
let metricsServer: ReturnType<typeof createServer> | null = null;
let retentionService: DataRetentionService | null = null;
let retentionTimerId: ReturnType<typeof setTimeout> | null = null;

/**
 * Calculate milliseconds until next 3 AM UTC.
 * Used to schedule daily data retention tasks.
 */
function msUntilNext3AmUtc(): number {
  const now = new Date();
  const next3Am = new Date(now);
  next3Am.setUTCHours(3, 0, 0, 0);

  // If it's already past 3 AM UTC today, schedule for tomorrow
  if (now >= next3Am) {
    next3Am.setUTCDate(next3Am.getUTCDate() + 1);
  }

  return next3Am.getTime() - now.getTime();
}

/**
 * Schedule and run data retention tasks daily at 3 AM UTC.
 * GDPR compliance requires automated cleanup of soft-deleted data.
 */
function scheduleDataRetentionTask(): void {
  const runAndReschedule = async () => {
    if (retentionService) {
      try {
        const report = await retentionService.runRetentionTasks();
        logger.info('Scheduled data retention tasks completed', { report });
      } catch (error) {
        logger.error('Scheduled data retention tasks failed', {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    // Schedule next run at 3 AM UTC tomorrow
    const msUntilNext = msUntilNext3AmUtc();
    retentionTimerId = setTimeout(runAndReschedule, msUntilNext);
    logger.debug('Next data retention run scheduled', {
      msUntilNext,
      nextRunAt: new Date(Date.now() + msUntilNext).toISOString(),
    });
  };

  // Schedule first run
  const msUntilFirst = msUntilNext3AmUtc();
  retentionTimerId = setTimeout(runAndReschedule, msUntilFirst);
  logger.info('Data retention scheduler initialized', {
    firstRunAt: new Date(Date.now() + msUntilFirst).toISOString(),
    retentionConfig: retentionService?.getConfig(),
  });
}

// Trust proxy headers for accurate client IP detection behind nginx/Cloudflare.
// This allows rate limiters and logging to use the real client IP from
// X-Forwarded-For instead of always seeing 127.0.0.1 (the proxy IP).
app.set('trust proxy', true);

const metricsEnabled = config.metrics.enabled;
const metricsExposeOnMain = config.metrics.exposeOnMain;
const metricsUseSeparateServer = metricsEnabled && !metricsExposeOnMain;
const healthChecksEnabled = config.healthChecks.enabled;

// Register default Prometheus metrics for the Node.js process.
if (metricsEnabled) {
  client.collectDefaultMetrics();
}

// Initialize MetricsService singleton (registers all custom metrics)
const metricsService = getMetricsService();

// Wire up ServiceStatusManager to emit metrics on status changes
const statusManager = getServiceStatusManager();
statusManager.on('statusChange', (service, _oldStatus, newStatus) => {
  metricsService.updateServiceStatus(service, newStatus);
});
statusManager.on('degradationLevelChange', (_oldLevel, newLevel) => {
  metricsService.updateDegradationLevel(newLevel);
});
// Security Middleware - helmet for security headers, CORS for cross-origin requests
// Configuration is centralized in ./middleware/securityHeaders.ts
app.use(securityMiddleware.headers);
app.use(securityMiddleware.cors);

app.use(compression());

// Attach lightweight per-request context for correlation IDs. This middleware
// runs before any API routes and before the global error handler so that
// requestId is available throughout the HTTP pipeline. It also establishes
// the AsyncLocalStorage context for automatic log correlation.
app.use(requestContext);

// Structured request logging middleware (replaces Morgan).
// Outputs JSON logs suitable for log aggregation (ELK, CloudWatch, Datadog).
app.use(apiRequestLogger);

// HTTP metrics middleware - collects request duration, count, and sizes
// Placed after requestContext so correlation IDs are available, and after
// requestLogger so logging completes before metrics are recorded.
if (metricsEnabled) {
  app.use(metricsMiddleware);
}

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check endpoints - placed BEFORE rate limiter to bypass rate limiting.
// These endpoints are called frequently by orchestrators (Kubernetes, load balancers)
// and should not be rate limited or require authentication.
if (healthChecksEnabled) {
  /**
   * Liveness probe endpoint - /health or /healthz
   * Returns 200 if the process is alive and responding.
   * Fast response with minimal checks.
   * Used by orchestrators to restart dead containers.
   */
  app.get(['/health', '/healthz'], (_req: Request, res: Response) => {
    const status = HealthCheckService.getLivenessStatus();
    res.status(200).json(status);
  });

  /**
   * Readiness probe endpoint - /ready or /readyz
   * Returns 200 if the service is ready to serve traffic.
   * Checks database, Redis, and AI service connectivity.
   * Returns 503 if critical dependencies (database) are unavailable.
   * Used by orchestrators to remove instances from service endpoints.
   */
  app.get(['/ready', '/readyz'], async (_req: Request, res: Response) => {
    const status = await HealthCheckService.getReadinessStatus();
    const httpStatus = isServiceReady(status) ? 200 : 503;
    res.status(httpStatus).json(status);
  });
}

// Rate limiting - placed AFTER health endpoints.
// Apply the general API limiter only to non-auth, non-game API routes.
// Auth endpoints have their own dedicated limiters in src/server/routes/auth.ts.
// Game creation and game routes use their own quotas in src/server/routes/game.ts.
app.use((req, res, next) => {
  const path = req.path || req.originalUrl || '';

  // Only apply the generic API limiter to /api/* routes.
  if (!path.startsWith('/api/')) {
    return next();
  }

  // Skip /api/auth/*: these routes are protected by auth-specific limiters.
  if (path.startsWith('/api/auth/')) {
    return next();
  }

  // Skip /api/games* so that game routes rely on their dedicated limiters
  // (adaptiveRateLimiter('game','api') and gameCreateUser/gameCreateIp quotas)
  // without being double-limited by the generic API limiter. This is especially
  // important for load testing scenarios that generate high game creation rates.
  if (path.startsWith('/api/games')) {
    return next();
  }

  return rateLimiter(req, res, next);
});

// Prometheus metrics endpoint
// Uses MetricsService which consolidates all custom metrics with the default
// Node.js metrics (memory, CPU, event loop, GC) from prom-client.
//
// SECURITY: When METRICS_API_KEY is configured, the endpoint requires
// authentication via Bearer token or X-Metrics-Key header.
const metricsHandler = async (req: Request, res: Response) => {
  // Check authentication if API key is configured
  const metricsApiKey = config.metrics.apiKey;
  if (metricsApiKey) {
    const authHeader = req.headers.authorization;
    const metricsKeyHeader = req.headers['x-metrics-key'];

    const providedKey =
      (typeof authHeader === 'string' && authHeader.startsWith('Bearer ')
        ? authHeader.slice(7)
        : undefined) || (typeof metricsKeyHeader === 'string' ? metricsKeyHeader : undefined);

    if (!providedKey || providedKey !== metricsApiKey) {
      logger.warn('Unauthorized /metrics access attempt', {
        ip: req.ip,
        hasAuthHeader: !!authHeader,
        hasMetricsKeyHeader: !!metricsKeyHeader,
      });
      res.status(401).json({
        success: false,
        error: {
          code: 'UNAUTHORIZED',
          message: 'Valid API key required for metrics access',
        },
      });
      return;
    }
  }

  try {
    res.set('Content-Type', metricsService.getContentType());
    const metrics = await metricsService.getMetrics();
    res.send(metrics);
  } catch (err) {
    logger.error('Failed to generate /metrics payload', {
      error: err instanceof Error ? err.message : String(err),
    });
    res.status(500).send('metrics_unavailable');
  }
};

if (metricsEnabled && metricsExposeOnMain) {
  app.get('/metrics', metricsHandler);
}

function startMetricsServer(host: string): void {
  if (!metricsUseSeparateServer) {
    return;
  }

  const metricsApp = express();
  metricsApp.get('/metrics', metricsHandler);

  metricsServer = createServer(metricsApp);
  metricsServer.on('error', (error) => {
    logger.error('Failed to start metrics server', {
      error: error instanceof Error ? error.message : String(error),
    });
  });

  metricsServer.listen(config.metrics.port, host, () => {
    logger.info(`Metrics server listening on ${host}:${config.metrics.port}`);
  });
}

// In production, the server runs from dist/server/server/index.js
// Client files are in dist/client, so we need to go up two levels
const clientBuildPath = path.resolve(__dirname, '../../client');

if (config.isProduction) {
  app.use(express.static(clientBuildPath));
}

// WebSocket setup
const wsServer = new WebSocketServer(server);

// API routes (pass wsServer for lobby broadcasting)
app.use('/api', setupRoutes(wsServer));

if (config.isProduction) {
  // Express 5 uses path-to-regexp v8, which no longer accepts a bare "*" path.
  // Use a named wildcard parameter to catch all routes for SPA fallback.
  app.get('/{*path}', (req: Request, res: Response, next: NextFunction) => {
    if (
      req.path.startsWith('/api') ||
      req.path.startsWith('/socket.io') ||
      req.path.startsWith('/metrics') ||
      req.path.startsWith('/health') ||
      req.path.startsWith('/ready')
    ) {
      return next();
    }

    res.sendFile(path.join(clientBuildPath, 'index.html'));
  });
}

// Error handling
app.use(errorHandler);

// 404 handler (must be last). In Express 5, omit the path string to avoid
// path-to-regexp errors and match all remaining unmatched routes.
app.use((_req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    error: {
      message: 'Route not found',
      code: 'NOT_FOUND',
      timestamp: new Date(),
    },
  });
});

async function startServer() {
  try {
    enforceAppTopology(config);

    // Connect to database
    await connectDatabase();
    logger.info('Database connected successfully');

    // Connect to Redis (optional in local development). Failure should not
    // prevent the HTTP server from starting so that developers can run
    // without a local Redis instance.
    try {
      await connectRedis();
      logger.info('Redis connected successfully');
    } catch (redisError) {
      logger.warn('Redis connection failed; continuing without Redis', {
        error: redisError instanceof Error ? redisError.message : String(redisError),
      });
    }

    // Initialize data retention service for GDPR compliance
    // Runs daily at 3 AM UTC to clean up soft-deleted users, expired tokens, etc.
    const prisma = getDatabaseClient();
    if (prisma) {
      retentionService = createDataRetentionService(prisma);
      scheduleDataRetentionTask();
    } else {
      logger.warn('Data retention scheduler not started: database not connected');
    }

    // Start server
    const PORT = config.server.port;
    const HOST = config.server.host;

    server.listen(PORT, HOST, () => {
      logger.info(`Server running on port ${PORT}`);
      logger.info(`Server host: ${HOST}`);
      logger.info(`WebSocket server attached to HTTP server on port ${PORT}`);
      logger.info(`Environment: ${config.nodeEnv}`);

      // Log orchestrator adapter mode for observability
      // FSM is now the canonical validator (RR-CANON compliance)
      const orchestratorEnabled = config.featureFlags.orchestrator.adapterEnabled;
      logger.info(`Orchestrator adapter mode: ${orchestratorEnabled ? 'ENABLED' : 'DISABLED'}`, {
        orchestratorAdapterEnabled: orchestratorEnabled,
        circuitBreakerEnabled: config.featureFlags.orchestrator.circuitBreaker.enabled,
        engineMode: orchestratorEnabled
          ? 'TurnEngineAdapter (FSM canonical)'
          : 'TurnEngine (legacy)',
      });

      // In production, having the orchestrator adapter disabled should be
      // an explicit, documented rollback per ORCHESTRATOR_ROLLOUT_PLAN.
      // Emit a loud warning so accidental misconfiguration is visible.
      if (config.nodeEnv === 'production' && !orchestratorEnabled) {
        logger.warn(
          'Orchestrator adapter is DISABLED in production; this should only occur during a documented rollback (see docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md).',
          {
            orchestratorAdapterEnabled: orchestratorEnabled,
            rulesMode: config.rules.mode,
          }
        );
      }

      // Initialize orchestrator rollout percentage gauge for metrics (100% = permanently enabled)
      getMetricsService().setOrchestratorRolloutPercentage(100);
    });

    startMetricsServer(HOST);

    // Graceful shutdown
    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

async function gracefulShutdown(signal: string) {
  logger.info(`Received ${signal}. Starting graceful shutdown...`);

  // Set a hard timeout for the entire shutdown process
  const forceExitTimeout = setTimeout(() => {
    logger.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 30000);

  try {
    // 1. Stop accepting new connections
    await new Promise<void>((resolve, reject) => {
      server.close((err) => {
        if (err) {
          logger.warn('Error closing HTTP server:', err);
          reject(err);
        } else {
          logger.info('HTTP server closed');
          resolve();
        }
      });
    });

    if (metricsServer) {
      await new Promise<void>((resolve, reject) => {
        metricsServer?.close((err) => {
          if (err) {
            logger.warn('Error closing metrics server:', err);
            reject(err);
          } else {
            logger.info('Metrics server closed');
            resolve();
          }
        });
      });
    }

    // 2. Cancel scheduled data retention task
    if (retentionTimerId) {
      clearTimeout(retentionTimerId);
      retentionTimerId = null;
      logger.info('Data retention scheduler cancelled');
    }

    // 3. Close Redis connection
    try {
      await disconnectRedis();
      logger.info('Redis connection closed');
    } catch (err) {
      logger.warn('Error closing Redis connection:', err);
    }

    // 4. Close database connection
    try {
      await disconnectDatabase();
      logger.info('Database connection closed');
    } catch (err) {
      logger.warn('Error closing database connection:', err);
    }

    clearTimeout(forceExitTimeout);
    logger.info('Graceful shutdown complete');
    process.exit(0);
  } catch (err) {
    logger.error('Error during graceful shutdown:', err);
    clearTimeout(forceExitTimeout);
    process.exit(1);
  }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Start the server
startServer();

export { app, server };
