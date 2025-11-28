import express from 'express';
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
import { connectDatabase } from './database/connection';
import { connectRedis } from './cache/redis';
import client from 'prom-client';
import { config, enforceAppTopology } from './config';
import { HealthCheckService, isServiceReady } from './services/HealthCheckService';
import { getMetricsService } from './services/MetricsService';
import { getServiceStatusManager } from './services/ServiceStatusManager';

const app = express();
const server = createServer(app);

// Register default Prometheus metrics for the Node.js process.
client.collectDefaultMetrics();

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
app.use(metricsMiddleware);

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check endpoints - placed BEFORE rate limiter to bypass rate limiting.
// These endpoints are called frequently by orchestrators (Kubernetes, load balancers)
// and should not be rate limited or require authentication.

/**
 * Liveness probe endpoint - /health or /healthz
 * Returns 200 if the process is alive and responding.
 * Fast response with minimal checks.
 * Used by orchestrators to restart dead containers.
 */
app.get(['/health', '/healthz'], (_req, res) => {
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
app.get(['/ready', '/readyz'], async (_req, res) => {
  const status = await HealthCheckService.getReadinessStatus();
  const httpStatus = isServiceReady(status) ? 200 : 503;
  res.status(httpStatus).json(status);
});

// Rate limiting - placed AFTER health endpoints
app.use(rateLimiter);

// Prometheus metrics endpoint
// Uses MetricsService which consolidates all custom metrics with the default
// Node.js metrics (memory, CPU, event loop, GC) from prom-client.
app.get('/metrics', async (_req, res) => {
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
});

const clientBuildPath = path.resolve(__dirname, '../client');

if (config.isProduction) {
  app.use(express.static(clientBuildPath));
}

// WebSocket setup
const wsServer = new WebSocketServer(server);

// API routes (pass wsServer for lobby broadcasting)
app.use('/api', setupRoutes(wsServer));

if (config.isProduction) {
  app.get('*', (req, res, next) => {
    if (
      req.path.startsWith('/api') ||
      req.path.startsWith('/socket.io') ||
      req.path.startsWith('/metrics') ||
      req.path.startsWith('/health')
    ) {
      return next();
    }

    res.sendFile(path.join(clientBuildPath, 'index.html'));
  });
}

// Error handling
app.use(errorHandler);

// 404 handler
app.use('*', (_req, res) => {
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

    // Start server
    const PORT = config.server.port;

    server.listen(PORT, () => {
      logger.info(`Server running on port ${PORT}`);
      logger.info(`WebSocket server attached to HTTP server on port ${PORT}`);
      logger.info(`Environment: ${config.nodeEnv}`);

      // Log orchestrator adapter mode for observability
      const orchestratorEnabled = config.featureFlags.orchestrator.adapterEnabled;
      logger.info(`Orchestrator adapter mode: ${orchestratorEnabled ? 'ENABLED' : 'DISABLED'}`, {
        orchestratorAdapterEnabled: orchestratorEnabled,
        rolloutPercentage: config.featureFlags.orchestrator.rolloutPercentage,
        shadowModeEnabled: config.featureFlags.orchestrator.shadowModeEnabled,
        circuitBreakerEnabled: config.featureFlags.orchestrator.circuitBreaker.enabled,
        engineMode: orchestratorEnabled
          ? 'TurnEngineAdapter (shared orchestrator)'
          : 'TurnEngine (legacy)',
      });

      // In production, having the orchestrator adapter disabled should be
      // an explicit, documented rollback per ORCHESTRATOR_ROLLOUT_PLAN.
      // Emit a loud warning so accidental misconfiguration is visible.
      if (config.nodeEnv === 'production' && !orchestratorEnabled) {
        logger.warn(
          'Orchestrator adapter is DISABLED in production; this should only occur during a documented rollback (see docs/ORCHESTRATOR_ROLLOUT_PLAN.md).',
          {
            orchestratorAdapterEnabled: orchestratorEnabled,
            rolloutPercentage: config.featureFlags.orchestrator.rolloutPercentage,
            shadowModeEnabled: config.featureFlags.orchestrator.shadowModeEnabled,
            rulesMode: config.rulesEngine.rulesMode,
          }
        );
      }

      // Initialize orchestrator rollout percentage gauge for metrics.
      getMetricsService().setOrchestratorRolloutPercentage(
        config.featureFlags.orchestrator.rolloutPercentage
      );
    });

    // Graceful shutdown
    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

function gracefulShutdown(signal: string) {
  logger.info(`Received ${signal}. Starting graceful shutdown...`);

  server.close(() => {
    logger.info('HTTP server closed');

    // Close database connections, Redis, etc.
    process.exit(0);
  });

  // Force close after 30 seconds
  setTimeout(() => {
    logger.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 30000);
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
