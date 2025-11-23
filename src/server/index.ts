import express from 'express';
import { createServer } from 'http';
import path from 'path';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import { WebSocketServer } from './websocket/server';
import { setupRoutes } from './routes';
import { errorHandler } from './middleware/errorHandler';
import { rateLimiter } from './middleware/rateLimiter';
import { requestContext } from './middleware/requestContext';
import { logger } from './utils/logger';
import { connectDatabase } from './database/connection';
import { connectRedis } from './cache/redis';
import client from 'prom-client';
import { config } from './config';
const app = express();
const server = createServer(app);

// Register default Prometheus metrics for the Node.js process.
client.collectDefaultMetrics();
// Middleware
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", 'data:', 'https:'],
        connectSrc: ["'self'", 'ws:', 'wss:'],
      },
    },
  })
);

app.use(
  cors({
    origin: config.server.corsOrigin,
    credentials: true,
  })
);

app.use(compression());
app.use(morgan('combined', { stream: { write: (message) => logger.info(message.trim()) } }));
// Attach lightweight per-request context for correlation IDs. This middleware
// runs before any API routes and before the global error handler so that
// requestId is available throughout the HTTP pipeline.
app.use(requestContext);
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use(rateLimiter);

// Health check endpoint
app.get('/health', (_req, res) => {
  res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: config.app.version,
  });
});

// Prometheus metrics endpoint
app.get('/metrics', async (_req, res) => {
  try {
    res.set('Content-Type', client.register.contentType);
    const metrics = await client.register.metrics();
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

function enforceAppTopology() {
  const topology = config.app.topology;
  const nodeEnv = config.nodeEnv;

  if (topology === 'single') {
    logger.info(
      'App topology: single-instance mode (RINGRIFT_APP_TOPOLOGY=single). ' +
        'The server assumes it is the only app instance talking to this database and Redis ' +
        'for authoritative game sessions.'
    );
    return;
  }

  const logContext = { topology, nodeEnv };

  if (topology === 'multi-unsafe') {
    const message =
      'RINGRIFT_APP_TOPOLOGY=multi-unsafe: multi-instance deployment without sticky sessions ' +
      'or shared state is unsupported. Each instance will maintain its own in-process game sessions.';

    if (config.isProduction) {
      logger.error(
        `${message} Refusing to start in NODE_ENV=production. ` +
          'Configure infrastructure-enforced sticky sessions and/or shared game state before using multiple app instances.',
        logContext
      );
      throw new Error(
        'Unsupported app topology "multi-unsafe" in production. Refusing to start with multiple app instances.'
      );
    }

    logger.warn(
      `${message} Continuing because NODE_ENV=${nodeEnv}. ` +
        'This mode is intended only for development/experimentation.',
      logContext
    );
    return;
  }

  if (topology === 'multi-sticky') {
    logger.warn(
      'RINGRIFT_APP_TOPOLOGY=multi-sticky: backend assumes infrastructure-enforced sticky sessions ' +
        '(HTTP + WebSocket) for all game-affecting traffic to a given game. ' +
        'Correctness is not guaranteed if sticky sessions are misconfigured or absent.',
      logContext
    );
  }
}

async function startServer() {
  try {
    enforceAppTopology();

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
