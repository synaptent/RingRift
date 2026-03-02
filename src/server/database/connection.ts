import { PrismaClient } from '@prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';
import pg from 'pg';
import { logger } from '../utils/logger';
import { config } from '../config';

/**
 * Prisma event types for logging
 */
interface PrismaQueryEvent {
  timestamp: Date;
  query: string;
  params: string;
  duration: number;
  target: string;
}

interface PrismaLogEvent {
  timestamp: Date;
  message: string;
  target: string;
}

let prisma: PrismaClient | null = null;

export const connectDatabase = async (): Promise<PrismaClient> => {
  try {
    if (prisma) {
      return prisma;
    }

    // Prisma 7: Use pg driver adapter for direct database connection.
    // The adapter replaces the old url-in-schema pattern.
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error('DATABASE_URL environment variable is required');
    }
    const pool = new pg.Pool({ connectionString });
    const adapter = new PrismaPg(pool);

    prisma = new PrismaClient({
      adapter,
      log: [
        {
          emit: 'event',
          level: 'query',
        },
        {
          emit: 'event',
          level: 'error',
        },
        {
          emit: 'event',
          level: 'info',
        },
        {
          emit: 'event',
          level: 'warn',
        },
      ],
      errorFormat: 'pretty',
    });

    // Log database queries in development
    // Note: $on events require Prisma log configuration which we set above
    if (config.isDevelopment) {
      // Type-safe event handling using Prisma's built-in event types
      const prismaWithEvents = prisma as PrismaClient & {
        $on: (
          event: 'query' | 'error' | 'info' | 'warn',
          callback: (e: PrismaQueryEvent | PrismaLogEvent) => void
        ) => void;
      };

      prismaWithEvents.$on('query', (e) => {
        const event = e as PrismaQueryEvent;
        logger.debug('Database Query:', {
          query: event.query,
          params: event.params,
          duration: `${event.duration}ms`,
        });
      });

      prismaWithEvents.$on('error', (e) => {
        logger.error('Database Error:', e);
      });

      prismaWithEvents.$on('info', (e) => {
        const event = e as PrismaLogEvent;
        logger.info('Database Info:', event.message);
      });

      prismaWithEvents.$on('warn', (e) => {
        const event = e as PrismaLogEvent;
        logger.warn('Database Warning:', event.message);
      });
    }

    // Test the connection
    await prisma.$connect();
    logger.info('Database connected successfully');

    return prisma;
  } catch (error) {
    logger.error('Failed to connect to database:', error);
    throw error;
  }
};

export const getDatabaseClient = (): PrismaClient | null => {
  return prisma;
};

export const disconnectDatabase = async (): Promise<void> => {
  if (prisma) {
    await prisma.$disconnect();
    prisma = null;
    logger.info('Database disconnected');
  }
};

// Health check function
export const checkDatabaseHealth = async (): Promise<boolean> => {
  try {
    if (!prisma) {
      return false;
    }

    await prisma.$queryRaw`SELECT 1`;
    return true;
  } catch (error) {
    logger.error('Database health check failed:', error);
    return false;
  }
};

/**
 * Prisma transaction client type (excludes interactive transaction methods)
 */
export type TransactionClient = Omit<
  PrismaClient,
  '$connect' | '$disconnect' | '$on' | '$transaction' | '$extends'
>;

// Transaction wrapper
export const withTransaction = async <T>(
  callback: (tx: TransactionClient) => Promise<T>
): Promise<T> => {
  if (!prisma) {
    throw new Error('Database not connected');
  }

  return await prisma.$transaction(async (tx) => {
    return await callback(tx);
  });
};

/**
 * Default timeout for database queries in milliseconds.
 * Can be overridden per-query.
 */
export const DEFAULT_DB_QUERY_TIMEOUT_MS = 5000;

/**
 * Wraps a database query promise with a timeout.
 * Returns null if the timeout is reached, otherwise returns the query result.
 *
 * @param queryPromise - The database query promise to wrap
 * @param timeoutMs - Timeout in milliseconds (defaults to DEFAULT_DB_QUERY_TIMEOUT_MS)
 * @returns The query result or null if timed out
 *
 * @example
 * const user = await withQueryTimeout(
 *   prisma.user.findUnique({ where: { id } }),
 *   3000
 * );
 * if (user === null) {
 *   // Handle timeout - could be actual null result or timeout
 *   // For nullable queries, consider using withQueryTimeoutStrict instead
 * }
 */
export async function withQueryTimeout<T>(
  queryPromise: Promise<T>,
  timeoutMs: number = DEFAULT_DB_QUERY_TIMEOUT_MS
): Promise<T | null> {
  const timeoutPromise = new Promise<null>((resolve) => setTimeout(() => resolve(null), timeoutMs));
  return Promise.race([queryPromise, timeoutPromise]);
}

/**
 * Result type for strict timeout wrapper that distinguishes timeout from null results.
 */
export type QueryTimeoutResult<T> =
  | { success: true; data: T }
  | { success: false; reason: 'timeout' };

/**
 * Wraps a database query promise with a timeout, distinguishing between
 * timeout and actual null/undefined results from the query.
 *
 * @param queryPromise - The database query promise to wrap
 * @param timeoutMs - Timeout in milliseconds (defaults to DEFAULT_DB_QUERY_TIMEOUT_MS)
 * @returns QueryTimeoutResult indicating success with data or timeout failure
 *
 * @example
 * const result = await withQueryTimeoutStrict(
 *   prisma.user.findUnique({ where: { id } }),
 *   3000
 * );
 * if (!result.success) {
 *   // Definitely a timeout, not a null query result
 *   throw new Error('Database query timed out');
 * }
 * const user = result.data; // Could be null if user doesn't exist
 */
export async function withQueryTimeoutStrict<T>(
  queryPromise: Promise<T>,
  timeoutMs: number = DEFAULT_DB_QUERY_TIMEOUT_MS
): Promise<QueryTimeoutResult<T>> {
  const TIMEOUT_SENTINEL = Symbol('timeout');
  const timeoutPromise = new Promise<typeof TIMEOUT_SENTINEL>((resolve) =>
    setTimeout(() => resolve(TIMEOUT_SENTINEL), timeoutMs)
  );

  const result = await Promise.race([queryPromise, timeoutPromise]);

  if (result === TIMEOUT_SENTINEL) {
    return { success: false, reason: 'timeout' };
  }

  return { success: true, data: result as T };
}

// Graceful shutdown on process exit (not on signals - those are handled by index.ts)
// beforeExit fires when the Node.js event loop has no additional work to schedule
process.on('beforeExit', async () => {
  await disconnectDatabase();
});
