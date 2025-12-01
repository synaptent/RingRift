import { Prisma, PrismaClient } from '@prisma/client';
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

    prisma = new PrismaClient({
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
        $on: (event: 'query' | 'error' | 'info' | 'warn', callback: (e: PrismaQueryEvent | PrismaLogEvent) => void) => void;
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
  '$connect' | '$disconnect' | '$on' | '$transaction' | '$use' | '$extends'
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

// Graceful shutdown
process.on('beforeExit', async () => {
  await disconnectDatabase();
});

process.on('SIGINT', async () => {
  await disconnectDatabase();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await disconnectDatabase();
  process.exit(0);
});
