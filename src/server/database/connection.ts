import { PrismaClient } from '@prisma/client';
import { logger } from '../utils/logger';

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
    if (process.env.NODE_ENV === 'development') {
      prisma.$on('query', (e: any) => {
        logger.debug('Database Query:', {
          query: e.query,
          params: e.params,
          duration: `${e.duration}ms`,
        });
      });
    }

    // Log database errors
    prisma.$on('error', (e: any) => {
      logger.error('Database Error:', e);
    });

    // Log database info
    prisma.$on('info', (e: any) => {
      logger.info('Database Info:', e.message);
    });

    // Log database warnings
    prisma.$on('warn', (e: any) => {
      logger.warn('Database Warning:', e.message);
    });

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

// Transaction wrapper
export const withTransaction = async <T>(
  callback: (tx: PrismaClient) => Promise<T>
): Promise<T> => {
  if (!prisma) {
    throw new Error('Database not connected');
  }

  return await prisma.$transaction(async (tx: any) => {
    return await callback(tx as PrismaClient);
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