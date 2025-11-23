import winston from 'winston';
import path from 'path';
import { config } from '../config';

// Create logs directory if it doesn't exist
const logsDir = path.join(process.cwd(), 'logs');

const logger = winston.createLogger({
  level: config.logging.level,
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss',
    }),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'ringrift' },
  transports: [
    // Write all logs with importance level of `error` or less to `error.log`
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    // Write all logs with importance level of `info` or less to `combined.log`
    new winston.transports.File({
      filename: path.join(logsDir, 'combined.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
  ],
});

// If we're not in production, log to the console with a simple format
if (!config.isProduction) {
  logger.add(
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple(),
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          return `${timestamp} [${level}]: ${message} ${Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''}`;
        })
      ),
    })
  );
}

type LogMeta = Record<string, any>;

/**
 * Merge a per-request correlation id into log metadata when available.
 */
const withRequestContext = (req: any, meta: LogMeta = {}): LogMeta => {
  const requestId = (req as any)?.requestId;
  if (requestId) {
    return { requestId, ...meta };
  }
  return meta;
};

/**
 * Convenience logger for HTTP handlers where an Express Request (or
 * compatible object) is available. Ensures requestId is consistently
 * attached to log metadata.
 */
const httpLogger = {
  info: (req: any, message: string, meta?: LogMeta) =>
    logger.info(message, withRequestContext(req, meta)),
  warn: (req: any, message: string, meta?: LogMeta) =>
    logger.warn(message, withRequestContext(req, meta)),
  error: (req: any, message: string, meta?: LogMeta) =>
    logger.error(message, withRequestContext(req, meta)),
  debug: (req: any, message: string, meta?: LogMeta) =>
    logger.debug(message, withRequestContext(req, meta)),
};

// Create a stream object for Morgan HTTP logger
const stream = {
  write: (message: string) => {
    logger.info(message.trim());
  },
};

export { logger, stream, httpLogger, withRequestContext };
