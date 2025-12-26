import winston from 'winston';
import fs from 'fs';
import path from 'path';
import { AsyncLocalStorage } from 'async_hooks';
import { config } from '../config';

// ============================================================================
// Types
// ============================================================================

export type LogMeta = Record<string, unknown>;

/**
 * Request context stored in AsyncLocalStorage for automatic propagation
 * throughout the request lifecycle.
 */
export interface RequestContext {
  requestId: string;
  userId?: string;
  method?: string;
  path?: string;
  startTime?: number;
}

// ============================================================================
// Request Context (AsyncLocalStorage)
// ============================================================================

/**
 * AsyncLocalStorage for propagating request context through async call chains.
 * This allows any code to access the current request's correlation ID without
 * explicitly passing it through function parameters.
 */
export const requestContextStorage = new AsyncLocalStorage<RequestContext>();

/**
 * Get the current request context from AsyncLocalStorage.
 * Returns undefined if called outside of a request context.
 */
export const getRequestContext = (): RequestContext | undefined => {
  return requestContextStorage.getStore();
};

/**
 * Run a function within a request context. All logs and async operations
 * within the callback will have access to the context.
 */
export const runWithContext = <T>(context: RequestContext, fn: () => T): T => {
  return requestContextStorage.run(context, fn);
};

// ============================================================================
// Sensitive Data Masking
// ============================================================================

/**
 * Patterns for detecting sensitive keys in objects.
 * These are matched case-insensitively.
 */
const SENSITIVE_KEY_PATTERNS = [
  /password/i,
  /secret/i,
  /token/i,
  /api[_-]?key/i,
  /auth/i,
  /bearer/i,
  /credential/i,
  /private[_-]?key/i,
  /access[_-]?key/i,
  /session/i,
  /cookie/i,
];

/**
 * Sensitive HTTP headers that should be redacted.
 */
const SENSITIVE_HEADERS = new Set([
  'authorization',
  'cookie',
  'set-cookie',
  'x-api-key',
  'x-auth-token',
  'x-access-token',
  'proxy-authorization',
]);

/**
 * Check if a key name indicates sensitive data.
 */
const isSensitiveKey = (key: string): boolean => {
  const lowerKey = key.toLowerCase();
  return SENSITIVE_KEY_PATTERNS.some((pattern) => pattern.test(lowerKey));
};

/**
 * Redact email addresses to show only first 3 characters of local part.
 * Example: "john.doe@example.com" -> "joh***@example.com"
 */
export const redactEmail = (email: string | null | undefined): string | undefined => {
  if (!email) return undefined;
  const trimmed = String(email).trim();
  const atIndex = trimmed.indexOf('@');
  if (atIndex <= 0 || atIndex === trimmed.length - 1) {
    return '[REDACTED_EMAIL]';
  }
  const local = trimmed.slice(0, atIndex);
  const domain = trimmed.slice(atIndex + 1);
  const visibleChars = Math.min(3, local.length);
  const visible = local.slice(0, visibleChars);
  return `${visible}***@${domain}`;
};

/**
 * Redact a sensitive string value.
 * Shows first 4 characters for debugging while hiding the rest.
 */
const redactSensitiveString = (value: string): string => {
  if (value.length <= 8) {
    return '[REDACTED]';
  }
  return `${value.slice(0, 4)}...[REDACTED]`;
};

/**
 * Recursively mask sensitive values in an object.
 * Returns a new object with sensitive values redacted.
 *
 * @param obj - The object to mask
 * @param maxDepth - Maximum recursion depth to prevent infinite loops (default: 5)
 */
export const maskSensitiveData = (obj: unknown, maxDepth: number = 5): unknown => {
  if (maxDepth <= 0) {
    return '[MAX_DEPTH_EXCEEDED]';
  }

  if (obj === null || obj === undefined) {
    return obj;
  }

  if (typeof obj === 'string') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => maskSensitiveData(item, maxDepth - 1));
  }

  if (typeof obj !== 'object') {
    return obj;
  }

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (isSensitiveKey(key)) {
      // Preserve null/undefined for sensitive keys
      if (value === null || value === undefined) {
        result[key] = value;
      } else if (typeof value === 'string') {
        result[key] = redactSensitiveString(value);
      } else if (typeof value === 'object') {
        // For objects/arrays with sensitive keys, recurse to mask nested content
        result[key] = maskSensitiveData(value, maxDepth - 1);
      } else {
        result[key] = '[REDACTED]';
      }
    } else if (key.toLowerCase() === 'email' && typeof value === 'string') {
      result[key] = redactEmail(value);
    } else {
      result[key] = maskSensitiveData(value, maxDepth - 1);
    }
  }
  return result;
};

/**
 * Mask sensitive HTTP headers.
 * Returns a new headers object with sensitive headers redacted.
 */
export const maskHeaders = (
  headers: Record<string, string | string[] | undefined>
): Record<string, string | string[] | undefined> => {
  const result: Record<string, string | string[] | undefined> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (SENSITIVE_HEADERS.has(key.toLowerCase())) {
      result[key] = '[REDACTED]';
    } else {
      result[key] = value;
    }
  }
  return result;
};

// ============================================================================
// Winston Logger Configuration
// ============================================================================

const logsDir = path.join(process.cwd(), 'logs');
const configuredLogFile = config.logging.file?.trim();
const combinedLogPath = configuredLogFile
  ? path.resolve(configuredLogFile)
  : path.join(logsDir, 'combined.log');

if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}
if (configuredLogFile) {
  const logDir = path.dirname(combinedLogPath);
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
}

/**
 * Custom format to add request context from AsyncLocalStorage to log entries.
 */
const addRequestContext = winston.format((info) => {
  const context = getRequestContext();
  if (context) {
    info.requestId = context.requestId;
    if (context.userId) {
      info.userId = context.userId;
    }
    if (context.method) {
      info.method = context.method;
    }
    if (context.path) {
      info.path = context.path;
    }
  }
  return info;
});

/**
 * Custom format to structure log metadata consistently.
 */
const structuredFormat = winston.format((info) => {
  // Ensure standard fields are always present
  if (!info.service) {
    info.service = 'ringrift-api';
  }
  if (!info.environment) {
    info.environment = config.nodeEnv;
  }

  // Handle Error objects specially
  if (info.error instanceof Error) {
    info.error = {
      message: info.error.message,
      name: info.error.name,
      stack: info.error.stack,
    };
  }

  // Mask any sensitive data in the log entry
  const { level, message, timestamp, ...rest } = info;
  const masked = maskSensitiveData(rest);
  return {
    level,
    message,
    timestamp,
    ...(masked as Record<string, unknown>),
  };
});

/**
 * Format for structured JSON logging (used in production and file transports).
 */
const jsonFormat = winston.format.combine(
  winston.format.timestamp({
    format: () => new Date().toISOString(),
  }),
  winston.format.errors({ stack: true }),
  addRequestContext(),
  structuredFormat(),
  winston.format.json()
);

/**
 * Format for human-readable console output (used in development).
 */
const consoleFormat = winston.format.combine(
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss',
  }),
  winston.format.errors({ stack: true }),
  addRequestContext(),
  winston.format.colorize(),
  winston.format.printf(({ timestamp, level, message, requestId, ...meta }) => {
    const reqIdStr = requestId ? ` [${requestId}]` : '';
    const metaStr = Object.keys(meta).length > 0 ? ` ${JSON.stringify(meta)}` : '';
    return `${timestamp} ${level}${reqIdStr}: ${message}${metaStr}`;
  })
);

/**
 * Create the Winston logger instance.
 */
const logger = winston.createLogger({
  level: config.logging.level,
  defaultMeta: {
    service: 'ringrift-api',
    environment: config.nodeEnv,
  },
  transports: [
    // File transport for errors - always use JSON format
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      format: jsonFormat,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    // File transport for all logs - always use JSON format
    new winston.transports.File({
      filename: combinedLogPath,
      format: jsonFormat,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
  ],
});

const consoleFormatForEnv = config.logging.format === 'json' ? jsonFormat : consoleFormat;

logger.add(
  new winston.transports.Console({
    format: consoleFormatForEnv,
  })
);

// ============================================================================
// Legacy Compatibility Helpers
// ============================================================================

/**
 * Merge a per-request correlation id into log metadata when available.
 * @deprecated Use AsyncLocalStorage context instead. This function is kept
 * for backward compatibility with existing code.
 */
export const withRequestContext = (req: unknown, meta: LogMeta = {}): LogMeta => {
  const requestId = (req as { requestId?: string })?.requestId;
  if (requestId) {
    return { requestId, ...meta };
  }
  return meta;
};

/**
 * Convenience logger for HTTP handlers where an Express Request (or
 * compatible object) is available. Ensures requestId is consistently
 * attached to log metadata.
 *
 * @deprecated Prefer using the standard `logger` with AsyncLocalStorage context.
 * This object is kept for backward compatibility.
 */
export const httpLogger = {
  info: (req: unknown, message: string, meta?: LogMeta) =>
    logger.info(message, withRequestContext(req, meta)),
  warn: (req: unknown, message: string, meta?: LogMeta) =>
    logger.warn(message, withRequestContext(req, meta)),
  error: (req: unknown, message: string, meta?: LogMeta) =>
    logger.error(message, withRequestContext(req, meta)),
  debug: (req: unknown, message: string, meta?: LogMeta) =>
    logger.debug(message, withRequestContext(req, meta)),
};

// ============================================================================
// Stream for Morgan (legacy compatibility)
// ============================================================================

/**
 * Create a stream object for Morgan HTTP logger.
 * @deprecated Use the new requestLogger middleware instead of Morgan.
 */
export const stream = {
  write: (message: string) => {
    logger.info(message.trim());
  },
};

// ============================================================================
// Exports
// ============================================================================

export { logger };
