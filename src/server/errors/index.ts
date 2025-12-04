/**
 * Centralized error handling module for standardized API error responses.
 *
 * @module errors
 *
 * @example
 * ```ts
 * import { ApiError, ErrorCodes, CommonErrors } from '../errors';
 *
 * // Throw with error code
 * throw new ApiError({ code: ErrorCodes.AUTH_INVALID_CREDENTIALS });
 *
 * // Use common error factory
 * throw CommonErrors.databaseUnavailable();
 *
 * // With validation details
 * throw CommonErrors.validationFailed([
 *   { field: 'email', message: 'Invalid email format' }
 * ]);
 * ```
 */

export { ApiError, CommonErrors, isAppError } from './ApiError';
export type { ApiErrorResponse, ApiErrorOptions, ValidationErrorDetail } from './ApiError';

export {
  ErrorCodes,
  ErrorCodeToStatus,
  ErrorCodeMessages,
  LegacyCodeMapping,
  normalizeErrorCode,
} from './errorCodes';
export type { ErrorCode } from './errorCodes';
