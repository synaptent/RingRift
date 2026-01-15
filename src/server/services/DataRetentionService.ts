import { PrismaClient } from '@prisma/client';
import { logger } from '../utils/logger';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for data retention periods.
 * All periods are configurable to comply with GDPR and organizational policies.
 */
export interface RetentionConfig {
  /** Soft-deleted users: days until hard delete (GDPR: typically 30 days) */
  deletedUserRetentionDays: number;
  /** Inactive users: days until marking inactive (typically 1 year) */
  inactiveUserThresholdDays: number;
  /** Unverified accounts: days until soft deletion (typically 7 days) */
  unverifiedAccountRetentionDays: number;
  /** Game data: months to retain (typically 24 months / 2 years) */
  gameDataRetentionMonths: number;
  /** Session data: hours to retain (typically 24 hours) */
  sessionDataRetentionHours: number;
  /** Refresh tokens: days to retain expired/revoked tokens (typically 7 days) */
  expiredTokenRetentionDays: number;
}

/**
 * Report of retention task execution results.
 */
export interface RetentionReport {
  /** Number of users permanently deleted (hard delete) */
  hardDeletedUsers: number;
  /** Number of expired/revoked refresh tokens deleted */
  deletedTokens: number;
  /** Number of unverified accounts soft-deleted */
  deletedUnverified: number;
  /** Timestamp when the retention tasks were run */
  executedAt: Date;
  /** Duration of execution in milliseconds */
  durationMs: number;
}

// ============================================================================
// Default Configuration
// ============================================================================

/**
 * Default retention periods aligned with GDPR requirements and best practices.
 * These can be overridden via constructor or environment variables.
 *
 * Reference: docs/security/DATA_LIFECYCLE_AND_PRIVACY.md Section 3
 */
export const DEFAULT_RETENTION: RetentionConfig = {
  deletedUserRetentionDays: 30, // GDPR: typically 30 days for recovery/compliance
  inactiveUserThresholdDays: 365, // 1 year of inactivity
  unverifiedAccountRetentionDays: 7, // 1 week to verify email
  gameDataRetentionMonths: 24, // 2 years for game history
  sessionDataRetentionHours: 24, // 1 day for session data
  expiredTokenRetentionDays: 7, // 1 week for expired token cleanup
};

// ============================================================================
// Service
// ============================================================================

/**
 * Service for managing data retention policies and cleanup tasks.
 *
 * Implements S-05.E.4 from the Data Lifecycle and Privacy requirements.
 * Responsible for:
 * - Hard deleting users past the soft-delete retention period
 * - Cleaning up expired and revoked refresh tokens
 * - Soft-deleting unverified accounts past threshold
 *
 * @example
 * ```typescript
 * const retentionService = new DataRetentionService(prisma);
 * const report = await retentionService.runRetentionTasks();
 * console.log(`Cleaned up ${report.hardDeletedUsers} users`);
 * ```
 *
 * @example Integration with server startup (implemented in src/server/index.ts)
 * ```typescript
 * // Scheduling is implemented via setTimeout in scheduleDataRetentionTask()
 * // Runs daily at 3 AM UTC for GDPR compliance
 * ```
 */
export class DataRetentionService {
  private prisma: PrismaClient;
  private config: RetentionConfig;

  constructor(prisma: PrismaClient, config?: Partial<RetentionConfig>) {
    this.prisma = prisma;
    this.config = { ...DEFAULT_RETENTION, ...config };

    logger.debug('DataRetentionService initialized', {
      config: this.config,
    });
  }

  // ==========================================================================
  // Public API
  // ==========================================================================

  /**
   * Run all retention cleanup tasks.
   *
   * This method should be called periodically (e.g., daily via cron job)
   * to enforce data retention policies.
   *
   * @returns Report of all cleanup operations performed
   */
  async runRetentionTasks(): Promise<RetentionReport> {
    const startTime = Date.now();
    logger.info('Starting data retention cleanup tasks');

    try {
      const [hardDeletedUsers, deletedTokens, deletedUnverified] = await Promise.all([
        this.hardDeleteExpiredUsers(),
        this.cleanupExpiredTokens(),
        this.cleanupUnverifiedAccounts(),
      ]);

      const durationMs = Date.now() - startTime;

      const report: RetentionReport = {
        hardDeletedUsers,
        deletedTokens,
        deletedUnverified,
        executedAt: new Date(),
        durationMs,
      };

      logger.info('Data retention cleanup tasks completed', {
        report,
      });

      return report;
    } catch (error) {
      const durationMs = Date.now() - startTime;
      logger.error('Data retention cleanup tasks failed', {
        error: error instanceof Error ? error.message : String(error),
        durationMs,
      });
      throw error;
    }
  }

  /**
   * Hard delete users who were soft-deleted longer than the retention period.
   *
   * Per GDPR requirements, soft-deleted accounts are kept for a grace period
   * (default: 30 days) to allow for recovery or compliance requests.
   * After this period, the account and all associated data are permanently deleted.
   *
   * Note: Related entities (games, moves) are preserved with anonymized references
   * per the data lifecycle policy. Cascade delete handles RefreshTokens.
   *
   * @returns Number of users permanently deleted
   */
  async hardDeleteExpiredUsers(): Promise<number> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.deletedUserRetentionDays);

    logger.debug('Hard deleting expired users', {
      cutoffDate: cutoffDate.toISOString(),
      retentionDays: this.config.deletedUserRetentionDays,
    });

    const result = await this.prisma.user.deleteMany({
      where: {
        deletedAt: {
          not: null,
          lt: cutoffDate,
        },
      },
    });

    if (result.count > 0) {
      logger.info(`Hard deleted ${result.count} expired user accounts`, {
        count: result.count,
        cutoffDate: cutoffDate.toISOString(),
      });
    }

    return result.count;
  }

  /**
   * Clean up expired and revoked refresh tokens.
   *
   * Removes tokens that are:
   * - Expired beyond the retention period
   * - Revoked beyond the retention period
   *
   * This keeps the refresh_tokens table size manageable and limits
   * the blast radius of historical session identifiers.
   *
   * @returns Number of tokens deleted
   */
  async cleanupExpiredTokens(): Promise<number> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.expiredTokenRetentionDays);

    logger.debug('Cleaning up expired refresh tokens', {
      cutoffDate: cutoffDate.toISOString(),
      retentionDays: this.config.expiredTokenRetentionDays,
    });

    // Delete tokens that are either:
    // 1. Expired beyond the retention period
    // 2. Revoked beyond the retention period
    const result = await this.prisma.refreshToken.deleteMany({
      where: {
        OR: [
          // Expired tokens past retention
          { expiresAt: { lt: cutoffDate } },
          // Revoked tokens past retention (revokedAt is non-null and old)
          {
            revokedAt: {
              not: null,
              lt: cutoffDate,
            },
          },
        ],
      },
    });

    if (result.count > 0) {
      logger.info(`Deleted ${result.count} expired/revoked refresh tokens`, {
        count: result.count,
        cutoffDate: cutoffDate.toISOString(),
      });
    }

    return result.count;
  }

  /**
   * Soft-delete unverified accounts older than the threshold.
   *
   * Accounts that haven't verified their email within the retention period
   * (default: 7 days) are marked as deleted. They will be hard deleted
   * after the soft-delete retention period passes.
   *
   * This prevents accumulation of abandoned registrations while still
   * allowing legitimate users time to verify.
   *
   * @returns Number of accounts soft-deleted
   */
  async cleanupUnverifiedAccounts(): Promise<number> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.unverifiedAccountRetentionDays);

    logger.debug('Cleaning up unverified accounts', {
      cutoffDate: cutoffDate.toISOString(),
      retentionDays: this.config.unverifiedAccountRetentionDays,
    });

    // Soft-delete unverified accounts that are:
    // - Not yet verified
    // - Created before the cutoff date
    // - Not already deleted
    const result = await this.prisma.user.updateMany({
      where: {
        emailVerified: false,
        createdAt: { lt: cutoffDate },
        deletedAt: null,
      },
      data: {
        deletedAt: new Date(),
        isActive: false,
      },
    });

    if (result.count > 0) {
      logger.info(`Soft-deleted ${result.count} unverified accounts`, {
        count: result.count,
        cutoffDate: cutoffDate.toISOString(),
      });
    }

    return result.count;
  }

  // ==========================================================================
  // Configuration Access
  // ==========================================================================

  /**
   * Get the current retention configuration.
   * Returns a copy to prevent external modification.
   *
   * @returns Current retention configuration
   */
  getConfig(): RetentionConfig {
    return { ...this.config };
  }

  /**
   * Update the retention configuration.
   * Only provided fields will be updated; others retain their current values.
   *
   * @param updates - Partial configuration updates
   */
  updateConfig(updates: Partial<RetentionConfig>): void {
    this.config = { ...this.config, ...updates };
    logger.info('Retention configuration updated', {
      config: this.config,
    });
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a DataRetentionService with configuration from environment variables.
 *
 * Environment variables:
 * - DATA_RETENTION_DELETED_USERS_DAYS: Days to retain soft-deleted users
 * - DATA_RETENTION_INACTIVE_USERS_DAYS: Days until marking users inactive
 * - DATA_RETENTION_UNVERIFIED_DAYS: Days to retain unverified accounts
 * - DATA_RETENTION_GAME_DATA_MONTHS: Months to retain game data
 * - DATA_RETENTION_SESSION_HOURS: Hours to retain session data
 * - DATA_RETENTION_EXPIRED_TOKEN_DAYS: Days to retain expired tokens
 *
 * @param prisma - PrismaClient instance
 * @returns Configured DataRetentionService
 */
export function createDataRetentionService(prisma: PrismaClient): DataRetentionService {
  const config: Partial<RetentionConfig> = {};

  // Parse environment variables with defaults from DEFAULT_RETENTION
  const envDeletedUsers = process.env.DATA_RETENTION_DELETED_USERS_DAYS;
  if (envDeletedUsers) {
    const parsed = parseInt(envDeletedUsers, 10);
    if (!isNaN(parsed) && parsed > 0) {
      config.deletedUserRetentionDays = parsed;
    }
  }

  const envInactiveUsers = process.env.DATA_RETENTION_INACTIVE_USERS_DAYS;
  if (envInactiveUsers) {
    const parsed = parseInt(envInactiveUsers, 10);
    if (!isNaN(parsed) && parsed > 0) {
      config.inactiveUserThresholdDays = parsed;
    }
  }

  const envUnverified = process.env.DATA_RETENTION_UNVERIFIED_DAYS;
  if (envUnverified) {
    const parsed = parseInt(envUnverified, 10);
    if (!isNaN(parsed) && parsed > 0) {
      config.unverifiedAccountRetentionDays = parsed;
    }
  }

  const envGameData = process.env.DATA_RETENTION_GAME_DATA_MONTHS;
  if (envGameData) {
    const parsed = parseInt(envGameData, 10);
    if (!isNaN(parsed) && parsed > 0) {
      config.gameDataRetentionMonths = parsed;
    }
  }

  const envSession = process.env.DATA_RETENTION_SESSION_HOURS;
  if (envSession) {
    const parsed = parseInt(envSession, 10);
    if (!isNaN(parsed) && parsed > 0) {
      config.sessionDataRetentionHours = parsed;
    }
  }

  const envExpiredToken = process.env.DATA_RETENTION_EXPIRED_TOKEN_DAYS;
  if (envExpiredToken) {
    const parsed = parseInt(envExpiredToken, 10);
    if (!isNaN(parsed) && parsed > 0) {
      config.expiredTokenRetentionDays = parsed;
    }
  }

  return new DataRetentionService(prisma, config);
}

// ============================================================================
// Admin Endpoint Types (for future implementation)
// ============================================================================

/**
 * Response type for GET /api/admin/retention/status endpoint.
 *
 * @example
 * ```typescript
 * // GET /api/admin/retention/status
 * // Returns current config and last run stats
 * router.get('/admin/retention/status', adminAuth, async (req, res) => {
 *   const status: RetentionStatusResponse = {
 *     config: retentionService.getConfig(),
 *     lastRunAt: lastRetentionRun?.executedAt,
 *     lastReport: lastRetentionRun,
 *   };
 *   res.json(status);
 * });
 * ```
 */
export interface RetentionStatusResponse {
  /** Current retention configuration */
  config: RetentionConfig;
  /** Timestamp of last retention run (if available) */
  lastRunAt?: Date;
  /** Report from last retention run (if available) */
  lastReport?: RetentionReport;
}
