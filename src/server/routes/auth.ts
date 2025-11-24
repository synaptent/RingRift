import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';
import { getDatabaseClient } from '../database/connection';
import {
  generateToken,
  generateRefreshToken,
  verifyRefreshToken,
  authenticate,
  validateUser,
} from '../middleware/auth';
import type { AuthenticatedRequest } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { authRateLimiter } from '../middleware/rateLimiter';
import { logger, httpLogger, redactEmail } from '../utils/logger';
import { sendVerificationEmail, sendPasswordResetEmail } from '../utils/email';
import {
  RegisterSchema,
  LoginSchema,
  RefreshTokenSchema,
  VerifyEmailSchema,
  ForgotPasswordSchema,
  ResetPasswordSchema,
} from '../../shared/validation/schemas';
import { getCacheService, CacheKeys } from '../cache/redis';
import { config } from '../config';

const router = Router();

type FailedLoginRecord = {
  count: number;
  firstFailureAt: number;
};

const inMemoryFailedLogins = new Map<string, FailedLoginRecord>();
const inMemoryLockouts = new Map<string, number>();

/**
 * Hash raw refresh tokens before persisting them, so that a database leak
 * does not expose long-lived bearer tokens directly. The JWT itself is still
 * returned to the client; only the hash is stored server-side.
 */
const hashRefreshToken = (token: string): string =>
  crypto.createHash('sha256').update(token).digest('hex');

const normalizeEmailForLockout = (email: string): string => email.trim().toLowerCase();

const isLoginLockoutEnabled = (): boolean => config.auth.loginLockoutEnabled;

/**
 * Check whether a given email is currently locked out from logging in.
 * Uses Redis when available, otherwise falls back to in-memory state.
 */
const isEmailLockedOut = async (email: string, ip?: string): Promise<boolean> => {
  if (!isLoginLockoutEnabled()) {
    return false;
  }

  const normalizedEmail = normalizeEmailForLockout(email);
  const now = Date.now();
  const cache = getCacheService();

  if (cache) {
    const lockoutKey = CacheKeys.authLoginLockout(normalizedEmail);
    const locked = await cache.exists(lockoutKey);
    if (locked) {
      logger.warn('Login attempt while email is locked out', {
        event: 'auth_lockout_attempt',
        email: redactEmail(normalizedEmail),
        ip,
      });
      return true;
    }
    return false;
  }

  const lockoutUntil = inMemoryLockouts.get(normalizedEmail);
  if (lockoutUntil && lockoutUntil > now) {
    logger.warn('Login attempt while email is locked out (in-memory)', {
      event: 'auth_lockout_attempt',
      email: redactEmail(normalizedEmail),
      ip,
    });
    return true;
  }

  if (lockoutUntil && lockoutUntil <= now) {
    inMemoryLockouts.delete(normalizedEmail);
  }

  return false;
};

/**
 * Record a failed login attempt for an email, potentially triggering
 * a temporary lockout when the configured threshold is exceeded
 * within the sliding window.
 */
const recordFailedLoginAttempt = async (email: string, ip?: string): Promise<void> => {
  if (!isLoginLockoutEnabled()) {
    return;
  }

  const normalizedEmail = normalizeEmailForLockout(email);
  const now = Date.now();
  const { maxFailedLoginAttempts, failedLoginWindowSeconds, lockoutDurationSeconds } = config.auth;
  const windowMs = failedLoginWindowSeconds * 1000;

  const cache = getCacheService();

  if (cache) {
    const failuresKey = CacheKeys.authLoginFailures(normalizedEmail);
    const lockoutKey = CacheKeys.authLoginLockout(normalizedEmail);

    const existing = (await cache.get<FailedLoginRecord>(failuresKey)) || {
      count: 0,
      firstFailureAt: now,
    };

    let record = existing;
    if (now - existing.firstFailureAt > windowMs) {
      record = { count: 0, firstFailureAt: now };
    }

    record = { ...record, count: record.count + 1 };

    if (record.count >= maxFailedLoginAttempts) {
      await cache.set(
        lockoutKey,
        {
          lockedAt: now,
          attempts: record.count,
        },
        lockoutDurationSeconds
      );
      await cache.del(failuresKey);

      logger.warn('Auth login lockout triggered', {
        event: 'auth_lockout',
        email: redactEmail(normalizedEmail),
        ip,
        attempts: record.count,
        lockoutDurationSeconds,
      });
    } else {
      await cache.set(failuresKey, record, failedLoginWindowSeconds);
    }

    return;
  }

  // In-memory fallback when Redis is not available.
  const existingRecord = inMemoryFailedLogins.get(normalizedEmail);
  let record: FailedLoginRecord;

  if (!existingRecord || now - existingRecord.firstFailureAt > windowMs) {
    record = { count: 1, firstFailureAt: now };
  } else {
    record = { ...existingRecord, count: existingRecord.count + 1 };
  }

  if (record.count >= maxFailedLoginAttempts) {
    inMemoryLockouts.set(normalizedEmail, now + lockoutDurationSeconds * 1000);
    inMemoryFailedLogins.delete(normalizedEmail);

    logger.warn('Auth login lockout triggered (in-memory)', {
      event: 'auth_lockout',
      email: redactEmail(normalizedEmail),
      ip,
      attempts: record.count,
      lockoutDurationSeconds,
    });
  } else {
    inMemoryFailedLogins.set(normalizedEmail, record);
  }
};

/**
 * Clear failure/lockout state after a successful login so that
 * occasional mistakes do not accumulate into a lockout.
 */
const resetLoginFailures = async (email: string): Promise<void> => {
  if (!isLoginLockoutEnabled()) {
    return;
  }

  const normalizedEmail = normalizeEmailForLockout(email);
  const cache = getCacheService();

  if (cache) {
    await cache.del(CacheKeys.authLoginFailures(normalizedEmail));
    await cache.del(CacheKeys.authLoginLockout(normalizedEmail));
    return;
  }

  inMemoryFailedLogins.delete(normalizedEmail);
  inMemoryLockouts.delete(normalizedEmail);
};

// Apply rate limiting to all auth routes
router.use(authRateLimiter);

// Register
router.post(
  '/register',
  asyncHandler(async (req: Request, res: Response) => {
    const { email, username, password } = RegisterSchema.parse(req.body);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Check if user already exists
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [{ email }, { username }],
      },
    });

    if (existingUser) {
      if (existingUser.email === email) {
        throw createError('Email already registered', 409, 'EMAIL_EXISTS');
      } else {
        throw createError('Username already taken', 409, 'USERNAME_EXISTS');
      }
    }

    // Hash password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Generate verification token
    const verificationToken = crypto.randomBytes(32).toString('hex');
    const verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours

    // Create user
    const user = await prisma.user.create({
      data: {
        email,
        username,
        // Persist hashed password into the schema's passwordHash field
        passwordHash: hashedPassword,
        role: 'USER',
        isActive: true,
        emailVerified: false,
        verificationToken,
        verificationTokenExpires,
      },
      select: {
        id: true,
        email: true,
        username: true,
        role: true,
        createdAt: true,
      },
    });

    // Send verification email
    try {
      await sendVerificationEmail(user.email, verificationToken);
    } catch (emailError) {
      httpLogger.error(req, 'Failed to send verification email', {
        userId: user.id,
        email: redactEmail(user.email),
        error: emailError instanceof Error ? emailError.message : String(emailError),
      });
      // Don't fail registration if email fails, but log it
    }

    // For newly registered users, tokenVersion always starts at 0. This is
    // embedded into the JWT so that future increments (via /logout-all) will
    // invalidate these tokens.
    const tokenVersion = 0;

    // Generate tokens
    const accessToken = generateToken({
      id: user.id,
      email: user.email,
      tokenVersion,
    });
    const refreshToken = generateRefreshToken({
      id: user.id,
      email: user.email,
      tokenVersion,
    });

    // Store refresh token in database if the model is available. In some
    // dev setups the RefreshToken model/table may not exist; in that case
    // we log and continue rather than throwing a hard runtime error.
    try {
      const refreshTokenModel = (prisma as any).refreshToken;
      if (refreshTokenModel && typeof refreshTokenModel.create === 'function') {
        const hashedToken = hashRefreshToken(refreshToken);

        // Ensure we only keep a single active refresh token per user. This is
        // stricter than required (per-tokenVersion) but keeps the model simple.
        if (typeof refreshTokenModel.deleteMany === 'function') {
          await refreshTokenModel.deleteMany({ where: { userId: user.id } });
        }

        await refreshTokenModel.create({
          data: {
            token: hashedToken,
            userId: user.id,
            expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
          },
        });
      } else {
        httpLogger.warn(
          req,
          'RefreshToken model not available; skipping refresh token persistence on register',
          {
            userId: user.id,
            email: redactEmail(user.email),
          }
        );
      }
    } catch (tokenError) {
      httpLogger.warn(
        req,
        'Failed to persist refresh token on register; continuing without DB-stored refresh token',
        {
          userId: user.id,
          email: redactEmail(user.email),
          error: tokenError instanceof Error ? tokenError.message : String(tokenError),
        }
      );
    }

    httpLogger.info(req, 'User registered successfully', {
      userId: user.id,
      email: redactEmail(user.email),
    });

    res.status(201).json({
      success: true,
      data: {
        user,
        accessToken,
        refreshToken,
      },
      message: 'User registered successfully',
    });
  })
);

// Login
router.post(
  '/login',
  asyncHandler(async (req: Request, res: Response) => {
    const { email, password } = LoginSchema.parse(req.body);
    const normalizedEmail = email.trim().toLowerCase();

    if (await isEmailLockedOut(normalizedEmail, req.ip)) {
      throw createError(
        'Too many failed login attempts. Please try again later.',
        429,
        'LOGIN_LOCKED_OUT'
      );
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Find user
    const user = await prisma.user.findUnique({
      where: { email },
      select: {
        id: true,
        email: true,
        username: true,
        // Load the hashed password from passwordHash for verification
        passwordHash: true,
        role: true,
        isActive: true,
        emailVerified: true,
      },
    });

    if (!user) {
      await recordFailedLoginAttempt(normalizedEmail, req.ip);
      throw createError('Invalid credentials', 401, 'INVALID_CREDENTIALS');
    }

    if (!user.isActive) {
      throw createError('Account is deactivated', 401, 'ACCOUNT_DEACTIVATED');
    }

    // Verify password with defensive guards so legacy/invalid hashes are treated
    // as invalid credentials rather than 500-level server errors.
    let isValidPassword = false;
    try {
      const hash = (user as any).passwordHash;
      if (typeof hash === 'string' && hash.length > 0) {
        isValidPassword = await bcrypt.compare(password, hash);
      } else {
        httpLogger.warn(
          req,
          'User record missing valid passwordHash; treating as invalid credentials',
          {
            userId: user.id,
            email: redactEmail(user.email),
          }
        );
        isValidPassword = false;
      }
    } catch (err) {
      httpLogger.warn(req, 'Password verification failed; treating as invalid credentials', {
        userId: user.id,
        email: redactEmail(user.email),
        error: err instanceof Error ? err.message : String(err),
      });
      isValidPassword = false;
    }

    if (!isValidPassword) {
      await recordFailedLoginAttempt(normalizedEmail, req.ip);
      throw createError('Invalid credentials', 401, 'INVALID_CREDENTIALS');
    }

    // Load the current tokenVersion for this user so that any future
    // increment (via /logout-all) will invalidate these tokens.
    let tokenVersion = 0;
    try {
      const userWithVersion = await (prisma as any).user.findUnique({
        where: { id: user.id },
        select: { tokenVersion: true },
      });
      if (userWithVersion && typeof userWithVersion.tokenVersion === 'number') {
        tokenVersion = userWithVersion.tokenVersion;
      }
    } catch (err) {
      httpLogger.warn(req, 'Failed to load tokenVersion for login; defaulting to 0', {
        userId: user.id,
        email: redactEmail(user.email),
        error: err instanceof Error ? err.message : String(err),
      });
    }

    // Generate tokens
    const accessToken = generateToken({
      id: user.id,
      email: user.email,
      tokenVersion,
    });
    const refreshToken = generateRefreshToken({
      id: user.id,
      email: user.email,
      tokenVersion,
    });

    // Store refresh token in database if the model is available. In some
    // dev setups the RefreshToken model/table may not exist; in that case
    // we log and continue rather than throwing a hard runtime error.
    try {
      const refreshTokenModel = (prisma as any).refreshToken;
      if (refreshTokenModel && typeof refreshTokenModel.create === 'function') {
        const hashedToken = hashRefreshToken(refreshToken);

        // Ensure a single active refresh token per user by removing any
        // previously stored tokens for this account.
        if (typeof refreshTokenModel.deleteMany === 'function') {
          await refreshTokenModel.deleteMany({ where: { userId: user.id } });
        }

        await refreshTokenModel.create({
          data: {
            token: hashedToken,
            userId: user.id,
            expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
          },
        });
      } else {
        httpLogger.warn(
          req,
          'RefreshToken model not available; skipping refresh token persistence on login',
          {
            userId: user.id,
            email: redactEmail(user.email),
          }
        );
      }
    } catch (tokenError) {
      httpLogger.warn(
        req,
        'Failed to persist refresh token on login; continuing without DB-stored refresh token',
        {
          userId: user.id,
          email: redactEmail(user.email),
          error: tokenError instanceof Error ? tokenError.message : String(tokenError),
        }
      );
    }
    // Update last login
    await prisma.user.update({
      where: { id: user.id },
      data: { lastLoginAt: new Date() },
    });

    await resetLoginFailures(normalizedEmail);

    httpLogger.info(req, 'User logged in successfully', {
      userId: user.id,
      email: redactEmail(user.email),
    });

    // Strip the passwordHash field before returning the user payload
    const { passwordHash: _, ...userWithoutPassword } = user;

    res.json({
      success: true,
      data: {
        user: userWithoutPassword,
        accessToken,
        refreshToken,
      },
      message: 'Login successful',
    });
  })
);

// Refresh token
router.post(
  '/refresh',
  asyncHandler(async (req: Request, res: Response) => {
    const parsed = RefreshTokenSchema.safeParse(req.body);

    if (!parsed.success) {
      const firstIssue = parsed.error.issues[0];
      if (firstIssue?.path[0] === 'refreshToken' && firstIssue.code === 'invalid_type') {
        // Preserve the specific error code used by existing consumers when the
        // token field is entirely missing from the payload.
        throw createError('Refresh token required', 400, 'REFRESH_TOKEN_REQUIRED');
      }

      throw createError('Invalid request', 400, 'INVALID_REQUEST');
    }

    const { refreshToken } = parsed.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Verify refresh token signature and basic payload.
    const decoded = verifyRefreshToken(refreshToken);

    // Enforce tokenVersion-based revocation in the same way as access
    // tokens by validating the user against the current DB state.
    await validateUser(decoded.userId, decoded.tokenVersion);

    const prismaAny = prisma as any;

    // Check if refresh token exists in database using its hashed value.
    const hashedToken = hashRefreshToken(refreshToken);

    const storedToken = await prismaAny.refreshToken.findFirst({
      where: {
        token: hashedToken,
        userId: decoded.userId,
        expiresAt: {
          gt: new Date(),
        },
      },
      include: {
        user: {
          select: {
            id: true,
            email: true,
            username: true,
            role: true,
            isActive: true,
          },
        },
      },
    });

    if (!storedToken) {
      throw createError('Invalid refresh token', 401, 'INVALID_REFRESH_TOKEN');
    }

    if (!storedToken.user.isActive) {
      throw createError('Account is deactivated', 401, 'ACCOUNT_DEACTIVATED');
    }

    const tokenVersion = typeof decoded.tokenVersion === 'number' ? decoded.tokenVersion : 0;

    // Generate new tokens carrying forward the same tokenVersion. If the
    // user's tokenVersion has been bumped (via /logout-all), the earlier
    // validateUser call will already have rejected this request.
    const newAccessToken = generateToken({
      id: storedToken.user.id,
      email: storedToken.user.email,
      tokenVersion,
    });
    const newRefreshToken = generateRefreshToken({
      id: storedToken.user.id,
      email: storedToken.user.email,
      tokenVersion,
    });

    const newHashedToken = hashRefreshToken(newRefreshToken);

    // Delete all existing refresh tokens for this user and store the new one,
    // enforcing a single active refresh token per user.
    await prisma.$transaction([
      prismaAny.refreshToken.deleteMany({
        where: { userId: storedToken.user.id },
      }),
      prismaAny.refreshToken.create({
        data: {
          token: newHashedToken,
          userId: storedToken.user.id,
          expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
        },
      }),
    ]);

    res.json({
      success: true,
      data: {
        accessToken: newAccessToken,
        refreshToken: newRefreshToken,
      },
      message: 'Tokens refreshed successfully',
    });
  })
);

// Logout
router.post(
  '/logout',
  authenticate,
  asyncHandler(async (_req: AuthenticatedRequest, res: Response) => {
    // For stateless access tokens, logout is a client-driven operation: the
    // browser discards its tokens. The server simply confirms that the
    // authenticated request was accepted without mutating tokenVersion.
    res.json({
      success: true,
      message: 'Logged out successfully',
    });
  })
);

// Logout all devices
router.post(
  '/logout-all',
  authenticate,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    if (!req.user) {
      throw createError('Authentication required', 401, 'AUTH_REQUIRED');
    }

    const prismaAny = prisma as any;

    // Increment the per-user tokenVersion to revoke all existing access and
    // refresh tokens for this account.
    if (prismaAny.user && typeof prismaAny.user.update === 'function') {
      await prismaAny.user.update({
        where: { id: req.user.id },
        data: {
          tokenVersion: {
            increment: 1,
          },
        },
      });
    }

    // Best-effort: clear any stored refresh tokens for this user so that
    // refresh attempts with old tokens fail even if tokenVersion is not
    // checked for some reason.
    const refreshTokenModel = prismaAny.refreshToken;
    if (refreshTokenModel && typeof refreshTokenModel.deleteMany === 'function') {
      await refreshTokenModel.deleteMany({
        where: { userId: req.user.id },
      });
    }

    res.json({
      success: true,
      message: 'Logged out from all devices successfully',
    });
  })
);

// Verify email
router.post(
  '/verify-email',
  asyncHandler(async (req: Request, res: Response) => {
    const parsed = VerifyEmailSchema.safeParse(req.body);

    if (!parsed.success) {
      const firstIssue = parsed.error.issues[0];
      if (firstIssue?.path[0] === 'token' && firstIssue.code === 'invalid_type') {
        throw createError('Verification token required', 400, 'TOKEN_REQUIRED');
      }

      throw createError('Invalid request', 400, 'INVALID_REQUEST');
    }

    const { token } = parsed.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Find user with valid token
    const user = await prisma.user.findFirst({
      where: {
        verificationToken: token,
        verificationTokenExpires: {
          gt: new Date(),
        },
      },
    });

    if (!user) {
      throw createError('Invalid or expired verification token', 400, 'INVALID_TOKEN');
    }

    // Update user
    await prisma.user.update({
      where: { id: user.id },
      data: {
        emailVerified: true,
        verificationToken: null,
        verificationTokenExpires: null,
      },
    });

    httpLogger.info(req, 'Email verified successfully', {
      userId: user.id,
      email: redactEmail(user.email),
    });

    res.json({
      success: true,
      message: 'Email verified successfully',
    });
  })
);

// Request password reset
router.post(
  '/forgot-password',
  asyncHandler(async (req: Request, res: Response) => {
    const parsed = ForgotPasswordSchema.safeParse(req.body);

    if (!parsed.success) {
      const firstIssue = parsed.error.issues[0];
      if (firstIssue?.path[0] === 'email' && firstIssue.code === 'invalid_type') {
        throw createError('Email required', 400, 'EMAIL_REQUIRED');
      }

      throw createError('Invalid request', 400, 'INVALID_REQUEST');
    }

    const { email } = parsed.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    const user = await prisma.user.findUnique({
      where: { email },
    });

    if (!user) {
      // Don't reveal if user exists
      res.json({
        success: true,
        message: 'If an account exists with this email, a password reset link has been sent.',
      });
      return;
    }

    // Generate reset token
    const resetToken = crypto.randomBytes(32).toString('hex');
    const resetExpires = new Date(Date.now() + 60 * 60 * 1000); // 1 hour

    // Save token to user
    await prisma.user.update({
      where: { id: user.id },
      data: {
        passwordResetToken: resetToken,
        passwordResetExpires: resetExpires,
      },
    });

    // Send email
    try {
      await sendPasswordResetEmail(user.email, resetToken);
    } catch (emailError) {
      httpLogger.error(req, 'Failed to send password reset email', {
        userId: user.id,
        email: redactEmail(user.email),
        error: emailError instanceof Error ? emailError.message : String(emailError),
      });
      throw createError('Failed to send password reset email', 500, 'EMAIL_SEND_FAILED');
    }

    res.json({
      success: true,
      message: 'If an account exists with this email, a password reset link has been sent.',
    });
  })
);

router.post(
  '/reset-password',
  asyncHandler(async (req: Request, res: Response) => {
    const parsed = ResetPasswordSchema.safeParse(req.body);

    if (!parsed.success) {
      const firstIssue = parsed.error.issues[0];

      if (firstIssue?.path[0] === 'newPassword' && firstIssue.code === 'too_small') {
        // Preserve the existing WEAK_PASSWORD code and human-friendly message
        // while delegating length enforcement to the shared Zod schema.
        throw createError('Password must be at least 8 characters long', 400, 'WEAK_PASSWORD');
      }

      throw createError('Invalid request', 400, 'INVALID_REQUEST');
    }

    const { token, newPassword } = parsed.data;

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Find user with valid token
    const user = await prisma.user.findFirst({
      where: {
        passwordResetToken: token,
        passwordResetExpires: {
          gt: new Date(),
        },
      },
    });

    if (!user) {
      throw createError('Invalid or expired password reset token', 400, 'INVALID_TOKEN');
    }

    // Hash new password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

    // Update user
    await prisma.user.update({
      where: { id: user.id },
      data: {
        passwordHash: hashedPassword,
        passwordResetToken: null,
        passwordResetExpires: null,
      },
    });

    httpLogger.info(req, 'Password reset successfully', {
      userId: user.id,
      email: redactEmail(user.email),
    });

    res.json({
      success: true,
      message: 'Password reset successfully',
    });
  })
);

export const __testResetLoginLockoutState = () => {
  inMemoryFailedLogins.clear();
  inMemoryLockouts.clear();
};

export default router;
