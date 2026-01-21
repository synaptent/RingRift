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
import { authRegisterRateLimiter, authPasswordResetRateLimiter } from '../middleware/rateLimiter';
import { logger, httpLogger, redactEmail } from '../utils/logger';
import {
  audit,
  auditLoginSuccess,
  auditLoginFailed,
  auditLogout,
  auditTokenRefresh,
  auditLockout,
  auditRegister,
} from '../utils/auditLogger';
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
import { config, parseDurationToSeconds } from '../config';

const router = Router();

/**
 * Refresh token expiry in milliseconds (7 days).
 * This is used consistently across all token creation points.
 */
const REFRESH_TOKEN_EXPIRY_MS = 7 * 24 * 60 * 60 * 1000;

/**
 * Extended refresh token expiry in milliseconds (30 days).
 * Used when the user selects "Remember me" during login.
 */
const REMEMBER_ME_EXPIRY_MS = 30 * 24 * 60 * 60 * 1000;

/**
 * Cookie options for secure refresh token handling.
 * In production: httpOnly, secure (HTTPS only), sameSite=strict
 * In development: httpOnly, sameSite=lax (allows localhost testing)
 *
 * @param expiryMs - Custom expiry in milliseconds (defaults to REFRESH_TOKEN_EXPIRY_MS)
 */
const getRefreshTokenCookieOptions = (
  expiryMs: number = REFRESH_TOKEN_EXPIRY_MS
): {
  httpOnly: boolean;
  secure: boolean;
  sameSite: 'strict' | 'lax' | 'none';
  maxAge: number;
  path: string;
} => ({
  httpOnly: true,
  secure: config.isProduction,
  sameSite: config.isProduction ? 'strict' : 'lax',
  maxAge: expiryMs,
  path: '/api/auth', // Only send cookie to auth endpoints
});

/**
 * Generate a new token family ID for tracking refresh token chains.
 * Each login creates a new family; rotated tokens inherit the family.
 */
const generateFamilyId = (): string => crypto.randomUUID();

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

      // Audit log the lockout
      auditLockout(normalizedEmail, ip ? { ip, headers: {} } : undefined, {
        attempts: record.count,
        lockoutDuration: lockoutDurationSeconds,
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

    // Audit log the lockout
    auditLockout(normalizedEmail, ip ? { ip, headers: {} } : undefined, {
      attempts: record.count,
      lockoutDuration: lockoutDurationSeconds,
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

/**
 * @openapi
 * /auth/register:
 *   post:
 *     summary: Register a new user account
 *     description: |
 *       Creates a new user account with the provided credentials.
 *       A verification email will be sent to the provided email address.
 *       Returns access and refresh tokens upon successful registration.
 *     tags: [Authentication]
 *     security: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/RegisterRequest'
 *     responses:
 *       201:
 *         description: User registered successfully
 *         headers:
 *           Set-Cookie:
 *             schema:
 *               type: string
 *             description: httpOnly refresh token cookie
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/AuthResponse'
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       409:
 *         description: Email or username already exists
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               emailExists:
 *                 summary: Email already registered
 *                 value:
 *                   success: false
 *                   error:
 *                     code: RESOURCE_EMAIL_EXISTS
 *                     message: Email already registered
 *               usernameExists:
 *                 summary: Username already taken
 *                 value:
 *                   success: false
 *                   error:
 *                     code: RESOURCE_USERNAME_EXISTS
 *                     message: Username already taken
 *       429:
 *         $ref: '#/components/responses/TooManyRequests'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/register',
  authRegisterRateLimiter,
  asyncHandler(async (req: Request, res: Response) => {
    const { email, username, password } = RegisterSchema.parse(req.body);

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Check if user already exists (exclude soft-deleted users to allow email/username reuse)
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [{ email }, { username }],
        deletedAt: null,
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

    // Store refresh token in database with token family tracking.
    // Each registration creates a new token family for rotation tracking.
    const familyId = generateFamilyId();
    try {
      const refreshTokenModel = prisma.refreshToken;
      if (refreshTokenModel) {
        const hashedToken = hashRefreshToken(refreshToken);

        // Ensure we only keep a single active refresh token per user. This is
        // stricter than required (per-tokenVersion) but keeps the model simple.
        await refreshTokenModel.deleteMany({ where: { userId: user.id } });

        await refreshTokenModel.create({
          data: {
            token: hashedToken,
            userId: user.id,
            familyId,
            expiresAt: new Date(Date.now() + REFRESH_TOKEN_EXPIRY_MS),
            rememberMe: false, // Registration doesn't have "Remember me" option
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

    // Audit log successful registration
    auditRegister(user.id, user.email, req);

    // Set refresh token as httpOnly cookie for security
    res.cookie('refreshToken', refreshToken, getRefreshTokenCookieOptions());

    res.status(201).json({
      success: true,
      data: {
        user,
        accessToken,
        // Note: refreshToken still included in body for backward compatibility
        // and for clients that prefer to manage tokens themselves.
        // The httpOnly cookie provides an additional secure transport.
        refreshToken,
      },
      message: 'User registered successfully',
    });
  })
);

/**
 * @openapi
 * /auth/login:
 *   post:
 *     summary: Authenticate user and get tokens
 *     description: |
 *       Authenticates a user with email and password.
 *       Returns access and refresh tokens upon successful authentication.
 *       Implements login lockout after multiple failed attempts.
 *     tags: [Authentication]
 *     security: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/LoginRequest'
 *     responses:
 *       200:
 *         description: Login successful
 *         headers:
 *           Set-Cookie:
 *             schema:
 *               type: string
 *             description: httpOnly refresh token cookie
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/AuthResponse'
 *       400:
 *         $ref: '#/components/responses/BadRequest'
 *       401:
 *         description: Invalid credentials or account deactivated
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               invalidCredentials:
 *                 summary: Invalid credentials
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_INVALID_CREDENTIALS
 *                     message: Invalid credentials
 *               accountDeactivated:
 *                 summary: Account deactivated
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_ACCOUNT_DEACTIVATED
 *                     message: Account is deactivated
 *       429:
 *         description: Too many failed attempts - account locked
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: AUTH_LOGIN_LOCKED_OUT
 *                 message: Too many failed login attempts. Please try again later.
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/login',
  asyncHandler(async (req: Request, res: Response) => {
    const { email, password, rememberMe } = LoginSchema.parse(req.body);
    const normalizedEmail = email.trim().toLowerCase();

    // Determine token expiry based on rememberMe flag
    const tokenExpiryMs = rememberMe ? REMEMBER_ME_EXPIRY_MS : REFRESH_TOKEN_EXPIRY_MS;

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

    // Find user (exclude soft-deleted users)
    const user = await prisma.user.findFirst({
      where: {
        email,
        deletedAt: null,
      },
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
      auditLoginFailed(email, 'User not found', req);
      throw createError('Invalid credentials', 401, 'INVALID_CREDENTIALS');
    }

    if (!user.isActive) {
      throw createError('Account is deactivated', 401, 'ACCOUNT_DEACTIVATED');
    }

    // Verify password with defensive guards so legacy/invalid hashes are treated
    // as invalid credentials rather than 500-level server errors.
    let isValidPassword = false;
    try {
      const hash = user.passwordHash;
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
      auditLoginFailed(email, 'Invalid password', req);
      throw createError('Invalid credentials', 401, 'INVALID_CREDENTIALS');
    }

    // Load the current tokenVersion for this user so that any future
    // increment (via /logout-all) will invalidate these tokens.
    let tokenVersion = 0;
    try {
      const userWithVersion = await prisma.user.findUnique({
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

    // Store refresh token in database with a new token family.
    // Each login creates a new family for rotation tracking.
    const familyId = generateFamilyId();
    try {
      const refreshTokenModel = prisma.refreshToken;
      if (refreshTokenModel) {
        const hashedToken = hashRefreshToken(refreshToken);

        // Ensure a single active refresh token per user by removing any
        // previously stored tokens for this account.
        await refreshTokenModel.deleteMany({ where: { userId: user.id } });

        await refreshTokenModel.create({
          data: {
            token: hashedToken,
            userId: user.id,
            familyId,
            expiresAt: new Date(Date.now() + tokenExpiryMs),
            rememberMe: !!rememberMe,
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

    // Audit log successful login
    auditLoginSuccess(user.id, user.email, req);

    // Strip the passwordHash field before returning the user payload
    const { passwordHash: _, ...userWithoutPassword } = user;

    // Set refresh token as httpOnly cookie for security
    res.cookie('refreshToken', refreshToken, getRefreshTokenCookieOptions(tokenExpiryMs));

    // Calculate token TTL in seconds from config for client token management
    const expiresIn = parseDurationToSeconds(config.auth.accessTokenExpiresIn);

    res.json({
      success: true,
      data: {
        user: userWithoutPassword,
        accessToken,
        // Note: refreshToken still included in body for backward compatibility
        refreshToken,
        // Token TTL in seconds - enables clients to schedule proactive refresh
        expiresIn,
      },
      message: 'Login successful',
    });
  })
);

/**
 * @openapi
 * /auth/refresh:
 *   post:
 *     summary: Refresh access token
 *     description: |
 *       Exchanges a valid refresh token for new access and refresh tokens.
 *       Implements token rotation - each refresh token can only be used once.
 *       Detects token reuse attacks and invalidates the entire token family.
 *
 *       The refresh token can be provided in:
 *       - Request body (refreshToken field)
 *       - httpOnly cookie (set automatically by login/register)
 *     tags: [Authentication]
 *     security: []
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/RefreshTokenRequest'
 *     responses:
 *       200:
 *         description: Tokens refreshed successfully
 *         headers:
 *           Set-Cookie:
 *             schema:
 *               type: string
 *             description: New httpOnly refresh token cookie
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/TokenRefreshResponse'
 *       400:
 *         description: Refresh token required but not provided
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: AUTH_REFRESH_TOKEN_REQUIRED
 *                 message: Refresh token required
 *       401:
 *         description: Invalid, expired, or reused refresh token
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               invalidToken:
 *                 summary: Invalid token
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_REFRESH_TOKEN_INVALID
 *                     message: Invalid refresh token
 *               expiredToken:
 *                 summary: Token expired
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_REFRESH_TOKEN_EXPIRED
 *                     message: Refresh token has expired
 *               tokenReused:
 *                 summary: Token reuse detected
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_REFRESH_TOKEN_REUSED
 *                     message: Refresh token has been revoked due to suspicious activity
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/refresh',
  asyncHandler(async (req: Request, res: Response) => {
    // Accept refresh token from cookie OR request body for flexibility
    let refreshToken = req.cookies?.refreshToken;

    if (!refreshToken) {
      const parsed = RefreshTokenSchema.safeParse(req.body);
      if (!parsed.success) {
        const firstIssue = parsed.error.issues[0];
        if (firstIssue?.path[0] === 'refreshToken' && firstIssue.code === 'invalid_type') {
          throw createError('Refresh token required', 400, 'REFRESH_TOKEN_REQUIRED');
        }
        throw createError('Invalid request', 400, 'INVALID_REQUEST');
      }
      refreshToken = parsed.data.refreshToken;
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Verify refresh token signature and basic payload.
    const decoded = verifyRefreshToken(refreshToken);

    // Enforce tokenVersion-based revocation in the same way as access
    // tokens by validating the user against the current DB state.
    await validateUser(decoded.userId, decoded.tokenVersion);

    const hashedToken = hashRefreshToken(refreshToken);

    // Look up the token, including revoked ones to detect reuse attacks
    const storedToken = await prisma.refreshToken.findFirst({
      where: {
        token: hashedToken,
        userId: decoded.userId,
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

    // SECURITY: Detect refresh token reuse attack
    // If a previously rotated (revoked) token is being reused, this indicates
    // the token chain has been compromised. Revoke the entire token family.
    if (storedToken.revokedAt) {
      httpLogger.warn(req, 'Refresh token reuse detected - potential token theft', {
        event: 'refresh_token_reuse',
        userId: decoded.userId,
        familyId: storedToken.familyId,
        tokenId: storedToken.id,
        revokedAt: storedToken.revokedAt,
      });

      // Audit log the suspicious activity
      audit('security.suspicious.activity', 'blocked', {
        userId: decoded.userId,
        req,
        reason: 'Refresh token reuse detected - potential token theft',
        details: {
          familyId: storedToken.familyId,
          tokenId: storedToken.id,
          revokedAt: storedToken.revokedAt.toISOString(),
        },
      });

      // Revoke ALL tokens in this family to prevent further abuse
      if (storedToken.familyId) {
        await prisma.refreshToken.updateMany({
          where: { familyId: storedToken.familyId },
          data: { revokedAt: new Date() },
        });
      }

      // Also increment tokenVersion to invalidate all access tokens
      await prisma.user.update({
        where: { id: decoded.userId },
        data: { tokenVersion: { increment: 1 } },
      });

      throw createError(
        'Refresh token has been revoked due to suspicious activity',
        401,
        'REFRESH_TOKEN_REUSED'
      );
    }

    // Check expiry (only for non-revoked tokens)
    if (new Date() > storedToken.expiresAt) {
      throw createError('Refresh token has expired', 401, 'REFRESH_TOKEN_EXPIRED');
    }

    if (!storedToken.user.isActive) {
      throw createError('Account is deactivated', 401, 'ACCOUNT_DEACTIVATED');
    }

    const tokenVersion = typeof decoded.tokenVersion === 'number' ? decoded.tokenVersion : 0;

    // Generate new tokens carrying forward the same tokenVersion and family.
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

    // Determine expiry based on whether the original login used "Remember me"
    const tokenExpiryMs = storedToken.rememberMe ? REMEMBER_ME_EXPIRY_MS : REFRESH_TOKEN_EXPIRY_MS;

    // Rotate the refresh token: mark old one as revoked, create new one
    // This allows us to detect reuse of the old token later.
    await prisma.$transaction([
      // Mark the current token as revoked (not deleted, for reuse detection)
      prisma.refreshToken.update({
        where: { id: storedToken.id },
        data: { revokedAt: new Date() },
      }),
      // Create the new token in the same family, preserving rememberMe preference
      prisma.refreshToken.create({
        data: {
          token: newHashedToken,
          userId: storedToken.user.id,
          familyId: storedToken.familyId || generateFamilyId(),
          expiresAt: new Date(Date.now() + tokenExpiryMs),
          rememberMe: storedToken.rememberMe,
        },
      }),
    ]);

    // Set new refresh token as httpOnly cookie with appropriate expiry
    res.cookie('refreshToken', newRefreshToken, getRefreshTokenCookieOptions(tokenExpiryMs));

    // Audit log the token refresh
    auditTokenRefresh(storedToken.user.id, req, { familyId: storedToken.familyId || undefined });

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

/**
 * @openapi
 * /auth/logout:
 *   post:
 *     summary: Logout current session
 *     description: |
 *       Revokes the current refresh token and clears the refresh token cookie.
 *       The access token will remain valid until it expires, but the refresh
 *       token cannot be used to obtain new tokens.
 *     tags: [Authentication]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               refreshToken:
 *                 type: string
 *                 description: Optional - refresh token to revoke (if not using cookie)
 *     responses:
 *       200:
 *         description: Logged out successfully
 *         headers:
 *           Set-Cookie:
 *             schema:
 *               type: string
 *             description: Clears the refresh token cookie
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: Logged out successfully
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 */
router.post(
  '/logout',
  authenticate,
  asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
    const prisma = getDatabaseClient();

    // Best-effort: revoke the refresh token if present
    const refreshToken = req.cookies?.refreshToken || req.body?.refreshToken;

    if (prisma && refreshToken && req.user?.id) {
      try {
        const hashedToken = hashRefreshToken(refreshToken);

        // Mark the token as revoked (not deleted) for reuse detection
        await prisma.refreshToken.updateMany({
          where: {
            token: hashedToken,
            userId: req.user.id,
          },
          data: { revokedAt: new Date() },
        });
      } catch (err) {
        // Log but don't fail the logout
        httpLogger.warn(req, 'Failed to revoke refresh token on logout', {
          userId: req.user?.id,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    // Clear the refresh token cookie
    res.clearCookie('refreshToken', {
      httpOnly: true,
      secure: config.isProduction,
      sameSite: config.isProduction ? 'strict' : 'lax',
      path: '/api/auth',
    });

    // Audit log the logout
    if (req.user?.id) {
      auditLogout(req.user.id, req);
    }

    res.json({
      success: true,
      message: 'Logged out successfully',
    });
  })
);

/**
 * @openapi
 * /auth/logout-all:
 *   post:
 *     summary: Logout from all devices
 *     description: |
 *       Invalidates all existing access and refresh tokens for the user.
 *       Increments the user's tokenVersion, which causes all previously
 *       issued tokens to fail validation. Use this when:
 *       - User suspects account compromise
 *       - User wants to sign out everywhere
 *       - Security incident response
 *     tags: [Authentication]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Logged out from all devices
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: Logged out from all devices successfully
 *       401:
 *         $ref: '#/components/responses/Unauthorized'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
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

    // Increment the per-user tokenVersion to revoke all existing access and
    // refresh tokens for this account.
    await prisma.user.update({
      where: { id: req.user.id },
      data: {
        tokenVersion: {
          increment: 1,
        },
      },
    });

    // Best-effort: clear any stored refresh tokens for this user so that
    // refresh attempts with old tokens fail even if tokenVersion is not
    // checked for some reason.
    await prisma.refreshToken.deleteMany({
      where: { userId: req.user.id },
    });

    // Audit log the logout-all action
    auditLogout(req.user.id, req);

    res.json({
      success: true,
      message: 'Logged out from all devices successfully',
    });
  })
);

/**
 * @openapi
 * /auth/verify-email:
 *   post:
 *     summary: Verify email address
 *     description: |
 *       Verifies the user's email address using the token sent via email.
 *       The token expires after 24 hours.
 *     tags: [Authentication]
 *     security: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/VerifyEmailRequest'
 *     responses:
 *       200:
 *         description: Email verified successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: Email verified successfully
 *       400:
 *         description: Invalid or expired verification token
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               tokenRequired:
 *                 summary: Token required
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_VERIFICATION_TOKEN_REQUIRED
 *                     message: Verification token required
 *               invalidToken:
 *                 summary: Invalid or expired token
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_VERIFICATION_INVALID
 *                     message: Invalid or expired verification token
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
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

    // Find user with valid token (exclude soft-deleted users)
    const user = await prisma.user.findFirst({
      where: {
        verificationToken: token,
        verificationTokenExpires: {
          gt: new Date(),
        },
        deletedAt: null,
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

/**
 * @openapi
 * /auth/forgot-password:
 *   post:
 *     summary: Request password reset
 *     description: |
 *       Sends a password reset email to the specified address if an account exists.
 *       For security, the response is the same whether or not an account exists.
 *       The reset token expires after 1 hour.
 *     tags: [Authentication]
 *     security: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ForgotPasswordRequest'
 *     responses:
 *       200:
 *         description: Password reset email sent (if account exists)
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: If an account exists with this email, a password reset link has been sent.
 *       400:
 *         description: Email required
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: VALIDATION_EMAIL_REQUIRED
 *                 message: Email required
 *       429:
 *         $ref: '#/components/responses/TooManyRequests'
 *       500:
 *         description: Failed to send email
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error:
 *                 code: SERVER_EMAIL_SEND_FAILED
 *                 message: Failed to send password reset email
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/forgot-password',
  authPasswordResetRateLimiter,
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

    // Exclude soft-deleted users from password reset
    const user = await prisma.user.findFirst({
      where: {
        email,
        deletedAt: null,
      },
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
      const emailSent = await sendPasswordResetEmail(user.email, resetToken);
      if (!emailSent) {
        httpLogger.error(req, 'Failed to send password reset email (returned false)', {
          userId: user.id,
          email: redactEmail(user.email),
        });
        throw createError('Failed to send password reset email', 500, 'EMAIL_SEND_FAILED');
      }
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

/**
 * @openapi
 * /auth/reset-password:
 *   post:
 *     summary: Reset password with token
 *     description: |
 *       Resets the user's password using a valid reset token.
 *       Upon success, all existing sessions are invalidated (tokenVersion incremented).
 *       This ensures that any compromised sessions cannot be used.
 *     tags: [Authentication]
 *     security: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/ResetPasswordRequest'
 *     responses:
 *       200:
 *         description: Password reset successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: Password reset successfully
 *       400:
 *         description: Invalid token or weak password
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             examples:
 *               invalidToken:
 *                 summary: Invalid or expired token
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_RESET_TOKEN_INVALID
 *                     message: Invalid or expired password reset token
 *               weakPassword:
 *                 summary: Password too weak
 *                 value:
 *                   success: false
 *                   error:
 *                     code: AUTH_WEAK_PASSWORD
 *                     message: Password must be at least 8 characters long
 *       429:
 *         $ref: '#/components/responses/TooManyRequests'
 *       503:
 *         $ref: '#/components/responses/ServiceUnavailable'
 */
router.post(
  '/reset-password',
  authPasswordResetRateLimiter,
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

    // Find user with valid token (exclude soft-deleted users)
    const user = await prisma.user.findFirst({
      where: {
        passwordResetToken: token,
        passwordResetExpires: {
          gt: new Date(),
        },
        deletedAt: null,
      },
    });

    if (!user) {
      throw createError('Invalid or expired password reset token', 400, 'INVALID_TOKEN');
    }

    // Hash new password
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

    // Update user password and increment tokenVersion to invalidate all existing
    // tokens. This is a critical security measure: when a user resets their
    // password, all previous sessions should be invalidated.
    await prisma.user.update({
      where: { id: user.id },
      data: {
        passwordHash: hashedPassword,
        passwordResetToken: null,
        passwordResetExpires: null,
        tokenVersion: { increment: 1 },
      },
    });

    // Also revoke all refresh tokens for this user
    try {
      await prisma.refreshToken.updateMany({
        where: { userId: user.id },
        data: { revokedAt: new Date() },
      });
    } catch (err) {
      httpLogger.warn(req, 'Failed to revoke refresh tokens on password reset', {
        userId: user.id,
        error: err instanceof Error ? err.message : String(err),
      });
    }

    httpLogger.info(req, 'Password reset successfully - all sessions invalidated', {
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
