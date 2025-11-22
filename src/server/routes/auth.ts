import { Router, Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';
import { getDatabaseClient } from '../database/connection';
import { generateToken, generateRefreshToken, verifyRefreshToken } from '../middleware/auth';
import { createError, asyncHandler } from '../middleware/errorHandler';
import { authRateLimiter } from '../middleware/rateLimiter';
import { logger } from '../utils/logger';
import { sendVerificationEmail, sendPasswordResetEmail } from '../utils/email';
import { RegisterSchema, LoginSchema } from '../../shared/validation/schemas';

const router = Router();

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
      logger.error('Failed to send verification email', {
        userId: user.id,
        email: user.email,
        error: emailError instanceof Error ? emailError.message : String(emailError),
      });
      // Don't fail registration if email fails, but log it
    }

    // Generate tokens
    const accessToken = generateToken({ id: user.id, email: user.email });
    const refreshToken = generateRefreshToken({ id: user.id, email: user.email });

    // Store refresh token in database if the model is available. In some
    // dev setups the RefreshToken model/table may not exist; in that case
    // we log and continue rather than throwing a hard runtime error.
    try {
      const refreshTokenModel = (prisma as any).refreshToken;
      if (refreshTokenModel && typeof refreshTokenModel.create === 'function') {
        await refreshTokenModel.create({
          data: {
            token: refreshToken,
            userId: user.id,
            expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
          },
        });
      } else {
        logger.warn(
          'RefreshToken model not available; skipping refresh token persistence on register',
          {
            userId: user.id,
            email: user.email,
          }
        );
      }
    } catch (tokenError) {
      logger.warn(
        'Failed to persist refresh token on register; continuing without DB-stored refresh token',
        {
          userId: user.id,
          email: user.email,
          error: tokenError instanceof Error ? tokenError.message : String(tokenError),
        }
      );
    }

    logger.info('User registered successfully', { userId: user.id, email: user.email });

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
        logger.warn('User record missing valid passwordHash; treating as invalid credentials', {
          userId: user.id,
          email: user.email,
        });
        isValidPassword = false;
      }
    } catch (err) {
      logger.warn('Password verification failed; treating as invalid credentials', {
        userId: user.id,
        email: user.email,
        error: err instanceof Error ? err.message : String(err),
      });
      isValidPassword = false;
    }

    if (!isValidPassword) {
      throw createError('Invalid credentials', 401, 'INVALID_CREDENTIALS');
    }

    // Generate tokens
    const accessToken = generateToken({ id: user.id, email: user.email });
    const refreshToken = generateRefreshToken({ id: user.id, email: user.email });

    // Store refresh token in database if the model is available. In some
    // dev setups the RefreshToken model/table may not exist; in that case
    // we log and continue rather than throwing a hard runtime error.
    try {
      const refreshTokenModel = (prisma as any).refreshToken;
      if (refreshTokenModel && typeof refreshTokenModel.create === 'function') {
        await refreshTokenModel.create({
          data: {
            token: refreshToken,
            userId: user.id,
            expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
          },
        });
      } else {
        logger.warn(
          'RefreshToken model not available; skipping refresh token persistence on login',
          {
            userId: user.id,
            email: user.email,
          }
        );
      }
    } catch (tokenError) {
      logger.warn(
        'Failed to persist refresh token on login; continuing without DB-stored refresh token',
        {
          userId: user.id,
          email: user.email,
          error: tokenError instanceof Error ? tokenError.message : String(tokenError),
        }
      );
    }

    // Update last login
    await prisma.user.update({
      where: { id: user.id },
      data: { lastLoginAt: new Date() },
    });

    logger.info('User logged in successfully', { userId: user.id, email: user.email });

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
    const { refreshToken } = req.body;

    if (!refreshToken) {
      throw createError('Refresh token required', 400, 'REFRESH_TOKEN_REQUIRED');
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Verify refresh token
    const decoded = verifyRefreshToken(refreshToken);

    // Check if refresh token exists in database
    const storedToken = await (prisma as any).refreshToken.findFirst({
      where: {
        token: refreshToken,
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

    // Generate new tokens
    const newAccessToken = generateToken({
      id: storedToken.user.id,
      email: storedToken.user.email,
    });
    const newRefreshToken = generateRefreshToken({
      id: storedToken.user.id,
      email: storedToken.user.email,
    });

    // Delete old refresh token and create new one
    await prisma.$transaction([
      (prisma as any).refreshToken.delete({
        where: { id: storedToken.id },
      }),
      (prisma as any).refreshToken.create({
        data: {
          token: newRefreshToken,
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
  asyncHandler(async (req: Request, res: Response) => {
    const { refreshToken } = req.body;

    if (!refreshToken) {
      return res.json({
        success: true,
        message: 'Logged out successfully',
      });
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Delete refresh token from database
    await (prisma as any).refreshToken.deleteMany({
      where: { token: refreshToken },
    });

    return res.json({
      success: true,
      message: 'Logged out successfully',
    });
  })
);

// Logout all devices
router.post(
  '/logout-all',
  asyncHandler(async (req: Request, res: Response) => {
    const { refreshToken } = req.body;

    if (!refreshToken) {
      throw createError('Refresh token required', 400, 'REFRESH_TOKEN_REQUIRED');
    }

    const prisma = getDatabaseClient();
    if (!prisma) {
      throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
    }

    // Verify refresh token to get user ID
    const decoded = verifyRefreshToken(refreshToken);

    // Delete all refresh tokens for this user
    await (prisma as any).refreshToken.deleteMany({
      where: { userId: decoded.userId },
    });

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
    const { token } = req.body;

    if (!token) {
      throw createError('Verification token required', 400, 'TOKEN_REQUIRED');
    }

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

    logger.info('Email verified successfully', { userId: user.id, email: user.email });

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
    const { email } = req.body;

    if (!email) {
      throw createError('Email required', 400, 'EMAIL_REQUIRED');
    }

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
      logger.error('Failed to send password reset email', {
        userId: user.id,
        email: user.email,
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

// Reset password
router.post(
  '/reset-password',
  asyncHandler(async (req: Request, res: Response) => {
    const { token, newPassword } = req.body;

    if (!token || !newPassword) {
      throw createError('Token and new password required', 400, 'INVALID_REQUEST');
    }

    if (newPassword.length < 8) {
      throw createError('Password must be at least 8 characters long', 400, 'WEAK_PASSWORD');
    }

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

    logger.info('Password reset successfully', { userId: user.id, email: user.email });

    res.json({
      success: true,
      message: 'Password reset successfully',
    });
  })
);

export default router;
