import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { getDatabaseClient } from '../database/connection';
import { createError } from './errorHandler';
import { logger } from '../utils/logger';
import { config } from '../config';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    username: string;
    role: string;
  };
}

export interface VerifiedAccessToken {
  userId: string;
  email: string;
  /**
   * Per-user token version captured at the time the token was issued.
   * When absent, it is treated as version 0 for backwards compatibility.
   */
  tokenVersion?: number;
}

export const authenticate = async (
  req: AuthenticatedRequest,
  _res: Response,
  next: NextFunction
) => {
  try {
    const token = extractToken(req);

    if (!token) {
      throw createError('Authentication token required', 401, 'TOKEN_REQUIRED');
    }

    const decoded = verifyToken(token);
    const user = await validateUser(decoded.userId, decoded.tokenVersion);

    req.user = user;
    next();
  } catch (error) {
    next(error);
  }
};

export const optionalAuth = async (
  req: AuthenticatedRequest,
  _res: Response,
  next: NextFunction
) => {
  try {
    const token = extractToken(req);

    if (token) {
      const decoded = verifyToken(token);
      const user = await validateUser(decoded.userId, decoded.tokenVersion);
      req.user = user;
    }

    next();
  } catch (error) {
    // For optional auth, we don't fail on invalid tokens
    logger.warn('Optional auth failed:', error);
    next();
  }
};

export const authorize = (roles: string[]) => {
  return (req: AuthenticatedRequest, _res: Response, next: NextFunction) => {
    if (!req.user) {
      return next(createError('Authentication required', 401, 'AUTH_REQUIRED'));
    }

    if (!roles.includes(req.user.role)) {
      return next(createError('Insufficient permissions', 403, 'INSUFFICIENT_PERMISSIONS'));
    }

    next();
  };
};

const extractToken = (req: Request): string | null => {
  // Check Authorization header
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    return authHeader.substring(7);
  }

  // Check cookie
  const cookieToken = req.cookies?.token;
  if (cookieToken) {
    return cookieToken;
  }

  // Check query parameter (for WebSocket connections)
  const queryToken = req.query.token as string;
  if (queryToken) {
    return queryToken;
  }

  return null;
};

export const getAccessTokenSecret = (): string => {
  return config.auth.jwtSecret;
};

const getRefreshTokenSecret = (): string => {
  return config.auth.jwtRefreshSecret;
};

/**
 * Verify an access token and extract the core identity claims.
 *
 * The payload includes an optional `tokenVersion` (tv) claim which is used
 * to enforce server-side revocation via the User.tokenVersion field.
 */
export const verifyToken = (token: string): VerifiedAccessToken => {
  try {
    const secret = getAccessTokenSecret();
    const decoded = jwt.verify(token, secret) as any;

    if (!decoded.userId || !decoded.email) {
      throw new Error('Invalid token payload');
    }

    const tokenVersion = typeof decoded.tv === 'number' ? decoded.tv : undefined;

    return {
      userId: decoded.userId,
      email: decoded.email,
      tokenVersion,
    };
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      throw createError('Token has expired', 401, 'TOKEN_EXPIRED');
    } else if (error instanceof jwt.JsonWebTokenError) {
      throw createError('Invalid token', 401, 'INVALID_TOKEN');
    } else {
      throw createError('Token verification failed', 401, 'TOKEN_VERIFICATION_FAILED');
    }
  }
};

/**
 * Load and validate a user for an authenticated request.
 *
 * In addition to basic existence / isActive checks, this enforces that the
 * token's embedded tokenVersion (if present) matches the current value in
 * the database. Incrementing User.tokenVersion therefore revokes all
 * previously issued tokens for that user.
 */
export const validateUser = async (userId: string, tokenVersion?: number) => {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  // Use a loosely-typed query here so that the code continues to compile
  // even if the generated Prisma client has not yet been regenerated with
  // the new `User.tokenVersion` field. At runtime, the field will be present
  // once migrations and `prisma generate` have been applied.
  const user = (await (prisma as any).user.findUnique({
    where: { id: userId },
  })) as any;

  if (!user) {
    throw createError('User not found', 401, 'USER_NOT_FOUND');
  }

  if (!user.isActive) {
    throw createError('Account is deactivated', 401, 'ACCOUNT_DEACTIVATED');
  }

  const currentVersion: number = typeof user.tokenVersion === 'number' ? user.tokenVersion : 0;
  const claimedVersion: number = typeof tokenVersion === 'number' ? tokenVersion : 0;

  if (claimedVersion !== currentVersion) {
    // Treat mismatched tokenVersion as a revoked / invalid token. We reuse the
    // existing INVALID_TOKEN code so that HTTP and WebSocket callers see a
    // consistent error surface.
    throw createError('Token has been revoked', 401, 'INVALID_TOKEN');
  }

  return {
    id: user.id,
    email: user.email,
    username: user.username,
    role: user.role,
  };
};

export const generateToken = (user: {
  id: string;
  email: string;
  tokenVersion?: number;
}): string => {
  const secret = getAccessTokenSecret();

  const payload: any = {
    userId: user.id,
    email: user.email,
  };

  // Embed the per-user token version so that server-side revocation can be
  // enforced on each authenticated request.
  if (typeof user.tokenVersion === 'number') {
    payload.tv = user.tokenVersion;
  }

  const options: any = {
    expiresIn: config.auth.accessTokenExpiresIn,
    issuer: 'ringrift',
    audience: 'ringrift-users',
  };

  return jwt.sign(payload, secret, options);
};

export const generateRefreshToken = (user: {
  id: string;
  email: string;
  tokenVersion?: number;
}): string => {
  const secret = getRefreshTokenSecret();

  const payload: any = {
    userId: user.id,
    email: user.email,
    type: 'refresh',
  };

  // Mirror the access-token tokenVersion claim on refresh tokens so that
  // rotation and revocation can be handled consistently.
  if (typeof user.tokenVersion === 'number') {
    payload.tv = user.tokenVersion;
  }

  const options: any = {
    expiresIn: config.auth.refreshTokenExpiresIn,
    issuer: 'ringrift',
    audience: 'ringrift-users',
  };

  return jwt.sign(payload, secret, options);
};

export const verifyRefreshToken = (
  token: string
): { userId: string; email: string; tokenVersion?: number } => {
  try {
    const secret = getRefreshTokenSecret();

    const decoded = jwt.verify(token, secret) as any;

    if (!decoded.userId || !decoded.email || decoded.type !== 'refresh') {
      throw new Error('Invalid refresh token payload');
    }

    const tokenVersion = typeof decoded.tv === 'number' ? decoded.tv : undefined;

    return {
      userId: decoded.userId,
      email: decoded.email,
      tokenVersion,
    };
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      throw createError('Refresh token has expired', 401, 'REFRESH_TOKEN_EXPIRED');
    } else if (error instanceof jwt.JsonWebTokenError) {
      throw createError('Invalid refresh token', 401, 'INVALID_REFRESH_TOKEN');
    } else {
      throw createError(
        'Refresh token verification failed',
        401,
        'REFRESH_TOKEN_VERIFICATION_FAILED'
      );
    }
  }
};
