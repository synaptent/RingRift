import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { getDatabaseClient } from '../database/connection';
import { createError } from './errorHandler';
import { logger } from '../utils/logger';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    username: string;
    role: string;
  };
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
    const user = await validateUser(decoded.userId);
    
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
      const user = await validateUser(decoded.userId);
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

const verifyToken = (token: string): { userId: string; email: string } => {
  try {
    const secret = process.env.JWT_SECRET;
    if (!secret) {
      throw new Error('JWT_SECRET not configured');
    }

    const decoded = jwt.verify(token, secret) as any;
    
    if (!decoded.userId || !decoded.email) {
      throw new Error('Invalid token payload');
    }

    return {
      userId: decoded.userId,
      email: decoded.email
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

const validateUser = async (userId: string) => {
  const prisma = getDatabaseClient();
  if (!prisma) {
    throw createError('Database not available', 500, 'DATABASE_UNAVAILABLE');
  }

  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      username: true,
      role: true,
      isActive: true,
      emailVerified: true
    }
  });

  if (!user) {
    throw createError('User not found', 401, 'USER_NOT_FOUND');
  }

  if (!user.isActive) {
    throw createError('Account is deactivated', 401, 'ACCOUNT_DEACTIVATED');
  }

  return {
    id: user.id,
    email: user.email,
    username: user.username,
    role: user.role
  };
};

export const generateToken = (user: { id: string; email: string }): string => {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('JWT_SECRET not configured');
  }

  const payload = {
    userId: user.id,
    email: user.email
  };

  const options: any = {
    expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    issuer: 'ringrift',
    audience: 'ringrift-users'
  };

  return jwt.sign(payload, secret, options);
};

export const generateRefreshToken = (user: { id: string; email: string }): string => {
  const secret = process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('JWT_REFRESH_SECRET not configured');
  }

  const payload = {
    userId: user.id,
    email: user.email,
    type: 'refresh'
  };

  const options: any = {
    expiresIn: process.env.JWT_REFRESH_EXPIRES_IN || '30d',
    issuer: 'ringrift',
    audience: 'ringrift-users'
  };

  return jwt.sign(payload, secret, options);
};

export const verifyRefreshToken = (token: string): { userId: string; email: string } => {
  try {
    const secret = process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET;
    if (!secret) {
      throw new Error('JWT_REFRESH_SECRET not configured');
    }

    const decoded = jwt.verify(token, secret) as any;
    
    if (!decoded.userId || !decoded.email || decoded.type !== 'refresh') {
      throw new Error('Invalid refresh token payload');
    }

    return {
      userId: decoded.userId,
      email: decoded.email
    };
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      throw createError('Refresh token has expired', 401, 'REFRESH_TOKEN_EXPIRED');
    } else if (error instanceof jwt.JsonWebTokenError) {
      throw createError('Invalid refresh token', 401, 'INVALID_REFRESH_TOKEN');
    } else {
      throw createError('Refresh token verification failed', 401, 'REFRESH_TOKEN_VERIFICATION_FAILED');
    }
  }
};