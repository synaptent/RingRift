/**
 * Integration tests for Data Lifecycle Validation
 *
 * Tests the complete data lifecycle including:
 * - Data export (GET /api/users/me/export)
 * - Data retention service cleanup
 * - Game history with deleted players
 *
 * Reference: S-05.E.5 Data Lifecycle Validation Tests
 */

import request from 'supertest';
import express, { Express } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

// Mock the database connection
jest.mock('../../src/server/database/connection');

// Mock the logger to reduce noise
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
  httpLogger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
  redactEmail: (email: string) => email.replace(/(.{2}).*(@.*)/, '$1***$2'),
  getRequestContext: jest.fn(() => ({})),
  withRequestContext: jest.fn((_req: any, context: any) => context),
}));

// Mock the cache service (redis)
jest.mock('../../src/server/cache/redis', () => ({
  getCacheService: jest.fn(() => null),
  CacheKeys: {
    authLoginLockout: (email: string) => `auth:lockout:${email}`,
    authLoginFailures: (email: string) => `auth:failures:${email}`,
  },
}));

// Mock email service
jest.mock('../../src/server/utils/email', () => ({
  sendVerificationEmail: jest.fn().mockResolvedValue(undefined),
  sendPasswordResetEmail: jest.fn().mockResolvedValue(undefined),
}));

// Mock config
jest.mock('../../src/server/config', () => ({
  config: {
    isProduction: false,
    isDevelopment: true,
    auth: {
      jwtSecret: 'test-jwt-secret-12345678901234567890',
      jwtRefreshSecret: 'test-refresh-secret-12345678901234567890',
      accessTokenExpiresIn: '15m',
      refreshTokenExpiresIn: '7d',
      loginLockoutEnabled: false,
      maxFailedLoginAttempts: 5,
      failedLoginWindowSeconds: 900,
      lockoutDurationSeconds: 1800,
    },
  },
}));

import { getDatabaseClient } from '../../src/server/database/connection';
import userRouter from '../../src/server/routes/user';
import authRouter from '../../src/server/routes/auth';
import { authenticate } from '../../src/server/middleware/auth';
import { errorHandler } from '../../src/server/middleware/errorHandler';
import {
  DataRetentionService,
  DEFAULT_RETENTION,
  RetentionConfig,
} from '../../src/server/services/DataRetentionService';
import {
  getDisplayUsername,
  isDeletedUserUsername,
  DELETED_USER_DISPLAY_NAME,
  DELETED_USER_PREFIX,
} from '../../src/server/routes/user';

// =============================================================================
// In-Memory Database Mock
// =============================================================================

interface MockPrismaClient {
  user: {
    findUnique: jest.Mock;
    findFirst: jest.Mock;
    findMany: jest.Mock;
    create: jest.Mock;
    update: jest.Mock;
    updateMany: jest.Mock;
    deleteMany: jest.Mock;
    count: jest.Mock;
  };
  refreshToken: {
    create: jest.Mock;
    findFirst: jest.Mock;
    findMany: jest.Mock;
    deleteMany: jest.Mock;
    updateMany: jest.Mock;
  };
  game: {
    findUnique: jest.Mock;
    findMany: jest.Mock;
    create: jest.Mock;
    count: jest.Mock;
  };
  move: {
    findMany: jest.Mock;
    create: jest.Mock;
  };
  $transaction: jest.Mock;
}

interface MockUser {
  id: string;
  email: string;
  username: string;
  passwordHash: string;
  role: string;
  rating: number;
  gamesPlayed: number;
  gamesWon: number;
  emailVerified: boolean;
  isActive: boolean;
  createdAt: Date;
  updatedAt: Date;
  lastLoginAt: Date | null;
  tokenVersion: number;
  deletedAt: Date | null;
  verificationToken: string | null;
  verificationTokenExpires: Date | null;
  passwordResetToken: string | null;
  passwordResetExpires: Date | null;
}

interface MockRefreshToken {
  id: string;
  token: string;
  userId: string;
  familyId: string | null;
  expiresAt: Date;
  createdAt: Date;
  revokedAt: Date | null;
}

interface MockGame {
  id: string;
  boardType: string;
  status: string;
  maxPlayers: number;
  isRated: boolean;
  player1Id: string | null;
  player2Id: string | null;
  player3Id: string | null;
  player4Id: string | null;
  winnerId: string | null;
  createdAt: Date;
  startedAt: Date | null;
  endedAt: Date | null;
  gameState: string;
}

interface MockMove {
  id: string;
  gameId: string;
  playerId: string;
  moveNumber: number;
  moveType: string;
  position: any;
  timestamp: Date;
}

// In-memory stores
let users: Map<string, MockUser>;
let refreshTokens: Map<string, MockRefreshToken>;
let games: Map<string, MockGame>;
let moves: Map<string, MockMove>;
let userIdCounter: number;
let gameIdCounter: number;
let moveIdCounter: number;

function resetDatabase() {
  users = new Map();
  refreshTokens = new Map();
  games = new Map();
  moves = new Map();
  userIdCounter = 0;
  gameIdCounter = 0;
  moveIdCounter = 0;
}

function createMockPrisma(): MockPrismaClient {
  return {
    user: {
      findUnique: jest.fn(
        async ({ where, select }: { where: { id?: string; email?: string }; select?: any }) => {
          let user: MockUser | undefined;
          if (where.id) {
            user = users.get(where.id);
          }
          if (where.email) {
            for (const u of users.values()) {
              if (u.email === where.email) {
                user = u;
                break;
              }
            }
          }
          if (!user) return null;
          if (select) {
            const result: any = {};
            for (const key of Object.keys(select)) {
              if (select[key] && key in user) {
                result[key] = (user as any)[key];
              }
            }
            return result;
          }
          return user;
        }
      ),
      findFirst: jest.fn(async ({ where }: { where: any }) => {
        for (const user of users.values()) {
          let matches = true;
          if (where.email && user.email !== where.email) matches = false;
          if (where.username && user.username !== where.username) matches = false;
          if (where.deletedAt === null && user.deletedAt !== null) matches = false;
          if (where.NOT?.id && user.id === where.NOT.id) matches = false;
          if (where.OR) {
            matches = where.OR.some((cond: any) => {
              if (cond.email && user.email === cond.email) return true;
              if (cond.username && user.username === cond.username) return true;
              return false;
            });
          }
          if (matches) return user;
        }
        return null;
      }),
      findMany: jest.fn(async ({ where }: { where?: any } = {}) => {
        const result: MockUser[] = [];
        for (const user of users.values()) {
          let matches = true;
          if (where) {
            if (where.isActive !== undefined && user.isActive !== where.isActive) matches = false;
            if (where.gamesPlayed?.gt !== undefined && user.gamesPlayed <= where.gamesPlayed.gt)
              matches = false;
          }
          if (matches) result.push(user);
        }
        return result;
      }),
      create: jest.fn(async ({ data }: { data: any }) => {
        userIdCounter++;
        const id = `user-${userIdCounter}`;
        const user: MockUser = {
          id,
          email: data.email,
          username: data.username,
          passwordHash: data.passwordHash,
          role: data.role || 'USER',
          rating: data.rating || 1200,
          gamesPlayed: data.gamesPlayed || 0,
          gamesWon: data.gamesWon || 0,
          emailVerified: data.emailVerified || false,
          isActive: data.isActive !== undefined ? data.isActive : true,
          createdAt: data.createdAt || new Date(),
          updatedAt: new Date(),
          lastLoginAt: null,
          tokenVersion: data.tokenVersion || 0,
          deletedAt: data.deletedAt || null,
          verificationToken: data.verificationToken || null,
          verificationTokenExpires: data.verificationTokenExpires || null,
          passwordResetToken: data.passwordResetToken || null,
          passwordResetExpires: data.passwordResetExpires || null,
        };
        users.set(id, user);
        return user;
      }),
      update: jest.fn(async ({ where, data }: { where: { id: string }; data: any }) => {
        const user = users.get(where.id);
        if (!user) throw new Error('User not found');

        if (data.isActive !== undefined) user.isActive = data.isActive;
        if (data.deletedAt !== undefined) user.deletedAt = data.deletedAt;
        if (data.email !== undefined) user.email = data.email;
        if (data.username !== undefined) user.username = data.username;
        if (data.lastLoginAt !== undefined) user.lastLoginAt = data.lastLoginAt;
        if (data.passwordHash !== undefined) user.passwordHash = data.passwordHash;
        if (data.verificationToken !== undefined) user.verificationToken = data.verificationToken;
        if (data.verificationTokenExpires !== undefined)
          user.verificationTokenExpires = data.verificationTokenExpires;
        if (data.passwordResetToken !== undefined)
          user.passwordResetToken = data.passwordResetToken;
        if (data.passwordResetExpires !== undefined)
          user.passwordResetExpires = data.passwordResetExpires;
        if (data.tokenVersion !== undefined) {
          if (typeof data.tokenVersion === 'object' && data.tokenVersion.increment) {
            user.tokenVersion += data.tokenVersion.increment;
          } else {
            user.tokenVersion = data.tokenVersion;
          }
        }
        user.updatedAt = new Date();

        return user;
      }),
      updateMany: jest.fn(async ({ where, data }: { where: any; data: any }) => {
        let count = 0;
        for (const user of users.values()) {
          let matches = true;
          if (where.emailVerified !== undefined && user.emailVerified !== where.emailVerified)
            matches = false;
          if (where.createdAt?.lt && user.createdAt >= where.createdAt.lt) matches = false;
          if (where.deletedAt === null && user.deletedAt !== null) matches = false;
          if (matches) {
            if (data.deletedAt !== undefined) user.deletedAt = data.deletedAt;
            if (data.isActive !== undefined) user.isActive = data.isActive;
            count++;
          }
        }
        return { count };
      }),
      deleteMany: jest.fn(async ({ where }: { where: any }) => {
        const toDelete: string[] = [];
        for (const [id, user] of users.entries()) {
          let matches = true;
          if (where.deletedAt?.not === null && user.deletedAt === null) matches = false;
          if (where.deletedAt?.lt && (!user.deletedAt || user.deletedAt >= where.deletedAt.lt))
            matches = false;
          if (matches) toDelete.push(id);
        }
        toDelete.forEach((id) => users.delete(id));
        return { count: toDelete.length };
      }),
      count: jest.fn(async ({ where }: { where?: any } = {}) => {
        let count = 0;
        for (const user of users.values()) {
          let matches = true;
          if (where) {
            if (where.isActive !== undefined && user.isActive !== where.isActive) matches = false;
            if (where.gamesPlayed?.gt !== undefined && user.gamesPlayed <= where.gamesPlayed.gt)
              matches = false;
          }
          if (matches) count++;
        }
        return count;
      }),
    },
    refreshToken: {
      create: jest.fn(async ({ data }: { data: any }) => {
        const id = `rt-${Date.now()}-${Math.random()}`;
        const token: MockRefreshToken = {
          id,
          token: data.token,
          userId: data.userId,
          familyId: data.familyId || null,
          expiresAt: data.expiresAt,
          createdAt: data.createdAt || new Date(),
          revokedAt: data.revokedAt || null,
        };
        refreshTokens.set(id, token);
        return token;
      }),
      findFirst: jest.fn(async ({ where }: { where: any }) => {
        for (const token of refreshTokens.values()) {
          let matches = true;
          if (where.token && token.token !== where.token) matches = false;
          if (where.userId && token.userId !== where.userId) matches = false;
          if (matches) {
            const user = users.get(token.userId);
            return { ...token, user };
          }
        }
        return null;
      }),
      findMany: jest.fn(async ({ where }: { where?: any } = {}) => {
        const result: MockRefreshToken[] = [];
        for (const token of refreshTokens.values()) {
          let matches = true;
          if (where?.userId && token.userId !== where.userId) matches = false;
          if (matches) result.push(token);
        }
        return result;
      }),
      deleteMany: jest.fn(async ({ where }: { where: any }) => {
        const toDelete: string[] = [];
        for (const [id, token] of refreshTokens.entries()) {
          let matches = true;
          if (where.userId && token.userId !== where.userId) matches = false;
          if (where.OR) {
            matches = where.OR.some((cond: any) => {
              if (cond.expiresAt?.lt && token.expiresAt < cond.expiresAt.lt) return true;
              if (
                cond.revokedAt?.not === null &&
                cond.revokedAt?.lt &&
                token.revokedAt &&
                token.revokedAt < cond.revokedAt.lt
              )
                return true;
              return false;
            });
          }
          if (where.expiresAt?.lt && token.expiresAt >= where.expiresAt.lt) matches = false;
          if (matches) toDelete.push(id);
        }
        toDelete.forEach((id) => refreshTokens.delete(id));
        return { count: toDelete.length };
      }),
      updateMany: jest.fn(async ({ where, data }: { where: any; data: any }) => {
        let count = 0;
        for (const token of refreshTokens.values()) {
          let matches = true;
          if (where.token && token.token !== where.token) matches = false;
          if (where.userId && token.userId !== where.userId) matches = false;
          if (where.familyId && token.familyId !== where.familyId) matches = false;
          if (matches) {
            if (data.revokedAt !== undefined) token.revokedAt = data.revokedAt;
            count++;
          }
        }
        return { count };
      }),
    },
    game: {
      findUnique: jest.fn(async ({ where, include }: { where: { id: string }; include?: any }) => {
        const game = games.get(where.id);
        if (!game) return null;

        const result: any = { ...game };
        if (include?.player1 && game.player1Id) {
          const player = users.get(game.player1Id);
          result.player1 = player ? { id: player.id, username: player.username } : null;
        }
        if (include?.player2 && game.player2Id) {
          const player = users.get(game.player2Id);
          result.player2 = player ? { id: player.id, username: player.username } : null;
        }
        if (include?.player3 && game.player3Id) {
          const player = users.get(game.player3Id);
          result.player3 = player ? { id: player.id, username: player.username } : null;
        }
        if (include?.player4 && game.player4Id) {
          const player = users.get(game.player4Id);
          result.player4 = player ? { id: player.id, username: player.username } : null;
        }
        if (include?.moves) {
          result.moves = [];
          for (const move of moves.values()) {
            if (move.gameId === game.id) result.moves.push(move);
          }
          result.moves.sort((a: MockMove, b: MockMove) => a.moveNumber - b.moveNumber);
        }
        return result;
      }),
      findMany: jest.fn(async ({ where, select, include, orderBy, take, skip }: any = {}) => {
        const result: any[] = [];
        for (const game of games.values()) {
          let matches = true;
          if (where?.OR) {
            matches = where.OR.some((cond: any) => {
              if (cond.player1Id && game.player1Id === cond.player1Id) return true;
              if (cond.player2Id && game.player2Id === cond.player2Id) return true;
              if (cond.player3Id && game.player3Id === cond.player3Id) return true;
              if (cond.player4Id && game.player4Id === cond.player4Id) return true;
              return false;
            });
          }
          if (where?.status && game.status !== where.status) matches = false;
          if (matches) {
            const gameResult: any = { ...game };

            // Handle select or include for player relations
            const playerSelect = select?.player1?.select || include?.player1?.select;
            if (playerSelect || include?.player1 || select?.player1) {
              if (game.player1Id) {
                const player = users.get(game.player1Id);
                gameResult.player1 = player ? { id: player.id, username: player.username } : null;
              } else {
                gameResult.player1 = null;
              }
            }
            if (playerSelect || include?.player2 || select?.player2) {
              if (game.player2Id) {
                const player = users.get(game.player2Id);
                gameResult.player2 = player ? { id: player.id, username: player.username } : null;
              } else {
                gameResult.player2 = null;
              }
            }
            if (playerSelect || include?.player3 || select?.player3) {
              if (game.player3Id) {
                const player = users.get(game.player3Id);
                gameResult.player3 = player ? { id: player.id, username: player.username } : null;
              } else {
                gameResult.player3 = null;
              }
            }
            if (playerSelect || include?.player4 || select?.player4) {
              if (game.player4Id) {
                const player = users.get(game.player4Id);
                gameResult.player4 = player ? { id: player.id, username: player.username } : null;
              } else {
                gameResult.player4 = null;
              }
            }

            // Handle moves
            if (include?.moves || select?.moves) {
              gameResult.moves = [];
              for (const move of moves.values()) {
                if (move.gameId === game.id) {
                  gameResult.moves.push({
                    moveNumber: move.moveNumber,
                    moveType: move.moveType,
                    timestamp: move.timestamp,
                    playerId: move.playerId,
                    position: move.position,
                  });
                }
              }
              gameResult.moves.sort((a: MockMove, b: MockMove) => a.moveNumber - b.moveNumber);
            }
            if (include?._count?.select?.moves) {
              let moveCount = 0;
              for (const move of moves.values()) {
                if (move.gameId === game.id) moveCount++;
              }
              gameResult._count = { moves: moveCount };
            }
            result.push(gameResult);
          }
        }
        // Sort by createdAt desc
        result.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
        if (skip) result.splice(0, skip);
        if (take) result.splice(take);
        return result;
      }),
      create: jest.fn(async ({ data }: { data: any }) => {
        gameIdCounter++;
        const id = `game-${gameIdCounter}`;
        const game: MockGame = {
          id,
          boardType: data.boardType || 'square8',
          status: data.status || 'waiting',
          maxPlayers: data.maxPlayers || 2,
          isRated: data.isRated || false,
          player1Id: data.player1Id || null,
          player2Id: data.player2Id || null,
          player3Id: data.player3Id || null,
          player4Id: data.player4Id || null,
          winnerId: data.winnerId || null,
          createdAt: data.createdAt || new Date(),
          startedAt: data.startedAt || null,
          endedAt: data.endedAt || null,
          gameState: data.gameState || '{}',
        };
        games.set(id, game);
        return game;
      }),
      count: jest.fn(async ({ where }: { where?: any } = {}) => {
        let count = 0;
        for (const game of games.values()) {
          let matches = true;
          if (where?.OR) {
            matches = where.OR.some((cond: any) => {
              if (cond.player1Id && game.player1Id === cond.player1Id) return true;
              if (cond.player2Id && game.player2Id === cond.player2Id) return true;
              if (cond.player3Id && game.player3Id === cond.player3Id) return true;
              if (cond.player4Id && game.player4Id === cond.player4Id) return true;
              return false;
            });
          }
          if (matches) count++;
        }
        return count;
      }),
    },
    move: {
      findMany: jest.fn(async ({ where, orderBy }: { where?: any; orderBy?: any } = {}) => {
        const result: MockMove[] = [];
        for (const move of moves.values()) {
          let matches = true;
          if (where?.gameId && move.gameId !== where.gameId) matches = false;
          if (matches) result.push(move);
        }
        result.sort((a, b) => a.moveNumber - b.moveNumber);
        return result;
      }),
      create: jest.fn(async ({ data }: { data: any }) => {
        moveIdCounter++;
        const id = `move-${moveIdCounter}`;
        const move: MockMove = {
          id,
          gameId: data.gameId,
          playerId: data.playerId,
          moveNumber: data.moveNumber,
          moveType: data.moveType || 'place',
          position: data.position || { x: 0, y: 0 },
          timestamp: data.timestamp || new Date(),
        };
        moves.set(id, move);
        return move;
      }),
    },
    $transaction: jest.fn(async (callback: (tx: MockPrismaClient) => Promise<unknown>) => {
      const currentPrisma = getDatabaseClient() as unknown as MockPrismaClient;
      return callback(currentPrisma);
    }),
  };
}

// =============================================================================
// Test Setup
// =============================================================================

let app: Express;
let mockPrisma: MockPrismaClient;

function setupApp(): Express {
  app = express();
  app.use(express.json());
  app.use('/api/auth', authRouter);
  app.use('/api/users', authenticate, userRouter);
  app.use(errorHandler);
  return app;
}

// =============================================================================
// Test Helpers
// =============================================================================

async function createTestUser(
  email: string = 'test@example.com',
  username: string = 'testuser',
  options: {
    password?: string;
    emailVerified?: boolean;
    createdAt?: Date;
    gamesPlayed?: number;
    gamesWon?: number;
    rating?: number;
  } = {}
): Promise<{ user: MockUser; plainPassword: string }> {
  const password = options.password || 'Password123';
  const passwordHash = await bcrypt.hash(password, 10);
  const user = await mockPrisma.user.create({
    data: {
      email,
      username,
      passwordHash,
      role: 'USER',
      isActive: true,
      emailVerified: options.emailVerified !== undefined ? options.emailVerified : true,
      createdAt: options.createdAt || new Date(),
      gamesPlayed: options.gamesPlayed || 0,
      gamesWon: options.gamesWon || 0,
      rating: options.rating || 1200,
    },
  });
  return { user: user as MockUser, plainPassword: password };
}

function generateAccessToken(user: MockUser): string {
  return jwt.sign(
    {
      userId: user.id,
      email: user.email,
      tv: user.tokenVersion,
    },
    'test-jwt-secret-12345678901234567890',
    {
      expiresIn: '15m',
      issuer: 'ringrift',
      audience: 'ringrift-users',
    }
  );
}

async function createTestGame(
  player1Id: string,
  player2Id: string,
  options: {
    status?: string;
    winnerId?: string | null;
    isRated?: boolean;
    boardType?: string;
  } = {}
): Promise<MockGame> {
  const game = await mockPrisma.game.create({
    data: {
      player1Id,
      player2Id,
      status: options.status || 'completed',
      winnerId: options.winnerId !== undefined ? options.winnerId : player1Id,
      isRated: options.isRated || false,
      boardType: options.boardType || 'square8',
      startedAt: new Date(),
      endedAt: options.status === 'completed' ? new Date() : null,
    },
  });
  return game;
}

async function createTestMove(
  gameId: string,
  playerId: string,
  moveNumber: number
): Promise<MockMove> {
  const move = await mockPrisma.move.create({
    data: {
      gameId,
      playerId,
      moveNumber,
      moveType: 'place',
      position: { x: moveNumber, y: 0 },
      timestamp: new Date(),
    },
  });
  return move;
}

function daysAgo(days: number): Date {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date;
}

// =============================================================================
// Tests
// =============================================================================

describe('Data Lifecycle Validation Tests', () => {
  beforeEach(() => {
    resetDatabase();
    mockPrisma = createMockPrisma();
    (getDatabaseClient as jest.Mock).mockReturnValue(mockPrisma);
    setupApp();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  // ===========================================================================
  // Data Export Tests
  // ===========================================================================

  describe('GET /api/users/me/export', () => {
    test('should export user profile data', async () => {
      // Create user with specific data
      const { user } = await createTestUser('export@example.com', 'exportuser', {
        gamesPlayed: 10,
        gamesWon: 6,
        rating: 1350,
      });
      const accessToken = generateAccessToken(user);

      // Request data export
      const res = await request(app)
        .get('/api/users/me/export')
        .set('Authorization', `Bearer ${accessToken}`);

      // Verify response
      expect(res.status).toBe(200);
      expect(res.headers['content-type']).toMatch(/application\/json/);
      expect(res.headers['content-disposition']).toContain('attachment');
      expect(res.headers['content-disposition']).toContain('ringrift-data-export');

      // Verify export structure
      expect(res.body).toHaveProperty('exportedAt');
      expect(res.body).toHaveProperty('exportFormat', '1.0');
      expect(res.body).toHaveProperty('profile');
      expect(res.body).toHaveProperty('statistics');
      expect(res.body).toHaveProperty('games');

      // Verify profile data
      expect(res.body.profile.id).toBe(user.id);
      expect(res.body.profile.username).toBe('exportuser');
      expect(res.body.profile.email).toBe('export@example.com');

      // Verify statistics
      expect(res.body.statistics.rating).toBe(1350);
      expect(res.body.statistics.gamesPlayed).toBe(10);
      expect(res.body.statistics.wins).toBe(6);
      expect(res.body.statistics.losses).toBe(4);
      expect(res.body.statistics.winRate).toBe(60);
    });

    test('should include game history in export', async () => {
      // Create user and opponent
      const { user } = await createTestUser('player1@example.com', 'player1');
      const { user: opponent } = await createTestUser('player2@example.com', 'player2');

      // Create a completed game
      const game = await createTestGame(user.id, opponent.id, {
        status: 'completed',
        winnerId: user.id,
      });

      // Add some moves
      await createTestMove(game.id, user.id, 1);
      await createTestMove(game.id, opponent.id, 2);
      await createTestMove(game.id, user.id, 3);

      const accessToken = generateAccessToken(user);

      // Request data export
      const res = await request(app)
        .get('/api/users/me/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(res.status).toBe(200);
      expect(res.body.games).toHaveLength(1);

      const exportedGame = res.body.games[0];
      expect(exportedGame.id).toBe(game.id);
      expect(exportedGame.result).toBe('win');
      expect(exportedGame.status).toBe('completed');
      expect(exportedGame.moves).toHaveLength(3);

      // Verify move data includes isUserMove flag
      const userMoves = exportedGame.moves.filter((m: any) => m.isUserMove);
      expect(userMoves).toHaveLength(2);
    });

    test('should exclude sensitive data from export', async () => {
      const { user } = await createTestUser('sensitive@example.com', 'sensitiveuser');
      const accessToken = generateAccessToken(user);

      const res = await request(app)
        .get('/api/users/me/export')
        .set('Authorization', `Bearer ${accessToken}`);

      expect(res.status).toBe(200);

      // Verify sensitive fields are NOT included
      expect(res.body.profile).not.toHaveProperty('passwordHash');
      expect(res.body.profile).not.toHaveProperty('tokenVersion');
      expect(res.body.profile).not.toHaveProperty('verificationToken');
      expect(res.body.profile).not.toHaveProperty('verificationTokenExpires');
      expect(res.body.profile).not.toHaveProperty('passwordResetToken');
      expect(res.body.profile).not.toHaveProperty('passwordResetExpires');
      expect(res.body.profile).not.toHaveProperty('deletedAt');
    });

    test('should require authentication for data export', async () => {
      // Request without auth header
      const res = await request(app).get('/api/users/me/export');

      expect(res.status).toBe(401);
      expect(res.body.success).toBe(false);
    });
  });

  // ===========================================================================
  // DataRetentionService Tests
  // ===========================================================================

  describe('DataRetentionService', () => {
    test('should hard delete users after retention period', async () => {
      // Create user, soft delete, set deletedAt to past (beyond retention period)
      const { user } = await createTestUser('old-deleted@example.com', 'olddeleted');
      const cutoffDays = DEFAULT_RETENTION.deletedUserRetentionDays + 1;
      user.deletedAt = daysAgo(cutoffDays);
      user.isActive = false;

      // Verify user exists
      expect(users.size).toBe(1);

      // Run retention
      const retentionService = new DataRetentionService(mockPrisma as any);
      const report = await retentionService.runRetentionTasks();

      // Verify user was hard deleted
      expect(report.hardDeletedUsers).toBe(1);
      expect(users.size).toBe(0);
    });

    test('should not delete users within retention period', async () => {
      // Create user, soft delete with recent deletedAt (within retention period)
      const { user } = await createTestUser('recent-deleted@example.com', 'recentdeleted');
      user.deletedAt = daysAgo(5); // Only 5 days ago, within 30 day retention
      user.isActive = false;

      expect(users.size).toBe(1);

      // Run retention
      const retentionService = new DataRetentionService(mockPrisma as any);
      const report = await retentionService.runRetentionTasks();

      // Verify user still exists
      expect(report.hardDeletedUsers).toBe(0);
      expect(users.size).toBe(1);
    });

    test('should cleanup expired refresh tokens', async () => {
      // Create user
      const { user } = await createTestUser('token-user@example.com', 'tokenuser');

      // Create expired token (beyond retention period)
      const expiredTokenDays = DEFAULT_RETENTION.expiredTokenRetentionDays + 1;
      await mockPrisma.refreshToken.create({
        data: {
          token: 'expired-token-hash',
          userId: user.id,
          expiresAt: daysAgo(expiredTokenDays),
          createdAt: daysAgo(expiredTokenDays + 7),
        },
      });

      // Create valid token
      await mockPrisma.refreshToken.create({
        data: {
          token: 'valid-token-hash',
          userId: user.id,
          expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
          createdAt: new Date(),
        },
      });

      expect(refreshTokens.size).toBe(2);

      // Run retention
      const retentionService = new DataRetentionService(mockPrisma as any);
      const report = await retentionService.runRetentionTasks();

      // Verify expired token was deleted
      expect(report.deletedTokens).toBe(1);
      expect(refreshTokens.size).toBe(1);
    });

    test('should soft delete unverified accounts past threshold', async () => {
      // Create unverified account with old createdAt
      const unverifiedDays = DEFAULT_RETENTION.unverifiedAccountRetentionDays + 1;
      const { user } = await createTestUser('unverified@example.com', 'unverifieduser', {
        emailVerified: false,
        createdAt: daysAgo(unverifiedDays),
      });

      expect(user.emailVerified).toBe(false);
      expect(user.deletedAt).toBeNull();

      // Run retention
      const retentionService = new DataRetentionService(mockPrisma as any);
      const report = await retentionService.runRetentionTasks();

      // Verify account was soft-deleted
      expect(report.deletedUnverified).toBe(1);
      const updatedUser = users.get(user.id);
      expect(updatedUser?.deletedAt).not.toBeNull();
      expect(updatedUser?.isActive).toBe(false);
    });

    test('should not soft delete recently created unverified accounts', async () => {
      // Create unverified account with recent createdAt
      const { user } = await createTestUser('new-unverified@example.com', 'newunverified', {
        emailVerified: false,
        createdAt: daysAgo(2), // Only 2 days old
      });

      expect(user.emailVerified).toBe(false);
      expect(user.deletedAt).toBeNull();

      // Run retention
      const retentionService = new DataRetentionService(mockPrisma as any);
      const report = await retentionService.runRetentionTasks();

      // Verify account was NOT soft-deleted
      expect(report.deletedUnverified).toBe(0);
      const updatedUser = users.get(user.id);
      expect(updatedUser?.deletedAt).toBeNull();
      expect(updatedUser?.isActive).toBe(true);
    });

    test('should respect custom retention config', async () => {
      // Create custom config with short retention period
      const customConfig: Partial<RetentionConfig> = {
        deletedUserRetentionDays: 3, // Only 3 days instead of default 30
      };

      // Create user, soft delete 5 days ago (beyond custom 3 day retention)
      const { user } = await createTestUser('custom-config@example.com', 'customconfig');
      user.deletedAt = daysAgo(5);
      user.isActive = false;

      // Run retention with custom config
      const retentionService = new DataRetentionService(mockPrisma as any, customConfig);
      const config = retentionService.getConfig();
      expect(config.deletedUserRetentionDays).toBe(3);

      const report = await retentionService.runRetentionTasks();

      // Verify user was hard deleted with custom retention period
      expect(report.hardDeletedUsers).toBe(1);
      expect(users.size).toBe(0);
    });

    test('should return complete retention report', async () => {
      // Setup various scenarios
      const { user: oldDeletedUser } = await createTestUser('old@example.com', 'old');
      oldDeletedUser.deletedAt = daysAgo(35);
      oldDeletedUser.isActive = false;

      const { user: tokenUser } = await createTestUser('tok@example.com', 'tok');
      await mockPrisma.refreshToken.create({
        data: {
          token: 'old-token',
          userId: tokenUser.id,
          expiresAt: daysAgo(10),
          createdAt: daysAgo(17),
        },
      });

      const { user: _unverified } = await createTestUser('unv@example.com', 'unv', {
        emailVerified: false,
        createdAt: daysAgo(10),
      });

      // Run retention
      const retentionService = new DataRetentionService(mockPrisma as any);
      const report = await retentionService.runRetentionTasks();

      // Verify report structure
      expect(report).toHaveProperty('hardDeletedUsers');
      expect(report).toHaveProperty('deletedTokens');
      expect(report).toHaveProperty('deletedUnverified');
      expect(report).toHaveProperty('executedAt');
      expect(report).toHaveProperty('durationMs');
      expect(report.executedAt).toBeInstanceOf(Date);
      expect(typeof report.durationMs).toBe('number');
    });
  });

  // ===========================================================================
  // Game History with Deleted Players Tests
  // ===========================================================================

  describe('Game history with deleted players', () => {
    test('should show "Deleted Player" for anonymized users', async () => {
      // Create two users
      const { user: player1 } = await createTestUser('p1@example.com', 'player1');
      const { user: player2 } = await createTestUser('p2@example.com', 'player2');

      // Create a game between them
      const game = await createTestGame(player1.id, player2.id, {
        status: 'completed',
        winnerId: player1.id,
      });

      // Delete player2 (anonymize)
      const accessToken2 = generateAccessToken(player2);
      await request(app).delete('/api/users/me').set('Authorization', `Bearer ${accessToken2}`);

      // Verify player2 is anonymized
      const deletedPlayer = users.get(player2.id);
      expect(deletedPlayer?.username).toMatch(new RegExp(`^${DELETED_USER_PREFIX}`));

      // Test helper functions
      expect(isDeletedUserUsername(deletedPlayer!.username)).toBe(true);
      expect(getDisplayUsername(deletedPlayer!.username)).toBe(DELETED_USER_DISPLAY_NAME);

      // Player1 should see their game history with "Deleted Player"
      const accessToken1 = generateAccessToken(player1);
      const res = await request(app)
        .get('/api/users/me/export')
        .set('Authorization', `Bearer ${accessToken1}`);

      expect(res.status).toBe(200);
      expect(res.body.games).toHaveLength(1);
      expect(res.body.games[0].opponent).toBe(DELETED_USER_DISPLAY_NAME);
    });

    test('opponent should still see game history after player deletion', async () => {
      // Create two users
      const { user: activePlayer } = await createTestUser('active@example.com', 'activeplayer');
      const { user: deletingPlayer } = await createTestUser(
        'deleting@example.com',
        'deletingplayer'
      );

      // Create a game
      const game = await createTestGame(activePlayer.id, deletingPlayer.id, {
        status: 'completed',
        winnerId: activePlayer.id,
      });

      // Add moves
      await createTestMove(game.id, activePlayer.id, 1);
      await createTestMove(game.id, deletingPlayer.id, 2);

      // Delete one player
      const deleteToken = generateAccessToken(deletingPlayer);
      await request(app).delete('/api/users/me').set('Authorization', `Bearer ${deleteToken}`);

      // Active player queries their game history
      const activeToken = generateAccessToken(activePlayer);
      const res = await request(app)
        .get('/api/users/me/export')
        .set('Authorization', `Bearer ${activeToken}`);

      // Game should still be visible
      expect(res.status).toBe(200);
      expect(res.body.games).toHaveLength(1);

      // Game details preserved
      expect(res.body.games[0].id).toBe(game.id);
      expect(res.body.games[0].result).toBe('win');
      expect(res.body.games[0].moves).toHaveLength(2);

      // Opponent shown as "Deleted Player"
      expect(res.body.games[0].opponent).toBe(DELETED_USER_DISPLAY_NAME);
    });

    test('getDisplayUsername helper handles various inputs correctly', () => {
      // Normal username
      expect(getDisplayUsername('NormalUser')).toBe('NormalUser');

      // Deleted user username
      expect(getDisplayUsername(`${DELETED_USER_PREFIX}abc12345`)).toBe(DELETED_USER_DISPLAY_NAME);

      // Null/undefined
      expect(getDisplayUsername(null)).toBe(DELETED_USER_DISPLAY_NAME);
      expect(getDisplayUsername(undefined)).toBe(DELETED_USER_DISPLAY_NAME);

      // Empty string
      expect(getDisplayUsername('')).toBe(DELETED_USER_DISPLAY_NAME);
    });

    test('isDeletedUserUsername helper identifies deleted users', () => {
      expect(isDeletedUserUsername('NormalUser')).toBe(false);
      expect(isDeletedUserUsername(`${DELETED_USER_PREFIX}abc12345`)).toBe(true);
      expect(isDeletedUserUsername('DeletedPlayerFake')).toBe(false);
      expect(isDeletedUserUsername(`${DELETED_USER_PREFIX}`)).toBe(true);
    });
  });
});
