// Reusable in-memory Prisma-like stub for route tests (auth, users, games, etc.).
//
// This mirrors the subset of PrismaClient used by the auth routes today:
// - prisma.user.findFirst / findUnique / create / update
// - prisma.refreshToken.create / findFirst / delete / deleteMany
// - prisma.$transaction([...])
//
// Tests can import { mockDb, prismaStub, resetPrismaMockDb } and then wire
// prismaStub into the database connection module via jest.mock, e.g.:
//
//   import { prismaStub, resetPrismaMockDb } from '../utils/prismaTestUtils';
//
//   jest.mock('../../src/server/database/connection', () => ({
//     getDatabaseClient: () => prismaStub,
//   }));
//
//   beforeEach(() => {
//     resetPrismaMockDb();
//   });
//
// This keeps the auth harness behavior identical while making it easy to
// reuse the same stub for other route tests.

// ============================================================================
// Types for mock database records
// ============================================================================

/** Mock user record type for in-memory database */
export interface MockUser {
  id: string;
  email: string;
  username: string;
  passwordHash?: string;
  role?: string;
  createdAt: Date;
  deletedAt?: Date | null;
  tokenVersion?: number;
  passwordResetToken?: string;
  passwordResetExpires?: Date;
  [key: string]: unknown; // Allow additional fields for flexibility
}

/** Mock refresh token record type for in-memory database */
export interface MockRefreshToken {
  id: string;
  token: string;
  userId: string;
  expiresAt: Date;
  revokedAt?: Date | null;
  familyId?: string | null;
  user?: Partial<MockUser>;
  [key: string]: unknown; // Allow additional fields for flexibility
}

/** Shape of the mock database */
export interface MockDatabase {
  users: MockUser[];
  refreshTokens: MockRefreshToken[];
}

// Shared in-memory data store backing the Prisma stub.
export const mockDb: MockDatabase = {
  users: [],
  refreshTokens: [],
};

// ============================================================================
// Type definitions for Prisma-like query arguments
// ============================================================================

interface UserWhereInput {
  id?: string;
  email?: string;
  username?: string;
  deletedAt?: Date | null;
  passwordResetToken?: string;
  passwordResetExpires?: { gt?: Date };
  OR?: Array<{ email?: string; username?: string }>;
  [key: string]: unknown;
}

interface UserFindArgs {
  where?: UserWhereInput;
}

interface UserCreateArgs {
  data?: Partial<MockUser>;
}

/** Prisma-style increment operation type */
interface PrismaIncrement {
  increment: number;
}

interface UserUpdateArgs {
  where: { id: string };
  data?: Omit<Partial<MockUser>, 'tokenVersion'> & {
    tokenVersion?: PrismaIncrement | number;
  };
}

interface RefreshTokenWhereInput {
  id?: string;
  token?: string;
  userId?: string;
  familyId?: string;
  expiresAt?: { gt?: Date };
}

interface RefreshTokenFindArgs {
  where?: RefreshTokenWhereInput;
  include?: {
    user?: boolean | { select?: Record<string, boolean> };
  };
}

interface RefreshTokenCreateArgs {
  data: Partial<MockRefreshToken>;
}

interface RefreshTokenDeleteArgs {
  where: { id: string };
}

interface RefreshTokenDeleteManyArgs {
  where?: RefreshTokenWhereInput;
}

interface RefreshTokenUpdateArgs {
  where: { id: string };
  data: Partial<MockRefreshToken>;
}

interface RefreshTokenUpdateManyArgs {
  where?: RefreshTokenWhereInput;
  data?: Partial<MockRefreshToken>;
}

// ============================================================================
// Minimal Prisma-like client stub for tests
// ============================================================================

const userModelStub = {
  findFirst: jest.fn(async (args: UserFindArgs) => {
    if (!args || !args.where) return null;

    // Handle OR-based queries (for checking existing users by email/username)
    if (args.where.OR) {
      const { email, username } = args.where.OR.reduce(
        (acc: { email?: string; username?: string }, cond) => ({
          email: cond.email ?? acc.email,
          username: cond.username ?? acc.username,
        }),
        { email: undefined, username: undefined }
      );
      return (
        mockDb.users.find(
          (u) => (email && u.email === email) || (username && u.username === username)
        ) || null
      );
    }

    // Handle password reset token queries
    if (args.where.passwordResetToken) {
      const token = args.where.passwordResetToken;
      const expiresGt = args.where.passwordResetExpires?.gt;

      return (
        mockDb.users.find((u) => {
          if (u.passwordResetToken !== token) return false;
          if (expiresGt && u.passwordResetExpires) {
            // Check if the token hasn't expired
            return u.passwordResetExpires > expiresGt;
          }
          return true;
        }) || null
      );
    }

    // Handle generic field-based queries, including soft-delete semantics.
    return (
      mockDb.users.find((u) => {
        for (const [key, value] of Object.entries(args.where as Record<string, unknown>)) {
          // For deletedAt: null, treat undefined and null as equivalent so that
          // tests can omit deletedAt on active users while routes still filter
          // with { deletedAt: null }.
          if (key === 'deletedAt' && value === null) {
            if (u.deletedAt === null || typeof u.deletedAt === 'undefined') continue;
            return false;
          }

          if (u[key] !== value) return false;
        }
        return true;
      }) || null
    );
  }),
  findUnique: jest.fn(async (args: UserFindArgs) => {
    if (!args || !args.where) return null;
    const { email, id } = args.where;
    if (email) {
      return mockDb.users.find((u) => u.email === email) || null;
    }
    if (id) {
      return mockDb.users.find((u) => u.id === id) || null;
    }
    return null;
  }),
  create: jest.fn(async (args: UserCreateArgs) => {
    const data = args?.data || {};
    const user: MockUser = {
      id: `user-${mockDb.users.length + 1}`,
      email: data.email || '',
      username: data.username || '',
      createdAt: new Date(),
      ...data,
    };
    mockDb.users.push(user);
    return {
      id: user.id,
      email: user.email,
      username: user.username,
      role: user.role,
      createdAt: user.createdAt,
    };
  }),
  update: jest.fn(async (args: UserUpdateArgs) => {
    const { id } = args.where;
    const idx = mockDb.users.findIndex((u) => u.id === id);
    if (idx === -1) return null;

    const current = mockDb.users[idx];
    const data = args.data || {};

    // Support Prisma-style increment syntax for tokenVersion so that
    // auth tests can observe version bumps from /logout-all.
    const { tokenVersion, ...rest } = data;
    const next: MockUser = { ...current, ...rest };

    if (tokenVersion !== undefined) {
      if (typeof tokenVersion === 'object' && tokenVersion !== null) {
        // Prisma-style increment syntax
        const prev = typeof next.tokenVersion === 'number' ? next.tokenVersion : 0;
        next.tokenVersion = prev + (tokenVersion as PrismaIncrement).increment;
      } else if (typeof tokenVersion === 'number') {
        next.tokenVersion = tokenVersion;
      }
    }

    mockDb.users[idx] = next;
    return mockDb.users[idx];
  }),
};

const refreshTokenModelStub = {
  create: jest.fn(async (args: RefreshTokenCreateArgs) => {
    const token: MockRefreshToken = {
      id: `rt-${mockDb.refreshTokens.length + 1}`,
      token: args.data.token || '',
      userId: args.data.userId || '',
      expiresAt: args.data.expiresAt || new Date(),
      revokedAt: null,
      familyId: null,
      ...args.data,
    };
    mockDb.refreshTokens.push(token);
    return token;
  }),
  findFirst: jest.fn(async (args: RefreshTokenFindArgs) => {
    const { token, userId, expiresAt } = args.where || {};
    const found =
      mockDb.refreshTokens.find((rt) => {
        if (token && rt.token !== token) return false;
        if (userId && rt.userId !== userId) return false;
        // Only check expiry if explicitly requested (for non-revoked token lookups)
        if (expiresAt?.gt && !(rt.expiresAt instanceof Date)) return false;
        if (expiresAt?.gt && rt.expiresAt <= expiresAt.gt) return false;
        return true;
      }) || null;

    // If args.include.user is present, attach the user to the result
    if (found && args.include?.user) {
      const user = mockDb.users.find((u) => u.id === found.userId);
      if (user) {
        const includeConfig = args.include.user;
        if (typeof includeConfig === 'object' && includeConfig.select) {
          found.user = Object.fromEntries(
            Object.entries(user).filter(([k]) =>
              Object.keys(includeConfig.select as Record<string, boolean>).includes(k)
            )
          );
        } else {
          found.user = user;
        }
      }
    }
    return found;
  }),
  delete: jest.fn(async (args: RefreshTokenDeleteArgs) => {
    const { id } = args.where;
    const idx = mockDb.refreshTokens.findIndex((rt) => rt.id === id);
    if (idx === -1) return null;
    const [deleted] = mockDb.refreshTokens.splice(idx, 1);
    return deleted;
  }),
  deleteMany: jest.fn(async (args: RefreshTokenDeleteManyArgs) => {
    if (!args || !args.where) {
      const count = mockDb.refreshTokens.length;
      mockDb.refreshTokens = [];
      return { count };
    }
    const { token, userId } = args.where;
    const before = mockDb.refreshTokens.length;
    mockDb.refreshTokens = mockDb.refreshTokens.filter((rt) => {
      if (token && rt.token !== token) return true;
      if (userId && rt.userId !== userId) return true;
      return false;
    });
    return { count: before - mockDb.refreshTokens.length };
  }),
  update: jest.fn(async (args: RefreshTokenUpdateArgs) => {
    const { id } = args.where;
    const idx = mockDb.refreshTokens.findIndex((rt) => rt.id === id);
    if (idx === -1) return null;
    const current = mockDb.refreshTokens[idx];
    mockDb.refreshTokens[idx] = { ...current, ...args.data };
    return mockDb.refreshTokens[idx];
  }),
  updateMany: jest.fn(async (args: RefreshTokenUpdateManyArgs) => {
    if (!args || !args.where) {
      return { count: 0 };
    }
    const { token, userId, familyId } = args.where;
    let count = 0;
    mockDb.refreshTokens = mockDb.refreshTokens.map((rt) => {
      let matches = true;
      if (token && rt.token !== token) matches = false;
      if (userId && rt.userId !== userId) matches = false;
      if (familyId && rt.familyId !== familyId) matches = false;
      if (matches) {
        count++;
        return { ...rt, ...args.data };
      }
      return rt;
    });
    return { count };
  }),
};

/** Transaction callback type for prisma.$transaction */
type TransactionCallback = (tx: {
  user: typeof userModelStub;
  refreshToken: typeof refreshTokenModelStub;
}) => Promise<unknown>;

export const prismaStub = {
  user: userModelStub,
  refreshToken: refreshTokenModelStub,
  $transaction: jest.fn(async (arg: Promise<unknown>[] | TransactionCallback) => {
    // Support array-of-promises style used in some routes (for example auth refresh).
    if (Array.isArray(arg)) {
      for (const op of arg) {
        await op;
      }
      return undefined;
    }

    // Support callback-style transactions used in other routes (for example user deletion).
    if (typeof arg === 'function') {
      const tx = {
        user: userModelStub,
        refreshToken: refreshTokenModelStub,
      };
      return arg(tx);
    }

    return undefined;
  }),
};

// Helper to reset in-memory state between tests. Does NOT clear Jest mock
// call histories; use Jest's clearMocks/restoreMocks configuration or
// per-suite hooks for that.
export function resetPrismaMockDb(): void {
  mockDb.users.length = 0;
  mockDb.refreshTokens.length = 0;
}
