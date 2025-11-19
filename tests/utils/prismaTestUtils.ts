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

// Shared in-memory data store backing the Prisma stub.
export const mockDb = {
  users: [] as any[],
  refreshTokens: [] as any[],
};

// Minimal Prisma-like client stub for tests.
export const prismaStub = {
  user: {
    findFirst: jest.fn(async (args: any) => {
      if (!args || !args.where || !args.where.OR) return null;
      const { email, username } = args.where.OR.reduce(
        (acc: any, cond: any) => ({
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
    }),
    findUnique: jest.fn(async (args: any) => {
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
    create: jest.fn(async (args: any) => {
      const data = args?.data || {};
      const user = {
        id: `user-${mockDb.users.length + 1}`,
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
    update: jest.fn(async (args: any) => {
      const { id } = args.where;
      const idx = mockDb.users.findIndex((u) => u.id === id);
      if (idx === -1) return null;
      mockDb.users[idx] = { ...mockDb.users[idx], ...args.data };
      return mockDb.users[idx];
    }),
  },
  refreshToken: {
    create: jest.fn(async (args: any) => {
      const token = {
        id: `rt-${mockDb.refreshTokens.length + 1}`,
        ...args.data,
      };
      mockDb.refreshTokens.push(token);
      return token;
    }),
    findFirst: jest.fn(async (args: any) => {
      const { token, userId, expiresAt } = args.where;
      return (
        mockDb.refreshTokens.find((rt) => {
          if (token && rt.token !== token) return false;
          if (userId && rt.userId !== userId) return false;
          if (expiresAt?.gt && !(rt.expiresAt instanceof Date)) return false;
          if (expiresAt?.gt && rt.expiresAt <= expiresAt.gt) return false;
          return true;
        }) || null
      );
    }),
    delete: jest.fn(async (args: any) => {
      const { id } = args.where;
      const idx = mockDb.refreshTokens.findIndex((rt) => rt.id === id);
      if (idx === -1) return null;
      const [deleted] = mockDb.refreshTokens.splice(idx, 1);
      return deleted;
    }),
    deleteMany: jest.fn(async (args: any) => {
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
  },
  $transaction: jest.fn(async (ops: any[]) => {
    for (const op of ops) {
      // eslint-disable-next-line no-await-in-loop
      await op;
    }
    return undefined;
  }),
} as any;

// Helper to reset in-memory state between tests. Does NOT clear Jest mock
// call histories; use Jest's clearMocks/restoreMocks configuration or
// per-suite hooks for that.
export function resetPrismaMockDb(): void {
  mockDb.users.length = 0;
  mockDb.refreshTokens.length = 0;
}
