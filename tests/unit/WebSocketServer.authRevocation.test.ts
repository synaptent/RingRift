import { WebSocketServer } from '../../src/server/websocket/server';

type MiddlewareFn = (socket: any, next: (err?: Error) => void) => void;

let capturedMiddleware: MiddlewareFn | null = null;

const mockUserFindUnique = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => ({
    user: {
      findUnique: mockUserFindUnique,
    },
  }),
}));

jest.mock('../../src/server/middleware/auth', () => {
  const actual = jest.requireActual('../../src/server/middleware/auth');
  const verifyToken = jest.fn(() => ({
    userId: 'user-1',
    email: 'user1@example.com',
    tokenVersion: 0,
  }));
  return {
    ...actual,
    verifyToken,
  };
});

jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public use = jest.fn((fn: MiddlewareFn) => {
      capturedMiddleware = fn;
    });

    public on = jest.fn();

    public to = jest.fn(() => ({
      emit: jest.fn(),
    }));

    public sockets = {
      adapter: {
        rooms: new Map<string, Set<string>>(),
      },
      sockets: new Map<string, any>(),
    };

    constructor(..._args: any[]) {}
  }

  return {
    Server: FakeSocketIOServer,
  };
});

describe('WebSocketServer auth revocation via tokenVersion', () => {
  it('emits ACCESS_DENIED and fails connection when tokenVersion is stale', async () => {
    const httpServerStub: any = {};
    // Constructing WebSocketServer will register the auth middleware on the
    // Socket.IO server stub via io.use(...) and store it in capturedMiddleware.
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const wsServer = new WebSocketServer(httpServerStub as any);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    // DB user has tokenVersion=1, but verifyToken returns tokenVersion=0,
    // so validateUser should treat the token as revoked.
    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 1,
    });

    const socket: any = {
      id: 'socket-1',
      handshake: {
        auth: { token: 'ACCESS_TOKEN' },
        query: {},
      },
      emit: jest.fn(),
    };

    await new Promise<void>((resolve) => {
      capturedMiddleware!(socket, (_err?: Error) => {
        resolve();
      });
    });

    expect(socket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'ACCESS_DENIED',
        message: 'Authentication failed',
      })
    );
  });
});
