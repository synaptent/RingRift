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
  const validateUser = jest.fn().mockResolvedValue({
    id: 'user-1',
    username: 'Tester',
    email: 'user1@example.com',
    isActive: true,
    tokenVersion: 0,
  });
  return {
    ...actual,
    verifyToken,
    validateUser,
  };
});

// Create a mock socket that simulates Socket.IO socket behavior
const createMockSocket = (id: string, userId: string) => ({
  id,
  userId,
  emit: jest.fn(),
  disconnect: jest.fn(),
  handshake: {
    auth: { token: 'ACCESS_TOKEN' },
    query: {},
  },
  join: jest.fn(),
  leave: jest.fn(),
  rooms: new Set<string>(),
  on: jest.fn(),
  to: jest.fn(() => ({ emit: jest.fn() })),
});

// Track sockets created during connection events
const mockSockets = new Map<string, any>();
let connectionHandler: ((socket: any) => void) | null = null;

jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public use = jest.fn((fn: MiddlewareFn) => {
      capturedMiddleware = fn;
    });

    public on = jest.fn((event: string, handler: any) => {
      if (event === 'connection') {
        connectionHandler = handler;
      }
    });

    public to = jest.fn(() => ({
      emit: jest.fn(),
    }));

    public sockets = {
      adapter: {
        rooms: new Map<string, Set<string>>(),
      },
      sockets: mockSockets,
    };

    constructor(..._args: any[]) {}
  }

  return {
    Server: FakeSocketIOServer,
  };
});

describe('WebSocketServer.terminateUserSessions', () => {
  let wsServer: WebSocketServer;

  beforeEach(() => {
    jest.clearAllMocks();
    mockSockets.clear();
    capturedMiddleware = null;
    connectionHandler = null;

    const httpServerStub: any = {};
    wsServer = new WebSocketServer(httpServerStub as any);
  });

  it('returns 0 when user has no active connections', () => {
    const terminatedCount = wsServer.terminateUserSessions('nonexistent-user', 'Test reason');

    expect(terminatedCount).toBe(0);
  });

  it('terminates active WebSocket connection for user', async () => {
    // Simulate a user connecting
    const mockSocket = createMockSocket('socket-1', 'user-1');
    mockSockets.set('socket-1', mockSocket);

    // Run the auth middleware to register the connection
    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    // Simulate the connection event handler to register the socket
    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Now terminate the user's sessions
    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');

    // Socket should have been disconnected
    expect(mockSocket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'ACCESS_DENIED',
        message: 'Account deleted',
      })
    );
    expect(mockSocket.disconnect).toHaveBeenCalledWith(true);
    expect(terminatedCount).toBe(1);
  });

  it('cleans up stale socket mapping when socket no longer exists', async () => {
    // Simulate a user connecting
    const mockSocket = createMockSocket('socket-1', 'user-1');
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Now remove the socket from the Socket.IO server (simulating it was already closed)
    mockSockets.delete('socket-1');

    // Terminate should handle this gracefully
    const terminatedCount = wsServer.terminateUserSessions('user-1', 'Account deleted');

    expect(terminatedCount).toBe(0);
    // Verify no crash occurred
  });

  it('uses default reason when not provided', async () => {
    const mockSocket = createMockSocket('socket-1', 'user-1');
    mockSockets.set('socket-1', mockSocket);

    if (!capturedMiddleware) {
      throw new Error('Auth middleware was not registered');
    }

    mockUserFindUnique.mockResolvedValue({
      id: 'user-1',
      username: 'Tester',
      email: 'user1@example.com',
      isActive: true,
      tokenVersion: 0,
    });

    await new Promise<void>((resolve) => {
      capturedMiddleware!(mockSocket, (_err?: Error) => {
        resolve();
      });
    });

    if (connectionHandler) {
      connectionHandler(mockSocket);
    }

    // Terminate without providing a reason
    wsServer.terminateUserSessions('user-1');

    expect(mockSocket.emit).toHaveBeenCalledWith(
      'error',
      expect.objectContaining({
        type: 'error',
        code: 'ACCESS_DENIED',
        message: 'Session terminated',
      })
    );
  });
});
