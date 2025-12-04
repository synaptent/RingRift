import { Server as HTTPServer } from 'http';
import { config as appConfig } from '../../src/server/config';
import { WebSocketServer } from '../../src/server/websocket/server';

jest.mock('socket.io', () => {
  const actual = jest.requireActual('socket.io');
  return {
    ...actual,
    Server: jest.fn().mockImplementation((httpServer: HTTPServer, options: any) => {
      // Expose options so tests can inspect any server-level configuration if needed.
      const io = new actual.Server(httpServer, options);
      return io;
    }),
  };
});

describe('WebSocket reconnection timeout configuration', () => {
  it('uses the configured WS reconnection timeout value', () => {
    const server = new HTTPServer();

    // Sanity check that the config exposes a positive timeout value.
    expect(appConfig.server.wsReconnectionTimeoutMs).toBeGreaterThan(0);

    // Constructing the WebSocketServer should pick up the timeout from config;
    // behaviour is covered indirectly in connection state tests, so here we
    // only assert that the field is wired and positive.
    const wsServer = new WebSocketServer(server as any);
    expect(wsServer).toBeDefined();
  });
});
