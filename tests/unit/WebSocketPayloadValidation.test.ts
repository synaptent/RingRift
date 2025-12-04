import { EventEmitter } from 'events';
import { WebSocketServer } from '../../src/server/websocket/server';

// Mock the database layer so authentication / handlers never touch a real DB.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => null,
}));

// Minimal Socket.IO Server + Socket stubs for exercising WebSocketServer's
// per-event payload validation without opening a real network connection.

type ConnectionHandler = (socket: any) => void;

class FakeSocket extends EventEmitter {
  public id: string;
  public userId?: string;
  public username?: string;
  public gameId?: string;
  public rooms: Set<string> = new Set();
  public emittedEvents: Array<{ event: string; payload: any }> = [];

  private handlers: Map<string, ((data: any) => void)[]> = new Map();

  constructor(id: string) {
    super();
    this.id = id;
  }

  on(event: string, handler: (data: any) => void): this {
    const list = this.handlers.get(event) ?? [];
    list.push(handler);
    this.handlers.set(event, list);
    return this;
  }

  /**
   * Simulate an inbound client event (e.g. join_game) by invoking the
   * registered handler(s).
   */
  trigger(event: string, data: any) {
    const list = this.handlers.get(event) ?? [];
    for (const handler of list) {
      handler(data);
    }
  }

  join(room: string) {
    this.rooms.add(room);
  }

  leave(room: string) {
    this.rooms.delete(room);
  }

  emit(event: string, payload: any): boolean {
    this.emittedEvents.push({ event, payload });
    return true;
  }
}

let mockLastIoInstance: any = null;

function mockSetLastIoInstance(instance: any) {
  mockLastIoInstance = instance;
}

// Replace the real Socket.IO Server with our in-memory stub. The mock factory
// defines FakeSocketIOServer inside its own scope to satisfy Jest's
// "no out-of-scope variables" rule for jest.mock module factories.
jest.mock('socket.io', () => {
  class FakeSocketIOServer {
    public connectionHandler: ConnectionHandler | null = null;
    public use = jest.fn();
    public to = jest.fn(() => ({
      emit: jest.fn(),
    }));

    public sockets = {
      adapter: {
        rooms: new Map<string, Set<string>>(),
      },
      sockets: new Map<string, FakeSocket>(),
    };

    constructor(..._args: any[]) {
      mockSetLastIoInstance(this);
    }

    on(event: string, handler: (...args: any[]) => void) {
      if (event === 'connection') {
        this.connectionHandler = handler as ConnectionHandler;
      }
    }
  }

  return {
    Server: FakeSocketIOServer,
  };
});

describe('WebSocket payload validation with Zod', () => {
  function setupServerAndSocket() {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const io = mockLastIoInstance as {
      connectionHandler: ConnectionHandler | null;
      sockets: {
        sockets: Map<string, FakeSocket>;
      };
    } | null;
    if (!io || !io.connectionHandler) {
      throw new Error('FakeSocketIOServer was not initialised correctly');
    }

    const socket = new FakeSocket('socket-1');
    // Simulate a connected, authenticated user for log context.
    socket.userId = 'user-1';
    socket.username = 'Tester';

    // Let WebSocketServer attach its per-connection handlers.
    io.sockets.sockets.set(socket.id, socket);
    io.connectionHandler(socket);

    return { wsServer, serverAny, socket };
  }

  it('rejects join_game payloads missing gameId with a validation_error and does not call handleJoinGame', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const joinSpy = jest.spyOn(serverAny, 'handleJoinGame');

    // Missing gameId entirely
    socket.trigger('join_game', {});

    expect(joinSpy).not.toHaveBeenCalled();

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'INVALID_PAYLOAD',
      event: 'join_game',
      message: expect.any(String),
    });
  });

  it('accepts valid join_game payload and forwards to handler without emitting an error', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const joinSpy = jest.spyOn(serverAny, 'handleJoinGame').mockResolvedValue(undefined);

    socket.trigger('join_game', { gameId: 'game-1' });

    expect(joinSpy).toHaveBeenCalledTimes(1);
    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(0);
  });

  it('rejects player_move payloads with malformed move and does not call handlePlayerMove', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const moveSpy = jest.spyOn(serverAny, 'handlePlayerMove');

    // gameId is present but move is missing required "position" field
    const badPayload = {
      gameId: 'game-1',
      move: {
        moveType: 'place_ring',
        // position omitted - MoveSchema requires it
      } as any,
    };

    socket.trigger('player_move', badPayload);

    expect(moveSpy).not.toHaveBeenCalled();

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'INVALID_PAYLOAD',
      event: 'player_move',
      message: expect.any(String),
    });
  });

  it('accepts valid player_move payload and forwards to handler without emitting an error', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const moveSpy = jest.spyOn(serverAny, 'handlePlayerMove').mockResolvedValue(undefined);

    const goodPayload = {
      gameId: 'game-1',
      move: {
        moveType: 'place_ring',
        position: '{"x":0,"y":0}',
        moveNumber: 1,
      },
    };

    socket.trigger('player_move', goodPayload);

    expect(moveSpy).toHaveBeenCalledTimes(1);
    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(0);
  });

  it('rejects chat_message payloads missing text and does not call handleChatMessage', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const chatSpy = jest.spyOn(serverAny, 'handleChatMessage');

    // text is required and must be a non-empty string
    const badPayload = {
      gameId: 'game-1',
    };

    socket.trigger('chat_message', badPayload);

    expect(chatSpy).not.toHaveBeenCalled();

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'INVALID_PAYLOAD',
      event: 'chat_message',
      message: expect.any(String),
    });
  });

  it('accepts valid chat_message payload and forwards to handler without emitting an error', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const chatSpy = jest.spyOn(serverAny, 'handleChatMessage').mockResolvedValue(undefined);

    const goodPayload = {
      gameId: 'game-1',
      text: 'Hello world',
    };

    socket.trigger('chat_message', goodPayload);

    expect(chatSpy).toHaveBeenCalledTimes(1);
    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(0);
  });

  it('rejects player_choice_response payloads missing choiceId and does not reach interaction handler', () => {
    const { serverAny, socket } = setupServerAndSocket();

    // Spy on the interaction handler at the GameSessionManager layer. For this
    // test we only care that WebSocketServer does not even try to look up a
    // session when validation fails.
    const sessionManagerSpy = jest.spyOn(serverAny.sessionManager, 'getSession');

    const badPayload = {
      // choiceId missing
      playerNumber: 1,
      selectedOption: { any: 'value' },
    };

    socket.trigger('player_choice_response', badPayload);

    expect(sessionManagerSpy).not.toHaveBeenCalled();

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'INVALID_PAYLOAD',
      event: 'player_choice_response',
      message: expect.any(String),
    });
  });

  it('emits GAME_NOT_FOUND when join_game fails with a missing game', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const joinSpy = jest.spyOn(serverAny, 'handleJoinGame').mockImplementation(() => {
      throw new Error('Game not found');
    });

    socket.trigger('join_game', { gameId: 'missing-game' });

    expect(joinSpy).toHaveBeenCalledTimes(1);

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'GAME_NOT_FOUND',
      event: 'join_game',
      message: expect.any(String),
    });

    const gameStateEvents = socket.emittedEvents.filter((e) => e.event === 'game_state');
    expect(gameStateEvents.length).toBe(0);
  });

  it('emits ACCESS_DENIED when join_game fails with an access error', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const joinSpy = jest.spyOn(serverAny, 'handleJoinGame').mockImplementation(() => {
      throw new Error('Access denied');
    });

    socket.trigger('join_game', { gameId: 'private-game' });

    expect(joinSpy).toHaveBeenCalledTimes(1);

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'ACCESS_DENIED',
      event: 'join_game',
      message: expect.any(String),
    });

    const gameStateEvents = socket.emittedEvents.filter((e) => e.event === 'game_state');
    expect(gameStateEvents.length).toBe(0);
  });

  it('emits MOVE_REJECTED when player_move is rejected by the rules engine', () => {
    const { serverAny, socket } = setupServerAndSocket();
    const moveSpy = jest.spyOn(serverAny, 'handlePlayerMove').mockImplementation(() => {
      throw new Error('Illegal move');
    });

    // Valid payload that passes schema validation
    const goodPayload = {
      gameId: 'game-1',
      move: {
        moveType: 'place_ring',
        position: '{"x":0,"y":0}',
        moveNumber: 1,
      },
    };

    socket.trigger('player_move', goodPayload);

    expect(moveSpy).toHaveBeenCalledTimes(1);

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(1);

    const payload = errorEvents[0].payload;
    expect(payload).toMatchObject({
      type: 'error',
      code: 'MOVE_REJECTED',
      event: 'player_move',
      message: expect.any(String),
    });

    const gameStateEvents = socket.emittedEvents.filter((e) => e.event === 'game_state');
    expect(gameStateEvents.length).toBe(0);
  });

  it('accepts valid diagnostic:ping payload and echoes a diagnostic:pong with round-trip metadata', () => {
    const { socket } = setupServerAndSocket();

    const now = Date.now();
    const pingPayload = {
      timestamp: now,
      vu: 42,
      sequence: 7,
    };

    socket.trigger('diagnostic:ping', pingPayload);

    const errorEvents = socket.emittedEvents.filter((e) => e.event === 'error');
    expect(errorEvents.length).toBe(0);

    const pongEvents = socket.emittedEvents.filter((e) => e.event === 'diagnostic:pong');
    expect(pongEvents.length).toBe(1);

    const pongPayload = pongEvents[0].payload;
    expect(pongPayload.timestamp).toBe(now);
    expect(pongPayload.vu).toBe(42);
    expect(pongPayload.sequence).toBe(7);
    expect(typeof pongPayload.serverTimestamp).toBe('string');
  });
});
