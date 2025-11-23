import React from 'react';
import { render } from '@testing-library/react';
import { GameProvider, useGame } from '../../src/client/contexts/GameContext';

type HandlerMap = { [event: string]: (...args: any[]) => void };
const socketEventHandlers: HandlerMap = {};
// Named with `mock` prefix so Jest allows it to be referenced from jest.mock factory.
const mockEmit = jest.fn();

jest.mock('socket.io-client', () => {
  return {
    __esModule: true,
    io: jest.fn((_url: string, _options?: any) => ({
      on: jest.fn((event: string, handler: (...args: any[]) => void) => {
        socketEventHandlers[event] = handler;
      }),
      emit: mockEmit,
      disconnect: jest.fn(),
    })),
    Socket: jest.fn(),
  };
});

jest.mock('react-hot-toast', () => {
  const base = jest.fn();
  (base as any).success = jest.fn();
  (base as any).error = jest.fn();
  return {
    __esModule: true,
    toast: base,
  };
});

function TestHarness({ gameId }: { gameId: string }) {
  const { connectToGame } = useGame();
  React.useEffect(() => {
    void connectToGame(gameId);
  }, [connectToGame, gameId]);
  return null;
}

describe('GameContext WebSocket reconnection', () => {
  beforeEach(() => {
    mockEmit.mockClear();
    for (const key of Object.keys(socketEventHandlers)) {
      delete socketEventHandlers[key];
    }
  });

  it('re-emits join_game with the current gameId on socket reconnect', () => {
    const targetGameId = 'game-123';

    render(
      <GameProvider>
        <TestHarness gameId={targetGameId} />
      </GameProvider>
    );

    // Simulate initial connection.
    const connectHandler = socketEventHandlers['connect'];
    expect(typeof connectHandler).toBe('function');
    connectHandler?.();

    expect(mockEmit).toHaveBeenCalledWith('join_game', { gameId: targetGameId });

    mockEmit.mockClear();

    // Simulate a socket.io-level reconnect.
    const reconnectHandler = socketEventHandlers['reconnect'];
    expect(typeof reconnectHandler).toBe('function');
    reconnectHandler?.();

    expect(mockEmit).toHaveBeenCalledWith('join_game', { gameId: targetGameId });
  });
});
