import { WebSocketServer } from '../../src/server/websocket/server';
import { Move } from '../../src/shared/types/game';

// Mock the database layer so maybePerformAITurn does not touch a real DB.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => null
}));

// Mock the global AI engine so we can control AI configuration and responses.
jest.mock('../../src/server/game/ai/AIEngine', () => {
  const getAIConfig = jest.fn();
  const createAI = jest.fn();
  const getAIMove = jest.fn();

  return {
    globalAIEngine: {
      getAIConfig,
      createAI,
      getAIMove
    }
  };
});

// Minimal Socket.IO "server" stub that records game_state emissions.
class FakeSocketIOServer {
  public toCalls: Array<{ gameId: string; event: string; payload: any }> = [];

  to(gameId: string) {
    return {
      emit: (event: string, payload: any) => {
        this.toCalls.push({ gameId, event, payload });
      }
    };
  }
}

describe('WebSocketServer.maybePerformAITurn', () => {
  it('requests a move from the AI engine, applies it via GameEngine, and emits game_state', async () => {
    // Arrange: set up a WebSocketServer with a fake Socket.IO layer.
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    const { globalAIEngine } = require('../../src/server/game/ai/AIEngine');

    // Current game state: active game, current player is an AI (player 2).
    const state: any = {
      gameStatus: 'active',
      currentPlayer: 2,
      players: [
        { id: 'p1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'p2', username: 'AI', playerNumber: 2, type: 'ai', aiDifficulty: 5 }
      ],
      moveHistory: []
    };

    const aiMove: Move = {
      id: 'ai-move-1',
      type: 'place_ring',
      player: 2,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    // AI config already exists; no need to call createAI.
    globalAIEngine.getAIConfig.mockReturnValue({ difficulty: 5 });
    globalAIEngine.getAIMove.mockResolvedValue(aiMove);

    const makeMove = jest.fn(async (move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>) => {
      // Simulate the engine applying the move and advancing the turn.
      state.moveHistory.push({ ...aiMove, ...move, timestamp: new Date() });
      state.currentPlayer = 1; // turn passes to player 1
      return { success: true };
    });

    const fakeEngine: any = {
      getGameState: () => state,
      makeMove,
      // maybePerformAITurn calls getValidMoves(updatedState.currentPlayer) to
      // compute the next player's legal moves for the broadcast payload.
      // For this integration test we only care that it can be called
      // successfully, so we return an empty list.
      getValidMoves: jest.fn(() => [])
    };

    // Act: invoke maybePerformAITurn for an AI turn.
    await serverAny.maybePerformAITurn('game-1', fakeEngine);

    // Assert: AI config and move were requested as expected.
    expect(globalAIEngine.getAIConfig).toHaveBeenCalledWith(2);
    expect(globalAIEngine.getAIMove).toHaveBeenCalledWith(2, state);

    // The engine should have attempted to apply the AI move.
    expect(makeMove).toHaveBeenCalledTimes(1);
    const appliedMoveArg = makeMove.mock.calls[0][0];
    expect(appliedMoveArg.player).toBe(2);
    expect(appliedMoveArg.type).toBe('place_ring');

    // A game_state event should have been emitted with the updated state.
    const gameStateCalls = fakeIo.toCalls.filter(call => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);

    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe('game-1');
    expect(payload.data.gameState.currentPlayer).toBe(1);
  });
});
