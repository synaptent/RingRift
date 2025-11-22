import { WebSocketServer } from '../../src/server/websocket/server';
import { Move } from '../../src/shared/types/game';

// Mock the database layer so maybePerformAITurn does not touch a real DB.
jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: () => null,
}));

// Mock the global AI engine so we can control AI configuration and responses.
jest.mock('../../src/server/game/ai/AIEngine', () => {
  const getAIConfig = jest.fn();
  const createAI = jest.fn();
  const getAIMove = jest.fn();
  const chooseLocalMoveFromCandidates = jest.fn();

  return {
    globalAIEngine: {
      getAIConfig,
      createAI,
      getAIMove,
      chooseLocalMoveFromCandidates,
    },
  };
});

// Minimal Socket.IO "server" stub that records game_state emissions.
class FakeSocketIOServer {
  public toCalls: Array<{ gameId: string; event: string; payload: any }> = [];

  to(gameId: string) {
    return {
      emit: (event: string, payload: any) => {
        this.toCalls.push({ gameId, event, payload });
      },
    };
  }
}

describe('WebSocketServer.maybePerformAITurn', () => {
  it('requests a move from the AI engine, applies it via GameEngine, and emits game_state in normal phases', async () => {
    // Arrange: set up a WebSocketServer with a fake Socket.IO layer.
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { globalAIEngine } = require('../../src/server/game/ai/AIEngine');

    // Current game state: active game, current player is an AI (player 2)
    // in a normal interactive phase (movement).
    const state: any = {
      gameStatus: 'active',
      currentPhase: 'movement',
      currentPlayer: 2,
      players: [
        { id: 'p1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'p2', username: 'AI', playerNumber: 2, type: 'ai', aiDifficulty: 5 },
      ],
      moveHistory: [],
    };

    const aiMove: Move = {
      id: 'ai-move-1',
      type: 'place_ring',
      player: 2,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
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
      getValidMoves: jest.fn(() => []),
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
    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);

    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe('game-1');
    expect(payload.data.gameState.currentPlayer).toBe(1);
  });

  it('uses local decision policy for line_processing / territory_processing and does not call getAIMove', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { globalAIEngine } = require('../../src/server/game/ai/AIEngine');

    const baseState: any = {
      gameStatus: 'active',
      currentPhase: 'line_processing',
      currentPlayer: 2,
      players: [
        { id: 'p1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'p2', username: 'AI', playerNumber: 2, type: 'ai', aiDifficulty: 5 },
      ],
      moveHistory: [],
    };

    const decisionMove: Move = {
      id: 'process-line-0-0,0',
      type: 'process_line',
      player: 2,
      formedLines: [],
      // Decision moves are phase-driven and do not use `to`, but the
      // shared Move type requires it; provide a harmless sentinel.
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    // Ensure an AI config exists.
    globalAIEngine.getAIConfig.mockReturnValue({ difficulty: 5 });
    // For decision phases, maybePerformAITurn should *not* call getAIMove.
    globalAIEngine.getAIMove.mockReset();
    // Instead it should call chooseLocalMoveFromCandidates with the
    // decision candidates from getValidMoves.
    globalAIEngine.chooseLocalMoveFromCandidates.mockReturnValue(decisionMove);

    const state = { ...baseState };

    const makeMove = jest.fn(async (move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>) => {
      state.moveHistory.push({ ...decisionMove, ...move, timestamp: new Date() });
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      return { success: true };
    });

    const fakeEngine: any = {
      getGameState: () => state,
      makeMove,
      getValidMoves: jest.fn(() => [decisionMove]),
    };

    await serverAny.maybePerformAITurn('game-2', fakeEngine);

    // For decision phases, we should use the local decision policy and
    // not the service-backed getAIMove path.
    expect(globalAIEngine.chooseLocalMoveFromCandidates).toHaveBeenCalled();
    expect(globalAIEngine.getAIMove).not.toHaveBeenCalled();

    // Engine should have been asked to apply a canonical decision move.
    expect(makeMove).toHaveBeenCalledTimes(1);
    const appliedMoveArg = makeMove.mock.calls[0][0];
    expect(appliedMoveArg.type).toBe('process_line');
    expect(appliedMoveArg.player).toBe(2);

    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);
    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe('game-2');
    expect(payload.data.gameState.currentPlayer).toBe(1);
  });

  it('uses local decision policy for eliminate_rings_from_stack in territory_processing and does not call getAIMove', async () => {
    const httpServerStub: any = {};
    const wsServer = new WebSocketServer(httpServerStub as any);
    const serverAny: any = wsServer as any;

    const fakeIo = new FakeSocketIOServer();
    serverAny.io = fakeIo;

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { globalAIEngine } = require('../../src/server/game/ai/AIEngine');

    const baseState: any = {
      gameStatus: 'active',
      currentPhase: 'territory_processing',
      currentPlayer: 2,
      players: [
        { id: 'p1', username: 'Human', playerNumber: 1, type: 'human' },
        { id: 'p2', username: 'AI', playerNumber: 2, type: 'ai', aiDifficulty: 5 },
      ],
      moveHistory: [],
    };

    const eliminationMove: Move = {
      id: 'eliminate-0,1',
      type: 'eliminate_rings_from_stack',
      player: 2,
      to: { x: 0, y: 1 },
      eliminatedRings: [{ player: 2, count: 1 }],
      eliminationFromStack: {
        position: { x: 0, y: 1 },
        capHeight: 1,
        totalHeight: 2,
      },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    // Ensure an AI config exists.
    globalAIEngine.getAIConfig.mockReturnValue({ difficulty: 5 });
    // For decision phases, maybePerformAITurn should *not* call getAIMove.
    globalAIEngine.getAIMove.mockReset();
    // Instead it should call chooseLocalMoveFromCandidates with the
    // decision candidates from getValidMoves.
    globalAIEngine.chooseLocalMoveFromCandidates.mockReturnValue(eliminationMove);

    const state = { ...baseState };

    const makeMove = jest.fn(async (move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>) => {
      state.moveHistory.push({ ...eliminationMove, ...move, timestamp: new Date() });
      state.currentPhase = 'movement';
      state.currentPlayer = 1;
      return { success: true };
    });

    const fakeEngine: any = {
      getGameState: () => state,
      makeMove,
      getValidMoves: jest.fn(() => [eliminationMove]),
    };

    await serverAny.maybePerformAITurn('game-3', fakeEngine);

    // For elimination decisions in territory_processing, we should use the
    // local decision policy and not the service-backed getAIMove path.
    expect(globalAIEngine.chooseLocalMoveFromCandidates).toHaveBeenCalled();
    expect(globalAIEngine.getAIMove).not.toHaveBeenCalled();

    // Engine should have been asked to apply a canonical elimination move.
    expect(makeMove).toHaveBeenCalledTimes(1);
    const appliedMoveArg = makeMove.mock.calls[0][0];
    expect(appliedMoveArg.type).toBe('eliminate_rings_from_stack');
    expect(appliedMoveArg.player).toBe(2);

    const gameStateCalls = fakeIo.toCalls.filter((call) => call.event === 'game_state');
    expect(gameStateCalls.length).toBe(1);
    const payload = gameStateCalls[0].payload;
    expect(payload.data.gameId).toBe('game-3');
    expect(payload.data.gameState.currentPlayer).toBe(1);
  });
});
