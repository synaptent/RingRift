import express from 'express';
import request from 'supertest';

import { errorHandler } from '../../src/server/middleware/errorHandler';
import { sandboxHelperRoutes } from '../../src/server/routes/game';
import { getAIServiceClient } from '../../src/server/services/AIServiceClient';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { serializeGameState } from '../../src/shared/engine/contracts/serialization';
import type { GameState, Move, Player, TimeControl } from '../../src/shared/types/game';

jest.mock('../../src/server/services/AIServiceClient', () => ({
  getAIServiceClient: jest.fn(),
}));

function createTestApp() {
  const app = express();
  app.use(express.json());
  app.use('/api/games', sandboxHelperRoutes);
  app.use(errorHandler as any);
  return app;
}

function createMinimalActiveState(
  boardType: GameState['boardType'],
  numPlayers: number
): GameState {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const players: Player[] = Array.from({ length: numPlayers }, (_, idx) => ({
    id: `p${idx + 1}`,
    username: `P${idx + 1}`,
    type: 'ai',
    playerNumber: idx + 1,
    isReady: true,
    timeRemaining: 0,
    ringsInHand: 0,
    eliminatedRings: 0,
    territorySpaces: 0,
  }));

  const state = createInitialGameState(
    'sandbox-ai-contract-test',
    boardType,
    players,
    timeControl,
    false,
    123
  );

  return { ...state, gameStatus: 'active' };
}

describe('POST /api/games/sandbox/ai/move (contract)', () => {
  it('threads difficulty + playerNumber into AI service and returns metadata fields', async () => {
    const app = createTestApp();
    const state = createMinimalActiveState('square8', 2);
    const serialized = serializeGameState(state);

    const mockGetAIMove = jest.fn(
      async (_state: GameState, playerNumber: number, difficulty: number) => {
        const move: Move = {
          type: 'skip_placement',
          player: playerNumber,
          to: { x: 0, y: 0 },
          moveNumber: 1,
        };

        return {
          move,
          evaluation: null,
          thinking_time_ms: 12,
          ai_type: 'mcts',
          difficulty,
          heuristic_profile_id: 'heuristic_sq8_2p_v1',
          use_neural_net: true,
          nn_model_id: 'ringrift_best_sq8_2p',
          nn_checkpoint: 'ckpt_123',
          nnue_checkpoint: null,
        };
      }
    );

    (getAIServiceClient as unknown as jest.Mock).mockReturnValue({
      getAIMove: mockGetAIMove,
    });

    const res = await request(app)
      .post('/api/games/sandbox/ai/move')
      .send({
        state: serialized,
        difficulty: 11, // should clamp to 10
        playerNumber: 2,
      })
      .expect(200);

    expect(mockGetAIMove).toHaveBeenCalledTimes(1);
    expect(mockGetAIMove.mock.calls[0][1]).toBe(2);
    expect(mockGetAIMove.mock.calls[0][2]).toBe(10);

    expect(res.body).toMatchObject({
      move: expect.objectContaining({ type: 'skip_placement', player: 2 }),
      evaluation: null,
      thinkingTimeMs: 12,
      aiType: 'mcts',
      difficulty: 10,
      heuristicProfileId: 'heuristic_sq8_2p_v1',
      useNeuralNet: true,
      nnModelId: 'ringrift_best_sq8_2p',
      nnCheckpoint: 'ckpt_123',
      nnueCheckpoint: null,
    });
  });
});
