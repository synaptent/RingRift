import axios from 'axios';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';
import { logger } from '../../src/server/utils/logger';

jest.mock('axios');
jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    error: jest.fn(),
  },
}));

describe('PythonRulesClient.evaluateMove', () => {
  const createAxiosPostMock = () => {
    const postMock = jest.fn();
    (axios as any).create.mockReturnValue({
      post: postMock,
    });
    return postMock;
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('sends correct payload and maps snake_case fields to camelCase', async () => {
    const postMock = createAxiosPostMock();

    const client = new PythonRulesClient('http://python-rules.test');

    const state = { id: 'game-1' } as any;
    const move = { type: 'move_stack', player: 1 } as any;

    const wireResponse = {
      valid: true,
      validation_error: 'ok',
      next_state: { id: 'game-1', gameStatus: 'active' } as any,
      state_hash: 'hash-123',
      s_invariant: 7,
      game_status: 'active' as any,
    };

    postMock.mockResolvedValue({ data: wireResponse });

    const result = await client.evaluateMove(state, move);

    expect(postMock).toHaveBeenCalledTimes(1);
    expect(postMock).toHaveBeenCalledWith('/rules/evaluate_move', {
      game_state: state,
      move,
    });

    expect(result.valid).toBe(true);
    expect(result.validationError).toBe('ok');
    expect(result.nextState).toEqual(wireResponse.next_state);
    expect(result.stateHash).toBe('hash-123');
    expect(result.sInvariant).toBe(7);
    expect(result.gameStatus).toBe('active');
  });

  it('logs and rethrows errors from the HTTP client', async () => {
    const postMock = createAxiosPostMock();
    const client = new PythonRulesClient('http://python-rules.test');

    const state = { id: 'game-1' } as any;
    const move = { type: 'move_stack', player: 1 } as any;

    const error = {
      message: 'boom',
      response: {
        status: 500,
        data: { detail: 'internal error' },
      },
    };

    postMock.mockRejectedValue(error);

    await expect(client.evaluateMove(state, move)).rejects.toBe(error);

    expect(logger.error).toHaveBeenCalledTimes(1);
    expect(logger.error).toHaveBeenCalledWith(
      'Python rules evaluate_move failed',
      expect.objectContaining({
        message: 'boom',
        status: 500,
      })
    );
  });
});
