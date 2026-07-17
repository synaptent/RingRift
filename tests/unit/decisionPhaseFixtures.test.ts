const mockGameCreate = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: { create: mockGameCreate },
  })),
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: { info: jest.fn(), warn: jest.fn() },
}));

import { createDecisionPhaseFixtureGame } from '../../src/server/game/testFixtures/decisionPhaseFixtures';

describe('decision-phase fixture persistence', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGameCreate.mockResolvedValue({ id: 'fixture-game-123' });
  });

  it('persists the registered opponent as Player 2', async () => {
    await expect(
      createDecisionPhaseFixtureGame({
        creatorUserId: 'user-123',
        secondPlayerUserId: 'user-456',
        scenario: 'line_processing',
        isRated: false,
      })
    ).resolves.toBe('fixture-game-123');

    expect(mockGameCreate).toHaveBeenCalledWith({
      data: expect.objectContaining({
        maxPlayers: 2,
        player1Id: 'user-123',
        player2Id: 'user-456',
      }),
    });
  });
});
