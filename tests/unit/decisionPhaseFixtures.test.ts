const mockGameCreate = jest.fn();

jest.mock('../../src/server/database/connection', () => ({
  getDatabaseClient: jest.fn(() => ({
    game: { create: mockGameCreate },
  })),
}));

jest.mock('../../src/server/utils/logger', () => ({
  logger: { info: jest.fn(), warn: jest.fn(), error: jest.fn(), debug: jest.fn() },
}));

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  applyDecisionPhaseFixtureIfNeeded,
  createDecisionPhaseFixtureGame,
} from '../../src/server/game/testFixtures/decisionPhaseFixtures';
import { BOARD_CONFIGS, type Player, type TimeControl } from '../../src/shared/types/game';

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

  it('lets the canonical adapter resolve the line timeout fixture in one move', async () => {
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
    const players: Player[] = [1, 2].map((playerNumber) => ({
      id: `player-${playerNumber}`,
      username: `Player ${playerNumber}`,
      playerNumber,
      type: 'human',
      isReady: true,
      timeRemaining: 600_000,
      ringsInHand: BOARD_CONFIGS.square8.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
    const engine = new GameEngine('fixture-game-123', 'square8', players, timeControl, false);

    expect(
      applyDecisionPhaseFixtureIfNeeded(engine, {
        fixture: { kind: 'decision_phase_fixture', scenario: 'line_processing', version: 1 },
      })
    ).toBe(true);

    const processLineMove = engine.getValidMoves(1).find((move) => move.type === 'process_line');
    expect(processLineMove).toBeDefined();
    if (!processLineMove) {
      throw new Error('Expected the line fixture to expose a process_line move');
    }

    const result = await engine.makeMoveById(1, processLineMove.id);

    expect(result.success).toBe(true);
    expect(engine.getGameState().currentPhase).not.toBe('line_processing');
  });

  it('exposes a canonical territory choice from the territory timeout fixture', () => {
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
    const players: Player[] = [1, 2].map((playerNumber) => ({
      id: `player-${playerNumber}`,
      username: `Player ${playerNumber}`,
      playerNumber,
      type: 'human',
      isReady: true,
      timeRemaining: 600_000,
      ringsInHand: BOARD_CONFIGS.square8.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
    const engine = new GameEngine('fixture-game-123', 'square8', players, timeControl, false);

    expect(
      applyDecisionPhaseFixtureIfNeeded(engine, {
        fixture: { kind: 'decision_phase_fixture', scenario: 'territory_processing', version: 1 },
      })
    ).toBe(true);

    const territoryMoves = engine
      .getValidMoves(1)
      .filter((move) => move.type === 'choose_territory_option');

    expect(territoryMoves.length).toBeGreaterThan(0);
    expect(territoryMoves[0]?.disconnectedRegions?.[0]?.isDisconnected).toBe(true);
  });

  it('exposes canonical continuation moves from the chain-capture timeout fixture', () => {
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
    const players: Player[] = [1, 2].map((playerNumber) => ({
      id: `player-${playerNumber}`,
      username: `Player ${playerNumber}`,
      playerNumber,
      type: 'human',
      isReady: true,
      timeRemaining: 600_000,
      ringsInHand: BOARD_CONFIGS.square8.ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
    const engine = new GameEngine('fixture-game-123', 'square8', players, timeControl, false);

    expect(
      applyDecisionPhaseFixtureIfNeeded(engine, {
        fixture: { kind: 'decision_phase_fixture', scenario: 'chain_capture_choice', version: 1 },
      })
    ).toBe(true);

    const state = engine.getGameState();
    const continuationMoves = engine
      .getValidMoves(1)
      .filter((move) => move.type === 'continue_capture_segment');

    expect(state.currentPhase).toBe('chain_capture');
    expect(continuationMoves.length).toBeGreaterThan(1);
  });
});
