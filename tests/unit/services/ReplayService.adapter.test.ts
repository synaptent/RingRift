import { adaptHistoryToGameRecord } from '../../../src/client/services/ReplayService';
import type { GameHistoryResponse, GameDetailsResponse } from '../../../src/client/services/api';

describe('ReplayService adaptHistoryToGameRecord – outcome mapping', () => {
  function createHistory(overrides: Partial<GameHistoryResponse> = {}): GameHistoryResponse {
    return {
      gameId: 'history-game-1',
      moves: [],
      totalMoves: 0,
      ...overrides,
    };
  }

  function createDetails(overrides: Partial<GameDetailsResponse> = {}): GameDetailsResponse {
    const baseTimestamp = new Date('2024-01-01T00:00:00Z').toISOString();
    return {
      id: 'history-game-1',
      status: 'completed',
      boardType: 'square8',
      maxPlayers: 2,
      isRated: false,
      allowSpectators: true,
      players: [
        { id: 'p1', username: 'P1', rating: 1200 },
        { id: 'p2', username: 'P2', rating: 1200 },
      ],
      winner: null,
      createdAt: baseTimestamp,
      updatedAt: baseTimestamp,
      startedAt: baseTimestamp,
      endedAt: baseTimestamp,
      moveCount: 0,
      ...overrides,
    };
  }

  it('maps terminal reason game_completed to GameOutcome draw', () => {
    const history: GameHistoryResponse = createHistory({
      result: {
        // Structural stalemate / generic completion
        reason: 'game_completed' as any,
        winner: null,
      },
    });

    const details: GameDetailsResponse = createDetails();

    const { record } = adaptHistoryToGameRecord(history, details);

    expect(record.outcome).toBe('draw');
  });

  it('passes through canonical GameOutcome reasons unchanged', () => {
    const canonicalReasons: Array<GameHistoryResponse['result']> = [
      { reason: 'ring_elimination', winner: 1 } as any,
      { reason: 'territory_control', winner: 1 } as any,
      { reason: 'last_player_standing', winner: 1 } as any,
      { reason: 'timeout', winner: 1 } as any,
      { reason: 'resignation', winner: 1 } as any,
      { reason: 'draw', winner: null } as any,
      { reason: 'abandonment', winner: null } as any,
    ];

    const details = createDetails();

    canonicalReasons.forEach((result) => {
      const history = createHistory({ result });
      const { record } = adaptHistoryToGameRecord(history, details);
      expect(record.outcome).toBe(result.reason);
    });
  });
});
