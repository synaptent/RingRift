import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import type { LoadableScenario } from '../../../src/client/sandbox/scenarioTypes';
import { SelfPlayBrowser } from '../../../src/client/components/SelfPlayBrowser';

describe('SelfPlayBrowser – self-play move normalization', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    jest.resetAllMocks();
  });

  afterAll(() => {
    global.fetch = originalFetch;
  });

  it('normalizes forced_elimination moves to eliminate_rings_from_stack in selfPlayMeta.moves', async () => {
    const databasesResponse = {
      success: true,
      databases: [
        {
          path: '/tmp/selfplay.db',
          name: 'selfplay.db',
          gameCount: 1,
          createdAt: '2025-12-01T10:56:51.370554Z',
        },
      ],
    };

    const gamesResponse = {
      success: true,
      games: [
        {
          gameId: 'game-1',
          boardType: 'square8',
          numPlayers: 2,
          winner: 1,
          totalMoves: 2,
          totalTurns: 2,
          createdAt: '2025-12-01T10:56:51.370554Z',
          completedAt: '2025-12-01T10:57:51.370554Z',
          source: 'selfplay',
          terminationReason: null,
          durationMs: 60000,
        },
      ],
      pagination: {
        limit: 100,
        offset: 0,
        returned: 1,
      },
    };

    const forcedMove = {
      id: 'm2',
      type: 'forced_elimination',
      player: 1,
      from: null,
      to: { x: 4, y: 0, z: null },
      timestamp: '2025-12-01T10:56:51.370554',
      thinkTime: 0,
      moveNumber: 2,
    };

    const placeMove = {
      id: 'm1',
      type: 'place_ring',
      player: 1,
      from: null,
      to: { x: 0, y: 0, z: null },
      timestamp: '2025-12-01T10:56:51.370554',
      thinkTime: 0,
      moveNumber: 1,
    };

    const gameDetailResponse = {
      success: true,
      game: {
        gameId: 'game-1',
        boardType: 'square8',
        numPlayers: 2,
        winner: 1,
        totalMoves: 2,
        totalTurns: 2,
        createdAt: '2025-12-01T10:56:51.370554Z',
        completedAt: '2025-12-01T10:57:51.370554Z',
        source: 'selfplay',
        terminationReason: null,
        durationMs: 60000,
        initialState: {},
        moves: [
          {
            moveNumber: 1,
            turnNumber: 1,
            player: 1,
            phase: 'ring_placement',
            moveType: 'place_ring',
            move: placeMove,
            thinkTimeMs: 0,
            engineEval: null,
          },
          {
            moveNumber: 2,
            turnNumber: 2,
            player: 1,
            phase: 'territory_processing',
            moveType: 'forced_elimination',
            move: forcedMove,
            thinkTimeMs: 0,
            engineEval: null,
          },
        ],
        players: [],
      },
    };

    const fetchMock = jest.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.startsWith('/api/selfplay/databases')) {
        return { json: async () => databasesResponse } as any;
      }
      if (url.startsWith('/api/selfplay/games?')) {
        return { json: async () => gamesResponse } as any;
      }
      if (url.startsWith('/api/selfplay/games/game-1')) {
        return { json: async () => gameDetailResponse } as any;
      }
      throw new Error(`Unexpected fetch URL in SelfPlayBrowser test: ${url}`);
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).fetch = fetchMock;

    const handleSelectGame = jest.fn<(scenario: LoadableScenario) => void>();

    render(
      <SelfPlayBrowser isOpen={true} onClose={() => undefined} onSelectGame={handleSelectGame} />
    );

    const loadButton = await screen.findByRole('button', { name: 'Load' });
    fireEvent.click(loadButton);

    await waitFor(() => {
      expect(handleSelectGame).toHaveBeenCalledTimes(1);
    });

    const scenario = handleSelectGame.mock.calls[0][0] as LoadableScenario;
    expect(scenario.selfPlayMeta).toBeDefined();
    const moves = scenario.selfPlayMeta?.moves;
    expect(moves).toBeDefined();
    expect(moves).toHaveLength(2);

    const [first, second] = moves!;
    expect(first.type).toBe('place_ring');
    expect(second.type).toBe('eliminate_rings_from_stack');
    expect(second.to).toEqual(forcedMove.to);
    expect(first.timestamp).toBeInstanceOf(Date);
    expect(second.timestamp).toBeInstanceOf(Date);
  });
});
