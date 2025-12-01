import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHistoryPanel } from '../../../src/client/components/GameHistoryPanel';
import type { GameHistoryResponse, GameHistoryMove } from '../../../src/client/services/api';

// Mock the game API so we can control the history payload returned to the panel
const mockGetGameHistory = jest.fn<Promise<GameHistoryResponse>, [string]>();

jest.mock('../../../src/client/services/api', () => {
  const actual = jest.requireActual('../../../src/client/services/api');
  return {
    ...actual,
    gameApi: {
      ...actual.gameApi,
      getGameHistory: (gameId: string) => mockGetGameHistory(gameId),
    },
  };
});

describe('GameHistoryPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function createMove(overrides: Partial<GameHistoryMove> = {}): GameHistoryMove {
    return {
      moveNumber: 1,
      playerId: 'player-1',
      playerName: 'Player One',
      moveType: 'place_ring',
      moveData: {
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      },
      timestamp: new Date('2024-01-15T10:05:30Z').toISOString(),
      ...overrides,
    };
  }

  it('renders an auto-resolve badge for moves with autoResolved metadata', async () => {
    const autoResolvedMove = createMove({
      moveNumber: 1,
      autoResolved: {
        reason: 'timeout',
        choiceKind: 'line',
        choiceType: 'reward',
      },
    });

    const normalMove = createMove({
      moveNumber: 2,
      playerId: 'player-2',
      playerName: 'Player Two',
      autoResolved: undefined,
    });

    const history: GameHistoryResponse = {
      gameId: 'game-1',
      moves: [autoResolvedMove, normalMove],
      totalMoves: 2,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-1" />);

    // Wait for history to load and the badge to appear
    const badge = await waitFor(() => screen.getByTestId('auto-resolved-badge'));

    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('Auto-resolved (timeout)');

    // There should be exactly one auto-resolve badge even though we have two moves
    const allBadges = screen.getAllByTestId('auto-resolved-badge');
    expect(allBadges).toHaveLength(1);
  });

  it('does not render an auto-resolve badge when no moves are auto-resolved', async () => {
    const normalMove = createMove({
      moveNumber: 1,
      autoResolved: undefined,
    });

    const history: GameHistoryResponse = {
      gameId: 'game-2',
      moves: [normalMove],
      totalMoves: 1,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-2" />);

    // Wait for loading to complete by asserting that the move row appears
    await waitFor(() => {
      expect(screen.getByText('Player One')).toBeInTheDocument();
    });

    expect(screen.queryByTestId('auto-resolved-badge')).not.toBeInTheDocument();
  });

  it.each([
    { reason: 'timeout', winner: 1, expectedLabel: 'Result: Timeout', expectWinner: true },
    {
      reason: 'resignation',
      winner: 2,
      expectedLabel: 'Result: Resignation',
      expectWinner: true,
    },
    {
      reason: 'abandonment',
      winner: null,
      expectedLabel: 'Result: Abandonment',
      expectWinner: false,
    },
  ] as const)(
    'renders terminal result banner for $reason games',
    async ({ reason, winner, expectedLabel, expectWinner }) => {
      const history: GameHistoryResponse = {
        gameId: 'game-terminal',
        moves: [createMove()],
        totalMoves: 1,
        result: {
          // Cast needed because reason is a narrowed string literal in this test
          reason: reason as GameHistoryResponse['result'] extends { reason: infer R } ? R : never,
          winner,
        },
      };

      mockGetGameHistory.mockResolvedValue(history);

      render(<GameHistoryPanel gameId="game-terminal" />);

      await waitFor(() => {
        expect(screen.getByText('Player One')).toBeInTheDocument();
      });

      expect(screen.getByText(expectedLabel)).toBeInTheDocument();

      const winnerText = screen.queryByText(/Winner: P/i);
      if (expectWinner) {
        expect(winnerText).toBeInTheDocument();
      } else {
        expect(winnerText).not.toBeInTheDocument();
      }
    }
  );
});
