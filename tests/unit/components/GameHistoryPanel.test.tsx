import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHistoryPanel } from '../../../src/client/components/GameHistoryPanel';
import { gameApi, type GameHistoryResponse } from '../../../src/client/services/api';

jest.mock('../../../src/client/services/api', () => ({
  gameApi: {
    getGameHistory: jest.fn(),
  },
}));

const mockGetGameHistory = gameApi.getGameHistory as jest.MockedFunction<
  typeof gameApi.getGameHistory
>;

function createMockHistory(overrides: Partial<GameHistoryResponse> = {}): GameHistoryResponse {
  return {
    gameId: 'game-123',
    totalMoves: 1,
    moves: [
      {
        moveNumber: 1,
        playerId: 'player-1',
        playerName: 'Player 1',
        moveType: 'move_stack',
        moveData: {
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
        },
        timestamp: '2024-01-01T12:00:00.000Z',
      },
    ],
    ...overrides,
  };
}

describe('GameHistoryPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('does not fetch history while collapsed and fetches when expanded', async () => {
    const history = createMockHistory();
    mockGetGameHistory.mockResolvedValueOnce(history);

    render(<GameHistoryPanel gameId="game-123" defaultCollapsed={true} />);

    // Initially collapsed: no fetch yet
    expect(mockGetGameHistory).not.toHaveBeenCalled();

    // Expand the panel via header button
    fireEvent.click(screen.getByRole('button', { name: /Move History/i }));

    await waitFor(() => {
      expect(mockGetGameHistory).toHaveBeenCalledWith('game-123');
    });

    // Move type from formatted history should appear
    expect(await screen.findByText('Move Stack')).toBeInTheDocument();
  });

  it('shows loading state, then renders moves when fetch succeeds', async () => {
    const history = createMockHistory({ totalMoves: 2 });
    mockGetGameHistory.mockResolvedValueOnce(history);

    render(<GameHistoryPanel gameId="game-123" defaultCollapsed={false} />);

    // Loading indicator should appear first
    expect(screen.getByText('Loading history...')).toBeInTheDocument();

    // After fetch resolves, moves and total count should be shown
    expect(await screen.findByText('(2 moves)')).toBeInTheDocument();
    expect(screen.getByText('Move Stack')).toBeInTheDocument();
  });

  it('renders error state and calls onError when fetch fails', async () => {
    const error = new Error('HTTP 500');
    mockGetGameHistory.mockRejectedValueOnce(error);
    const onError = jest.fn();

    render(<GameHistoryPanel gameId="game-123" defaultCollapsed={false} onError={onError} />);

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(error);
    });

    // User-facing error message should be visible
    expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
  });
});
