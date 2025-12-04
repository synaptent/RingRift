import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReplayPanel } from '../../../src/client/components/ReplayPanel';

// Mock the ReplayService
const mockListGames = jest.fn();
const mockGetGame = jest.fn();
const mockGetStateAtMove = jest.fn();
const mockGetMoves = jest.fn();

jest.mock('../../../src/client/services/ReplayService', () => ({
  getReplayService: () => ({
    listGames: mockListGames,
    getGame: mockGetGame,
    getStateAtMove: mockGetStateAtMove,
    getMoves: mockGetMoves,
  }),
}));

// Mock MoveHistory component - use require('react').createElement to avoid JSX hoisting issues
jest.mock('../../../src/client/components/MoveHistory', () => ({
  MoveHistory: function MockMoveHistory(props: {
    moves: any[];
    onMoveClick?: (index: number) => void;
  }) {
    const React = require('react');
    return React.createElement(
      'div',
      { 'data-testid': 'move-history' },
      props.moves.map((_: any, i: number) =>
        React.createElement(
          'button',
          {
            key: i,
            'data-testid': `move-${i}`,
            onClick: () => props.onMoveClick?.(i),
          },
          `Move ${i}`
        )
      )
    );
  },
}));

describe('ReplayPanel', () => {
  const mockOnStateChange = jest.fn();
  const mockOnReplayModeChange = jest.fn();
  const mockOnForkFromPosition = jest.fn();
  const mockOnAnimationChange = jest.fn();

  const defaultProps = {
    onStateChange: mockOnStateChange,
    onReplayModeChange: mockOnReplayModeChange,
    onForkFromPosition: mockOnForkFromPosition,
    onAnimationChange: mockOnAnimationChange,
  };

  const mockGames = [
    {
      gameId: 'game-1',
      boardType: 'square8',
      numPlayers: 2,
      winner: 1,
      totalMoves: 50,
      source: 'cmaes',
      createdAt: '2025-01-01T00:00:00Z',
    },
    {
      gameId: 'game-2',
      boardType: 'hex',
      numPlayers: 3,
      winner: null,
      totalMoves: 75,
      source: 'self-play',
      createdAt: '2025-01-02T00:00:00Z',
    },
  ];

  const mockGameState = {
    id: 'state-1',
    players: [],
    board: { type: 'square8', cells: {} },
    currentTurn: 1,
    phase: 'movement',
  };

  const mockMoves = [
    {
      moveNumber: 1,
      player: 1,
      moveType: 'movement',
      move: { from: { x: 0, y: 0 }, to: { x: 1, y: 0 } },
    },
    {
      moveNumber: 2,
      player: 2,
      moveType: 'movement',
      move: { from: { x: 7, y: 7 }, to: { x: 6, y: 7 } },
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    mockListGames.mockResolvedValue({ games: mockGames, total: 2, hasMore: false });
    mockGetGame.mockResolvedValue({
      gameId: 'game-1',
      boardType: 'square8',
      numPlayers: 2,
      winner: 1,
      totalMoves: 50,
      source: 'cmaes',
    });
    mockGetStateAtMove.mockResolvedValue({ gameState: mockGameState });
    mockGetMoves.mockResolvedValue({ moves: mockMoves });
  });

  it('renders collapsed by default with header button', () => {
    render(<ReplayPanel {...defaultProps} />);

    expect(screen.getByText('Game Replay')).toBeInTheDocument();
    expect(screen.queryByText('Browse Recorded Games')).not.toBeInTheDocument();
  });

  it('expands on click and shows browse button', async () => {
    render(<ReplayPanel {...defaultProps} />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /game replay/i }));
    });

    await waitFor(() => {
      expect(screen.getByText('Browse Recorded Games')).toBeInTheDocument();
    });
  });

  it('renders expanded when defaultCollapsed is false', () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    expect(screen.getByText('Browse Recorded Games')).toBeInTheDocument();
  });

  it('loads games when Browse Recorded Games is clicked', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => {
      expect(mockListGames).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(screen.getByText(/square8 • 2p/)).toBeInTheDocument();
    });
  });

  it('shows loading state while fetching games', async () => {
    mockListGames.mockImplementation(() => new Promise(() => {})); // Never resolves

    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => {
      expect(screen.getByText('Loading games...')).toBeInTheDocument();
    });
  });

  it('shows error state when game loading fails', async () => {
    // Mock listGames to reject immediately
    mockListGames.mockRejectedValueOnce(new Error('Network error'));

    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    // Wait for the error to appear
    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('loads a game when selected from list', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Open game picker and wait for games to load
    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => {
      expect(screen.getByText(/square8 • 2p/)).toBeInTheDocument();
    });

    // Click the first game
    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      expect(mockGetGame).toHaveBeenCalledWith('game-1');
      expect(mockGetMoves).toHaveBeenCalledWith('game-1', 0, undefined, 1000);
      expect(mockGetStateAtMove).toHaveBeenCalledWith('game-1', 0);
    });

    await waitFor(() => {
      expect(mockOnReplayModeChange).toHaveBeenCalledWith(true);
      expect(mockOnStateChange).toHaveBeenCalled();
    });
  });

  it('shows playback controls when a game is loaded', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Load a game
    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => {
      expect(screen.getByText(/square8 • 2p/)).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      // Use exact aria-label match to distinguish from header button
      expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /step forward/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /step back/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /go to start/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /go to end/i })).toBeInTheDocument();
    });
  });

  it('shows Active badge when in replay mode', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      expect(screen.getByText('Active')).toBeInTheDocument();
    });
  });

  it('shows scrubber with correct move range', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      const scrubber = screen.getByRole('slider', { name: /move scrubber/i });
      expect(scrubber).toHaveAttribute('min', '0');
      expect(scrubber).toHaveAttribute('max', '50');
    });
  });

  it('shows speed control buttons', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      expect(screen.getByText('0.5x')).toBeInTheDocument();
      expect(screen.getByText('1x')).toBeInTheDocument();
      expect(screen.getByText('2x')).toBeInTheDocument();
      expect(screen.getByText('4x')).toBeInTheDocument();
    });
  });

  it('steps forward when step forward button is clicked', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByRole('button', { name: /step forward/i }));

    mockGetStateAtMove.mockClear();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /step forward/i }));
    });

    await waitFor(() => {
      expect(mockGetStateAtMove).toHaveBeenCalledWith('game-1', 1);
    });
  });

  it('calls onForkFromPosition when Fork button is clicked', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByText('Fork'));

    await act(async () => {
      fireEvent.click(screen.getByText('Fork'));
    });

    expect(mockOnForkFromPosition).toHaveBeenCalled();
  });

  it('exits replay mode when Exit Replay button is clicked', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByText('Exit Replay'));

    await act(async () => {
      fireEvent.click(screen.getByText('Exit Replay'));
    });

    expect(mockOnReplayModeChange).toHaveBeenCalledWith(false);
    expect(mockOnStateChange).toHaveBeenCalledWith(null);
  });

  it('responds to keyboard shortcuts in replay mode', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Load a game
    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByRole('button', { name: /step forward/i }));

    mockGetStateAtMove.mockClear();

    // Arrow keys for stepping
    await act(async () => {
      fireEvent.keyDown(window, { key: 'ArrowRight' });
    });

    await waitFor(() => {
      expect(mockGetStateAtMove).toHaveBeenCalledWith('game-1', 1);
    });

    // Navigate to move 1 first, then test ArrowLeft
    mockGetStateAtMove.mockClear();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'ArrowLeft' });
    });

    await waitFor(() => {
      expect(mockGetStateAtMove).toHaveBeenCalledWith('game-1', 0);
    });
  });

  it('exits replay on Escape key', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByText('Exit Replay'));

    mockOnReplayModeChange.mockClear();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });

    expect(mockOnReplayModeChange).toHaveBeenCalledWith(false);
  });

  it('ignores keyboard shortcuts when focus is on input', async () => {
    render(
      <div>
        <input data-testid="text-input" />
        <ReplayPanel {...defaultProps} defaultCollapsed={false} />
      </div>
    );

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByRole('button', { name: /step forward/i }));

    mockGetStateAtMove.mockClear();

    const input = screen.getByTestId('text-input');
    input.focus();

    await act(async () => {
      fireEvent.keyDown(input, { key: 'ArrowRight' });
    });

    // Should not have stepped forward because focus was on input
    expect(mockGetStateAtMove).not.toHaveBeenCalled();
  });

  it('shows keyboard hints in replay mode', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      // Keyboard hints div exists with these text fragments
      expect(screen.getByText('Space')).toBeInTheDocument();
      expect(screen.getByText('Esc')).toBeInTheDocument();
    });
  });

  it('displays game metadata when loaded', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => {
      // Game info section
      expect(screen.getByText(/P1 won/)).toBeInTheDocument();
      expect(screen.getByText(/cmaes/)).toBeInTheDocument();
    });
  });

  it('changes speed when speed button is clicked', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByText('2x'));

    // Click 2x speed button
    await act(async () => {
      fireEvent.click(screen.getByText('2x'));
    });

    // The 2x button should now be pressed
    expect(screen.getByText('2x')).toHaveAttribute('aria-pressed', 'true');
  });

  it('seeks to move when scrubber is changed', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await act(async () => {
      fireEvent.click(screen.getByText('Browse Recorded Games'));
    });

    await waitFor(() => screen.getByText(/square8 • 2p/));

    await act(async () => {
      fireEvent.click(screen.getByText(/square8 • 2p/));
    });

    await waitFor(() => screen.getByRole('slider', { name: /move scrubber/i }));

    mockGetStateAtMove.mockClear();

    const scrubber = screen.getByRole('slider', { name: /move scrubber/i });

    await act(async () => {
      fireEvent.change(scrubber, { target: { value: '25' } });
    });

    await waitFor(() => {
      expect(mockGetStateAtMove).toHaveBeenCalledWith('game-1', 25);
    });
  });
});
