import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReplayPanel } from '../../../src/client/components/ReplayPanel';

// ─────────────────────────────────────────────────────────────────────────────
// Mocks
// ─────────────────────────────────────────────────────────────────────────────

const mockGetUserGames = jest.fn();
const mockGetGameHistory = jest.fn();
const mockGetGameDetails = jest.fn();
const mockAdaptHistoryToGameRecord = jest.fn();
const mockReconstructStateAtMove = jest.fn();

// AuthContext: always provide a logged-in user so ReplayPanel will attempt to
// load backend games via /api/games/user/:id.
jest.mock('../../../src/client/contexts/AuthContext', () => ({
  useAuth: () => ({
    user: {
      id: 'user-1',
      username: 'TestUser',
      email: 'test@example.com',
      role: 'player',
      rating: 1500,
      gamesPlayed: 10,
      gamesWon: 5,
      createdAt: new Date(),
      lastActive: new Date(),
      status: 'online',
      avatar: null,
      preferences: {
        boardTheme: 'classic',
        pieceStyle: 'traditional',
        soundEnabled: true,
        animationsEnabled: true,
        autoPromoteQueen: true,
        showCoordinates: true,
        highlightLastMove: true,
        confirmMoves: false,
        timeZone: 'UTC',
        language: 'en',
      },
    },
    isLoading: false,
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
    updateUser: jest.fn(),
  }),
}));

// Backend Game API used by ReplayPanel for multi-game backend replay.
jest.mock('../../../src/client/services/api', () => ({
  gameApi: {
    getUserGames: (...args: any[]) => mockGetUserGames(...args),
    getGameHistory: (...args: any[]) => mockGetGameHistory(...args),
    getGameDetails: (...args: any[]) => mockGetGameDetails(...args),
  },
}));

// Shared adapter that projects backend history+details into canonical GameRecord
// + Move[] for replay. We stub this so tests don't depend on its internal mapping.
jest.mock('../../../src/client/services/ReplayService', () => ({
  adaptHistoryToGameRecord: (...args: any[]) => mockAdaptHistoryToGameRecord(...args),
}));

// Canonical replay helper; for ReplayPanel tests we don't care about the exact
// GameState shape, only that callbacks are invoked, so we stub this as well.
jest.mock('../../../src/shared/engine/replayHelpers', () => ({
  reconstructStateAtMove: (...args: any[]) => mockReconstructStateAtMove(...args),
}));

// Mock MoveHistory component - keep the contract (moves + onMoveClick) but
// render a simple button list so we can trigger jump-to-move behaviour.
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
          `Move ${i + 1}`
        )
      )
    );
  },
}));

// Capture GameList props so tests can assert on passed games and invoke
// onSelectGame directly without depending on its internal markup.
const mockGameListProps = jest.fn();

jest.mock('../../../src/client/components/ReplayPanel/GameList', () => ({
  GameList: function MockGameList(props: any) {
    mockGameListProps(props);

    const React = require('react');
    // We don't render individual game rows here; tests interact via props.
    return React.createElement('div', { 'data-testid': 'game-list' });
  },
}));

describe('ReplayPanel (backend multi-game replay browser)', () => {
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

  beforeEach(() => {
    jest.clearAllMocks();

    // Stub backend user-games response
    mockGetUserGames.mockResolvedValue({
      games: [
        {
          id: 'game-1',
          boardType: 'square8',
          status: 'completed',
          playerCount: 2,
          maxPlayers: 2,
          winnerId: 'user-1',
          winnerName: 'TestUser',
          createdAt: '2025-01-01T00:00:00Z',
          endedAt: '2025-01-01T00:30:00Z',
          moveCount: 2,
          numPlayers: 2,
          isRated: true,
          source: 'online_game',
          outcome: 'ring_elimination',
          resultReason: 'ring_elimination',
        },
        {
          id: 'game-2',
          boardType: 'hexagonal',
          status: 'completed',
          playerCount: 3,
          maxPlayers: 3,
          winnerId: null,
          winnerName: null,
          createdAt: '2025-01-02T00:00:00Z',
          endedAt: '2025-01-02T00:30:00Z',
          moveCount: 3,
          numPlayers: 3,
          isRated: false,
          source: 'self_play',
          outcome: 'timeout',
          resultReason: 'timeout',
        },
      ],
      pagination: {
        total: 2,
        limit: 100,
        offset: 0,
        hasMore: false,
      },
    });

    // Stub history/details; ReplayPanel will call these but we delegate projection
    // to adaptHistoryToGameRecord, which we also stub.
    mockGetGameHistory.mockResolvedValue({
      gameId: 'game-1',
      moves: [
        {
          moveNumber: 1,
          playerId: 'p1',
          playerName: 'P1',
          moveType: 'place_ring',
          moveData: { to: { x: 0, y: 0 } },
          timestamp: '2025-01-01T00:00:01Z',
        },
        {
          moveNumber: 2,
          playerId: 'p2',
          playerName: 'P2',
          moveType: 'place_ring',
          moveData: { to: { x: 1, y: 0 } },
          timestamp: '2025-01-01T00:00:05Z',
        },
      ],
      totalMoves: 2,
      result: {
        reason: 'ring_elimination',
        winner: 1,
      },
    });

    mockGetGameDetails.mockResolvedValue({
      id: 'game-1',
      status: 'completed',
      boardType: 'square8',
      maxPlayers: 2,
      isRated: true,
      allowSpectators: true,
      players: [
        { id: 'p1', username: 'P1', rating: 1500 },
        { id: 'p2', username: 'P2', rating: 1500 },
      ],
      winner: { id: 'p1', username: 'P1' },
      createdAt: '2025-01-01T00:00:00Z',
      updatedAt: '2025-01-01T00:30:00Z',
      startedAt: '2025-01-01T00:00:00Z',
      endedAt: '2025-01-01T00:30:00Z',
      moveCount: 2,
    });

    const mockRecord = {
      id: 'game-1',
      boardType: 'square8',
      numPlayers: 2,
      isRated: true,
      players: [
        { playerNumber: 1, userId: 'p1', username: 'P1', rating: 1500 },
        { playerNumber: 2, userId: 'p2', username: 'P2', rating: 1500 },
      ],
      winner: 1,
      outcome: 'ring_elimination',
      finalScore: {
        ringsEliminated: { 1: 3, 2: 0 },
        territorySpaces: { 1: 0, 2: 0 },
        ringsRemaining: { 1: 15, 2: 18 },
      },
      startedAt: '2025-01-01T00:00:00Z',
      endedAt: '2025-01-01T00:30:00Z',
      totalMoves: 2,
      totalDurationMs: 1000,
      moves: [
        {
          player: 1,
          type: 'place_ring',
          to: { x: 0, y: 0 },
          placementCount: 1,
        },
        {
          player: 2,
          type: 'place_ring',
          to: { x: 1, y: 0 },
          placementCount: 1,
        },
      ],
      metadata: {
        recordVersion: 'test',
        createdAt: '2025-01-01T00:30:00Z',
        source: 'online_game',
        sourceId: 'game-1',
        tags: [],
      },
    };

    const mockMovesForDisplay = [
      {
        id: 'm1',
        type: 'place_ring',
        player: 1,
        to: { x: 0, y: 0 },
        placementCount: 1,
        timestamp: new Date('2025-01-01T00:00:01Z'),
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'place_ring',
        player: 2,
        to: { x: 1, y: 0 },
        placementCount: 1,
        timestamp: new Date('2025-01-01T00:00:05Z'),
        thinkTime: 0,
        moveNumber: 2,
      },
    ];

    mockAdaptHistoryToGameRecord.mockReturnValue({
      record: mockRecord,
      movesForDisplay: mockMovesForDisplay,
    });

    // For these tests, we don't care about the exact GameState shape; just
    // return a simple object to satisfy callbacks.
    mockReconstructStateAtMove.mockReturnValue({} as any);
  });

  it('renders collapsed by default with header button', () => {
    render(<ReplayPanel {...defaultProps} />);

    expect(screen.getByText('Game Replay')).toBeInTheDocument();
    // Content should be hidden while collapsed
    expect(
      screen.queryByText(/Browse and replay your completed backend games/i)
    ).not.toBeInTheDocument();
  });

  it('loads and displays user games when expanded by default', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Should request the logged-in user's completed games
    await waitFor(() => {
      expect(mockGetUserGames).toHaveBeenCalledWith(
        'user-1',
        expect.objectContaining({ status: 'completed' })
      );
    });

    // GameList should be rendered with some games prop
    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });
  });

  it('selecting a game loads replay data and enters replay mode', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Wait for GameList to receive games
    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });

    const latestCall = mockGameListProps.mock.calls[mockGameListProps.mock.calls.length - 1][0] as {
      onSelectGame: (gameId: string) => void;
    };

    // Simulate selecting game-1 via GameList's onSelectGame callback
    await act(async () => {
      latestCall.onSelectGame('game-1');
    });

    await waitFor(() => {
      expect(mockGetGameHistory).toHaveBeenCalledWith('game-1');
      expect(mockGetGameDetails).toHaveBeenCalledWith('game-1');
      expect(mockAdaptHistoryToGameRecord).toHaveBeenCalled();
    });

    await waitFor(() => {
      expect(mockOnReplayModeChange).toHaveBeenCalledWith(true);
    });

    // Playback controls should be visible (speed buttons, etc.)
    await waitFor(() => {
      expect(screen.getByText('Speed:')).toBeInTheDocument();
    });
  });

  it('exits replay mode when Exit Replay is clicked', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Load games and get GameList props
    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });

    const latestCall = mockGameListProps.mock.calls[mockGameListProps.mock.calls.length - 1][0] as {
      onSelectGame: (gameId: string) => void;
    };

    // Enter replay mode by selecting game-1
    await act(async () => {
      latestCall.onSelectGame('game-1');
    });

    await waitFor(() => {
      expect(mockOnReplayModeChange).toHaveBeenCalledWith(true);
    });

    // Clear previous state-change notifications
    mockOnReplayModeChange.mockClear();
    mockOnStateChange.mockClear();

    // Click Exit Replay
    await waitFor(() => screen.getByText('Exit Replay'));

    await act(async () => {
      fireEvent.click(screen.getByText('Exit Replay'));
    });

    expect(mockOnReplayModeChange).toHaveBeenCalledWith(false);
    expect(mockOnStateChange).toHaveBeenCalledWith(null);
  });

  it('clicking a move in MoveHistory jumps to that move index (via MoveInfo)', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    // Load games and get GameList props
    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });

    const latestCall = mockGameListProps.mock.calls[mockGameListProps.mock.calls.length - 1][0] as {
      onSelectGame: (gameId: string) => void;
    };

    // Enter replay mode by selecting game-1
    await act(async () => {
      latestCall.onSelectGame('game-1');
    });

    // Initial move index should be at end (2), reflected in MoveInfo header
    await waitFor(() => {
      expect(screen.getAllByText(/Move 2/i).length).toBeGreaterThan(0);
    });

    // Click first move in mocked MoveHistory (index 0 → move number 1)
    await waitFor(() => screen.getByTestId('move-0'));

    await act(async () => {
      fireEvent.click(screen.getByTestId('move-0'));
    });

    // MoveInfo should now show Move 1 as the current move
    await waitFor(() => {
      expect(screen.getAllByText(/Move 1/i).length).toBeGreaterThan(0);
    });
  });

  it('supports keyboard controls: arrow keys step moves and Escape exits replay', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });

    const latestCall = mockGameListProps.mock.calls[mockGameListProps.mock.calls.length - 1][0] as {
      onSelectGame: (gameId: string) => void;
    };

    await act(async () => {
      latestCall.onSelectGame('game-1');
    });

    await waitFor(() => {
      expect(screen.getByText('Speed:')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getAllByText(/Move 2/i).length).toBeGreaterThan(0);
    });

    await act(async () => {
      fireEvent.keyDown(window, { key: 'ArrowLeft' });
    });

    await waitFor(() => {
      expect(screen.getAllByText(/Move 1/i).length).toBeGreaterThan(0);
    });

    mockOnReplayModeChange.mockClear();
    mockOnStateChange.mockClear();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' });
    });

    expect(mockOnReplayModeChange).toHaveBeenCalledWith(false);
    expect(mockOnStateChange).toHaveBeenCalledWith(null);
  });

  it('supports Home/End keyboard shortcuts to jump to start/end of replay', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });

    const latestCall = mockGameListProps.mock.calls[mockGameListProps.mock.calls.length - 1][0] as {
      onSelectGame: (gameId: string) => void;
    };

    await act(async () => {
      latestCall.onSelectGame('game-1');
    });

    await waitFor(() => {
      expect(screen.getByText('Speed:')).toBeInTheDocument();
    });

    // Clear initial reconstruct calls so we can assert on keyboard-driven ones.
    mockReconstructStateAtMove.mockClear();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'Home' });
    });

    await waitFor(() => {
      expect(mockReconstructStateAtMove).toHaveBeenCalledWith(expect.any(Object), 0);
    });

    mockReconstructStateAtMove.mockClear();

    await act(async () => {
      fireEvent.keyDown(window, { key: 'End' });
    });

    await waitFor(() => {
      expect(mockReconstructStateAtMove).toHaveBeenCalledWith(expect.any(Object), 2);
    });
  });

  it('toggles play/pause via Space key and updates play button label', async () => {
    render(<ReplayPanel {...defaultProps} defaultCollapsed={false} />);

    await waitFor(() => {
      expect(mockGameListProps).toHaveBeenCalled();
    });

    const latestCall = mockGameListProps.mock.calls[mockGameListProps.mock.calls.length - 1][0] as {
      onSelectGame: (gameId: string) => void;
    };

    await act(async () => {
      latestCall.onSelectGame('game-1');
    });

    // Play button should initially show "Play"
    await waitFor(() => {
      expect(screen.getByLabelText('Play')).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.keyDown(window, { key: ' ' });
    });

    await waitFor(() => {
      expect(screen.getByLabelText('Pause')).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.keyDown(window, { key: ' ' });
    });

    await waitFor(() => {
      expect(screen.getByLabelText('Play')).toBeInTheDocument();
    });
  });
});
