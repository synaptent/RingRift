/**
 * ReplayPanel.branchCoverage.test.tsx
 *
 * Branch coverage tests for ReplayPanel.tsx
 */

import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ReplayPanel } from '../../../src/client/components/ReplayPanel/ReplayPanel';

// Mock hooks
const mockLoadGame = jest.fn();
const mockUnloadGame = jest.fn();
const mockStepForward = jest.fn();
const mockStepBackward = jest.fn();
const mockTogglePlay = jest.fn();
const mockJumpToStart = jest.fn();
const mockJumpToEnd = jest.fn();
const mockJumpToMove = jest.fn();
const mockSetSpeed = jest.fn();
const mockGetCurrentMove = jest.fn();

const defaultPlaybackState = {
  gameId: null,
  isLoading: false,
  error: null,
  currentState: null,
  currentMoveNumber: 0,
  totalMoves: 0,
  moves: [],
  isPlaying: false,
  playbackSpeed: 1,
  canStepForward: false,
  canStepBackward: false,
  metadata: null,
  loadGame: mockLoadGame,
  unloadGame: mockUnloadGame,
  stepForward: mockStepForward,
  stepBackward: mockStepBackward,
  togglePlay: mockTogglePlay,
  jumpToStart: mockJumpToStart,
  jumpToEnd: mockJumpToEnd,
  jumpToMove: mockJumpToMove,
  setSpeed: mockSetSpeed,
  getCurrentMove: mockGetCurrentMove,
};

let mockPlaybackState = { ...defaultPlaybackState };
let mockIsAvailable: boolean | undefined = true;
let mockIsCheckingAvailability = false;
let mockGameListData: { games: unknown[]; total: number; hasMore: boolean } | null = null;
let mockGameListLoading = false;
let mockGameListError: Error | null = null;

jest.mock('../../../src/client/hooks/useReplayService', () => ({
  useReplayServiceAvailable: () => ({
    data: mockIsAvailable,
    isLoading: mockIsCheckingAvailability,
  }),
  useGameList: () => ({
    data: mockGameListData,
    isLoading: mockGameListLoading,
    error: mockGameListError,
  }),
}));

jest.mock('../../../src/client/hooks/useReplayPlayback', () => ({
  useReplayPlayback: () => mockPlaybackState,
}));

jest.mock('../../../src/client/hooks/useReplayAnimation', () => ({
  useReplayAnimation: () => ({
    pendingAnimation: null,
  }),
}));

// Mock sub-components to simplify testing - must use require for React
jest.mock('../../../src/client/components/ReplayPanel/GameFilters', () => {
  const React = require('react');
  return {
    GameFilters: ({ onFilterChange }: { onFilterChange: (f: unknown) => void }) =>
      React.createElement(
        'div',
        { 'data-testid': 'game-filters' },
        React.createElement(
          'button',
          { onClick: () => onFilterChange({ boardType: 'square8' }) },
          'Apply Filter'
        )
      ),
  };
});

jest.mock('../../../src/client/components/ReplayPanel/GameList', () => {
  const React = require('react');
  return {
    GameList: ({
      onSelectGame,
      onPageChange,
    }: {
      onSelectGame: (id: string) => void;
      onPageChange: (offset: number) => void;
    }) =>
      React.createElement(
        'div',
        { 'data-testid': 'game-list' },
        React.createElement(
          'button',
          { onClick: () => onSelectGame('test-game-id') },
          'Select Game'
        ),
        React.createElement('button', { onClick: () => onPageChange(10) }, 'Next Page')
      ),
  };
});

jest.mock('../../../src/client/components/ReplayPanel/PlaybackControls', () => {
  const React = require('react');
  return {
    PlaybackControls: () =>
      React.createElement('div', { 'data-testid': 'playback-controls' }, 'Playback Controls'),
  };
});

jest.mock('../../../src/client/components/ReplayPanel/MoveInfo', () => {
  const React = require('react');
  return {
    MoveInfo: () => React.createElement('div', { 'data-testid': 'move-info' }, 'Move Info'),
  };
});

describe('ReplayPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPlaybackState = { ...defaultPlaybackState };
    mockIsAvailable = true;
    mockIsCheckingAvailability = false;
    mockGameListData = { games: [], total: 0, hasMore: false };
    mockGameListLoading = false;
    mockGameListError = null;
    mockLoadGame.mockResolvedValue(undefined);
  });

  describe('collapsed state', () => {
    it('renders collapsed by default', () => {
      render(<ReplayPanel />);

      expect(screen.getByText('Game Database')).toBeInTheDocument();
      expect(screen.getByText('▼ Expand')).toBeInTheDocument();
    });

    it('expands when clicking expand button', () => {
      render(<ReplayPanel />);

      fireEvent.click(screen.getByText('▼ Expand'));

      // After expanding, should show filters and game list
      expect(screen.getByTestId('game-filters')).toBeInTheDocument();
      expect(screen.getByTestId('game-list')).toBeInTheDocument();
    });

    it('starts expanded when defaultCollapsed is false', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByTestId('game-filters')).toBeInTheDocument();
    });
  });

  describe('checking availability state', () => {
    it('shows checking message while loading', () => {
      mockIsCheckingAvailability = true;
      mockIsAvailable = undefined;

      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByText('Checking replay service...')).toBeInTheDocument();
    });
  });

  describe('service unavailable state', () => {
    it('shows unavailable message when service is down', () => {
      mockIsAvailable = false;

      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByText(/Replay service unavailable/)).toBeInTheDocument();
      expect(screen.getByText(/cd ai-service/)).toBeInTheDocument();
    });

    it('allows collapsing when unavailable', () => {
      mockIsAvailable = false;

      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.click(screen.getByText('▲ Collapse'));

      expect(screen.getByText('▼ Expand')).toBeInTheDocument();
    });
  });

  describe('browse mode', () => {
    it('renders game list and filters when available', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByTestId('game-filters')).toBeInTheDocument();
      expect(screen.getByTestId('game-list')).toBeInTheDocument();
    });

    it('handles game selection', async () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Select Game'));
      });

      expect(mockLoadGame).toHaveBeenCalledWith('test-game-id');
    });

    it('handles page change', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.click(screen.getByText('Next Page'));

      // Filters should be updated with new offset
      // This is internal state, so we just verify no errors
    });

    it('allows collapsing in browse mode', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.click(screen.getByText('▲ Collapse'));

      expect(screen.getByText('▼ Expand')).toBeInTheDocument();
    });
  });

  describe('replay mode', () => {
    beforeEach(() => {
      mockPlaybackState = {
        ...defaultPlaybackState,
        gameId: 'test-game-123',
        currentMoveNumber: 5,
        totalMoves: 20,
        isPlaying: false,
        canStepForward: true,
        canStepBackward: true,
        metadata: {
          gameId: 'test-game-123456789012',
          boardType: 'square8',
          numPlayers: 2,
          winner: 1,
          totalMoves: 20,
        },
        currentState: {
          id: 'test',
          boardType: 'square8',
          board: {
            stacks: new Map(),
            markers: new Map(),
            collapsedSpaces: new Map(),
            territories: new Map(),
            formedLines: [],
            eliminatedRings: {},
            size: 8,
            type: 'square8',
          },
          players: [],
          currentPhase: 'ring_placement',
          currentPlayer: 1,
          moveHistory: [],
          history: [],
          timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
          spectators: [],
          gameStatus: 'active',
          createdAt: new Date(),
          lastMoveAt: new Date(),
          isRated: false,
          maxPlayers: 2,
          totalRingsInPlay: 36,
          totalRingsEliminated: 0,
          victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
          territoryVictoryThreshold: 33,
        },
      };
    });

    it('renders replay mode UI when game is loaded', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByText('Replay Mode')).toBeInTheDocument();
      expect(screen.getByTestId('playback-controls')).toBeInTheDocument();
      expect(screen.getByTestId('move-info')).toBeInTheDocument();
    });

    it('shows game metadata', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      // Game ID is truncated to 12 chars in display
      expect(screen.getByText(/test-game-12/)).toBeInTheDocument();
      expect(screen.getByText('square8')).toBeInTheDocument();
      // Player count "2P" rendered as text
      expect(screen.getByText(/2.*P/)).toBeInTheDocument();
      // Winner info is rendered
      expect(screen.getByText(/Winner/)).toBeInTheDocument();
    });

    it('shows error when present', () => {
      mockPlaybackState = {
        ...mockPlaybackState,
        error: 'Failed to load game',
      };

      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByText('Failed to load game')).toBeInTheDocument();
    });

    it('handles close replay', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.click(screen.getByText('✕ Close'));

      expect(mockUnloadGame).toHaveBeenCalled();
    });

    it('renders fork button when callback provided', () => {
      const onFork = jest.fn();

      render(<ReplayPanel defaultCollapsed={false} onForkFromPosition={onFork} />);

      expect(screen.getByText('Fork from this position')).toBeInTheDocument();
    });

    it('handles fork action', () => {
      const onFork = jest.fn();

      render(<ReplayPanel defaultCollapsed={false} onForkFromPosition={onFork} />);

      fireEvent.click(screen.getByText('Fork from this position'));

      expect(onFork).toHaveBeenCalled();
      expect(mockUnloadGame).toHaveBeenCalled();
    });

    it('shows keyboard hints', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      expect(screen.getByText(/← → Step/)).toBeInTheDocument();
      expect(screen.getByText(/Home\/End Jump/)).toBeInTheDocument();
    });
  });

  describe('keyboard shortcuts', () => {
    beforeEach(() => {
      mockPlaybackState = {
        ...defaultPlaybackState,
        gameId: 'test-game-123',
        currentMoveNumber: 5,
        totalMoves: 20,
        playbackSpeed: 1,
      };
    });

    it('handles ArrowLeft for step backward', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'ArrowLeft' });

      expect(mockStepBackward).toHaveBeenCalled();
    });

    it('handles ArrowRight for step forward', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'ArrowRight' });

      expect(mockStepForward).toHaveBeenCalled();
    });

    it('handles h for step backward', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'h' });

      expect(mockStepBackward).toHaveBeenCalled();
    });

    it('handles l for step forward', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'l' });

      expect(mockStepForward).toHaveBeenCalled();
    });

    it('handles Space for toggle play', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: ' ' });

      expect(mockTogglePlay).toHaveBeenCalled();
    });

    it('handles Home for jump to start', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'Home' });

      expect(mockJumpToStart).toHaveBeenCalled();
    });

    it('handles End for jump to end', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'End' });

      expect(mockJumpToEnd).toHaveBeenCalled();
    });

    it('handles 0 for jump to start', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: '0' });

      expect(mockJumpToStart).toHaveBeenCalled();
    });

    it('handles $ for jump to end', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: '$' });

      expect(mockJumpToEnd).toHaveBeenCalled();
    });

    it('handles [ for speed decrease', () => {
      mockPlaybackState.playbackSpeed = 2;

      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: '[' });

      expect(mockSetSpeed).toHaveBeenCalledWith(1);
    });

    it('handles ] for speed increase', () => {
      mockPlaybackState.playbackSpeed = 1;

      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: ']' });

      expect(mockSetSpeed).toHaveBeenCalledWith(2);
    });

    it('handles Escape to close replay', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      fireEvent.keyDown(window, { key: 'Escape' });

      expect(mockUnloadGame).toHaveBeenCalled();
    });

    it('ignores keyboard shortcuts when typing in input', () => {
      render(<ReplayPanel defaultCollapsed={false} />);

      const mockInput = document.createElement('input');
      document.body.appendChild(mockInput);
      mockInput.focus();

      fireEvent.keyDown(mockInput, { key: 'ArrowLeft', target: mockInput });

      expect(mockStepBackward).not.toHaveBeenCalled();

      document.body.removeChild(mockInput);
    });
  });

  describe('callbacks', () => {
    it('calls onStateChange when state changes', () => {
      const onStateChange = jest.fn();
      mockPlaybackState = {
        ...defaultPlaybackState,
        gameId: 'test',
        currentState: { id: 'test' } as any,
      };

      render(<ReplayPanel defaultCollapsed={false} onStateChange={onStateChange} />);

      expect(onStateChange).toHaveBeenCalled();
    });

    it('calls onReplayModeChange when entering replay mode', () => {
      const onReplayModeChange = jest.fn();
      mockPlaybackState = {
        ...defaultPlaybackState,
        gameId: 'test-game',
      };

      render(<ReplayPanel defaultCollapsed={false} onReplayModeChange={onReplayModeChange} />);

      expect(onReplayModeChange).toHaveBeenCalledWith(true);
    });
  });

  describe('auto-load requested game', () => {
    it('auto-loads when requestedGameId is provided', async () => {
      render(<ReplayPanel defaultCollapsed={true} requestedGameId="requested-game-123" />);

      // Should expand and attempt to load
      await act(async () => {
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockLoadGame).toHaveBeenCalledWith('requested-game-123');
    });

    it('shows error when auto-load fails', async () => {
      mockLoadGame.mockRejectedValue(new Error('Game not found'));
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      render(<ReplayPanel defaultCollapsed={true} requestedGameId="missing-game" />);

      await act(async () => {
        await new Promise((r) => setTimeout(r, 10));
      });

      // Error should be logged
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('does not auto-load when service unavailable', async () => {
      mockIsAvailable = false;

      render(<ReplayPanel defaultCollapsed={true} requestedGameId="requested-game" />);

      await act(async () => {
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockLoadGame).not.toHaveBeenCalled();
    });
  });

  describe('className prop', () => {
    it('applies custom className', () => {
      const { container } = render(<ReplayPanel className="custom-class" />);

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
