import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { GameList } from '../../../src/client/components/ReplayPanel/GameList';
import type { ReplayGameMetadata } from '../../../src/client/types/replay';

describe('GameList', () => {
  const mockGame: ReplayGameMetadata = {
    gameId: 'abc12345-6789-0123-4567-890abcdef012',
    boardType: 'square8',
    numPlayers: 2,
    winner: 0,
    totalMoves: 42,
    terminationReason: 'ring_elimination',
    createdAt: '2024-12-13T10:30:00Z',
    source: 'self_play',
  };

  const defaultProps = {
    games: [mockGame],
    selectedGameId: null,
    onSelectGame: jest.fn(),
    total: 1,
    offset: 0,
    limit: 10,
    hasMore: false,
    onPageChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders game list with games', () => {
      render(<GameList {...defaultProps} />);

      // Game ID prefix should be visible
      expect(screen.getByText('abc12345')).toBeInTheDocument();
    });

    it('renders board type badge', () => {
      render(<GameList {...defaultProps} />);

      expect(screen.getByText('8×8')).toBeInTheDocument();
    });

    it('renders player count', () => {
      render(<GameList {...defaultProps} />);

      expect(screen.getByText('2P')).toBeInTheDocument();
    });

    it('renders winner information', () => {
      render(<GameList {...defaultProps} />);

      expect(screen.getByText('P0 won')).toBeInTheDocument();
    });

    it('renders move count', () => {
      render(<GameList {...defaultProps} />);

      expect(screen.getByText('42 moves')).toBeInTheDocument();
    });

    it('renders termination reason', () => {
      render(<GameList {...defaultProps} />);

      expect(screen.getByText('Ring Elim.')).toBeInTheDocument();
    });

    it('renders formatted date', () => {
      render(<GameList {...defaultProps} />);

      // Date format depends on locale, but should contain the month
      // Get the game row button (not pagination buttons)
      const gameButtons = screen.getAllByRole('button');
      const gameButton = gameButtons.find((btn) => btn.textContent?.includes('abc12345'));
      expect(gameButton).toHaveTextContent(/Dec/i);
    });
  });

  describe('board type formatting', () => {
    it('formats square8 as 8×8', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, boardType: 'square8' }]} />);

      expect(screen.getByText('8×8')).toBeInTheDocument();
    });

    it('formats square19 as 19×19', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, boardType: 'square19' }]} />);

      expect(screen.getByText('19×19')).toBeInTheDocument();
    });

    it('formats hexagonal as Hex', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, boardType: 'hexagonal' }]} />);

      expect(screen.getByText('Hex')).toBeInTheDocument();
    });

    it('shows unknown board types as-is', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, boardType: 'custom_board' }]} />);

      expect(screen.getByText('custom_board')).toBeInTheDocument();
    });
  });

  describe('termination reason formatting', () => {
    it('formats ring_elimination', () => {
      render(
        <GameList
          {...defaultProps}
          games={[{ ...mockGame, terminationReason: 'ring_elimination' }]}
        />
      );

      expect(screen.getByText('Ring Elim.')).toBeInTheDocument();
    });

    it('formats territory', () => {
      render(
        <GameList {...defaultProps} games={[{ ...mockGame, terminationReason: 'territory' }]} />
      );

      expect(screen.getByText('Territory')).toBeInTheDocument();
    });

    it('formats last_player_standing', () => {
      render(
        <GameList
          {...defaultProps}
          games={[{ ...mockGame, terminationReason: 'last_player_standing' }]}
        />
      );

      expect(screen.getByText('LPS')).toBeInTheDocument();
    });

    it('formats stalemate', () => {
      render(
        <GameList {...defaultProps} games={[{ ...mockGame, terminationReason: 'stalemate' }]} />
      );

      expect(screen.getByText('Stalemate')).toBeInTheDocument();
    });

    it('shows dash for null termination reason', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, terminationReason: null }]} />);

      expect(screen.getByText('—')).toBeInTheDocument();
    });
  });

  describe('selection', () => {
    it('calls onSelectGame when game is clicked', async () => {
      const onSelectGame = jest.fn();
      const user = userEvent.setup();
      render(<GameList {...defaultProps} onSelectGame={onSelectGame} />);

      // Get the game row button (not pagination buttons)
      const gameButtons = screen.getAllByRole('button');
      const gameButton = gameButtons.find((btn) => btn.textContent?.includes('abc12345'))!;
      await user.click(gameButton);

      expect(onSelectGame).toHaveBeenCalledWith(mockGame.gameId);
    });

    it('applies selected styling to selected game', () => {
      render(<GameList {...defaultProps} selectedGameId={mockGame.gameId} />);

      // Get the game row button (not pagination buttons)
      const gameButtons = screen.getAllByRole('button');
      const gameButton = gameButtons.find((btn) => btn.textContent?.includes('abc12345'));
      expect(gameButton).toHaveClass('bg-emerald-900/40');
      expect(gameButton).toHaveClass('border-emerald-500/50');
    });

    it('applies default styling to non-selected game', () => {
      render(<GameList {...defaultProps} selectedGameId={null} />);

      // Get the game row button (not pagination buttons)
      const gameButtons = screen.getAllByRole('button');
      const gameButton = gameButtons.find((btn) => btn.textContent?.includes('abc12345'));
      expect(gameButton).toHaveClass('bg-slate-800/60');
      expect(gameButton).not.toHaveClass('bg-emerald-900/40');
    });
  });

  describe('loading state', () => {
    it('shows loading message when loading with no games', () => {
      render(<GameList {...defaultProps} games={[]} isLoading={true} total={0} />);

      expect(screen.getByText('Loading games...')).toBeInTheDocument();
    });

    it('shows games when loading with existing games', () => {
      render(<GameList {...defaultProps} isLoading={true} />);

      expect(screen.getByText('abc12345')).toBeInTheDocument();
      expect(screen.queryByText('Loading games...')).not.toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when error prop is set', () => {
      render(<GameList {...defaultProps} error="Failed to load games" />);

      expect(screen.getByText('Error: Failed to load games')).toBeInTheDocument();
    });

    it('does not show games when error is present', () => {
      render(<GameList {...defaultProps} error="Failed to load games" />);

      expect(screen.queryByText('abc12345')).not.toBeInTheDocument();
    });
  });

  describe('empty state', () => {
    it('shows empty message when no games', () => {
      render(<GameList {...defaultProps} games={[]} total={0} />);

      expect(
        screen.getByText('No games found. Try adjusting your filters or run some self-play games.')
      ).toBeInTheDocument();
    });
  });

  describe('pagination', () => {
    it('shows pagination info', () => {
      render(<GameList {...defaultProps} total={25} offset={0} limit={10} />);

      expect(screen.getByText('1–1 of 25')).toBeInTheDocument();
    });

    it('shows current page and total pages', () => {
      render(<GameList {...defaultProps} total={25} offset={0} limit={10} />);

      expect(screen.getByText('1 / 3')).toBeInTheDocument();
    });

    it('shows 0 games text when total is 0', () => {
      render(<GameList {...defaultProps} games={[]} total={0} />);

      // Empty state message is shown instead
      expect(screen.getByText(/No games found/)).toBeInTheDocument();
    });

    it('renders previous page button', () => {
      render(<GameList {...defaultProps} total={25} offset={10} limit={10} hasMore={true} />);

      expect(screen.getByRole('button', { name: /Previous page/i })).toBeInTheDocument();
    });

    it('renders next page button', () => {
      render(<GameList {...defaultProps} total={25} offset={0} limit={10} hasMore={true} />);

      expect(screen.getByRole('button', { name: /Next page/i })).toBeInTheDocument();
    });

    it('disables previous button on first page', () => {
      render(<GameList {...defaultProps} total={25} offset={0} limit={10} />);

      expect(screen.getByRole('button', { name: /Previous page/i })).toBeDisabled();
    });

    it('disables next button when no more pages', () => {
      render(<GameList {...defaultProps} total={10} offset={0} limit={10} hasMore={false} />);

      expect(screen.getByRole('button', { name: /Next page/i })).toBeDisabled();
    });

    it('calls onPageChange with previous offset when previous clicked', async () => {
      const onPageChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameList
          {...defaultProps}
          total={25}
          offset={10}
          limit={10}
          hasMore={true}
          onPageChange={onPageChange}
        />
      );

      await user.click(screen.getByRole('button', { name: /Previous page/i }));

      expect(onPageChange).toHaveBeenCalledWith(0);
    });

    it('calls onPageChange with next offset when next clicked', async () => {
      const onPageChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameList
          {...defaultProps}
          total={25}
          offset={0}
          limit={10}
          hasMore={true}
          onPageChange={onPageChange}
        />
      );

      await user.click(screen.getByRole('button', { name: /Next page/i }));

      expect(onPageChange).toHaveBeenCalledWith(10);
    });

    it('does not go below offset 0', async () => {
      const onPageChange = jest.fn();
      const user = userEvent.setup();
      render(
        <GameList
          {...defaultProps}
          total={25}
          offset={5}
          limit={10}
          hasMore={true}
          onPageChange={onPageChange}
        />
      );

      await user.click(screen.getByRole('button', { name: /Previous page/i }));

      expect(onPageChange).toHaveBeenCalledWith(0);
    });
  });

  describe('multiple games', () => {
    it('renders all games in the list', () => {
      const games = [
        { ...mockGame, gameId: 'game1-xxx', totalMoves: 10 },
        { ...mockGame, gameId: 'game2-xxx', totalMoves: 20 },
        { ...mockGame, gameId: 'game3-xxx', totalMoves: 30 },
      ];
      render(<GameList {...defaultProps} games={games} total={3} />);

      expect(screen.getByText('game1-xx')).toBeInTheDocument();
      expect(screen.getByText('game2-xx')).toBeInTheDocument();
      expect(screen.getByText('game3-xx')).toBeInTheDocument();
    });

    it('only highlights the selected game', () => {
      const games = [
        { ...mockGame, gameId: 'game1-xxx' },
        { ...mockGame, gameId: 'game2-xxx' },
      ];
      render(<GameList {...defaultProps} games={games} total={2} selectedGameId="game1-xxx" />);

      const buttons = screen.getAllByRole('button');
      expect(buttons[0]).toHaveClass('bg-emerald-900/40');
      expect(buttons[1]).not.toHaveClass('bg-emerald-900/40');
    });
  });

  describe('winner display', () => {
    it('shows winner when present', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, winner: 1 }]} />);

      expect(screen.getByText('P1 won')).toBeInTheDocument();
    });

    it('does not show winner when null', () => {
      render(<GameList {...defaultProps} games={[{ ...mockGame, winner: null }]} />);

      expect(screen.queryByText(/won/)).not.toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      const { container } = render(<GameList {...defaultProps} className="custom-class" />);

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
