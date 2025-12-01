import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MoveHistory, MoveHistoryFromEntries } from '../../../src/client/components/MoveHistory';
import type { Move, BoardType, GameHistoryEntry, GameState } from '../../../src/shared/types/game';

// Mock scrollIntoView which is not available in jsdom
beforeAll(() => {
  Element.prototype.scrollIntoView = jest.fn();
});

describe('MoveHistory', () => {
  const boardType: BoardType = 'square8';

  // Helper to create test moves
  function createTestMove(
    index: number,
    type: Move['type'],
    player: number,
    overrides: Partial<Move> = {}
  ): Move {
    return {
      id: `move-${index}`,
      type,
      player,
      to: { x: index % 8, y: Math.floor(index / 8) },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: index + 1,
      ...overrides,
    } as Move;
  }

  describe('empty state', () => {
    it('should display empty message when no moves', () => {
      render(<MoveHistory moves={[]} boardType={boardType} />);

      expect(screen.getByTestId('move-history')).toBeInTheDocument();
      expect(screen.getByText('No moves yet.')).toBeInTheDocument();
    });

    it('should display header even when empty', () => {
      render(<MoveHistory moves={[]} boardType={boardType} />);

      expect(screen.getByText('Moves')).toBeInTheDocument();
    });
  });

  describe('with moves', () => {
    it('should display move count', () => {
      const moves = [
        createTestMove(0, 'place_ring', 1),
        createTestMove(1, 'place_ring', 2),
        createTestMove(2, 'place_ring', 1),
      ];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByText('3 total')).toBeInTheDocument();
    });

    it('should display move numbers', () => {
      const moves = [createTestMove(0, 'place_ring', 1), createTestMove(1, 'place_ring', 2)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByText('1.')).toBeInTheDocument();
      expect(screen.getByText('2.')).toBeInTheDocument();
    });

    it('should highlight current move', () => {
      const moves = [
        createTestMove(0, 'place_ring', 1),
        createTestMove(1, 'place_ring', 2),
        createTestMove(2, 'place_ring', 1),
      ];

      render(<MoveHistory moves={moves} boardType={boardType} currentMoveIndex={1} />);

      // Find the button for move 2 (index 1)
      const moveButtons = screen.getAllByRole('button');
      expect(moveButtons[1]).toHaveAttribute('aria-current', 'true');
    });

    it('should default current move to last move', () => {
      const moves = [createTestMove(0, 'place_ring', 1), createTestMove(1, 'place_ring', 2)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      // Last move should be current when currentMoveIndex is not specified
      const moveButtons = screen.getAllByRole('button');
      expect(moveButtons[1]).toHaveAttribute('aria-current', 'true');
    });
  });

  describe('move types', () => {
    it('should display ring placement notation', () => {
      const moves = [createTestMove(0, 'place_ring', 1, { to: { x: 0, y: 0 } })];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      // Should contain the ring symbol
      expect(screen.getByText(/◯/)).toBeInTheDocument();
    });

    it('should display movement notation', () => {
      const moves = [
        createTestMove(0, 'move_stack', 1, {
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
        }),
      ];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      // Should contain arrow symbol
      expect(screen.getByText(/→/)).toBeInTheDocument();
    });

    it('should display capture notation', () => {
      const moves = [
        createTestMove(0, 'overtaking_capture', 1, {
          from: { x: 0, y: 0 },
          to: { x: 2, y: 2 },
          captureTarget: { x: 1, y: 1 },
        }),
      ];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      // Should contain cross symbol for capture
      expect(screen.getByText(/×/)).toBeInTheDocument();
    });

    it('should display swap notation', () => {
      const moves = [createTestMove(0, 'swap_sides', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByText(/Swap/)).toBeInTheDocument();
    });

    it('should display skip notation', () => {
      const moves = [createTestMove(0, 'skip_placement', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByText(/Pass/)).toBeInTheDocument();
    });

    it('should display line processing notation', () => {
      const moves = [createTestMove(0, 'process_line', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByText(/Line/)).toBeInTheDocument();
    });

    it('should display territory processing notation', () => {
      const moves = [createTestMove(0, 'process_territory_region', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByText(/Territory/)).toBeInTheDocument();
    });
  });

  describe('player colors', () => {
    it('should display player 1 indicator', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      const playerIndicator = screen.getByLabelText('Player 1');
      expect(playerIndicator).toBeInTheDocument();
      expect(playerIndicator).toHaveClass('bg-emerald-500');
    });

    it('should display player 2 indicator', () => {
      const moves = [createTestMove(0, 'place_ring', 2)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      const playerIndicator = screen.getByLabelText('Player 2');
      expect(playerIndicator).toBeInTheDocument();
      expect(playerIndicator).toHaveClass('bg-sky-500');
    });
  });

  describe('click handler', () => {
    it('should call onMoveClick when move is clicked', () => {
      const mockOnMoveClick = jest.fn();
      const moves = [createTestMove(0, 'place_ring', 1), createTestMove(1, 'place_ring', 2)];

      render(<MoveHistory moves={moves} boardType={boardType} onMoveClick={mockOnMoveClick} />);

      const moveButtons = screen.getAllByRole('button');
      fireEvent.click(moveButtons[0]);

      expect(mockOnMoveClick).toHaveBeenCalledWith(0);
    });

    it('should call onMoveClick with correct index', () => {
      const mockOnMoveClick = jest.fn();
      const moves = [
        createTestMove(0, 'place_ring', 1),
        createTestMove(1, 'place_ring', 2),
        createTestMove(2, 'place_ring', 1),
      ];

      render(<MoveHistory moves={moves} boardType={boardType} onMoveClick={mockOnMoveClick} />);

      const moveButtons = screen.getAllByRole('button');
      fireEvent.click(moveButtons[2]);

      expect(mockOnMoveClick).toHaveBeenCalledWith(2);
    });

    it('should disable buttons when no onClick handler', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      const moveButton = screen.getByRole('button');
      expect(moveButton).toBeDisabled();
    });
  });

  describe('accessibility', () => {
    it('should have list role', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByRole('list')).toBeInTheDocument();
    });

    it('should have listitem role for each move', () => {
      const moves = [createTestMove(0, 'place_ring', 1), createTestMove(1, 'place_ring', 2)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      const listItems = screen.getAllByRole('listitem');
      expect(listItems).toHaveLength(2);
    });

    it('should have aria-label on move list', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(screen.getByRole('list')).toHaveAttribute('aria-label', 'Move history');
    });
  });

  describe('styling props', () => {
    it('should apply custom className', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} className="custom-class" />);

      expect(screen.getByTestId('move-history')).toHaveClass('custom-class');
    });

    it('should apply default maxHeight', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} />);

      const list = screen.getByRole('list');
      expect(list).toHaveClass('max-h-48');
    });

    it('should apply custom maxHeight', () => {
      const moves = [createTestMove(0, 'place_ring', 1)];

      render(<MoveHistory moves={moves} boardType={boardType} maxHeight="max-h-96" />);

      const list = screen.getByRole('list');
      expect(list).toHaveClass('max-h-96');
    });
  });
});

describe('MoveHistoryFromEntries', () => {
  const boardType: BoardType = 'square8';

  function createTestEntry(index: number, player: number): GameHistoryEntry {
    const move: Move = {
      id: `move-${index}`,
      type: 'place_ring',
      player,
      to: { x: index % 8, y: Math.floor(index / 8) },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: index + 1,
    };

    return {
      action: move,
      stateBefore: {} as GameState,
      stateAfter: {} as GameState,
    };
  }

  it('should extract moves from entries and display them', () => {
    const entries = [createTestEntry(0, 1), createTestEntry(1, 2)];

    render(<MoveHistoryFromEntries entries={entries} boardType={boardType} />);

    expect(screen.getByText('2 total')).toBeInTheDocument();
    expect(screen.getByText('1.')).toBeInTheDocument();
    expect(screen.getByText('2.')).toBeInTheDocument();
  });

  it('should pass through all props to MoveHistory', () => {
    const entries = [createTestEntry(0, 1)];
    const mockOnMoveClick = jest.fn();

    render(
      <MoveHistoryFromEntries
        entries={entries}
        boardType={boardType}
        currentMoveIndex={0}
        onMoveClick={mockOnMoveClick}
        maxHeight="max-h-64"
        className="custom-class"
      />
    );

    expect(screen.getByTestId('move-history')).toHaveClass('custom-class');
    expect(screen.getByRole('list')).toHaveClass('max-h-64');
  });
});
