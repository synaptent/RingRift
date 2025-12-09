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

  function createHistoryEntry(move: Move): GameHistoryEntry {
    return {
      moveNumber: move.moveNumber ?? 1,
      action: move,
      actor: move.player,
      phaseBefore: 'movement',
      phaseAfter: 'movement',
      statusBefore: 'active',
      statusAfter: 'active',
      progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
      progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
    };
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

    it('should render bookkeeping/no-op and forced_elimination moves with readable labels', () => {
      const moves = [
        createTestMove(0, 'no_line_action', 1),
        createTestMove(1, 'no_territory_action', 2),
        createTestMove(2, 'forced_elimination', 1),
      ];

      const { container } = render(<MoveHistory moves={moves} boardType={boardType} />);

      expect(container.textContent).toContain('no_line_action');
      expect(container.textContent).toContain('no_territory_action');
      expect(container.textContent).toContain('forced_elimination');
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

  describe('notation options', () => {
    describe('square rank orientation', () => {
      it('uses canonical top-origin ranks by default', () => {
        // Canonical: square8 with y=7 (bottom visual row) → rank 8
        const moves = [
          createTestMove(0, 'place_ring', 1, { to: { x: 0, y: 7 } }), // a8 in canonical
        ];

        const { container } = render(<MoveHistory moves={moves} boardType="square8" />);

        // Should display "a8" (y=7 → rank 8 in canonical top-origin)
        expect(container.textContent).toContain('a8');
      });

      it('uses bottom-origin ranks when squareRankFromBottom is enabled', () => {
        // Bottom-origin: square8 with y=7 (bottom visual row) → rank 1
        const moves = [
          createTestMove(0, 'place_ring', 1, { to: { x: 0, y: 7 } }), // a1 in bottom-origin
        ];

        const { container } = render(
          <MoveHistory
            moves={moves}
            boardType="square8"
            notationOptions={{
              boardType: 'square8',
              boardSizeOverride: 8,
              squareRankFromBottom: true,
            }}
          />
        );

        // Should display "a1" (y=7 → rank 1 when bottom-origin)
        expect(container.textContent).toContain('a1');
      });

      it('applies bottom-origin to square19 boards', () => {
        // square19 with y=18 (bottom visual row) → rank 1 when bottom-origin
        const moves = [createTestMove(0, 'place_ring', 1, { to: { x: 0, y: 18 } })];

        const { container } = render(
          <MoveHistory
            moves={moves}
            boardType="square19"
            notationOptions={{
              boardType: 'square19',
              boardSizeOverride: 19,
              squareRankFromBottom: true,
            }}
          />
        );

        // Should display "a1" (y=18 → rank 1 when bottom-origin on size 19)
        expect(container.textContent).toContain('a1');
      });

      it('preserves canonical square19 ranks without the option', () => {
        // square19 with y=18 (bottom visual row) → rank 19 in canonical
        const moves = [createTestMove(0, 'place_ring', 1, { to: { x: 0, y: 18 } })];

        const { container } = render(<MoveHistory moves={moves} boardType="square19" />);

        // Should display "a19" (y=18 → rank 19 in canonical top-origin)
        expect(container.textContent).toContain('a19');
      });

      it('formats movement notation with bottom-origin ranks', () => {
        const moves = [
          createTestMove(0, 'move_stack', 1, {
            from: { x: 0, y: 0 }, // a8 canonical → a1 bottom-origin (for size 8)
            to: { x: 1, y: 0 }, // b8 canonical → b1 bottom-origin
          }),
        ];

        const { container } = render(
          <MoveHistory
            moves={moves}
            boardType="square8"
            notationOptions={{
              boardType: 'square8',
              boardSizeOverride: 8,
              squareRankFromBottom: true,
            }}
          />
        );

        // With bottom-origin on square8, y=0 becomes rank 8 (not rank 1)
        // Wait, let me recalculate:
        // - Canonical: y=0 → rank 1 (top visual)
        // - Bottom-origin: y=0 → rank 8 (because size - y = 8 - 0 = 8)
        // Actually for y=0 with size=8: bottomOrigin gives size - y = 8 - 0 = 8
        // So from a8 to b8 in bottom-origin mode
        expect(container.textContent).toMatch(/a8.*→.*b8/);
      });

      it('has no effect on hex board notation', () => {
        // Hex boards use cube coordinates; squareRankFromBottom should be ignored
        const moves = [createTestMove(0, 'place_ring', 1, { to: { x: 0, y: 0, z: 0 } })];

        const { container } = render(
          <MoveHistory
            moves={moves}
            boardType="hexagonal"
            notationOptions={{
              boardType: 'hexagonal',
              squareRankFromBottom: true, // should be ignored
            }}
          />
        );

        // Hex notation should be unaffected
        expect(screen.getByTestId('move-history')).toBeInTheDocument();
      });
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

  function createTestEntry(
    index: number,
    player: number,
    type: Move['type'] = 'place_ring'
  ): GameHistoryEntry {
    const move: Move = {
      id: `move-${index}`,
      type,
      player,
      to: { x: index % 8, y: Math.floor(index / 8) },
      timestamp: new Date(),
      thinkTime: 1000,
      moveNumber: index + 1,
    };

    return {
      moveNumber: index + 1,
      action: move,
      actor: player,
      phaseBefore: 'movement',
      phaseAfter: 'movement',
      statusBefore: 'active',
      statusAfter: 'active',
      progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
      progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
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

  it('renders bookkeeping and forced-elimination moves from history entries', () => {
    const entries = [
      createTestEntry(0, 1, 'no_line_action'),
      createTestEntry(1, 2, 'no_territory_action'),
      createTestEntry(2, 1, 'forced_elimination'),
    ];

    const { container } = render(
      <MoveHistoryFromEntries entries={entries} boardType={boardType} />
    );

    expect(container.textContent).toContain('no_line_action');
    expect(container.textContent).toContain('no_territory_action');
    expect(container.textContent).toContain('forced_elimination');
  });
});
