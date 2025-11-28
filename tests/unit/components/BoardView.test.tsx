import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import { BoardState, Position, RingStack } from '../../../src/shared/types/game';

// Helper to create empty board state
function createEmptyBoardState(type: 'square8' | 'square19' | 'hexagonal' = 'square8'): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: type === 'square8' ? 8 : type === 'square19' ? 19 : 11,
    type,
  };
}

// Helper to create a board with stacks
function createBoardWithStacks(positions: Array<{ pos: Position; rings: number[] }>): BoardState {
  const board = createEmptyBoardState('square8');
  positions.forEach(({ pos, rings }) => {
    const key = `${pos.x},${pos.y}`;
    board.stacks.set(key, {
      position: pos,
      rings,
      stackHeight: rings.length,
      capHeight: 1,
      controllingPlayer: rings[0] ?? 1,
    });
  });
  return board;
}

describe('BoardView', () => {
  describe('rendering', () => {
    it('renders without crashing', () => {
      const board = createEmptyBoardState('square8');
      render(<BoardView boardType="square8" board={board} />);
      expect(screen.getByTestId('board-view')).toBeInTheDocument();
    });

    it('renders square8 board correctly', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Should have 8x8 = 64 cells
      const buttons = container.querySelectorAll('button');
      expect(buttons).toHaveLength(64);
    });

    it('renders square19 board correctly', () => {
      const board = createEmptyBoardState('square19');
      board.size = 19;
      const { container } = render(<BoardView boardType="square19" board={board} />);

      // Should have 19x19 = 361 cells
      const buttons = container.querySelectorAll('button');
      expect(buttons).toHaveLength(361);
    });

    it('renders hexagonal board correctly', () => {
      const board = createEmptyBoardState('hexagonal');
      board.size = 11;
      board.type = 'hexagonal';
      const { container } = render(<BoardView boardType="hexagonal" board={board} />);

      // Hex board has variable cell count based on radius
      const buttons = container.querySelectorAll('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('returns null for unknown board type', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType={'unknown' as any} board={board} />);
      expect(container.firstChild).toBeNull();
    });
  });

  describe('click handling', () => {
    it('calls onCellClick when a cell is clicked', () => {
      const board = createEmptyBoardState('square8');
      const handleCellClick = jest.fn();

      render(<BoardView boardType="square8" board={board} onCellClick={handleCellClick} />);

      const cells = screen.getByTestId('board-view').querySelectorAll('button');
      fireEvent.click(cells[0]);

      expect(handleCellClick).toHaveBeenCalledTimes(1);
      expect(handleCellClick).toHaveBeenCalledWith(expect.objectContaining({ x: 0, y: 0 }));
    });

    it('calls onCellDoubleClick when a cell is double-clicked', () => {
      const board = createEmptyBoardState('square8');
      const handleDoubleClick = jest.fn();

      render(<BoardView boardType="square8" board={board} onCellDoubleClick={handleDoubleClick} />);

      const cells = screen.getByTestId('board-view').querySelectorAll('button');
      fireEvent.doubleClick(cells[0]);

      expect(handleDoubleClick).toHaveBeenCalledTimes(1);
    });

    it('calls onCellContextMenu on right-click', () => {
      const board = createEmptyBoardState('square8');
      const handleContextMenu = jest.fn();

      render(<BoardView boardType="square8" board={board} onCellContextMenu={handleContextMenu} />);

      const cells = screen.getByTestId('board-view').querySelectorAll('button');
      fireEvent.contextMenu(cells[0]);

      expect(handleContextMenu).toHaveBeenCalledTimes(1);
    });

    it('does not call click handlers when spectating', () => {
      const board = createEmptyBoardState('square8');
      const handleCellClick = jest.fn();

      render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={handleCellClick}
          isSpectator={true}
        />
      );

      const cells = screen.getByTestId('board-view').querySelectorAll('button');
      fireEvent.click(cells[0]);

      expect(handleCellClick).not.toHaveBeenCalled();
    });
  });

  describe('selected position', () => {
    it('highlights selected position', () => {
      const board = createEmptyBoardState('square8');
      const selectedPosition: Position = { x: 3, y: 3 };

      const { container } = render(
        <BoardView boardType="square8" board={board} selectedPosition={selectedPosition} />
      );

      // Check that selected cell has ring highlighting
      const selectedCell = container.querySelector('.ring-emerald-400');
      expect(selectedCell).toBeInTheDocument();
    });
  });

  describe('valid targets', () => {
    it('highlights valid target positions', () => {
      const board = createEmptyBoardState('square8');
      const validTargets: Position[] = [
        { x: 2, y: 2 },
        { x: 4, y: 4 },
      ];

      const { container } = render(
        <BoardView boardType="square8" board={board} validTargets={validTargets} />
      );

      // Check that valid targets have outline styling
      const highlightedCells = container.querySelectorAll('.outline-emerald-300\\/90');
      expect(highlightedCells.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('stacks rendering', () => {
    it('renders ring stacks on the board', () => {
      const board = createBoardWithStacks([{ pos: { x: 3, y: 3 }, rings: [1, 2] }]);

      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Stack should display height and cap info
      expect(container.textContent).toContain('H2');
      expect(container.textContent).toContain('C1');
    });
  });

  describe('markers rendering', () => {
    it('renders markers on the board', () => {
      const board = createEmptyBoardState('square8');
      board.markers.set('4,4', { type: 'regular', player: 1, position: { x: 4, y: 4 } });

      render(<BoardView boardType="square8" board={board} />);

      // Marker should be visible (checking for marker styling)
      const markerElement = document.querySelector('.border-emerald-400');
      expect(markerElement).toBeInTheDocument();
    });
  });

  describe('collapsed spaces', () => {
    it('renders collapsed territory spaces with player colors', () => {
      const board = createEmptyBoardState('square8');
      board.collapsedSpaces.set('5,5', 1);

      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Territory should have player-specific background
      const territoryCell = container.querySelector('.bg-emerald-700\\/85');
      expect(territoryCell).toBeInTheDocument();
    });
  });

  describe('coordinate labels', () => {
    it('shows coordinate labels when enabled', () => {
      const board = createEmptyBoardState('square8');

      const { container } = render(
        <BoardView boardType="square8" board={board} showCoordinateLabels={true} />
      );

      // Should show file labels (a-h for square8)
      expect(container.textContent).toContain('a');
      expect(container.textContent).toContain('h');
      // Should show rank labels (1-8)
      expect(container.textContent).toContain('1');
      expect(container.textContent).toContain('8');
    });

    it('does not show coordinate labels when disabled', () => {
      const board = createEmptyBoardState('square8');

      const { container } = render(
        <BoardView boardType="square8" board={board} showCoordinateLabels={false} />
      );

      // Coordinate labels should not be present in label elements
      const labelElements = container.querySelectorAll('.text-slate-400');
      expect(labelElements.length).toBe(0);
    });
  });

  describe('spectator mode', () => {
    it('disables cells in spectator mode', () => {
      const board = createEmptyBoardState('square8');

      const { container } = render(
        <BoardView boardType="square8" board={board} isSpectator={true} />
      );

      const buttons = container.querySelectorAll('button');
      buttons.forEach((button) => {
        expect(button).toBeDisabled();
      });
    });

    it('changes cursor to default in spectator mode', () => {
      const board = createEmptyBoardState('square8');

      const { container } = render(
        <BoardView boardType="square8" board={board} isSpectator={true} />
      );

      const button = container.querySelector('button');
      expect(button).toHaveClass('cursor-default');
    });
  });

  describe('data-testid', () => {
    it('has board-view data-testid for square8', () => {
      const board = createEmptyBoardState('square8');
      render(<BoardView boardType="square8" board={board} />);
      expect(screen.getByTestId('board-view')).toBeInTheDocument();
    });

    it('has board-view data-testid for hexagonal', () => {
      const board = createEmptyBoardState('hexagonal');
      board.type = 'hexagonal';
      render(<BoardView boardType="hexagonal" board={board} />);
      expect(screen.getByTestId('board-view')).toBeInTheDocument();
    });
  });
});
