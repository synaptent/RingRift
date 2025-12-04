import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import { BoardState, Position, RingStack } from '../../../src/shared/types/game';
import type { BoardViewModel } from '../../../src/client/adapters/gameViewModels';

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

    it('renders an empty container for unknown board type', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType={'unknown' as any} board={board} />);

      // The accessibility wrapper remains, but no interactive board cells
      // should be rendered when the board type is unknown.
      const wrapper = screen.getByTestId('board-view');
      expect(wrapper).toBeInTheDocument();
      const buttons = container.querySelectorAll('button');
      expect(buttons.length).toBe(0);
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

  describe('decision highlights', () => {
    it('applies primary and secondary decision highlight classes on square boards', () => {
      const board = createEmptyBoardState('square8');
      const baseVM: BoardViewModel = {
        boardType: 'square8',
        size: 8,
        cells: [],
        rows: [],
        decisionHighlights: {
          choiceKind: 'line_reward',
          highlights: [
            { positionKey: '1,1', intensity: 'primary' },
            { positionKey: '2,2', intensity: 'secondary' },
          ],
        },
      };

      const { container } = render(
        <BoardView boardType="square8" board={board} viewModel={baseVM} />
      );

      const primaryCell = container.querySelector('button[data-x="1"][data-y="1"]');
      const secondaryCell = container.querySelector('button[data-x="2"][data-y="2"]');

      expect(primaryCell).toHaveClass('decision-highlight-primary');
      expect(primaryCell).toHaveClass('line-formation-burst');
      expect(primaryCell).toHaveAttribute('data-decision-highlight', 'primary');

      expect(secondaryCell).toHaveClass('decision-highlight-secondary');
      expect(secondaryCell).toHaveAttribute('data-decision-highlight', 'secondary');
    });

    it('applies decision highlight classes on hex boards using cube coordinates', () => {
      const board = createEmptyBoardState('hexagonal');
      board.type = 'hexagonal';
      board.size = 3; // small radius to keep button count manageable

      const baseVM: BoardViewModel = {
        boardType: 'hexagonal',
        size: 3,
        cells: [],
        decisionHighlights: {
          choiceKind: 'capture_direction',
          highlights: [{ positionKey: '0,0,0', intensity: 'primary' }],
        },
      };

      const { container } = render(
        <BoardView boardType="hexagonal" board={board} viewModel={baseVM} />
      );

      const highlightedCell = container.querySelector('button[data-x="0"][data-y="0"][data-z="0"]');
      expect(highlightedCell).toHaveClass('decision-highlight-primary');
      expect(highlightedCell).toHaveAttribute('data-decision-highlight', 'primary');
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

    it('does not render a marker when a stack is present on the same cell', () => {
      const board = createEmptyBoardState('square8');

      // Stack and marker at the same position; view model should prefer stack.
      const pos: Position = { x: 2, y: 2 };
      const key = '2,2';

      const stack: RingStack = {
        position: pos,
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };

      board.stacks.set(key, stack);
      board.markers.set(key, { type: 'regular', player: 1, position: pos });

      const { container } = render(<BoardView boardType="square8" board={board} />);

      const cell = container.querySelector('button[data-x="2"][data-y="2"]');
      expect(cell).toBeInTheDocument();

      // Cell should show stack info but no marker border class.
      expect(container.textContent).toContain('H2');
      expect(container.textContent).toContain('C2');

      const markerInCell = (cell as HTMLElement).querySelector('.border-emerald-400');
      expect(markerInCell).toBeNull();
    });
  });

  describe('movement destination pulses', () => {
    it('applies destination pulse only to landing stack on hex boards', async () => {
      const origin: Position = { x: 0, y: 0, z: 0 };
      const dest: Position = { x: 0, y: 2, z: -2 };

      // Initial board: stack at origin only
      const preBoard = createEmptyBoardState('hexagonal');
      preBoard.type = 'hexagonal';
      preBoard.size = 11;
      preBoard.stacks.set('0,0,0', {
        position: origin,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      const { container, rerender } = render(<BoardView boardType="hexagonal" board={preBoard} />);

      // Post-move board: stack moved to dest, marker left at origin
      const postBoard = createEmptyBoardState('hexagonal');
      postBoard.type = 'hexagonal';
      postBoard.size = 11;
      postBoard.stacks.set('0,2,-2', {
        position: dest,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });
      postBoard.markers.set('0,0,0', {
        position: origin,
        player: 1,
        type: 'regular',
      });

      rerender(<BoardView boardType="hexagonal" board={postBoard} />);

      // Wait for BoardView's board-diff effect to run and apply animations.
      const getCells = () => {
        const originCell = container.querySelector(
          'button[data-x="0"][data-y="0"][data-z="0"]'
        ) as HTMLButtonElement | null;
        const destCell = container.querySelector(
          'button[data-x="0"][data-y="2"][data-z="-2"]'
        ) as HTMLButtonElement | null;
        return { originCell, destCell };
      };

      await screen.findByTestId('board-view');

      const { originCell, destCell } = getCells();
      expect(originCell).not.toBeNull();
      expect(destCell).not.toBeNull();

      // The origin (marker-only) cell should not have the destination pulse,
      // while the landing stack cell should.
      expect(originCell).not.toHaveClass('move-destination-pulse');
      expect(destCell).toHaveClass('move-destination-pulse');
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

  describe('rules-lab overlays', () => {
    it('sets line overlay data attributes when showLineOverlays is true', () => {
      const board = createEmptyBoardState('square8');
      board.formedLines = [
        {
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
          ],
          player: 1,
          length: 2,
          direction: { x: 1, y: 0 },
        },
      ];

      const { container } = render(
        <BoardView boardType="square8" board={board} showLineOverlays={true} />
      );

      const cell = container.querySelector(
        'button[data-x="0"][data-y="0"]'
      ) as HTMLButtonElement | null;
      expect(cell).not.toBeNull();
      expect(cell).toHaveAttribute('data-line-overlay', 'true');
      expect(cell).toHaveAttribute('data-line-overlay-player', '1');
    });

    it('sets territory region overlay data attributes when showTerritoryRegionOverlays is true', () => {
      const board = createEmptyBoardState('square8');
      board.territories.set('region-1', {
        spaces: [{ x: 2, y: 2 }],
        controllingPlayer: 2,
        isDisconnected: true,
      });

      const { container } = render(
        <BoardView boardType="square8" board={board} showTerritoryRegionOverlays={true} />
      );

      const cell = container.querySelector(
        'button[data-x="2"][data-y="2"]'
      ) as HTMLButtonElement | null;
      expect(cell).not.toBeNull();
      expect(cell).toHaveAttribute('data-region-overlay', 'true');
      expect(cell).toHaveAttribute('data-region-overlay-player', '2');
      expect(cell).toHaveAttribute('data-region-overlay-disconnected', 'true');
    });
  });

  describe('chain capture path visualization', () => {
    it('accepts chainCapturePath prop and renders board without errors', () => {
      const board = createEmptyBoardState('square8');
      const chainCapturePath = [
        { x: 2, y: 2 },
        { x: 4, y: 2 },
        { x: 6, y: 2 },
      ];

      // The SVG overlay relies on DOM geometry (getBoundingClientRect) which
      // isn't available in JSDOM. This test verifies the component accepts
      // the prop and renders the board correctly without errors.
      render(<BoardView boardType="square8" board={board} chainCapturePath={chainCapturePath} />);

      // Board should render correctly with the chainCapturePath prop
      expect(screen.getByTestId('board-view')).toBeInTheDocument();
    });

    it('does not render chain capture overlay when path has fewer than 2 positions', () => {
      const board = createEmptyBoardState('square8');
      const chainCapturePath = [{ x: 2, y: 2 }]; // Only one position

      const { container } = render(
        <BoardView boardType="square8" board={board} chainCapturePath={chainCapturePath} />
      );

      // Should not have the chain capture arrow marker (path too short)
      const arrowMarker = container.querySelector('#chain-capture-arrow');
      expect(arrowMarker).toBeNull();
    });

    it('does not render chain capture overlay when path is undefined', () => {
      const board = createEmptyBoardState('square8');

      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Should not have the chain capture arrow marker
      const arrowMarker = container.querySelector('#chain-capture-arrow');
      expect(arrowMarker).toBeNull();
    });

    it('accepts chainCapturePath prop on hex boards', () => {
      const board = createEmptyBoardState('hexagonal');
      board.type = 'hexagonal';
      board.size = 11;

      const chainCapturePath = [
        { x: 0, y: 0, z: 0 },
        { x: 1, y: -1, z: 0 },
        { x: 2, y: -2, z: 0 },
      ];

      // Verify component accepts the prop without errors on hex boards
      render(<BoardView boardType="hexagonal" board={board} chainCapturePath={chainCapturePath} />);

      expect(screen.getByTestId('board-view')).toBeInTheDocument();
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

  describe('keyboard navigation', () => {
    it('makes the board container focusable with tabIndex=0', () => {
      const board = createEmptyBoardState('square8');
      render(<BoardView boardType="square8" board={board} />);

      const boardView = screen.getByTestId('board-view');
      expect(boardView).toHaveAttribute('tabIndex', '0');

      // Simulate keyboard tab focus landing on the board container.
      (boardView as HTMLElement).focus();
      expect(document.activeElement).toBe(boardView);
    });

    it('focuses cell on arrow key press from initial state', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      const boardView = screen.getByTestId('board-view');
      fireEvent.keyDown(boardView, { key: 'ArrowDown' });

      // First cell should receive focus
      const firstCell = container.querySelector('button[data-x="0"][data-y="0"]');
      expect(document.activeElement).toBe(firstCell);
    });

    it('moves focus right with ArrowRight key', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Focus first cell
      const firstCell = container.querySelector(
        'button[data-x="0"][data-y="0"]'
      ) as HTMLButtonElement;
      firstCell.focus();
      fireEvent.focus(firstCell);

      // Press ArrowRight
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'ArrowRight' });

      const nextCell = container.querySelector('button[data-x="1"][data-y="0"]');
      expect(document.activeElement).toBe(nextCell);
    });

    it('moves focus down with ArrowDown key', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Focus first cell
      const firstCell = container.querySelector(
        'button[data-x="0"][data-y="0"]'
      ) as HTMLButtonElement;
      firstCell.focus();
      fireEvent.focus(firstCell);

      // Press ArrowDown
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'ArrowDown' });

      const nextCell = container.querySelector('button[data-x="0"][data-y="1"]');
      expect(document.activeElement).toBe(nextCell);
    });

    it('calls onCellClick when Enter is pressed on focused cell', () => {
      const board = createEmptyBoardState('square8');
      const handleCellClick = jest.fn();
      const { container } = render(
        <BoardView boardType="square8" board={board} onCellClick={handleCellClick} />
      );

      // Focus a cell
      const cell = container.querySelector('button[data-x="3"][data-y="3"]') as HTMLButtonElement;
      cell.focus();
      fireEvent.focus(cell);

      // Press Enter
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'Enter' });

      expect(handleCellClick).toHaveBeenCalledTimes(1);
      expect(handleCellClick).toHaveBeenCalledWith(expect.objectContaining({ x: 3, y: 3 }));
    });

    it('calls onCellClick when Space is pressed on focused cell', () => {
      const board = createEmptyBoardState('square8');
      const handleCellClick = jest.fn();
      const { container } = render(
        <BoardView boardType="square8" board={board} onCellClick={handleCellClick} />
      );

      // Focus a cell
      const cell = container.querySelector('button[data-x="4"][data-y="4"]') as HTMLButtonElement;
      cell.focus();
      fireEvent.focus(cell);

      // Press Space
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: ' ' });

      expect(handleCellClick).toHaveBeenCalledTimes(1);
      expect(handleCellClick).toHaveBeenCalledWith(expect.objectContaining({ x: 4, y: 4 }));
    });

    it('does not trigger click when in spectator mode', () => {
      const board = createEmptyBoardState('square8');
      const handleCellClick = jest.fn();
      const { container } = render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={handleCellClick}
          isSpectator={true}
        />
      );

      // Focus a cell
      const cell = container.querySelector('button[data-x="3"][data-y="3"]') as HTMLButtonElement;
      cell.focus();
      fireEvent.focus(cell);

      // Press Enter
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'Enter' });

      expect(handleCellClick).not.toHaveBeenCalled();
    });

    it('does not move focus beyond board boundaries', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Focus top-left cell
      const topLeftCell = container.querySelector(
        'button[data-x="0"][data-y="0"]'
      ) as HTMLButtonElement;
      topLeftCell.focus();
      fireEvent.focus(topLeftCell);

      // Try to move up (should stay in place)
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'ArrowUp' });
      expect(document.activeElement).toBe(topLeftCell);

      // Try to move left (should stay in place)
      fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'ArrowLeft' });
      expect(document.activeElement).toBe(topLeftCell);
    });

    it('shows focus ring styling on focused cell', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      // Focus a cell
      const cell = container.querySelector('button[data-x="3"][data-y="3"]') as HTMLButtonElement;
      cell.focus();
      fireEvent.focus(cell);

      // Cell should have amber focus ring class
      expect(cell).toHaveClass('ring-amber-400');
    });

    it('navigates hexagonal board with arrow keys', () => {
      const board = createEmptyBoardState('hexagonal');
      board.type = 'hexagonal';
      board.size = 3; // Small size for testing

      const { container } = render(<BoardView boardType="hexagonal" board={board} />);

      // Focus center cell (0,0,0)
      const centerCell = container.querySelector(
        'button[data-x="0"][data-y="0"][data-z="0"]'
      ) as HTMLButtonElement;
      centerCell?.focus();
      if (centerCell) {
        fireEvent.focus(centerCell);
      }

      // Navigate with arrow key - should move to adjacent hex
      if (centerCell) {
        fireEvent.keyDown(screen.getByTestId('board-view'), { key: 'ArrowRight' });
        // Active element should be different from center
        expect(document.activeElement).not.toBe(centerCell);
      }
    });

    it('has proper ARIA attributes for accessibility', () => {
      const board = createEmptyBoardState('square8');
      const { container } = render(<BoardView boardType="square8" board={board} />);

      const boardView = screen.getByTestId('board-view');
      expect(boardView).toHaveAttribute('role', 'grid');
      expect(boardView).toHaveAttribute('aria-label');

      // Check that cells have proper roles and labels
      const cell = container.querySelector('button[data-x="0"][data-y="0"]');
      expect(cell).toHaveAttribute('role', 'gridcell');
      expect(cell).toHaveAttribute('aria-label');
    });

    it('has screen reader announcement region', () => {
      const board = createEmptyBoardState('square8');
      render(<BoardView boardType="square8" board={board} />);

      const srRegion = document.querySelector('[role="status"][aria-live="polite"]');
      expect(srRegion).toBeInTheDocument();
    });
  });

  describe('cell coordinate attributes', () => {
    it('renders square8 cells with data-x and data-y attributes matching their coordinates', () => {
      const board = createEmptyBoardState('square8');

      const { container } = render(<BoardView boardType="square8" board={board} />);

      const cells = Array.from(container.querySelectorAll('button'));
      expect(cells).toHaveLength(64);

      // Ensure that each cell exposes its coordinate via data-x/data-y so that
      // E2E helpers and movement tests can target cells deterministically.
      cells.forEach((cell) => {
        const xAttr = cell.getAttribute('data-x');
        const yAttr = cell.getAttribute('data-y');
        expect(xAttr).not.toBeNull();
        expect(yAttr).not.toBeNull();

        const x = Number(xAttr);
        const y = Number(yAttr);
        expect(Number.isInteger(x)).toBe(true);
        expect(Number.isInteger(y)).toBe(true);
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThan(8);
        expect(y).toBeGreaterThanOrEqual(0);
        expect(y).toBeLessThan(8);
      });
    });

    it('renders hexagonal cells with cube-coordinate data-x/data-y/data-z attributes', () => {
      const board = createEmptyBoardState('hexagonal');
      board.type = 'hexagonal';
      board.size = 11; // canonical side length for the hex board in core rules

      const { container } = render(<BoardView boardType="hexagonal" board={board} />);

      const cells = Array.from(container.querySelectorAll('button'));
      expect(cells.length).toBeGreaterThan(0);

      const radius = board.size - 1;

      cells.forEach((cell) => {
        const xAttr = cell.getAttribute('data-x');
        const yAttr = cell.getAttribute('data-y');
        const zAttr = cell.getAttribute('data-z');

        expect(xAttr).not.toBeNull();
        expect(yAttr).not.toBeNull();
        expect(zAttr).not.toBeNull();

        const q = Number(xAttr);
        const r = Number(yAttr);
        const s = Number(zAttr);

        expect(Number.isInteger(q)).toBe(true);
        expect(Number.isInteger(r)).toBe(true);
        expect(Number.isInteger(s)).toBe(true);

        // Cube coordinate invariant for hex boards: q + r + s === 0 within radius.
        expect(q + r + s).toBe(0);
        expect(Math.abs(q)).toBeLessThanOrEqual(radius);
        expect(Math.abs(r)).toBeLessThanOrEqual(radius);
        expect(Math.abs(s)).toBeLessThanOrEqual(radius);
      });
    });
  });
});
