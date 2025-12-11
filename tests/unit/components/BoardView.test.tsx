import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import type { BoardState, BoardType, Position, RingStack } from '../../../src/shared/types/game';

const createRingStack = (position: Position, player = 1): RingStack => ({
  position,
  rings: [player],
  stackHeight: 1,
  capHeight: 1,
  controllingPlayer: player,
});

const createBoardState = (boardType: BoardType = 'square8'): BoardState => ({
  stacks: new Map(),
  markers: new Map(),
  collapsedSpaces: new Map(),
  territories: new Map(),
  formedLines: [],
  eliminatedRings: {},
  size: boardType === 'hexagonal' ? 3 : boardType === 'square19' ? 19 : 8,
  type: boardType,
});

const renderBoard = (props: Partial<React.ComponentProps<typeof BoardView>> = {}) => {
  const board = createBoardState(props.boardType as BoardType | undefined);
  return render(
    <BoardView
      boardType="square8"
      board={board}
      validTargets={[]}
      onCellClick={jest.fn()}
      onCellDoubleClick={jest.fn()}
      onCellContextMenu={jest.fn()}
      {...props}
    />
  );
};

describe('BoardView', () => {
  it('renders square board cells and coordinate labels when enabled', () => {
    renderBoard({ showCoordinateLabels: true });

    const cells = screen.getAllByRole('gridcell');
    expect(cells.length).toBe(64); // 8x8 board
    expect(screen.getAllByText('a').length).toBeGreaterThan(0);
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);
  });

  it('invokes click, double-click, and context menu handlers for a cell', async () => {
    const onCellClick = jest.fn();
    const onCellDoubleClick = jest.fn();
    const onCellContextMenu = jest.fn();

    renderBoard({ onCellClick, onCellDoubleClick, onCellContextMenu });
    const user = userEvent.setup();
    const firstCell = screen.getAllByRole('gridcell')[0];

    await user.click(firstCell);
    await user.dblClick(firstCell);
    fireEvent.contextMenu(firstCell);

    expect(onCellClick).toHaveBeenCalledWith({ x: 0, y: 0 });
    expect(onCellDoubleClick).toHaveBeenCalledWith({ x: 0, y: 0 });
    expect(onCellContextMenu).toHaveBeenCalledWith({ x: 0, y: 0 });
  });

  it('suppresses input handlers when rendered for spectators', async () => {
    const onCellClick = jest.fn();
    const onCellDoubleClick = jest.fn();
    const onCellContextMenu = jest.fn();

    renderBoard({
      onCellClick,
      onCellDoubleClick,
      onCellContextMenu,
      isSpectator: true,
    });
    const user = userEvent.setup();
    const firstCell = screen.getAllByRole('gridcell')[0];

    await user.click(firstCell);
    await user.dblClick(firstCell);
    fireEvent.contextMenu(firstCell);

    expect(onCellClick).not.toHaveBeenCalled();
    expect(onCellDoubleClick).not.toHaveBeenCalled();
    expect(onCellContextMenu).not.toHaveBeenCalled();
  });

  it('handles keyboard navigation and selection', () => {
    const onCellClick = jest.fn();
    renderBoard({ onCellClick });

    const boardContainer = screen.getByTestId('board-view');
    boardContainer.focus();

    fireEvent.keyDown(boardContainer, { key: 'ArrowRight' });
    fireEvent.keyDown(boardContainer, { key: 'Enter' });

    expect(onCellClick).toHaveBeenCalledWith({ x: 0, y: 0 });
  });

  it('fires keyboard help callback on question-mark press', () => {
    const onShowKeyboardHelp = jest.fn();
    renderBoard({ onShowKeyboardHelp });

    const boardContainer = screen.getByTestId('board-view');
    boardContainer.focus();
    fireEvent.keyDown(boardContainer, { key: '?' });

    expect(onShowKeyboardHelp).toHaveBeenCalled();
  });

  it('announces stack details from view model when present', () => {
    const board = createBoardState('square8');
    const pos: Position = { x: 1, y: 1 };
    const stack = createRingStack(pos, 2);
    board.stacks.set('1,1', stack);

    render(
      <BoardView
        boardType="square8"
        board={board}
        validTargets={[]}
        viewModel={{
          boardType: 'square8',
          size: 8,
          cells: [
            {
              position: pos,
              positionKey: '1,1',
              stack: {
                position: pos,
                positionKey: '1,1',
                rings: [
                  {
                    playerNumber: 2,
                    colorClass: 'bg-blue-500',
                    borderClass: 'border-blue-300',
                    isTop: true,
                    isInCap: true,
                  },
                ],
                stackHeight: 1,
                capHeight: 1,
                controllingPlayer: 2,
              },
              isSelected: true,
              isValidTarget: false,
              isDarkSquare: false,
            },
          ],
        }}
      />
    );

    expect(
      screen.getByLabelText(/Row 2, Column 2\. Stack height 1, cap 1, player 2/)
    ).toBeInTheDocument();
  });
});
