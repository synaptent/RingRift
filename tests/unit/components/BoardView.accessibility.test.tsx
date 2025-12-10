import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import type { BoardState, Position } from '../../../src/shared/types/game';

function emptyBoard(): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };
}

describe('BoardView accessibility announcements', () => {
  it('announces valid move count when a piece is selected', () => {
    const selected: Position = { x: 1, y: 1 };
    const validTargets: Position[] = [
      { x: 1, y: 2 },
      { x: 2, y: 1 },
    ];

    render(
      <BoardView
        boardType="square8"
        board={emptyBoard()}
        selectedPosition={selected}
        validTargets={validTargets}
      />
    );

    const announcement = screen.getByRole('status');
    expect(announcement).toHaveTextContent('Piece selected. 2 valid moves available');
  });

  it('supports keyboard navigation and help shortcut', async () => {
    const onCellClick = jest.fn();
    const onShowKeyboardHelp = jest.fn();

    render(
      <BoardView
        boardType="square8"
        board={emptyBoard()}
        onCellClick={onCellClick}
        onShowKeyboardHelp={onShowKeyboardHelp}
      />
    );

    const grid = screen.getByTestId('board-view');

    await act(async () => {
      grid.focus();
    });

    // First ArrowRight initializes focus on the first cell; second moves right.
    await act(async () => {
      grid.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    });
    await act(async () => {
      grid.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    });

    // Press Enter to invoke onCellClick at the focused cell.
    await act(async () => {
      grid.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    });

    await waitFor(() => {
      expect(onCellClick).toHaveBeenCalledTimes(1);
    });

    // Question mark should open keyboard help.
    await act(async () => {
      grid.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true }));
    });

    await waitFor(() => {
      expect(onShowKeyboardHelp).toHaveBeenCalledTimes(1);
    });
  });

  it('announces when selection is cleared via Escape', async () => {
    const selected: Position = { x: 0, y: 0 };
    const validTargets: Position[] = [{ x: 0, y: 1 }];

    render(
      <BoardView
        boardType="square8"
        board={emptyBoard()}
        selectedPosition={selected}
        validTargets={validTargets}
      />
    );

    const grid = screen.getByTestId('board-view');
    const status = screen.getByRole('status');

    // Initial announcement includes valid move count.
    expect(status).toHaveTextContent('1 valid move');

    await act(async () => {
      grid.focus();
    });

    await act(async () => {
      grid.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    });

    await waitFor(() => {
      expect(status).toHaveTextContent('Selection cleared');
    });
  });
});
