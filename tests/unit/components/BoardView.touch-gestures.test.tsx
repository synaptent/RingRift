import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import { BoardState, Position } from '../../../src/shared/types/game';

// Touch timing constants (must match BoardView.tsx)
const LONG_PRESS_DELAY_MS = 500;
const DOUBLE_TAP_DELAY_MS = 300;

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

// Helper to create touch events
function createTouchEvent(type: string, target: Element, options: Partial<Touch> = {}) {
  const touch = {
    identifier: 0,
    target,
    clientX: 100,
    clientY: 100,
    pageX: 100,
    pageY: 100,
    screenX: 100,
    screenY: 100,
    radiusX: 0,
    radiusY: 0,
    rotationAngle: 0,
    force: 1,
    ...options,
  } as Touch;

  return new TouchEvent(type, {
    bubbles: true,
    cancelable: true,
    touches: type === 'touchend' ? [] : [touch],
    targetTouches: type === 'touchend' ? [] : [touch],
    changedTouches: [touch],
  });
}

describe('BoardView Touch Gestures', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Long Press Detection', () => {
    it('should trigger onCellLongPress after 500ms of continuous touch', async () => {
      const onCellLongPress = jest.fn();
      const board = createEmptyBoardState('square8');

      render(<BoardView boardType="square8" board={board} onCellLongPress={onCellLongPress} />);

      const cell = screen.getAllByRole('button')[0];

      // Start touch
      fireEvent(cell, createTouchEvent('touchstart', cell));

      // Before threshold - should not trigger
      act(() => {
        jest.advanceTimersByTime(LONG_PRESS_DELAY_MS - 100);
      });
      expect(onCellLongPress).not.toHaveBeenCalled();

      // At threshold - should trigger
      act(() => {
        jest.advanceTimersByTime(100);
      });
      expect(onCellLongPress).toHaveBeenCalledTimes(1);
    });

    it('should not trigger long press if touch ends before threshold', () => {
      const onCellLongPress = jest.fn();
      const board = createEmptyBoardState('square8');

      render(<BoardView boardType="square8" board={board} onCellLongPress={onCellLongPress} />);

      const cell = screen.getAllByRole('button')[0];

      // Start touch
      fireEvent(cell, createTouchEvent('touchstart', cell));

      // Release before threshold
      act(() => {
        jest.advanceTimersByTime(LONG_PRESS_DELAY_MS - 100);
      });
      fireEvent(cell, createTouchEvent('touchend', cell));

      // Advance past threshold
      act(() => {
        jest.advanceTimersByTime(200);
      });

      expect(onCellLongPress).not.toHaveBeenCalled();
    });

    it('should cancel long press if touch moves away', () => {
      const onCellLongPress = jest.fn();
      const board = createEmptyBoardState('square8');

      render(<BoardView boardType="square8" board={board} onCellLongPress={onCellLongPress} />);

      const cell = screen.getAllByRole('button')[0];

      // Start touch
      fireEvent(cell, createTouchEvent('touchstart', cell));

      // Move touch (cancel gesture)
      act(() => {
        jest.advanceTimersByTime(200);
      });
      fireEvent(cell, createTouchEvent('touchmove', cell, { clientX: 200, clientY: 200 }));

      // Advance past threshold
      act(() => {
        jest.advanceTimersByTime(LONG_PRESS_DELAY_MS);
      });

      expect(onCellLongPress).not.toHaveBeenCalled();
    });

    it('should fall back to onCellContextMenu if onCellLongPress not provided', () => {
      const onCellContextMenu = jest.fn();
      const board = createEmptyBoardState('square8');

      render(<BoardView boardType="square8" board={board} onCellContextMenu={onCellContextMenu} />);

      const cell = screen.getAllByRole('button')[0];

      // Start touch and wait for long press
      fireEvent(cell, createTouchEvent('touchstart', cell));
      act(() => {
        jest.advanceTimersByTime(LONG_PRESS_DELAY_MS);
      });

      expect(onCellContextMenu).toHaveBeenCalledTimes(1);
    });
  });

  describe('Double Tap Detection', () => {
    it('should trigger onCellDoubleClick for two taps within 300ms', () => {
      const onCellDoubleClick = jest.fn();
      const onCellClick = jest.fn();
      const board = createEmptyBoardState('square8');

      render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={onCellClick}
          onCellDoubleClick={onCellDoubleClick}
        />
      );

      const cell = screen.getAllByRole('button')[0];

      // First tap
      fireEvent(cell, createTouchEvent('touchstart', cell));
      fireEvent(cell, createTouchEvent('touchend', cell));

      // Second tap within threshold
      act(() => {
        jest.advanceTimersByTime(DOUBLE_TAP_DELAY_MS - 100);
      });
      fireEvent(cell, createTouchEvent('touchstart', cell));
      fireEvent(cell, createTouchEvent('touchend', cell));

      expect(onCellDoubleClick).toHaveBeenCalledTimes(1);
    });

    it('should trigger single click for single tap', () => {
      const onCellClick = jest.fn();
      const onCellDoubleClick = jest.fn();
      const board = createEmptyBoardState('square8');

      render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={onCellClick}
          onCellDoubleClick={onCellDoubleClick}
        />
      );

      const cell = screen.getAllByRole('button')[0];

      // Regular click (non-touch) should trigger onCellClick
      fireEvent.click(cell);

      expect(onCellClick).toHaveBeenCalledTimes(1);
      expect(onCellDoubleClick).not.toHaveBeenCalled();
    });

    it('should not trigger double click for taps outside threshold', () => {
      const onCellDoubleClick = jest.fn();
      const onCellClick = jest.fn();
      const board = createEmptyBoardState('square8');

      render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={onCellClick}
          onCellDoubleClick={onCellDoubleClick}
        />
      );

      const cell = screen.getAllByRole('button')[0];

      // First tap
      fireEvent(cell, createTouchEvent('touchstart', cell));
      fireEvent(cell, createTouchEvent('touchend', cell));

      // Wait past threshold
      act(() => {
        jest.advanceTimersByTime(DOUBLE_TAP_DELAY_MS + 100);
      });

      // Second tap - should be separate single tap (not double click)
      fireEvent(cell, createTouchEvent('touchstart', cell));
      fireEvent(cell, createTouchEvent('touchend', cell));

      // Neither tap triggers double click
      expect(onCellDoubleClick).not.toHaveBeenCalled();
      // onCellClick is called via the button's onClick handler, not touch events
      // Touch events only handle double-tap and long-press detection
    });
  });

  describe('Touch vs Click Disambiguation', () => {
    it('should not trigger click after long press', () => {
      const onCellClick = jest.fn();
      const onCellLongPress = jest.fn();
      const board = createEmptyBoardState('square8');

      render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={onCellClick}
          onCellLongPress={onCellLongPress}
        />
      );

      const cell = screen.getAllByRole('button')[0];

      // Long press
      fireEvent(cell, createTouchEvent('touchstart', cell));
      act(() => {
        jest.advanceTimersByTime(LONG_PRESS_DELAY_MS);
      });
      fireEvent(cell, createTouchEvent('touchend', cell));

      // Should trigger long press but not click
      expect(onCellLongPress).toHaveBeenCalledTimes(1);
      expect(onCellClick).not.toHaveBeenCalled();
    });

    it('should handle context menu on right click', () => {
      const onCellContextMenu = jest.fn();
      const board = createEmptyBoardState('square8');

      render(<BoardView boardType="square8" board={board} onCellContextMenu={onCellContextMenu} />);

      const cell = screen.getAllByRole('button')[0];

      // Right click
      fireEvent.contextMenu(cell);

      expect(onCellContextMenu).toHaveBeenCalledTimes(1);
    });
  });

  describe('Multi-touch Handling', () => {
    it('should ignore multi-touch events at start', () => {
      const onCellLongPress = jest.fn();
      const board = createEmptyBoardState('square8');

      render(<BoardView boardType="square8" board={board} onCellLongPress={onCellLongPress} />);

      const cell = screen.getAllByRole('button')[0];

      // Start with multi-touch (should be ignored)
      const multiTouchEvent = new TouchEvent('touchstart', {
        bubbles: true,
        cancelable: true,
        touches: [
          { identifier: 0, target: cell } as Touch,
          { identifier: 1, target: cell } as Touch,
        ],
        targetTouches: [
          { identifier: 0, target: cell } as Touch,
          { identifier: 1, target: cell } as Touch,
        ],
        changedTouches: [{ identifier: 0, target: cell } as Touch],
      });
      fireEvent(cell, multiTouchEvent);

      // Advance past threshold
      act(() => {
        jest.advanceTimersByTime(LONG_PRESS_DELAY_MS + 100);
      });

      // Should not trigger - multi-touch is ignored at start
      expect(onCellLongPress).not.toHaveBeenCalled();
    });
  });

  describe('Spectator Mode', () => {
    it('should not trigger interactions when isSpectator is true', () => {
      const onCellClick = jest.fn();
      const onCellLongPress = jest.fn();
      const board = createEmptyBoardState('square8');

      render(
        <BoardView
          boardType="square8"
          board={board}
          onCellClick={onCellClick}
          onCellLongPress={onCellLongPress}
          isSpectator={true}
        />
      );

      const cells = screen.getAllByRole('button');

      // Cells should be disabled in spectator mode
      cells.forEach((cell) => {
        expect(cell).toBeDisabled();
      });
    });
  });
});
