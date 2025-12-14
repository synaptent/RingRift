import { renderHook, act } from '@testing-library/react';
import React from 'react';
import {
  useKeyboardNavigation,
  useGlobalGameShortcuts,
  getPlayerFocusRingClass,
  PLAYER_FOCUS_RING_CLASSES,
} from '../../../src/client/hooks/useKeyboardNavigation';
import type { Position } from '../../../src/shared/types/game';

describe('useKeyboardNavigation', () => {
  describe('initialization', () => {
    it('starts with no focused position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      expect(result.current.focusedPosition).toBeNull();
    });

    it('provides all expected methods', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      expect(typeof result.current.setFocusedPosition).toBe('function');
      expect(typeof result.current.moveFocus).toBe('function');
      expect(typeof result.current.clearSelection).toBe('function');
      expect(typeof result.current.handleKeyDown).toBe('function');
      expect(typeof result.current.handleCellFocus).toBe('function');
      expect(typeof result.current.registerCellRef).toBe('function');
      expect(typeof result.current.getCellRef).toBe('function');
      expect(typeof result.current.isFocused).toBe('function');
    });
  });

  describe('square8 board navigation', () => {
    it('moves focus down from a middle position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      act(() => {
        result.current.moveFocus(0, 1); // down
      });

      expect(result.current.focusedPosition).toEqual({ x: 3, y: 4 });
    });

    it('moves focus up from a middle position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      act(() => {
        result.current.moveFocus(0, -1); // up
      });

      expect(result.current.focusedPosition).toEqual({ x: 3, y: 2 });
    });

    it('moves focus left from a middle position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      act(() => {
        result.current.moveFocus(-1, 0); // left
      });

      expect(result.current.focusedPosition).toEqual({ x: 2, y: 3 });
    });

    it('moves focus right from a middle position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      act(() => {
        result.current.moveFocus(1, 0); // right
      });

      expect(result.current.focusedPosition).toEqual({ x: 4, y: 3 });
    });

    it('does not move past left edge', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 0, y: 3 });
      });

      act(() => {
        result.current.moveFocus(-1, 0); // left at edge
      });

      expect(result.current.focusedPosition).toEqual({ x: 0, y: 3 });
    });

    it('does not move past right edge', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 7, y: 3 });
      });

      act(() => {
        result.current.moveFocus(1, 0); // right at edge
      });

      expect(result.current.focusedPosition).toEqual({ x: 7, y: 3 });
    });

    it('does not move past top edge', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 0 });
      });

      act(() => {
        result.current.moveFocus(0, -1); // up at edge
      });

      expect(result.current.focusedPosition).toEqual({ x: 3, y: 0 });
    });

    it('does not move past bottom edge', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 7 });
      });

      act(() => {
        result.current.moveFocus(0, 1); // down at edge
      });

      expect(result.current.focusedPosition).toEqual({ x: 3, y: 7 });
    });
  });

  describe('square19 board navigation', () => {
    it('respects 19x19 bounds', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square19',
          size: 19,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 18, y: 18 });
      });

      act(() => {
        result.current.moveFocus(1, 0); // right at edge
      });

      expect(result.current.focusedPosition).toEqual({ x: 18, y: 18 });

      act(() => {
        result.current.moveFocus(0, 1); // down at edge
      });

      expect(result.current.focusedPosition).toEqual({ x: 18, y: 18 });
    });
  });

  describe('hexagonal board navigation', () => {
    it('moves focus in hex grid directions', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'hexagonal',
          size: 5, // radius 4
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 0, y: 0, z: 0 });
      });

      // Move right (increase r)
      act(() => {
        result.current.moveFocus(1, 0);
      });

      expect(result.current.focusedPosition).toEqual({ x: 0, y: 1, z: -1 });
    });

    it('does not move outside hex radius', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'hexagonal',
          size: 3, // radius 2
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 2, y: 0, z: -2 });
      });

      // Try to move down (increase q) - would go outside radius
      act(() => {
        result.current.moveFocus(0, 1);
      });

      // Should stay at current position
      expect(result.current.focusedPosition).toEqual({ x: 2, y: 0, z: -2 });
    });
  });

  describe('keyboard event handling', () => {
    it('handles ArrowUp key', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'ArrowUp',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toEqual({ x: 3, y: 2 });
    });

    it('handles ArrowDown key', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'ArrowDown',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toEqual({ x: 3, y: 4 });
    });

    it('handles ArrowLeft key', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'ArrowLeft',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toEqual({ x: 2, y: 3 });
    });

    it('handles ArrowRight key', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'ArrowRight',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toEqual({ x: 4, y: 3 });
    });

    it('handles Escape key to clear selection', () => {
      const onClear = jest.fn();
      const onAnnounce = jest.fn();
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
          onClear,
          onAnnounce,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'Escape',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toBeNull();
      expect(onClear).toHaveBeenCalled();
      expect(onAnnounce).toHaveBeenCalledWith('Selection cleared');
    });

    it('handles Enter key for selection', () => {
      const onSelect = jest.fn();
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
          onSelect,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'Enter',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(onSelect).toHaveBeenCalledWith({ x: 3, y: 3 });
    });

    it('handles Space key for selection', () => {
      const onSelect = jest.fn();
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
          onSelect,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: ' ',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(onSelect).toHaveBeenCalledWith({ x: 3, y: 3 });
    });

    it('does not select when spectator', () => {
      const onSelect = jest.fn();
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
          isSpectator: true,
          onSelect,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      const event = {
        key: 'Enter',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(onSelect).not.toHaveBeenCalled();
    });

    it('handles Home key to jump to first cell', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 5, y: 5 });
      });

      const event = {
        key: 'Home',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toEqual({ x: 0, y: 0 });
    });

    it('handles End key to jump to last cell', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 0, y: 0 });
      });

      const event = {
        key: 'End',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(result.current.focusedPosition).toEqual({ x: 7, y: 7 });
    });

    it('handles ? key for help', () => {
      const onShowHelp = jest.fn();
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
          onShowHelp,
        })
      );

      const event = {
        key: '?',
        preventDefault: jest.fn(),
      } as unknown as React.KeyboardEvent<HTMLElement>;

      act(() => {
        result.current.handleKeyDown(event);
      });

      expect(event.preventDefault).toHaveBeenCalled();
      expect(onShowHelp).toHaveBeenCalled();
    });
  });

  describe('cell focus handling', () => {
    it('updates focused position on cell focus', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.handleCellFocus({ x: 4, y: 5 });
      });

      expect(result.current.focusedPosition).toEqual({ x: 4, y: 5 });
    });
  });

  describe('isFocused utility', () => {
    it('returns true for focused position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      expect(result.current.isFocused({ x: 3, y: 3 })).toBe(true);
    });

    it('returns false for non-focused position', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      act(() => {
        result.current.setFocusedPosition({ x: 3, y: 3 });
      });

      expect(result.current.isFocused({ x: 4, y: 4 })).toBe(false);
    });

    it('returns false when no position is focused', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      expect(result.current.isFocused({ x: 3, y: 3 })).toBe(false);
    });
  });

  describe('cell ref management', () => {
    it('registers and retrieves cell refs', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      const mockButton = document.createElement('button');

      act(() => {
        result.current.registerCellRef('0,0', mockButton as unknown as HTMLButtonElement);
      });

      expect(result.current.getCellRef('0,0')).toBe(mockButton);
    });

    it('removes cell refs when null is passed', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      const mockButton = document.createElement('button');

      act(() => {
        result.current.registerCellRef('0,0', mockButton as unknown as HTMLButtonElement);
      });

      act(() => {
        result.current.registerCellRef('0,0', null);
      });

      expect(result.current.getCellRef('0,0')).toBeUndefined();
    });
  });

  describe('first movement when no focus', () => {
    it('focuses first position when moving with no focus', () => {
      const { result } = renderHook(() =>
        useKeyboardNavigation({
          boardType: 'square8',
          size: 8,
        })
      );

      expect(result.current.focusedPosition).toBeNull();

      act(() => {
        result.current.moveFocus(0, 1); // Any direction
      });

      // Should focus first position (0,0)
      expect(result.current.focusedPosition).toEqual({ x: 0, y: 0 });
    });
  });
});

describe('useGlobalGameShortcuts', () => {
  let originalAddEventListener: typeof window.addEventListener;
  let originalRemoveEventListener: typeof window.removeEventListener;
  let keydownHandler: ((e: KeyboardEvent) => void) | null = null;

  beforeEach(() => {
    originalAddEventListener = window.addEventListener;
    originalRemoveEventListener = window.removeEventListener;

    window.addEventListener = jest.fn(
      (event: string, handler: EventListenerOrEventListenerObject) => {
        if (event === 'keydown') {
          keydownHandler = handler as (e: KeyboardEvent) => void;
        }
      }
    );

    window.removeEventListener = jest.fn();
  });

  afterEach(() => {
    window.addEventListener = originalAddEventListener;
    window.removeEventListener = originalRemoveEventListener;
    keydownHandler = null;
  });

  it('registers keydown listener on mount', () => {
    renderHook(() =>
      useGlobalGameShortcuts({
        onShowHelp: jest.fn(),
      })
    );

    expect(window.addEventListener).toHaveBeenCalledWith('keydown', expect.any(Function));
  });

  it('unregisters keydown listener on unmount', () => {
    const { unmount } = renderHook(() =>
      useGlobalGameShortcuts({
        onShowHelp: jest.fn(),
      })
    );

    unmount();

    expect(window.removeEventListener).toHaveBeenCalledWith('keydown', expect.any(Function));
  });

  it('calls onShowHelp for ? key', () => {
    const onShowHelp = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onShowHelp,
      })
    );

    const event = new KeyboardEvent('keydown', { key: '?' });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onShowHelp).toHaveBeenCalled();
  });

  it('calls onResign for R key', () => {
    const onResign = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onResign,
      })
    );

    const event = new KeyboardEvent('keydown', { key: 'r' });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onResign).toHaveBeenCalled();
  });

  it('calls onToggleMute for M key', () => {
    const onToggleMute = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onToggleMute,
      })
    );

    const event = new KeyboardEvent('keydown', { key: 'm' });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onToggleMute).toHaveBeenCalled();
  });

  it('calls onToggleFullscreen for F key', () => {
    const onToggleFullscreen = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onToggleFullscreen,
      })
    );

    const event = new KeyboardEvent('keydown', { key: 'f' });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onToggleFullscreen).toHaveBeenCalled();
  });

  it('calls onUndo for Ctrl+Z', () => {
    const onUndo = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onUndo,
      })
    );

    const event = new KeyboardEvent('keydown', { key: 'z', ctrlKey: true });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onUndo).toHaveBeenCalled();
  });

  it('calls onRedo for Ctrl+Shift+Z', () => {
    const onRedo = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onRedo,
      })
    );

    const event = new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, shiftKey: true });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onRedo).toHaveBeenCalled();
  });

  it('does not trigger shortcuts when typing in input', () => {
    const onResign = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onResign,
      })
    );

    const input = document.createElement('input');
    const event = new KeyboardEvent('keydown', { key: 'r' });
    Object.defineProperty(event, 'target', { value: input });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onResign).not.toHaveBeenCalled();
  });

  it('does not trigger shortcuts when typing in textarea', () => {
    const onResign = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onResign,
      })
    );

    const textarea = document.createElement('textarea');
    const event = new KeyboardEvent('keydown', { key: 'r' });
    Object.defineProperty(event, 'target', { value: textarea });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onResign).not.toHaveBeenCalled();
  });

  it('does not call onResign with Ctrl+R (browser refresh)', () => {
    const onResign = jest.fn();
    renderHook(() =>
      useGlobalGameShortcuts({
        onResign,
      })
    );

    const event = new KeyboardEvent('keydown', { key: 'r', ctrlKey: true });
    Object.defineProperty(event, 'target', { value: document.body });
    Object.defineProperty(event, 'preventDefault', { value: jest.fn() });

    keydownHandler?.(event);

    expect(onResign).not.toHaveBeenCalled();
  });
});

describe('getPlayerFocusRingClass', () => {
  it('returns emerald for player 1', () => {
    expect(getPlayerFocusRingClass(1)).toBe('ring-emerald-400');
  });

  it('returns sky for player 2', () => {
    expect(getPlayerFocusRingClass(2)).toBe('ring-sky-400');
  });

  it('returns amber for player 3', () => {
    expect(getPlayerFocusRingClass(3)).toBe('ring-amber-400');
  });

  it('returns fuchsia for player 4', () => {
    expect(getPlayerFocusRingClass(4)).toBe('ring-fuchsia-400');
  });

  it('returns amber fallback for undefined', () => {
    expect(getPlayerFocusRingClass(undefined)).toBe('ring-amber-400');
  });

  it('returns amber fallback for invalid player number', () => {
    expect(getPlayerFocusRingClass(5)).toBe('ring-amber-400');
    expect(getPlayerFocusRingClass(0)).toBe('ring-amber-400');
  });
});

describe('PLAYER_FOCUS_RING_CLASSES constant', () => {
  it('maps all 4 players to unique classes', () => {
    expect(PLAYER_FOCUS_RING_CLASSES[1]).toBe('ring-emerald-400');
    expect(PLAYER_FOCUS_RING_CLASSES[2]).toBe('ring-sky-400');
    expect(PLAYER_FOCUS_RING_CLASSES[3]).toBe('ring-amber-400');
    expect(PLAYER_FOCUS_RING_CLASSES[4]).toBe('ring-fuchsia-400');
  });

  it('has 4 entries', () => {
    expect(Object.keys(PLAYER_FOCUS_RING_CLASSES).length).toBe(4);
  });
});
