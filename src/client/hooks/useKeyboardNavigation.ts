import { useState, useCallback, useRef, useEffect } from 'react';
import type { Position, BoardType } from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';

export interface KeyboardNavigationOptions {
  /** Board type for navigation geometry */
  boardType: BoardType;
  /** Board size (for hex: radius+1, for square: dimension) */
  size: number;
  /** Whether the user is a spectator (disables selection) */
  isSpectator?: boolean;
  /** Callback when a cell is selected (Enter/Space) */
  onSelect?: (position: Position) => void;
  /** Callback when selection is cleared (Escape) */
  onClear?: () => void;
  /** Callback for screen reader announcements */
  onAnnounce?: (message: string) => void;
  /** Callback when help is requested (? key) */
  onShowHelp?: () => void;
}

export interface KeyboardNavigationState {
  /** Currently focused position */
  focusedPosition: Position | null;
  /** Set focused position directly */
  setFocusedPosition: (position: Position | null) => void;
  /** Move focus in a direction (dx, dy) */
  moveFocus: (dx: number, dy: number) => void;
  /** Clear selection and focus */
  clearSelection: () => void;
  /** Handle keyboard events on the board container */
  handleKeyDown: (e: React.KeyboardEvent<HTMLElement>) => void;
  /** Handle cell focus event */
  handleCellFocus: (position: Position) => void;
  /** Register a cell ref for focus management */
  registerCellRef: (key: string, ref: HTMLButtonElement | null) => void;
  /** Get cell ref by position key */
  getCellRef: (key: string) => HTMLButtonElement | undefined;
  /** Check if a position is keyboard-focused */
  isFocused: (position: Position) => boolean;
}

/**
 * Build list of all valid positions for a given board type and size.
 */
function getAllPositions(boardType: BoardType, size: number): Position[] {
  const positions: Position[] = [];

  if (boardType === 'square8' || boardType === 'square19') {
    const dim = boardType === 'square8' ? 8 : 19;
    for (let y = 0; y < dim; y++) {
      for (let x = 0; x < dim; x++) {
        positions.push({ x, y });
      }
    }
  } else if (boardType === 'hexagonal' || boardType === 'hex8') {
    const radius = (size - 1) / 2; // size is bounding box, radius = (size-1)/2
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        positions.push({ x: q, y: r, z: s });
      }
    }
  }

  return positions;
}

/**
 * Find neighboring position in a given direction for keyboard navigation.
 */
function findNeighbor(
  current: Position,
  dx: number,
  dy: number,
  boardType: BoardType,
  size: number
): Position | null {
  if (boardType === 'square8' || boardType === 'square19') {
    const dim = boardType === 'square8' ? 8 : 19;
    const newX = current.x + dx;
    const newY = current.y + dy;

    if (newX >= 0 && newX < dim && newY >= 0 && newY < dim) {
      return { x: newX, y: newY };
    }
    return null;
  }

  // Hex board navigation - map arrow keys to approximate hex directions
  if (boardType === 'hexagonal' || boardType === 'hex8') {
    const radius = (size - 1) / 2; // size is bounding box, radius = (size-1)/2
    let newQ = current.x;
    let newR = current.y;

    // Arrow key mappings for hex grid (approximate visual directions)
    if (dy === -1) {
      // Up: decrease q
      newQ = current.x - 1;
      newR = current.y;
    } else if (dy === 1) {
      // Down: increase q
      newQ = current.x + 1;
      newR = current.y;
    } else if (dx === -1) {
      // Left: decrease r
      newR = current.y - 1;
    } else if (dx === 1) {
      // Right: increase r
      newR = current.y + 1;
    }

    const newS = -newQ - newR;

    // Check if position is valid (within hex radius)
    if (Math.abs(newQ) <= radius && Math.abs(newR) <= radius && Math.abs(newS) <= radius) {
      return { x: newQ, y: newR, z: newS };
    }
    return null;
  }

  return null;
}

/**
 * Hook for managing keyboard navigation on a game board.
 *
 * Provides arrow key navigation, Enter/Space selection, Escape to clear,
 * and focus management for all board types (square8, square19, hexagonal).
 *
 * @example
 * ```tsx
 * const nav = useKeyboardNavigation({
 *   boardType: 'square8',
 *   size: 8,
 *   onSelect: (pos) => handleCellClick(pos),
 *   onAnnounce: (msg) => setAnnouncement(msg),
 * });
 *
 * return (
 *   <div onKeyDown={nav.handleKeyDown} tabIndex={0}>
 *     {cells.map((cell) => (
 *       <button
 *         ref={(ref) => nav.registerCellRef(cell.key, ref)}
 *         onFocus={() => nav.handleCellFocus(cell.position)}
 *         className={nav.isFocused(cell.position) ? 'focused' : ''}
 *       />
 *     ))}
 *   </div>
 * );
 * ```
 */
export function useKeyboardNavigation(options: KeyboardNavigationOptions): KeyboardNavigationState {
  const { boardType, size, isSpectator, onSelect, onClear, onAnnounce, onShowHelp } = options;

  const [focusedPosition, setFocusedPosition] = useState<Position | null>(null);
  const cellRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

  // Get all valid positions for this board
  const allPositions = getAllPositions(boardType, size);

  const registerCellRef = useCallback((key: string, ref: HTMLButtonElement | null) => {
    if (ref) {
      cellRefs.current.set(key, ref);
    } else {
      cellRefs.current.delete(key);
    }
  }, []);

  const getCellRef = useCallback((key: string): HTMLButtonElement | undefined => {
    return cellRefs.current.get(key);
  }, []);

  const moveFocus = useCallback(
    (dx: number, dy: number) => {
      if (!focusedPosition) {
        // If no focus, start at first position
        if (allPositions.length > 0) {
          const firstPos = allPositions[0];
          setFocusedPosition(firstPos);
          const key = positionToString(firstPos);
          const cellRef = cellRefs.current.get(key);
          if (cellRef) {
            cellRef.focus();
          }
        }
        return;
      }

      const neighbor = findNeighbor(focusedPosition, dx, dy, boardType, size);
      if (neighbor) {
        setFocusedPosition(neighbor);
        const key = positionToString(neighbor);
        const cellRef = cellRefs.current.get(key);
        if (cellRef) {
          cellRef.focus();
        }
      }
    },
    [focusedPosition, boardType, size, allPositions]
  );

  const clearSelection = useCallback(() => {
    setFocusedPosition(null);
    onClear?.();
    onAnnounce?.('Selection cleared');
  }, [onClear, onAnnounce]);

  const handleCellFocus = useCallback((position: Position) => {
    setFocusedPosition(position);
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLElement>) => {
      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          moveFocus(0, -1);
          break;
        case 'ArrowDown':
          e.preventDefault();
          moveFocus(0, 1);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          moveFocus(-1, 0);
          break;
        case 'ArrowRight':
          e.preventDefault();
          moveFocus(1, 0);
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (focusedPosition && !isSpectator) {
            onSelect?.(focusedPosition);
          }
          break;
        case 'Escape':
          e.preventDefault();
          clearSelection();
          break;
        case '?':
          e.preventDefault();
          onShowHelp?.();
          break;
        // Home/End for jumping to corners
        case 'Home':
          e.preventDefault();
          if (allPositions.length > 0) {
            const firstPos = allPositions[0];
            setFocusedPosition(firstPos);
            const key = positionToString(firstPos);
            const cellRef = cellRefs.current.get(key);
            if (cellRef) {
              cellRef.focus();
            }
          }
          break;
        case 'End':
          e.preventDefault();
          if (allPositions.length > 0) {
            const lastPos = allPositions[allPositions.length - 1];
            setFocusedPosition(lastPos);
            const key = positionToString(lastPos);
            const cellRef = cellRefs.current.get(key);
            if (cellRef) {
              cellRef.focus();
            }
          }
          break;
        default:
          break;
      }
    },
    [focusedPosition, isSpectator, moveFocus, clearSelection, onSelect, onShowHelp, allPositions]
  );

  const isFocused = useCallback(
    (position: Position): boolean => {
      return focusedPosition !== null && positionsEqual(focusedPosition, position);
    },
    [focusedPosition]
  );

  return {
    focusedPosition,
    setFocusedPosition,
    moveFocus,
    clearSelection,
    handleKeyDown,
    handleCellFocus,
    registerCellRef,
    getCellRef,
    isFocused,
  };
}

/**
 * Global game keyboard shortcuts configuration.
 * These are actions that can be triggered from anywhere in the game view.
 */
export interface GlobalGameShortcuts {
  /** Show keyboard shortcuts help (?) */
  onShowHelp?: () => void;
  /** Resign from the game (R) */
  onResign?: () => void;
  /** Toggle sound/mute (M) */
  onToggleMute?: () => void;
  /** Undo last move (Ctrl/Cmd+Z) - sandbox only */
  onUndo?: () => void;
  /** Redo last undone move (Ctrl/Cmd+Shift+Z) - sandbox only */
  onRedo?: () => void;
  /** Toggle fullscreen (F) */
  onToggleFullscreen?: () => void;
}

/**
 * Hook for global game keyboard shortcuts.
 * These shortcuts work when focus is anywhere in the game view,
 * not just on the board.
 *
 * @example
 * ```tsx
 * useGlobalGameShortcuts({
 *   onShowHelp: () => setHelpOpen(true),
 *   onResign: () => setResignConfirmOpen(true),
 *   onToggleMute: () => setSoundEnabled(!soundEnabled),
 * });
 * ```
 */
export function useGlobalGameShortcuts(shortcuts: GlobalGameShortcuts) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.defaultPrevented) {
        return;
      }
      // Don't trigger shortcuts when typing in an input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      const isModifier = e.ctrlKey || e.metaKey;

      switch (e.key.toLowerCase()) {
        case '?':
          e.preventDefault();
          shortcuts.onShowHelp?.();
          break;
        case 'r':
          if (!isModifier) {
            e.preventDefault();
            shortcuts.onResign?.();
          }
          break;
        case 'm':
          if (!isModifier) {
            e.preventDefault();
            shortcuts.onToggleMute?.();
          }
          break;
        case 'f':
          if (!isModifier) {
            e.preventDefault();
            shortcuts.onToggleFullscreen?.();
          }
          break;
        case 'z':
          if (isModifier) {
            e.preventDefault();
            if (e.shiftKey) {
              shortcuts.onRedo?.();
            } else {
              shortcuts.onUndo?.();
            }
          }
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}

/**
 * Player color classes for focus indicators.
 * Maps player number to Tailwind classes for focus rings.
 */
export const PLAYER_FOCUS_RING_CLASSES: Record<number, string> = {
  1: 'ring-emerald-400',
  2: 'ring-sky-400',
  3: 'ring-amber-400',
  4: 'ring-fuchsia-400',
};

/**
 * Get focus ring classes for a player.
 * Returns an amber ring as fallback for no player / spectator.
 */
export function getPlayerFocusRingClass(playerNumber?: number): string {
  if (!playerNumber) return 'ring-amber-400';
  return PLAYER_FOCUS_RING_CLASSES[playerNumber] || 'ring-amber-400';
}
