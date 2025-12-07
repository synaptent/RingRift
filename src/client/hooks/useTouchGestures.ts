/**
 * useTouchGestures Hook
 *
 * Provides touch gesture detection for mobile devices:
 * - Tap: single touch (equivalent to click)
 * - Double-tap: two taps within 300ms (equivalent to double-click)
 * - Long-press: hold for 500ms+ (equivalent to context menu / right-click)
 *
 * This hook returns event handlers that can be spread onto a touchable element.
 * It distinguishes between gestures to enable proper mobile interactions on
 * the game board where different actions map to different gestures.
 *
 * Per RR-CANON-R076, this is a Host/Adapter Layer concern - it doesn't affect
 * game rules, only the UX for interacting with the board on touch devices.
 */

import { useCallback, useRef } from 'react';

/** Configuration options for touch gesture detection */
export interface TouchGestureOptions {
  /** Time window for double-tap detection (ms). Default: 300 */
  doubleTapDelay?: number;
  /** Duration to trigger long-press (ms). Default: 500 */
  longPressDelay?: number;
  /** Distance threshold for canceling gestures on move (px). Default: 10 */
  moveThreshold?: number;
  /** Disable gesture detection (e.g., for spectators). Default: false */
  disabled?: boolean;
}

/** Callbacks for different touch gestures */
export interface TouchGestureCallbacks<T = void> {
  /** Called on single tap */
  onTap?: (data: T) => void;
  /** Called on double-tap */
  onDoubleTap?: (data: T) => void;
  /** Called on long-press */
  onLongPress?: (data: T) => void;
}

/** Return value from useTouchGestures hook */
export interface TouchGestureHandlers {
  onTouchStart: (e: React.TouchEvent) => void;
  onTouchEnd: (e: React.TouchEvent) => void;
  onTouchMove: (e: React.TouchEvent) => void;
  onTouchCancel: (e: React.TouchEvent) => void;
}

const DEFAULT_DOUBLE_TAP_DELAY = 300;
const DEFAULT_LONG_PRESS_DELAY = 500;
const DEFAULT_MOVE_THRESHOLD = 10;

/**
 * Hook for detecting touch gestures on mobile devices.
 *
 * @param callbacks - Object containing callback functions for each gesture type
 * @param data - Data to pass to callbacks (e.g., cell position)
 * @param options - Configuration options
 * @returns Event handlers to spread onto the touchable element
 *
 * @example
 * ```tsx
 * const handlers = useTouchGestures(
 *   {
 *     onTap: (pos) => handleSelect(pos),
 *     onDoubleTap: (pos) => handleStackedPlacement(pos),
 *     onLongPress: (pos) => handleContextMenu(pos),
 *   },
 *   position,
 *   { disabled: isSpectator }
 * );
 *
 * return <div {...handlers}>Cell</div>;
 * ```
 */
export function useTouchGestures<T>(
  callbacks: TouchGestureCallbacks<T>,
  data: T,
  options: TouchGestureOptions = {}
): TouchGestureHandlers {
  const {
    doubleTapDelay = DEFAULT_DOUBLE_TAP_DELAY,
    longPressDelay = DEFAULT_LONG_PRESS_DELAY,
    moveThreshold = DEFAULT_MOVE_THRESHOLD,
    disabled = false,
  } = options;

  // Track touch state
  const touchStartRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const lastTapRef = useRef<number>(0);
  const longPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const gestureCompletedRef = useRef<boolean>(false);

  const clearLongPressTimer = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }, []);

  const onTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (disabled || e.touches.length !== 1) {
        clearLongPressTimer();
        return;
      }

      const touch = e.touches[0];
      touchStartRef.current = {
        x: touch.clientX,
        y: touch.clientY,
        time: Date.now(),
      };
      gestureCompletedRef.current = false;

      // Start long-press timer
      clearLongPressTimer();
      if (callbacks.onLongPress) {
        longPressTimerRef.current = setTimeout(() => {
          if (touchStartRef.current && !gestureCompletedRef.current) {
            gestureCompletedRef.current = true;
            callbacks.onLongPress?.(data);
            // Prevent subsequent tap from firing
            touchStartRef.current = null;
          }
        }, longPressDelay);
      }
    },
    [disabled, callbacks, data, longPressDelay, clearLongPressTimer]
  );

  const onTouchEnd = useCallback(
    (e: React.TouchEvent) => {
      clearLongPressTimer();

      if (disabled || !touchStartRef.current || gestureCompletedRef.current) {
        touchStartRef.current = null;
        return;
      }

      const now = Date.now();
      const touchDuration = now - touchStartRef.current.time;

      // If touch was too long, it might be a long-press (handled by timer)
      if (touchDuration >= longPressDelay) {
        touchStartRef.current = null;
        return;
      }

      // Check for double-tap
      const timeSinceLastTap = now - lastTapRef.current;
      if (timeSinceLastTap < doubleTapDelay && callbacks.onDoubleTap) {
        // Double-tap detected
        e.preventDefault();
        gestureCompletedRef.current = true;
        lastTapRef.current = 0; // Reset to prevent triple-tap
        callbacks.onDoubleTap(data);
      } else {
        // Single tap - fire immediately for responsiveness
        lastTapRef.current = now;
        callbacks.onTap?.(data);
      }

      touchStartRef.current = null;
    },
    [disabled, callbacks, data, doubleTapDelay, longPressDelay, clearLongPressTimer]
  );

  const onTouchMove = useCallback(
    (e: React.TouchEvent) => {
      if (!touchStartRef.current || gestureCompletedRef.current) {
        return;
      }

      const touch = e.touches[0];
      const dx = touch.clientX - touchStartRef.current.x;
      const dy = touch.clientY - touchStartRef.current.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      // Cancel gesture if finger moved too far
      if (distance > moveThreshold) {
        clearLongPressTimer();
        touchStartRef.current = null;
      }
    },
    [moveThreshold, clearLongPressTimer]
  );

  const onTouchCancel = useCallback(() => {
    clearLongPressTimer();
    touchStartRef.current = null;
    gestureCompletedRef.current = false;
  }, [clearLongPressTimer]);

  return {
    onTouchStart,
    onTouchEnd,
    onTouchMove,
    onTouchCancel,
  };
}

/**
 * Simplified hook for single-element touch handling.
 * Creates handlers that can be spread onto a single element.
 */
export function useCellTouchGestures(
  onTap: (() => void) | undefined,
  onDoubleTap: (() => void) | undefined,
  onLongPress: (() => void) | undefined,
  disabled: boolean = false
): TouchGestureHandlers {
  return useTouchGestures<void>(
    {
      onTap: onTap ? (_data: void) => onTap() : undefined,
      onDoubleTap: onDoubleTap ? (_data: void) => onDoubleTap() : undefined,
      onLongPress: onLongPress ? (_data: void) => onLongPress() : undefined,
    },
    undefined,
    { disabled }
  );
}
