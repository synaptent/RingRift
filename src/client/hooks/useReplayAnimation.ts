/**
 * useReplayAnimation - Hook for triggering move animations during replay playback.
 *
 * Unlike useAutoMoveAnimation which watches for new moves being appended,
 * this hook tracks move number changes and triggers animations based on
 * the move records from the replay database.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import type { Position } from '../../shared/types/game';
import type { MoveAnimationData } from '../components/BoardView';
import type { ReplayMoveRecord } from '../types/replay';

export interface UseReplayAnimationOptions {
  /** Current move number in replay */
  currentMoveNumber: number;
  /** All move records for the game */
  moves: ReplayMoveRecord[];
  /** Whether replay is currently playing (auto-advance) */
  isPlaying: boolean;
  /** Whether animations should be enabled */
  enabled?: boolean;
}

export interface UseReplayAnimationReturn {
  /** Current pending animation (if any) */
  pendingAnimation: MoveAnimationData | null;
  /** Clear the current animation */
  clearAnimation: () => void;
}

/**
 * Extract position from move data.
 * Move data from replay can have various formats.
 */
function extractPosition(data: unknown): Position | null {
  if (!data || typeof data !== 'object') return null;
  const obj = data as Record<string, unknown>;

  if (typeof obj.x === 'number' && typeof obj.y === 'number') {
    return {
      x: obj.x,
      y: obj.y,
      ...(typeof obj.z === 'number' ? { z: obj.z } : {}),
    };
  }
  return null;
}

/**
 * Determine animation type from move type string.
 */
function getAnimationType(moveType: string): MoveAnimationData['type'] {
  switch (moveType) {
    case 'place_ring':
      return 'place';
    case 'overtaking_capture':
    case 'chain_capture':
      return 'capture';
    case 'continue_capture_segment':
      return 'chain_capture';
    case 'recovery_slide':
      return 'move'; // Recovery slides markers like movement
    default:
      return 'move';
  }
}

export function useReplayAnimation({
  currentMoveNumber,
  moves,
  isPlaying,
  enabled = true,
}: UseReplayAnimationOptions): UseReplayAnimationReturn {
  const [pendingAnimation, setPendingAnimation] = useState<MoveAnimationData | null>(null);
  const prevMoveNumberRef = useRef<number>(currentMoveNumber);
  const animationIdRef = useRef(0);

  /**
   * Clear the current animation.
   */
  const clearAnimation = useCallback(() => {
    setPendingAnimation(null);
  }, []);

  /**
   * Trigger animation for a move.
   */
  const triggerAnimation = useCallback((move: ReplayMoveRecord) => {
    const moveData = move.move;

    // Extract from/to positions from move data
    const from = extractPosition(moveData.from);
    const to = extractPosition(moveData.to);

    // Skip if no destination (some moves like skip don't have positions)
    if (!to) return;

    const id = `replay-anim-${++animationIdRef.current}`;
    const type = getAnimationType(move.moveType);

    const animation: MoveAnimationData = {
      type,
      ...(from ? { from } : {}),
      to,
      playerNumber: move.player,
      id,
    };

    setPendingAnimation(animation);
  }, []);

  /**
   * Watch for move number changes and trigger animations.
   */
  useEffect(() => {
    if (!enabled) {
      prevMoveNumberRef.current = currentMoveNumber;
      return;
    }

    const prevMove = prevMoveNumberRef.current;
    const direction = currentMoveNumber - prevMove;

    // Only animate when stepping forward by 1
    // (Skip animations for large jumps, backward steps, or initial load)
    if (direction === 1 && currentMoveNumber > 0 && moves.length > 0) {
      // Get the move that was just played (move at index currentMoveNumber - 1)
      const moveRecord = moves[currentMoveNumber - 1];

      if (moveRecord) {
        triggerAnimation(moveRecord);
      }
    } else if (direction !== 0) {
      // Clear any pending animation on backward step or jump
      setPendingAnimation(null);
    }

    prevMoveNumberRef.current = currentMoveNumber;
  }, [currentMoveNumber, moves, enabled, triggerAnimation]);

  /**
   * Auto-clear animation after a delay when playing.
   * This ensures animations complete even if the next move comes quickly.
   */
  useEffect(() => {
    if (!pendingAnimation || !isPlaying) return;

    // Auto-clear after animation duration (slightly less than move interval)
    const timer = setTimeout(() => {
      clearAnimation();
    }, 400); // 400ms animation, leaving buffer for next move

    return () => clearTimeout(timer);
  }, [pendingAnimation, isPlaying, clearAnimation]);

  return {
    pendingAnimation,
    clearAnimation,
  };
}
