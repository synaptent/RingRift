/**
 * useGameSounds - Hook for playing sound effects based on game events
 *
 * Automatically plays appropriate sounds when:
 * - Moves are made (place, move, capture)
 * - Turn changes (your turn notification)
 * - Game ends (victory, defeat, draw)
 * - Lines are formed or territories claimed
 *
 * Usage:
 * ```tsx
 * const { playMoveSound, playTurnStartSound, playGameEndSound } = useGameSounds();
 *
 * // After a move is processed
 * playMoveSound(move.type, isCapture);
 *
 * // When it becomes your turn
 * playTurnStartSound();
 * ```
 */

import { useCallback } from 'react';
import { useSoundOptional } from '../contexts/SoundContext';
import type { GameResult } from '../../shared/types/game';

export type MoveType =
  | 'place_ring'
  | 'move_stack'
  | 'capture'
  | 'chain_capture'
  | 'process_line'
  | 'process_territory'
  | 'eliminate_ring'
  | 'no_line_action'
  | 'no_territory_action'
  | 'swap';

export interface UseGameSoundsResult {
  /** Play sound for a move based on its type */
  playMoveSound: (moveType: MoveType, isCapture?: boolean) => void;
  /** Play sound when it becomes the player's turn */
  playTurnStartSound: () => void;
  /** Play sound when game ends */
  playGameEndSound: (result: 'victory' | 'defeat' | 'draw') => void;
  /** Play sound when a line is formed */
  playLineFormedSound: () => void;
  /** Play sound when territory is claimed */
  playTerritoryClaimedSound: () => void;
  /** Play sound when a ring is eliminated */
  playEliminationSound: () => void;
  /** Play sound when phase changes */
  playPhaseChangeSound: () => void;
  /** Play sound when piece is selected */
  playSelectSound: () => void;
  /** Play sound when selection is cleared */
  playDeselectSound: () => void;
  /** Play low-time warning tick */
  playTickSound: () => void;
}

/**
 * Hook for playing game-related sound effects.
 * Uses the SoundContext for audio playback.
 * Safe to use outside of SoundProvider (sounds will be silently skipped).
 */
export function useGameSounds(): UseGameSoundsResult {
  const sound = useSoundOptional();

  const playMoveSound = useCallback(
    (moveType: MoveType, isCapture = false) => {
      if (!sound) return;

      if (isCapture || moveType === 'capture') {
        sound.playSound('capture');
      } else if (moveType === 'chain_capture') {
        sound.playSound('chain_capture');
      } else if (moveType === 'place_ring') {
        sound.playSound('place');
      } else if (moveType === 'move_stack') {
        sound.playSound('move');
      } else if (moveType === 'process_line') {
        sound.playSound('line_formed');
      } else if (moveType === 'process_territory') {
        sound.playSound('territory_claimed');
      } else if (moveType === 'eliminate_ring') {
        sound.playSound('elimination');
      }
      // no_line_action, no_territory_action, swap are silent
    },
    [sound]
  );

  const playTurnStartSound = useCallback(() => {
    if (!sound?.turnStartSound) return;
    sound.playSound('turn_start');
  }, [sound]);

  const playGameEndSound = useCallback(
    (result: 'victory' | 'defeat' | 'draw') => {
      if (!sound) return;
      sound.playSound(result);
    },
    [sound]
  );

  const playLineFormedSound = useCallback(() => {
    sound?.playSound('line_formed');
  }, [sound]);

  const playTerritoryClaimedSound = useCallback(() => {
    sound?.playSound('territory_claimed');
  }, [sound]);

  const playEliminationSound = useCallback(() => {
    sound?.playSound('elimination');
  }, [sound]);

  const playPhaseChangeSound = useCallback(() => {
    sound?.playSound('phase_change');
  }, [sound]);

  const playSelectSound = useCallback(() => {
    sound?.playSound('select');
  }, [sound]);

  const playDeselectSound = useCallback(() => {
    sound?.playSound('deselect');
  }, [sound]);

  const playTickSound = useCallback(() => {
    sound?.playSound('tick');
  }, [sound]);

  return {
    playMoveSound,
    playTurnStartSound,
    playGameEndSound,
    playLineFormedSound,
    playTerritoryClaimedSound,
    playEliminationSound,
    playPhaseChangeSound,
    playSelectSound,
    playDeselectSound,
    playTickSound,
  };
}

/**
 * Derive game end result for sound from GameResult
 */
export function getGameEndSoundType(
  gameResult: GameResult,
  myPlayerNumber?: number
): 'victory' | 'defeat' | 'draw' | null {
  if (gameResult.reason === 'draw') {
    return 'draw';
  }

  if (!gameResult.winner) {
    return null;
  }

  // If we know the player number, determine if they won or lost
  if (myPlayerNumber !== undefined) {
    return gameResult.winner === myPlayerNumber ? 'victory' : 'defeat';
  }

  // Default to victory sound for spectators/unknown
  return 'victory';
}

export default useGameSounds;
