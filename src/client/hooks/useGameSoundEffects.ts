/**
 * useGameSoundEffects - Hook that automatically plays sound effects for game events
 *
 * Watches game state changes and plays appropriate sounds for:
 * - Phase changes
 * - Turn changes (your turn notification)
 * - Moves (based on moveHistory changes)
 * - Game end (victory, defeat, draw)
 *
 * Usage:
 * ```tsx
 * // In BackendGameHost or SandboxGameHost
 * useGameSoundEffects({
 *   gameState,
 *   victoryState,
 *   currentUserId: user?.id,
 *   myPlayerNumber,
 * });
 * ```
 */

import { useEffect, useRef } from 'react';
import { useGameSounds, getGameEndSoundType } from './useGameSounds';
import type { GameState, GameResult, Move } from '../../shared/types/game';
import { normalizeLegacyMoveType } from '../../shared/engine/legacy/legacyMoveTypes';

export interface UseGameSoundEffectsOptions {
  /** Current game state */
  gameState: GameState | null;
  /** Victory state when game ends */
  victoryState: GameResult | null;
  /** Current user ID (to determine victory/defeat) */
  currentUserId?: string;
  /** Player number of the current user (if playing) */
  myPlayerNumber?: number;
}

/**
 * Maps canonical move types to sound effect types
 */
function getMoveTypeForSound(
  move: Move
):
  | 'place_ring'
  | 'move_stack'
  | 'capture'
  | 'chain_capture'
  | 'process_line'
  | 'process_territory'
  | 'eliminate_ring'
  | null {
  if (move.type === 'line_formation') {
    return 'process_line';
  }
  if (move.type === 'territory_claim') {
    return 'process_territory';
  }

  const canonicalType = normalizeLegacyMoveType(move.type);
  switch (canonicalType) {
    case 'place_ring':
      return 'place_ring';
    case 'move_stack':
      return 'move_stack';
    case 'overtaking_capture':
      return 'capture';
    case 'continue_capture_segment':
      return 'chain_capture';
    case 'process_line':
    case 'choose_line_option':
      return 'process_line';
    case 'choose_territory_option':
      return 'process_territory';
    case 'eliminate_rings_from_stack':
    case 'forced_elimination':
      return 'eliminate_ring';
    // Silent moves
    case 'no_line_action':
    case 'no_territory_action':
    case 'skip_capture':
    case 'skip_placement':
    case 'skip_recovery':
    case 'swap_sides':
    case 'resign':
    case 'timeout':
      return null;
    default:
      return null;
  }
}

/**
 * Hook that automatically plays sound effects based on game state changes.
 * Must be called within a SoundProvider context.
 */
export function useGameSoundEffects({
  gameState,
  victoryState,
  currentUserId,
  myPlayerNumber,
}: UseGameSoundEffectsOptions): void {
  const { playMoveSound, playTurnStartSound, playGameEndSound, playPhaseChangeSound } =
    useGameSounds();

  // Track previous state for change detection
  const prevPhaseRef = useRef<string | null>(null);
  const prevCurrentPlayerRef = useRef<number | null>(null);
  const prevMoveHistoryLengthRef = useRef<number>(0);
  const gameEndSoundPlayedRef = useRef<boolean>(false);

  // Phase change sounds
  useEffect(() => {
    if (!gameState) {
      prevPhaseRef.current = null;
      return;
    }

    const currentPhase = gameState.currentPhase;
    const prevPhase = prevPhaseRef.current;

    // Only play sound if phase actually changed (not initial load)
    if (prevPhase !== null && currentPhase !== prevPhase) {
      playPhaseChangeSound();
    }

    prevPhaseRef.current = currentPhase;
  }, [gameState?.currentPhase, playPhaseChangeSound]);

  // Turn start sounds (when it becomes your turn)
  useEffect(() => {
    if (!gameState || myPlayerNumber === undefined) {
      prevCurrentPlayerRef.current = null;
      return;
    }

    const currentPlayer = gameState.currentPlayer;
    const prevPlayer = prevCurrentPlayerRef.current;

    // Only play sound if:
    // 1. Player actually changed (not initial load)
    // 2. It's now your turn
    // 3. Game is still active
    if (
      prevPlayer !== null &&
      currentPlayer !== prevPlayer &&
      currentPlayer === myPlayerNumber &&
      gameState.gameStatus === 'active'
    ) {
      playTurnStartSound();
    }

    prevCurrentPlayerRef.current = currentPlayer;
  }, [gameState?.currentPlayer, gameState?.gameStatus, myPlayerNumber, playTurnStartSound]);

  // Move sounds (when new moves are added to history)
  useEffect(() => {
    if (!gameState?.moveHistory) {
      prevMoveHistoryLengthRef.current = 0;
      return;
    }

    const currentLength = gameState.moveHistory.length;
    const prevLength = prevMoveHistoryLengthRef.current;

    // Play sound for each new move
    if (currentLength > prevLength) {
      // Get the most recent new move
      const latestMove = gameState.moveHistory[currentLength - 1];
      if (latestMove) {
        const soundType = getMoveTypeForSound(latestMove);
        if (soundType) {
          // Check if it's a capture move
          const isCapture =
            latestMove.type === 'overtaking_capture' ||
            latestMove.type === 'continue_capture_segment';
          playMoveSound(soundType, isCapture);
        }
      }
    }

    prevMoveHistoryLengthRef.current = currentLength;
  }, [gameState?.moveHistory?.length, playMoveSound]);

  // Game end sounds
  useEffect(() => {
    if (!victoryState) {
      // Reset when victoryState is cleared (new game)
      gameEndSoundPlayedRef.current = false;
      return;
    }

    // Only play game end sound once
    if (gameEndSoundPlayedRef.current) {
      return;
    }

    const soundType = getGameEndSoundType(victoryState, myPlayerNumber);
    if (soundType) {
      playGameEndSound(soundType);
      gameEndSoundPlayedRef.current = true;
    }
  }, [victoryState, currentUserId, playGameEndSound]);
}

export default useGameSoundEffects;
