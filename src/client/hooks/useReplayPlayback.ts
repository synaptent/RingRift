/**
 * useReplayPlayback - React hook for managing game replay playback.
 *
 * Handles:
 * - Loading a game for replay
 * - Stepping forward/backward through moves
 * - Auto-play (movie mode) at configurable speed
 * - State caching and prefetching
 *
 * Usage:
 *   const playback = useReplayPlayback();
 *   await playback.loadGame(gameId);
 *   playback.stepForward();
 *   playback.play();
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { getReplayService } from '../services/ReplayService';
import type { ReplayMoveRecord, ReplayPlaybackState, PlaybackSpeed } from '../types/replay';
import type { GameState } from '../../shared/types/game';

const INITIAL_STATE: ReplayPlaybackState = {
  gameId: null,
  metadata: null,
  currentMoveNumber: 0,
  totalMoves: 0,
  currentState: null,
  isPlaying: false,
  playbackSpeed: 1,
  isLoading: false,
  error: null,
  moves: [],
};

/**
 * Calculate delay between moves based on speed.
 * Base delay is 1000ms (1 second per move at 1x speed).
 */
function getDelayMs(speed: PlaybackSpeed): number {
  return 1000 / speed;
}

/**
 * Minimum delay to allow animations to complete.
 * Even at 4x speed, we want at least 200ms for visual feedback.
 */
const MIN_DELAY_MS = 200;

export interface UseReplayPlaybackReturn extends ReplayPlaybackState {
  /** Load a game for replay */
  loadGame: (gameId: string) => Promise<void>;
  /** Unload current game */
  unloadGame: () => void;
  /** Step forward one move */
  stepForward: () => Promise<void>;
  /** Step backward one move */
  stepBackward: () => Promise<void>;
  /** Jump to specific move */
  jumpToMove: (moveNumber: number) => Promise<void>;
  /** Jump to start (move 0) */
  jumpToStart: () => Promise<void>;
  /** Jump to end (last move) */
  jumpToEnd: () => Promise<void>;
  /** Start auto-play */
  play: () => void;
  /** Pause auto-play */
  pause: () => void;
  /** Toggle play/pause */
  togglePlay: () => void;
  /** Set playback speed */
  setSpeed: (speed: PlaybackSpeed) => void;
  /** Get move record at current position */
  getCurrentMove: () => ReplayMoveRecord | null;
  /** Check if can step forward */
  canStepForward: boolean;
  /** Check if can step backward */
  canStepBackward: boolean;
}

export function useReplayPlayback(): UseReplayPlaybackReturn {
  const [state, setState] = useState<ReplayPlaybackState>(INITIAL_STATE);
  const queryClient = useQueryClient();
  const service = getReplayService();
  const playIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (playIntervalRef.current) {
        clearTimeout(playIntervalRef.current);
      }
    };
  }, []);

  // Stop playback when reaching the end
  useEffect(() => {
    if (state.isPlaying && state.currentMoveNumber >= state.totalMoves) {
      setState((s) => ({ ...s, isPlaying: false }));
      if (playIntervalRef.current) {
        clearTimeout(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    }
  }, [state.isPlaying, state.currentMoveNumber, state.totalMoves]);

  /**
   * Fetch and cache state at a move number.
   */
  const fetchState = useCallback(
    async (gameId: string, moveNumber: number): Promise<GameState> => {
      const response = await service.getStateAtMove(gameId, moveNumber);
      return response.gameState;
    },
    [service]
  );

  /**
   * Get a cached state if available (no fetch).
   */
  const getCachedState = useCallback(
    (gameId: string, moveNumber: number): GameState | undefined => {
      const data = queryClient.getQueryData<{ gameState: GameState }>([
        'replay',
        'games',
        gameId,
        'state',
        moveNumber,
      ]);
      return data?.gameState;
    },
    [queryClient]
  );

  /**
   * Prefetch adjacent states for smooth navigation.
   * Adapts prefetch depth based on whether we're playing or manually navigating.
   */
  const prefetchAdjacent = useCallback(
    async (
      gameId: string,
      currentMove: number,
      totalMoves: number,
      options?: { isPlaying?: boolean; speed?: PlaybackSpeed }
    ) => {
      const toPrefetch: number[] = [];
      const isPlaying = options?.isPlaying ?? false;
      const speed = options?.speed ?? 1;

      // Determine how many moves ahead to prefetch based on playback state
      // At higher speeds, prefetch more aggressively
      const lookAhead = isPlaying ? Math.max(3, Math.ceil(4 / speed)) : 2;
      const lookBehind = 1;

      // Prefetch forward (more important during playback)
      for (let i = 1; i <= lookAhead; i++) {
        if (currentMove + i <= totalMoves) {
          toPrefetch.push(currentMove + i);
        }
      }

      // Prefetch backward (for stepping back)
      for (let i = 1; i <= lookBehind; i++) {
        if (currentMove - i >= 0) {
          toPrefetch.push(currentMove - i);
        }
      }

      // Also prefetch key positions: start, end, and some intermediate points
      // for quick scrubber jumps (only if game is long enough)
      if (totalMoves > 20) {
        const keyPositions = [
          0, // Start
          totalMoves, // End
          Math.floor(totalMoves / 4), // 25%
          Math.floor(totalMoves / 2), // 50%
          Math.floor((totalMoves * 3) / 4), // 75%
        ];
        keyPositions.forEach((pos) => {
          if (!toPrefetch.includes(pos)) {
            toPrefetch.push(pos);
          }
        });
      }

      // Fire off prefetches in background (don't await)
      toPrefetch.forEach((move) => {
        // Skip if already cached
        if (getCachedState(gameId, move)) return;

        queryClient.prefetchQuery({
          queryKey: ['replay', 'games', gameId, 'state', move],
          queryFn: () => service.getStateAtMove(gameId, move),
          staleTime: Infinity,
        });
      });
    },
    [queryClient, service, getCachedState]
  );

  /**
   * Load a game for replay.
   */
  const loadGame = useCallback(
    async (gameId: string) => {
      setState((s) => ({
        ...s,
        isLoading: true,
        error: null,
        isPlaying: false,
      }));

      // Stop any existing playback
      if (playIntervalRef.current) {
        clearTimeout(playIntervalRef.current);
        playIntervalRef.current = null;
      }

      try {
        // Fetch game metadata
        const metadata = await service.getGame(gameId);

        // Fetch initial state (move 0)
        const initialStateResponse = await service.getStateAtMove(gameId, 0);

        // Fetch all moves
        const movesResponse = await service.getMoves(gameId, 0, undefined, metadata.totalMoves);

        setState({
          gameId,
          metadata,
          currentMoveNumber: 0,
          totalMoves: metadata.totalMoves,
          currentState: initialStateResponse.gameState,
          isPlaying: false,
          playbackSpeed: 1,
          isLoading: false,
          error: null,
          moves: movesResponse.moves,
        });

        // Prefetch adjacent states
        prefetchAdjacent(gameId, 0, metadata.totalMoves);
      } catch (err) {
        setState((s) => ({
          ...s,
          isLoading: false,
          error: err instanceof Error ? err.message : 'Failed to load game',
        }));
      }
    },
    [service, prefetchAdjacent]
  );

  /**
   * Unload the current game.
   */
  const unloadGame = useCallback(() => {
    if (playIntervalRef.current) {
      clearTimeout(playIntervalRef.current);
      playIntervalRef.current = null;
    }
    setState(INITIAL_STATE);
  }, []);

  /**
   * Jump to a specific move number.
   * Uses cached state if available for instant navigation.
   */
  const jumpToMove = useCallback(
    async (moveNumber: number) => {
      const { gameId, totalMoves, isPlaying, playbackSpeed } = state;
      if (!gameId) return;

      // Clamp to valid range
      const targetMove = Math.max(0, Math.min(moveNumber, totalMoves));

      // Check if we have a cached state for instant navigation
      const cachedState = getCachedState(gameId, targetMove);

      if (cachedState) {
        // Instant update from cache - no loading state needed
        setState((s) => ({
          ...s,
          currentMoveNumber: targetMove,
          currentState: cachedState,
          isLoading: false,
        }));

        // Prefetch adjacent states in background
        prefetchAdjacent(gameId, targetMove, totalMoves, {
          isPlaying,
          speed: playbackSpeed,
        });
        return;
      }

      // No cache - need to fetch
      setState((s) => ({ ...s, isLoading: true }));

      try {
        const newState = await fetchState(gameId, targetMove);
        setState((s) => ({
          ...s,
          currentMoveNumber: targetMove,
          currentState: newState,
          isLoading: false,
        }));

        // Prefetch adjacent
        prefetchAdjacent(gameId, targetMove, totalMoves, {
          isPlaying,
          speed: playbackSpeed,
        });
      } catch (err) {
        setState((s) => ({
          ...s,
          isLoading: false,
          error: err instanceof Error ? err.message : 'Failed to load state',
        }));
      }
    },
    [state, fetchState, prefetchAdjacent, getCachedState]
  );

  /**
   * Step forward one move.
   */
  const stepForward = useCallback(async () => {
    if (state.currentMoveNumber < state.totalMoves) {
      await jumpToMove(state.currentMoveNumber + 1);
    }
  }, [state.currentMoveNumber, state.totalMoves, jumpToMove]);

  /**
   * Step backward one move.
   */
  const stepBackward = useCallback(async () => {
    if (state.currentMoveNumber > 0) {
      await jumpToMove(state.currentMoveNumber - 1);
    }
  }, [state.currentMoveNumber, jumpToMove]);

  /**
   * Jump to start.
   */
  const jumpToStart = useCallback(async () => {
    await jumpToMove(0);
  }, [jumpToMove]);

  /**
   * Jump to end.
   */
  const jumpToEnd = useCallback(async () => {
    await jumpToMove(state.totalMoves);
  }, [state.totalMoves, jumpToMove]);

  /**
   * Ref to track if we should continue playing.
   * Using a ref so the play loop can check it synchronously.
   */
  const isPlayingRef = useRef(false);

  /**
   * Ref to track current speed for the play loop.
   */
  const speedRef = useRef<PlaybackSpeed>(1);

  // Keep refs in sync with state
  useEffect(() => {
    isPlayingRef.current = state.isPlaying;
  }, [state.isPlaying]);

  useEffect(() => {
    speedRef.current = state.playbackSpeed;
  }, [state.playbackSpeed]);

  /**
   * Start auto-play using a recursive setTimeout pattern.
   * This is more reliable than setInterval because:
   * 1. It waits for the async fetch to complete before scheduling next
   * 2. It doesn't drift over time
   * 3. It can cleanly stop mid-iteration
   */
  const play = useCallback(() => {
    if (state.currentMoveNumber >= state.totalMoves) {
      return; // Already at end
    }

    setState((s) => ({ ...s, isPlaying: true }));
    isPlayingRef.current = true;

    // Clear any existing timeout
    if (playIntervalRef.current) {
      clearTimeout(playIntervalRef.current);
      playIntervalRef.current = null;
    }

    // Recursive play loop
    const playLoop = async () => {
      // Check if we should still be playing
      if (!isPlayingRef.current) {
        return;
      }

      const startTime = performance.now();

      // Step forward
      await stepForward();

      // Check again after async operation
      if (!isPlayingRef.current) {
        return;
      }

      // Get current state to check if we've reached the end
      setState((current) => {
        if (current.currentMoveNumber >= current.totalMoves) {
          // Reached end, stop playing
          isPlayingRef.current = false;
          return { ...current, isPlaying: false };
        }
        return current;
      });

      // If still playing, schedule next iteration
      if (isPlayingRef.current) {
        const elapsed = performance.now() - startTime;
        const targetDelay = Math.max(MIN_DELAY_MS, getDelayMs(speedRef.current));
        const remainingDelay = Math.max(0, targetDelay - elapsed);

        playIntervalRef.current = setTimeout(playLoop, remainingDelay);
      }
    };

    // Start the first iteration after initial delay
    playIntervalRef.current = setTimeout(playLoop, getDelayMs(state.playbackSpeed));
  }, [state.currentMoveNumber, state.totalMoves, state.playbackSpeed, stepForward]);

  /**
   * Pause auto-play.
   */
  const pause = useCallback(() => {
    isPlayingRef.current = false;
    setState((s) => ({ ...s, isPlaying: false }));
    if (playIntervalRef.current) {
      clearTimeout(playIntervalRef.current);
      playIntervalRef.current = null;
    }
  }, []);

  /**
   * Toggle play/pause.
   */
  const togglePlay = useCallback(() => {
    if (state.isPlaying) {
      pause();
    } else {
      play();
    }
  }, [state.isPlaying, play, pause]);

  /**
   * Set playback speed.
   * The play loop automatically picks up the new speed via speedRef.
   */
  const setSpeed = useCallback((speed: PlaybackSpeed) => {
    setState((s) => ({ ...s, playbackSpeed: speed }));
    // speedRef is updated via useEffect, play loop will use new speed on next iteration
  }, []);

  /**
   * Get the move record at current position.
   */
  const getCurrentMove = useCallback((): ReplayMoveRecord | null => {
    if (state.currentMoveNumber === 0 || state.moves.length === 0) {
      return null;
    }
    return state.moves[state.currentMoveNumber - 1] ?? null;
  }, [state.currentMoveNumber, state.moves]);

  return {
    ...state,
    loadGame,
    unloadGame,
    stepForward,
    stepBackward,
    jumpToMove,
    jumpToStart,
    jumpToEnd,
    play,
    pause,
    togglePlay,
    setSpeed,
    getCurrentMove,
    canStepForward: state.currentMoveNumber < state.totalMoves,
    canStepBackward: state.currentMoveNumber > 0,
  };
}
