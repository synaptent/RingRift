/**
 * useMatchmaking Hook
 *
 * Manages matchmaking queue state and provides actions to join/leave the queue.
 * This hook uses socket.io to communicate with the server's MatchmakingService.
 *
 * Features:
 * - Join matchmaking queue with preferences
 * - Leave matchmaking queue
 * - Track queue position and estimated wait time
 * - Auto-navigate to game when match is found
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { io, Socket } from 'socket.io-client';
import type {
  MatchmakingPreferences,
  MatchmakingStatus,
  MatchFoundPayload,
  ClientToServerEvents,
  ServerToClientEvents,
} from '../../shared/types/websocket';
import { readEnv } from '../../shared/utils/envFlags';

/**
 * Current state of the matchmaking queue
 */
export interface MatchmakingState {
  /** Whether the user is currently in the matchmaking queue */
  inQueue: boolean;
  /** Estimated wait time in milliseconds */
  estimatedWaitTime: number | null;
  /** Current position in the queue */
  queuePosition: number | null;
  /** Search criteria used when joining the queue */
  searchCriteria: MatchmakingPreferences | null;
  /** Whether a match has been found (transitioning to game) */
  matchFound: boolean;
  /** ID of the game that was matched (if match found) */
  matchedGameId: string | null;
  /** Error message from matchmaking attempt */
  error: string | null;
  /** Whether the socket is currently connecting */
  isConnecting: boolean;
}

/**
 * Actions for matchmaking
 */
export interface MatchmakingActions {
  /** Join the matchmaking queue with the given preferences */
  joinQueue: (preferences: MatchmakingPreferences) => void;
  /** Leave the matchmaking queue */
  leaveQueue: () => void;
  /** Clear any matchmaking error */
  clearError: () => void;
}

/**
 * Full matchmaking state and actions
 */
export type UseMatchmakingResult = MatchmakingState & MatchmakingActions;

function getSocketBaseUrl(): string {
  const wsUrl = readEnv('VITE_WS_URL');
  if (wsUrl) return wsUrl.replace(/\/$/, '');

  const apiUrl = readEnv('VITE_API_URL');
  if (apiUrl) {
    const base = apiUrl.replace(/\/?api\/?$/, '');
    return base.replace(/\/$/, '');
  }

  if (typeof window !== 'undefined' && window.location?.origin) {
    const origin = window.location.origin;
    if (origin.startsWith('http://localhost:5173') || origin.startsWith('https://localhost:5173')) {
      return 'http://localhost:3000';
    }
    return origin;
  }

  return 'http://localhost:3000';
}

/**
 * Hook for managing matchmaking queue state
 *
 * @param autoNavigate - Whether to automatically navigate to the game when a match is found (default: true)
 */
export function useMatchmaking(autoNavigate = true): UseMatchmakingResult {
  const navigate = useNavigate();
  const socketRef = useRef<Socket<ServerToClientEvents, ClientToServerEvents> | null>(null);

  const [state, setState] = useState<MatchmakingState>({
    inQueue: false,
    estimatedWaitTime: null,
    queuePosition: null,
    searchCriteria: null,
    matchFound: false,
    matchedGameId: null,
    error: null,
    isConnecting: false,
  });

  // Initialize socket connection
  const ensureSocket = useCallback((): Socket<ServerToClientEvents, ClientToServerEvents> => {
    if (socketRef.current?.connected) {
      return socketRef.current;
    }

    // Get auth token from localStorage
    const token = localStorage.getItem('accessToken');

    const socket = io(getSocketBaseUrl(), {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      auth: token ? { token } : undefined,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    }) as Socket<ServerToClientEvents, ClientToServerEvents>;

    // Set up event listeners
    socket.on('connect', () => {
      setState((prev) => ({ ...prev, isConnecting: false, error: null }));
    });

    socket.on('connect_error', (err) => {
      setState((prev) => ({
        ...prev,
        isConnecting: false,
        error: `Connection error: ${err.message}`,
      }));
    });

    socket.on('disconnect', () => {
      setState((prev) => ({
        ...prev,
        inQueue: false,
        isConnecting: false,
      }));
    });

    // Matchmaking events
    socket.on('matchmaking-status', (status: MatchmakingStatus) => {
      setState((prev) => ({
        ...prev,
        inQueue: status.inQueue,
        estimatedWaitTime: status.estimatedWaitTime,
        queuePosition: status.queuePosition,
        searchCriteria: status.searchCriteria,
      }));
    });

    socket.on('match-found', (payload: MatchFoundPayload) => {
      setState((prev) => ({
        ...prev,
        matchFound: true,
        matchedGameId: payload.gameId,
        inQueue: false,
      }));

      // Auto-navigate to game if enabled
      if (autoNavigate) {
        navigate(`/game/${payload.gameId}`);
      }
    });

    socket.on('error', (errorPayload) => {
      if (errorPayload.event === 'matchmaking:join' || errorPayload.event === 'matchmaking:leave') {
        setState((prev) => ({
          ...prev,
          error: errorPayload.message,
        }));
      }
    });

    socketRef.current = socket;
    return socket;
  }, [autoNavigate, navigate]);

  // Cleanup socket on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        // Leave queue before disconnecting
        if (state.inQueue) {
          socketRef.current.emit('matchmaking:leave');
        }
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [state.inQueue]);

  const joinQueue = useCallback(
    (preferences: MatchmakingPreferences) => {
      setState((prev) => ({
        ...prev,
        isConnecting: true,
        error: null,
        matchFound: false,
        matchedGameId: null,
      }));

      const socket = ensureSocket();
      socket.emit('matchmaking:join', { preferences });
    },
    [ensureSocket]
  );

  const leaveQueue = useCallback(() => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('matchmaking:leave');
    }

    setState((prev) => ({
      ...prev,
      inQueue: false,
      estimatedWaitTime: null,
      queuePosition: null,
      searchCriteria: null,
    }));
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  return {
    ...state,
    joinQueue,
    leaveQueue,
    clearError,
  };
}

export default useMatchmaking;
