/**
 * ═══════════════════════════════════════════════════════════════════════════
 * useGameConnection Hook
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides connection status and management functions for WebSocket-based
 * game connections. This hook wraps GameContext to expose only connection-
 * related state and actions.
 *
 * Benefits:
 * - Clear separation of connection concerns from game state
 * - Focused API for connection management
 * - Easy to mock for testing components
 */

import { useMemo, useCallback } from 'react';
import { useGame, ConnectionStatus, DisconnectedPlayer } from '../contexts/GameContext';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Connection health status with derived flags
 */
export interface ConnectionHealth {
  /** Current connection status */
  status: ConnectionStatus;
  /** Whether actively connecting/reconnecting */
  isConnecting: boolean;
  /** Whether fully connected and healthy */
  isHealthy: boolean;
  /** Whether connection is stale (no recent heartbeat) */
  isStale: boolean;
  /** Whether disconnected */
  isDisconnected: boolean;
  /** Human-readable status label */
  statusLabel: string;
  /** Tailwind color class for status indicator */
  statusColorClass: string;
  /** Time since last heartbeat (ms), null if never received */
  timeSinceHeartbeat: number | null;
}

/**
 * Connection actions
 */
export interface ConnectionActions {
  /** Connect to a specific game by ID */
  connectToGame: (gameId: string) => Promise<void>;
  /** Disconnect from current game */
  disconnect: () => void;
}

/**
 * Full connection state and actions
 */
export interface GameConnectionState extends ConnectionHealth, ConnectionActions {
  /** Current game ID (null if not connected) */
  gameId: string | null;
  /** Error message from last connection attempt */
  error: string | null;
  /** Timestamp of last heartbeat */
  lastHeartbeatAt: number | null;
  /** Players who have disconnected but may reconnect */
  disconnectedOpponents: DisconnectedPlayer[];
  /** Whether the game ended due to abandonment (reconnection timeout expired) */
  gameEndedByAbandonment: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/** Heartbeat staleness threshold in milliseconds */
const HEARTBEAT_STALE_THRESHOLD_MS = 8000;

// ═══════════════════════════════════════════════════════════════════════════
// Hook Implementation
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing WebSocket connection to game server
 *
 * Usage:
 * ```tsx
 * const {
 *   status,
 *   isHealthy,
 *   isStale,
 *   connectToGame,
 *   disconnect,
 *   error
 * } = useGameConnection();
 *
 * useEffect(() => {
 *   if (gameId) {
 *     connectToGame(gameId);
 *   }
 *   return () => disconnect();
 * }, [gameId, connectToGame, disconnect]);
 *
 * if (!isHealthy) {
 *   return <ReconnectingBanner status={status} />;
 * }
 * ```
 */
export function useGameConnection(): GameConnectionState {
  const {
    gameId,
    connectionStatus,
    lastHeartbeatAt,
    error,
    isConnecting: _isConnecting,
    connectToGame,
    disconnect,
    disconnectedOpponents,
    gameEndedByAbandonment,
  } = useGame();

  // Derive connection health
  const connectionHealth = useMemo((): ConnectionHealth => {
    const now = Date.now();
    const timeSinceHeartbeat = lastHeartbeatAt ? now - lastHeartbeatAt : null;
    const isStale =
      timeSinceHeartbeat !== null &&
      timeSinceHeartbeat > HEARTBEAT_STALE_THRESHOLD_MS &&
      connectionStatus === 'connected';

    const isHealthy = connectionStatus === 'connected' && !isStale;
    const isDisconnected = connectionStatus === 'disconnected';

    // Status label
    let statusLabel: string;
    switch (connectionStatus) {
      case 'connected':
        statusLabel = isStale ? 'Awaiting update…' : 'Connected';
        break;
      case 'connecting':
        statusLabel = 'Connecting…';
        break;
      case 'reconnecting':
        statusLabel = 'Reconnecting…';
        break;
      case 'disconnected':
      default:
        statusLabel = 'Disconnected';
    }

    // Color class
    let statusColorClass: string;
    if (isHealthy) {
      statusColorClass = 'text-emerald-300';
    } else if (connectionStatus === 'reconnecting' || isStale) {
      statusColorClass = 'text-amber-300';
    } else {
      statusColorClass = 'text-rose-300';
    }

    return {
      status: connectionStatus,
      isConnecting: connectionStatus === 'connecting' || connectionStatus === 'reconnecting',
      isHealthy,
      isStale,
      isDisconnected,
      statusLabel,
      statusColorClass,
      timeSinceHeartbeat,
    };
  }, [connectionStatus, lastHeartbeatAt]);

  // Stable action references
  const stableConnectToGame = useCallback(
    async (targetGameId: string) => {
      await connectToGame(targetGameId);
    },
    [connectToGame]
  );

  const stableDisconnect = useCallback(() => {
    disconnect();
  }, [disconnect]);

  return {
    gameId,
    ...connectionHealth,
    connectToGame: stableConnectToGame,
    disconnect: stableDisconnect,
    error,
    lastHeartbeatAt,
    disconnectedOpponents,
    gameEndedByAbandonment,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useConnectionStatus (Lightweight)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Lightweight hook that only provides connection status (no actions)
 *
 * Usage:
 * ```tsx
 * const { isHealthy, statusLabel } = useConnectionStatus();
 *
 * return (
 *   <div className={isHealthy ? 'text-green-500' : 'text-red-500'}>
 *     {statusLabel}
 *   </div>
 * );
 * ```
 */
export function useConnectionStatus(): ConnectionHealth {
  const { connectionStatus, lastHeartbeatAt } = useGame();

  return useMemo((): ConnectionHealth => {
    const now = Date.now();
    const timeSinceHeartbeat = lastHeartbeatAt ? now - lastHeartbeatAt : null;
    const isStale =
      timeSinceHeartbeat !== null &&
      timeSinceHeartbeat > HEARTBEAT_STALE_THRESHOLD_MS &&
      connectionStatus === 'connected';

    const isHealthy = connectionStatus === 'connected' && !isStale;
    const isDisconnected = connectionStatus === 'disconnected';

    let statusLabel: string;
    switch (connectionStatus) {
      case 'connected':
        statusLabel = isStale ? 'Awaiting update…' : 'Connected';
        break;
      case 'connecting':
        statusLabel = 'Connecting…';
        break;
      case 'reconnecting':
        statusLabel = 'Reconnecting…';
        break;
      case 'disconnected':
      default:
        statusLabel = 'Disconnected';
    }

    let statusColorClass: string;
    if (isHealthy) {
      statusColorClass = 'text-emerald-300';
    } else if (connectionStatus === 'reconnecting' || isStale) {
      statusColorClass = 'text-amber-300';
    } else {
      statusColorClass = 'text-rose-300';
    }

    return {
      status: connectionStatus,
      isConnecting: connectionStatus === 'connecting' || connectionStatus === 'reconnecting',
      isHealthy,
      isStale,
      isDisconnected,
      statusLabel,
      statusColorClass,
      timeSinceHeartbeat,
    };
  }, [connectionStatus, lastHeartbeatAt]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hook: useIsConnected (Boolean)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Simple boolean check for connection status
 *
 * Usage:
 * ```tsx
 * const isConnected = useIsConnected();
 *
 * if (!isConnected) {
 *   return <NotConnectedMessage />;
 * }
 * ```
 */
export function useIsConnected(): boolean {
  const { connectionStatus, lastHeartbeatAt } = useGame();

  return useMemo(() => {
    if (connectionStatus !== 'connected') return false;

    // Check for stale heartbeat
    if (lastHeartbeatAt) {
      const timeSinceHeartbeat = Date.now() - lastHeartbeatAt;
      if (timeSinceHeartbeat > HEARTBEAT_STALE_THRESHOLD_MS) {
        return false;
      }
    }

    return true;
  }, [connectionStatus, lastHeartbeatAt]);
}
