import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { toast } from 'react-hot-toast';
import {
  reportClientError,
  isErrorReportingEnabled,
  extractErrorMessage,
} from '../utils/errorReporting';
import { SocketGameConnection } from '../services/GameConnection';
import type {
  GameConnection,
  GameEventHandlers,
  ConnectionStatus as DomainConnectionStatus,
} from '../domain/GameAPI';
import {
  BoardState,
  GameState,
  Move,
  PlayerChoice,
  GameResult,
  RingStack,
  MarkerInfo,
  Territory,
} from '../../shared/types/game';
import type {
  WebSocketErrorPayload,
  GameStateUpdateMessage,
  GameOverMessage,
  ChatMessageServerPayload,
  ChatMessagePersisted,
  ChatHistoryPayload,
  DecisionAutoResolvedMeta,
  DecisionPhaseTimeoutWarningPayload,
  RematchRequestPayload,
  RematchResponsePayload,
  PositionEvaluationPayload,
  PlayerDisconnectedPayload,
  PlayerReconnectedPayload,
} from '../../shared/types/websocket';

export type ConnectionStatus = DomainConnectionStatus;

/**
 * Information about a player who has disconnected from the game.
 * Used to show opponent disconnect banners in the UI.
 */
export interface DisconnectedPlayer {
  id: string;
  username: string;
  /** Timestamp when disconnect was detected (ms since epoch). */
  disconnectedAt: number;
}

interface GameContextType {
  gameId: string | null;
  gameState: GameState | null;
  /**
   * Optional list of valid moves for the current player, as provided by the
   * backend GameEngine in the latest game_state payload. Currently this is
   * primarily intended for highlighting legal targets in the UI.
   */
  validMoves: Move[] | null;
  isConnecting: boolean;
  error: string | null;
  /**
   * When defined, contains the terminal GameResult for the current game.
   * This is set in response to the server-emitted game_over event.
   */
  victoryState: GameResult | null;
  connectToGame: (gameId: string) => Promise<void>;
  disconnect: () => void;

  // Choice handling
  pendingChoice: PlayerChoice | null;
  /**
   * Deadline timestamp (ms since epoch) when the current choice will time out,
   * or null if no explicit timeout was provided by the server.
   */
  choiceDeadline: number | null;
  respondToChoice: (choice: PlayerChoice, selectedOption: unknown) => void;

  // Move submission (backend game mode)
  submitMove: (partialMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>) => void;

  // Chat
  sendChatMessage: (text: string) => void;
  chatMessages: { sender: string; text: string }[];
  connectionStatus: ConnectionStatus;
  /** Timestamp of the most recent game_state heartbeat (ms since epoch). */
  lastHeartbeatAt: number | null;
  /** Summary of the most recently auto-resolved decision, if any, for the latest update. */
  decisionAutoResolved: DecisionAutoResolvedMeta | null;
  /**
   * Optional payload for an in-progress decision-phase timeout warning, if the
   * server has indicated that a pending decision is approaching auto-resolution.
   */
  decisionPhaseTimeoutWarning: DecisionPhaseTimeoutWarningPayload | null;

  // Rematch
  /** Pending rematch request, if any. */
  pendingRematchRequest: RematchRequestPayload | null;
  /** Request a rematch for the current completed game. */
  requestRematch: () => void;
  /** Accept a pending rematch request. */
  acceptRematch: (requestId: string) => void;
  /** Decline a pending rematch request. */
  declineRematch: (requestId: string) => void;
  /** ID of the new game created from an accepted rematch, if any. */
  rematchGameId: string | null;
  /** Last terminal status received for a rematch request, if any. */
  rematchLastStatus: 'accepted' | 'declined' | 'expired' | null;
  /** Streaming AI evaluation history for the current game (analysis mode). */
  evaluationHistory: PositionEvaluationPayload['data'][];

  // Opponent connection state (Wave 2.2)
  /**
   * List of players who have disconnected from the game but may still reconnect.
   * Cleared when game ends or when player_reconnected is received.
   */
  disconnectedOpponents: DisconnectedPlayer[];
  /**
   * Derived flag: true if the game ended due to abandonment (reconnection timeout).
   * Consumers can use this to show appropriate game-over messaging.
   */
  gameEndedByAbandonment: boolean;
}

const GameContext = createContext<GameContextType | null>(null);

// Hydrate a BoardState coming over the wire, where Maps have been
// serialized to plain objects.
function hydrateBoardState(rawBoard: Record<string, unknown> | null | undefined): BoardState {
  if (!rawBoard) {
    // RR-FIX-2026-01-15: Log when returning empty fallback board - this can cause blank board display.
    console.error(
      '[GameContext] hydrateBoardState received null/undefined rawBoard, returning empty fallback'
    );
    // Fallback empty board; callers should guard against this.
    return {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 0,
      type: 'square8',
    };
  }

  const rawStacks = (rawBoard.stacks || {}) as Record<string, RingStack>;
  const rawMarkers = (rawBoard.markers || {}) as Record<string, MarkerInfo>;
  const rawCollapsed = (rawBoard.collapsedSpaces || {}) as Record<string, number>;
  const rawTerritories = (rawBoard.territories || {}) as Record<string, Territory>;

  const stacks = new Map<string, RingStack>(Object.entries(rawStacks));
  const markers = new Map<string, MarkerInfo>(Object.entries(rawMarkers));
  const collapsedSpaces = new Map<string, number>(Object.entries(rawCollapsed));
  const territories = new Map<string, Territory>(Object.entries(rawTerritories));

  // RR-DEBUG-2026-01-15: Log hydration details to diagnose blank board issues
  const boardSize = rawBoard.size as number;
  if (!boardSize || boardSize < 1) {
    console.error('[GameContext] hydrateBoardState: Invalid board size after hydration:', {
      size: boardSize,
      type: rawBoard.type,
      stacksCount: stacks.size,
      markersCount: markers.size,
      collapsedCount: collapsedSpaces.size,
      rawBoardKeys: Object.keys(rawBoard),
    });
  }

  return {
    stacks,
    markers,
    collapsedSpaces,
    territories,
    formedLines: (rawBoard.formedLines || []) as BoardState['formedLines'],
    eliminatedRings: (rawBoard.eliminatedRings || {}) as BoardState['eliminatedRings'],
    size: rawBoard.size as number,
    type: rawBoard.type as BoardState['type'],
  };
}

function hydrateGameState(rawState: GameState | Record<string, unknown>): GameState {
  // Handle both wire-serialized payloads and already-typed GameState objects
  const state = rawState as GameState & Record<string, unknown>;
  return {
    ...state,
    board: hydrateBoardState(state.board as unknown as Record<string, unknown>),
  } as GameState;
}

export function GameProvider({ children }: { children: React.ReactNode }) {
  const [gameId, setGameId] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [validMoves, setValidMoves] = useState<Move[] | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingChoice, setPendingChoice] = useState<PlayerChoice | null>(null);
  const [choiceDeadline, setChoiceDeadline] = useState<number | null>(null);
  const [victoryState, setVictoryState] = useState<GameResult | null>(null);
  const [chatMessages, setChatMessages] = useState<{ sender: string; text: string }[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [lastHeartbeatAt, setLastHeartbeatAt] = useState<number | null>(null);
  const [decisionAutoResolved, setDecisionAutoResolved] = useState<DecisionAutoResolvedMeta | null>(
    null
  );
  const [decisionPhaseTimeoutWarning, setDecisionPhaseTimeoutWarning] =
    useState<DecisionPhaseTimeoutWarningPayload | null>(null);
  const [pendingRematchRequest, setPendingRematchRequest] = useState<RematchRequestPayload | null>(
    null
  );
  const [rematchGameId, setRematchGameId] = useState<string | null>(null);
  const [rematchLastStatus, setRematchLastStatus] = useState<
    'accepted' | 'declined' | 'expired' | null
  >(null);
  const [evaluationHistory, setEvaluationHistory] = useState<PositionEvaluationPayload['data'][]>(
    []
  );
  const [disconnectedOpponents, setDisconnectedOpponents] = useState<DisconnectedPlayer[]>([]);
  const connectionRef = useRef<GameConnection | null>(null);
  const lastStatusRef = useRef<ConnectionStatus | null>(null);
  const hasEverConnectedRef = useRef(false);
  // Ref to track current gameId without causing connectToGame to re-create on gameId changes
  const gameIdRef = useRef<string | null>(null);

  // Keep gameIdRef in sync with state (for use in callbacks without deps)
  useEffect(() => {
    gameIdRef.current = gameId;
  }, [gameId]);

  const getConnection = useCallback((): GameConnection => {
    if (connectionRef.current) {
      return connectionRef.current;
    }

    const handlers: GameEventHandlers = {
      onGameState: (payload: GameStateUpdateMessage) => {
        const { data } = payload || ({} as GameStateUpdateMessage);
        if (!data?.gameId || !data.gameState) return;

        setGameId(data.gameId);
        setGameState(hydrateGameState(data.gameState));
        setValidMoves(Array.isArray(data.validMoves) ? data.validMoves : null);
        setIsConnecting(false);
        setError(null);
        setConnectionStatus('connected');
        setLastHeartbeatAt(Date.now());
        // Any fresh game_state snapshot is treated as authoritative and
        // clears any previously pending choices/deadlines so the HUD does
        // not display stale decision banners after a reconnect or resync.
        setPendingChoice(null);
        setChoiceDeadline(null);
        // Each fresh game_state diff summary replaces any prior auto-resolve
        // metadata and clears outstanding timeout warnings.
        setDecisionAutoResolved(data.meta?.diffSummary?.decisionAutoResolved ?? null);
        setDecisionPhaseTimeoutWarning(null);
      },
      onGameOver: (payload: GameOverMessage) => {
        const { data } = payload || ({} as GameOverMessage);
        if (!data?.gameId) return;

        setGameId(data.gameId);
        if (data.gameState) {
          setGameState(hydrateGameState(data.gameState));
        }
        if (data.gameResult) {
          setVictoryState(data.gameResult as GameResult);
        }
        setValidMoves(null);
        setIsConnecting(false);
        setError(null);
        setPendingChoice(null);
        setChoiceDeadline(null);
        setDecisionAutoResolved(null);
        setDecisionPhaseTimeoutWarning(null);
      },
      onChoiceRequired: (choice: PlayerChoice) => {
        setPendingChoice(choice);
        const deadline = choice.timeoutMs ? Date.now() + choice.timeoutMs : null;
        setChoiceDeadline(deadline);
      },
      onChoiceCanceled: (choiceId: string) => {
        // Only clear deadline if we're actually clearing the matching pending choice
        setPendingChoice((current) => {
          if (current && current.id === choiceId) {
            // Choice matches - clear it and also clear the deadline
            setChoiceDeadline(null);
            return null;
          }
          return current;
        });
      },
      onChatMessage: (payload: ChatMessageServerPayload) => {
        setChatMessages((prev) => [...prev, { sender: payload.sender, text: payload.text }]);
      },
      onError: (payload: WebSocketErrorPayload | unknown) => {
        let message: string;
        // Type guard for extracting optional properties from unknown payloads
        const asRecord = payload as Record<string, unknown> | null | undefined;

        if (
          payload &&
          (payload as WebSocketErrorPayload).type === 'error' &&
          (payload as WebSocketErrorPayload).code
        ) {
          const err = payload as WebSocketErrorPayload;
          console.warn('Game socket error', err.code, err.event, err.message);
          message = err.message || 'Game error';
        } else {
          console.error('Game socket error', payload);
          message =
            (typeof asRecord?.message === 'string' ? asRecord.message : null) || 'Game error';
        }

        setError(message);
        toast.error(message);
        setIsConnecting(false);

        if (isErrorReportingEnabled()) {
          void reportClientError(payload, {
            type: 'game_socket_error',
            code: typeof asRecord?.code === 'string' ? asRecord.code : undefined,
            event: typeof asRecord?.event === 'string' ? asRecord.event : undefined,
            gameId: gameIdRef.current,
            message,
          });
        }
      },
      onDisconnect: (reason: string) => {
        console.warn('Socket disconnected:', reason);
        setLastHeartbeatAt(null);
        if (reason !== 'io client disconnect' && isErrorReportingEnabled()) {
          void reportClientError(new Error(`Socket disconnected: ${reason}`), {
            type: 'socket_disconnect',
            gameId: gameIdRef.current,
            reason,
          });
        }
      },
      onConnectionStatusChange: (status: ConnectionStatus) => {
        const previous = lastStatusRef.current;
        lastStatusRef.current = status;

        setConnectionStatus(status);
        setIsConnecting(status === 'connecting' || status === 'reconnecting');

        if (status === 'reconnecting') {
          toast('Reconnecting...', { icon: '🔄', id: 'reconnecting' });
        } else if (status === 'connected') {
          // Clear any connection-related error when we successfully reconnect.
          // While onGameState will also clear the error, this provides more
          // immediate feedback and handles edge cases where game_state may
          // be delayed after reconnection.
          if (previous === 'reconnecting') {
            setError(null);
          }
          if (hasEverConnectedRef.current && previous === 'reconnecting') {
            toast.success('Reconnected!', { id: 'reconnecting' });
          }
          hasEverConnectedRef.current = true;
        }
      },
      onDecisionPhaseTimeoutWarning: (payload) => {
        // Surface the latest timeout warning; BackendGameHost and other
        // consumers can use this for HUD banners and diagnostics logs.
        setDecisionPhaseTimeoutWarning(payload);
      },
      onDecisionPhaseTimedOut: () => {
        // Once a decision has timed out and been auto-resolved, clear any
        // outstanding warning; the resulting game_state update carries the
        // decisionAutoResolved metadata instead.
        setDecisionPhaseTimeoutWarning(null);
      },
      onChatMessagePersisted: (payload: ChatMessagePersisted) => {
        // Prefer persisted messages over legacy chat_message when both are emitted.
        // The server emits both for backward compatibility; we dedupe here by
        // checking if we already have a message with this sender+text recently.
        setChatMessages((prev) => {
          const isDuplicate = prev.some(
            (m) =>
              m.sender === payload.username &&
              m.text === payload.message &&
              prev.indexOf(m) >= prev.length - 3 // Only check recent messages
          );
          if (isDuplicate) return prev;
          return [...prev, { sender: payload.username, text: payload.message }];
        });
      },
      onChatHistory: (payload: ChatHistoryPayload) => {
        // Prepend historical messages to the chat, avoiding duplicates
        const historicalMessages = payload.messages.map((m) => ({
          sender: m.username,
          text: m.message,
        }));
        setChatMessages((prev) => {
          // Only add messages that aren't already present
          const existingKeys = new Set(prev.map((m) => `${m.sender}:${m.text}`));
          const newMessages = historicalMessages.filter(
            (m) => !existingKeys.has(`${m.sender}:${m.text}`)
          );
          return [...newMessages, ...prev];
        });
      },
      onRematchRequested: (payload: RematchRequestPayload) => {
        setPendingRematchRequest(payload);
        setRematchGameId(null); // Clear any previous rematch game ID
        setRematchLastStatus(null);
      },
      onRematchResponse: (payload: RematchResponsePayload) => {
        if (payload.status === 'accepted' && payload.newGameId) {
          setRematchGameId(payload.newGameId);
          setRematchLastStatus('accepted');
          toast.success('Rematch accepted! Redirecting to new game...');
        } else if (payload.status === 'declined') {
          setRematchLastStatus('declined');
          toast('Rematch declined', { icon: '❌' });
        } else if (payload.status === 'expired') {
          setRematchLastStatus('expired');
          toast('Rematch request expired', { icon: '⏰' });
        }
        setPendingRematchRequest(null);
      },
      onPositionEvaluation: (payload: PositionEvaluationPayload) => {
        const { data } = payload || ({} as PositionEvaluationPayload);
        if (!data?.gameId) return;
        setEvaluationHistory((prev) => [...prev, data]);
      },
      onPlayerDisconnected: (payload: PlayerDisconnectedPayload) => {
        // Add the disconnected player to our tracking list
        const { player, reconnectionWindowMs } = payload.data;
        setDisconnectedOpponents((prev) => {
          // Avoid duplicates
          if (prev.some((p) => p.id === player.id)) return prev;
          return [
            ...prev,
            {
              id: player.id,
              username: player.username ?? 'Player',
              disconnectedAt: Date.now(),
            },
          ];
        });
        // Show a toast notification for UX with reconnection window info
        const windowSecs = reconnectionWindowMs ? Math.round(reconnectionWindowMs / 1000) : 30;
        toast(`${player.username ?? 'A player'} disconnected (${windowSecs}s to reconnect)`, {
          icon: '⚠️',
          id: `disconnect-${player.id}`,
          duration: 5000,
        });
      },
      onPlayerReconnected: (payload: PlayerReconnectedPayload) => {
        // Remove the reconnected player from our tracking list
        const { player } = payload.data;
        setDisconnectedOpponents((prev) => prev.filter((p) => p.id !== player.id));
        // Show a toast notification for UX
        toast.success(`${player.username ?? 'Player'} reconnected`, {
          id: `disconnect-${player.id}`,
        });
      },
    };

    const connection = new SocketGameConnection(handlers);
    connectionRef.current = connection;
    return connection;
  }, []);

  const disconnect = useCallback(() => {
    if (connectionRef.current) {
      connectionRef.current.disconnect();
    }
    setGameId(null);
    setGameState(null);
    setValidMoves(null);
    setIsConnecting(false);
    setError(null);
    setPendingChoice(null);
    setChoiceDeadline(null);
    setVictoryState(null);
    setChatMessages([]);
    setConnectionStatus('disconnected');
    setLastHeartbeatAt(null);
    setDecisionAutoResolved(null);
    setDecisionPhaseTimeoutWarning(null);
    setPendingRematchRequest(null);
    setRematchGameId(null);
    setRematchLastStatus(null);
    setEvaluationHistory([]);
    setDisconnectedOpponents([]);
  }, []);

  const connectToGame = useCallback(
    async (targetGameId: string) => {
      const connection = getConnection();

      // Use ref instead of state to avoid re-creating this callback when gameId changes
      if (gameIdRef.current === targetGameId && connection.status !== 'disconnected') {
        return;
      }

      // Tear down any existing connection state.
      disconnect();

      setIsConnecting(true);
      setError(null);
      setConnectionStatus('connecting');

      try {
        await connection.connect(targetGameId);
      } catch (error: unknown) {
        // Any connection errors should also surface via onError, but we
        // defensively set local state here as well.

        console.error('Failed to connect to game', error);
        setError(extractErrorMessage(error, 'Failed to connect to game'));
        setIsConnecting(false);
        setConnectionStatus('disconnected');
      }
    },
    [disconnect, getConnection]
  );
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (connectionRef.current) {
        connectionRef.current.disconnect();
      }
    };
  }, []);

  // Show toast when a decision is auto-resolved due to timeout
  useEffect(() => {
    if (decisionAutoResolved) {
      const reason =
        decisionAutoResolved.reason === 'timeout'
          ? 'timed out'
          : decisionAutoResolved.reason === 'disconnected'
            ? 'disconnected'
            : 'auto-resolved';
      toast(`Decision ${reason} - move applied automatically`, {
        icon: '⏱️',
        id: 'decision-auto-resolved',
        duration: 4000,
      });
    }
  }, [decisionAutoResolved]);

  const respondToChoice = useCallback(
    (choice: PlayerChoice, selectedOption: unknown) => {
      const connection = connectionRef.current;
      if (!connection || !gameId) {
        console.warn('respondToChoice called without active connection/game');
        return;
      }

      connection.respondToChoice(choice, selectedOption);
      setPendingChoice(null);
      setChoiceDeadline(null);
    },
    [gameId]
  );

  const submitMove = useCallback(
    (partialMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>) => {
      const connection = connectionRef.current;
      if (!connection || !gameId) {
        console.warn('submitMove called without active connection/game');
        return;
      }

      // Spread all fields from partialMove (includes placementCount, placedOnStack,
      // captureTarget, recoveryOption, etc.) and override auto-generated fields
      const move: Move = {
        ...partialMove,
        id: '',
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: (gameState?.moveHistory.length ?? 0) + 1,
      };

      connection.submitMove(move);
    },
    [gameId, gameState]
  );

  const sendChatMessage = useCallback(
    (text: string) => {
      const connection = connectionRef.current;
      if (!connection || !gameId) {
        console.warn('sendChatMessage called without active connection/game');
        return;
      }

      connection.sendChatMessage(text);
    },
    [gameId]
  );

  const requestRematch = useCallback(() => {
    const connection = connectionRef.current;
    if (!connection || !gameId) {
      console.warn('requestRematch called without active connection/game');
      return;
    }

    connection.requestRematch();
  }, [gameId]);

  const acceptRematch = useCallback((requestId: string) => {
    const connection = connectionRef.current;
    if (!connection) {
      console.warn('acceptRematch called without active connection');
      return;
    }

    connection.respondToRematch(requestId, true);
  }, []);

  const declineRematch = useCallback((requestId: string) => {
    const connection = connectionRef.current;
    if (!connection) {
      console.warn('declineRematch called without active connection');
      return;
    }

    connection.respondToRematch(requestId, false);
  }, []);

  const value: GameContextType = {
    gameId,
    gameState,
    isConnecting,
    error,
    victoryState,
    connectToGame,
    disconnect,
    pendingChoice,
    choiceDeadline,
    respondToChoice,
    submitMove,
    validMoves,
    sendChatMessage,
    chatMessages,
    connectionStatus,
    lastHeartbeatAt,
    decisionAutoResolved,
    decisionPhaseTimeoutWarning,
    pendingRematchRequest,
    requestRematch,
    acceptRematch,
    declineRematch,
    rematchGameId,
    rematchLastStatus,
    evaluationHistory,
    // Opponent connection state (Wave 2.2)
    disconnectedOpponents,
    gameEndedByAbandonment: victoryState?.reason === 'abandonment',
  };

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame(): GameContextType {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error('useGame must be used within a GameProvider');
  }
  return context;
}
