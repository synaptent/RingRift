import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { toast } from 'react-hot-toast';
import { reportClientError, isErrorReportingEnabled } from '../utils/errorReporting';
import { SocketGameConnection } from '../services/GameConnection';
import type {
  GameConnection,
  GameEventHandlers,
  ConnectionStatus as DomainConnectionStatus,
} from '../domain/GameAPI';
import { BoardState, GameState, Move, PlayerChoice, GameResult } from '../../shared/types/game';
import type {
  WebSocketErrorPayload,
  GameStateUpdateMessage,
  GameOverMessage,
  ChatMessageServerPayload,
} from '../../shared/types/websocket';

export type ConnectionStatus = DomainConnectionStatus;

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
  respondToChoice: (choice: PlayerChoice, selectedOption: any) => void;

  // Move submission (backend game mode)
  submitMove: (partialMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>) => void;

  // Chat
  sendChatMessage: (text: string) => void;
  chatMessages: { sender: string; text: string }[];
  connectionStatus: ConnectionStatus;
  /** Timestamp of the most recent game_state heartbeat (ms since epoch). */
  lastHeartbeatAt: number | null;
}

const GameContext = createContext<GameContextType | null>(null);

// Hydrate a BoardState coming over the wire, where Maps have been
// serialized to plain objects.
function hydrateBoardState(rawBoard: any): BoardState {
  if (!rawBoard) {
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

  const stacks = new Map<string, any>(Object.entries(rawBoard.stacks || {}));
  const markers = new Map<string, any>(Object.entries(rawBoard.markers || {}));
  const collapsedSpaces = new Map<string, number>(
    Object.entries(rawBoard.collapsedSpaces || {}) as [string, number][]
  );
  const territories = new Map<string, any>(Object.entries(rawBoard.territories || {}));

  return {
    stacks,
    markers,
    collapsedSpaces,
    territories,
    formedLines: rawBoard.formedLines || [],
    eliminatedRings: rawBoard.eliminatedRings || {},
    size: rawBoard.size,
    type: rawBoard.type,
  };
}

function hydrateGameState(rawState: any): GameState {
  return {
    ...rawState,
    board: hydrateBoardState(rawState.board),
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
  const connectionRef = useRef<GameConnection | null>(null);
  const lastStatusRef = useRef<ConnectionStatus | null>(null);
  const hasEverConnectedRef = useRef(false);

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
      },
      onChoiceRequired: (choice: PlayerChoice) => {
        setPendingChoice(choice);
        const deadline = choice.timeoutMs ? Date.now() + choice.timeoutMs : null;
        setChoiceDeadline(deadline);
      },
      onChoiceCanceled: (choiceId: string) => {
        setPendingChoice((current) => (current && current.id === choiceId ? null : current));
        setChoiceDeadline((current) => (current ? null : current));
      },
      onChatMessage: (payload: ChatMessageServerPayload) => {
        setChatMessages((prev) => [...prev, { sender: payload.sender, text: payload.text }]);
      },
      onError: (payload: WebSocketErrorPayload | unknown) => {
        let message: string;

        if (
          payload &&
          (payload as WebSocketErrorPayload).type === 'error' &&
          (payload as WebSocketErrorPayload).code
        ) {
          const err = payload as WebSocketErrorPayload;
          // eslint-disable-next-line no-console
          console.warn('Game socket error', err.code, err.event, err.message);
          message = err.message || 'Game error';
        } else {
          // eslint-disable-next-line no-console
          console.error('Game socket error', payload);
          message = (payload as any)?.message || 'Game error';
        }

        setError(message);
        toast.error(message);
        setIsConnecting(false);

        if (isErrorReportingEnabled()) {
          void reportClientError(payload, {
            type: 'game_socket_error',
            code: (payload as any)?.code,
            event: (payload as any)?.event,
            gameId,
            message,
          });
        }
      },
      onDisconnect: (reason: string) => {
        // eslint-disable-next-line no-console
        console.log('Socket disconnected:', reason);
        setLastHeartbeatAt(null);
        if (reason !== 'io client disconnect' && isErrorReportingEnabled()) {
          void reportClientError(new Error(`Socket disconnected: ${reason}`), {
            type: 'socket_disconnect',
            gameId,
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
          if (hasEverConnectedRef.current && previous === 'reconnecting') {
            toast.success('Reconnected!', { id: 'reconnecting' });
          }
          hasEverConnectedRef.current = true;
        }
      },
    };

    const connection = new SocketGameConnection(handlers);
    connectionRef.current = connection;
    return connection;
  }, [gameId]);

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
  }, []);

  const connectToGame = useCallback(
    async (targetGameId: string) => {
      const connection = getConnection();

      if (gameId === targetGameId && connection.status !== 'disconnected') {
        return;
      }

      // Tear down any existing connection state.
      disconnect();

      setIsConnecting(true);
      setError(null);
      setConnectionStatus('connecting');

      try {
        await connection.connect(targetGameId);
      } catch (err: any) {
        // Any connection errors should also surface via onError, but we
        // defensively set local state here as well.
        // eslint-disable-next-line no-console
        console.error('Failed to connect to game', err);
        setError(err?.message || 'Failed to connect to game');
        setIsConnecting(false);
        setConnectionStatus('disconnected');
      }
    },
    [gameId, disconnect, getConnection]
  );
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (connectionRef.current) {
        connectionRef.current.disconnect();
      }
    };
  }, []);

  const respondToChoice = useCallback(
    (choice: PlayerChoice, selectedOption: any) => {
      const connection = connectionRef.current;
      if (!connection || !gameId) {
        // eslint-disable-next-line no-console
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
        // eslint-disable-next-line no-console
        console.warn('submitMove called without active connection/game');
        return;
      }

      const move: Move = {
        id: '',
        type: partialMove.type,
        player: partialMove.player,
        from: partialMove.from,
        to: partialMove.to,
        captureTarget: (partialMove as any).captureTarget,
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
        // eslint-disable-next-line no-console
        console.warn('sendChatMessage called without active connection/game');
        return;
      }

      connection.sendChatMessage(text);
    },
    [gameId]
  );

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
