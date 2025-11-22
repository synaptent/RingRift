import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';
import { toast } from 'react-hot-toast';
import { BoardState, GameState, Move, PlayerChoice, GameResult } from '../../shared/types/game';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

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

// Derive the WebSocket base URL from environment configuration, falling back
// to a sensible dev default. In local development:
//   - Vite runs on http://localhost:5173
//   - The backend (Express + Socket.IO) runs on http://localhost:3000
// We therefore prefer talking to the backend origin directly rather than
// relying on the Vite proxy for WebSocket connections.
function getSocketBaseUrl(): string {
  const env = (import.meta as any).env ?? {};

  // Prefer an explicit WebSocket URL when provided.
  const wsUrl = env.VITE_WS_URL as string | undefined;
  if (wsUrl) {
    return wsUrl.replace(/\/$/, '');
  }

  // Next, derive from an API URL by stripping any trailing "/api".
  const apiUrl = env.VITE_API_URL as string | undefined;
  if (apiUrl) {
    const base = apiUrl.replace(/\/?api\/?$/, '');
    return base.replace(/\/$/, '');
  }

  // In the browser (Vite dev, built client), detect the common local dev
  // case (frontend on :5173, backend on :3000) and talk to the backend
  // origin directly. For any other origin, just reuse window.location.origin.
  if (typeof window !== 'undefined' && window.location?.origin) {
    const origin = window.location.origin;
    if (origin.startsWith('http://localhost:5173') || origin.startsWith('https://localhost:5173')) {
      return 'http://localhost:3000';
    }
    return origin;
  }

  // Fallback for tests/SSR when no window is available.
  return 'http://localhost:3000';
}

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
  const socketRef = useRef<Socket | null>(null);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
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
      // If already connected to this game, do nothing.
      if (gameId === targetGameId && socketRef.current) {
        return;
      }

      // Tear down any existing connection.
      disconnect();

      setIsConnecting(true);
      setError(null);
      setConnectionStatus('connecting');

      try {
        const token = localStorage.getItem('token');
        const baseUrl = getSocketBaseUrl();

        const socket = io(baseUrl, {
          transports: ['websocket', 'polling'],
          auth: token ? { token } : undefined,
        });

        socketRef.current = socket;

        socket.on('connect_error', (err: Error) => {
          console.error('Socket connect_error', err);
          const msg = err.message || 'Failed to connect to game server';
          setError(msg);
          toast.error(msg);
          setIsConnecting(false);
          setConnectionStatus('disconnected');
          setLastHeartbeatAt(null);
        });

        socket.on('connect', () => {
          setConnectionStatus('connected');
          // After connecting, join the specific game room.
          socket.emit('join_game', { gameId: targetGameId });
        });

        socket.on('reconnect_attempt', () => {
          setIsConnecting(true);
          setConnectionStatus('reconnecting');
          toast('Reconnecting...', { icon: '🔄', id: 'reconnecting' });
        });

        socket.on('reconnect', () => {
          setIsConnecting(false);
          setConnectionStatus('connected');
          toast.success('Reconnected!', { id: 'reconnecting' });
          // Re-join the game room and request latest state
          socket.emit('join_game', { gameId: targetGameId });
        });

        // Handle explicit reconnection requests from the server or client logic
        socket.on('request_reconnect', () => {
          console.log('Server requested reconnection/resync');
          socket.emit('join_game', { gameId: targetGameId });
        });

        socket.on('game_state', (payload: any) => {
          // Payload shape from server: { type: 'game_update', data: { gameId, gameState, validMoves }, timestamp }
          const { data } = payload || {};
          if (data?.gameId === targetGameId && data?.gameState) {
            setGameId(targetGameId);
            setGameState(hydrateGameState(data.gameState));
            setValidMoves(Array.isArray(data.validMoves) ? data.validMoves : null);
            setIsConnecting(false);
            setError(null);
            setConnectionStatus('connected');
            setLastHeartbeatAt(Date.now());
          }
        });

        // Terminal game event carrying the final GameResult and snapshot.
        socket.on('game_over', (payload: any) => {
          const { data } = payload || {};
          if (!data || data.gameId !== targetGameId) return;

          setGameId(targetGameId);
          if (data.gameState) {
            setGameState(hydrateGameState(data.gameState));
          }
          if (data.gameResult) {
            setVictoryState(data.gameResult as GameResult);
          }
          setValidMoves(null);
          setIsConnecting(false);
          setError(null);
          // Any pending choices are no longer relevant once the game ends.
          setPendingChoice(null);
          setChoiceDeadline(null);
        });

        // Choice system events
        socket.on('player_choice_required', (choice: PlayerChoice) => {
          setPendingChoice(choice);
          const deadline = choice.timeoutMs ? Date.now() + choice.timeoutMs : null;
          setChoiceDeadline(deadline);
        });

        socket.on('player_choice_canceled', (choiceId: string) => {
          setPendingChoice((current) => (current && current.id === choiceId ? null : current));
          setChoiceDeadline((current) => (current ? null : current));
        });

        socket.on('chat_message', (payload: any) => {
          // Payload: { sender: string, text: string }
          setChatMessages((prev) => [...prev, payload]);
        });

        socket.on('error', (payload: any) => {
          const message = payload?.message || 'Game error';
          console.error('Game socket error', payload);
          setError(message);
          toast.error(message);
          setIsConnecting(false);
        });

        socket.on('disconnect', (reason) => {
          console.log('Socket disconnected:', reason);
          // If the disconnection was initiated by the server or network,
          // we want to keep the socketRef so auto-reconnect can work.
          // Only clear socketRef if we explicitly called disconnect().
          if (reason === 'io client disconnect') {
            socketRef.current = null;
          }

          setConnectionStatus('disconnected');
          setLastHeartbeatAt(null);

          if (reason !== 'io client disconnect') {
            // Attempt to reconnect immediately if it wasn't an intentional disconnect
            // Note: Socket.IO's auto-reconnect will handle the actual connection retry,
            // but we update UI state here.
            setConnectionStatus('reconnecting');
          }
        });
      } catch (err: any) {
        console.error('Failed to connect to game', err);
        setError(err?.message || 'Failed to connect to game');
        setIsConnecting(false);
        setConnectionStatus('disconnected');
      }
    },
    [gameId, disconnect]
  );

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const respondToChoice = useCallback(
    (choice: PlayerChoice, selectedOption: any) => {
      const socket = socketRef.current;
      if (!socket || !gameId) {
        console.warn('respondToChoice called without active socket/game');
        return;
      }

      // When an option carries a canonical moveId, we prefer the Move-driven
      // decision path by submitting a player_move_by_id request. This ensures
      // that all decisions are recorded as canonical Moves in the history.
      let moveId: string | undefined;

      if (
        choice.type === 'line_order' ||
        choice.type === 'region_order' ||
        choice.type === 'ring_elimination'
      ) {
        // These types have options as objects which may contain a moveId.
        moveId =
          selectedOption && typeof (selectedOption as any).moveId === 'string'
            ? (selectedOption as any).moveId
            : undefined;
      } else if (choice.type === 'line_reward_option') {
        // This type has options as strings, but the choice object itself
        // carries a map of option strings to moveIds.
        const optionKey = selectedOption as string;
        moveId = choice.moveIds?.[optionKey as keyof typeof choice.moveIds];
      }

      if (moveId) {
        socket.emit('player_move_by_id', { gameId, moveId });
      } else {
        const response = {
          choiceId: choice.id,
          playerNumber: choice.playerNumber,
          choiceType: choice.type,
          selectedOption,
        };

        socket.emit('player_choice_response', response);
      }

      setPendingChoice(null);
      setChoiceDeadline(null);
    },
    [gameId]
  );

  const submitMove = useCallback(
    (partialMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>) => {
      const socket = socketRef.current;
      if (!socket || !gameId) {
        console.warn('submitMove called without active socket/game');
        return;
      }

      // Bridge between the richer Move type used by GameEngine and the
      // current WebSocket/DB schema expected by handlePlayerMove. This is a
      // transitional implementation and should be revisited when the
      // WebSocket layer is refactored to speak Move directly.
      const movePayload = {
        moveNumber: (gameState?.moveHistory.length ?? 0) + 1,
        position: JSON.stringify({ from: partialMove.from, to: partialMove.to }),
        moveType: partialMove.type,
      };

      socket.emit('player_move', {
        gameId,
        move: movePayload,
      });
    },
    [gameId, gameState]
  );

  const sendChatMessage = useCallback(
    (text: string) => {
      const socket = socketRef.current;
      if (!socket || !gameId) {
        console.warn('sendChatMessage called without active socket/game');
        return;
      }

      socket.emit('chat_message', {
        gameId,
        text,
      });
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
