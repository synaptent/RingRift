import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState
} from 'react';
import { io, Socket } from 'socket.io-client';
import { BoardState, GameState, Move, PlayerChoice, GameResult } from '../../shared/types/game';

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
  submitMove: (
    partialMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>
  ) => void;
}

const GameContext = createContext<GameContextType | null>(null);

// Derive the WebSocket base URL from the API URL, falling back to localhost:5000.
function getSocketBaseUrl(): string {
  // Access Vite env via a loose import.meta cast so this file stays portable
  // even if bundler-specific typings are not present in tsconfig.
  const apiUrl = (import.meta as any).env?.VITE_API_URL as string | undefined;
  if (apiUrl) {
    // Strip trailing "/api" if present
    const base = apiUrl.replace(/\/?api\/?$/, '');
    return base;
  }
  return 'http://localhost:5000';
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
      type: 'square8'
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
    type: rawBoard.type
  };
}

function hydrateGameState(rawState: any): GameState {
  return {
    ...rawState,
    board: hydrateBoardState(rawState.board)
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
  }, []);

  const connectToGame = useCallback(async (targetGameId: string) => {
    // If already connected to this game, do nothing.
    if (gameId === targetGameId && socketRef.current) {
      return;
    }

    // Tear down any existing connection.
    disconnect();

    setIsConnecting(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const baseUrl = getSocketBaseUrl();

      const socket = io(baseUrl, {
        transports: ['websocket', 'polling'],
        auth: token ? { token } : undefined
      });

      socketRef.current = socket;

      socket.on('connect_error', (err: Error) => {
        console.error('Socket connect_error', err);
        setError(err.message || 'Failed to connect to game server');
        setIsConnecting(false);
      });

      socket.on('connect', () => {
        // After connecting, join the specific game room.
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
        setPendingChoice(current =>
          current && current.id === choiceId ? null : current
        );
        setChoiceDeadline(current => (current ? null : current));
      });

      socket.on('error', (payload: any) => {
        const message = payload?.message || 'Game error';
        console.error('Game socket error', payload);
        setError(message);
        setIsConnecting(false);
      });

      socket.on('disconnect', () => {
        // Keep error state, but clear connection.
        socketRef.current = null;
      });
    } catch (err: any) {
      console.error('Failed to connect to game', err);
      setError(err?.message || 'Failed to connect to game');
      setIsConnecting(false);
    }
  }, [gameId, disconnect]);

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

      const response = {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        choiceType: choice.type,
        selectedOption
      };

      socket.emit('player_choice_response', response);
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
        moveType: partialMove.type
      };

      socket.emit('player_move', {
        gameId,
        move: movePayload
      });
    },
    [gameId, gameState]
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
    validMoves
  };

  return (
    <GameContext.Provider value={value}>
      {children}
    </GameContext.Provider>
  );
}

export function useGame(): GameContextType {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error('useGame must be used within a GameProvider');
  }
  return context;
}
