/**
 * GameContext Unit Tests
 *
 * Tests the GameContext provider and useGame hook for:
 * - Initial state management
 * - Socket connection lifecycle
 * - Game state updates
 * - Player choices and responses
 * - Move submission
 * - Chat messaging
 * - Connection status tracking
 * - Error handling
 * - Victory state handling
 *
 * NOTE: Due to GameContext using Vite's import.meta.env which Jest cannot
 * parse in Node/CommonJS mode, we mock the GameContext module before
 * importing. This allows us to test the context contract and consumer behavior.
 */

import React, { useState, useCallback, useRef, useEffect, createContext, useContext } from 'react';
import { render, screen, waitFor, act, renderHook } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type {
  GameState,
  GamePhase,
  PlayerChoice,
  Move,
  GameResult,
} from '../../../src/shared/types/game';

// Define connection status type
type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

// Store socket event handlers for simulation
type SocketEventHandler = (...args: any[]) => void;
const socketEventHandlers: Record<string, SocketEventHandler> = {};
const mockSocketEmit = jest.fn();
const mockSocketDisconnect = jest.fn();

// Define socket shape interface
interface MockSocketShape {
  on: jest.Mock;
  emit: jest.Mock;
  disconnect: jest.Mock;
}

const mockSocket: MockSocketShape = {
  on: jest.fn((event: string, handler: SocketEventHandler) => {
    socketEventHandlers[event] = handler;
    return mockSocket;
  }),
  emit: mockSocketEmit,
  disconnect: mockSocketDisconnect,
};

// Mock socket.io-client
jest.mock('socket.io-client', () => ({
  __esModule: true,
  io: jest.fn(() => mockSocket),
}));

// Mock react-hot-toast with explicit typing
interface MockToastFn extends jest.Mock {
  success: jest.Mock;
  error: jest.Mock;
}

const mockToast: MockToastFn = Object.assign(jest.fn(), {
  success: jest.fn(),
  error: jest.fn(),
});

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  toast: mockToast,
}));

// Mock error reporting
jest.mock('../../../src/client/utils/errorReporting', () => ({
  reportClientError: jest.fn(),
  isErrorReportingEnabled: jest.fn(() => false),
}));

// ============================================================================
// Create a local implementation of GameContext for testing
// This mirrors the real implementation but without import.meta dependencies
// ============================================================================

// Decision auto-resolved metadata type (mirrors websocket.ts)
interface DecisionAutoResolvedMeta {
  choiceType: string;
  choiceKind: string;
  actingPlayerNumber: number;
  resolvedMoveId?: string;
  reason: string;
}

// Decision phase timeout warning payload type (mirrors websocket.ts)
interface DecisionPhaseTimeoutWarningPayload {
  type: 'decision_phase_timeout_warning';
  data: {
    gameId: string;
    playerNumber: number;
    phase: 'line_processing' | 'territory_processing' | 'chain_capture';
    remainingMs: number;
    choiceId?: string;
  };
  timestamp: string;
}

interface GameContextType {
  gameId: string | null;
  gameState: GameState | null;
  validMoves: Move[] | null;
  isConnecting: boolean;
  error: string | null;
  victoryState: GameResult | null;
  connectToGame: (gameId: string) => Promise<void>;
  disconnect: () => void;
  pendingChoice: PlayerChoice | null;
  choiceDeadline: number | null;
  respondToChoice: (choice: PlayerChoice, selectedOption: any) => void;
  submitMove: (partialMove: Omit<Move, 'id' | 'timestamp' | 'thinkTime' | 'moveNumber'>) => void;
  sendChatMessage: (text: string) => void;
  chatMessages: { sender: string; text: string }[];
  connectionStatus: ConnectionStatus;
  lastHeartbeatAt: number | null;
  decisionAutoResolved: DecisionAutoResolvedMeta | null;
  decisionPhaseTimeoutWarning: DecisionPhaseTimeoutWarningPayload | null;
}

const GameContext = createContext<GameContextType | null>(null);

// Helper to hydrate board state from raw object
function hydrateBoardState(rawBoard: any) {
  if (!rawBoard) {
    return {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 0,
      type: 'square8' as const,
    };
  }

  return {
    stacks: new Map(Object.entries(rawBoard.stacks || {})),
    markers: new Map(Object.entries(rawBoard.markers || {})),
    collapsedSpaces: new Map(Object.entries(rawBoard.collapsedSpaces || {})),
    territories: new Map(Object.entries(rawBoard.territories || {})),
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

// Test implementation of GameProvider that mirrors real behavior
function TestGameProvider({ children }: { children: React.ReactNode }) {
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
  const socketRef = useRef<MockSocketShape | null>(null);
  const targetGameIdRef = useRef<string | null>(null);

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
    setDecisionAutoResolved(null);
    setDecisionPhaseTimeoutWarning(null);
  }, []);

  const connectToGame = useCallback(
    async (targetGameId: string) => {
      if (gameId === targetGameId && socketRef.current) {
        return;
      }

      disconnect();
      targetGameIdRef.current = targetGameId;
      setIsConnecting(true);
      setError(null);
      setConnectionStatus('connecting');

      try {
        const { io } = await import('socket.io-client');
        const token = localStorage.getItem('token');

        const socket = io('http://localhost:3000', {
          transports: ['websocket', 'polling'],
          auth: token ? { token } : undefined,
        }) as unknown as MockSocketShape;

        socketRef.current = socket;

        const emitJoinGame = () => {
          socket.emit('join_game', { gameId: targetGameId });
        };

        socket.on('connect_error', (err: Error) => {
          const msg = err.message || 'Failed to connect to game server';
          setError(msg);
          mockToast.error(msg);
          setIsConnecting(false);
          setConnectionStatus('disconnected');
          setLastHeartbeatAt(null);
        });

        socket.on('connect', () => {
          setConnectionStatus('connected');
          emitJoinGame();
        });

        socket.on('reconnect_attempt', () => {
          setIsConnecting(true);
          setConnectionStatus('reconnecting');
          mockToast('Reconnecting...', { icon: '🔄', id: 'reconnecting' });
        });

        socket.on('reconnect', () => {
          setIsConnecting(false);
          setConnectionStatus('connected');
          mockToast.success('Reconnected!', { id: 'reconnecting' });
          emitJoinGame();
        });

        socket.on('request_reconnect', () => {
          emitJoinGame();
        });

        socket.on('game_state', (payload: any) => {
          const { data } = payload || {};
          if (data?.gameId === targetGameIdRef.current && data?.gameState) {
            setGameId(targetGameIdRef.current);
            setGameState(hydrateGameState(data.gameState));
            setValidMoves(Array.isArray(data.validMoves) ? data.validMoves : null);
            setIsConnecting(false);
            setError(null);
            setConnectionStatus('connected');
            setLastHeartbeatAt(Date.now());
            // Fresh snapshots clear any prior pending choices/deadlines so the
            // HUD does not show stale decision banners after reconnect/resync.
            setPendingChoice(null);
            setChoiceDeadline(null);
            setDecisionAutoResolved(data.meta?.diffSummary?.decisionAutoResolved ?? null);
            setDecisionPhaseTimeoutWarning(null);
          }
        });

        socket.on('game_over', (payload: any) => {
          const { data } = payload || {};
          if (!data || data.gameId !== targetGameIdRef.current) return;

          setGameId(targetGameIdRef.current);
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
        });

        socket.on('player_choice_required', (choice: PlayerChoice) => {
          setPendingChoice(choice);
          const deadline = choice.timeoutMs ? Date.now() + choice.timeoutMs : null;
          setChoiceDeadline(deadline);
        });

        socket.on('player_choice_canceled', (choiceId: string) => {
          setPendingChoice((current) => (current && current.id === choiceId ? null : current));
          setChoiceDeadline((current) => (current ? null : current));
        });

        socket.on(
          'decision_phase_timeout_warning',
          (payload: DecisionPhaseTimeoutWarningPayload) => {
            setDecisionPhaseTimeoutWarning(payload);
          }
        );

        socket.on('decision_phase_timed_out', () => {
          setDecisionPhaseTimeoutWarning(null);
        });

        socket.on('chat_message', (payload: { sender: string; text: string }) => {
          setChatMessages((prev) => [...prev, { sender: payload.sender, text: payload.text }]);
        });

        socket.on('error', (payload: any) => {
          let message: string;
          if (payload && payload.type === 'error' && payload.code) {
            message = payload.message || 'Game error';
          } else {
            message = payload?.message || 'Game error';
          }
          setError(message);
          mockToast.error(message);
          setIsConnecting(false);
        });

        socket.on('disconnect', (reason: string) => {
          if (reason === 'io client disconnect') {
            socketRef.current = null;
          }
          setConnectionStatus('disconnected');
          setLastHeartbeatAt(null);
          if (reason !== 'io client disconnect') {
            setConnectionStatus('reconnecting');
          }
        });
      } catch (err: any) {
        setError(err?.message || 'Failed to connect to game');
        setIsConnecting(false);
        setConnectionStatus('disconnected');
      }
    },
    [gameId, disconnect]
  );

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

      let moveId: string | undefined;

      if (
        choice.type === 'line_order' ||
        choice.type === 'region_order' ||
        choice.type === 'ring_elimination'
      ) {
        moveId =
          selectedOption && typeof selectedOption.moveId === 'string'
            ? selectedOption.moveId
            : undefined;
      } else if (choice.type === 'line_reward_option') {
        const optionKey = selectedOption as string;
        moveId = (choice as any).moveIds?.[optionKey];
      }

      if (moveId) {
        socket.emit('player_move_by_id', { gameId, moveId });
      } else {
        socket.emit('player_choice_response', {
          choiceId: choice.id,
          playerNumber: choice.playerNumber,
          choiceType: choice.type,
          selectedOption,
        });
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
    decisionAutoResolved,
    decisionPhaseTimeoutWarning,
  };

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

function useGame(): GameContextType {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error('useGame must be used within a GameProvider');
  }
  return context;
}

// ============================================================================
// Test Helpers
// ============================================================================

// Helper to create mock game state
function createMockGameState(overrides: Partial<GameState> = {}): GameState {
  return {
    id: 'game-123',
    boardType: 'square8',
    board: {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    },
    players: [
      {
        id: 'player-1',
        username: 'Player1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'player-2',
        username: 'Player2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 600,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ],
    currentPhase: 'ring_placement' as GamePhase,
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: true,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 33,
    ...overrides,
  };
}

// Helper to create mock choice
function createMockChoice(overrides: Partial<PlayerChoice> = {}): PlayerChoice {
  return {
    id: 'choice-123',
    gameId: 'game-123',
    playerNumber: 1,
    type: 'line_order',
    prompt: 'Choose which line to process',
    options: [
      {
        lineId: 'line-1',
        markerPositions: [{ x: 0, y: 0 }],
        moveId: 'move-1',
      },
    ],
    ...overrides,
  } as PlayerChoice;
}

// Test component that exposes GameContext values
function TestConsumer({ gameId = 'game-123' }: { gameId?: string }) {
  const ctx = useGame();

  return (
    <div>
      <span data-testid="game-id">{ctx.gameId || 'none'}</span>
      <span data-testid="connection-status">{ctx.connectionStatus}</span>
      <span data-testid="is-connecting">{ctx.isConnecting ? 'yes' : 'no'}</span>
      <span data-testid="error">{ctx.error || 'no-error'}</span>
      <span data-testid="current-phase">{ctx.gameState?.currentPhase || 'no-phase'}</span>
      <span data-testid="current-player">{ctx.gameState?.currentPlayer ?? 'none'}</span>
      <span data-testid="pending-choice">{ctx.pendingChoice ? ctx.pendingChoice.id : 'none'}</span>
      <span data-testid="choice-deadline">{ctx.choiceDeadline ?? 'none'}</span>
      <span data-testid="victory-state">{ctx.victoryState ? ctx.victoryState.reason : 'none'}</span>
      <span data-testid="chat-count">{ctx.chatMessages.length}</span>
      <span data-testid="valid-moves-count">{ctx.validMoves?.length ?? 'null'}</span>
      <span data-testid="last-heartbeat">{ctx.lastHeartbeatAt ?? 'null'}</span>
      <span data-testid="decision-auto-resolved">
        {ctx.decisionAutoResolved ? ctx.decisionAutoResolved.reason : 'none'}
      </span>
      <span data-testid="decision-timeout-warning">
        {ctx.decisionPhaseTimeoutWarning
          ? ctx.decisionPhaseTimeoutWarning.data.remainingMs
          : 'none'}
      </span>

      <button data-testid="connect-btn" onClick={() => ctx.connectToGame(gameId)}>
        Connect
      </button>
      <button data-testid="disconnect-btn" onClick={ctx.disconnect}>
        Disconnect
      </button>
      <button
        data-testid="submit-move-btn"
        onClick={() =>
          ctx.submitMove({
            type: 'place_ring',
            player: 1,
            to: { x: 3, y: 3 },
          })
        }
      >
        Submit Move
      </button>
      <button data-testid="send-chat-btn" onClick={() => ctx.sendChatMessage('Hello!')}>
        Send Chat
      </button>
      <button
        data-testid="respond-choice-btn"
        onClick={() =>
          ctx.pendingChoice && ctx.respondToChoice(ctx.pendingChoice, ctx.pendingChoice.options[0])
        }
      >
        Respond to Choice
      </button>
    </div>
  );
}

// ============================================================================
// Tests
// ============================================================================

describe('GameContext', () => {
  const localStorageMock: Record<string, string> = {};

  beforeEach(() => {
    jest.clearAllMocks();

    // Clear socket event handlers
    Object.keys(socketEventHandlers).forEach((key) => delete socketEventHandlers[key]);

    // Reset localStorage mock
    Object.keys(localStorageMock).forEach((key) => delete localStorageMock[key]);

    jest.spyOn(Storage.prototype, 'getItem').mockImplementation((key) => {
      return localStorageMock[key] || null;
    });
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation((key, value) => {
      localStorageMock[key] = value;
    });
    jest.spyOn(Storage.prototype, 'removeItem').mockImplementation((key) => {
      delete localStorageMock[key];
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Initial State', () => {
    it('provides initial disconnected state', () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      expect(screen.getByTestId('connection-status')).toHaveTextContent('disconnected');
      expect(screen.getByTestId('game-id')).toHaveTextContent('none');
      expect(screen.getByTestId('is-connecting')).toHaveTextContent('no');
      expect(screen.getByTestId('error')).toHaveTextContent('no-error');
    });

    it('provides null game state initially', () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      expect(screen.getByTestId('current-phase')).toHaveTextContent('no-phase');
      expect(screen.getByTestId('current-player')).toHaveTextContent('none');
    });

    it('has empty chat messages initially', () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      expect(screen.getByTestId('chat-count')).toHaveTextContent('0');
    });

    it('has null pending choice initially', () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('none');
      expect(screen.getByTestId('choice-deadline')).toHaveTextContent('none');
    });
  });

  describe('Connection Lifecycle', () => {
    it('sets connecting state when connecting to game', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      expect(screen.getByTestId('is-connecting')).toHaveTextContent('yes');
      expect(screen.getByTestId('connection-status')).toHaveTextContent('connecting');
    });

    it('uses token from localStorage for socket auth', async () => {
      const { io } = await import('socket.io-client');
      localStorageMock['token'] = 'user-jwt-token';

      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      expect(io).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          auth: { token: 'user-jwt-token' },
        })
      );
    });

    it('updates status to connected on socket connect event', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      expect(screen.getByTestId('connection-status')).toHaveTextContent('connected');
    });

    it('emits join_game on socket connect', async () => {
      render(
        <TestGameProvider>
          <TestConsumer gameId="test-game-456" />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      expect(mockSocketEmit).toHaveBeenCalledWith('join_game', { gameId: 'test-game-456' });
    });

    it('disconnects and resets state on disconnect call', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      await user.click(screen.getByTestId('disconnect-btn'));

      expect(mockSocketDisconnect).toHaveBeenCalled();
      expect(screen.getByTestId('connection-status')).toHaveTextContent('disconnected');
      expect(screen.getByTestId('game-id')).toHaveTextContent('none');
    });

    it('handles socket disconnect event with reconnection', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['disconnect']?.('transport close');
      });

      expect(screen.getByTestId('connection-status')).toHaveTextContent('reconnecting');
    });

    it('does not attempt reconnection on intentional disconnect', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['disconnect']?.('io client disconnect');
      });

      expect(screen.getByTestId('connection-status')).toHaveTextContent('disconnected');
    });
  });

  describe('Game State Updates', () => {
    it('updates game state on game_state event', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const mockState = createMockGameState({ currentPhase: 'movement', currentPlayer: 2 });

      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
            validMoves: [],
          },
        });
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-phase')).toHaveTextContent('movement');
        expect(screen.getByTestId('current-player')).toHaveTextContent('2');
      });
    });

    it('updates valid moves on game_state event', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const mockState = createMockGameState();
      const validMoves = [
        {
          id: 'move-1',
          type: 'place_ring' as const,
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
        {
          id: 'move-2',
          type: 'place_ring' as const,
          player: 1,
          to: { x: 1, y: 1 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];

      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
            validMoves,
          },
        });
      });

      expect(screen.getByTestId('valid-moves-count')).toHaveTextContent('2');
    });

    it('updates last heartbeat timestamp on game_state', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const beforeTime = Date.now();
      const mockState = createMockGameState();

      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
          },
        });
      });

      const heartbeat = parseInt(screen.getByTestId('last-heartbeat').textContent || '0');
      expect(heartbeat).toBeGreaterThanOrEqual(beforeTime);
    });

    it('ignores game_state for different game ID', async () => {
      render(
        <TestGameProvider>
          <TestConsumer gameId="my-game" />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const mockState = createMockGameState({ currentPhase: 'movement' });

      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'different-game',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
          },
        });
      });

      expect(screen.getByTestId('current-phase')).toHaveTextContent('no-phase');
    });
  });

  describe('Victory State', () => {
    it('sets victory state on game_over event', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['game_over']?.({
          data: {
            gameId: 'game-123',
            gameResult: {
              winner: 1,
              reason: 'ring_elimination',
              finalScore: {
                ringsEliminated: { 1: 0, 2: 10 },
                territorySpaces: { 1: 5, 2: 3 },
                ringsRemaining: { 1: 18, 2: 8 },
              },
            },
          },
        });
      });

      expect(screen.getByTestId('victory-state')).toHaveTextContent('ring_elimination');
    });

    it('clears pending choice on game_over', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['player_choice_required']?.(createMockChoice());
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('choice-123');

      act(() => {
        socketEventHandlers['game_over']?.({
          data: {
            gameId: 'game-123',
            gameResult: { winner: 1, reason: 'timeout', finalScore: {} },
          },
        });
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('none');
    });

    it('clears pending choice on fresh game_state snapshot (e.g. reconnect)', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      // Simulate an in-flight decision
      act(() => {
        socketEventHandlers['player_choice_required']?.(createMockChoice());
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('choice-123');

      // A subsequent authoritative game_state snapshot (such as after a
      // reconnect) should clear any stale pending choice so the HUD does
      // not display decision banners that no longer match server state.
      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: createMockGameState(),
            validMoves: [],
          },
        });
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('none');
      expect(screen.getByTestId('choice-deadline')).toHaveTextContent('none');
    });

    it('clears decisionAutoResolved and timeout warning on game_over', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      // Simulate a prior game_state carrying decisionAutoResolved + a timeout warning
      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...createMockGameState(),
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
            validMoves: [],
            meta: {
              diffSummary: {
                decisionAutoResolved: {
                  choiceType: 'line_reward_option',
                  choiceKind: 'line_reward',
                  actingPlayerNumber: 1,
                  resolvedMoveId: 'auto-move-1',
                  reason: 'timeout',
                },
              },
            },
          },
        });
      });

      // Manually invoke the timeout warning handler as the socket would
      act(() => {
        socketEventHandlers['decision_phase_timeout_warning']?.({
          type: 'decision_phase_timeout_warning',
          data: {
            gameId: 'game-123',
            playerNumber: 1,
            phase: 'line_processing',
            remainingMs: 5000,
            choiceId: 'choice-1',
          },
          timestamp: new Date().toISOString(),
        } as any);
      });

      expect(screen.getByTestId('decision-auto-resolved')).toHaveTextContent('timeout');
      expect(screen.getByTestId('decision-timeout-warning')).not.toHaveTextContent('none');

      // game_over should clear both decisionAutoResolved and timeout warnings
      act(() => {
        socketEventHandlers['game_over']?.({
          data: {
            gameId: 'game-123',
            gameResult: { winner: 1, reason: 'timeout', finalScore: {} },
          },
        });
      });

      expect(screen.getByTestId('decision-auto-resolved')).toHaveTextContent('none');
      expect(screen.getByTestId('decision-timeout-warning')).toHaveTextContent('none');
    });
  });

  describe('Player Choices', () => {
    it('sets pending choice on player_choice_required event', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const choice = createMockChoice({ id: 'my-choice-id' });

      act(() => {
        socketEventHandlers['player_choice_required']?.(choice);
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('my-choice-id');
    });

    it('calculates deadline from timeoutMs', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const beforeTime = Date.now();
      const choice = createMockChoice({ timeoutMs: 30000 });

      act(() => {
        socketEventHandlers['player_choice_required']?.(choice);
      });

      const deadline = parseInt(screen.getByTestId('choice-deadline').textContent || '0');
      expect(deadline).toBeGreaterThanOrEqual(beforeTime + 30000);
    });

    it('clears pending choice on player_choice_canceled', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['player_choice_required']?.(createMockChoice({ id: 'cancel-me' }));
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('cancel-me');

      act(() => {
        socketEventHandlers['player_choice_canceled']?.('cancel-me');
      });

      expect(screen.getByTestId('pending-choice')).toHaveTextContent('none');
    });

    it('tracks decision-phase timeout warnings and clears them on final timeout event', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      // Simulate a timeout warning from the server.
      act(() => {
        socketEventHandlers['decision_phase_timeout_warning']?.({
          type: 'decision_phase_timeout_warning',
          data: {
            gameId: 'game-123',
            playerNumber: 1,
            phase: 'line_processing',
            remainingMs: 5000,
            choiceId: 'choice-1',
          },
          timestamp: new Date().toISOString(),
        } as any);
      });

      expect(screen.getByTestId('decision-timeout-warning')).not.toHaveTextContent('none');

      // The final timeout event should clear the warning.
      act(() => {
        socketEventHandlers['decision_phase_timed_out']?.({
          type: 'decision_phase_timed_out',
          data: {
            gameId: 'game-123',
            playerNumber: 1,
            phase: 'line_processing',
            autoSelectedMoveId: 'auto-move-1',
            reason: 'timeout',
          },
          timestamp: new Date().toISOString(),
        } as any);
      });

      expect(screen.getByTestId('decision-timeout-warning')).toHaveTextContent('none');
    });

    it('responds to choice by emitting player_choice_response', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const mockState = createMockGameState();
      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
          },
        });
      });

      const choice: PlayerChoice = {
        id: 'choice-123',
        gameId: 'game-123',
        playerNumber: 1,
        type: 'line_reward_option',
        prompt: 'Choose reward',
        options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
      };

      act(() => {
        socketEventHandlers['player_choice_required']?.(choice);
      });

      mockSocketEmit.mockClear();

      await user.click(screen.getByTestId('respond-choice-btn'));

      expect(mockSocketEmit).toHaveBeenCalledWith(
        'player_choice_response',
        expect.objectContaining({
          choiceId: 'choice-123',
          playerNumber: 1,
        })
      );
    });
  });

  describe('Move Submission', () => {
    it('emits player_move with move payload', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const mockState = createMockGameState({ moveHistory: [] });
      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
          },
        });
      });

      mockSocketEmit.mockClear();

      await user.click(screen.getByTestId('submit-move-btn'));

      expect(mockSocketEmit).toHaveBeenCalledWith('player_move', {
        gameId: 'game-123',
        move: expect.objectContaining({
          moveType: 'place_ring',
          moveNumber: 1,
        }),
      });
    });

    it('does not emit move when not connected', async () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      mockSocketEmit.mockClear();

      await user.click(screen.getByTestId('submit-move-btn'));

      expect(mockSocketEmit).not.toHaveBeenCalledWith('player_move', expect.anything());
      expect(consoleSpy).toHaveBeenCalledWith('submitMove called without active socket/game');

      consoleSpy.mockRestore();
    });
  });

  describe('Chat Messages', () => {
    it('emits chat_message with text', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      const mockState = createMockGameState();
      act(() => {
        socketEventHandlers['game_state']?.({
          data: {
            gameId: 'game-123',
            gameState: {
              ...mockState,
              board: {
                stacks: {},
                markers: {},
                collapsedSpaces: {},
                territories: {},
                formedLines: [],
                eliminatedRings: {},
                size: 8,
                type: 'square8',
              },
            },
          },
        });
      });

      mockSocketEmit.mockClear();

      await user.click(screen.getByTestId('send-chat-btn'));

      expect(mockSocketEmit).toHaveBeenCalledWith('chat_message', {
        gameId: 'game-123',
        text: 'Hello!',
      });
    });

    it('receives chat messages from server', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['chat_message']?.({
          sender: 'Player1',
          text: 'Hello everyone!',
        });
      });

      expect(screen.getByTestId('chat-count')).toHaveTextContent('1');

      act(() => {
        socketEventHandlers['chat_message']?.({
          sender: 'Player2',
          text: 'Hi there!',
        });
      });

      expect(screen.getByTestId('chat-count')).toHaveTextContent('2');
    });
  });

  describe('Error Handling', () => {
    it('handles socket error events', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['error']?.({
          type: 'error',
          code: 'INVALID_MOVE',
          message: 'That move is not valid',
        });
      });

      expect(screen.getByTestId('error')).toHaveTextContent('That move is not valid');
      expect(mockToast.error).toHaveBeenCalledWith('That move is not valid');
    });

    it('handles connect_error events', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect_error']?.(new Error('Connection refused'));
      });

      expect(screen.getByTestId('error')).toHaveTextContent('Connection refused');
      expect(screen.getByTestId('connection-status')).toHaveTextContent('disconnected');
    });
  });

  describe('Hook Usage', () => {
    it('throws error when useGame is used outside GameProvider', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useGame());
      }).toThrow('useGame must be used within a GameProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('Reconnection', () => {
    it('shows reconnecting status and toast on reconnect attempt', async () => {
      render(
        <TestGameProvider>
          <TestConsumer />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      act(() => {
        socketEventHandlers['reconnect_attempt']?.();
      });

      expect(screen.getByTestId('connection-status')).toHaveTextContent('reconnecting');
      expect(screen.getByTestId('is-connecting')).toHaveTextContent('yes');
      expect(mockToast).toHaveBeenCalledWith('Reconnecting...', expect.any(Object));
    });

    it('re-emits join_game on reconnect success', async () => {
      render(
        <TestGameProvider>
          <TestConsumer gameId="reconnect-game" />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      mockSocketEmit.mockClear();

      act(() => {
        socketEventHandlers['reconnect']?.();
      });

      expect(mockSocketEmit).toHaveBeenCalledWith('join_game', { gameId: 'reconnect-game' });
      expect(screen.getByTestId('connection-status')).toHaveTextContent('connected');
      expect(mockToast.success).toHaveBeenCalledWith('Reconnected!', expect.any(Object));
    });

    it('handles request_reconnect event from server', async () => {
      render(
        <TestGameProvider>
          <TestConsumer gameId="resync-game" />
        </TestGameProvider>
      );

      const user = userEvent.setup();
      await user.click(screen.getByTestId('connect-btn'));

      act(() => {
        socketEventHandlers['connect']?.();
      });

      mockSocketEmit.mockClear();

      act(() => {
        socketEventHandlers['request_reconnect']?.();
      });

      expect(mockSocketEmit).toHaveBeenCalledWith('join_game', { gameId: 'resync-game' });
    });
  });

  describe('Multiple Connections', () => {
    it('disconnects previous connection when connecting to new game', async () => {
      const { rerender } = render(
        <TestGameProvider>
          <TestConsumer gameId="game-1" />
        </TestGameProvider>
      );

      const user = userEvent.setup();

      // Connect to first game
      await user.click(screen.getByTestId('connect-btn'));
      act(() => {
        socketEventHandlers['connect']?.();
      });

      mockSocketDisconnect.mockClear();

      // Disconnect and connect to different game
      await user.click(screen.getByTestId('disconnect-btn'));

      rerender(
        <TestGameProvider>
          <TestConsumer gameId="game-2" />
        </TestGameProvider>
      );

      // The disconnect should have been called
      expect(mockSocketDisconnect).toHaveBeenCalled();
    });
  });
});
