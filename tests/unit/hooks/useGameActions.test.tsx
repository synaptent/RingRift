/**
 * Unit tests for useGameActions hook
 *
 * Tests action submission functions including move submission,
 * choice handling, chat, and capabilities computation.
 */

import { renderHook, act } from '@testing-library/react';
import {
  useGameActions,
  usePendingChoice,
  useChatMessages,
  useValidMoves,
} from '@/client/hooks/useGameActions';
import type { PlayerChoice, Move, Position } from '@/shared/types/game';

// ─────────────────────────────────────────────────────────────────────────────
// Mock Setup
// ─────────────────────────────────────────────────────────────────────────────

const mockSubmitMove = jest.fn();
const mockRespondToChoice = jest.fn();
const mockSendChatMessage = jest.fn();

const createMockGameContext = (overrides: Record<string, unknown> = {}) => ({
  gameId: 'game-123',
  gameState: {
    gameStatus: 'active',
    currentPhase: 'movement',
    currentPlayer: 1,
    players: [
      { playerNumber: 1, userId: 'user-1', rings: 3 },
      { playerNumber: 2, userId: 'user-2', rings: 3 },
    ],
    board: { stacks: new Map(), markers: new Map() },
    moveHistory: [],
  },
  submitMove: mockSubmitMove,
  respondToChoice: mockRespondToChoice,
  sendChatMessage: mockSendChatMessage,
  pendingChoice: null,
  choiceDeadline: null,
  validMoves: null,
  chatMessages: [],
  ...overrides,
});

let mockContextValue = createMockGameContext();

jest.mock('@/client/contexts/GameContext', () => ({
  useGame: () => mockContextValue,
}));

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useGameActions
// ─────────────────────────────────────────────────────────────────────────────

describe('useGameActions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns all expected action functions and state', () => {
    const { result } = renderHook(() => useGameActions());

    expect(result.current.submitMove).toBeDefined();
    expect(result.current.submitPlacement).toBeDefined();
    expect(result.current.submitMovement).toBeDefined();
    expect(result.current.respondToChoice).toBeDefined();
    expect(result.current.sendChat).toBeDefined();
    expect(result.current.pendingChoice).toBeDefined();
    expect(result.current.capabilities).toBeDefined();
  });

  it('submitMove calls context submitMove with partial move', () => {
    const { result } = renderHook(() => useGameActions());

    const partialMove = {
      type: 'move_stack' as const,
      player: 1,
      from: { x: 2, y: 3 },
      to: { x: 4, y: 5 },
    };

    act(() => {
      result.current.submitMove(partialMove);
    });

    expect(mockSubmitMove).toHaveBeenCalledWith(partialMove);
    expect(mockSubmitMove).toHaveBeenCalledTimes(1);
  });

  it('submitPlacement calls submitMove with placement action', () => {
    const { result } = renderHook(() => useGameActions());

    const placementAction = {
      type: 'place_ring' as const,
      to: { x: 3, y: 4 },
      player: 1,
      placementCount: 1,
    };

    act(() => {
      result.current.submitPlacement(placementAction);
    });

    expect(mockSubmitMove).toHaveBeenCalledWith(placementAction);
  });

  it('submitMovement calls submitMove with movement action', () => {
    const { result } = renderHook(() => useGameActions());

    const movementAction = {
      type: 'move_ring' as const,
      from: { x: 1, y: 1 },
      to: { x: 2, y: 2 },
      player: 2,
    };

    act(() => {
      result.current.submitMovement(movementAction);
    });

    expect(mockSubmitMove).toHaveBeenCalledWith(movementAction);
  });

  it('respondToChoice calls context respondToChoice with choice and option', () => {
    const { result } = renderHook(() => useGameActions());

    const mockChoice: PlayerChoice = {
      id: 'choice-1',
      type: 'line_reward_option',
      playerNumber: 1,
      options: ['add_ring', 'add_stack'],
    };

    act(() => {
      result.current.respondToChoice(mockChoice, 'add_ring');
    });

    expect(mockRespondToChoice).toHaveBeenCalledWith(mockChoice, 'add_ring');
  });

  it('sendChat calls context sendChatMessage', () => {
    const { result } = renderHook(() => useGameActions());

    act(() => {
      result.current.sendChat('Hello world!');
    });

    expect(mockSendChatMessage).toHaveBeenCalledWith('Hello world!');
  });

  describe('pendingChoice state', () => {
    it('returns hasPendingChoice false when no choice pending', () => {
      mockContextValue = createMockGameContext({ pendingChoice: null });
      const { result } = renderHook(() => useGameActions());

      expect(result.current.pendingChoice.hasPendingChoice).toBe(false);
      expect(result.current.pendingChoice.choice).toBeNull();
      expect(result.current.pendingChoice.deadline).toBeNull();
    });

    it('returns hasPendingChoice true when choice pending', () => {
      const mockChoice: PlayerChoice = {
        id: 'choice-1',
        type: 'ring_elimination',
        playerNumber: 1,
        options: [{ x: 1, y: 1 }],
      };
      const deadline = Date.now() + 30000;
      mockContextValue = createMockGameContext({
        pendingChoice: mockChoice,
        choiceDeadline: deadline,
      });

      const { result } = renderHook(() => useGameActions());

      expect(result.current.pendingChoice.hasPendingChoice).toBe(true);
      expect(result.current.pendingChoice.choice).toEqual(mockChoice);
      expect(result.current.pendingChoice.deadline).toBe(deadline);
    });
  });

  describe('capabilities', () => {
    it('returns disabled capabilities when not connected to game', () => {
      mockContextValue = createMockGameContext({ gameId: null, gameState: null });
      const { result } = renderHook(() => useGameActions());

      expect(result.current.capabilities.canSubmitMove).toBe(false);
      expect(result.current.capabilities.canRespondToChoice).toBe(false);
      expect(result.current.capabilities.canSendChat).toBe(false);
      expect(result.current.capabilities.disabledReason).toBe('Not connected to a game');
    });

    it('returns disabled move capabilities when game not active', () => {
      mockContextValue = createMockGameContext({
        gameState: {
          ...createMockGameContext().gameState,
          gameStatus: 'finished',
        },
      });
      const { result } = renderHook(() => useGameActions());

      expect(result.current.capabilities.canSubmitMove).toBe(false);
      expect(result.current.capabilities.canRespondToChoice).toBe(false);
      expect(result.current.capabilities.canSendChat).toBe(true); // Chat still works
      expect(result.current.capabilities.disabledReason).toBe('Game is finished');
    });

    it('returns enabled capabilities when game is active', () => {
      mockContextValue = createMockGameContext();
      const { result } = renderHook(() => useGameActions());

      expect(result.current.capabilities.canSubmitMove).toBe(true);
      expect(result.current.capabilities.canSendChat).toBe(true);
      expect(result.current.capabilities.canRespondToChoice).toBe(false); // No pending choice
      expect(result.current.capabilities.disabledReason).toBeUndefined();
    });

    it('enables canRespondToChoice when choice is pending', () => {
      const mockChoice: PlayerChoice = {
        id: 'choice-1',
        type: 'line_order',
        playerNumber: 1,
        options: [],
      };
      mockContextValue = createMockGameContext({ pendingChoice: mockChoice });
      const { result } = renderHook(() => useGameActions());

      expect(result.current.capabilities.canRespondToChoice).toBe(true);
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: usePendingChoice
// ─────────────────────────────────────────────────────────────────────────────

describe('usePendingChoice', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns hasChoice false when no pending choice', () => {
    const { result } = renderHook(() => usePendingChoice());

    expect(result.current.hasChoice).toBe(false);
    expect(result.current.choice).toBeNull();
    expect(result.current.choiceType).toBeNull();
  });

  it('returns choice details when choice is pending', () => {
    const mockChoice: PlayerChoice = {
      id: 'choice-abc',
      type: 'territory_elimination',
      playerNumber: 2,
      options: ['option1', 'option2'],
    };
    mockContextValue = createMockGameContext({ pendingChoice: mockChoice });

    const { result } = renderHook(() => usePendingChoice());

    expect(result.current.hasChoice).toBe(true);
    expect(result.current.choice).toEqual(mockChoice);
    expect(result.current.choiceType).toBe('territory_elimination');
  });

  it('respond calls respondToChoice when choice is pending', () => {
    const mockChoice: PlayerChoice = {
      id: 'choice-xyz',
      type: 'line_reward_option',
      playerNumber: 1,
      options: ['add_ring', 'add_stack'],
    };
    mockContextValue = createMockGameContext({ pendingChoice: mockChoice });

    const { result } = renderHook(() => usePendingChoice());

    act(() => {
      result.current.respond('add_stack');
    });

    expect(mockRespondToChoice).toHaveBeenCalledWith(mockChoice, 'add_stack');
  });

  it('respond does nothing when no choice is pending', () => {
    mockContextValue = createMockGameContext({ pendingChoice: null });

    const { result } = renderHook(() => usePendingChoice());

    act(() => {
      result.current.respond('some_option');
    });

    expect(mockRespondToChoice).not.toHaveBeenCalled();
  });

  it('timeRemaining returns null when no deadline', () => {
    mockContextValue = createMockGameContext({ choiceDeadline: null });

    const { result } = renderHook(() => usePendingChoice());

    expect(result.current.timeRemaining).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useChatMessages
// ─────────────────────────────────────────────────────────────────────────────

describe('useChatMessages', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns empty messages array initially', () => {
    const { result } = renderHook(() => useChatMessages());

    expect(result.current.messages).toEqual([]);
    expect(result.current.messageCount).toBe(0);
  });

  it('returns chat messages from context', () => {
    const messages = [
      { sender: 'Player 1', text: 'Hello!' },
      { sender: 'Player 2', text: 'Hi there!' },
    ];
    mockContextValue = createMockGameContext({ chatMessages: messages });

    const { result } = renderHook(() => useChatMessages());

    expect(result.current.messages).toEqual(messages);
    expect(result.current.messageCount).toBe(2);
  });

  it('sendMessage calls context sendChatMessage with trimmed text', () => {
    const { result } = renderHook(() => useChatMessages());

    act(() => {
      result.current.sendMessage('  Hello world!  ');
    });

    expect(mockSendChatMessage).toHaveBeenCalledWith('Hello world!');
  });

  it('sendMessage does not send empty or whitespace-only messages', () => {
    const { result } = renderHook(() => useChatMessages());

    act(() => {
      result.current.sendMessage('   ');
    });

    expect(mockSendChatMessage).not.toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Test Suite: useValidMoves
// ─────────────────────────────────────────────────────────────────────────────

describe('useValidMoves', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockContextValue = createMockGameContext();
  });

  it('returns empty moves when validMoves is null', () => {
    mockContextValue = createMockGameContext({ validMoves: null });
    const { result } = renderHook(() => useValidMoves());

    expect(result.current.moves).toEqual([]);
    expect(result.current.hasValidMoves).toBe(false);
  });

  it('returns moves when validMoves is provided', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'move_ring',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 2,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    expect(result.current.moves).toEqual(validMoves);
    expect(result.current.hasValidMoves).toBe(true);
  });

  it('findMoveFor returns matching move', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'move_ring',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 2,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    const found = result.current.findMoveFor({ x: 1, y: 1 }, { x: 2, y: 2 });
    expect(found).toEqual(validMoves[0]);
  });

  it('findMoveFor returns undefined when no match', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    const found = result.current.findMoveFor({ x: 5, y: 5 }, { x: 6, y: 6 });
    expect(found).toBeUndefined();
  });

  it('getTargetsFrom returns valid target positions', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'move_ring',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 3, y: 3 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 2,
      },
      {
        id: 'm3',
        type: 'move_stack',
        player: 1,
        from: { x: 4, y: 4 },
        to: { x: 5, y: 5 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 3,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    const targets = result.current.getTargetsFrom({ x: 1, y: 1 });
    expect(targets).toHaveLength(2);
    expect(targets).toContainEqual({ x: 2, y: 2 });
    expect(targets).toContainEqual({ x: 3, y: 3 });
  });

  it('getPlacementPositions returns placement move targets', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'place_ring',
        player: 1,
        to: { x: 1, y: 1 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'm2',
        type: 'place_ring',
        player: 1,
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 2,
      },
      {
        id: 'm3',
        type: 'move_stack',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 4, y: 4 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 3,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    const placements = result.current.getPlacementPositions();
    expect(placements).toHaveLength(2);
    expect(placements).toContainEqual({ x: 1, y: 1 });
    expect(placements).toContainEqual({ x: 2, y: 2 });
  });

  it('isValidTarget returns true for valid placement', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'place_ring',
        player: 1,
        to: { x: 1, y: 1 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    expect(result.current.isValidTarget(undefined, { x: 1, y: 1 })).toBe(true);
    expect(result.current.isValidTarget(undefined, { x: 9, y: 9 })).toBe(false);
  });

  it('isValidTarget returns true for valid movement', () => {
    const validMoves: Move[] = [
      {
        id: 'm1',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: 0,
        thinkTime: 0,
        moveNumber: 1,
      },
    ];
    mockContextValue = createMockGameContext({ validMoves });

    const { result } = renderHook(() => useValidMoves());

    expect(result.current.isValidTarget({ x: 1, y: 1 }, { x: 2, y: 2 })).toBe(true);
    expect(result.current.isValidTarget({ x: 1, y: 1 }, { x: 9, y: 9 })).toBe(false);
  });
});
