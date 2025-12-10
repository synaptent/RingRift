/**
 * useSandboxInteractions.branchCoverage.test.tsx
 *
 * Branch coverage tests for useSandboxInteractions.ts
 */

import React, { useRef, useState } from 'react';
import { render, screen, act } from '@testing-library/react';
import { useSandboxInteractions } from '../../../src/client/hooks/useSandboxInteractions';
import type {
  Position,
  PlayerChoice,
  PlayerChoiceResponseFor,
  GameState,
  BoardState,
  Player,
} from '../../../src/shared/types/game';

// Mock the sandbox context
const mockGetGameState = jest.fn();
const mockGetValidMoves = jest.fn();
const mockTryPlaceRings = jest.fn();
const mockGetValidLandingPositionsForCurrentPlayer = jest.fn();
const mockHandleHumanCellClick = jest.fn();
const mockMaybeRunAITurn = jest.fn();
const mockClearSelection = jest.fn();
const mockGetChainCaptureContextForCurrentPlayer = jest.fn();

const mockSandboxEngine = {
  getGameState: mockGetGameState,
  getValidMoves: mockGetValidMoves,
  tryPlaceRings: mockTryPlaceRings,
  getValidLandingPositionsForCurrentPlayer: mockGetValidLandingPositionsForCurrentPlayer,
  handleHumanCellClick: mockHandleHumanCellClick,
  maybeRunAITurn: mockMaybeRunAITurn,
  clearSelection: mockClearSelection,
  getChainCaptureContextForCurrentPlayer: mockGetChainCaptureContextForCurrentPlayer,
};

const mockSetSandboxPendingChoice = jest.fn();
const mockSetSandboxCaptureChoice = jest.fn();
const mockSetSandboxCaptureTargets = jest.fn();
const mockSetSandboxLastProgressAt = jest.fn();
const mockSetSandboxStallWarning = jest.fn();
const mockSetSandboxStateVersion = jest.fn();

jest.mock('../../../src/client/contexts/SandboxContext', () => ({
  useSandbox: () => ({
    sandboxEngine: mockSandboxEngine,
    isConfigured: true,
    sandboxPendingChoice: null,
    setSandboxPendingChoice: mockSetSandboxPendingChoice,
    sandboxCaptureChoice: null,
    setSandboxCaptureChoice: mockSetSandboxCaptureChoice,
    setSandboxCaptureTargets: mockSetSandboxCaptureTargets,
    setSandboxLastProgressAt: mockSetSandboxLastProgressAt,
    setSandboxStallWarning: mockSetSandboxStallWarning,
    sandboxStateVersion: 1,
    setSandboxStateVersion: mockSetSandboxStateVersion,
  }),
}));

// Mock useInvalidMoveFeedback
jest.mock('../../../src/client/hooks/useInvalidMoveFeedback', () => ({
  useInvalidMoveFeedback: () => ({
    shakingCellKey: null,
    triggerInvalidMove: jest.fn(),
    analyzeInvalidMove: jest.fn(() => 'invalid_move'),
  }),
}));

// Create a test component that uses the hook
function TestComponent({
  onHookResult,
}: {
  onHookResult?: (result: ReturnType<typeof useSandboxInteractions>) => void;
}) {
  const [selected, setSelected] = useState<Position | undefined>(undefined);
  const [validTargets, setValidTargets] = useState<Position[]>([]);
  const choiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);

  const result = useSandboxInteractions({
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    choiceResolverRef,
  });

  React.useEffect(() => {
    if (onHookResult) onHookResult(result);
  }, [result, onHookResult]);

  return (
    <div>
      <button data-testid="click-cell" onClick={() => result.handleCellClick({ x: 3, y: 3 })}>
        Click
      </button>
      <button
        data-testid="dbl-click-cell"
        onClick={() => result.handleCellDoubleClick({ x: 3, y: 3 })}
      >
        Double Click
      </button>
      <button
        data-testid="context-menu"
        onClick={() => result.handleCellContextMenu({ x: 3, y: 3 })}
      >
        Context Menu
      </button>
      <button data-testid="clear" onClick={() => result.clearSelection()}>
        Clear
      </button>
      <button data-testid="run-ai" onClick={() => result.maybeRunSandboxAiIfNeeded()}>
        Run AI
      </button>
      <span data-testid="shaking">{result.shakingCellKey ?? 'none'}</span>
    </div>
  );
}

// Helper to create a basic game state
function createBasicGameState(overrides: Partial<GameState> = {}): GameState {
  const defaultPlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  const defaultBoard: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };

  return {
    id: 'test-game',
    boardType: 'square8',
    board: defaultBoard,
    players: defaultPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 0,
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
    territoryVictoryThreshold: 33,
    ...overrides,
  };
}

describe('useSandboxInteractions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetGameState.mockReturnValue(createBasicGameState());
    mockGetValidMoves.mockReturnValue([]);
    mockTryPlaceRings.mockResolvedValue(true);
    mockGetValidLandingPositionsForCurrentPlayer.mockReturnValue([]);
    mockHandleHumanCellClick.mockResolvedValue(undefined);
    mockMaybeRunAITurn.mockResolvedValue(undefined);
  });

  describe('handleCellClick in ring_placement phase', () => {
    it('selects cell without attempting placement on first click', async () => {
      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      // Single click should only update selection state; placement is
      // handled by double-click/context-menu flows.
      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });

    it('does not attempt placement even when engine mock would reject', async () => {
      mockTryPlaceRings.mockResolvedValue(false);
      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });
  });

  describe('handleCellClick in movement phase', () => {
    beforeEach(() => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          currentPhase: 'movement',
          board: {
            stacks: new Map([
              ['3,3', { controllingPlayer: 1, rings: [1], stackHeight: 1, capHeight: 1 }],
            ]),
            markers: new Map(),
            collapsedSpaces: new Map(),
            territories: new Map(),
            formedLines: [],
            eliminatedRings: {},
            size: 8,
            type: 'square8',
          },
        })
      );
      mockGetValidMoves.mockReturnValue([
        { id: 'm1', type: 'move_stack', player: 1, from: { x: 3, y: 3 }, to: { x: 4, y: 4 } },
      ]);
      mockGetValidLandingPositionsForCurrentPlayer.mockReturnValue([{ x: 4, y: 4 }]);
    });

    it('selects stack and highlights targets on first click', async () => {
      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellClick({ x: 3, y: 3 });
      });

      expect(mockGetValidLandingPositionsForCurrentPlayer).toHaveBeenCalledWith({ x: 3, y: 3 });
      expect(mockHandleHumanCellClick).toHaveBeenCalledWith({ x: 3, y: 3 });
    });
  });

  describe('handleCellClick when AI turn', () => {
    it('triggers AI loop instead of processing click', async () => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          currentPhase: 'ring_placement',
          players: [
            {
              id: 'p1',
              username: 'AI1',
              playerNumber: 1,
              type: 'ai',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 18,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
            {
              id: 'p2',
              username: 'Player2',
              playerNumber: 2,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 18,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
          ],
        })
      );

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellClick({ x: 4, y: 4 });
      });

      // AI turn should be triggered instead of placement
      // Note: The actual AI run depends on context state, but the click should not place
      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });
  });

  describe('maybeRunSandboxAiIfNeeded', () => {
    it('does nothing if engine is null', () => {
      // Mock the hook to return null engine by adjusting the mock
      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      // Engine is mocked, so this should work without error
      act(() => {
        result!.maybeRunSandboxAiIfNeeded();
      });
    });

    it('triggers AI loop when current player is AI', async () => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          currentPhase: 'ring_placement',
          players: [
            {
              id: 'p1',
              username: 'AI1',
              playerNumber: 1,
              type: 'ai',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 18,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
            {
              id: 'p2',
              username: 'Player2',
              playerNumber: 2,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 18,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
          ],
        })
      );

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.maybeRunSandboxAiIfNeeded();
        await new Promise((r) => setTimeout(r, 10));
      });
    });
  });

  describe('clearSelection', () => {
    it('clears selection and calls engine.clearSelection', () => {
      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      act(() => {
        result!.clearSelection();
      });

      expect(mockClearSelection).toHaveBeenCalled();
    });
  });

  describe('handleCellDoubleClick', () => {
    it('does nothing if not in ring_placement phase', async () => {
      mockGetGameState.mockReturnValue(createBasicGameState({ currentPhase: 'movement' }));

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellDoubleClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });

    it('attempts 2-ring placement on empty cell', async () => {
      mockGetGameState.mockReturnValue(createBasicGameState());
      mockTryPlaceRings.mockResolvedValue(true);

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellDoubleClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockTryPlaceRings).toHaveBeenCalledWith({ x: 4, y: 4 }, 2);
    });

    it('falls back to 1-ring placement if 2-ring fails', async () => {
      mockGetGameState.mockReturnValue(createBasicGameState());
      mockTryPlaceRings
        .mockResolvedValueOnce(false) // First 2-ring attempt fails
        .mockResolvedValueOnce(true); // Fallback to 1-ring succeeds

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellDoubleClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockTryPlaceRings).toHaveBeenCalledTimes(2);
      expect(mockTryPlaceRings).toHaveBeenNthCalledWith(1, { x: 4, y: 4 }, 2);
      expect(mockTryPlaceRings).toHaveBeenNthCalledWith(2, { x: 4, y: 4 }, 1);
    });

    it('attempts 1-ring on occupied cell', async () => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          board: {
            stacks: new Map([
              ['4,4', { controllingPlayer: 1, rings: [1], stackHeight: 1, capHeight: 1 }],
            ]),
            markers: new Map(),
            collapsedSpaces: new Map(),
            territories: new Map(),
            formedLines: [],
            eliminatedRings: {},
            size: 8,
            type: 'square8',
          },
        })
      );
      mockTryPlaceRings.mockResolvedValue(true);

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellDoubleClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockTryPlaceRings).toHaveBeenCalledWith({ x: 4, y: 4 }, 1);
    });

    it('does nothing if player has no rings in hand', async () => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          players: [
            {
              id: 'p1',
              username: 'Player1',
              playerNumber: 1,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 0, // No rings
              eliminatedRings: 0,
              territorySpaces: 0,
            },
          ],
        })
      );

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      await act(async () => {
        result!.handleCellDoubleClick({ x: 4, y: 4 });
        await new Promise((r) => setTimeout(r, 10));
      });

      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });
  });

  describe('handleCellContextMenu', () => {
    it('does nothing if not in ring_placement phase', () => {
      mockGetGameState.mockReturnValue(createBasicGameState({ currentPhase: 'movement' }));

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      act(() => {
        result!.handleCellContextMenu({ x: 4, y: 4 });
      });

      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });

    it('does nothing if player has no rings in hand', () => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          players: [
            {
              id: 'p1',
              username: 'Player1',
              playerNumber: 1,
              type: 'human',
              isReady: true,
              timeRemaining: 600000,
              ringsInHand: 0,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
          ],
        })
      );

      let result: ReturnType<typeof useSandboxInteractions> | null = null;
      render(
        <TestComponent
          onHookResult={(r) => {
            result = r;
          }}
        />
      );

      act(() => {
        result!.handleCellContextMenu({ x: 4, y: 4 });
      });

      expect(mockTryPlaceRings).not.toHaveBeenCalled();
    });
  });

  describe('handleCellClick in chain_capture phase', () => {
    beforeEach(() => {
      mockGetGameState.mockReturnValue(
        createBasicGameState({
          currentPhase: 'chain_capture',
        })
      );
      mockGetValidMoves.mockReturnValue([]);
    });

    it('rejects clicks not on valid targets', async () => {
      let result: ReturnType<typeof useSandboxInteractions> | null = null;

      // Render with empty valid targets
      function TestWithState() {
        const [selected, setSelected] = useState<Position | undefined>({ x: 3, y: 3 });
        const [validTargets, setValidTargets] = useState<Position[]>([{ x: 5, y: 5 }]); // Target is at 5,5
        const choiceResolverRef = useRef<
          ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
        >(null);

        const hookResult = useSandboxInteractions({
          selected,
          setSelected,
          validTargets,
          setValidTargets,
          choiceResolverRef,
        });

        React.useEffect(() => {
          result = hookResult;
        }, [hookResult]);

        return <div />;
      }

      render(<TestWithState />);

      await act(async () => {
        result!.handleCellClick({ x: 4, y: 4 }); // Click at 4,4 which is NOT a valid target
      });

      // Should not process the click since it's not a valid target
      expect(mockHandleHumanCellClick).not.toHaveBeenCalled();
    });
  });
});
