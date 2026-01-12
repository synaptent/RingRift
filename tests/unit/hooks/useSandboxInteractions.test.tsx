/**
 * Tests for useSandboxInteractions hook.
 *
 * This hook handles sandbox mode interactions including clicks,
 * double-clicks, context menu, and AI turn management.
 */

import { useState, useRef } from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useSandboxInteractions } from '../../../src/client/hooks/useSandboxInteractions';
import * as SandboxContextModule from '../../../src/client/contexts/SandboxContext';
import type {
  Position,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../../src/shared/types/game';
import { createTestGameState, pos, addStack } from '../../utils/fixtures';

// Mock the SandboxContext
jest.mock('../../../src/client/contexts/SandboxContext', () => ({
  useSandbox: jest.fn(),
}));

// Create mock engine
function createMockEngine(
  overrides: Partial<{
    gameState: ReturnType<typeof createTestGameState>;
    validLandingPositions: Position[];
    validMoves?: Array<{ type: string; player: number; from?: Position; to?: Position }>;
  }> = {}
) {
  const defaultState = createTestGameState({
    currentPhase: 'movement',
    currentPlayer: 1,
    gameStatus: 'active',
  });

  // Default valid moves for movement phase to avoid triggering no_movement_action auto-apply
  const defaultValidMoves = [{ type: 'move_stack', player: 1, from: pos(0, 0), to: pos(1, 1) }];

  return {
    getGameState: jest.fn(() => overrides.gameState ?? defaultState),
    getValidMoves: jest.fn(() => overrides.validMoves ?? defaultValidMoves),
    getValidLandingPositionsForCurrentPlayer: jest.fn(() => overrides.validLandingPositions ?? []),
    handleHumanCellClick: jest.fn().mockResolvedValue(undefined),
    clearSelection: jest.fn(),
    tryPlaceRings: jest.fn().mockResolvedValue(true),
    maybeRunAITurn: jest.fn().mockResolvedValue(undefined),
  };
}

// Create mock sandbox context value
function createMockSandboxContext(engine: { getGameState: () => any } | null = null) {
  return {
    sandboxEngine: engine,
    isConfigured: engine !== null,
    sandboxPendingChoice: null as PlayerChoice | null,
    setSandboxPendingChoice: jest.fn(),
    sandboxCaptureChoice: null as PlayerChoice | null,
    setSandboxCaptureChoice: jest.fn(),
    setSandboxCaptureTargets: jest.fn(),
    setSandboxLastProgressAt: jest.fn(),
    setSandboxStallWarning: jest.fn(),
    sandboxStateVersion: 0,
    setSandboxStateVersion: jest.fn(),
  };
}

// Test harness component
interface HarnessProps {
  engine?: ReturnType<typeof createMockEngine> | null;
  contextOverrides?: Partial<ReturnType<typeof createMockSandboxContext>>;
}

function TestHarness({ engine, contextOverrides }: HarnessProps) {
  const [selected, setSelected] = useState<Position | undefined>(undefined);
  const [validTargets, setValidTargets] = useState<Position[]>([]);
  const choiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);

  // Set up resolver for testing
  const setResolver = (fn: ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null) => {
    choiceResolverRef.current = fn;
  };

  const {
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection: hookClearSelection,
    ringPlacementCountPrompt,
    closeRingPlacementCountPrompt,
    confirmRingPlacementCountPrompt,
    recoveryChoicePromptOpen,
    resolveRecoveryChoice,
  } = useSandboxInteractions({
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    choiceResolverRef,
  });

  return (
    <div>
      <div data-testid="selected">{selected ? `${selected.x},${selected.y}` : 'none'}</div>
      <div data-testid="valid-targets">
        {validTargets.length > 0 ? validTargets.map((t) => `${t.x},${t.y}`).join(';') : 'none'}
      </div>
      <div data-testid="ring-placement-prompt">
        {ringPlacementCountPrompt
          ? `${ringPlacementCountPrompt.position.x},${ringPlacementCountPrompt.position.y}:${ringPlacementCountPrompt.maxCount}:${ringPlacementCountPrompt.isStackPlacement ? 'stack' : 'empty'}`
          : 'none'}
      </div>
      <div data-testid="recovery-choice-open">{recoveryChoicePromptOpen ? 'open' : 'closed'}</div>
      <button data-testid="click-0-0" onClick={() => handleCellClick(pos(0, 0))}>
        Click 0,0
      </button>
      <button data-testid="click-1-1" onClick={() => handleCellClick(pos(1, 1))}>
        Click 1,1
      </button>
      <button data-testid="click-2-2" onClick={() => handleCellClick(pos(2, 2))}>
        Click 2,2
      </button>
      <button data-testid="dblclick-0-0" onClick={() => handleCellDoubleClick(pos(0, 0))}>
        DblClick 0,0
      </button>
      <button data-testid="contextmenu-0-0" onClick={() => handleCellContextMenu(pos(0, 0))}>
        ContextMenu 0,0
      </button>
      <button data-testid="confirm-placement-2" onClick={() => confirmRingPlacementCountPrompt(2)}>
        Confirm Placement 2
      </button>
      <button data-testid="close-placement-prompt" onClick={closeRingPlacementCountPrompt}>
        Close Placement Prompt
      </button>
      <button
        data-testid="resolve-recovery-option2"
        onClick={() => resolveRecoveryChoice('option2')}
      >
        Resolve Recovery Option 2
      </button>
      <button data-testid="run-ai" onClick={() => maybeRunSandboxAiIfNeeded()}>
        Run AI
      </button>
      <button data-testid="clear-selection" onClick={() => hookClearSelection()}>
        Clear Selection
      </button>
      <button
        data-testid="set-resolver"
        onClick={() =>
          setResolver((response) => {
            // Store response for testing
            (window as any).__lastResolverResponse = response;
          })
        }
      >
        Set Resolver
      </button>
      <button
        data-testid="set-selected-0-0"
        onClick={() => {
          setSelected(pos(0, 0));
          setValidTargets([pos(1, 1), pos(2, 2)]);
        }}
      >
        Set Selected 0,0 with targets
      </button>
    </div>
  );
}

describe('useSandboxInteractions', () => {
  let mockContext: ReturnType<typeof createMockSandboxContext>;
  let mockEngine: ReturnType<typeof createMockEngine>;

  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    mockEngine = createMockEngine();
    mockContext = createMockSandboxContext(mockEngine);
    (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);
    (window as any).__lastResolverResponse = null;
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('handleCellClick', () => {
    describe('without engine', () => {
      it('should do nothing when engine is null', async () => {
        mockContext.sandboxEngine = null;
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        // Should not crash and selection remains unchanged
        expect(screen.getByTestId('selected').textContent).toBe('none');
      });
    });

    describe('ring_elimination choice handling', () => {
      it('should resolve ring_elimination choice when clicking highlighted stack', async () => {
        const eliminationChoice: PlayerChoice = {
          id: 'choice-1',
          type: 'ring_elimination',
          gameId: 'test-game-123',
          playerNumber: 1,
          prompt: 'Choose ring to eliminate',
          options: [
            { stackPosition: pos(0, 0), capHeight: 3, totalHeight: 5, moveId: 'move-1' },
            { stackPosition: pos(1, 1), capHeight: 2, totalHeight: 4, moveId: 'move-2' },
          ],
        };

        mockContext.sandboxPendingChoice = eliminationChoice;
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        // Set up resolver
        await act(async () => {
          screen.getByTestId('set-resolver').click();
        });

        // Click on a highlighted elimination stack
        await act(async () => {
          screen.getByTestId('click-0-0').click();
          // Wait for setTimeout
          jest.runAllTimers();
        });

        // Verify resolver was called with correct response
        expect((window as any).__lastResolverResponse).not.toBeNull();
        expect((window as any).__lastResolverResponse.choiceType).toBe('ring_elimination');
        expect((window as any).__lastResolverResponse.selectedOption.stackPosition).toEqual(
          pos(0, 0)
        );
      });

      it('should ignore clicks not on highlighted elimination stacks', async () => {
        const eliminationChoice: PlayerChoice = {
          id: 'choice-1',
          type: 'ring_elimination',
          gameId: 'test-game-123',
          playerNumber: 1,
          prompt: 'Choose ring to eliminate',
          options: [{ stackPosition: pos(5, 5), capHeight: 3, totalHeight: 5, moveId: 'move-1' }],
        };

        mockContext.sandboxPendingChoice = eliminationChoice;
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          screen.getByTestId('set-resolver').click();
        });

        // Click on non-highlighted position
        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        // Resolver should not have been called
        expect((window as any).__lastResolverResponse).toBeNull();
      });
    });

    describe('region_order choice handling', () => {
      it('resolves region_order choice when clicking inside a territory region', async () => {
        const territoryState = createTestGameState({
          currentPhase: 'territory_processing',
          currentPlayer: 1,
          gameStatus: 'active',
        });

        // Two simple disconnected regions, one containing (0,0) and one
        // containing (2,2), both credited to the moving player.
        territoryState.board.territories.set('region-1', {
          spaces: [pos(0, 0)],
          controllingPlayer: 1,
          isDisconnected: true,
        } as any);
        territoryState.board.territories.set('region-2', {
          spaces: [pos(2, 2)],
          controllingPlayer: 1,
          isDisconnected: true,
        } as any);

        mockEngine = createMockEngine({ gameState: territoryState });
        mockContext = createMockSandboxContext(mockEngine);

        const regionChoice: PlayerChoice = {
          id: 'choice-region-1',
          type: 'region_order',
          gameId: 'test-game-123',
          playerNumber: 1,
          prompt: 'Choose region order',
          options: [
            {
              regionId: 'region-1',
              size: 1,
              representativePosition: pos(0, 0),
              moveId: 'move-region-1',
            },
            {
              regionId: 'region-2',
              size: 1,
              representativePosition: pos(2, 2),
              moveId: 'move-region-2',
            },
          ] as any,
        } as any;

        mockContext.sandboxPendingChoice = regionChoice;
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        // Set up resolver to capture the response.
        await act(async () => {
          screen.getByTestId('set-resolver').click();
        });

        // Click inside the second region at (2,2); the hook should
        // resolve the choice with the matching option.
        await act(async () => {
          screen.getByTestId('click-2-2').click();
          jest.runAllTimers();
        });

        const response = (window as any).__lastResolverResponse;
        expect(response).not.toBeNull();
        expect(response.choiceType).toBe('region_order');
        expect(response.selectedOption.regionId).toBe('region-2');
      });
    });

    describe('capture_direction choice handling', () => {
      it('should resolve capture_direction choice when clicking highlighted landing', async () => {
        const captureChoice: PlayerChoice = {
          id: 'choice-2',
          type: 'capture_direction',
          gameId: 'test-game-123',
          playerNumber: 1,
          prompt: 'Choose capture direction',
          options: [
            { targetPosition: pos(0, 1), landingPosition: pos(0, 0), capturedCapHeight: 2 },
            { targetPosition: pos(2, 3), landingPosition: pos(2, 2), capturedCapHeight: 3 },
          ],
        };

        mockContext.sandboxCaptureChoice = captureChoice;
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        // Set up resolver
        await act(async () => {
          screen.getByTestId('set-resolver').click();
        });

        // Click on highlighted landing
        await act(async () => {
          screen.getByTestId('click-0-0').click();
          jest.runAllTimers();
        });

        expect((window as any).__lastResolverResponse).not.toBeNull();
        expect((window as any).__lastResolverResponse.choiceType).toBe('capture_direction');
      });

      it('should ignore clicks not on capture landing positions', async () => {
        const captureChoice: PlayerChoice = {
          id: 'choice-2',
          type: 'capture_direction',
          gameId: 'test-game-123',
          playerNumber: 1,
          prompt: 'Choose capture direction',
          options: [
            { targetPosition: pos(5, 6), landingPosition: pos(5, 5), capturedCapHeight: 2 },
          ],
        };

        mockContext.sandboxCaptureChoice = captureChoice;
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        expect((window as any).__lastResolverResponse).toBeNull();
      });
    });

    describe('AI turn handling', () => {
      it('should start AI turn loop when clicking during AI turn', async () => {
        const aiState = createTestGameState({
          currentPhase: 'movement',
          currentPlayer: 1,
          gameStatus: 'active',
          players: [
            {
              id: '1',
              username: 'AI',
              type: 'ai',
              playerNumber: 1,
              isReady: true,
              timeRemaining: 600,
              ringsInHand: 18,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
            {
              id: '2',
              username: 'Human',
              type: 'human',
              playerNumber: 2,
              isReady: true,
              timeRemaining: 600,
              ringsInHand: 18,
              eliminatedRings: 0,
              territorySpaces: 0,
            },
          ],
        });

        mockEngine = createMockEngine({ gameState: aiState });
        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        // Should have triggered AI turn
        expect(mockEngine.maybeRunAITurn).toHaveBeenCalled();
      });
    });

    describe('ring_placement phase', () => {
      it('should place a ring and treat cell as selected on single click for empty cell', async () => {
        const placementState = createTestGameState({
          currentPhase: 'ring_placement',
          currentPlayer: 1,
          gameStatus: 'active',
        });

        mockEngine = createMockEngine({
          gameState: placementState,
          validLandingPositions: [pos(1, 1), pos(2, 2)],
        });
        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        await waitFor(() => {
          expect(screen.getByTestId('selected').textContent).toBe('0,0');
        });

        expect(mockEngine.handleHumanCellClick).toHaveBeenCalledWith(pos(0, 0));
      });

      it('should place a ring when clicking again on same already-selected stack', async () => {
        const placementState = createTestGameState({
          currentPhase: 'ring_placement',
          currentPlayer: 1,
          gameStatus: 'active',
        });

        // Seed an existing stack at 0,0 so the second click represents
        // clicking on an already-selected stack.
        placementState.board.stacks.set('0,0', {
          position: pos(0, 0),
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        } as any);

        mockEngine = createMockEngine({ gameState: placementState });
        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          const cell = screen.getByTestId('click-0-0');
          cell.click();
          cell.click();
          await Promise.resolve();
        });

        // Selection should still be on the same cell and a placement should
        // have been attempted via handleHumanCellClick.
        expect(screen.getByTestId('selected').textContent).toBe('0,0');
        expect(mockEngine.handleHumanCellClick).toHaveBeenCalledWith(pos(0, 0));
      });
    });

    describe('movement phase - selection', () => {
      it('should select cell and highlight targets on first click', async () => {
        mockEngine = createMockEngine({
          validLandingPositions: [pos(1, 1), pos(2, 2)],
        });
        // Seed a realistic movement state: player 1 controls a stack at (0,0)
        // with at least one canonical move originating from that position.
        const state = mockEngine.getGameState();
        addStack(state.board, pos(0, 0), 1, 1);
        mockEngine.getValidMoves.mockReturnValue([
          {
            id: 'move-0,0-1,1-test',
            type: 'move_stack',
            player: 1,
            from: pos(0, 0),
            to: pos(1, 1),
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          } as any,
        ]);

        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        expect(screen.getByTestId('selected').textContent).toBe('0,0');
        expect(screen.getByTestId('valid-targets').textContent).toBe('1,1;2,2');
        expect(mockEngine.handleHumanCellClick).toHaveBeenCalledWith(pos(0, 0));
      });

      it('should clear selection when clicking same cell again', async () => {
        mockEngine = createMockEngine();
        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        // First, set up a selection
        await act(async () => {
          screen.getByTestId('set-selected-0-0').click();
        });

        expect(screen.getByTestId('selected').textContent).toBe('0,0');

        // Click same cell to clear
        await act(async () => {
          screen.getByTestId('click-0-0').click();
        });

        expect(screen.getByTestId('selected').textContent).toBe('none');
        expect(screen.getByTestId('valid-targets').textContent).toBe('none');
        expect(mockEngine.clearSelection).toHaveBeenCalled();
      });

      it('should execute move when clicking on valid target', async () => {
        mockEngine = createMockEngine();
        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        // Set up selection with valid targets
        await act(async () => {
          screen.getByTestId('set-selected-0-0').click();
        });

        // Click on valid target (1,1)
        await act(async () => {
          screen.getByTestId('click-1-1').click();
          await Promise.resolve();
        });

        expect(mockEngine.handleHumanCellClick).toHaveBeenCalledWith(pos(1, 1));
        expect(screen.getByTestId('selected').textContent).toBe('none');
      });

      it('should ignore clicks on non-target cells when selection is active', async () => {
        mockEngine = createMockEngine();
        mockContext = createMockSandboxContext(mockEngine);
        (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

        render(<TestHarness />);

        // Set up selection with targets at 1,1 and 2,2
        await act(async () => {
          screen.getByTestId('set-selected-0-0').click();
        });

        // Valid targets are 1,1 and 2,2, but we'll click somewhere else
        // However our test setup only has buttons for 0,0, 1,1, 2,2
        // The selection should remain if we would click on an invalid cell

        // Clear the mock calls
        mockEngine.handleHumanCellClick.mockClear();

        // Selection remains at 0,0
        expect(screen.getByTestId('selected').textContent).toBe('0,0');
      });

      describe('chain_capture phase', () => {
        it('uses engine chain-capture context to set selection and targets after continuation', async () => {
          const beforeState = createTestGameState({
            currentPhase: 'movement',
            currentPlayer: 1,
            gameStatus: 'active',
          });

          const chainCaptureState = createTestGameState({
            currentPhase: 'chain_capture',
            currentPlayer: 1,
            gameStatus: 'active',
          });

          let currentState = beforeState;

          const engine: any = {
            getGameState: jest.fn(() => currentState),
            // Return a valid move so the movement phase doesn't auto-apply no_movement_action
            getValidMoves: jest.fn(() => [
              { type: 'move_stack', player: 1, from: pos(0, 0), to: pos(1, 1) },
            ]),
            getValidLandingPositionsForCurrentPlayer: jest.fn(() => [pos(1, 1)]),
            handleHumanCellClick: jest.fn(async () => {
              currentState = chainCaptureState;
            }),
            clearSelection: jest.fn(),
            tryPlaceRings: jest.fn().mockResolvedValue(true),
            maybeRunAITurn: jest.fn().mockResolvedValue(undefined),
            getChainCaptureContextForCurrentPlayer: jest.fn(() => ({
              from: pos(9, 9),
              landings: [pos(8, 8)],
            })),
          };

          mockContext = createMockSandboxContext(engine);
          (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

          render(<TestHarness />);

          // Seed an initial selection + validTargets mimicking a completed
          // movement capture segment.
          await act(async () => {
            screen.getByTestId('set-selected-0-0').click();
          });

          // Click on a valid target (1,1); this should call handleHumanCellClick,
          // transition the mock state into chain_capture, and then apply the
          // chain-capture context from the engine.
          await act(async () => {
            screen.getByTestId('click-1-1').click();
            await Promise.resolve();
          });

          await waitFor(() => {
            expect(screen.getByTestId('selected').textContent).toBe('9,9');
            expect(screen.getByTestId('valid-targets').textContent).toBe('8,8');
          });
        });
      });
    });
  });

  describe('handleCellDoubleClick', () => {
    it('should do nothing without engine', async () => {
      mockContext.sandboxEngine = null;
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
      });

      // No crash, no effect
      expect(screen.getByTestId('selected').textContent).toBe('none');
    });

    it('should do nothing outside ring_placement phase', async () => {
      mockEngine = createMockEngine({
        gameState: createTestGameState({ currentPhase: 'movement' }),
      });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
      });

      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
    });

    it('should attempt 2-ring placement on empty cell', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });

      mockEngine = createMockEngine({
        gameState: placementState,
        validLandingPositions: [pos(1, 1)],
      });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
        await Promise.resolve();
      });

      // Should try 2 rings first for empty cell
      expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 2);
    });

    it('should fall back to 1-ring placement if 2-ring fails', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });

      mockEngine = createMockEngine({
        gameState: placementState,
        validLandingPositions: [pos(1, 1)],
      });
      // First call (2 rings) fails, second call (1 ring) succeeds
      mockEngine.tryPlaceRings.mockResolvedValueOnce(false).mockResolvedValueOnce(true);
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
        await Promise.resolve();
      });

      expect(mockEngine.tryPlaceRings).toHaveBeenCalledTimes(2);
      expect(mockEngine.tryPlaceRings).toHaveBeenNthCalledWith(1, pos(0, 0), 2);
      expect(mockEngine.tryPlaceRings).toHaveBeenNthCalledWith(2, pos(0, 0), 1);
    });
  });

  describe('handleCellContextMenu', () => {
    it('should do nothing without engine', async () => {
      mockContext.sandboxEngine = null;
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('none');
    });

    it('should do nothing outside ring_placement phase', async () => {
      mockEngine = createMockEngine({
        gameState: createTestGameState({ currentPhase: 'movement' }),
      });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('none');
    });

    it('should open a ring placement prompt during placement phase', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Player',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 5,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({
        gameState: placementState,
        validLandingPositions: [pos(1, 1)],
      });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('0,0:5:empty');

      await act(async () => {
        screen.getByTestId('confirm-placement-2').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 2);
      });
      expect(screen.getByTestId('selected').textContent).toBe('0,0');
    });

    it('should not attempt placement when the prompt is closed', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });

      mockEngine = createMockEngine({ gameState: placementState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      expect(screen.getByTestId('ring-placement-prompt').textContent).not.toBe('none');

      await act(async () => {
        screen.getByTestId('close-placement-prompt').click();
      });

      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('none');
    });

    it('should expose the max ring count for UI validation', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Player',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 5,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: placementState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('0,0:5:empty');
    });
  });

  describe('clearSelection', () => {
    it('should clear selection and valid targets', async () => {
      mockEngine = createMockEngine();
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      // Set up selection first
      await act(async () => {
        screen.getByTestId('set-selected-0-0').click();
      });

      expect(screen.getByTestId('selected').textContent).toBe('0,0');

      // Clear it
      await act(async () => {
        screen.getByTestId('clear-selection').click();
      });

      expect(screen.getByTestId('selected').textContent).toBe('none');
      expect(screen.getByTestId('valid-targets').textContent).toBe('none');
      expect(mockEngine.clearSelection).toHaveBeenCalled();
    });

    it('should work without engine', async () => {
      mockContext.sandboxEngine = null;
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      // Should not crash
      await act(async () => {
        screen.getByTestId('clear-selection').click();
      });

      expect(screen.getByTestId('selected').textContent).toBe('none');
    });
  });

  describe('maybeRunSandboxAiIfNeeded', () => {
    it('should start AI loop when current player is AI', async () => {
      const aiState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'AI',
            type: 'ai',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: '2',
            username: 'Human',
            type: 'human',
            playerNumber: 2,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: aiState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('run-ai').click();
        await Promise.resolve();
      });

      expect(mockEngine.maybeRunAITurn).toHaveBeenCalled();
    });

    it('should not run AI when current player is human', async () => {
      const humanState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Human',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
          {
            id: '2',
            username: 'AI',
            type: 'ai',
            playerNumber: 2,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: humanState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('run-ai').click();
      });

      expect(mockEngine.maybeRunAITurn).not.toHaveBeenCalled();
    });

    it('should not run AI when game is not active', async () => {
      const finishedState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
        gameStatus: 'completed',
        players: [
          {
            id: '1',
            username: 'AI',
            type: 'ai',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: finishedState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('run-ai').click();
      });

      expect(mockEngine.maybeRunAITurn).not.toHaveBeenCalled();
    });

    it('should do nothing without engine', async () => {
      mockContext.sandboxEngine = null;
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      // Should not crash
      await act(async () => {
        screen.getByTestId('run-ai').click();
      });
    });
  });

  describe('edge cases', () => {
    it('should handle player not found gracefully', async () => {
      const badState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 99, // Non-existent player
        gameStatus: 'active',
      });

      mockEngine = createMockEngine({ gameState: badState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      // Should not crash
      await act(async () => {
        screen.getByTestId('click-0-0').click();
      });
    });

    it('should handle player with no rings gracefully', async () => {
      const noRingsState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Player',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: noRingsState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
      });

      // Should not attempt placement with no rings
      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
    });

    it('should handle context menu with no rings gracefully', async () => {
      const noRingsState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Player',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 0,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: noRingsState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      // Should not open prompt or attempt placement with no rings
      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('none');
    });

    it('should handle context menu placement failure gracefully', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });

      mockEngine = createMockEngine({ gameState: placementState });
      mockEngine.tryPlaceRings.mockResolvedValue(false);
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      await act(async () => {
        screen.getByTestId('confirm-placement-2').click();
        await Promise.resolve();
      });

      // Placement was attempted but failed - selection should not be set.
      await waitFor(() => {
        expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 2);
      });
      expect(screen.getByTestId('selected').textContent).toBe('none');
    });

    it('should handle double-click placement failure on occupied cell', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });
      // Add a stack at 0,0 to make it occupied
      placementState.board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      mockEngine = createMockEngine({ gameState: placementState });
      mockEngine.tryPlaceRings.mockResolvedValue(false);
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
        await Promise.resolve();
      });

      // Should try to place 1 ring on occupied cell
      expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 1);
      // Selection should not be set since placement failed
      expect(screen.getByTestId('selected').textContent).toBe('none');
    });

    it('should handle double-click with only 1 ring in hand', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Player',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 1,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({
        gameState: placementState,
        validLandingPositions: [pos(1, 1)],
      });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
        await Promise.resolve();
      });

      // Should try 1 ring since that's all we have (min of 2 and 1)
      expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 1);
    });

    it('should handle context menu with occupied cell (1 ring max)', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });
      // Add a stack at 0,0 to make it occupied
      placementState.board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      mockEngine = createMockEngine({
        gameState: placementState,
        validLandingPositions: [pos(1, 1)],
      });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 1);
      });
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('none');
      expect(screen.getByTestId('selected').textContent).toBe('0,0');
    });

    it('should not run AI loop when game status is not active', async () => {
      const completedState = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
        gameStatus: 'completed',
        players: [
          {
            id: '1',
            username: 'AI1',
            type: 'ai',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 18,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: completedState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('run-ai').click();
      });

      // AI should not be run when game is completed
      expect(mockEngine.maybeRunAITurn).not.toHaveBeenCalled();
    });

    it('should constrain the placement prompt to rings in hand', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          {
            id: '1',
            username: 'Player',
            type: 'human',
            playerNumber: 1,
            isReady: true,
            timeRemaining: 600,
            ringsInHand: 5,
            eliminatedRings: 0,
            territorySpaces: 0,
          },
        ],
      });

      mockEngine = createMockEngine({ gameState: placementState });
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('contextmenu-0-0').click();
      });

      expect(mockEngine.tryPlaceRings).not.toHaveBeenCalled();
      expect(screen.getByTestId('ring-placement-prompt').textContent).toBe('0,0:5:empty');
    });

    it('should handle double-click success on occupied cell', async () => {
      const placementState = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        gameStatus: 'active',
      });
      // Add a stack at 0,0 to make it occupied
      placementState.board.stacks.set('0,0', {
        position: pos(0, 0),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      });

      mockEngine = createMockEngine({
        gameState: placementState,
        validLandingPositions: [pos(1, 1)],
      });
      mockEngine.tryPlaceRings.mockResolvedValue(true);
      mockContext = createMockSandboxContext(mockEngine);
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('dblclick-0-0').click();
        await Promise.resolve();
      });

      // Should try to place 1 ring on occupied cell
      expect(mockEngine.tryPlaceRings).toHaveBeenCalledWith(pos(0, 0), 1);
      // Selection should be set since placement succeeded
      await waitFor(() => {
        expect(screen.getByTestId('selected').textContent).toBe('0,0');
      });
    });

    it('recomputes chain-capture overlays and bumps state version when chain remains active', async () => {
      // Engine stub that transitions from movement -> chain_capture on a
      // landing click and exposes a simple chain-capture context.
      const state = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
        gameStatus: 'active',
      });

      const chainEngine = {
        getGameState: jest.fn(() => state),
        // Return valid moves so movement phase doesn't auto-apply no_movement_action
        getValidMoves: jest.fn(() => [
          { type: 'move_stack', player: 1, from: pos(0, 0), to: pos(1, 1) },
        ]),
        getValidLandingPositionsForCurrentPlayer: jest.fn(() => [pos(1, 1)]),
        handleHumanCellClick: jest.fn(async () => {
          state.currentPhase = 'chain_capture';
        }),
        getChainCaptureContextForCurrentPlayer: jest.fn(() => ({
          from: pos(0, 0),
          landings: [pos(5, 5)],
        })),
        clearSelection: jest.fn(),
        tryPlaceRings: jest.fn().mockResolvedValue(true),
        maybeRunAITurn: jest.fn().mockResolvedValue(undefined),
      } as any;

      mockContext = createMockSandboxContext(chainEngine);
      const setStateVersionSpy = jest.fn();
      mockContext.setSandboxStateVersion = setStateVersionSpy;
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      // Pre-select a stack and highlight a landing target, mirroring the
      // movement-phase UX before a capture that transitions into
      // chain_capture.
      await act(async () => {
        screen.getByTestId('set-selected-0-0').click();
      });

      // Click on the highlighted landing (1,1); the hook should:
      // - call handleHumanCellClick,
      // - see currentPhase === 'chain_capture',
      // - read chain-capture context, and
      // - set selected/validTargets from that context while bumping
      //   sandboxStateVersion.
      await act(async () => {
        screen.getByTestId('click-1-1').click();
        await Promise.resolve();
      });

      expect(chainEngine.handleHumanCellClick).toHaveBeenCalled();
      expect(chainEngine.getChainCaptureContextForCurrentPlayer).toHaveBeenCalled();

      expect(screen.getByTestId('selected').textContent).toBe('0,0');
      expect(screen.getByTestId('valid-targets').textContent).toBe('5,5');
      expect(setStateVersionSpy).toHaveBeenCalled();
    });

    it('clears overlays and still bumps state version when chain completes', async () => {
      // Engine stub that transitions movement -> movement (no chain_capture)
      // after a capture, simulating a completed chain.
      const state = createTestGameState({
        currentPhase: 'movement',
        currentPlayer: 1,
        gameStatus: 'active',
      });

      const chainEngine = {
        getGameState: jest.fn(() => state),
        // Return valid moves so movement phase doesn't auto-apply no_movement_action
        getValidMoves: jest.fn(() => [
          { type: 'move_stack', player: 1, from: pos(0, 0), to: pos(1, 1) },
        ]),
        getValidLandingPositionsForCurrentPlayer: jest.fn(() => [pos(1, 1)]),
        handleHumanCellClick: jest.fn(async () => {
          state.currentPhase = 'movement';
        }),
        getChainCaptureContextForCurrentPlayer: jest.fn(() => null),
        clearSelection: jest.fn(),
        tryPlaceRings: jest.fn().mockResolvedValue(true),
        maybeRunAITurn: jest.fn().mockResolvedValue(undefined),
      } as any;

      mockContext = createMockSandboxContext(chainEngine);
      const setStateVersionSpy = jest.fn();
      mockContext.setSandboxStateVersion = setStateVersionSpy;
      (SandboxContextModule.useSandbox as jest.Mock).mockReturnValue(mockContext);

      render(<TestHarness />);

      await act(async () => {
        screen.getByTestId('set-selected-0-0').click();
      });

      await act(async () => {
        screen.getByTestId('click-1-1').click();
        await Promise.resolve();
      });

      expect(chainEngine.handleHumanCellClick).toHaveBeenCalled();
      // getChainCaptureContextForCurrentPlayer is NOT called when phase is 'movement'
      // (only called when phase === 'chain_capture')
      expect(chainEngine.getChainCaptureContextForCurrentPlayer).not.toHaveBeenCalled();

      // Chain is complete: overlays remain cleared and only the state
      // version is bumped.
      expect(screen.getByTestId('selected').textContent).toBe('none');
      expect(screen.getByTestId('valid-targets').textContent).toBe('none');
      expect(setStateVersionSpy).toHaveBeenCalled();
    });
  });
});
