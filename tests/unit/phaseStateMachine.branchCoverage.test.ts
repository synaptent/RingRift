/**
 * phaseStateMachine.branchCoverage.test.ts
 *
 * Branch coverage tests for phaseStateMachine.ts targeting uncovered branches:
 * - determineNextPhase: all phase transition branches
 * - createTurnLogicDelegates: getNextPlayerNumber branches
 * - phaseRequiresDecision: all phase cases
 * - shouldAutoAdvancePhase: all phase cases
 * - PhaseStateMachine class methods
 * - createTurnProcessingState factory
 *
 * @deprecated These tests cover phaseStateMachine.ts which is deprecated.
 * The FSM-based orchestration in TurnStateMachine.ts is now canonical for
 * phase transitions and validation (RR-CANON-R070). These tests remain to
 * ensure backward compatibility while phaseStateMachine is still used by
 * processPostMovePhases(). When that function is fully migrated to FSM,
 * these tests can be removed along with phaseStateMachine.ts.
 */

import {
  determineNextPhase,
  createTurnLogicDelegates,
  toPerTurnState,
  updateFlagsFromPerTurnState,
  phaseRequiresDecision,
  shouldAutoAdvancePhase,
  PhaseStateMachine,
  createTurnProcessingState,
  PhaseContext,
} from '../../src/shared/engine/orchestration/phaseStateMachine';
import type {
  TurnProcessingState,
  PerTurnFlags,
} from '../../src/shared/engine/orchestration/types';
import type { GameState, GamePhase, Move, Position, Player } from '../../src/shared/types/game';

// Helper to create a minimal Player
function makePlayer(playerNumber: number, overrides: Partial<Player> = {}): Player {
  return {
    id: `p${playerNumber}`,
    username: `Player${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand: 10,
    eliminatedRings: 0,
    territorySpaces: 0,
    ...overrides,
  };
}

// Helper to create a minimal GameState
function makeGameState(overrides: Partial<GameState> = {}): GameState {
  return {
    id: 'test-game',
    boardType: 'square8',
    board: {
      type: 'square8',
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      formedLines: [],
      territories: new Map(),
      eliminatedRings: { 1: 0, 2: 0 },
    },
    players: [makePlayer(1), makePlayer(2)],
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    moveHistory: [],
    history: [],
    gameStatus: 'active',
    winner: undefined,
    timeControl: { initialTime: 600000, increment: 0, type: 'rapid' },
    spectators: [],
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 15,
    territoryVictoryThreshold: 8,
    ...overrides,
  } as GameState;
}

// Helper to create a minimal Move
function makeMove(overrides: Partial<Move> = {}): Move {
  return {
    id: 'test-move',
    type: 'placement',
    player: 1,
    to: { x: 0, y: 0 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    ...overrides,
  } as Move;
}

// Helper to create PerTurnFlags
function makePerTurnFlags(overrides: Partial<PerTurnFlags> = {}): PerTurnFlags {
  return {
    hasPlacedThisTurn: false,
    mustMoveFromStackKey: undefined,
    eliminationRewardPending: false,
    eliminationRewardCount: 0,
    ...overrides,
  };
}

// Helper to create TurnProcessingState
function makeTurnProcessingState(
  gameState: GameState,
  move: Move,
  overrides: Partial<TurnProcessingState> = {}
): TurnProcessingState {
  return {
    gameState,
    originalMove: move,
    perTurnFlags: makePerTurnFlags(),
    pendingLines: [],
    pendingRegions: [],
    chainCaptureInProgress: false,
    chainCapturePosition: undefined,
    events: [],
    phasesTraversed: [gameState.currentPhase],
    startTime: Date.now(),
    ...overrides,
  };
}

describe('phaseStateMachine branch coverage', () => {
  describe('determineNextPhase', () => {
    describe('ring_placement phase', () => {
      it('transitions to movement when hasAnyMovement is true', () => {
        const result = determineNextPhase('ring_placement', 'placement', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: true,
          hasAnyCapture: false,
        });

        expect(result).toBe('movement');
      });

      it('transitions to movement when hasAnyCapture is true', () => {
        const result = determineNextPhase('ring_placement', 'placement', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: true,
        });

        expect(result).toBe('movement');
      });

      it('transitions to line_processing when no movement or capture', () => {
        const result = determineNextPhase('ring_placement', 'placement', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('line_processing');
      });
    });

    describe('movement phase', () => {
      it('transitions to chain_capture when available', () => {
        const result = determineNextPhase('movement', 'movement', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: true,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('chain_capture');
      });

      it('transitions to line_processing when no chain captures', () => {
        const result = determineNextPhase('movement', 'movement', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('line_processing');
      });
    });

    describe('capture phase', () => {
      it('transitions to chain_capture when available', () => {
        const result = determineNextPhase('capture', 'capture', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: true,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('chain_capture');
      });

      it('transitions to line_processing when no chain captures', () => {
        const result = determineNextPhase('capture', 'capture', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('line_processing');
      });
    });

    describe('chain_capture phase', () => {
      it('stays in chain_capture when more captures available', () => {
        const result = determineNextPhase('chain_capture', 'capture', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: true,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('chain_capture');
      });

      it('transitions to line_processing when no more chain captures', () => {
        const result = determineNextPhase('chain_capture', 'capture', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('line_processing');
      });
    });

    describe('line_processing phase', () => {
      it('stays in line_processing when more lines to process', () => {
        const result = determineNextPhase('line_processing', 'process_line', {
          hasMoreLinesToProcess: true,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('line_processing');
      });

      it('transitions to territory_processing when no more lines', () => {
        const result = determineNextPhase('line_processing', 'process_line', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('territory_processing');
      });
    });

    describe('territory_processing phase', () => {
      it('stays in territory_processing (handled by turn advance)', () => {
        const result = determineNextPhase('territory_processing', 'choose_territory_option', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: true,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('territory_processing');
      });
    });

    describe('default case', () => {
      it('returns current phase for unknown phases', () => {
        const result = determineNextPhase('game_over' as GamePhase, 'placement', {
          hasMoreLinesToProcess: false,
          hasMoreRegionsToProcess: false,
          chainCapturesAvailable: false,
          hasAnyMovement: false,
          hasAnyCapture: false,
        });

        expect(result).toBe('game_over');
      });
    });
  });

  describe('createTurnLogicDelegates', () => {
    describe('getNextPlayerNumber', () => {
      it('returns current player when players array is empty', () => {
        const processingState = makeTurnProcessingState(makeGameState({ players: [] }), makeMove());
        const delegates = createTurnLogicDelegates(processingState, {
          getPlayerStacks: () => [],
          hasAnyPlacement: () => false,
          hasAnyMovement: () => false,
          hasAnyCapture: () => false,
          applyForcedElimination: (state) => state,
        });

        const result = delegates.getNextPlayerNumber(makeGameState({ players: [] }), 1);

        expect(result).toBe(1);
      });

      it('returns first player when current not found', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const delegates = createTurnLogicDelegates(processingState, {
          getPlayerStacks: () => [],
          hasAnyPlacement: () => false,
          hasAnyMovement: () => false,
          hasAnyCapture: () => false,
          applyForcedElimination: (state) => state,
        });

        const result = delegates.getNextPlayerNumber(makeGameState(), 99);

        expect(result).toBe(1); // First player
      });

      it('wraps around to first player after last', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const delegates = createTurnLogicDelegates(processingState, {
          getPlayerStacks: () => [],
          hasAnyPlacement: () => false,
          hasAnyMovement: () => false,
          hasAnyCapture: () => false,
          applyForcedElimination: (state) => state,
        });

        const result = delegates.getNextPlayerNumber(makeGameState(), 2);

        expect(result).toBe(1); // Wraps back to player 1
      });

      it('advances to next player normally', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const delegates = createTurnLogicDelegates(processingState, {
          getPlayerStacks: () => [],
          hasAnyPlacement: () => false,
          hasAnyMovement: () => false,
          hasAnyCapture: () => false,
          applyForcedElimination: (state) => state,
        });

        const result = delegates.getNextPlayerNumber(makeGameState(), 1);

        expect(result).toBe(2);
      });
    });

    it('passes through callbacks correctly', () => {
      const mockGetPlayerStacks = jest.fn().mockReturnValue([]);
      const mockHasAnyPlacement = jest.fn().mockReturnValue(true);
      const mockHasAnyMovement = jest.fn().mockReturnValue(true);
      const mockHasAnyCapture = jest.fn().mockReturnValue(true);
      const mockApplyForcedElimination = jest.fn((state) => state);

      const processingState = makeTurnProcessingState(makeGameState(), makeMove());
      const delegates = createTurnLogicDelegates(processingState, {
        getPlayerStacks: mockGetPlayerStacks,
        hasAnyPlacement: mockHasAnyPlacement,
        hasAnyMovement: mockHasAnyMovement,
        hasAnyCapture: mockHasAnyCapture,
        applyForcedElimination: mockApplyForcedElimination,
      });

      const state = makeGameState();
      delegates.getPlayerStacks(state, 1);
      delegates.hasAnyPlacement(state, 1);
      delegates.hasAnyMovement(state, 1, {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
      });
      delegates.hasAnyCapture(state, 1, {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
      });
      delegates.applyForcedElimination(state, 1);

      expect(mockGetPlayerStacks).toHaveBeenCalledWith(state, 1);
      expect(mockHasAnyPlacement).toHaveBeenCalledWith(state, 1);
      expect(mockHasAnyMovement).toHaveBeenCalled();
      expect(mockHasAnyCapture).toHaveBeenCalled();
      expect(mockApplyForcedElimination).toHaveBeenCalledWith(state, 1);
    });
  });

  describe('toPerTurnState', () => {
    it('converts PerTurnFlags to PerTurnState', () => {
      const flags: PerTurnFlags = {
        hasPlacedThisTurn: true,
        mustMoveFromStackKey: '1,1',
        eliminationRewardPending: true,
        eliminationRewardCount: 3,
      };

      const result = toPerTurnState(flags);

      expect(result.hasPlacedThisTurn).toBe(true);
      expect(result.mustMoveFromStackKey).toBe('1,1');
    });
  });

  describe('updateFlagsFromPerTurnState', () => {
    it('updates flags from PerTurnState', () => {
      const flags: PerTurnFlags = {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
        eliminationRewardPending: false,
        eliminationRewardCount: 0,
      };
      const state = {
        hasPlacedThisTurn: true,
        mustMoveFromStackKey: '2,2',
      };

      const result = updateFlagsFromPerTurnState(flags, state);

      expect(result.hasPlacedThisTurn).toBe(true);
      expect(result.mustMoveFromStackKey).toBe('2,2');
      expect(result.eliminationRewardPending).toBe(false); // Preserved
      expect(result.eliminationRewardCount).toBe(0); // Preserved
    });
  });

  describe('phaseRequiresDecision', () => {
    it('returns true for line_processing with multiple lines', () => {
      const context: PhaseContext = {
        pendingLineCount: 2,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 0,
      };

      expect(phaseRequiresDecision('line_processing', context)).toBe(true);
    });

    it('returns false for line_processing with single line', () => {
      const context: PhaseContext = {
        pendingLineCount: 1,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 0,
      };

      expect(phaseRequiresDecision('line_processing', context)).toBe(false);
    });

    it('returns true for territory_processing with multiple regions', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 3,
        chainCaptureOptionsCount: 0,
      };

      expect(phaseRequiresDecision('territory_processing', context)).toBe(true);
    });

    it('returns false for territory_processing with single region', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 1,
        chainCaptureOptionsCount: 0,
      };

      expect(phaseRequiresDecision('territory_processing', context)).toBe(false);
    });

    it('returns true for chain_capture with multiple options', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 2,
      };

      expect(phaseRequiresDecision('chain_capture', context)).toBe(true);
    });

    it('returns false for chain_capture with single option', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 1,
      };

      expect(phaseRequiresDecision('chain_capture', context)).toBe(false);
    });

    it('returns false for other phases', () => {
      const context: PhaseContext = {
        pendingLineCount: 5,
        pendingRegionCount: 5,
        chainCaptureOptionsCount: 5,
      };

      expect(phaseRequiresDecision('ring_placement', context)).toBe(false);
      expect(phaseRequiresDecision('movement', context)).toBe(false);
    });
  });

  describe('shouldAutoAdvancePhase', () => {
    it('returns true for line_processing with 0 or 1 lines', () => {
      expect(
        shouldAutoAdvancePhase('line_processing', {
          pendingLineCount: 0,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(true);

      expect(
        shouldAutoAdvancePhase('line_processing', {
          pendingLineCount: 1,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(true);
    });

    it('returns false for line_processing with multiple lines', () => {
      expect(
        shouldAutoAdvancePhase('line_processing', {
          pendingLineCount: 2,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(false);
    });

    it('returns true for territory_processing with 0 or 1 regions', () => {
      expect(
        shouldAutoAdvancePhase('territory_processing', {
          pendingLineCount: 0,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(true);

      expect(
        shouldAutoAdvancePhase('territory_processing', {
          pendingLineCount: 0,
          pendingRegionCount: 1,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(true);
    });

    it('returns false for territory_processing with multiple regions', () => {
      expect(
        shouldAutoAdvancePhase('territory_processing', {
          pendingLineCount: 0,
          pendingRegionCount: 2,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(false);
    });

    it('returns true for chain_capture with exactly 1 option', () => {
      expect(
        shouldAutoAdvancePhase('chain_capture', {
          pendingLineCount: 0,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 1,
        })
      ).toBe(true);
    });

    it('returns false for chain_capture with 0 or multiple options', () => {
      expect(
        shouldAutoAdvancePhase('chain_capture', {
          pendingLineCount: 0,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 0,
        })
      ).toBe(false);

      expect(
        shouldAutoAdvancePhase('chain_capture', {
          pendingLineCount: 0,
          pendingRegionCount: 0,
          chainCaptureOptionsCount: 2,
        })
      ).toBe(false);
    });

    it('returns false for other phases', () => {
      const context: PhaseContext = {
        pendingLineCount: 1,
        pendingRegionCount: 1,
        chainCaptureOptionsCount: 1,
      };

      expect(shouldAutoAdvancePhase('ring_placement', context)).toBe(false);
      expect(shouldAutoAdvancePhase('movement', context)).toBe(false);
    });
  });

  describe('PhaseStateMachine', () => {
    describe('constructor and getters', () => {
      it('initializes with provided state', () => {
        const gameState = makeGameState({ currentPhase: 'movement' });
        const processingState = makeTurnProcessingState(gameState, makeMove());
        const machine = new PhaseStateMachine(processingState);

        expect(machine.currentPhase).toBe('movement');
        expect(machine.gameState).toBe(gameState);
        expect(machine.processingState).toBe(processingState);
      });
    });

    describe('updateGameState', () => {
      it('updates game state and tracks phase', () => {
        const initialState = makeGameState({ currentPhase: 'ring_placement' });
        const processingState = makeTurnProcessingState(initialState, makeMove());
        const machine = new PhaseStateMachine(processingState);

        const newState = makeGameState({ currentPhase: 'movement' });
        machine.updateGameState(newState);

        expect(machine.gameState).toBe(newState);
        expect(machine.processingState.phasesTraversed).toContain('movement');
      });
    });

    describe('updateFlags', () => {
      it('merges partial flags', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        machine.updateFlags({ hasPlacedThisTurn: true });

        expect(machine.processingState.perTurnFlags.hasPlacedThisTurn).toBe(true);
      });
    });

    describe('setChainCapture', () => {
      it('sets chain capture state', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const position: Position = { x: 1, y: 2 };
        machine.setChainCapture(true, position);

        expect(machine.processingState.chainCaptureInProgress).toBe(true);
        expect(machine.processingState.chainCapturePosition).toEqual(position);
      });

      it('clears chain capture state', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove(), {
          chainCaptureInProgress: true,
          chainCapturePosition: { x: 1, y: 1 },
        });
        const machine = new PhaseStateMachine(processingState);

        machine.setChainCapture(false);

        expect(machine.processingState.chainCaptureInProgress).toBe(false);
        expect(machine.processingState.chainCapturePosition).toBeUndefined();
      });
    });

    describe('addEvent', () => {
      it('adds event to events array', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        machine.addEvent('phase_transition', { from: 'ring_placement', to: 'movement' });

        expect(machine.processingState.events.length).toBe(1);
        expect(machine.processingState.events[0].type).toBe('phase_transition');
        expect(machine.processingState.events[0].payload).toEqual({
          from: 'ring_placement',
          to: 'movement',
        });
        expect(machine.processingState.events[0].timestamp).toBeInstanceOf(Date);
      });
    });

    describe('transitionTo', () => {
      it('transitions to new phase', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        machine.transitionTo('line_processing');

        expect(machine.currentPhase).toBe('line_processing');
        expect(machine.processingState.phasesTraversed).toContain('line_processing');
      });
    });

    describe('isGameOver', () => {
      it('returns true when game status is not active', () => {
        const processingState = makeTurnProcessingState(
          makeGameState({ gameStatus: 'completed' }),
          makeMove()
        );
        const machine = new PhaseStateMachine(processingState);

        expect(machine.isGameOver()).toBe(true);
      });

      it('returns false when game status is active', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        expect(machine.isGameOver()).toBe(false);
      });
    });

    describe('pending lines management', () => {
      it('sets and gets pending lines', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const lines = [
          { positions: [{ x: 0, y: 0 }], player: 1 },
          { positions: [{ x: 1, y: 1 }], player: 2 },
        ];
        machine.setPendingLines(lines as TurnProcessingState['pendingLines']);

        expect(machine.pendingLineCount).toBe(2);
      });

      it('pops pending line', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const firstLine = { positions: [{ x: 0, y: 0 }], player: 1 };
        const lines = [firstLine, { positions: [{ x: 1, y: 1 }], player: 2 }];
        machine.setPendingLines(lines as TurnProcessingState['pendingLines']);

        const popped = machine.popPendingLine();

        // shift() removes from front of array (FIFO queue) and mutates original
        expect(popped).toEqual(firstLine);
        expect(machine.pendingLineCount).toBe(1);
      });

      it('returns undefined when no pending lines', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const popped = machine.popPendingLine();

        expect(popped).toBeUndefined();
      });
    });

    describe('pending regions management', () => {
      it('sets and gets pending regions', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const regions = [
          { spaces: [{ x: 0, y: 0 }], controllingPlayer: 1 },
          { spaces: [{ x: 1, y: 1 }], controllingPlayer: 2 },
        ];
        machine.setPendingRegions(regions as TurnProcessingState['pendingRegions']);

        expect(machine.pendingRegionCount).toBe(2);
      });

      it('pops pending region', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const firstRegion = { spaces: [{ x: 0, y: 0 }], controllingPlayer: 1 };
        const regions = [firstRegion, { spaces: [{ x: 1, y: 1 }], controllingPlayer: 2 }];
        machine.setPendingRegions(regions as TurnProcessingState['pendingRegions']);

        const popped = machine.popPendingRegion();

        // shift() removes from front of array (FIFO queue) and mutates original
        expect(popped).toEqual(firstRegion);
        expect(machine.pendingRegionCount).toBe(1);
      });

      it('returns undefined when no pending regions', () => {
        const processingState = makeTurnProcessingState(makeGameState(), makeMove());
        const machine = new PhaseStateMachine(processingState);

        const popped = machine.popPendingRegion();

        expect(popped).toBeUndefined();
      });
    });
  });

  describe('createTurnProcessingState', () => {
    it('creates initial processing state', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const move = makeMove();

      const result = createTurnProcessingState(gameState, move);

      expect(result).toMatchObject({
        gameState: gameState,
        originalMove: move,
        perTurnFlags: {
          hasPlacedThisTurn: false,
          mustMoveFromStackKey: undefined,
          eliminationRewardPending: false,
          eliminationRewardCount: 0,
        },
        pendingLines: [],
        pendingRegions: [],
        chainCaptureInProgress: false,
        chainCapturePosition: undefined,
        events: [],
        phasesTraversed: ['ring_placement'],
        startTime: expect.any(Number),
      });
    });
  });
});
