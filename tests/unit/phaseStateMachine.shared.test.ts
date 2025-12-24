/**
 * phaseStateMachine.shared.test.ts
 *
 * Comprehensive tests for phase state machine functions.
 * Covers: determineNextPhase, createTurnLogicDelegates, toPerTurnState,
 * updateFlagsFromPerTurnState, phaseRequiresDecision, shouldAutoAdvancePhase,
 * PhaseStateMachine class, createTurnProcessingState
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
  type PhaseContext,
} from '../../src/shared/engine/orchestration/phaseStateMachine';
import type {
  TurnProcessingState,
  PerTurnFlags,
} from '../../src/shared/engine/orchestration/types';
import { createTestGameState } from '../utils/fixtures';
import type { GamePhase, Move } from '../../src/shared/types/game';

describe('phaseStateMachine', () => {
  describe('determineNextPhase', () => {
    const baseContext = {
      hasMoreLinesToProcess: false,
      hasMoreRegionsToProcess: false,
      chainCapturesAvailable: false,
      hasAnyMovement: false,
      hasAnyCapture: false,
    };

    describe('from ring_placement', () => {
      it('transitions to movement when movement available', () => {
        const next = determineNextPhase('ring_placement', 'place_ring', {
          ...baseContext,
          hasAnyMovement: true,
        });
        expect(next).toBe('movement');
      });

      it('transitions to movement when capture available', () => {
        const next = determineNextPhase('ring_placement', 'place_ring', {
          ...baseContext,
          hasAnyCapture: true,
        });
        expect(next).toBe('movement');
      });

      it('transitions to line_processing when no movement/capture', () => {
        const next = determineNextPhase('ring_placement', 'place_ring', baseContext);
        expect(next).toBe('line_processing');
      });
    });

    describe('from movement', () => {
      it('transitions to chain_capture when chain captures available', () => {
        const next = determineNextPhase('movement', 'move_stack', {
          ...baseContext,
          chainCapturesAvailable: true,
        });
        expect(next).toBe('chain_capture');
      });

      it('transitions to line_processing when no chain captures', () => {
        const next = determineNextPhase('movement', 'move_stack', baseContext);
        expect(next).toBe('line_processing');
      });
    });

    describe('from capture', () => {
      it('transitions to chain_capture when chain captures available', () => {
        const next = determineNextPhase('capture', 'overtaking_capture', {
          ...baseContext,
          chainCapturesAvailable: true,
        });
        expect(next).toBe('chain_capture');
      });

      it('transitions to line_processing when no chain captures', () => {
        const next = determineNextPhase('capture', 'overtaking_capture', baseContext);
        expect(next).toBe('line_processing');
      });
    });

    describe('from chain_capture', () => {
      it('stays in chain_capture when more chain captures available', () => {
        const next = determineNextPhase('chain_capture', 'chain_capture', {
          ...baseContext,
          chainCapturesAvailable: true,
        });
        expect(next).toBe('chain_capture');
      });

      it('transitions to line_processing when no more chain captures', () => {
        const next = determineNextPhase('chain_capture', 'chain_capture', baseContext);
        expect(next).toBe('line_processing');
      });
    });

    describe('from line_processing', () => {
      it('stays in line_processing when more lines to process', () => {
        const next = determineNextPhase('line_processing', 'process_line', {
          ...baseContext,
          hasMoreLinesToProcess: true,
        });
        expect(next).toBe('line_processing');
      });

      it('transitions to territory_processing when no more lines', () => {
        const next = determineNextPhase('line_processing', 'process_line', baseContext);
        expect(next).toBe('territory_processing');
      });
    });

    describe('from territory_processing', () => {
      it('stays in territory_processing (handled by turn advance)', () => {
        const next = determineNextPhase('territory_processing', 'process_territory', baseContext);
        expect(next).toBe('territory_processing');
      });
    });

    it('returns current phase for unknown phases', () => {
      const next = determineNextPhase('unknown_phase' as GamePhase, 'move_stack', baseContext);
      expect(next).toBe('unknown_phase');
    });
  });

  describe('toPerTurnState', () => {
    it('converts PerTurnFlags to PerTurnState', () => {
      const flags: PerTurnFlags = {
        hasPlacedThisTurn: true,
        mustMoveFromStackKey: '1,2',
        eliminationRewardPending: false,
        eliminationRewardCount: 0,
      };

      const state = toPerTurnState(flags);

      expect(state.hasPlacedThisTurn).toBe(true);
      expect(state.mustMoveFromStackKey).toBe('1,2');
    });

    it('handles undefined mustMoveFromStackKey', () => {
      const flags: PerTurnFlags = {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
        eliminationRewardPending: false,
        eliminationRewardCount: 0,
      };

      const state = toPerTurnState(flags);

      expect(state.hasPlacedThisTurn).toBe(false);
      expect(state.mustMoveFromStackKey).toBeUndefined();
    });
  });

  describe('updateFlagsFromPerTurnState', () => {
    it('updates flags from PerTurnState', () => {
      const flags: PerTurnFlags = {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
        eliminationRewardPending: true,
        eliminationRewardCount: 2,
      };

      const newState = {
        hasPlacedThisTurn: true,
        mustMoveFromStackKey: '3,4',
      };

      const updated = updateFlagsFromPerTurnState(flags, newState);

      expect(updated.hasPlacedThisTurn).toBe(true);
      expect(updated.mustMoveFromStackKey).toBe('3,4');
      // Should preserve other flags
      expect(updated.eliminationRewardPending).toBe(true);
      expect(updated.eliminationRewardCount).toBe(2);
    });
  });

  describe('phaseRequiresDecision', () => {
    it('returns true for line_processing with multiple lines', () => {
      const context: PhaseContext = {
        pendingLineCount: 3,
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
        pendingRegionCount: 2,
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
        chainCaptureOptionsCount: 3,
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

    it('returns false for movement phase', () => {
      const context: PhaseContext = {
        pendingLineCount: 5,
        pendingRegionCount: 5,
        chainCaptureOptionsCount: 5,
      };
      expect(phaseRequiresDecision('movement', context)).toBe(false);
    });
  });

  describe('shouldAutoAdvancePhase', () => {
    it('returns true for line_processing with 0 lines', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 0,
      };
      expect(shouldAutoAdvancePhase('line_processing', context)).toBe(true);
    });

    it('returns true for line_processing with 1 line', () => {
      const context: PhaseContext = {
        pendingLineCount: 1,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 0,
      };
      expect(shouldAutoAdvancePhase('line_processing', context)).toBe(true);
    });

    it('returns false for line_processing with multiple lines', () => {
      const context: PhaseContext = {
        pendingLineCount: 2,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 0,
      };
      expect(shouldAutoAdvancePhase('line_processing', context)).toBe(false);
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

    it('returns true for chain_capture with exactly 1 option', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 1,
      };
      expect(shouldAutoAdvancePhase('chain_capture', context)).toBe(true);
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

    it('returns false for movement phase', () => {
      const context: PhaseContext = {
        pendingLineCount: 0,
        pendingRegionCount: 0,
        chainCaptureOptionsCount: 0,
      };
      expect(shouldAutoAdvancePhase('movement', context)).toBe(false);
    });
  });

  describe('createTurnProcessingState', () => {
    it('creates initial processing state with correct defaults', () => {
      const gameState = createTestGameState();
      const move: Move = { type: 'place_ring', player: 1, position: { x: 0, y: 0 } };

      const state = createTurnProcessingState(gameState, move);

      expect(state).toMatchObject({
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
        startTime: expect.any(Number),
      });
      expect(state.phasesTraversed).toContain(gameState.currentPhase);
    });
  });

  describe('createTurnLogicDelegates', () => {
    it('creates delegates with correct structure', () => {
      const gameState = createTestGameState();
      const move: Move = { type: 'place_ring', player: 1, position: { x: 0, y: 0 } };
      const processingState = createTurnProcessingState(gameState, move);

      const callbacks = {
        getPlayerStacks: jest.fn().mockReturnValue([]),
        hasAnyPlacement: jest.fn().mockReturnValue(false),
        hasAnyMovement: jest.fn().mockReturnValue(false),
        hasAnyCapture: jest.fn().mockReturnValue(false),
        applyForcedElimination: jest.fn().mockReturnValue(gameState),
      };

      const delegates = createTurnLogicDelegates(processingState, callbacks);

      expect(delegates.getPlayerStacks).toBe(callbacks.getPlayerStacks);
      expect(delegates.hasAnyPlacement).toBe(callbacks.hasAnyPlacement);
      expect(delegates.hasAnyMovement).toBe(callbacks.hasAnyMovement);
      expect(delegates.hasAnyCapture).toBe(callbacks.hasAnyCapture);
      expect(delegates.applyForcedElimination).toBe(callbacks.applyForcedElimination);
      expect(typeof delegates.getNextPlayerNumber).toBe('function');
    });

    it('getNextPlayerNumber returns next player in order', () => {
      const gameState = createTestGameState();
      const move: Move = { type: 'place_ring', player: 1, position: { x: 0, y: 0 } };
      const processingState = createTurnProcessingState(gameState, move);

      const callbacks = {
        getPlayerStacks: jest.fn(),
        hasAnyPlacement: jest.fn(),
        hasAnyMovement: jest.fn(),
        hasAnyCapture: jest.fn(),
        applyForcedElimination: jest.fn(),
      };

      const delegates = createTurnLogicDelegates(processingState, callbacks);

      // Player 1 -> Player 2
      expect(delegates.getNextPlayerNumber(gameState, 1)).toBe(2);
      // Player 2 -> Player 1 (wrap around)
      expect(delegates.getNextPlayerNumber(gameState, 2)).toBe(1);
    });

    it('getNextPlayerNumber returns current when no players', () => {
      const gameState = createTestGameState();
      gameState.players = [];
      const move: Move = { type: 'place_ring', player: 1, position: { x: 0, y: 0 } };
      const processingState = createTurnProcessingState(gameState, move);

      const callbacks = {
        getPlayerStacks: jest.fn(),
        hasAnyPlacement: jest.fn(),
        hasAnyMovement: jest.fn(),
        hasAnyCapture: jest.fn(),
        applyForcedElimination: jest.fn(),
      };

      const delegates = createTurnLogicDelegates(processingState, callbacks);

      expect(delegates.getNextPlayerNumber(gameState, 1)).toBe(1);
    });

    it('getNextPlayerNumber returns first player when current not found', () => {
      const gameState = createTestGameState();
      const move: Move = { type: 'place_ring', player: 1, position: { x: 0, y: 0 } };
      const processingState = createTurnProcessingState(gameState, move);

      const callbacks = {
        getPlayerStacks: jest.fn(),
        hasAnyPlacement: jest.fn(),
        hasAnyMovement: jest.fn(),
        hasAnyCapture: jest.fn(),
        applyForcedElimination: jest.fn(),
      };

      const delegates = createTurnLogicDelegates(processingState, callbacks);

      expect(delegates.getNextPlayerNumber(gameState, 999)).toBe(1);
    });
  });

  describe('PhaseStateMachine', () => {
    let initialState: TurnProcessingState;
    let machine: PhaseStateMachine;

    beforeEach(() => {
      const gameState = createTestGameState();
      const move: Move = { type: 'place_ring', player: 1, position: { x: 0, y: 0 } };
      initialState = createTurnProcessingState(gameState, move);
      machine = new PhaseStateMachine(initialState);
    });

    it('returns current phase from game state', () => {
      expect(machine.currentPhase).toBe(initialState.gameState.currentPhase);
    });

    it('returns game state', () => {
      expect(machine.gameState).toBe(initialState.gameState);
    });

    it('returns processing state', () => {
      expect(machine.processingState).toBe(initialState);
    });

    it('updates game state and tracks phases', () => {
      const newGameState = { ...initialState.gameState, currentPhase: 'movement' as GamePhase };
      machine.updateGameState(newGameState);

      expect(machine.gameState).toBe(newGameState);
      expect(machine.processingState.phasesTraversed).toContain('movement');
    });

    it('updates flags', () => {
      machine.updateFlags({ hasPlacedThisTurn: true });

      expect(machine.processingState.perTurnFlags.hasPlacedThisTurn).toBe(true);
    });

    it('sets chain capture state', () => {
      machine.setChainCapture(true, { x: 1, y: 2 });

      expect(machine.processingState.chainCaptureInProgress).toBe(true);
      expect(machine.processingState.chainCapturePosition).toEqual({ x: 1, y: 2 });
    });

    it('adds events', () => {
      machine.addEvent('move_applied', { moveType: 'place_ring' });

      expect(machine.processingState.events.length).toBe(1);
      expect(machine.processingState.events[0].type).toBe('move_applied');
      expect(machine.processingState.events[0].payload).toEqual({ moveType: 'place_ring' });
    });

    it('transitions to specific phase', () => {
      machine.transitionTo('chain_capture');

      expect(machine.currentPhase).toBe('chain_capture');
      expect(machine.processingState.phasesTraversed).toContain('chain_capture');
    });

    it('checks if game is over', () => {
      expect(machine.isGameOver()).toBe(false);

      machine.updateGameState({ ...machine.gameState, gameStatus: 'completed' });
      expect(machine.isGameOver()).toBe(true);
    });

    it('manages pending lines', () => {
      const lines = [{ player: 1, positions: [{ x: 0, y: 0 }] }];
      machine.setPendingLines(lines as TurnProcessingState['pendingLines']);

      expect(machine.pendingLineCount).toBe(1);

      const popped = machine.popPendingLine();
      expect(popped).toBeDefined();
      expect(popped?.player).toBe(1);
      expect(machine.pendingLineCount).toBe(0);
    });

    it('manages pending regions', () => {
      const regions = [{ player: 1, positions: [{ x: 0, y: 0 }] }];
      machine.setPendingRegions(regions as TurnProcessingState['pendingRegions']);

      expect(machine.pendingRegionCount).toBe(1);

      const popped = machine.popPendingRegion();
      expect(popped).toBeDefined();
      expect(popped?.player).toBe(1);
      expect(machine.pendingRegionCount).toBe(0);
    });

    it('popPendingLine returns undefined when empty', () => {
      expect(machine.popPendingLine()).toBeUndefined();
    });

    it('popPendingRegion returns undefined when empty', () => {
      expect(machine.popPendingRegion()).toBeUndefined();
    });
  });
});
