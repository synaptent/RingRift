/**
 * Unit tests for FSMAdapter orchestration integration
 *
 * Tests the orchestration-level functions added to FSMAdapter:
 * - determineNextPhaseFromFSM
 * - attemptFSMTransition
 * - getCurrentFSMState
 * - isFSMTerminalState
 */

import {
  determineNextPhaseFromFSM,
  attemptFSMTransition,
  getCurrentFSMState,
  isFSMTerminalState,
  type PhaseTransitionContext,
} from '../../src/shared/engine/fsm/FSMAdapter';
import type { GameState, Move, BoardState, Player, RingStack } from '../../src/shared/types/game';
import { inferTotalRingsInPlay } from '../utils/fixtures';

describe('FSMAdapter orchestration integration', () => {
  // Helper to create a minimal game state
  function makeGameState(overrides: Partial<GameState> = {}): GameState {
    const board: BoardState = {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    } as unknown as BoardState;

    const players: Player[] = [
      {
        id: 'p1',
        username: 'P1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as unknown as Player,
      {
        id: 'p2',
        username: 'P2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as unknown as Player,
    ];
    const totalRingsInPlay = inferTotalRingsInPlay(players, board);

    const base: GameState = {
      id: 'fsm-test',
      gameId: 'fsm-test',
      boardType: 'square8',
      board,
      players,
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay,
      totalRingsEliminated: 0,
      victoryThreshold: 10,
      territoryVictoryThreshold: 33,
    } as unknown as GameState;

    return { ...base, ...overrides };
  }

  describe('getCurrentFSMState', () => {
    it('derives FSM state from game state in ring_placement phase', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const fsmState = getCurrentFSMState(gameState);

      expect(fsmState.phase).toBe('ring_placement');
      // FSM uses 'player' not 'currentPlayer'
      expect((fsmState as any).player).toBe(1);
    });

    it('derives FSM state from game state in movement phase', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const fsmState = getCurrentFSMState(gameState);

      expect(fsmState.phase).toBe('movement');
    });

    it('derives FSM state from game state in chain_capture phase', () => {
      const gameState = makeGameState({ currentPhase: 'chain_capture' });
      const fsmState = getCurrentFSMState(gameState);

      expect(fsmState.phase).toBe('chain_capture');
    });

    it('derives FSM state from game state in line_processing phase', () => {
      const gameState = makeGameState({ currentPhase: 'line_processing' });
      const fsmState = getCurrentFSMState(gameState);

      expect(fsmState.phase).toBe('line_processing');
    });

    it('derives FSM state from game state in territory_processing phase', () => {
      const gameState = makeGameState({ currentPhase: 'territory_processing' });
      const fsmState = getCurrentFSMState(gameState);

      expect(fsmState.phase).toBe('territory_processing');
    });
  });

  describe('isFSMTerminalState', () => {
    it('returns false for active game', () => {
      const gameState = makeGameState({ gameStatus: 'active' });
      expect(isFSMTerminalState(gameState)).toBe(false);
    });

    it('returns true for completed game', () => {
      const gameState = makeGameState({
        gameStatus: 'completed',
        currentPhase: 'game_over' as any,
      });
      expect(isFSMTerminalState(gameState)).toBe(true);
    });
  });

  describe('determineNextPhaseFromFSM', () => {
    it('transitions from ring_placement to movement when moves available', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: true,
        hasAnyCapture: false,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'place_ring', context);
      expect(nextPhase).toBe('movement');
    });

    it('transitions from ring_placement to movement when captures available', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: false,
        hasAnyCapture: true,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'place_ring', context);
      expect(nextPhase).toBe('movement');
    });

    it('transitions from ring_placement to line_processing when no moves/captures', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: false,
        hasAnyCapture: false,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'skip_placement', context);
      expect(nextPhase).toBe('line_processing');
    });

    it('transitions from movement to chain_capture when chains available', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: true,
        hasAnyMovement: true,
        hasAnyCapture: true,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'overtaking_capture', context);
      expect(nextPhase).toBe('chain_capture');
    });

    it('transitions from movement to line_processing when no chains', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: true,
        hasAnyCapture: false,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'move_stack', context);
      expect(nextPhase).toBe('line_processing');
    });

    it('stays in chain_capture when more chains available', () => {
      const gameState = makeGameState({ currentPhase: 'chain_capture' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: true,
        hasAnyMovement: false,
        hasAnyCapture: true,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'continue_capture_segment', context);
      expect(nextPhase).toBe('chain_capture');
    });

    it('transitions from chain_capture to line_processing when no more chains', () => {
      const gameState = makeGameState({ currentPhase: 'chain_capture' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: false,
        hasAnyCapture: false,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'skip_capture', context);
      expect(nextPhase).toBe('line_processing');
    });

    it('stays in line_processing when more lines to process', () => {
      const gameState = makeGameState({ currentPhase: 'line_processing' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: true,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: false,
        hasAnyCapture: false,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'process_line', context);
      expect(nextPhase).toBe('line_processing');
    });

    it('transitions from line_processing to territory_processing', () => {
      const gameState = makeGameState({ currentPhase: 'line_processing' });
      const context: PhaseTransitionContext = {
        hasMoreLinesToProcess: false,
        hasMoreRegionsToProcess: false,
        chainCapturesAvailable: false,
        hasAnyMovement: false,
        hasAnyCapture: false,
      };

      const nextPhase = determineNextPhaseFromFSM(gameState, 'no_line_action', context);
      expect(nextPhase).toBe('territory_processing');
    });
  });

  describe('attemptFSMTransition', () => {
    it('converts place_ring event type correctly in ring_placement phase', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const move: Move = {
        type: 'place_ring',
        player: 1,
        to: { row: 3, col: 3 },
        timestamp: new Date().toISOString(),
      };

      const result = attemptFSMTransition(gameState, move);

      // The event type is accepted for the phase - conversion succeeds
      // Full validation may fail due to guards (e.g., valid position check)
      expect(result.error?.code).not.toBe('CONVERSION_FAILED');
    });

    it('returns valid:false for movement move in ring_placement phase', () => {
      const gameState = makeGameState({ currentPhase: 'ring_placement' });
      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: { row: 3, col: 3 },
        to: { row: 5, col: 5 },
        timestamp: new Date().toISOString(),
      };

      const result = attemptFSMTransition(gameState, move);

      // FSM rejects movement events during ring_placement phase
      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('converts move_stack event type correctly', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const move: Move = {
        type: 'move_stack',
        player: 1,
        from: { row: 3, col: 3 }, // from is required for move_stack
        to: { row: 5, col: 5 },
        timestamp: new Date().toISOString(),
      };

      const result = attemptFSMTransition(gameState, move);

      // FSM accepts movement events during movement phase
      // Even if guards fail, it shouldn't be a conversion failure
      expect(result.error?.code).not.toBe('CONVERSION_FAILED');
    });

    it('converts overtaking_capture event type correctly', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: { row: 3, col: 3 },
        to: { row: 5, col: 5 },
        captureTarget: { row: 4, col: 4 }, // captureTarget is required for FSM conversion
        timestamp: new Date().toISOString(),
      };

      const result = attemptFSMTransition(gameState, move);

      // FSM accepts capture events during movement phase
      expect(result.error?.code).not.toBe('CONVERSION_FAILED');
    });

    it('returns conversion error when captureTarget is missing', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const move: Move = {
        type: 'overtaking_capture',
        player: 1,
        from: { row: 3, col: 3 },
        to: { row: 5, col: 5 },
        // Missing captureTarget - should fail conversion
        timestamp: new Date().toISOString(),
      };

      const result = attemptFSMTransition(gameState, move);

      expect(result.valid).toBe(false);
      expect(result.error?.code).toBe('CONVERSION_FAILED');
    });

    it('returns error for unconvertible move type', () => {
      const gameState = makeGameState({ currentPhase: 'movement' });
      const move: Move = {
        type: 'swap_sides' as any, // Meta-move not handled by FSM
        player: 1,
        timestamp: new Date().toISOString(),
      };

      const result = attemptFSMTransition(gameState, move);

      expect(result.valid).toBe(false);
      expect(result.error?.code).toBe('CONVERSION_FAILED');
    });
  });
});
