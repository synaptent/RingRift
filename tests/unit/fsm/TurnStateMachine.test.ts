/**
 * TurnStateMachine unit tests
 */

import {
  TurnStateMachine,
  transition,
  type TurnEvent,
  type GameContext,
  type RingPlacementState,
  type MovementState,
  type CaptureState,
  type ChainCaptureState,
  type LineProcessingState,
  type TerritoryProcessingState,
  type ForcedEliminationState,
  type TurnEndState,
  type GameOverState,
} from '../../../src/shared/engine/fsm';

describe('TurnStateMachine', () => {
  const context: GameContext = {
    boardType: 'square8',
    numPlayers: 2,
    ringsPerPlayer: 18,
    lineLength: 3,
  };

  describe('transition function', () => {
    it('should transition from ring_placement to movement on PLACE_RING', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'PLACE_RING', to: { x: 3, y: 3 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('movement');
        expect(result.actions).toContainEqual({
          type: 'PLACE_RING',
          position: { x: 3, y: 3 },
          player: 1,
        });
        expect(result.actions).toContainEqual({
          type: 'LEAVE_MARKER',
          position: { x: 3, y: 3 },
          player: 1,
        });
      }
    });

    it('should reject PLACE_RING when canPlace is false', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 0,
        canPlace: false,
        validPositions: [],
      };

      const event: TurnEvent = { type: 'PLACE_RING', to: { x: 3, y: 3 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should reject invalid event for phase', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      // MOVE_STACK is not valid in ring_placement phase
      const event: TurnEvent = { type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('INVALID_EVENT');
        expect(result.error.currentPhase).toBe('ring_placement');
        expect(result.error.eventType).toBe('MOVE_STACK');
      }
    });

    it('should reject SKIP_PLACEMENT when ringsInHand is 0', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 0,
        canPlace: false,
        validPositions: [],
      };

      const event: TurnEvent = { type: 'SKIP_PLACEMENT' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toMatch(/no rings in hand/i);
      }
    });

    it('should reject NO_PLACEMENT_ACTION when valid placements exist', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'NO_PLACEMENT_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should allow NO_PLACEMENT_ACTION when no placements are available', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 0,
        canPlace: false,
        validPositions: [],
      };

      const event: TurnEvent = { type: 'NO_PLACEMENT_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('movement');
      }
    });

    it('should transition from movement to line_processing on MOVE_STACK', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        recoveryEligible: false,
        recoveryMovesAvailable: false,
        placedRingAt: { x: 3, y: 3 },
      };

      const event: TurnEvent = { type: 'MOVE_STACK', from: { x: 3, y: 3 }, to: { x: 4, y: 4 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
        expect(result.actions).toContainEqual({
          type: 'MOVE_STACK',
          from: { x: 3, y: 3 },
          to: { x: 4, y: 4 },
        });
      }
    });

    it('should reject NO_MOVEMENT_ACTION when movement is possible', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        recoveryEligible: false,
        recoveryMovesAvailable: false,
        placedRingAt: { x: 3, y: 3 },
      };

      const event: TurnEvent = { type: 'NO_MOVEMENT_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should allow NO_MOVEMENT_ACTION when no movement is possible', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: false,
        recoveryEligible: false,
        recoveryMovesAvailable: false,
        placedRingAt: { x: 3, y: 3 },
      };

      const event: TurnEvent = { type: 'NO_MOVEMENT_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
      }
    });

    it('should transition to game_over on RESIGN', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'RESIGN', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('resignation');
        }
      }
    });
  });

  describe('TurnStateMachine class', () => {
    it('should track state through transitions', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      expect(fsm.phase).toBe('ring_placement');
      expect(fsm.currentPlayer).toBe(1);

      const actions = fsm.send({ type: 'PLACE_RING', to: { x: 3, y: 3 } });
      expect(fsm.phase).toBe('movement');
      expect(actions.length).toBeGreaterThan(0);
    });

    it('should throw on invalid transitions', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      // MOVE_STACK is not valid in ring_placement phase
      expect(() => {
        fsm.send({ type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } });
      }).toThrow('[FSM]');
    });

    it('should track history', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      fsm.send({ type: 'PLACE_RING', to: { x: 3, y: 3 } });

      const history = fsm.getHistory();
      expect(history.length).toBe(1);
      expect(history[0].event.type).toBe('PLACE_RING');
    });

    it('canSend should return true for valid events', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      expect(fsm.canSend({ type: 'PLACE_RING', to: { x: 3, y: 3 } })).toBe(true);
      expect(fsm.canSend({ type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } })).toBe(
        false
      );
    });
  });

  describe('chain capture transitions', () => {
    it('should allow CONTINUE_CHAIN when valid target exists', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [
          { target: { x: 5, y: 5 }, capturingPlayer: 1, isChainCapture: true },
        ],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'CONTINUE_CHAIN', target: { x: 5, y: 5 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('chain_capture');
        expect(result.actions).toContainEqual({
          type: 'EXECUTE_CAPTURE',
          target: { x: 5, y: 5 },
          capturer: 1,
        });
        if (result.state.phase === 'chain_capture') {
          expect(result.state.capturedTargets).toContainEqual({ x: 5, y: 5 });
          expect(result.state.segmentCount).toBe(2);
        }
      }
    });

    it('should reject CONTINUE_CHAIN with invalid target', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [
          { target: { x: 5, y: 5 }, capturingPlayer: 1, isChainCapture: true },
        ],
        segmentCount: 1,
        isFirstSegment: false,
      };

      // Invalid target - not in availableContinuations
      const event: TurnEvent = { type: 'CONTINUE_CHAIN', target: { x: 6, y: 6 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should allow END_CHAIN when no mandatory captures remain', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [], // No more targets
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'END_CHAIN' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
      }
    });

    it('should reject END_CHAIN when mandatory captures remain', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [
          { target: { x: 5, y: 5 }, capturingPlayer: 1, isChainCapture: true },
        ],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'END_CHAIN' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should transition to game_over on RESIGN from chain_capture', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'RESIGN', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('resignation');
        }
      }
    });
  });

  describe('line_processing guards', () => {
    it('should reject CHOOSE_LINE_REWARD when awaitingReward is false and line does not require choice', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
              { x: 2, y: 2 },
            ],
            player: 1,
            requiresChoice: false, // Line does NOT require choice (exact length)
          },
        ],
        currentLineIndex: 0,
        awaitingReward: false, // Not awaiting reward
      };

      const event: TurnEvent = { type: 'CHOOSE_LINE_REWARD', choice: 'eliminate' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Not awaiting line reward choice');
      }
    });

    it('should allow CHOOSE_LINE_REWARD when awaitingReward is false but line requires choice (legacy replay)', () => {
      // This test verifies legacy replay support: when a line requires a choice,
      // CHOOSE_LINE_REWARD should be allowed even without a prior PROCESS_LINE event.
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
              { x: 2, y: 2 },
              { x: 3, y: 3 },
            ],
            player: 1,
            requiresChoice: true, // Line requires choice (overlength)
          },
        ],
        currentLineIndex: 0,
        awaitingReward: false, // Not awaiting reward (legacy replay scenario)
      };

      const event: TurnEvent = { type: 'CHOOSE_LINE_REWARD', choice: 'eliminate' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
    });

    it('should allow CHOOSE_LINE_REWARD when awaitingReward is true', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
              { x: 2, y: 2 },
            ],
            player: 1,
            requiresChoice: true,
          },
        ],
        currentLineIndex: 0,
        awaitingReward: true, // Awaiting reward
      };

      const event: TurnEvent = { type: 'CHOOSE_LINE_REWARD', choice: 'eliminate' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
    });

    it('should reject PROCESS_LINE with invalid lineIndex', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
              { x: 2, y: 2 },
            ],
            player: 1,
            requiresChoice: false,
          },
        ],
        currentLineIndex: 0,
        awaitingReward: false,
      };

      // Invalid index - only one line exists (index 0)
      const event: TurnEvent = { type: 'PROCESS_LINE', lineIndex: 5 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Invalid line index');
      }
    });

    it('should reject NO_LINE_ACTION when lines exist', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
              { x: 2, y: 2 },
            ],
            player: 1,
            requiresChoice: false,
          },
        ],
        currentLineIndex: 0,
        awaitingReward: false,
      };

      const event: TurnEvent = { type: 'NO_LINE_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Cannot skip line processing when lines exist');
      }
    });

    it('should allow NO_LINE_ACTION when no lines exist', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [],
        currentLineIndex: 0,
        awaitingReward: false,
      };

      const event: TurnEvent = { type: 'NO_LINE_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
      }
    });
  });

  describe('territory_processing guards', () => {
    it('should reject PROCESS_REGION with invalid regionIndex', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          { positions: [{ x: 0, y: 0 }], controllingPlayer: 1, eliminationsRequired: 1 },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      // Invalid index - only one region exists (index 0)
      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 5 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should allow NO_TERRITORY_ACTION when no regions require processing', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'NO_TERRITORY_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
    });

    it('should reject NO_TERRITORY_ACTION when regions exist', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          { positions: [{ x: 0, y: 0 }], controllingPlayer: 1, eliminationsRequired: 1 },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'NO_TERRITORY_ACTION' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should reject ELIMINATE_FROM_STACK when no eliminations pending', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [], // No pending eliminations
      };

      const event: TurnEvent = { type: 'ELIMINATE_FROM_STACK', target: { x: 1, y: 1 }, count: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('No eliminations pending');
      }
    });

    it('should allow ELIMINATE_FROM_STACK when eliminations are pending', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [{ position: { x: 1, y: 1 }, player: 1, count: 1 }],
      };

      const event: TurnEvent = { type: 'ELIMINATE_FROM_STACK', target: { x: 1, y: 1 }, count: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.actions).toContainEqual({
          type: 'ELIMINATE_RINGS',
          target: { x: 1, y: 1 },
          count: 1,
        });
      }
    });
  });

  describe('capture phase guards', () => {
    it('should reject CAPTURE with invalid target', () => {
      const state: CaptureState = {
        phase: 'capture',
        player: 1,
        pendingCaptures: [{ target: { x: 3, y: 3 }, capturingPlayer: 1, isChainCapture: false }],
        chainInProgress: false,
        capturesMade: 0,
      };

      // Invalid target - not in pendingCaptures
      const event: TurnEvent = { type: 'CAPTURE', target: { x: 5, y: 5 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Invalid capture target');
      }
    });

    it('should allow CAPTURE with valid target', () => {
      const state: CaptureState = {
        phase: 'capture',
        player: 1,
        pendingCaptures: [{ target: { x: 3, y: 3 }, capturingPlayer: 1, isChainCapture: false }],
        chainInProgress: false,
        capturesMade: 0,
      };

      const event: TurnEvent = { type: 'CAPTURE', target: { x: 3, y: 3 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('chain_capture');
        expect(result.actions).toContainEqual({
          type: 'EXECUTE_CAPTURE',
          target: { x: 3, y: 3 },
          capturer: 1,
        });
      }
    });

    it('should allow END_CHAIN in capture phase (skip capture)', () => {
      const state: CaptureState = {
        phase: 'capture',
        player: 1,
        pendingCaptures: [],
        chainInProgress: false,
        capturesMade: 0,
      };

      const event: TurnEvent = { type: 'END_CHAIN' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
      }
    });

    it('should transition to game_over on RESIGN from capture phase', () => {
      const state: CaptureState = {
        phase: 'capture',
        player: 1,
        pendingCaptures: [],
        chainInProgress: false,
        capturesMade: 0,
      };

      const event: TurnEvent = { type: 'RESIGN', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
      }
    });

    it('should transition to game_over on TIMEOUT from capture phase', () => {
      const state: CaptureState = {
        phase: 'capture',
        player: 1,
        pendingCaptures: [],
        chainInProgress: false,
        capturesMade: 0,
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });
  });

  describe('forced_elimination phase guards', () => {
    it('should allow FORCED_ELIMINATE and transition to turn_end when complete', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 1,
        eliminationsDone: 0,
      };

      const event: TurnEvent = { type: 'FORCED_ELIMINATE', target: { x: 2, y: 2 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        expect(result.actions).toContainEqual({
          type: 'FORCED_ELIMINATE',
          target: { x: 2, y: 2 },
          player: 1,
        });
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });

    it('should stay in forced_elimination when more eliminations needed', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 3,
        eliminationsDone: 0,
      };

      const event: TurnEvent = { type: 'FORCED_ELIMINATE', target: { x: 2, y: 2 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('forced_elimination');
        if (result.state.phase === 'forced_elimination') {
          expect(result.state.eliminationsDone).toBe(1);
        }
      }
    });

    it('should transition to game_over on RESIGN from forced_elimination', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 2,
        eliminationsDone: 0,
      };

      const event: TurnEvent = { type: 'RESIGN', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
      }
    });

    it('should transition to game_over on TIMEOUT from forced_elimination', () => {
      const state: ForcedEliminationState = {
        phase: 'forced_elimination',
        player: 1,
        ringsOverLimit: 2,
        eliminationsDone: 0,
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });
  });

  describe('turn_end phase', () => {
    it('should transition to ring_placement on _ADVANCE_TURN', () => {
      const state: TurnEndState = {
        phase: 'turn_end',
        completedPlayer: 1,
        nextPlayer: 2,
      };

      const event: TurnEvent = { type: '_ADVANCE_TURN' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('ring_placement');
        if (result.state.phase === 'ring_placement') {
          expect(result.state.player).toBe(2);
        }
        expect(result.actions).toContainEqual({
          type: 'ADVANCE_PLAYER',
          from: 1,
          to: 2,
        });
      }
    });

    it('should reject non-_ADVANCE_TURN events in turn_end', () => {
      const state: TurnEndState = {
        phase: 'turn_end',
        completedPlayer: 1,
        nextPlayer: 2,
      };

      const event: TurnEvent = { type: 'PLACE_RING', to: { x: 0, y: 0 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('INVALID_EVENT');
      }
    });
  });

  describe('movement phase CAPTURE and RECOVERY_SLIDE', () => {
    it('should allow CAPTURE in movement phase', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        recoveryEligible: false,
        recoveryMovesAvailable: false,
        placedRingAt: { x: 3, y: 3 },
      };

      const event: TurnEvent = { type: 'CAPTURE', target: { x: 4, y: 4 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
        expect(result.actions).toContainEqual({
          type: 'EXECUTE_CAPTURE',
          target: { x: 4, y: 4 },
          capturer: 1,
        });
      }
    });

    it('should allow RECOVERY_SLIDE in movement phase', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        recoveryEligible: true,
        recoveryMovesAvailable: true,
        placedRingAt: null,
      };

      const event: TurnEvent = {
        type: 'RECOVERY_SLIDE',
        from: { x: 2, y: 2 },
        to: { x: 2, y: 4 },
        option: 1,
      };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
        expect(result.actions).toContainEqual({
          type: 'MOVE_STACK',
          from: { x: 2, y: 2 },
          to: { x: 2, y: 4 },
        });
      }
    });

    it('should allow SKIP_RECOVERY for recovery-eligible player', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        recoveryEligible: true,
        recoveryMovesAvailable: false,
        placedRingAt: null,
      };

      const event: TurnEvent = { type: 'SKIP_RECOVERY' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
      }
    });

    it('should transition to game_over on TIMEOUT from movement phase', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        recoveryEligible: false,
        recoveryMovesAvailable: false,
        placedRingAt: { x: 3, y: 3 },
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });
  });

  describe('timeout events in various phases', () => {
    it('should transition to game_over on TIMEOUT from ring_placement', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });

    it('should transition to game_over on TIMEOUT from chain_capture', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });

    it('should transition to game_over on TIMEOUT from line_processing', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [],
        currentLineIndex: 0,
        awaitingReward: false,
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });

    it('should transition to game_over on TIMEOUT from territory_processing', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'TIMEOUT', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('timeout');
        }
      }
    });
  });

  describe('game_over phase', () => {
    it('should reject all events in game_over phase', () => {
      const state: GameOverState = {
        phase: 'game_over',
        winner: 2,
        reason: 'ring_elimination',
      };

      const events: TurnEvent[] = [
        { type: 'PLACE_RING', to: { x: 0, y: 0 } },
        { type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } },
        { type: 'RESIGN', player: 1 },
        { type: 'TIMEOUT', player: 1 },
      ];

      events.forEach((event) => {
        const result = transition(state, event, context);
        expect(result.ok).toBe(false);
        if (!result.ok) {
          expect(result.error.message).toContain('Game is over');
        }
      });
    });
  });

  describe('phase completion helpers', () => {
    it('onLineProcessingComplete should return territory_processing when regions exist', () => {
      const { onLineProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onLineProcessingComplete(true, true, true);
      expect(result).toBe('territory_processing');
    });

    it('onLineProcessingComplete should return forced_elimination when no action and has stacks', () => {
      const { onLineProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onLineProcessingComplete(false, false, true);
      expect(result).toBe('forced_elimination');
    });

    it('onLineProcessingComplete should return turn_end when player had action', () => {
      const { onLineProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onLineProcessingComplete(false, true, true);
      expect(result).toBe('turn_end');
    });

    it('onLineProcessingComplete should return turn_end when player has no stacks', () => {
      const { onLineProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onLineProcessingComplete(false, false, false);
      expect(result).toBe('turn_end');
    });

    it('onTerritoryProcessingComplete should return forced_elimination when no action and has stacks', () => {
      const { onTerritoryProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onTerritoryProcessingComplete(false, true);
      expect(result).toBe('forced_elimination');
    });

    it('onTerritoryProcessingComplete should return turn_end when player had action', () => {
      const { onTerritoryProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onTerritoryProcessingComplete(true, true);
      expect(result).toBe('turn_end');
    });

    it('onTerritoryProcessingComplete should return turn_end when player has no stacks', () => {
      const { onTerritoryProcessingComplete } = require('../../../src/shared/engine/fsm');
      const result = onTerritoryProcessingComplete(false, false);
      expect(result).toBe('turn_end');
    });
  });
});
