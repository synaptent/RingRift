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
  type ChainCaptureState,
  type LineProcessingState,
  type TerritoryProcessingState,
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

    it('should transition from movement to line_processing on MOVE_STACK', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
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

    it('should transition to game_over on RESIGN', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
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
    it('should reject CHOOSE_LINE_REWARD when awaitingReward is false', () => {
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
        awaitingReward: false, // Not awaiting reward
      };

      const event: TurnEvent = { type: 'CHOOSE_LINE_REWARD', choice: 'eliminate_opponent' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Not awaiting line reward choice');
      }
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

      const event: TurnEvent = { type: 'CHOOSE_LINE_REWARD', choice: 'eliminate_opponent' };
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
  });
});
