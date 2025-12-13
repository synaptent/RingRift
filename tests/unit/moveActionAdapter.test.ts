/**
 * MoveActionAdapter Unit Tests
 *
 * Tests for bidirectional Move <-> GameAction conversion.
 * Covers moveToGameAction and gameActionToMove functions.
 */

import {
  moveToGameAction,
  gameActionToMove,
  MoveMappingError,
} from '../../src/shared/engine/moveActionAdapter';
import type { Move, Position } from '../../src/shared/types/game';
import type {
  GameState as EngineGameState,
  PlaceRingAction,
  SkipPlacementAction,
  MoveStackAction,
  OvertakingCaptureAction,
  ContinueChainAction,
  ProcessLineAction,
  ChooseLineRewardAction,
  ProcessTerritoryAction,
  EliminateStackAction,
} from '../../src/shared/engine/types';

describe('moveActionAdapter', () => {
  const createMockState = (overrides: Partial<EngineGameState> = {}): EngineGameState =>
    ({
      board: {
        stacks: new Map(),
        markers: new Map(),
        formedLines: [],
        territories: new Map(),
        collapsedSpaces: new Map(),
      },
      currentPlayer: 1,
      currentPhase: 'movement',
      players: [],
      ...overrides,
    }) as unknown as EngineGameState;

  describe('moveToGameAction', () => {
    describe('place_ring', () => {
      it('should convert place_ring Move to PlaceRingAction', () => {
        const move: Move = {
          type: 'place_ring',
          player: 1,
          to: { x: 3, y: 4 },
          placementCount: 2,
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('PLACE_RING');
        expect((action as PlaceRingAction).playerId).toBe(1);
        expect((action as PlaceRingAction).position).toEqual({ x: 3, y: 4 });
        expect((action as PlaceRingAction).count).toBe(2);
      });

      it('should default placementCount to 1 if not provided', () => {
        const move: Move = {
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect((action as PlaceRingAction).count).toBe(1);
      });
    });

    describe('skip_placement', () => {
      it('should convert skip_placement Move to SkipPlacementAction', () => {
        const move: Move = {
          type: 'skip_placement',
          player: 2,
          to: { x: 0, y: 0 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('SKIP_PLACEMENT');
        expect((action as SkipPlacementAction).playerId).toBe(2);
      });
    });

    describe('move_stack/move_ring', () => {
      it('should convert move_stack Move to MoveStackAction', () => {
        const move: Move = {
          type: 'move_stack',
          player: 1,
          from: { x: 1, y: 1 },
          to: { x: 2, y: 2 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('MOVE_STACK');
        expect((action as MoveStackAction).from).toEqual({ x: 1, y: 1 });
        expect((action as MoveStackAction).to).toEqual({ x: 2, y: 2 });
      });

      it('should convert move_ring Move to MoveStackAction', () => {
        const move: Move = {
          type: 'move_ring',
          player: 1,
          from: { x: 0, y: 0 },
          to: { x: 1, y: 0 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('MOVE_STACK');
      });

      it('should throw MoveMappingError if from is missing', () => {
        const move: Move = {
          type: 'move_stack',
          player: 1,
          to: { x: 2, y: 2 },
        } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow(MoveMappingError);
        expect(() => moveToGameAction(move, state)).toThrow(
          'move_stack/move_ring Move is missing from position'
        );
      });
    });

    describe('overtaking_capture', () => {
      it('should convert overtaking_capture Move to OvertakingCaptureAction', () => {
        const move: Move = {
          type: 'overtaking_capture',
          player: 1,
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
          captureTarget: { x: 2, y: 2 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('OVERTAKING_CAPTURE');
        expect((action as OvertakingCaptureAction).captureTarget).toEqual({ x: 2, y: 2 });
      });

      it('should throw if from is missing', () => {
        const move: Move = {
          type: 'overtaking_capture',
          player: 1,
          to: { x: 1, y: 1 },
          captureTarget: { x: 2, y: 2 },
        } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow(MoveMappingError);
      });

      it('should throw if captureTarget is missing', () => {
        const move: Move = {
          type: 'overtaking_capture',
          player: 1,
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
        } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow(MoveMappingError);
      });
    });

    describe('continue_capture_segment', () => {
      it('should convert continue_capture_segment to ContinueChainAction', () => {
        const move: Move = {
          type: 'continue_capture_segment',
          player: 1,
          from: { x: 1, y: 1 },
          to: { x: 2, y: 2 },
          captureTarget: { x: 3, y: 3 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('CONTINUE_CHAIN');
        expect((action as ContinueChainAction).captureTarget).toEqual({ x: 3, y: 3 });
      });

      it('should throw if from or captureTarget is missing', () => {
        const move: Move = {
          type: 'continue_capture_segment',
          player: 1,
          to: { x: 2, y: 2 },
        } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow(MoveMappingError);
      });
    });

    describe('process_line', () => {
      it('should convert process_line Move to ProcessLineAction', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'process_line',
          player: 1,
          to: { x: 0, y: 0 },
          formedLines: [
            {
              player: 1,
              positions: [
                { x: 0, y: 0 },
                { x: 1, y: 0 },
                { x: 2, y: 0 },
              ],
              length: 3,
              direction: { x: 1, y: 0 },
            },
          ],
        } as Move;

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('PROCESS_LINE');
        expect((action as ProcessLineAction).lineIndex).toBe(0);
      });

      it('should throw if no formedLines available', () => {
        const state = createMockState();
        state.board.formedLines = [];

        const move: Move = {
          type: 'process_line',
          player: 1,
          to: { x: 0, y: 0 },
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow(MoveMappingError);
        expect(() => moveToGameAction(move, state)).toThrow('No formedLines available');
      });

      it('should fallback to first line for player if no formedLines metadata', () => {
        const state = createMockState();
        state.board.formedLines = [
          { player: 2, positions: [{ x: 0, y: 0 }] },
          { player: 1, positions: [{ x: 1, y: 1 }] },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'process_line',
          player: 1,
          to: { x: 0, y: 0 },
          // No formedLines metadata
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ProcessLineAction).lineIndex).toBe(1);
      });

      it('should throw if no matching line found', () => {
        const state = createMockState();
        state.board.formedLines = [
          { player: 2, positions: [{ x: 0, y: 0 }] },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'process_line',
          player: 1,
          to: { x: 0, y: 0 },
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow('No line for Move.player');
      });

      it('should throw if formedLines metadata doesnt match', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [{ x: 5, y: 5 }],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'process_line',
          player: 1,
          to: { x: 0, y: 0 },
          formedLines: [
            {
              player: 1,
              positions: [{ x: 99, y: 99 }], // Different positions
              length: 1,
              direction: { x: 1, y: 0 },
            },
          ],
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow('Could not match Move.formedLines[0]');
      });
    });

    describe('choose_line_reward', () => {
      it('should convert to COLLAPSE_ALL when no collapsedMarkers', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
            ],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'choose_line_reward',
          player: 1,
          to: { x: 0, y: 0 },
          formedLines: [
            {
              player: 1,
              positions: [
                { x: 0, y: 0 },
                { x: 1, y: 0 },
              ],
              length: 2,
              direction: { x: 1, y: 0 },
            },
          ],
        } as Move;

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('CHOOSE_LINE_REWARD');
        expect((action as ChooseLineRewardAction).selection).toBe('COLLAPSE_ALL');
      });

      it('should convert to COLLAPSE_ALL when collapsedMarkers equals line length', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
            ],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'choose_line_reward',
          player: 1,
          to: { x: 0, y: 0 },
          formedLines: [
            {
              player: 1,
              positions: [
                { x: 0, y: 0 },
                { x: 1, y: 0 },
              ],
              length: 2,
              direction: { x: 1, y: 0 },
            },
          ],
          collapsedMarkers: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
          ],
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ChooseLineRewardAction).selection).toBe('COLLAPSE_ALL');
      });

      it('should convert to MINIMUM_COLLAPSE when collapsedMarkers less than line length', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const move: Move = {
          type: 'choose_line_reward',
          player: 1,
          to: { x: 0, y: 0 },
          formedLines: [
            {
              player: 1,
              positions: [
                { x: 0, y: 0 },
                { x: 1, y: 0 },
                { x: 2, y: 0 },
              ],
              length: 3,
              direction: { x: 1, y: 0 },
            },
          ],
          collapsedMarkers: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
          ],
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ChooseLineRewardAction).selection).toBe('MINIMUM_COLLAPSE');
        expect((action as ChooseLineRewardAction).collapsedPositions).toHaveLength(2);
      });
    });

    describe('process_territory_region', () => {
      it('should convert to ProcessTerritoryAction', () => {
        const state = createMockState();
        state.board.territories = new Map([
          ['region-1', { isDisconnected: true, controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
          disconnectedRegions: [{ controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        } as Move;

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('PROCESS_TERRITORY');
        expect((action as ProcessTerritoryAction).regionId).toBe('region-1');
      });

      it('should throw if no territories available', () => {
        const state = createMockState();
        state.board.territories = new Map();

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow('No territories available');
      });

      it('should fallback to first disconnected region for player', () => {
        const state = createMockState();
        state.board.territories = new Map([
          ['region-1', { isDisconnected: false, controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
          ['region-2', { isDisconnected: true, controllingPlayer: 1, spaces: [{ x: 1, y: 1 }] }],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
          // No disconnectedRegions metadata
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ProcessTerritoryAction).regionId).toBe('region-2');
      });

      it('should throw if no disconnected region found for player', () => {
        const state = createMockState();
        state.board.territories = new Map([
          ['region-1', { isDisconnected: true, controllingPlayer: 2, spaces: [{ x: 0, y: 0 }] }],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow('No disconnected territory region');
      });
    });

    describe('eliminate_rings_from_stack', () => {
      it('should convert to EliminateStackAction', () => {
        const move: Move = {
          type: 'eliminate_rings_from_stack',
          player: 1,
          to: { x: 3, y: 3 },
        } as Move;
        const state = createMockState();

        const action = moveToGameAction(move, state);

        expect(action.type).toBe('ELIMINATE_STACK');
        expect((action as EliminateStackAction).stackPosition).toEqual({ x: 3, y: 3 });
      });

      it('should throw if to is missing', () => {
        const move: Move = {
          type: 'eliminate_rings_from_stack',
          player: 1,
        } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow('missing stack position');
      });
    });

    describe('legacy/experimental types', () => {
      it('should throw for build_stack', () => {
        const move = { type: 'build_stack', player: 1, to: { x: 0, y: 0 } } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow('legacy/experimental');
      });

      it('should throw for line_formation', () => {
        const move = { type: 'line_formation', player: 1, to: { x: 0, y: 0 } } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow('legacy/experimental');
      });

      it('should throw for territory_claim', () => {
        const move = { type: 'territory_claim', player: 1, to: { x: 0, y: 0 } } as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow('legacy/experimental');
      });

      it('should throw for unknown move type', () => {
        const move = { type: 'unknown_type', player: 1, to: { x: 0, y: 0 } } as unknown as Move;
        const state = createMockState();

        expect(() => moveToGameAction(move, state)).toThrow('Unknown Move type');
      });
    });

    describe('process_territory_region edge cases', () => {
      it('should skip non-disconnected regions when looking for match', () => {
        const state = createMockState();
        state.board.territories = new Map([
          [
            'region-connected',
            { isDisconnected: false, controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] },
          ],
          [
            'region-disconnected',
            { isDisconnected: true, controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] },
          ],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
          disconnectedRegions: [{ controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ProcessTerritoryAction).regionId).toBe('region-disconnected');
      });

      it('should skip regions with different controlling player', () => {
        const state = createMockState();
        state.board.territories = new Map([
          ['region-p2', { isDisconnected: true, controllingPlayer: 2, spaces: [{ x: 0, y: 0 }] }],
          ['region-p1', { isDisconnected: true, controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
          disconnectedRegions: [{ controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ProcessTerritoryAction).regionId).toBe('region-p1');
      });

      it('should skip regions with different size', () => {
        const state = createMockState();
        state.board.territories = new Map([
          [
            'region-big',
            {
              isDisconnected: true,
              controllingPlayer: 1,
              spaces: [
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ],
            },
          ],
          [
            'region-small',
            { isDisconnected: true, controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] },
          ],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
          disconnectedRegions: [{ controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        } as Move;

        const action = moveToGameAction(move, state);

        expect((action as ProcessTerritoryAction).regionId).toBe('region-small');
      });

      it('should detect mismatch when spaces differ', () => {
        const state = createMockState();
        state.board.territories = new Map([
          ['region-1', { isDisconnected: true, controllingPlayer: 1, spaces: [{ x: 5, y: 5 }] }],
        ]) as unknown as EngineGameState['board']['territories'];

        const move: Move = {
          type: 'process_territory_region',
          player: 1,
          to: { x: 0, y: 0 },
          disconnectedRegions: [{ controllingPlayer: 1, spaces: [{ x: 0, y: 0 }] }],
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow(
          'Could not match disconnectedRegions[0]'
        );
      });
    });
  });

  describe('gameActionToMove', () => {
    describe('PLACE_RING', () => {
      it('should convert PlaceRingAction to place_ring Move', () => {
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: { x: 2, y: 3 },
          count: 1,
        };
        const state = createMockState();

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('place_ring');
        expect(move.player).toBe(1);
        expect(move.to).toEqual({ x: 2, y: 3 });
      });

      it('should set placedOnStack based on existing stack', () => {
        const action: PlaceRingAction = {
          type: 'PLACE_RING',
          playerId: 1,
          position: { x: 2, y: 3 },
          count: 1,
        };
        const state = createMockState();
        state.board.stacks.set('2,3', { rings: [1, 2], stackHeight: 2 } as any);

        const move = gameActionToMove(action, state);

        expect(move.placedOnStack).toBe(true);
      });
    });

    describe('SKIP_PLACEMENT', () => {
      it('should convert SkipPlacementAction to skip_placement Move', () => {
        const action: SkipPlacementAction = {
          type: 'SKIP_PLACEMENT',
          playerId: 2,
        };
        const state = createMockState();

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('skip_placement');
        expect(move.player).toBe(2);
      });
    });

    describe('MOVE_STACK', () => {
      it('should convert MoveStackAction to move_stack Move', () => {
        const action: MoveStackAction = {
          type: 'MOVE_STACK',
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
        };
        const state = createMockState();

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('move_stack');
        expect(move.from).toEqual({ x: 0, y: 0 });
        expect(move.to).toEqual({ x: 1, y: 1 });
      });
    });

    describe('OVERTAKING_CAPTURE', () => {
      it('should convert OvertakingCaptureAction to overtaking_capture Move', () => {
        const action: OvertakingCaptureAction = {
          type: 'OVERTAKING_CAPTURE',
          playerId: 1,
          from: { x: 0, y: 0 },
          to: { x: 1, y: 1 },
          captureTarget: { x: 2, y: 2 },
        };
        const state = createMockState();

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('overtaking_capture');
        expect(move.captureTarget).toEqual({ x: 2, y: 2 });
      });
    });

    describe('CONTINUE_CHAIN', () => {
      it('should convert ContinueChainAction to continue_capture_segment Move', () => {
        const action: ContinueChainAction = {
          type: 'CONTINUE_CHAIN',
          playerId: 1,
          from: { x: 1, y: 1 },
          to: { x: 2, y: 2 },
          captureTarget: { x: 3, y: 3 },
        };
        const state = createMockState();

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('continue_capture_segment');
        expect(move.captureTarget).toEqual({ x: 3, y: 3 });
      });
    });

    describe('PROCESS_LINE', () => {
      it('should convert ProcessLineAction to process_line Move', () => {
        const state = createMockState();
        state.board.formedLines = [
          { player: 1, positions: [{ x: 0, y: 0 }], length: 1, direction: { x: 1, y: 0 } },
        ] as unknown as EngineGameState['board']['formedLines'];

        const action: ProcessLineAction = {
          type: 'PROCESS_LINE',
          playerId: 1,
          lineIndex: 0,
        };

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('process_line');
        expect(move.formedLines).toHaveLength(1);
      });

      it('should throw if lineIndex is out of bounds', () => {
        const state = createMockState();
        state.board.formedLines = [];

        const action: ProcessLineAction = {
          type: 'PROCESS_LINE',
          playerId: 1,
          lineIndex: 0,
        };

        expect(() => gameActionToMove(action, state)).toThrow(
          'PROCESS_LINE action references missing formedLines index'
        );
      });

      it('should use default values for missing length/direction', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
            ],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const action: ProcessLineAction = {
          type: 'PROCESS_LINE',
          playerId: 1,
          lineIndex: 0,
        };

        const move = gameActionToMove(action, state);

        expect(move.formedLines![0].length).toBe(2); // Uses positions.length
        expect(move.formedLines![0].direction).toEqual({ x: 0, y: 0 }); // Default
      });
    });

    describe('CHOOSE_LINE_REWARD', () => {
      it('should convert MINIMUM_COLLAPSE with positions', () => {
        const state = createMockState();
        state.board.formedLines = [
          {
            player: 1,
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
          },
        ] as unknown as EngineGameState['board']['formedLines'];

        const action: ChooseLineRewardAction = {
          type: 'CHOOSE_LINE_REWARD',
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE',
          collapsedPositions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
          ],
        };

        const move = gameActionToMove(action, state);

        // Canonical line option move type (legacy alias: choose_line_reward).
        expect(move.type).toBe('choose_line_option');
        expect(move.collapsedMarkers).toHaveLength(2);
      });

      it('should throw if lineIndex is out of bounds', () => {
        const state = createMockState();
        state.board.formedLines = [];

        const action: ChooseLineRewardAction = {
          type: 'CHOOSE_LINE_REWARD',
          playerId: 1,
          lineIndex: 5,
          selection: 'COLLAPSE_ALL',
        };

        expect(() => gameActionToMove(action, state)).toThrow('missing formedLines index');
      });
    });

    describe('PROCESS_TERRITORY', () => {
      it('should convert ProcessTerritoryAction to choose_territory_option Move', () => {
        const state = createMockState();
        state.board.territories = new Map([
          ['region-abc', { spaces: [{ x: 1, y: 2 }], controllingPlayer: 1 }],
        ]) as unknown as EngineGameState['board']['territories'];

        const action: ProcessTerritoryAction = {
          type: 'PROCESS_TERRITORY',
          playerId: 1,
          regionId: 'region-abc',
        };

        const move = gameActionToMove(action, state);

        // Canonical territory decision move type (legacy alias: process_territory_region).
        expect(move.type).toBe('choose_territory_option');
        expect(move.disconnectedRegions).toHaveLength(1);
      });

      it('should throw if regionId not found', () => {
        const state = createMockState();
        state.board.territories = new Map();

        const action: ProcessTerritoryAction = {
          type: 'PROCESS_TERRITORY',
          playerId: 1,
          regionId: 'nonexistent',
        };

        expect(() => gameActionToMove(action, state)).toThrow('unknown regionId');
      });
    });

    describe('ELIMINATE_STACK', () => {
      it('should convert EliminateStackAction to eliminate_rings_from_stack Move', () => {
        const state = createMockState();
        state.board.stacks.set('3,3', { capHeight: 2, stackHeight: 5 } as any);

        const action: EliminateStackAction = {
          type: 'ELIMINATE_STACK',
          playerId: 1,
          stackPosition: { x: 3, y: 3 },
        };

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('eliminate_rings_from_stack');
        expect(move.eliminatedRings).toBeDefined();
        expect(move.eliminationFromStack?.capHeight).toBe(2);
      });

      it('should handle missing stack gracefully', () => {
        const state = createMockState();

        const action: EliminateStackAction = {
          type: 'ELIMINATE_STACK',
          playerId: 1,
          stackPosition: { x: 9, y: 9 },
        };

        const move = gameActionToMove(action, state);

        expect(move.type).toBe('eliminate_rings_from_stack');
        expect(move.eliminatedRings).toBeUndefined();
      });
    });

    describe('unknown action type', () => {
      it('should throw for unknown action type', () => {
        const action = { type: 'UNKNOWN_ACTION', playerId: 1 } as unknown;
        const state = createMockState();

        expect(() => gameActionToMove(action as any, state)).toThrow('Unsupported GameAction type');
      });
    });
  });

  describe('MoveMappingError', () => {
    it('should preserve move on error', () => {
      const move: Move = { type: 'move_stack', player: 1, to: { x: 0, y: 0 } } as Move;

      try {
        moveToGameAction(move, createMockState());
      } catch (error) {
        expect(error).toBeInstanceOf(MoveMappingError);
        expect((error as MoveMappingError).move).toBe(move);
      }
    });

    it('should have correct name property', () => {
      const error = new MoveMappingError('test');
      expect(error.name).toBe('MoveMappingError');
    });
  });
});
