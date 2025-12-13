/**
 * moveActionAdapter.branchCoverage.test.ts
 *
 * Branch coverage tests for moveActionAdapter.ts targeting uncovered branches:
 * - moveToGameAction all move type cases
 * - gameActionToMove all action type cases
 * - Error cases for missing fields
 * - Line/region resolution with fallbacks
 * - COLLAPSE_ALL vs MINIMUM_COLLAPSE selection
 */

import {
  moveToGameAction,
  gameActionToMove,
  MoveMappingError,
} from '../../src/shared/engine/moveActionAdapter';
import type { Move, Position, Territory, LineInfo } from '../../src/shared/types/game';
import type { GameState as EngineGameState } from '../../src/shared/engine/types';

// Helper to create position
const pos = (x: number, y: number): Position => ({ x, y });

// Helper to create a minimal engine state
const makeEngineState = (overrides?: Partial<EngineGameState>): EngineGameState =>
  ({
    board: {
      type: 'square8',
      size: 8,
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: { 1: 0, 2: 0 },
    },
    players: [
      { playerNumber: 1, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
      { playerNumber: 2, ringsInHand: 18, eliminatedRings: 0, territorySpaces: 0 },
    ],
    currentPlayer: 1,
    currentPhase: 'ring_placement',
    gameStatus: 'active',
    ...overrides,
  }) as EngineGameState;

// Helper to create a line info
const makeLine = (player: number, positions: Position[]): LineInfo => ({
  player,
  positions,
  length: positions.length,
  direction: { x: 1, y: 0 },
});

// Helper to create a territory
const makeTerritory = (player: number, spaces: Position[], isDisconnected = true): Territory => ({
  controllingPlayer: player,
  spaces,
  isDisconnected,
});

describe('moveActionAdapter branch coverage', () => {
  describe('moveToGameAction', () => {
    describe('place_ring', () => {
      it('maps place_ring move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'place_ring',
          player: 1,
          to: pos(3, 3),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('PLACE_RING');
        expect(action.playerId).toBe(1);
        expect((action as { position: Position }).position).toEqual(pos(3, 3));
      });

      it('uses default count of 1 when placementCount not specified', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'place_ring',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect((action as { count: number }).count).toBe(1);
      });

      it('uses specified placementCount', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'place_ring',
          player: 1,
          to: pos(0, 0),
          placementCount: 3,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect((action as { count: number }).count).toBe(3);
      });
    });

    describe('skip_placement', () => {
      it('maps skip_placement move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'skip_placement',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('SKIP_PLACEMENT');
        expect(action.playerId).toBe(1);
      });
    });

    describe('move_stack and move_ring', () => {
      it('maps move_stack move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'move_stack',
          player: 1,
          from: pos(0, 0),
          to: pos(1, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('MOVE_STACK');
        expect((action as { from: Position }).from).toEqual(pos(0, 0));
        expect((action as { to: Position }).to).toEqual(pos(1, 0));
      });

      it('maps move_ring move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'move_ring',
          player: 1,
          from: pos(2, 2),
          to: pos(3, 3),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('MOVE_STACK');
      });

      it('throws when from position is missing', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'move_stack',
          player: 1,
          to: pos(1, 0),
          // No from
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(MoveMappingError);
        expect(() => moveToGameAction(move, state)).toThrow('missing from position');
      });
    });

    describe('overtaking_capture', () => {
      it('maps overtaking_capture move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'overtaking_capture',
          player: 1,
          from: pos(0, 0),
          to: pos(2, 0),
          captureTarget: pos(1, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('OVERTAKING_CAPTURE');
        expect((action as { captureTarget: Position }).captureTarget).toEqual(pos(1, 0));
      });

      it('throws when from is missing', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'overtaking_capture',
          player: 1,
          to: pos(2, 0),
          captureTarget: pos(1, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow('missing from or captureTarget');
      });

      it('throws when captureTarget is missing', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'overtaking_capture',
          player: 1,
          from: pos(0, 0),
          to: pos(2, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow('missing from or captureTarget');
      });
    });

    describe('continue_capture_segment', () => {
      it('maps continue_capture_segment move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'continue_capture_segment',
          player: 1,
          from: pos(0, 0),
          to: pos(2, 0),
          captureTarget: pos(1, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('CONTINUE_CHAIN');
      });

      it('throws when from or captureTarget missing', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'continue_capture_segment',
          player: 1,
          to: pos(2, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'continue_capture_segment Move is missing from or captureTarget'
        );
      });
    });

    describe('process_line', () => {
      it('maps process_line move using formedLines match', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('PROCESS_LINE');
        expect((action as { lineIndex: number }).lineIndex).toBe(0);
      });

      it('throws when no formedLines available', () => {
        const state = makeEngineState();
        state.board.formedLines = [];

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'No formedLines available on state for line-processing Move'
        );
      });

      it('falls back to first line for player when no exact match', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          // No formedLines on move
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect((action as { lineIndex: number }).lineIndex).toBe(0);
      });

      it('throws when formedLines mismatch', () => {
        const state = makeEngineState();
        const stateLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [stateLine];

        const moveLine = makeLine(1, [pos(5, 5), pos(6, 5), pos(7, 5), pos(8, 5)]);
        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(5, 5),
          formedLines: [moveLine],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'Could not match Move.formedLines[0] to any board.formedLines entry'
        );
      });

      it('throws when no line for player found', () => {
        const state = makeEngineState();
        const line = makeLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const move: Move = {
          id: 'test',
          type: 'process_line',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'No line for Move.player found in board.formedLines'
        );
      });
    });

    describe('choose_line_reward', () => {
      it('maps to COLLAPSE_ALL when no collapsedMarkers', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const move: Move = {
          id: 'test',
          type: 'choose_line_reward',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('CHOOSE_LINE_REWARD');
        expect((action as { selection: string }).selection).toBe('COLLAPSE_ALL');
      });

      it('maps to COLLAPSE_ALL when collapsedMarkers equals line length', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const move: Move = {
          id: 'test',
          type: 'choose_line_reward',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          collapsedMarkers: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect((action as { selection: string }).selection).toBe('COLLAPSE_ALL');
      });

      it('maps to MINIMUM_COLLAPSE when collapsedMarkers less than line length', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        state.board.formedLines = [line];

        const move: Move = {
          id: 'test',
          type: 'choose_line_reward',
          player: 1,
          to: pos(0, 0),
          formedLines: [line],
          collapsedMarkers: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)], // 4 < 5
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect((action as { selection: string }).selection).toBe('MINIMUM_COLLAPSE');
        expect((action as { collapsedPositions?: Position[] }).collapsedPositions).toHaveLength(4);
      });
    });

    describe('process_territory_region', () => {
      it('maps process_territory_region move', () => {
        const state = makeEngineState();
        const territory = makeTerritory(1, [pos(0, 0), pos(1, 0)]);
        state.board.territories = new Map([['region-0', territory]]);

        const move: Move = {
          id: 'test',
          type: 'process_territory_region',
          player: 1,
          to: pos(0, 0),
          disconnectedRegions: [territory],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('PROCESS_TERRITORY');
        expect((action as { regionId: string }).regionId).toBe('region-0');
      });

      it('throws when no territories available', () => {
        const state = makeEngineState();
        state.board.territories = new Map();

        const move: Move = {
          id: 'test',
          type: 'process_territory_region',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'No territories available on state for choose_territory_option Move'
        );
      });

      it('falls back to first disconnected region for player', () => {
        const state = makeEngineState();
        const territory = makeTerritory(1, [pos(0, 0)], true);
        state.board.territories = new Map([['region-0', territory]]);

        const move: Move = {
          id: 'test',
          type: 'process_territory_region',
          player: 1,
          to: pos(0, 0),
          // No disconnectedRegions
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect((action as { regionId: string }).regionId).toBe('region-0');
      });

      it('throws when no disconnected territory for player', () => {
        const state = makeEngineState();
        const territory = makeTerritory(2, [pos(0, 0)], true);
        state.board.territories = new Map([['region-0', territory]]);

        const move: Move = {
          id: 'test',
          type: 'process_territory_region',
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'No disconnected territory region found for moving player'
        );
      });

      it('throws when disconnectedRegions mismatch', () => {
        const state = makeEngineState();
        const stateTerritory = makeTerritory(1, [pos(0, 0), pos(1, 0)]);
        state.board.territories = new Map([['region-0', stateTerritory]]);

        const moveTerritory = makeTerritory(1, [pos(5, 5), pos(6, 5)]);
        const move: Move = {
          id: 'test',
          type: 'process_territory_region',
          player: 1,
          to: pos(5, 5),
          disconnectedRegions: [moveTerritory],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'Could not match disconnectedRegions[0] to any board.territories entry'
        );
      });
    });

    describe('eliminate_rings_from_stack', () => {
      it('maps eliminate_rings_from_stack move', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'eliminate_rings_from_stack',
          player: 1,
          to: pos(3, 3),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        const action = moveToGameAction(move, state);
        expect(action.type).toBe('ELIMINATE_STACK');
        expect((action as { stackPosition: Position }).stackPosition).toEqual(pos(3, 3));
      });

      it('throws when to position is missing', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'eliminate_rings_from_stack',
          player: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        } as Move;

        expect(() => moveToGameAction(move, state)).toThrow(
          'eliminate_rings_from_stack Move is missing stack position'
        );
      });
    });

    describe('legacy/unknown types', () => {
      it('throws for build_stack', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'build_stack' as Move['type'],
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow(
          'legacy/experimental and is not supported'
        );
      });

      it('throws for line_formation', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'line_formation' as Move['type'],
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow('legacy/experimental');
      });

      it('throws for territory_claim', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'territory_claim' as Move['type'],
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow('legacy/experimental');
      });

      it('throws for unknown move type', () => {
        const state = makeEngineState();
        const move: Move = {
          id: 'test',
          type: 'unknown_type' as Move['type'],
          player: 1,
          to: pos(0, 0),
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        };

        expect(() => moveToGameAction(move, state)).toThrow('Unknown Move type');
      });
    });
  });

  describe('gameActionToMove', () => {
    describe('PLACE_RING', () => {
      it('converts PLACE_RING action', () => {
        const state = makeEngineState();
        const action = { type: 'PLACE_RING' as const, playerId: 1, position: pos(3, 3), count: 1 };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('place_ring');
        expect(move.player).toBe(1);
        expect(move.to).toEqual(pos(3, 3));
      });

      it('includes placedOnStack when stack exists', () => {
        const state = makeEngineState();
        state.board.stacks.set('3,3', {
          position: pos(3, 3),
          owner: 1,
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        });

        const action = { type: 'PLACE_RING' as const, playerId: 1, position: pos(3, 3), count: 1 };
        const move = gameActionToMove(action, state);
        expect(move.placedOnStack).toBe(true);
      });

      it('includes placedOnStack false when no stack', () => {
        const state = makeEngineState();
        const action = { type: 'PLACE_RING' as const, playerId: 1, position: pos(3, 3), count: 1 };
        const move = gameActionToMove(action, state);
        expect(move.placedOnStack).toBe(false);
      });
    });

    describe('SKIP_PLACEMENT', () => {
      it('converts SKIP_PLACEMENT action', () => {
        const state = makeEngineState();
        const action = { type: 'SKIP_PLACEMENT' as const, playerId: 1 };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('skip_placement');
        expect(move.to).toEqual({ x: 0, y: 0 });
      });
    });

    describe('MOVE_STACK', () => {
      it('converts MOVE_STACK action', () => {
        const state = makeEngineState();
        const action = {
          type: 'MOVE_STACK' as const,
          playerId: 1,
          from: pos(0, 0),
          to: pos(1, 0),
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('move_stack');
        expect(move.from).toEqual(pos(0, 0));
        expect(move.to).toEqual(pos(1, 0));
      });
    });

    describe('OVERTAKING_CAPTURE', () => {
      it('converts OVERTAKING_CAPTURE action', () => {
        const state = makeEngineState();
        const action = {
          type: 'OVERTAKING_CAPTURE' as const,
          playerId: 1,
          from: pos(0, 0),
          to: pos(2, 0),
          captureTarget: pos(1, 0),
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('overtaking_capture');
        expect(move.captureTarget).toEqual(pos(1, 0));
      });
    });

    describe('CONTINUE_CHAIN', () => {
      it('converts CONTINUE_CHAIN action', () => {
        const state = makeEngineState();
        const action = {
          type: 'CONTINUE_CHAIN' as const,
          playerId: 1,
          from: pos(0, 0),
          to: pos(2, 0),
          captureTarget: pos(1, 0),
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('continue_capture_segment');
      });
    });

    describe('PROCESS_LINE', () => {
      it('converts PROCESS_LINE action', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action = { type: 'PROCESS_LINE' as const, playerId: 1, lineIndex: 0 };
        const move = gameActionToMove(action, state);
        expect(move.type).toBe('process_line');
        expect(move.formedLines).toHaveLength(1);
      });

      it('throws when lineIndex is invalid', () => {
        const state = makeEngineState();
        state.board.formedLines = [];

        const action = { type: 'PROCESS_LINE' as const, playerId: 1, lineIndex: 0 };
        expect(() => gameActionToMove(action, state)).toThrow(
          'PROCESS_LINE action references missing formedLines index'
        );
      });
    });

    describe('CHOOSE_LINE_REWARD', () => {
      it('converts CHOOSE_LINE_REWARD action with COLLAPSE_ALL', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
        state.board.formedLines = [line];

        const action = {
          type: 'CHOOSE_LINE_REWARD' as const,
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL' as const,
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('choose_line_option');
        expect(move.collapsedMarkers).toEqual(line.positions);
      });

      it('converts CHOOSE_LINE_REWARD with MINIMUM_COLLAPSE', () => {
        const state = makeEngineState();
        const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
        state.board.formedLines = [line];

        const action = {
          type: 'CHOOSE_LINE_REWARD' as const,
          playerId: 1,
          lineIndex: 0,
          selection: 'MINIMUM_COLLAPSE' as const,
          collapsedPositions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
        };

        const move = gameActionToMove(action, state);
        expect(move.collapsedMarkers).toHaveLength(4);
      });

      it('throws when lineIndex is invalid', () => {
        const state = makeEngineState();
        state.board.formedLines = [];

        const action = {
          type: 'CHOOSE_LINE_REWARD' as const,
          playerId: 1,
          lineIndex: 0,
          selection: 'COLLAPSE_ALL' as const,
        };

        expect(() => gameActionToMove(action, state)).toThrow(
          'CHOOSE_LINE_REWARD action references missing formedLines index'
        );
      });
    });

    describe('PROCESS_TERRITORY', () => {
      it('converts PROCESS_TERRITORY action', () => {
        const state = makeEngineState();
        const territory = makeTerritory(1, [pos(0, 0), pos(1, 0)]);
        state.board.territories = new Map([['region-0', territory]]);

        const action = {
          type: 'PROCESS_TERRITORY' as const,
          playerId: 1,
          regionId: 'region-0',
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('choose_territory_option');
        expect(move.disconnectedRegions).toHaveLength(1);
      });

      it('throws when regionId not found', () => {
        const state = makeEngineState();
        state.board.territories = new Map();

        const action = {
          type: 'PROCESS_TERRITORY' as const,
          playerId: 1,
          regionId: 'region-unknown',
        };

        expect(() => gameActionToMove(action, state)).toThrow(
          'PROCESS_TERRITORY action references unknown regionId'
        );
      });
    });

    describe('ELIMINATE_STACK', () => {
      it('converts ELIMINATE_STACK action with stack present', () => {
        const state = makeEngineState();
        state.board.stacks.set('3,3', {
          position: pos(3, 3),
          owner: 1,
          rings: [1, 1, 2],
          stackHeight: 3,
          capHeight: 2,
          controllingPlayer: 1,
        });

        const action = {
          type: 'ELIMINATE_STACK' as const,
          playerId: 1,
          stackPosition: pos(3, 3),
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('eliminate_rings_from_stack');
        expect(move.eliminatedRings).toEqual([{ player: 1, count: 2 }]);
        expect(move.eliminationFromStack).toEqual({
          position: pos(3, 3),
          capHeight: 2,
          totalHeight: 3,
        });
      });

      it('converts ELIMINATE_STACK action without stack', () => {
        const state = makeEngineState();

        const action = {
          type: 'ELIMINATE_STACK' as const,
          playerId: 1,
          stackPosition: pos(3, 3),
        };

        const move = gameActionToMove(action, state);
        expect(move.type).toBe('eliminate_rings_from_stack');
        expect(move.eliminatedRings).toBeUndefined();
      });
    });

    describe('unknown action type', () => {
      it('throws for unknown action type', () => {
        const state = makeEngineState();
        const action = { type: 'UNKNOWN' as never, playerId: 1 };

        expect(() => gameActionToMove(action, state)).toThrow(MoveMappingError);
      });
    });
  });

  describe('MoveMappingError', () => {
    it('has correct name', () => {
      const error = new MoveMappingError('test error');
      expect(error.name).toBe('MoveMappingError');
    });

    it('includes move when provided', () => {
      const move: Move = {
        id: 'test',
        type: 'place_ring',
        player: 1,
        to: pos(0, 0),
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };
      const error = new MoveMappingError('test error', move);
      expect(error.move).toBe(move);
    });
  });

  // ==========================================================================
  // Additional branch coverage tests (lines 258, 292-295, 405-412, 435-443, 459, 487)
  // ==========================================================================
  describe('resolveLineIndexFromMove additional branches', () => {
    it('rejects line with different player (line 258)', () => {
      const state = makeEngineState();
      // Line on state belongs to player 2
      const stateLine = makeLine(2, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
      state.board.formedLines = [stateLine];

      // Move references a line with same positions but different player
      const moveLine = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)]);
      const move: Move = {
        id: 'test',
        type: 'process_line',
        player: 1,
        to: pos(0, 0),
        formedLines: [moveLine],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      // Should throw because move.formedLines[0].player !== line.player
      expect(() => moveToGameAction(move, state)).toThrow(
        'Could not match Move.formedLines[0] to any board.formedLines entry'
      );
    });
  });

  describe('resolveRegionIdFromMove additional branches', () => {
    it('skips non-disconnected regions (line 292)', () => {
      const state = makeEngineState();
      // Add a connected territory (isDisconnected = false)
      const connectedTerritory = makeTerritory(1, [pos(0, 0), pos(1, 0)], false);
      // Add a disconnected territory
      const disconnectedTerritory = makeTerritory(1, [pos(5, 5), pos(6, 5)], true);
      state.board.territories = new Map([
        ['region-connected', connectedTerritory],
        ['region-disconnected', disconnectedTerritory],
      ]);

      const move: Move = {
        id: 'test',
        type: 'process_territory_region',
        player: 1,
        to: pos(5, 5),
        disconnectedRegions: [disconnectedTerritory],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const action = moveToGameAction(move, state);
      // Should match the disconnected one, skipping the connected one
      expect((action as { regionId: string }).regionId).toBe('region-disconnected');
    });

    it('skips regions with different controlling player (line 293)', () => {
      const state = makeEngineState();
      // Add territory for player 2
      const player2Territory = makeTerritory(2, [pos(0, 0), pos(1, 0)], true);
      // Add territory for player 1 (matching)
      const player1Territory = makeTerritory(1, [pos(5, 5), pos(6, 5)], true);
      state.board.territories = new Map([
        ['region-player2', player2Territory],
        ['region-player1', player1Territory],
      ]);

      const move: Move = {
        id: 'test',
        type: 'process_territory_region',
        player: 1,
        to: pos(5, 5),
        disconnectedRegions: [player1Territory],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const action = moveToGameAction(move, state);
      // Should skip player 2's region
      expect((action as { regionId: string }).regionId).toBe('region-player1');
    });

    it('skips regions with different size (line 295)', () => {
      const state = makeEngineState();
      // Add territory with different size
      const smallTerritory = makeTerritory(1, [pos(0, 0)], true);
      // Add territory with matching size
      const matchingTerritory = makeTerritory(1, [pos(5, 5), pos(6, 5)], true);
      state.board.territories = new Map([
        ['region-small', smallTerritory],
        ['region-matching', matchingTerritory],
      ]);

      const move: Move = {
        id: 'test',
        type: 'process_territory_region',
        player: 1,
        to: pos(5, 5),
        disconnectedRegions: [matchingTerritory],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const action = moveToGameAction(move, state);
      expect((action as { regionId: string }).regionId).toBe('region-matching');
    });
  });

  describe('actionToProcessLineMove fallbacks', () => {
    it('uses fallback when line.length is undefined (line 405)', () => {
      const state = makeEngineState();
      // Create line without explicit length
      const line = {
        player: 1,
        positions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
        // No length property
        direction: { x: 1, y: 0 },
      };
      state.board.formedLines = [line as LineInfo];

      const action = { type: 'PROCESS_LINE' as const, playerId: 1, lineIndex: 0 };
      const move = gameActionToMove(action, state);

      // Should use positions.length as fallback
      expect(move.formedLines![0].length).toBe(4);
    });

    it('uses fallback when line.direction is undefined (line 406)', () => {
      const state = makeEngineState();
      // Create line without direction
      const line = {
        player: 1,
        positions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
        length: 4,
        // No direction property
      };
      state.board.formedLines = [line as LineInfo];

      const action = { type: 'PROCESS_LINE' as const, playerId: 1, lineIndex: 0 };
      const move = gameActionToMove(action, state);

      // Should use fallback direction
      expect(move.formedLines![0].direction).toEqual({ x: 0, y: 0 });
    });

    it('uses fallback when line.positions is empty (line 412)', () => {
      const state = makeEngineState();
      // Create line with empty positions
      const line = {
        player: 1,
        positions: [] as Position[],
        length: 0,
        direction: { x: 1, y: 0 },
      };
      state.board.formedLines = [line as LineInfo];

      const action = { type: 'PROCESS_LINE' as const, playerId: 1, lineIndex: 0 };
      const move = gameActionToMove(action, state);

      // Should use fallback position
      expect(move.to).toEqual({ x: 0, y: 0 });
    });
  });

  describe('actionToChooseLineRewardMove fallbacks', () => {
    it('uses fallback when line.length is undefined (line 435)', () => {
      const state = makeEngineState();
      const line = {
        player: 1,
        positions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
        // No length
        direction: { x: 1, y: 0 },
      };
      state.board.formedLines = [line as LineInfo];

      const action = {
        type: 'CHOOSE_LINE_REWARD' as const,
        playerId: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL' as const,
      };
      const move = gameActionToMove(action, state);

      expect(move.formedLines![0].length).toBe(4);
    });

    it('uses fallback when line.direction is undefined (line 436)', () => {
      const state = makeEngineState();
      const line = {
        player: 1,
        positions: [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0)],
        length: 4,
        // No direction
      };
      state.board.formedLines = [line as LineInfo];

      const action = {
        type: 'CHOOSE_LINE_REWARD' as const,
        playerId: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL' as const,
      };
      const move = gameActionToMove(action, state);

      expect(move.formedLines![0].direction).toEqual({ x: 0, y: 0 });
    });

    it('uses fallback when line.positions is empty (line 443)', () => {
      const state = makeEngineState();
      const line = {
        player: 1,
        positions: [] as Position[],
        length: 0,
        direction: { x: 1, y: 0 },
      };
      state.board.formedLines = [line as LineInfo];

      const action = {
        type: 'CHOOSE_LINE_REWARD' as const,
        playerId: 1,
        lineIndex: 0,
        selection: 'COLLAPSE_ALL' as const,
      };
      const move = gameActionToMove(action, state);

      expect(move.to).toEqual({ x: 0, y: 0 });
    });

    it('handles MINIMUM_COLLAPSE without collapsedPositions (line 439)', () => {
      const state = makeEngineState();
      const line = makeLine(1, [pos(0, 0), pos(1, 0), pos(2, 0), pos(3, 0), pos(4, 0)]);
      state.board.formedLines = [line];

      const action = {
        type: 'CHOOSE_LINE_REWARD' as const,
        playerId: 1,
        lineIndex: 0,
        selection: 'MINIMUM_COLLAPSE' as const,
        // No collapsedPositions - should use fallback
      };
      const move = gameActionToMove(action, state);

      // collapsedMarkers should be undefined or empty
      expect(move.collapsedMarkers).toBeUndefined();
    });
  });

  describe('actionToProcessTerritoryMove fallbacks', () => {
    it('uses fallback when region.spaces is empty (line 459)', () => {
      const state = makeEngineState();
      const territory = {
        controllingPlayer: 1,
        spaces: [] as Position[],
        isDisconnected: true,
      };
      state.board.territories = new Map([['region-empty', territory]]);

      const action = {
        type: 'PROCESS_TERRITORY' as const,
        playerId: 1,
        regionId: 'region-empty',
      };
      const move = gameActionToMove(action, state);

      // Should use fallback position
      expect(move.to).toEqual({ x: 0, y: 0 });
    });
  });

  describe('actionToEliminateStackMove fallbacks', () => {
    it('uses capHeight as fallback for totalHeight when stackHeight undefined (line 487)', () => {
      const state = makeEngineState();
      // Create a stack without stackHeight property
      const stack = {
        position: pos(3, 3),
        owner: 1,
        rings: [1, 1],
        capHeight: 2,
        controllingPlayer: 1,
        // stackHeight is undefined
      };
      state.board.stacks.set('3,3', stack as any);

      const action = {
        type: 'ELIMINATE_STACK' as const,
        playerId: 1,
        stackPosition: pos(3, 3),
      };
      const move = gameActionToMove(action, state);

      // totalHeight should fall back to capHeight
      expect(move.eliminationFromStack?.totalHeight).toBe(2);
    });
  });
});
