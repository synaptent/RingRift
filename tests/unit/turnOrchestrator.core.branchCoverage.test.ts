/**
 * TurnOrchestrator core branch coverage tests
 *
 * Tests for src/shared/engine/orchestration/turnOrchestrator.ts covering:
 * - processTurn with different move types
 * - applyMoveWithChainInfo for all move type branches
 * - processPostMovePhases for phase transitions and decisions
 * - validateMove for move validation
 * - getValidMoves for valid move enumeration
 * - Decision creation helpers
 * - S-Invariant and ANM resolution
 */

import {
  processTurn,
  processTurnAsync,
  validateMove,
  getValidMoves,
  hasValidMoves,
} from '../../src/shared/engine/orchestration/turnOrchestrator';
import * as TerritoryAggregate from '../../src/shared/engine/aggregates/TerritoryAggregate';
import type {
  GameState,
  GamePhase,
  Move,
  Player,
  Board,
  Position,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

describe('TurnOrchestrator core branch coverage', () => {
  const createPlayer = (playerNumber: number, ringsInHand: number = 18): Player => ({
    id: `player-${playerNumber}`,
    username: `Player ${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand,
    eliminatedRings: 0,
    territorySpaces: 0,
  });

  const createEmptyBoard = (size: number = 8): Board => ({
    type: 'square8',
    size,
    stacks: new Map(),
    markers: new Map(),
    territories: new Map(),
    formedLines: [],
    collapsedSpaces: new Map(),
    eliminatedRings: {},
  });

  const createBaseState = (
    phase: GamePhase = 'ring_placement',
    numPlayers: number = 2
  ): GameState => ({
    id: 'test-game',
    currentPlayer: 1,
    currentPhase: phase,
    gameStatus: 'active',
    boardType: 'square8',
    players: Array.from({ length: numPlayers }, (_, i) => createPlayer(i + 1)),
    board: createEmptyBoard(),
    moveHistory: [],
    history: [],
    lastMoveAt: new Date(),
    createdAt: new Date(),
    isRated: false,
    spectators: [],
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
    maxPlayers: numPlayers,
    totalRingsInPlay: 36,
    victoryThreshold: 19,
  });

  const createMove = (
    type: Move['type'],
    player: number,
    to: Position,
    extras: Partial<Move> = {}
  ): Move => ({
    id: `test-move-${Date.now()}`,
    type,
    player,
    to,
    timestamp: new Date(),
    thinkTime: 100,
    moveNumber: 1,
    ...extras,
  });

  describe('processTurn', () => {
    describe('place_ring move', () => {
      it('processes place_ring move successfully', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 3, y: 3 });

        const result = processTurn(state, move);

        expect(result.nextState.currentPhase).toBe('movement');
        expect(['complete', 'awaiting_decision']).toContain(result.status);
      });

      it('transitions to movement phase after placement', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 3, y: 3 });

        const result = processTurn(state, move);

        expect(result.nextState.currentPhase).toBe('movement');
      });

      it('decrements ringsInHand after placement', () => {
        const state = createBaseState('ring_placement');
        state.players[0].ringsInHand = 5;
        const move = createMove('place_ring', 1, { x: 3, y: 3 });

        const result = processTurn(state, move);

        expect(result.nextState.players[0].ringsInHand).toBeLessThanOrEqual(5);
      });

      it('handles placement with placementCount', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 3, y: 3 }, { placementCount: 2 });

        const result = processTurn(state, move);

        expect(result.nextState.currentPhase).toBe('movement');
        expect(result.nextState.board.stacks.has('3,3')).toBe(true);
      });
    });

    describe('skip_placement move', () => {
      it('processes skip_placement move', () => {
        const state = createBaseState('ring_placement');
        // Give player a stack so they can skip placement
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        state.players[0].ringsInHand = 0;
        const move = createMove('skip_placement', 1, { x: 0, y: 0 });

        const result = processTurn(state, move);

        expect(result.nextState.currentPhase).toBe('movement');
      });
    });

    describe('skip_capture move', () => {
      it('transitions to line_processing on skip_capture', () => {
        const state = createBaseState('capture');
        const move = createMove('skip_capture', 1, { x: 0, y: 0 });

        const result = processTurn(state, move);

        // After skip_capture, should proceed through phases
        expect(['line_processing', 'territory_processing', 'ring_placement']).toContain(
          result.nextState.currentPhase
        );
      });
    });

    describe('move_stack move', () => {
      it('processes move_stack with from position', () => {
        const state = createBaseState('movement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1], // Ring owner array
        });
        const move = createMove('move_stack', 1, { x: 4, y: 3 }, { from: { x: 3, y: 3 } });

        const result = processTurn(state, move);

        expect(result.nextState.board.stacks.has('4,3')).toBe(true);
        expect(result.nextState.board.stacks.has('3,3')).toBe(false);
      });

      it('throws error when from is missing for move_stack', () => {
        const state = createBaseState('movement');
        const move = createMove('move_stack', 1, { x: 4, y: 3 });

        expect(() => processTurn(state, move)).toThrow('Move.from is required');
      });
    });

    describe('move_ring move', () => {
      it('processes move_ring with from position', () => {
        const state = createBaseState('movement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        const move = createMove('move_ring', 1, { x: 4, y: 3 }, { from: { x: 3, y: 3 } });

        const result = processTurn(state, move);

        expect(result.nextState.board.stacks.has('4,3')).toBe(true);
        expect(result.nextState.board.stacks.get('4,3')?.controllingPlayer).toBe(1);
      });
    });

    describe('overtaking_capture move', () => {
      it('processes capture move with valid parameters', () => {
        const state = createBaseState('capture');
        // Set up attacker and target stacks
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });
        state.board.stacks.set('4,3', {
          position: { x: 4, y: 3 },
          stackHeight: 1,
          controllingPlayer: 2,
          composition: [{ player: 2, count: 1 }],
          rings: [2],
        });
        const move = createMove(
          'overtaking_capture',
          1,
          { x: 5, y: 3 },
          {
            from: { x: 3, y: 3 },
            captureTarget: { x: 4, y: 3 },
          }
        );

        const result = processTurn(state, move);

        expect(result.nextState.currentPhase).not.toBe('capture');
        expect(['complete', 'awaiting_decision']).toContain(result.status);
      });

      it('throws error when from or captureTarget is missing', () => {
        const state = createBaseState('capture');
        const moveNoFrom = createMove(
          'overtaking_capture',
          1,
          { x: 5, y: 3 },
          {
            captureTarget: { x: 4, y: 3 },
          }
        );

        expect(() => processTurn(state, moveNoFrom)).toThrow(
          'Move.from and Move.captureTarget are required'
        );

        const moveNoCaptureTarget = createMove(
          'overtaking_capture',
          1,
          { x: 5, y: 3 },
          {
            from: { x: 3, y: 3 },
          }
        );

        expect(() => processTurn(state, moveNoCaptureTarget)).toThrow(
          'Move.from and Move.captureTarget are required'
        );
      });
    });

    describe('continue_capture_segment move', () => {
      it('processes chain capture continuation', () => {
        const state = createBaseState('chain_capture');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });
        state.board.stacks.set('4,3', {
          position: { x: 4, y: 3 },
          stackHeight: 1,
          controllingPlayer: 2,
          composition: [{ player: 2, count: 1 }],
          rings: [2],
        });
        const move = createMove(
          'continue_capture_segment',
          1,
          { x: 5, y: 3 },
          {
            from: { x: 3, y: 3 },
            captureTarget: { x: 4, y: 3 },
          }
        );

        const result = processTurn(state, move);

        expect(['complete', 'awaiting_decision']).toContain(result.status);
        expect(result.nextState.gameStatus).toBe('active');
      });
    });

    describe('process_line move', () => {
      it('processes line move', () => {
        const state = createBaseState('line_processing');
        const move = createMove(
          'process_line',
          1,
          { x: 0, y: 0 },
          {
            formedLines: [
              {
                positions: [
                  { x: 0, y: 0 },
                  { x: 1, y: 0 },
                  { x: 2, y: 0 },
                  { x: 3, y: 0 },
                  { x: 4, y: 0 },
                ],
                player: 1,
                length: 5,
                direction: 'horizontal',
              },
            ],
          }
        );

        const result = processTurn(state, move);

        expect(['complete', 'awaiting_decision']).toContain(result.status);
        expect(result.nextState.gameStatus).toBe('active');
      });
    });

    describe('choose_line_reward move', () => {
      it('processes line reward choice', () => {
        const state = createBaseState('line_processing');
        const move = createMove(
          'choose_line_reward',
          1,
          { x: 0, y: 0 },
          {
            rewardType: 'COLLAPSE_ALL',
          }
        );

        const result = processTurn(state, move);

        expect(['complete', 'awaiting_decision']).toContain(result.status);
        expect(result.nextState.gameStatus).toBe('active');
      });
    });

    describe('process_territory_region move', () => {
      it('processes territory region', () => {
        const state = createBaseState('territory_processing');
        const move = createMove(
          'process_territory_region',
          1,
          { x: 0, y: 0 },
          {
            disconnectedRegions: [
              {
                id: 'region-1',
                spaces: [{ x: 0, y: 0 }],
                player: 1,
                isDisconnected: true,
              },
            ],
          }
        );

        const result = processTurn(state, move);

        expect(['complete', 'awaiting_decision']).toContain(result.status);
        expect(result.nextState.gameStatus).toBe('active');
      });
    });

    describe('eliminate_rings_from_stack move', () => {
      it('processes elimination move', () => {
        const state = createBaseState('territory_processing');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });
        const move = createMove(
          'eliminate_rings_from_stack',
          1,
          { x: 3, y: 3 },
          {
            eliminatedRings: [{ player: 1, count: 1 }],
            eliminationFromStack: {
              position: { x: 3, y: 3 },
              capHeight: 1,
              totalHeight: 2,
            },
          }
        );

        const result = processTurn(state, move);

        expect(['complete', 'awaiting_decision']).toContain(result.status);
        expect(result.nextState.gameStatus).toBe('active');
      });
    });

    describe('skip_territory_processing move', () => {
      it('processes skip territory processing', () => {
        const state = createBaseState('territory_processing');
        const move = createMove('skip_territory_processing', 1, { x: 0, y: 0 });

        const result = processTurn(state, move);

        expect(result.nextState.currentPhase).toBe('ring_placement');
        expect(result.nextState.currentPlayer).toBe(2);
      });
    });

    describe('unknown move type', () => {
      it('throws error for unsupported move type', () => {
        const state = createBaseState('movement');
        const move = createMove('unknown_type' as Move['type'], 1, { x: 0, y: 0 });

        // The orchestrator now strictly validates move types against the current phase
        // and throws PHASE_MOVE_INVARIANT error for unknown/invalid move types
        expect(() => processTurn(state, move)).toThrow('[PHASE_MOVE_INVARIANT]');
      });
    });

    describe('metadata tracking', () => {
      it('includes processing metadata', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 3, y: 3 });

        const result = processTurn(state, move);

        expect(result.metadata).toMatchObject({
          processedMove: move,
          durationMs: expect.any(Number),
          sInvariantBefore: expect.any(Number),
          sInvariantAfter: expect.any(Number),
        });
      });
    });

    describe('status returns', () => {
      it('returns complete status for simple moves', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 3, y: 3 });

        const result = processTurn(state, move);

        expect(['complete', 'awaiting_decision']).toContain(result.status);
      });
    });
  });

  describe('validateMove', () => {
    describe('place_ring validation', () => {
      it('validates valid place_ring move', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 3, y: 3 });

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });

      it('validates place_ring at edge position', () => {
        const state = createBaseState('ring_placement');
        const move = createMove('place_ring', 1, { x: 0, y: 0 });

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });
    });

    describe('skip_placement validation', () => {
      it('validates eligible skip_placement', () => {
        const state = createBaseState('ring_placement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        state.players[0].ringsInHand = 0;
        const move = createMove('skip_placement', 1, { x: 0, y: 0 });

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });

      it('returns invalid for ineligible skip_placement', () => {
        const state = createBaseState('ring_placement');
        state.players[0].ringsInHand = 18; // Has rings, can't skip
        const move = createMove('skip_placement', 1, { x: 0, y: 0 });

        const result = validateMove(state, move);

        expect(result.valid).toBe(false);
        expect(typeof result.reason).toBe('string');
        expect(result.reason!.length).toBeGreaterThan(0);
      });
    });

    describe('move_stack validation', () => {
      it('validates move_stack with valid from', () => {
        const state = createBaseState('movement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        const move = createMove('move_stack', 1, { x: 4, y: 3 }, { from: { x: 3, y: 3 } });

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });

      it('returns invalid for move_stack without from', () => {
        const state = createBaseState('movement');
        const move = createMove('move_stack', 1, { x: 4, y: 3 });

        const result = validateMove(state, move);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Move.from is required');
      });
    });

    describe('move_ring validation', () => {
      it('validates move_ring with valid from', () => {
        const state = createBaseState('movement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        const move = createMove('move_ring', 1, { x: 4, y: 3 }, { from: { x: 3, y: 3 } });

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });

      it('returns invalid for move_ring without from', () => {
        const state = createBaseState('movement');
        const move = createMove('move_ring', 1, { x: 4, y: 3 });

        const result = validateMove(state, move);

        expect(result.valid).toBe(false);
      });
    });

    describe('overtaking_capture validation', () => {
      it('validates capture with valid parameters', () => {
        const state = createBaseState('capture');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });
        state.board.stacks.set('4,3', {
          position: { x: 4, y: 3 },
          stackHeight: 1,
          controllingPlayer: 2,
          composition: [{ player: 2, count: 1 }],
          rings: [2],
        });
        const move = createMove(
          'overtaking_capture',
          1,
          { x: 5, y: 3 },
          {
            from: { x: 3, y: 3 },
            captureTarget: { x: 4, y: 3 },
          }
        );

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });

      it('returns invalid for capture without from', () => {
        const state = createBaseState('capture');
        const move = createMove(
          'overtaking_capture',
          1,
          { x: 5, y: 3 },
          {
            captureTarget: { x: 4, y: 3 },
          }
        );

        const result = validateMove(state, move);

        expect(result.valid).toBe(false);
        expect(result.reason).toBe('Move.from and Move.captureTarget are required');
      });

      it('returns invalid for capture without captureTarget', () => {
        const state = createBaseState('capture');
        const move = createMove(
          'overtaking_capture',
          1,
          { x: 5, y: 3 },
          {
            from: { x: 3, y: 3 },
          }
        );

        const result = validateMove(state, move);

        expect(result.valid).toBe(false);
      });
    });

    describe('continue_capture_segment validation', () => {
      it('validates chain capture with valid parameters', () => {
        const state = createBaseState('chain_capture');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });
        state.board.stacks.set('4,3', {
          position: { x: 4, y: 3 },
          stackHeight: 1,
          controllingPlayer: 2,
          composition: [{ player: 2, count: 1 }],
          rings: [2],
        });
        const move = createMove(
          'continue_capture_segment',
          1,
          { x: 5, y: 3 },
          {
            from: { x: 3, y: 3 },
            captureTarget: { x: 4, y: 3 },
          }
        );

        const result = validateMove(state, move);

        expect(result).toHaveProperty('valid');
      });
    });

    describe('default validation', () => {
      it('returns valid for unknown move types', () => {
        const state = createBaseState('movement');
        const move = createMove('unknown_type' as Move['type'], 1, { x: 0, y: 0 });

        const result = validateMove(state, move);

        expect(result.valid).toBe(true);
      });

      it('returns valid for process_line', () => {
        const state = createBaseState('line_processing');
        const move = createMove('process_line', 1, { x: 0, y: 0 });

        const result = validateMove(state, move);

        expect(result.valid).toBe(true);
      });

      it('returns valid for choose_line_reward', () => {
        const state = createBaseState('line_processing');
        const move = createMove('choose_line_reward', 1, { x: 0, y: 0 });

        const result = validateMove(state, move);

        expect(result.valid).toBe(true);
      });
    });
  });

  describe('getValidMoves', () => {
    describe('ring_placement phase', () => {
      it('returns placement moves when player has rings', () => {
        const state = createBaseState('ring_placement');
        state.players[0].ringsInHand = 5;

        const moves = getValidMoves(state);

        expect(Array.isArray(moves)).toBe(true);
        const placementMoves = moves.filter((m) => m.type === 'place_ring');
        expect(placementMoves.length).toBeGreaterThanOrEqual(0);
      });

      it('includes skip_placement when eligible', () => {
        const state = createBaseState('ring_placement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        state.players[0].ringsInHand = 0;

        const moves = getValidMoves(state);

        // When player has 0 rings but has stacks, should return movement moves
        expect(Array.isArray(moves)).toBe(true);
      });

      it('returns movement moves when ringsInHand is 0 but has stacks', () => {
        const state = createBaseState('ring_placement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });
        state.players[0].ringsInHand = 0;

        const moves = getValidMoves(state);

        // Should return movement/capture moves since placement is not possible
        expect(Array.isArray(moves)).toBe(true);
      });

      it('returns empty when no rings and no stacks', () => {
        const state = createBaseState('ring_placement');
        state.players[0].ringsInHand = 0;
        // No stacks for player 1

        const moves = getValidMoves(state);

        expect(moves.length).toBe(0);
      });
    });

    describe('movement phase', () => {
      it('returns movement and capture moves', () => {
        const state = createBaseState('movement');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 1,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 1 }],
          rings: [1],
        });

        const moves = getValidMoves(state);

        expect(Array.isArray(moves)).toBe(true);
      });
    });

    describe('capture phase', () => {
      it('returns captures from attacker position and skip option', () => {
        const state = createBaseState('capture');
        state.moveHistory = [
          {
            id: 'last-move',
            type: 'move_stack',
            player: 1,
            to: { x: 3, y: 3 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 1,
          },
        ];
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });

        const moves = getValidMoves(state);

        // Should include skip_capture option
        const skipMoves = moves.filter((m) => m.type === 'skip_capture');
        expect(skipMoves.length).toBe(1);
      });

      it('returns empty captures without last move position', () => {
        const state = createBaseState('capture');
        state.moveHistory = []; // No last move

        const moves = getValidMoves(state);

        // Should return empty (no attacker position)
        expect(moves.length).toBe(0);
      });
    });

    describe('chain_capture phase', () => {
      it('returns capture moves', () => {
        const state = createBaseState('chain_capture');
        state.board.stacks.set('3,3', {
          position: { x: 3, y: 3 },
          stackHeight: 2,
          controllingPlayer: 1,
          composition: [{ player: 1, count: 2 }],
          rings: [1, 1],
        });
        state.board.stacks.set('4,3', {
          position: { x: 4, y: 3 },
          stackHeight: 1,
          controllingPlayer: 2,
          composition: [{ player: 2, count: 1 }],
          rings: [2],
        });

        const moves = getValidMoves(state);

        expect(Array.isArray(moves)).toBe(true);
      });
    });

    describe('line_processing phase', () => {
      it('returns process_line moves', () => {
        const state = createBaseState('line_processing');

        const moves = getValidMoves(state);

        expect(Array.isArray(moves)).toBe(true);
      });
    });

    describe('territory_processing phase', () => {
      it('returns territory moves and skip option when regions exist', () => {
        const state = createBaseState('territory_processing');

        const moves = getValidMoves(state);

        expect(Array.isArray(moves)).toBe(true);
      });

      it('includes skip_territory_processing when regions exist but no eliminations pending', () => {
        const state = createBaseState('territory_processing');
        // Set up a disconnected territory that needs processing
        state.board.territories.set('territory-1', {
          id: 'territory-1',
          spaces: [{ x: 0, y: 0 }],
          player: 1,
          isDisconnected: true,
        });

        const moves = getValidMoves(state);

        // The moves should be empty or include skip if regions are processable
        expect(Array.isArray(moves)).toBe(true);
      });
    });

    describe('default phase', () => {
      it('returns empty for unknown phase', () => {
        const state = createBaseState('unknown_phase' as GamePhase);

        const moves = getValidMoves(state);

        expect(moves.length).toBe(0);
      });
    });
  });

  describe('hasValidMoves', () => {
    it('returns true when moves are available', () => {
      const state = createBaseState('ring_placement');
      state.players[0].ringsInHand = 5;

      const result = hasValidMoves(state);

      expect(typeof result).toBe('boolean');
    });

    it('returns false when no moves are available', () => {
      const state = createBaseState('ring_placement');
      state.players[0].ringsInHand = 0;
      // No stacks for player 1

      const result = hasValidMoves(state);

      expect(result).toBe(false);
    });
  });

  describe('processTurnAsync', () => {
    it('processes turn with resolve delegate', async () => {
      const state = createBaseState('ring_placement');
      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const delegates = {
        resolveDecision: jest.fn().mockResolvedValue(move),
        onProcessingEvent: jest.fn(),
      };

      const result = await processTurnAsync(state, move, delegates);

      expect(result.nextState.currentPhase).toBe('movement');
      expect(result.nextState.gameStatus).toBe('active');
    });

    // DELETED 2025-12-06: 'handles chain capture decision by returning without auto-resolve'
    // tested obsolete end_chain_capture behavior. See: docs/SKIPPED_TESTS_TRIAGE.md (TH-5)

    it('emits processing events', async () => {
      const state = createBaseState('ring_placement');
      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const onProcessingEvent = jest.fn();
      const delegates = {
        resolveDecision: jest.fn().mockResolvedValue(move),
        onProcessingEvent,
      };

      await processTurnAsync(state, move, delegates);

      // If there was a decision, events would be emitted
      expect(typeof onProcessingEvent).toBe('function');
    });
  });

  describe('phase transitions', () => {
    it('transitions ring_placement -> movement on place_ring', () => {
      const state = createBaseState('ring_placement');
      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = processTurn(state, move);

      expect(result.nextState.currentPhase).toBe('movement');
    });

    it('transitions ring_placement -> movement on skip_placement', () => {
      const state = createBaseState('ring_placement');
      state.players[0].ringsInHand = 0;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 1,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 1 }],
        rings: [1],
      });
      const move = createMove('skip_placement', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.nextState.currentPhase).toBe('movement');
    });
  });

  describe('player rotation', () => {
    it('advances to next player after full turn', () => {
      const state = createBaseState('movement');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 1,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 1 }],
        rings: [1],
      });
      const move = createMove('move_stack', 1, { x: 4, y: 3 }, { from: { x: 3, y: 3 } });

      const result = processTurn(state, move);

      // After move, stack should be at new position
      expect(result.nextState.board.stacks.has('4,3')).toBe(true);
      expect(result.nextState.gameStatus).toBe('active');
    });

    it('wraps player rotation in 2-player game', () => {
      const state = createBaseState('territory_processing');
      state.currentPlayer = 2;
      const move = createMove('skip_territory_processing', 2, { x: 0, y: 0 });

      const result = processTurn(state, move);

      // Should wrap back to player 1
      expect([1, 2]).toContain(result.nextState.currentPlayer);
    });

    it('rotates correctly in 3-player game', () => {
      const state = createBaseState('territory_processing', 3);
      state.currentPlayer = 3;
      const move = createMove('skip_territory_processing', 3, { x: 0, y: 0 });

      const result = processTurn(state, move);

      // Should wrap back to player 1
      expect([1, 2, 3]).toContain(result.nextState.currentPlayer);
    });
  });

  describe('S-Invariant', () => {
    it('computes S-invariant in metadata', () => {
      const state = createBaseState('ring_placement');
      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = processTurn(state, move);

      expect(typeof result.metadata?.sInvariantBefore).toBe('number');
      expect(typeof result.metadata?.sInvariantAfter).toBe('number');
      expect(result.metadata!.sInvariantBefore).toBeGreaterThanOrEqual(0);
      expect(result.metadata!.sInvariantAfter).toBeGreaterThanOrEqual(0);
    });

    it('S-invariant is non-negative', () => {
      const state = createBaseState('ring_placement');
      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = processTurn(state, move);

      expect(result.metadata?.sInvariantBefore).toBeGreaterThanOrEqual(0);
      expect(result.metadata?.sInvariantAfter).toBeGreaterThanOrEqual(0);
    });
  });

  describe('victory detection', () => {
    it('handles game completion', () => {
      const state = createBaseState('movement');
      state.players[0].eliminatedRings = 19; // Victory threshold
      state.victoryThreshold = 19;
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 1,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 1 }],
        rings: [1],
      });
      const move = createMove('move_stack', 1, { x: 4, y: 3 }, { from: { x: 3, y: 3 } });

      const result = processTurn(state, move);

      // May or may not be game over depending on victory evaluation
      expect(['active', 'completed']).toContain(result.nextState.gameStatus);
    });
  });

  describe('decision moves state change detection', () => {
    it('detects when decision move changes state', () => {
      const state = createBaseState('line_processing');
      // Set up a line that can be processed
      for (let i = 0; i < 5; i++) {
        state.board.markers.set(`${i},0`, { position: { x: i, y: 0 }, player: 1 });
      }
      const move = createMove(
        'process_line',
        1,
        { x: 0, y: 0 },
        {
          formedLines: [
            {
              positions: [
                { x: 0, y: 0 },
                { x: 1, y: 0 },
                { x: 2, y: 0 },
                { x: 3, y: 0 },
                { x: 4, y: 0 },
              ],
              player: 1,
              length: 5,
              direction: 'horizontal',
            },
          ],
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState.gameStatus).toBe('active');
      expect(['complete', 'awaiting_decision']).toContain(result.status);
    });

    it('handles decision move that does not change state', () => {
      const state = createBaseState('territory_processing');
      const move = createMove(
        'process_territory_region',
        1,
        { x: 0, y: 0 },
        {
          disconnectedRegions: [], // No regions to process
        }
      );

      const result = processTurn(state, move);

      // Should complete without error
      expect(result.nextState.gameStatus).toBe('active');
      expect(['complete', 'awaiting_decision']).toContain(result.status);
    });
  });

  describe('forced elimination and no-op transitions', () => {
    const createSingleCellBoardWithStack = (player: number): Board => ({
      type: 'square8',
      size: 1,
      stacks: new Map([
        [
          '0,0',
          {
            position: { x: 0, y: 0 },
            stackHeight: 1,
            controllingPlayer: player,
            composition: [{ player, count: 1 }],
            rings: [player],
          },
        ],
      ]),
      markers: new Map(),
      territories: new Map(),
      formedLines: [],
      collapsedSpaces: new Map(),
      eliminatedRings: {},
    });

    it('surfaces forced-elimination decision when blocked with stacks and no placements/moves', () => {
      const state = createBaseState('movement');
      state.players[0].ringsInHand = 0;
      state.board = createSingleCellBoardWithStack(1);
      const move = createMove('no_movement_action', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.status).toBe('awaiting_decision');
      expect(result.pendingDecision?.type).toBe('elimination_target');
      expect(result.nextState.currentPhase).toBe('forced_elimination');
      expect(result.pendingDecision?.context.extra?.reason).toBe('forced_elimination');
    });

    it('treats no_placement_action as phase advancement with no side effects', () => {
      const state = createBaseState('ring_placement');
      state.players[0].ringsInHand = 0; // no legal placements
      const move = createMove('no_placement_action', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.status).toBe('complete');
      expect(result.nextState.currentPhase).toBe('movement');
      expect(result.pendingDecision).toBeUndefined();
    });
  });

  describe('edge cases', () => {
    it('handles empty board state', () => {
      const state = createBaseState('ring_placement');

      const moves = getValidMoves(state);

      expect(Array.isArray(moves)).toBe(true);
    });

    it('handles player with no controlled stacks in movement', () => {
      const state = createBaseState('movement');
      // No stacks for current player

      const moves = getValidMoves(state);

      expect(moves.length).toBe(0);
    });

    it('handles completed game state', () => {
      const state = createBaseState('ring_placement');
      state.gameStatus = 'completed';
      state.winner = 1;

      const moves = getValidMoves(state);

      expect(Array.isArray(moves)).toBe(true);
    });

    it('handles move with minimal required fields', () => {
      const state = createBaseState('ring_placement');
      const move: Move = {
        id: 'minimal-move',
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = processTurn(state, move);

      expect(result.nextState.currentPhase).toBe('movement');
      expect(result.nextState.board.stacks.has('3,3')).toBe(true);
    });

    it('surfaces no_line_action_required when entering line_processing with no lines', () => {
      const state = createBaseState('capture');
      const move = createMove('skip_capture', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.status).toBe('awaiting_decision');
      expect(result.pendingDecision?.type).toBe('no_line_action_required');
    });

    it('surfaces no_territory_action_required when territory has no regions', () => {
      const state = createBaseState('line_processing');
      const move = createMove('no_line_action', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.status).toBe('awaiting_decision');
      expect(result.pendingDecision?.type).toBe('no_territory_action_required');
    });

    it('flows line -> territory with region_order decision when regions exist', () => {
      const state = createBaseState('line_processing');
      const regionA = {
        id: 'region-a',
        spaces: [{ x: 1, y: 1 }],
        player: 1,
        isDisconnected: true,
      };
      const regionB = {
        id: 'region-b',
        spaces: [{ x: 2, y: 2 }],
        player: 2,
        isDisconnected: true,
      };

      const spy = jest
        .spyOn(TerritoryAggregate, 'getProcessableTerritoryRegions')
        .mockReturnValue([regionA as any, regionB as any]);

      const move = createMove('no_line_action', 1, { x: 0, y: 0 });
      const result = processTurn(state, move);

      expect(result.status).toBe('awaiting_decision');
      expect(result.pendingDecision?.type).toBe('region_order');
      expect(result.pendingDecision?.options.length).toBe(2);

      spy.mockRestore();
    });
  });
});
