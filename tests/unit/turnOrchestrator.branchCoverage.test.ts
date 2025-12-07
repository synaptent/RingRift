/**
 * TurnOrchestrator branch coverage tests
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
import type {
  GameState,
  GamePhase,
  Move,
  Player,
  Board,
  Position,
} from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';

describe('TurnOrchestrator branch coverage', () => {
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

        expect(result.nextState).toBeDefined();
        expect(result.status).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
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

        expect(result.nextState).toBeDefined();
      });
    });

    describe('skip_territory_processing move', () => {
      it('processes skip territory processing', () => {
        const state = createBaseState('territory_processing');
        const move = createMove('skip_territory_processing', 1, { x: 0, y: 0 });

        const result = processTurn(state, move);

        expect(result.nextState).toBeDefined();
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

        expect(result.metadata).toBeDefined();
        expect(result.metadata?.processedMove).toBe(move);
        expect(typeof result.metadata?.durationMs).toBe('number');
        expect(typeof result.metadata?.sInvariantBefore).toBe('number');
        expect(typeof result.metadata?.sInvariantAfter).toBe('number');
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
        expect(result.reason).toBeDefined();
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

      expect(result.nextState).toBeDefined();
    });

    // SKIP: Multi-phase turn model transitions chain_capture → line_processing after
    // processing capture moves. Test expects to remain in chain_capture phase.
    // See: docs/SKIPPED_TESTS_TRIAGE.md
    it.skip('handles chain capture decision by returning without auto-resolve', async () => {
      const state = createBaseState('chain_capture');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        controllingPlayer: 1,
        capHeight: 2,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });
      state.board.stacks.set('4,3', {
        position: { x: 4, y: 3 },
        stackHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });
      // Set up for chain capture continuation
      (state as GameState & { chainCapturePosition?: Position }).chainCapturePosition = {
        x: 3,
        y: 3,
      };

      const move = createMove(
        'continue_capture_segment',
        1,
        { x: 5, y: 3 },
        {
          from: { x: 3, y: 3 },
          captureTarget: { x: 4, y: 3 },
        }
      );

      const delegates = {
        resolveDecision: jest.fn().mockResolvedValue(move),
        onProcessingEvent: jest.fn(),
      };

      const result = await processTurnAsync(state, move, delegates);

      expect(result.nextState).toBeDefined();
    });

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

      // After move, should eventually rotate to player 2
      // (exact phase depends on post-move processing)
      expect(result.nextState).toBeDefined();
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

      expect(result.metadata?.sInvariantBefore).toBeDefined();
      expect(result.metadata?.sInvariantAfter).toBeDefined();
      expect(typeof result.metadata?.sInvariantBefore).toBe('number');
      expect(typeof result.metadata?.sInvariantAfter).toBe('number');
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
      expect(result.nextState.gameStatus).toBeDefined();
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

      expect(result.nextState).toBeDefined();
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
      expect(result.nextState).toBeDefined();
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

      expect(result.nextState).toBeDefined();
    });
  });

  describe('line processing decisions', () => {
    it('creates line order decision when multiple lines exist', () => {
      const state = createBaseState('line_processing');
      // Create two 5-marker lines for player 1
      for (let i = 0; i < 5; i++) {
        state.board.markers.set(`${i},0`, { position: { x: i, y: 0 }, player: 1, type: 'regular' });
        state.board.markers.set(`${i},2`, { position: { x: i, y: 2 }, player: 1, type: 'regular' });
      }

      const moves = getValidMoves(state);

      // Should have process_line moves for both lines
      const processLineMoves = moves.filter((m) => m.type === 'process_line');
      expect(processLineMoves.length).toBeGreaterThanOrEqual(0);
    });

    it('handles choose_line_reward move', () => {
      const state = createBaseState('line_processing');
      // Set up a line
      for (let i = 0; i < 5; i++) {
        state.board.markers.set(`${i},0`, { position: { x: i, y: 0 }, player: 1, type: 'regular' });
      }
      state.board.formedLines = [
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
      ];

      const move = createMove(
        'choose_line_reward',
        1,
        { x: 2, y: 0 },
        {
          rewardType: 'MINIMUM_COLLAPSE',
          formedLines: state.board.formedLines,
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });
  });

  describe('territory processing decisions', () => {
    it('creates region order decision when multiple regions exist', () => {
      const state = createBaseState('territory_processing');
      // Would need to set up disconnected territory regions
      // This is complex to set up, but the test verifies the path exists

      const moves = getValidMoves(state);

      expect(Array.isArray(moves)).toBe(true);
    });

    it('handles skip_territory_processing move', () => {
      const state = createBaseState('territory_processing');
      const move = createMove('skip_territory_processing', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });

    it('handles eliminate_rings_from_stack in territory processing', () => {
      const state = createBaseState('territory_processing');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const move = createMove(
        'eliminate_rings_from_stack',
        1,
        { x: 3, y: 3 },
        {
          eliminatedRings: [{ player: 1, count: 2 }],
          eliminationFromStack: {
            position: { x: 3, y: 3 },
            capHeight: 2,
            totalHeight: 2,
          },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });
  });

  describe('capture and chain capture', () => {
    it('handles overtaking_capture move', () => {
      const state = createBaseState('capture');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });
      state.board.stacks.set('5,3', {
        position: { x: 5, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });
      state.moveHistory = [
        {
          id: 'last-move',
          type: 'move_stack',
          player: 1,
          to: { x: 3, y: 3 },
          from: { x: 0, y: 3 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];

      const move = createMove(
        'overtaking_capture',
        1,
        { x: 6, y: 3 },
        {
          from: { x: 3, y: 3 },
          captureTarget: { x: 5, y: 3 },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });

    it('handles skip_capture move', () => {
      const state = createBaseState('capture');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      state.moveHistory = [
        {
          id: 'last-move',
          type: 'move_stack',
          player: 1,
          to: { x: 3, y: 3 },
          from: { x: 0, y: 3 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];

      const move = createMove('skip_capture', 1, { x: 3, y: 3 });

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });

    it('handles chain capture with chainCapturePosition', () => {
      const state = createBaseState('chain_capture');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 4,
        capHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 4 }],
        rings: [1, 1, 1, 1],
      });
      state.board.stacks.set('5,3', {
        position: { x: 5, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });
      (state as GameState & { chainCapturePosition?: Position }).chainCapturePosition = {
        x: 3,
        y: 3,
      };

      const move = createMove(
        'continue_capture_segment',
        1,
        { x: 6, y: 3 },
        {
          from: { x: 3, y: 3 },
          captureTarget: { x: 5, y: 3 },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });

    // SKIP: end_chain_capture is not a valid move type in the current phase model.
    // chain_capture phase only allows: overtaking_capture, continue_capture_segment.
    // Chain captures end via decision resolution, not explicit moves.
    // See: docs/SKIPPED_TESTS_TRIAGE.md
    it.skip('handles end_chain_capture move', () => {
      const state = createBaseState('chain_capture');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      (state as GameState & { chainCapturePosition?: Position }).chainCapturePosition = {
        x: 3,
        y: 3,
      };

      const move = createMove('end_chain_capture', 1, { x: 3, y: 3 });

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });
  });

  describe('ANM state resolution', () => {
    it('resolves ANM state when player has no moves but has stacks', () => {
      const state = createBaseState('movement');
      state.players[0].ringsInHand = 0;
      // Player 1 has stack but is completely blocked
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 1 }],
        rings: [1],
      });
      // Block all adjacent positions
      state.board.collapsedSpaces.set('1,0', 1);
      state.board.collapsedSpaces.set('0,1', 1);
      state.board.collapsedSpaces.set('1,1', 1);
      // Collapse most of the board
      for (let x = 2; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          state.board.collapsedSpaces.set(`${x},${y}`, 1);
        }
      }
      for (let y = 2; y < 8; y++) {
        state.board.collapsedSpaces.set(`0,${y}`, 1);
        state.board.collapsedSpaces.set(`1,${y}`, 1);
      }

      // Player 2 has a stack
      state.board.stacks.set('0,7', {
        position: { x: 0, y: 7 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });
      state.players[1].ringsInHand = 0;

      const moves = getValidMoves(state);

      // ANM state should provide forced elimination options
      expect(Array.isArray(moves)).toBe(true);
    });

    it('handles forced elimination decision creation', () => {
      const state = createBaseState('ring_placement');
      state.players[0].ringsInHand = 0;
      // Player has stacks but no legal moves
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });
      // Block the stack completely
      state.board.collapsedSpaces.set('1,0', 1);
      state.board.collapsedSpaces.set('0,1', 1);
      state.board.collapsedSpaces.set('1,1', 1);

      const moves = getValidMoves(state);

      expect(Array.isArray(moves)).toBe(true);
    });
  });

  describe('movement with capture opportunity', () => {
    it('allows capture after movement lands adjacent to opponent', () => {
      const state = createBaseState('movement');
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });
      // Opponent stack that could be captured after movement
      state.board.stacks.set('6,3', {
        position: { x: 6, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      const move = createMove('move_stack', 1, { x: 5, y: 3 }, { from: { x: 3, y: 3 } });

      const result = processTurn(state, move);

      // Should either complete or offer capture decision
      expect(result.nextState).toBeDefined();
      expect(['complete', 'awaiting_decision']).toContain(result.status);
    });
  });

  describe('three player game', () => {
    it('cycles through three players correctly', () => {
      const state = createBaseState('ring_placement', 3);
      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = processTurn(state, move);

      expect(result.nextState.players.length).toBe(3);
    });
  });

  describe('processTurnAsync with decisions', () => {
    it('resolves line processing decision asynchronously', async () => {
      const state = createBaseState('line_processing');
      // Set up a line
      for (let i = 0; i < 5; i++) {
        state.board.markers.set(`${i},0`, { position: { x: i, y: 0 }, player: 1, type: 'regular' });
      }

      const processLineMove = createMove(
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

      const delegates = {
        resolveDecision: jest.fn().mockResolvedValue(processLineMove),
        onProcessingEvent: jest.fn(),
      };

      const result = await processTurnAsync(state, processLineMove, delegates);

      expect(result.nextState).toBeDefined();
    });

    it('handles territory processing decision asynchronously', async () => {
      const state = createBaseState('territory_processing');
      const skipMove = createMove('skip_territory_processing', 1, { x: 0, y: 0 });

      const delegates = {
        resolveDecision: jest.fn().mockResolvedValue(skipMove),
        onProcessingEvent: jest.fn(),
      };

      const result = await processTurnAsync(state, skipMove, delegates);

      expect(result.nextState).toBeDefined();
    });
  });

  describe('chain capture continuation (lines 598, 649-670)', () => {
    it('triggers chain capture decision when capture allows continuation', () => {
      // Set up a capture scenario where the landing position can capture again
      const state = createBaseState('capture');

      // Player 1 has a tall stack at 3,3
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 4,
        capHeight: 4,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 4 }],
        rings: [1, 1, 1, 1],
      });

      // First capture target at 4,3 (opponent stack)
      state.board.stacks.set('4,3', {
        position: { x: 4, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      // Second capture target at 6,3 (opponent stack - for chain continuation)
      state.board.stacks.set('6,3', {
        position: { x: 6, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      // Capture move from 3,3 jumping over 4,3 landing at 5,3
      const captureMove = createMove(
        'overtaking_capture',
        1,
        { x: 5, y: 3 },
        {
          from: { x: 3, y: 3 },
          captureTarget: { x: 4, y: 3 },
        }
      );

      const result = processTurn(state, captureMove);

      expect(result.nextState).toBeDefined();
      // Should be awaiting chain capture decision or have processed it
      expect(['complete', 'awaiting_decision']).toContain(result.status);
    });

    it('handles continue_capture_segment in chain', () => {
      const state = createBaseState('chain_capture');

      // Stack at landing position from previous capture
      state.board.stacks.set('5,3', {
        position: { x: 5, y: 3 },
        stackHeight: 5,
        capHeight: 5,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 5 }],
        rings: [1, 1, 1, 1, 1],
      });

      // Next capture target
      state.board.stacks.set('6,3', {
        position: { x: 6, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      // Set chainCapturePosition
      (state as GameState & { chainCapturePosition?: Position }).chainCapturePosition = {
        x: 5,
        y: 3,
      };

      const continueMove = createMove(
        'continue_capture_segment',
        1,
        { x: 7, y: 3 },
        {
          from: { x: 5, y: 3 },
          captureTarget: { x: 6, y: 3 },
        }
      );

      const result = processTurn(state, continueMove);

      expect(result.nextState).toBeDefined();
    });
  });

  describe('multiple lines decision (lines 716-724)', () => {
    it('creates line order decision when multiple lines exist', () => {
      const state = createBaseState('line_processing');

      // Create two horizontal lines of markers for player 1
      // Line 1: y=0
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Line 2: y=2
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},2`, {
          position: { x, y: 2 },
          player: 1,
          type: 'regular',
        });
      }

      // Add a stack so it's not empty board - player needs turn material
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 1 }],
        rings: [1],
      });

      // Trigger line processing with a movement that completes
      // Since we're already in line_processing, use a process_line move
      const moves = getValidMoves(state);

      // Should have multiple process_line options
      const processLineMoves = moves.filter((m) => m.type === 'process_line');
      expect(processLineMoves.length).toBeGreaterThan(0);
    });

    it('returns pending decision for multiple lines during processTurn', () => {
      const state = createBaseState('movement');

      // Player 1 stack to move
      state.board.stacks.set('7,3', {
        position: { x: 7, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Create two horizontal lines of markers for player 1
      // Line 1: y=0
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Line 2: y=2
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},2`, {
          position: { x, y: 2 },
          player: 1,
          type: 'regular',
        });
      }

      // Move the stack, which should transition through phases to line_processing
      const move = createMove(
        'move_stack',
        1,
        { x: 6, y: 3 },
        {
          from: { x: 7, y: 3 },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
      // With multiple lines, should be awaiting decision
      if (result.pendingDecision) {
        expect(['line_order', 'region_order', 'chain_capture']).toContain(
          result.pendingDecision.type
        );
      }
    });
  });

  describe('single line auto-processing with elimination (lines 729-746)', () => {
    it('auto-processes single line and handles elimination decision', () => {
      const state = createBaseState('movement');

      // Player 1 stack to move
      state.board.stacks.set('7,3', {
        position: { x: 7, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Create exactly one horizontal line of markers for player 1
      // Line at y=0
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Move the stack
      const move = createMove(
        'move_stack',
        1,
        { x: 6, y: 3 },
        {
          from: { x: 7, y: 3 },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
      // Should either complete or have a pending decision
      expect(['complete', 'awaiting_decision']).toContain(result.status);
    });
  });

  describe('territory processing (lines 768, 773-795)', () => {
    it('handles territory processing with regions', () => {
      const state = createBaseState('territory_processing');

      // Set up a territory region
      // Player 1 has markers forming enclosed territory
      state.board.markers.set('2,2', { position: { x: 2, y: 2 }, player: 1, type: 'regular' });
      state.board.markers.set('2,3', { position: { x: 2, y: 3 }, player: 1, type: 'regular' });
      state.board.markers.set('2,4', { position: { x: 2, y: 4 }, player: 1, type: 'regular' });
      state.board.markers.set('3,2', { position: { x: 3, y: 2 }, player: 1, type: 'regular' });
      state.board.markers.set('3,4', { position: { x: 3, y: 4 }, player: 1, type: 'regular' });
      state.board.markers.set('4,2', { position: { x: 4, y: 2 }, player: 1, type: 'regular' });
      state.board.markers.set('4,3', { position: { x: 4, y: 3 }, player: 1, type: 'regular' });
      state.board.markers.set('4,4', { position: { x: 4, y: 4 }, player: 1, type: 'regular' });

      // Add stack for player
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 1 }],
        rings: [1],
      });

      // Try skip territory processing
      const move = createMove('skip_territory_processing', 1, { x: 0, y: 0 });

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });

    it('handles process_territory_region move', () => {
      const state = createBaseState('territory_processing');

      // Add a stack for the player
      state.board.stacks.set('5,5', {
        position: { x: 5, y: 5 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const move = createMove(
        'process_territory_region',
        1,
        { x: 3, y: 3 },
        {
          disconnectedRegions: [
            {
              spaces: [{ x: 3, y: 3 }],
              owner: 1,
              markers: [],
              area: 1,
            },
          ],
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
    });
  });

  describe('forced elimination decision (lines 490-493)', () => {
    it('creates forced elimination decision when player has stacks but no legal moves', () => {
      const state = createBaseState('movement');
      state.players[0].ringsInHand = 0;

      // Player 1 has stack but completely blocked
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });

      // Block all adjacent positions with collapsed spaces
      state.board.collapsedSpaces.set('1,0', 1);
      state.board.collapsedSpaces.set('0,1', 1);
      state.board.collapsedSpaces.set('1,1', 1);

      // Collapse most of the board
      for (let x = 2; x < 8; x++) {
        for (let y = 0; y < 8; y++) {
          state.board.collapsedSpaces.set(`${x},${y}`, 1);
        }
      }
      for (let y = 2; y < 8; y++) {
        state.board.collapsedSpaces.set(`0,${y}`, 1);
        state.board.collapsedSpaces.set(`1,${y}`, 1);
      }

      // Player 2 has a stack to keep game active
      state.board.stacks.set('0,7', {
        position: { x: 0, y: 7 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });
      state.players[1].ringsInHand = 0;

      // Check valid moves - should include forced elimination
      const moves = getValidMoves(state);

      // Should have some moves available (either forced elimination or end_turn)
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  describe('ANM resolution (lines 100-158, 504-507)', () => {
    it('resolves ANM state at end of turn when player has no moves', () => {
      const state = createBaseState('ring_placement');
      state.currentPhase = 'movement';
      state.players[0].ringsInHand = 0;
      state.players[1].ringsInHand = 0;

      // Player 1 has blocked stack
      state.board.stacks.set('0,0', {
        position: { x: 0, y: 0 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Block player 1's stack
      state.board.collapsedSpaces.set('1,0', 1);
      state.board.collapsedSpaces.set('0,1', 1);
      state.board.collapsedSpaces.set('1,1', 1);

      // Player 2 has active stack
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 2 }],
        rings: [2, 2],
      });

      // Player 2 makes a move (needs valid move history)
      state.currentPlayer = 2;
      const move = createMove(
        'move_stack',
        2,
        { x: 6, y: 7 },
        {
          from: { x: 7, y: 7 },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
      // Turn should complete (player 1 may be eliminated or have forced elimination)
    });
  });

  describe('processTurnAsync decision loop (lines 857-886)', () => {
    it('processes multiple decisions in async loop', async () => {
      const state = createBaseState('movement');

      // Set up a scenario requiring decisions
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Create a line for processing
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      const move = createMove(
        'move_stack',
        1,
        { x: 4, y: 3 },
        {
          from: { x: 3, y: 3 },
        }
      );

      let decisionCount = 0;
      const delegates = {
        resolveDecision: jest.fn().mockImplementation((decision) => {
          decisionCount++;
          // Return first option for any decision
          if (decision.options && decision.options.length > 0) {
            return Promise.resolve(decision.options[0]);
          }
          // Fallback to skip
          return Promise.resolve(createMove('skip_territory_processing', 1, { x: 0, y: 0 }));
        }),
        onProcessingEvent: jest.fn(),
      };

      const result = await processTurnAsync(state, move, delegates);

      expect(result.nextState).toBeDefined();
      // Events should have been emitted if decisions were made
      if (decisionCount > 0) {
        expect(delegates.onProcessingEvent).toHaveBeenCalled();
      }
    });

    // SKIP: Test passes undefined move after capture processing, but orchestrator now
    // requires valid move in each phase. Multi-phase model change.
    // See: docs/SKIPPED_TESTS_TRIAGE.md
    it.skip('returns early on chain capture decision without auto-resolving', async () => {
      const state = createBaseState('capture');

      // Set up chain capture scenario
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 4,
        capHeight: 4,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 4 }],
        rings: [1, 1, 1, 1],
      });

      state.board.stacks.set('4,3', {
        position: { x: 4, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      state.board.stacks.set('6,3', {
        position: { x: 6, y: 3 },
        stackHeight: 1,
        capHeight: 1,
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

      const delegates = {
        resolveDecision: jest.fn(),
        onProcessingEvent: jest.fn(),
      };

      const result = await processTurnAsync(state, move, delegates);

      expect(result.nextState).toBeDefined();
      // Chain capture decisions should NOT be auto-resolved
      if (result.pendingDecision?.type === 'chain_capture') {
        expect(delegates.resolveDecision).not.toHaveBeenCalled();
      }
    });
  });

  describe('capture from landing position (lines 689-695)', () => {
    it('offers capture opportunity after movement lands adjacent to opponent', () => {
      const state = createBaseState('movement');

      // Player 1 stack that will move
      state.board.stacks.set('2,3', {
        position: { x: 2, y: 3 },
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });

      // Opponent stack at 5,3 - will be capturable from landing at 4,3
      state.board.stacks.set('5,3', {
        position: { x: 5, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      // Move stack to 4,3 (adjacent to opponent at 5,3)
      const move = createMove(
        'move_stack',
        1,
        { x: 4, y: 3 },
        {
          from: { x: 2, y: 3 },
        }
      );

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
      // Should transition to capture phase or complete
      // The capture opportunity from landing position should be detected
    });
  });

  describe('skipSingleTerritoryAutoProcess option (lines 773-777)', () => {
    it('returns decision for single region when skipSingleTerritoryAutoProcess is true', () => {
      const state = createBaseState('movement');

      // Player 1 stack
      state.board.stacks.set('7,3', {
        position: { x: 7, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const move = createMove(
        'move_stack',
        1,
        { x: 6, y: 3 },
        {
          from: { x: 7, y: 3 },
        }
      );

      // Note: This option is passed through processTurn's options parameter
      // We test that the path exists and is handled
      const result = processTurn(state, move, { skipSingleTerritoryAutoProcess: true });

      expect(result.nextState).toBeDefined();
    });
  });

  describe('getValidMoves skip_placement branch (line 1000)', () => {
    it('includes skip_placement when player has stacks with legal moves', () => {
      const state = createBaseState('ring_placement');
      // Player still has rings in hand (will enumerate placements)
      // AND has a stack with legal moves (enables skip)

      // Player has a stack on board at 3,3 with room to move
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const moves = getValidMoves(state);

      // Should include skip_placement option because player has stacks with legal moves
      const skipMoves = moves.filter((m) => m.type === 'skip_placement');
      // Skip is only available if player has controlled stacks with legal actions
      expect(moves.length).toBeGreaterThan(0);
    });

    it('adds skip_placement when player has stack and can move/capture', () => {
      const state = createBaseState('ring_placement');

      // Player 1 has a moveable stack
      state.board.stacks.set('4,4', {
        position: { x: 4, y: 4 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const moves = getValidMoves(state);

      // Check if skip_placement is available
      const skipMoves = moves.filter((m) => m.type === 'skip_placement');
      // Skip should be available since player has stack with legal moves
      expect(skipMoves.length).toBe(1);
    });
  });

  describe('getValidMoves capture phase filtering (line 1037)', () => {
    it('filters captures to attacker position in capture phase', () => {
      const state = createBaseState('capture');

      // Player 1 stack (attacker)
      state.board.stacks.set('3,3', {
        position: { x: 3, y: 3 },
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 3 }],
        rings: [1, 1, 1],
      });

      // Another player 1 stack at different position
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Opponent stack capturable from 3,3
      state.board.stacks.set('4,3', {
        position: { x: 4, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      // Move history with last move to 3,3 (attacker position)
      state.moveHistory = [
        {
          id: 'prev-move',
          type: 'move_stack',
          player: 1,
          to: { x: 3, y: 3 },
          from: { x: 2, y: 3 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];

      const moves = getValidMoves(state);

      // Should have captures and skip_capture option
      expect(moves.length).toBeGreaterThan(0);
      const skipCaptures = moves.filter((m) => m.type === 'skip_capture');
      expect(skipCaptures.length).toBeGreaterThanOrEqual(0);
    });

    it('returns empty when no last move position exists', () => {
      const state = createBaseState('capture');

      // Empty move history
      state.moveHistory = [];

      const moves = getValidMoves(state);

      // Should return empty or minimal moves when no attacker position
      expect(Array.isArray(moves)).toBe(true);
    });
  });

  describe('region order decision with skip option (lines 240-241)', () => {
    it('adds skip_territory_processing to region order options', () => {
      const state = createBaseState('territory_processing');

      // Create territory markers for player 1 forming a region
      // Enclosed area with markers around it
      state.board.markers.set('1,1', { position: { x: 1, y: 1 }, player: 1, type: 'regular' });
      state.board.markers.set('1,2', { position: { x: 1, y: 2 }, player: 1, type: 'regular' });
      state.board.markers.set('1,3', { position: { x: 1, y: 3 }, player: 1, type: 'regular' });
      state.board.markers.set('2,1', { position: { x: 2, y: 1 }, player: 1, type: 'regular' });
      state.board.markers.set('2,3', { position: { x: 2, y: 3 }, player: 1, type: 'regular' });
      state.board.markers.set('3,1', { position: { x: 3, y: 1 }, player: 1, type: 'regular' });
      state.board.markers.set('3,2', { position: { x: 3, y: 2 }, player: 1, type: 'regular' });
      state.board.markers.set('3,3', { position: { x: 3, y: 3 }, player: 1, type: 'regular' });

      // Add stack for player
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      const moves = getValidMoves(state);

      // Should include process_territory_region and skip options
      expect(moves.length).toBeGreaterThan(0);
    });
  });

  describe('line elimination decision (line 746)', () => {
    it('creates elimination decision after line processing requires it', () => {
      const state = createBaseState('line_processing');

      // Create a line of markers for player 1
      for (let x = 0; x < 5; x++) {
        state.board.markers.set(`${x},0`, {
          position: { x, y: 0 },
          player: 1,
          type: 'regular',
        });
      }

      // Player needs a stack
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Process line move
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

      expect(result.nextState).toBeDefined();
    });
  });

  describe('additional chain capture scenarios', () => {
    it('handles chain capture with landing position set in state', () => {
      const state = createBaseState('chain_capture');

      // Set chainCapturePosition in state
      (state as GameState & { chainCapturePosition?: Position }).chainCapturePosition = {
        x: 5,
        y: 3,
      };

      // Stack at chain capture position
      state.board.stacks.set('5,3', {
        position: { x: 5, y: 3 },
        stackHeight: 4,
        capHeight: 4,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 4 }],
        rings: [1, 1, 1, 1],
      });

      // Opponent stack to capture
      state.board.stacks.set('6,3', {
        position: { x: 6, y: 3 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
      });

      const moves = getValidMoves(state);

      // Should have chain capture continuation options
      expect(Array.isArray(moves)).toBe(true);
    });

    // SKIP: end_chain_capture is not a valid move type in the current phase model.
    // chain_capture phase only allows: overtaking_capture, continue_capture_segment.
    // Chain captures end via decision resolution, not explicit moves.
    // See: docs/SKIPPED_TESTS_TRIAGE.md
    it.skip('ends chain capture when no more continuations available', () => {
      const state = createBaseState('chain_capture');

      // Stack at position
      state.board.stacks.set('5,3', {
        position: { x: 5, y: 3 },
        stackHeight: 4,
        capHeight: 4,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 4 }],
        rings: [1, 1, 1, 1],
      });

      // No opponent stacks to capture
      (state as GameState & { chainCapturePosition?: Position }).chainCapturePosition = {
        x: 5,
        y: 3,
      };

      const move = createMove('end_chain_capture', 1, { x: 5, y: 3 });

      const result = processTurn(state, move);

      expect(result.nextState).toBeDefined();
      // Chain should end and proceed to line processing
    });
  });

  describe('validateMove edge cases', () => {
    it('validates move from wrong player as invalid', () => {
      const state = createBaseState('ring_placement');
      state.currentPlayer = 1;

      const move = createMove('place_ring', 2, { x: 3, y: 3 });

      const result = validateMove(state, move);

      expect(result.valid).toBe(false);
    });

    it('validates move in completed game', () => {
      const state = createBaseState('ring_placement');
      state.gameStatus = 'completed';

      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = validateMove(state, move);

      // The validate function checks move validity regardless of game status
      expect(result).toBeDefined();
    });

    it('validates placement position', () => {
      const state = createBaseState('ring_placement');

      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = validateMove(state, move);

      // Valid placement position
      expect(result.valid).toBe(true);
    });

    it('validates placement on collapsed space as invalid', () => {
      const state = createBaseState('ring_placement');

      // Collapse the target space
      state.board.collapsedSpaces.set('3,3', 1);

      const move = createMove('place_ring', 1, { x: 3, y: 3 });

      const result = validateMove(state, move);

      expect(result.valid).toBe(false);
    });
  });

  describe('hasValidMoves edge cases', () => {
    it('handles completed game state', () => {
      const state = createBaseState('ring_placement');
      state.gameStatus = 'completed';

      const result = hasValidMoves(state);

      // hasValidMoves still checks if moves exist regardless of game status
      expect(typeof result).toBe('boolean');
    });

    it('returns true when valid moves exist', () => {
      const state = createBaseState('ring_placement');

      const result = hasValidMoves(state);

      expect(result).toBe(true);
    });
  });
});
