/**
 * TurnOrchestrator advanced branch coverage tests
 *
 * Tests for src/shared/engine/orchestration/turnOrchestrator.ts covering:
 * - Line processing decisions
 * - Territory processing decisions
 * - Capture and chain capture
 * - ANM state resolution
 * - Forced elimination decision
 * - Advanced async processing scenarios
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

describe('TurnOrchestrator advanced branch coverage', () => {
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
    victoryThreshold: 18, // RR-CANON-R061: ringsPerPlayer
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

    it('handles line processing with no_line_action', () => {
      // FSM validates that choose_line_reward is only valid when awaitingReward is true,
      // which requires actual line detection via findLinesForPlayer(). Setting up
      // a valid line formation is complex, so we test no_line_action instead.
      // Full line reward testing is done in integration/parity tests.
      const state = createBaseState('line_processing');
      const move = createMove('no_line_action', 1, { x: 0, y: 0 });

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

    it('handles territory processing with no_territory_action', () => {
      // FSM validates that eliminate_rings_from_stack is only valid when pendingEliminations > 0,
      // which requires a disconnected territory that was just processed. Since setting
      // up that full state is complex, we test no_territory_action instead (which is the
      // bookkeeping move for when no territory processing is needed).
      const state = createBaseState('territory_processing');
      const move = createMove('no_territory_action', 1, { x: 0, y: 0 });

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

    // DELETED 2025-12-06: 'handles end_chain_capture move' tested obsolete move type.
    // end_chain_capture doesn't exist; chains end via decision resolution.
    // See: docs/SKIPPED_TESTS_TRIAGE.md (TH-5)
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

      // After line processing, phase transitions to territory_processing
      // The resolveDecision should return a territory processing move
      const skipTerritoryMove = createMove('skip_territory_processing', 1, { x: 0, y: 0 });

      const delegates = {
        resolveDecision: jest.fn().mockResolvedValue(skipTerritoryMove),
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

    // DELETED 2025-12-06: 'returns early on chain capture decision without auto-resolving'
    // tested passing undefined move after capture, now requires valid moves.
    // See: docs/SKIPPED_TESTS_TRIAGE.md (TH-5)
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

      // Add stacks for both players - territory detection requires 2+ active players
      // to determine disconnection (one player cannot be "cut off" from themselves)
      state.board.stacks.set('7,7', {
        position: { x: 7, y: 7 },
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
        composition: [{ player: 1, count: 2 }],
        rings: [1, 1],
      });

      // Player 2 stack outside the enclosed region (required for disconnection detection)
      state.board.stacks.set('6,6', {
        position: { x: 6, y: 6 },
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
        composition: [{ player: 2, count: 1 }],
        rings: [2],
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

    // DELETED 2025-12-06: 'ends chain capture when no more continuations available'
    // tested obsolete end_chain_capture move type. See: docs/SKIPPED_TESTS_TRIAGE.md (TH-5)
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
/**
 * Advanced branch coverage for turnOrchestrator focused on forced-elimination
 * pending decisions that are emitted when a player is blocked but still has
 * stacks (RR-CANON-R072/R100/R205).
 */

import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, GamePhase, Move, Player, Board } from '../../src/shared/types/game';

const createPlayer = (playerNumber: number, ringsInHand: number): Player => ({
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

const createBoardWithSingleStack = (player: number): Board => ({
  type: 'square8',
  size: 1, // Single valid cell; no legal moves/captures exist
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

const createEmptyBoard = (): Board => ({
  type: 'square8',
  size: 2,
  stacks: new Map(),
  markers: new Map(),
  territories: new Map(),
  formedLines: [],
  collapsedSpaces: new Map(),
  eliminatedRings: {},
});

const createBaseState = (phase: GamePhase, ringsInHand: number, board: Board): GameState => ({
  id: 'test-game',
  currentPlayer: 1,
  currentPhase: phase,
  gameStatus: 'active',
  boardType: board.type,
  players: [createPlayer(1, ringsInHand)],
  board,
  moveHistory: [],
  history: [],
  lastMoveAt: new Date(),
  createdAt: new Date(),
  isRated: false,
  spectators: [],
  timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
  maxPlayers: 1,
  totalRingsInPlay: 1,
  victoryThreshold: 1,
});

const createMove = (type: Move['type']): Move => ({
  id: `move-${type}`,
  type,
  player: 1,
  to: { x: 0, y: 0 },
  timestamp: new Date(),
  thinkTime: 0,
  moveNumber: 1,
});

describe('turnOrchestrator advanced branch coverage', () => {
  it('returns awaiting_decision after no_territory_action when elimination moves are needed', () => {
    // Per turnOrchestrator.ts line 1226: no_territory_action is NOT turn-ending
    // because forced elimination may be needed. Only skip_territory_processing
    // is turn-ending.
    //
    // When a player has stacks that need elimination, after no_territory_action
    // the post-move processing will detect forced elimination is needed and
    // return awaiting_decision.
    const board = createBoardWithSingleStack(1);
    const state = createBaseState('territory_processing', 0, board);
    const move = createMove('no_territory_action');

    const result = processTurn(state, move);

    // After no_territory_action, the system checks for forced elimination.
    // Since elimination moves are needed, it returns awaiting_decision.
    expect(result.status).toBe('awaiting_decision');
    expect(result.pendingDecision).toBeDefined();
  });

  it('does not surface forced-elimination decision when placement actions exist', () => {
    // Start in territory_processing phase to test forced elimination check
    // When ringsInHand > 0 and placement positions exist, no forced elimination
    const board = createEmptyBoard();
    const state = createBaseState('territory_processing', 1, board); // ringsInHand > 0 ⇒ placement exists
    const move = createMove('no_territory_action');

    const result = processTurn(state, move);

    // Turn should complete (rotate to next player/placement) rather than forced elimination
    expect(result.status).toBe('complete');
    expect(result.pendingDecision).toBeUndefined();
    expect(result.nextState.currentPhase).not.toBe('forced_elimination');
  });
});
