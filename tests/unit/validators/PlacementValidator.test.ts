/**
 * PlacementValidator Unit Tests
 *
 * Tests the shared engine placement validation logic including:
 * - Board position validation
 * - Ring supply constraints
 * - Collapsed space and marker blocking
 * - Multi-ring placement on empty vs occupied cells
 * - No-dead-placement rule enforcement
 * - Skip placement validation
 */

import {
  validatePlacementOnBoard,
  validatePlacement,
  validateSkipPlacement,
  PlacementContext,
} from '../../../src/shared/engine/validators/PlacementValidator';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  pos,
  posStr,
} from '../../utils/fixtures';
import { BoardState, RingStack, MarkerInfo } from '../../../src/shared/types/game';
import { PlaceRingAction, SkipPlacementAction, GameState } from '../../../src/shared/engine/types';

describe('PlacementValidator', () => {
  describe('validatePlacementOnBoard', () => {
    let board: BoardState;
    let ctx: PlacementContext;

    beforeEach(() => {
      board = createTestBoard('square8');
      ctx = {
        boardType: 'square8',
        player: 1,
        ringsInHand: 18,
        ringsPerPlayerCap: 18,
      };
    });

    describe('basic validation', () => {
      it('allows valid placement on empty cell', () => {
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(true);
        expect(result.maxPlacementCount).toBe(3); // Empty cell allows up to 3
      });

      it('allows placing up to 3 rings on empty cell', () => {
        const result = validatePlacementOnBoard(board, pos(3, 3), 3, ctx);
        expect(result.valid).toBe(true);
      });

      it('rejects placing more than 3 rings on empty cell', () => {
        const result = validatePlacementOnBoard(board, pos(3, 3), 4, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_COUNT');
      });

      it('rejects placement when no rings in hand', () => {
        ctx.ringsInHand = 0;
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INSUFFICIENT_RINGS');
      });

      it('rejects placement off board', () => {
        const result = validatePlacementOnBoard(board, pos(-1, 3), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
      });

      it('rejects placement on collapsed space', () => {
        board.collapsedSpaces.set(posStr(3, 3), 1);
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('COLLAPSED_SPACE');
      });

      it('rejects placement on marker', () => {
        const marker: MarkerInfo = { player: 1, position: pos(3, 3), type: 'regular' };
        board.markers.set(posStr(3, 3), marker);
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('MARKER_BLOCKED');
      });
    });

    describe('existing stack placement', () => {
      beforeEach(() => {
        // Add an existing stack at (3, 3)
        const stack: RingStack = {
          position: pos(3, 3),
          rings: [1],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 1,
        };
        board.stacks.set(posStr(3, 3), stack);
      });

      it('allows placing 1 ring on existing stack', () => {
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(true);
        expect(result.maxPlacementCount).toBe(1);
      });

      it('rejects placing more than 1 ring on existing stack', () => {
        const result = validatePlacementOnBoard(board, pos(3, 3), 2, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_COUNT');
        expect(result.reason).toContain('only place 1 ring');
      });
    });

    describe('ring cap constraints', () => {
      it('respects ringsPerPlayerCap', () => {
        // Player has 18 rings total cap, already has 17 on board
        ctx.ringsOnBoard = 17;
        ctx.ringsInHand = 5;
        // Can only place 1 more ring due to cap
        const result = validatePlacementOnBoard(board, pos(3, 3), 2, ctx);
        expect(result.valid).toBe(false);
        expect(result.maxPlacementCount).toBe(1);
      });

      it('rejects when cap is reached', () => {
        ctx.ringsOnBoard = 18;
        ctx.ringsInHand = 5;
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_RINGS_AVAILABLE');
      });

      it('uses precomputed maxAvailableGlobal when provided', () => {
        ctx.maxAvailableGlobal = 2;
        const result = validatePlacementOnBoard(board, pos(3, 3), 3, ctx);
        expect(result.valid).toBe(false);
        expect(result.maxPlacementCount).toBe(2);
      });
    });

    describe('no-dead-placement rule', () => {
      it('rejects placement that results in stack with no legal moves', () => {
        // Create a corner scenario where the stack would be blocked
        // Place stacks and collapsed spaces to block all directions from (0, 0)
        board.collapsedSpaces.set(posStr(1, 0), 1);
        board.collapsedSpaces.set(posStr(0, 1), 1);
        board.collapsedSpaces.set(posStr(1, 1), 1);

        // Stack of height 1 at (0,0) would need to move at least 1 space
        // but all adjacent spaces are blocked
        const result = validatePlacementOnBoard(board, pos(0, 0), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('NO_LEGAL_MOVES');
      });

      it('allows placement when at least one legal move exists', () => {
        // (3, 3) with height 1 can move in any direction
        const result = validatePlacementOnBoard(board, pos(3, 3), 1, ctx);
        expect(result.valid).toBe(true);
      });
    });

    describe('hexagonal board', () => {
      beforeEach(() => {
        board = createTestBoard('hexagonal');
        ctx.boardType = 'hexagonal';
      });

      it('validates positions using hex geometry', () => {
        // Center of hex board is valid
        const result = validatePlacementOnBoard(board, pos(0, 0, 0), 1, ctx);
        expect(result.valid).toBe(true);
      });

      it('rejects positions off hex board', () => {
        // Position far outside hex radius
        const result = validatePlacementOnBoard(board, pos(100, 100, -200), 1, ctx);
        expect(result.valid).toBe(false);
        expect(result.code).toBe('INVALID_POSITION');
      });
    });
  });

  describe('validatePlacement (GameState wrapper)', () => {
    let state: GameState;

    beforeEach(() => {
      state = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        players: [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
        ],
      });
    });

    it('validates placement in ring_placement phase', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects placement in wrong phase', () => {
      state.currentPhase = 'movement';
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects placement when not player turn', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 2, // Not current player
        position: pos(3, 3),
        count: 1,
      };
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects invalid count', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 0,
      };
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_COUNT');
    });

    it('rejects when not enough rings in hand', () => {
      state.players[0].ringsInHand = 2;
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 3,
      };
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INSUFFICIENT_RINGS');
    });
  });

  describe('validateSkipPlacement', () => {
    let state: GameState;

    beforeEach(() => {
      state = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        players: [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
        ],
      });

      // Add a controlled stack with legal moves
      const stack: RingStack = {
        position: pos(3, 3),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(3, 3), stack);
    });

    it('allows skip when player has controlled stack with legal moves', () => {
      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects skip in wrong phase', () => {
      state.currentPhase = 'movement';
      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('INVALID_PHASE');
    });

    it('rejects skip when not player turn', () => {
      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 2,
      };
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('rejects skip when player has no controlled stacks', () => {
      state.board.stacks.clear();
      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_CONTROLLED_STACKS');
    });

    it('rejects skip when no legal moves from controlled stacks', () => {
      // Block all directions from the stack
      state.board.collapsedSpaces.set(posStr(4, 3), 1);
      state.board.collapsedSpaces.set(posStr(2, 3), 1);
      state.board.collapsedSpaces.set(posStr(3, 4), 1);
      state.board.collapsedSpaces.set(posStr(3, 2), 1);
      state.board.collapsedSpaces.set(posStr(4, 4), 1);
      state.board.collapsedSpaces.set(posStr(2, 2), 1);
      state.board.collapsedSpaces.set(posStr(4, 2), 1);
      state.board.collapsedSpaces.set(posStr(2, 4), 1);

      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('NO_LEGAL_ACTIONS');
    });

    it('allows skip even with zero rings in hand', () => {
      // Per the code comment, skip is allowed when ringsInHand=0
      state.players[0].ringsInHand = 0;
      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(true);
    });

    it('rejects skip when player not found', () => {
      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 99, // Non-existent player
      };
      state.currentPlayer = 99; // Also set current player to bypass turn check
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('PLAYER_NOT_FOUND');
    });

    it('skips stacks with zero height when checking for legal actions', () => {
      // Add a stack with zero height (defensive case)
      const emptyStack: RingStack = {
        position: pos(5, 5),
        rings: [],
        stackHeight: 0,
        capHeight: 0,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(5, 5), emptyStack);

      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      // Should still succeed because we have the other stack with legal moves
      const result = validateSkipPlacement(state, action);
      expect(result.valid).toBe(true);
    });

    it('uses getMarkerOwner when checking legal captures', () => {
      // Add markers near the stack to test getMarkerOwner branch
      const marker: MarkerInfo = { player: 2, position: pos(4, 3), type: 'regular' };
      state.board.markers.set(posStr(4, 3), marker);

      const action: SkipPlacementAction = {
        type: 'skip_placement',
        playerId: 1,
      };
      const result = validateSkipPlacement(state, action);
      // Should still be valid - the marker doesn't block moves
      expect(result.valid).toBe(true);
    });
  });

  describe('validatePlacement edge cases', () => {
    let state: GameState;

    beforeEach(() => {
      state = createTestGameState({
        currentPhase: 'ring_placement',
        currentPlayer: 1,
        players: [
          createTestPlayer(1, { ringsInHand: 18 }),
          createTestPlayer(2, { ringsInHand: 18 }),
        ],
      });
    });

    it('rejects placement when player not found', () => {
      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 99, // Non-existent player
        position: pos(3, 3),
        count: 1,
      };
      state.currentPlayer = 99; // Bypass turn check
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(false);
      expect(result.code).toBe('PLAYER_NOT_FOUND');
    });

    it('considers existing stacks when validating placement moves', () => {
      // Add a stack adjacent to placement position to trigger getStackAt branch
      const adjacentStack: RingStack = {
        position: pos(4, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(4, 3), adjacentStack);

      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };
      // Placement should be valid - can still move/capture
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(true);
    });

    it('uses markers when checking legal moves from placement', () => {
      // Add markers near placement position
      const marker: MarkerInfo = { player: 2, position: pos(4, 3), type: 'regular' };
      state.board.markers.set(posStr(4, 3), marker);

      const action: PlaceRingAction = {
        type: 'place_ring',
        playerId: 1,
        position: pos(3, 3),
        count: 1,
      };
      const result = validatePlacement(state, action);
      expect(result.valid).toBe(true);
    });
  });
});
