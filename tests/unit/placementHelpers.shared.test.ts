import {
  BoardType,
  Player,
  TimeControl,
  GameState,
  Position,
  Move,
  RingStack,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  applyPlacementMove,
  evaluateSkipPlacementEligibility,
} from '../../src/shared/engine/placementHelpers';

describe('placementHelpers – shared placement application and skip-placement evaluation', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createTestState(id: string): GameState {
    const state = createInitialGameState(
      id,
      boardType,
      players,
      timeControl
    ) as unknown as GameState;

    state.currentPhase = 'ring_placement';
    state.currentPlayer = 1;
    state.gameStatus = 'active';
    return state;
  }

  function createPlaceRingMove(player: number, to: Position, placementCount: number = 1): Move {
    return {
      id: `place-${positionToString(to)}`,
      type: 'place_ring',
      player,
      to,
      placementCount,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };
  }

  function addStackToBoard(
    state: GameState,
    position: Position,
    rings: number[],
    controllingPlayer: number
  ): void {
    const key = positionToString(position);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: countCapHeight(rings),
      controllingPlayer,
    };
    state.board.stacks.set(key, stack);
  }

  function countCapHeight(rings: number[]): number {
    if (rings.length === 0) return 0;
    const topColor = rings[0];
    let count = 1;
    for (let i = 1; i < rings.length; i++) {
      if (rings[i] === topColor) {
        count++;
      } else {
        break;
      }
    }
    return count;
  }

  // ==========================================================================
  // applyPlacementMove Tests
  // ==========================================================================

  describe('applyPlacementMove', () => {
    it('places a single ring on an empty cell', () => {
      const state = createTestState('place-empty-cell');
      const pos: Position = { x: 3, y: 3 };
      const move = createPlaceRingMove(1, pos, 1);

      const outcome = applyPlacementMove(state, move);

      expect(outcome.placementCount).toBe(1);
      expect(outcome.placedOnStack).toBe(false);

      const key = positionToString(pos);
      const stack = outcome.nextState.board.stacks.get(key);
      expect(stack).toBeDefined();
      expect(stack!.rings).toEqual([1]);
      expect(stack!.controllingPlayer).toBe(1);
      expect(stack!.stackHeight).toBe(1);
      expect(stack!.capHeight).toBe(1);

      // Verify ringsInHand was decremented
      const player = outcome.nextState.players.find((p) => p.playerNumber === 1)!;
      expect(player.ringsInHand).toBe(17);
    });

    it('places a single ring on an existing stack', () => {
      const state = createTestState('place-on-stack');
      const pos: Position = { x: 3, y: 3 };

      // Add an existing stack on the position owned by player 2
      addStackToBoard(state, pos, [2, 2], 2);

      const move = createPlaceRingMove(1, pos, 1);
      const outcome = applyPlacementMove(state, move);

      expect(outcome.placementCount).toBe(1);
      expect(outcome.placedOnStack).toBe(true);

      const key = positionToString(pos);
      const stack = outcome.nextState.board.stacks.get(key);
      expect(stack).toBeDefined();
      expect(stack!.rings).toEqual([1, 2, 2]);
      expect(stack!.controllingPlayer).toBe(1);
      expect(stack!.stackHeight).toBe(3);
      expect(stack!.capHeight).toBe(1);
    });

    it('places multiple rings (up to 3) on an empty cell', () => {
      const state = createTestState('place-multiple');
      const pos: Position = { x: 4, y: 4 };
      const move = createPlaceRingMove(1, pos, 3);

      const outcome = applyPlacementMove(state, move);

      expect(outcome.placementCount).toBe(3);
      expect(outcome.placedOnStack).toBe(false);

      const key = positionToString(pos);
      const stack = outcome.nextState.board.stacks.get(key);
      expect(stack).toBeDefined();
      expect(stack!.rings).toEqual([1, 1, 1]);
      expect(stack!.stackHeight).toBe(3);
      expect(stack!.capHeight).toBe(3);

      // Verify ringsInHand was decremented by 3
      const player = outcome.nextState.players.find((p) => p.playerNumber === 1)!;
      expect(player.ringsInHand).toBe(15);
    });

    it('clamps placement count to ringsInHand', () => {
      const state = createTestState('place-clamp-rings');
      const pos: Position = { x: 2, y: 2 };

      // Set player to have only 2 rings
      state.players = state.players.map((p) =>
        p.playerNumber === 1 ? { ...p, ringsInHand: 2 } : p
      );

      const move = createPlaceRingMove(1, pos, 3);
      const outcome = applyPlacementMove(state, move);

      // Should have placed only 2 rings
      expect(outcome.placementCount).toBe(2);

      const key = positionToString(pos);
      const stack = outcome.nextState.board.stacks.get(key);
      expect(stack!.rings).toEqual([1, 1]);

      const player = outcome.nextState.players.find((p) => p.playerNumber === 1)!;
      expect(player.ringsInHand).toBe(0);
    });

    it('returns 0 placed when player has no rings in hand', () => {
      const state = createTestState('place-no-rings');
      const pos: Position = { x: 5, y: 5 };

      // Set player to have 0 rings
      state.players = state.players.map((p) =>
        p.playerNumber === 1 ? { ...p, ringsInHand: 0 } : p
      );

      const move = createPlaceRingMove(1, pos, 1);
      const outcome = applyPlacementMove(state, move);

      expect(outcome.placementCount).toBe(0);
      expect(outcome.placedOnStack).toBe(false);
      // State should be unchanged
      expect(outcome.nextState.board.stacks.has(positionToString(pos))).toBe(false);
    });

    it('throws error for wrong move type', () => {
      const state = createTestState('place-wrong-type');
      const move: Move = {
        id: 'wrong-move',
        type: 'move_stack', // Wrong type
        player: 1,
        to: { x: 3, y: 3 },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      expect(() => applyPlacementMove(state, move)).toThrow(/Expected 'place_ring' move/);
    });

    it('throws error for non-existent player', () => {
      const state = createTestState('place-no-player');
      const move = createPlaceRingMove(99, { x: 3, y: 3 }, 1);

      expect(() => applyPlacementMove(state, move)).toThrow(/Player not found/);
    });

    it('clears existing marker at destination when placing', () => {
      const state = createTestState('place-clear-marker');
      const pos: Position = { x: 6, y: 6 };
      const key = positionToString(pos);

      // Add a marker at the destination
      state.board.markers.set(key, {
        player: 2,
        position: pos,
        type: 'regular',
      });

      const move = createPlaceRingMove(1, pos, 1);
      const outcome = applyPlacementMove(state, move);

      // Marker should be cleared
      expect(outcome.nextState.board.markers.has(key)).toBe(false);
      // Stack should be created
      expect(outcome.nextState.board.stacks.has(key)).toBe(true);
    });
  });

  // ==========================================================================
  // evaluateSkipPlacementEligibility Tests
  // ==========================================================================

  describe('evaluateSkipPlacementEligibility', () => {
    it('returns canSkip=true when player has controlled stack with legal moves', () => {
      const state = createTestState('skip-eligible');
      // Add a stack for player 1 that can move
      addStackToBoard(state, { x: 3, y: 3 }, [1], 1);

      const result = evaluateSkipPlacementEligibility(state, 1);

      expect(result.canSkip).toBe(true);
      expect(result.reason).toBeUndefined();
      expect(result.code).toBeUndefined();
    });

    it('returns canSkip=false when not in ring_placement phase', () => {
      const state = createTestState('skip-wrong-phase');
      state.currentPhase = 'movement';
      addStackToBoard(state, { x: 3, y: 3 }, [1], 1);

      const result = evaluateSkipPlacementEligibility(state, 1);

      expect(result.canSkip).toBe(false);
      expect(result.reason).toBe('Not in ring placement phase');
      expect(result.code).toBe('INVALID_PHASE');
    });

    it("returns canSkip=false when not player's turn", () => {
      const state = createTestState('skip-not-turn');
      state.currentPlayer = 2;
      addStackToBoard(state, { x: 3, y: 3 }, [1], 1);

      const result = evaluateSkipPlacementEligibility(state, 1);

      expect(result.canSkip).toBe(false);
      expect(result.reason).toBe('Not your turn');
      expect(result.code).toBe('NOT_YOUR_TURN');
    });

    it('returns canSkip=false when player not found', () => {
      const state = createTestState('skip-no-player');
      // Set currentPlayer to 99 so it passes the turn check first
      state.currentPlayer = 99;

      const result = evaluateSkipPlacementEligibility(state, 99);

      expect(result.canSkip).toBe(false);
      expect(result.reason).toBe('Player not found');
      expect(result.code).toBe('PLAYER_NOT_FOUND');
    });

    it('returns canSkip=false when player controls no stacks', () => {
      const state = createTestState('skip-no-stacks');
      // Don't add any stacks for player 1

      const result = evaluateSkipPlacementEligibility(state, 1);

      expect(result.canSkip).toBe(false);
      expect(result.reason).toContain('control no stacks');
      expect(result.code).toBe('NO_CONTROLLED_STACKS');
    });

    it('returns canSkip=false when no legal moves available from controlled stacks', () => {
      const state = createTestState('skip-no-moves');

      // Create a corner situation where the stack is blocked
      // Add player 1 stack in a corner
      addStackToBoard(state, { x: 0, y: 0 }, [1], 1);

      // Fill all adjacent cells with collapsed spaces so no moves are possible
      const adjacentPositions: Position[] = [
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 1 },
      ];

      // Mark them all as collapsed
      for (const pos of adjacentPositions) {
        state.board.collapsedSpaces.set(positionToString(pos), 2);
      }

      const result = evaluateSkipPlacementEligibility(state, 1);

      expect(result.canSkip).toBe(false);
      expect(result.reason).toContain('no legal moves or captures');
      expect(result.code).toBe('NO_LEGAL_ACTIONS');
    });

    it('returns canSkip=true when player has zero rings but has legal moves from stacks', () => {
      const state = createTestState('skip-zero-rings');
      addStackToBoard(state, { x: 3, y: 3 }, [1], 1);

      // Set player to have 0 rings in hand
      state.players = state.players.map((p) =>
        p.playerNumber === 1 ? { ...p, ringsInHand: 0 } : p
      );

      const result = evaluateSkipPlacementEligibility(state, 1);

      // Should still be eligible to skip - having no rings in hand is fine
      expect(result.canSkip).toBe(true);
    });

    it('evaluates captures as legal actions for skip eligibility', () => {
      const state = createTestState('skip-with-capture');

      // Add player 1 stack with height 2
      addStackToBoard(state, { x: 2, y: 2 }, [1, 1], 1);

      // Add player 2 stack that can be captured (distance 1-2, empty landing)
      addStackToBoard(state, { x: 3, y: 2 }, [2], 2);

      // Block movement directions but leave capture path clear
      // The landing at {4, 2} is empty, so capture should be possible

      const result = evaluateSkipPlacementEligibility(state, 1);

      expect(result.canSkip).toBe(true);
    });
  });
});
