/**
 * MovementMutator Unit Tests
 *
 * Tests the shared engine movement mutation logic including:
 * - Stack movement to empty space
 * - Marker placement at origin
 * - Landing on own marker (ring elimination)
 * - State immutability
 */

import { mutateMovement } from '../../../src/shared/engine/mutators/MovementMutator';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  pos,
  posStr,
} from '../../utils/fixtures';
import { RingStack, MarkerInfo } from '../../../src/shared/types/game';
import { MoveStackAction, GameState } from '../../../src/shared/engine/types';

describe('MovementMutator', () => {
  let state: GameState;

  beforeEach(() => {
    state = createTestGameState({
      currentPhase: 'movement',
      currentPlayer: 1,
      players: [
        createTestPlayer(1, { ringsInHand: 10, eliminatedRings: 0 }),
        createTestPlayer(2, { ringsInHand: 10, eliminatedRings: 0 }),
      ],
    });

    // Add a stack for player 1 at (3, 3)
    const stack: RingStack = {
      position: pos(3, 3),
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };
    state.board.stacks.set(posStr(3, 3), stack);
  });

  describe('basic movement', () => {
    it('moves stack from origin to destination', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      expect(result.board.stacks.has(posStr(3, 3))).toBe(false);
      expect(result.board.stacks.has(posStr(5, 3))).toBe(true);

      const movedStack = result.board.stacks.get(posStr(5, 3));
      expect(movedStack!.rings).toEqual([1, 1]);
      expect(movedStack!.position).toEqual(pos(5, 3));
    });

    it('places marker at origin position', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);
      const marker = result.board.markers.get(posStr(3, 3));

      expect(marker).toBeDefined();
      expect(marker!.player).toBe(1);
      expect(marker!.position).toEqual(pos(3, 3));
      expect(marker!.type).toBe('regular');
    });

    it('updates lastMoveAt timestamp', () => {
      const originalTime = state.lastMoveAt;
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      expect(result.lastMoveAt.getTime()).toBeGreaterThanOrEqual(originalTime.getTime());
    });
  });

  describe('landing on own marker', () => {
    beforeEach(() => {
      // Add player 1's marker at destination
      const marker: MarkerInfo = { player: 1, position: pos(5, 3), type: 'regular' };
      state.board.markers.set(posStr(5, 3), marker);
    });

    it('removes the marker at destination', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      expect(result.board.markers.has(posStr(5, 3))).toBe(false);
    });

    it('eliminates top ring of moving stack', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);
      const stack = result.board.stacks.get(posStr(5, 3));

      // Original: [1, 1], after eliminating top: [1]
      expect(stack!.rings).toEqual([1]);
      expect(stack!.stackHeight).toBe(1);
    });

    it('updates elimination counters', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      expect(result.totalRingsEliminated).toBe(1);
      expect(result.board.eliminatedRings[1]).toBe(1);

      const player = result.players.find((p) => p.playerNumber === 1);
      expect(player!.eliminatedRings).toBe(1);
    });

    it('recalculates capHeight after elimination', () => {
      // Add a mixed stack for more interesting capHeight recalculation
      const mixedStack: RingStack = {
        position: pos(3, 3),
        rings: [1, 2, 2], // Top is player 1, then player 2's rings
        stackHeight: 3,
        capHeight: 1,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(3, 3), mixedStack);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);
      const stack = result.board.stacks.get(posStr(5, 3));

      // After eliminating player 1's top ring: [2, 2]
      expect(stack!.rings).toEqual([2, 2]);
      expect(stack!.capHeight).toBe(2); // Player 2 now controls with capHeight 2
      expect(stack!.controllingPlayer).toBe(2);
    });

    it('removes stack entirely if only one ring', () => {
      // Single ring stack
      const singleRingStack: RingStack = {
        position: pos(3, 3),
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      };
      state.board.stacks.set(posStr(3, 3), singleRingStack);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      // Stack is eliminated, no stack at destination
      expect(result.board.stacks.has(posStr(5, 3))).toBe(false);
      expect(result.totalRingsEliminated).toBe(1);
    });
  });

  describe('landing on opponent marker', () => {
    beforeEach(() => {
      // Add opponent marker at destination
      const marker: MarkerInfo = { player: 2, position: pos(5, 3), type: 'regular' };
      state.board.markers.set(posStr(5, 3), marker);
    });

    it('removes marker and eliminates top ring (per RR-CANON-R091/R092)', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      // Opponent marker should be removed
      expect(result.board.markers.has(posStr(5, 3))).toBe(false);
      // Top ring should be eliminated
      expect(result.totalRingsEliminated).toBe(1);
      // Stack should be at destination with reduced height
      const destStack = result.board.stacks.get(posStr(5, 3));
      expect(destStack?.stackHeight).toBe(1); // Was 2, now 1
    });
  });

  describe('error handling', () => {
    it('throws error when no stack at origin', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(0, 0), // No stack here
        to: pos(2, 0),
      };

      expect(() => mutateMovement(state, action)).toThrow('No stack at origin');
    });
  });

  describe('state immutability', () => {
    it('does not mutate original state', () => {
      const originalStackCount = state.board.stacks.size;
      const originalMarkerCount = state.board.markers.size;

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      mutateMovement(state, action);

      expect(state.board.stacks.size).toBe(originalStackCount);
      expect(state.board.markers.size).toBe(originalMarkerCount);
      expect(state.board.stacks.has(posStr(3, 3))).toBe(true);
      expect(state.board.stacks.has(posStr(5, 3))).toBe(false);
    });

    it('returns a new state object', () => {
      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 1,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      expect(result).not.toBe(state);
      expect(result.board).not.toBe(state.board);
      expect(result.board.stacks).not.toBe(state.board.stacks);
    });
  });

  describe('multi-player scenarios', () => {
    it('correctly attributes elimination to ring owner', () => {
      // Create a mixed stack where top ring belongs to player 2
      const mixedStack: RingStack = {
        position: pos(3, 3),
        rings: [2, 1], // Player 2 on top, player 1 below
        stackHeight: 2,
        capHeight: 1, // Player 2 has capHeight 1
        controllingPlayer: 2,
      };
      state.board.stacks.set(posStr(3, 3), mixedStack);
      state.currentPlayer = 2;

      // Add player 2's marker at destination
      const marker: MarkerInfo = { player: 2, position: pos(5, 3), type: 'regular' };
      state.board.markers.set(posStr(5, 3), marker);

      const action: MoveStackAction = {
        type: 'move_stack',
        playerId: 2,
        from: pos(3, 3),
        to: pos(5, 3),
      };

      const result = mutateMovement(state, action);

      // Top ring (player 2) is eliminated
      expect(result.board.eliminatedRings[2]).toBe(1);

      const player2 = result.players.find((p) => p.playerNumber === 2);
      expect(player2!.eliminatedRings).toBe(1);

      // Remaining stack is now controlled by player 1
      const stack = result.board.stacks.get(posStr(5, 3));
      expect(stack!.rings).toEqual([1]);
      expect(stack!.controllingPlayer).toBe(1);
    });
  });
});
