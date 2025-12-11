/**
 * EliminationAggregate Unit Tests
 *
 * Tests for the canonical elimination logic module.
 * Verifies eligibility, costs, and behavior for all elimination contexts.
 */

import {
  EliminationContext,
  EliminationParams,
  eliminateFromStack,
  isStackEligibleForElimination,
  enumerateEligibleStacks,
  hasEligibleEliminationTarget,
  calculateCapHeightElimination as calculateCapHeight,
  getRingsToEliminate,
} from '../../../src/shared/engine';
import type { BoardState, RingStack, Position } from '../../../src/shared/types/game';
import { positionToString } from '../../../src/shared/types/game';

// =============================================================================
// TEST HELPERS
// =============================================================================

function createStack(position: Position, rings: number[], controllingPlayer?: number): RingStack {
  const topPlayer = rings.length > 0 ? rings[0] : 0;
  let capHeight = 0;
  for (const ring of rings) {
    if (ring === topPlayer) capHeight++;
    else break;
  }
  return {
    position,
    rings,
    stackHeight: rings.length,
    capHeight,
    controllingPlayer: controllingPlayer ?? topPlayer,
  };
}

function createBoardWithStacks(stacks: RingStack[]): BoardState {
  const stackMap = new Map<string, RingStack>();
  for (const stack of stacks) {
    stackMap.set(positionToString(stack.position), stack);
  }
  return {
    stacks: stackMap,
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };
}

// =============================================================================
// CALCULATE CAP HEIGHT
// =============================================================================

describe('calculateCapHeight', () => {
  it('returns 0 for empty array', () => {
    expect(calculateCapHeight([])).toBe(0);
  });

  it('returns 1 for single ring', () => {
    expect(calculateCapHeight([1])).toBe(1);
  });

  it('returns full height for uniform stack', () => {
    expect(calculateCapHeight([1, 1, 1])).toBe(3);
  });

  it('returns cap height for multicolor stack', () => {
    expect(calculateCapHeight([1, 1, 2])).toBe(2);
    expect(calculateCapHeight([1, 2, 2])).toBe(1);
    expect(calculateCapHeight([2, 2, 1, 1])).toBe(2);
  });
});

// =============================================================================
// GET RINGS TO ELIMINATE
// =============================================================================

describe('getRingsToEliminate', () => {
  it('returns 1 for line context regardless of cap height', () => {
    const stack1 = createStack({ x: 0, y: 0 }, [1]);
    const stack3 = createStack({ x: 0, y: 0 }, [1, 1, 1]);
    const multicolor = createStack({ x: 0, y: 0 }, [1, 1, 2]);

    expect(getRingsToEliminate(stack1, 'line')).toBe(1);
    expect(getRingsToEliminate(stack3, 'line')).toBe(1);
    expect(getRingsToEliminate(multicolor, 'line')).toBe(1);
  });

  it('returns cap height for territory context', () => {
    const stack1 = createStack({ x: 0, y: 0 }, [1]);
    const stack3 = createStack({ x: 0, y: 0 }, [1, 1, 1]);
    const multicolor = createStack({ x: 0, y: 0 }, [1, 1, 2]);

    expect(getRingsToEliminate(stack1, 'territory')).toBe(1);
    expect(getRingsToEliminate(stack3, 'territory')).toBe(3);
    expect(getRingsToEliminate(multicolor, 'territory')).toBe(2);
  });

  it('returns cap height for forced context', () => {
    const stack1 = createStack({ x: 0, y: 0 }, [1]);
    const stack3 = createStack({ x: 0, y: 0 }, [1, 1, 1]);

    expect(getRingsToEliminate(stack1, 'forced')).toBe(1);
    expect(getRingsToEliminate(stack3, 'forced')).toBe(3);
  });

  it('returns 1 for recovery context', () => {
    const stack3 = createStack({ x: 0, y: 0 }, [1, 1, 2]);
    expect(getRingsToEliminate(stack3, 'recovery')).toBe(1);
  });
});

// =============================================================================
// STACK ELIGIBILITY
// =============================================================================

describe('isStackEligibleForElimination', () => {
  describe('line context (RR-CANON-R122)', () => {
    it('allows any controlled stack including height-1', () => {
      const height1 = createStack({ x: 0, y: 0 }, [1]);
      const height3 = createStack({ x: 0, y: 0 }, [1, 1, 1]);
      const multicolor = createStack({ x: 0, y: 0 }, [1, 1, 2]);

      expect(isStackEligibleForElimination(height1, 'line', 1).eligible).toBe(true);
      expect(isStackEligibleForElimination(height3, 'line', 1).eligible).toBe(true);
      expect(isStackEligibleForElimination(multicolor, 'line', 1).eligible).toBe(true);
    });

    it('rejects stacks not controlled by player', () => {
      const opponentStack = createStack({ x: 0, y: 0 }, [2, 2, 1]);
      expect(isStackEligibleForElimination(opponentStack, 'line', 1).eligible).toBe(false);
    });
  });

  describe('territory context (RR-CANON-R145)', () => {
    it('rejects height-1 standalone stacks', () => {
      const height1 = createStack({ x: 0, y: 0 }, [1]);
      const result = isStackEligibleForElimination(height1, 'territory', 1);
      expect(result.eligible).toBe(false);
      expect(result.reason.toLowerCase()).toContain('height-1');
    });

    it('allows multicolor stacks', () => {
      const multicolor = createStack({ x: 0, y: 0 }, [1, 1, 2]);
      expect(isStackEligibleForElimination(multicolor, 'territory', 1).eligible).toBe(true);
    });

    it('allows single-color stacks with height > 1', () => {
      const height2 = createStack({ x: 0, y: 0 }, [1, 1]);
      const height3 = createStack({ x: 0, y: 0 }, [1, 1, 1]);
      expect(isStackEligibleForElimination(height2, 'territory', 1).eligible).toBe(true);
      expect(isStackEligibleForElimination(height3, 'territory', 1).eligible).toBe(true);
    });

    it('rejects stacks not controlled by player', () => {
      const opponentMulticolor = createStack({ x: 0, y: 0 }, [2, 2, 1]);
      expect(isStackEligibleForElimination(opponentMulticolor, 'territory', 1).eligible).toBe(
        false
      );
    });
  });

  describe('forced context (RR-CANON-R100)', () => {
    it('allows any controlled stack including height-1', () => {
      const height1 = createStack({ x: 0, y: 0 }, [1]);
      const multicolor = createStack({ x: 0, y: 0 }, [1, 1, 2]);

      expect(isStackEligibleForElimination(height1, 'forced', 1).eligible).toBe(true);
      expect(isStackEligibleForElimination(multicolor, 'forced', 1).eligible).toBe(true);
    });
  });

  describe('recovery context (RR-CANON-R113)', () => {
    it('allows stacks with buried rings of player', () => {
      const buriedP1 = createStack({ x: 0, y: 0 }, [2, 1, 2]); // P1 ring buried
      expect(isStackEligibleForElimination(buriedP1, 'recovery', 1).eligible).toBe(true);
    });

    it('rejects stacks without buried rings of player', () => {
      const noBuried = createStack({ x: 0, y: 0 }, [1, 2, 2]); // P1 only on top
      const result = isStackEligibleForElimination(noBuried, 'recovery', 1);
      expect(result.eligible).toBe(false);
      expect(result.reason).toContain('buried');
    });

    it('does not require control for recovery', () => {
      // Player 1 can extract buried ring even if P2 controls
      const buriedP1 = createStack({ x: 0, y: 0 }, [2, 2, 1]); // P2 controls, P1 buried
      expect(isStackEligibleForElimination(buriedP1, 'recovery', 1).eligible).toBe(true);
    });
  });
});

// =============================================================================
// ENUMERATE ELIGIBLE STACKS
// =============================================================================

describe('enumerateEligibleStacks', () => {
  it('filters by context correctly', () => {
    const height1P1 = createStack({ x: 0, y: 0 }, [1]);
    const height2P1 = createStack({ x: 1, y: 0 }, [1, 1]);
    const multiP1 = createStack({ x: 2, y: 0 }, [1, 1, 2]);
    const height1P2 = createStack({ x: 3, y: 0 }, [2]);

    const board = createBoardWithStacks([height1P1, height2P1, multiP1, height1P2]);

    // Line: all P1 stacks (3)
    const lineEligible = enumerateEligibleStacks(board, 1, 'line');
    expect(lineEligible.length).toBe(3);

    // Territory: P1 stacks excluding height-1 (2)
    const territoryEligible = enumerateEligibleStacks(board, 1, 'territory');
    expect(territoryEligible.length).toBe(2);
    expect(territoryEligible.some((s) => s.stackHeight === 1)).toBe(false);
  });

  it('respects excludePositions', () => {
    const stack1 = createStack({ x: 0, y: 0 }, [1, 1]);
    const stack2 = createStack({ x: 1, y: 0 }, [1, 1]);
    const board = createBoardWithStacks([stack1, stack2]);

    const excluded = new Set(['0,0']);
    const eligible = enumerateEligibleStacks(board, 1, 'territory', excluded);
    expect(eligible.length).toBe(1);
    expect(positionToString(eligible[0].position)).toBe('1,0');
  });
});

// =============================================================================
// HAS ELIGIBLE ELIMINATION TARGET
// =============================================================================

describe('hasEligibleEliminationTarget', () => {
  it('returns true when eligible stacks exist', () => {
    const stack = createStack({ x: 0, y: 0 }, [1, 1]);
    const board = createBoardWithStacks([stack]);
    expect(hasEligibleEliminationTarget(board, 1, 'territory')).toBe(true);
  });

  it('returns false when no eligible stacks exist', () => {
    const height1 = createStack({ x: 0, y: 0 }, [1]);
    const board = createBoardWithStacks([height1]);
    expect(hasEligibleEliminationTarget(board, 1, 'territory')).toBe(false);
  });

  it('respects excludePositions', () => {
    const stack = createStack({ x: 0, y: 0 }, [1, 1]);
    const board = createBoardWithStacks([stack]);
    const excluded = new Set(['0,0']);
    expect(hasEligibleEliminationTarget(board, 1, 'territory', excluded)).toBe(false);
  });
});

// =============================================================================
// ELIMINATE FROM STACK
// =============================================================================

describe('eliminateFromStack', () => {
  describe('line context', () => {
    it('eliminates exactly 1 ring from top', () => {
      const stack = createStack({ x: 0, y: 0 }, [1, 1, 1]);
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'line',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
        reason: 'line_reward_exact',
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(true);
      expect(result.ringsEliminated).toBe(1);
      expect(result.updatedStack?.stackHeight).toBe(2);
      expect(result.updatedBoard.eliminatedRings[1]).toBe(1);
    });
  });

  describe('territory context', () => {
    it('eliminates entire cap', () => {
      const stack = createStack({ x: 0, y: 0 }, [1, 1, 2]); // cap = 2
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'territory',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
        reason: 'territory_self_elimination',
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(true);
      expect(result.ringsEliminated).toBe(2);
      expect(result.updatedStack?.stackHeight).toBe(1);
      expect(result.updatedStack?.controllingPlayer).toBe(2);
    });

    it('rejects height-1 stacks', () => {
      const stack = createStack({ x: 0, y: 0 }, [1]);
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'territory',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(false);
      expect(result.error?.toLowerCase()).toContain('height-1');
    });
  });

  describe('forced context', () => {
    it('eliminates entire cap including height-1', () => {
      const stack = createStack({ x: 0, y: 0 }, [1]);
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'forced',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
        reason: 'forced_elimination_anm',
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(true);
      expect(result.ringsEliminated).toBe(1);
      expect(result.updatedStack).toBeNull();
      // Stack should be removed from board
      expect(result.updatedBoard.stacks.has('0,0')).toBe(false);
    });
  });

  describe('recovery context', () => {
    it('extracts bottommost buried ring', () => {
      // Stack: [2, 1, 1, 2] - top to bottom
      // P1 rings at index 1 and 2, should extract bottommost (index 2)
      const stack = createStack({ x: 0, y: 0 }, [2, 1, 1, 2]);
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'recovery',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
        reason: 'recovery_buried_extraction',
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(true);
      expect(result.ringsEliminated).toBe(1);
      // One P1 ring should remain
      expect(result.updatedStack?.rings.filter((r) => r === 1).length).toBe(1);
    });

    it('fails when no buried rings of player exist', () => {
      const stack = createStack({ x: 0, y: 0 }, [1, 2, 2]); // P1 only on top
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'recovery',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(false);
      expect(result.error).toContain('buried');
    });
  });

  describe('immutability', () => {
    it('does not mutate original board', () => {
      const stack = createStack({ x: 0, y: 0 }, [1, 1, 1]);
      const board = createBoardWithStacks([stack]);
      const originalStackHeight = board.stacks.get('0,0')?.stackHeight;

      const params: EliminationParams = {
        context: 'line',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
      };

      eliminateFromStack(params);

      // Original unchanged
      expect(board.stacks.get('0,0')?.stackHeight).toBe(originalStackHeight);
    });
  });

  describe('error handling', () => {
    it('fails for non-existent stack', () => {
      const board = createBoardWithStacks([]);

      const params: EliminationParams = {
        context: 'line',
        player: 1,
        stackPosition: { x: 99, y: 99 },
        board,
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(false);
      expect(result.error).toContain('No stack');
    });

    it('fails when player does not control stack', () => {
      const stack = createStack({ x: 0, y: 0 }, [2, 2, 1]); // P2 controls
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'line',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
      };

      const result = eliminateFromStack(params);
      expect(result.success).toBe(false);
      expect(result.error).toContain('does not control');
    });
  });

  describe('audit events', () => {
    it('includes audit event in result', () => {
      const stack = createStack({ x: 0, y: 0 }, [1, 1]);
      const board = createBoardWithStacks([stack]);

      const params: EliminationParams = {
        context: 'territory',
        player: 1,
        stackPosition: { x: 0, y: 0 },
        board,
        reason: 'territory_self_elimination',
      };

      const result = eliminateFromStack(params);
      expect(result.auditEvent).toBeDefined();
      expect(result.auditEvent?.context).toBe('territory');
      expect(result.auditEvent?.reason).toBe('territory_self_elimination');
      expect(result.auditEvent?.ringsEliminated).toBe(2);
      expect(result.auditEvent?.stackHeightBefore).toBe(2);
      expect(result.auditEvent?.stackHeightAfter).toBe(0);
    });
  });
});
