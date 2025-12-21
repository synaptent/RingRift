import {
  chooseLocalMoveFromCandidates,
  type LocalAIRng,
} from '../../src/shared/engine/localAIMoveSelection';
import type { GameState, Move, RingStack } from '../../src/shared/types/game';

function makeSeededRng(seed: number): LocalAIRng {
  let s = seed >>> 0;
  return () => {
    // Simple LCG: Numerical Recipes parameters
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

function makeBaseGameState(): GameState {
  const now = new Date();
  return {
    id: 'swap-test-game',
    boardType: 'square8',
    board: {
      stacks: {} as unknown as Map<string, RingStack>, // overridden per test via `as any`
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    } as any,
    players: [
      {
        id: 'p1',
        username: 'P1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'P2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ],
    currentPhase: 'movement',
    currentPlayer: 2,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
    spectators: [],
    gameStatus: 'active',
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0,
  };
}

function makeSwapCandidate(player: number = 2): Move {
  const now = new Date();
  return {
    id: 'swap-move',
    type: 'swap_sides',
    player,
    from: undefined,
    to: { x: 0, y: 0 },
    timestamp: now,
    thinkTime: 0,
    moveNumber: 1,
  };
}

function makeSimpleMoveCandidate(): Move {
  const now = new Date();
  return {
    id: 'simple-move',
    type: 'move_stack',
    player: 2,
    from: { x: 0, y: 1 },
    to: { x: 0, y: 2 },
    timestamp: now,
    thinkTime: 0,
    moveNumber: 1,
  };
}

describe('localAIMoveSelection swap behaviour', () => {
  it('prefers swap_sides for a strong center opening', () => {
    const state = makeBaseGameState();

    // Simulate a strong center opening: P1 stack in the 8x8 center cluster.
    const centerStack: RingStack = {
      position: { x: 3, y: 3 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };

    (state.board as any).stacks = {
      '3,3': centerStack,
    };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const rng = makeSeededRng(1234);

    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove, simpleMove], rng, 0);

    expect(selected).toBe(swapMove);
  });

  it('declines swap_sides when opening provides no P1 stacks (neutral/weak opening)', () => {
    const state = makeBaseGameState();

    // No P1 stacks on board – evaluateSwapOpportunity returns 0 and
    // chooseLocalMoveFromCandidates must fall back to non-swap moves.
    (state.board as any).stacks = {};

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const rng = makeSeededRng(42);

    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove, simpleMove], rng, 0);

    expect(selected).toBe(simpleMove);
  });

  it('is deterministic for randomness = 0 with a fixed LocalAIRng', () => {
    const state = makeBaseGameState();

    // Use a modest but clearly positive opening so swap is preferred.
    const centerStack: RingStack = {
      position: { x: 4, y: 4 },
      rings: [1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };

    (state.board as any).stacks = {
      '4,4': centerStack,
    };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const candidates: Move[] = [swapMove, simpleMove];

    const rng1 = makeSeededRng(999);
    const rng2 = makeSeededRng(999);

    const selected1 = chooseLocalMoveFromCandidates(2, state as any, [...candidates], rng1, 0);
    const selected2 = chooseLocalMoveFromCandidates(2, state as any, [...candidates], rng2, 0);

    expect(selected1?.id).toBe(selected2?.id);
  });

  it('with randomness > 0 and borderline opening, sometimes swaps and sometimes declines', () => {
    const state = makeBaseGameState();

    // Construct a borderline opening: P1 stack with zero height in a
    // non-center, non-adjacent position so the base swap value is 0.0.
    const borderlineStack: RingStack = {
      position: { x: 0, y: 0 },
      rings: [],
      stackHeight: 0,
      capHeight: 0,
      controllingPlayer: 1,
    };

    (state.board as any).stacks = {
      '0,0': borderlineStack,
    };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const candidates: Move[] = [swapMove, simpleMove];

    const originalRandom = Math.random;
    try {
      // Seeded Math.random so evaluateSwapOpportunity's noise term is
      // reproducible and covers both positive and negative values over
      // multiple trials.
      let s = 123456789;
      Math.random = () => {
        s = (1103515245 * s + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
      };

      let swapCount = 0;
      let nonSwapCount = 0;

      for (let i = 0; i < 50; i++) {
        const rng = makeSeededRng(100 + i);
        const selected = chooseLocalMoveFromCandidates(
          2,
          state as any,
          [...candidates],
          rng,
          2.0 // large enough to produce both positive and negative swapValue
        );

        if (selected?.id === swapMove.id) {
          swapCount += 1;
        } else if (selected?.id === simpleMove.id) {
          nonSwapCount += 1;
        }
      }

      expect(swapCount).toBeGreaterThan(0);
      expect(nonSwapCount).toBeGreaterThan(0);
    } finally {
      Math.random = originalRandom;
    }
  });
});

describe('localAIMoveSelection branch coverage', () => {
  it('returns null for empty candidates array', () => {
    const state = makeBaseGameState();
    const rng = makeSeededRng(1);

    const selected = chooseLocalMoveFromCandidates(1, state as any, [], rng, 0);

    expect(selected).toBeNull();
  });

  it('returns null when swap is declined and no other moves exist', () => {
    const state = makeBaseGameState();
    // No P1 stacks, so swap value is 0 (not positive)
    (state.board as any).stacks = {};

    const swapMove = makeSwapCandidate(2);
    const rng = makeSeededRng(42);

    // Only swap move available, but P2 should not swap since value is 0
    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove], rng, 0);

    // After declining swap, candidates becomes empty, so null
    expect(selected).toBeNull();
  });

  it('handles non-2-player games (swap evaluation returns 0)', () => {
    const state = makeBaseGameState();
    // Add a third player
    state.players.push({
      id: 'p3',
      username: 'P3',
      type: 'human',
      playerNumber: 3,
      isReady: true,
      timeRemaining: 600_000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    } as any);

    const centerStack: RingStack = {
      position: { x: 3, y: 3 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };
    (state.board as any).stacks = { '3,3': centerStack };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const rng = makeSeededRng(1);

    // With 3 players, evaluateSwapOpportunity returns 0, so swap is declined
    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove, simpleMove], rng, 0);

    expect(selected).toBe(simpleMove);
  });

  it('handles odd-sized square board (single center position)', () => {
    const state = makeBaseGameState();
    state.board.size = 9;
    state.board.type = 'square19'; // Using square19 type but odd size

    // Center of 9x9 is (4,4)
    const centerStack: RingStack = {
      position: { x: 4, y: 4 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };
    (state.board as any).stacks = { '4,4': centerStack };

    const swapMove = makeSwapCandidate(2);
    const simpleMove = makeSimpleMoveCandidate();
    const rng = makeSeededRng(1);

    const selected = chooseLocalMoveFromCandidates(2, state as any, [swapMove, simpleMove], rng, 0);

    // Center stack gives 15+ points, so swap should be chosen
    expect(selected).toBe(swapMove);
  });

  it('handles ring_placement phase with both placement and non-placement moves', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'ring_placement';
    (state.board as any).stacks = {};

    const now = new Date();
    const placeMove: Move = {
      id: 'place-1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const skipMove: Move = {
      id: 'skip-1',
      type: 'skip_placement',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    // Run multiple times to test both branches of the probabilistic selection
    let placeCount = 0;
    let skipCount = 0;

    // Use 200 iterations to ensure statistical coverage
    for (let i = 0; i < 200; i++) {
      const rng = makeSeededRng(i * 7 + 13); // Different seed pattern
      const selected = chooseLocalMoveFromCandidates(
        1,
        state as any,
        [placeMove, skipMove],
        rng,
        0
      );

      if (selected?.type === 'place_ring') placeCount++;
      else if (selected?.type === 'skip_placement') skipCount++;
    }

    // With 50% probability each and 200 runs, both should appear
    expect(placeCount).toBeGreaterThan(0);
    expect(skipCount).toBeGreaterThan(0);
  });

  it('prioritizes capture moves in movement phase', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const captureMove: Move = {
      id: 'capture-1',
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 2, y: 0 },
      captureTarget: { x: 1, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const simpleMove: Move = {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from: { x: 3, y: 3 },
      to: { x: 4, y: 4 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(
      1,
      state as any,
      [simpleMove, captureMove],
      rng,
      0
    );

    // Capture moves are prioritized
    expect(selected?.type).toBe('overtaking_capture');
  });

  it('falls back to simple movement when no captures available', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const simpleMove: Move = {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from: { x: 3, y: 3 },
      to: { x: 4, y: 4 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [simpleMove], rng, 0);

    expect(selected?.type).toBe('move_stack');
  });

  it('handles mandatory placement (only place_ring moves)', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'ring_placement';
    (state.board as any).stacks = {};

    const now = new Date();
    const placeMove1: Move = {
      id: 'place-1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const placeMove2: Move = {
      id: 'place-2',
      type: 'place_ring',
      player: 1,
      to: { x: 1, y: 1 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(
      1,
      state as any,
      [placeMove1, placeMove2],
      rng,
      0
    );

    expect(selected?.type).toBe('place_ring');
  });

  it('uses global fallback for non-standard phases', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'line_processing';
    (state.board as any).stacks = {};

    const now = new Date();
    const processLineMove: Move = {
      id: 'process-1',
      type: 'process_line',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [processLineMove], rng, 0);

    // Global fallback should select the process_line move
    expect(selected?.type).toBe('process_line');
  });

  it('sorts moves deterministically for RNG parity', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const moves: Move[] = [
      {
        id: 'move-b',
        type: 'move_stack',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 3, y: 3 },
        timestamp: now,
        thinkTime: 0,
        moveNumber: 1,
      },
      {
        id: 'move-a',
        type: 'move_stack',
        player: 1,
        from: { x: 1, y: 1 },
        to: { x: 2, y: 2 },
        timestamp: now,
        thinkTime: 0,
        moveNumber: 1,
      },
    ];

    // Same seed, same order
    const rng1 = makeSeededRng(42);
    const rng2 = makeSeededRng(42);

    const selected1 = chooseLocalMoveFromCandidates(1, state as any, [...moves], rng1, 0);
    const selected2 = chooseLocalMoveFromCandidates(1, state as any, [...moves], rng2, 0);

    expect(selected1?.id).toBe(selected2?.id);
  });

  it('handles moves without from position in sorting', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const moveWithFrom: Move = {
      id: 'with-from',
      type: 'move_stack',
      player: 1,
      from: { x: 1, y: 1 },
      to: { x: 2, y: 2 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const moveWithoutFrom: Move = {
      id: 'without-from',
      type: 'move_stack',
      player: 1,
      from: undefined,
      to: { x: 3, y: 3 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(
      1,
      state as any,
      [moveWithFrom, moveWithoutFrom],
      rng,
      0
    );

    // Should not crash, and select one of the moves
    expect(selected).not.toBeNull();
  });

  it('handles moves without to position in sorting', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'line_processing';
    (state.board as any).stacks = {};

    const now = new Date();
    const moveWithTo: Move = {
      id: 'with-to',
      type: 'process_line',
      player: 1,
      to: { x: 1, y: 1 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const moveWithoutTo: Move = {
      id: 'without-to',
      type: 'process_line',
      player: 1,
      to: undefined as any,
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(
      1,
      state as any,
      [moveWithTo, moveWithoutTo],
      rng,
      0
    );

    expect(selected).not.toBeNull();
  });

  it('handles moves with captureTarget in sorting', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const capture1: Move = {
      id: 'capture-1',
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 2, y: 0 },
      captureTarget: { x: 1, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const capture2: Move = {
      id: 'capture-2',
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 2, y: 0 },
      captureTarget: { x: 1, y: 1 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [capture2, capture1], rng, 0);

    expect(selected?.type).toBe('overtaking_capture');
  });

  it('handles continue_capture_segment moves', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'chain_capture';
    (state.board as any).stacks = {};

    const now = new Date();
    const chainCapture: Move = {
      id: 'chain-1',
      type: 'continue_capture_segment',
      player: 1,
      from: { x: 2, y: 0 },
      to: { x: 4, y: 0 },
      captureTarget: { x: 3, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [chainCapture], rng, 0);

    expect(selected?.type).toBe('continue_capture_segment');
  });

  it('handles capture phase', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'capture';
    (state.board as any).stacks = {};

    const now = new Date();
    const captureMove: Move = {
      id: 'capture-1',
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 2, y: 0 },
      captureTarget: { x: 1, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [captureMove], rng, 0);

    expect(selected?.type).toBe('overtaking_capture');
  });

  it('returns 0 when comparing identical moves in sort', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    // Two moves with identical sortable properties but different IDs
    const move1: Move = {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 1 },
      captureTarget: undefined,
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const move2: Move = {
      id: 'move-2',
      type: 'move_stack',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 1 },
      captureTarget: undefined,
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [move1, move2], rng, 0);

    // Should select one of the identical moves
    expect(selected).not.toBeNull();
    expect(['move-1', 'move-2']).toContain(selected?.id);
  });

  it('handles sorting when only one move has captureTarget', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const captureMove: Move = {
      id: 'capture-with-target',
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 2, y: 0 },
      captureTarget: { x: 1, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const moveWithoutTarget: Move = {
      id: 'move-without-target',
      type: 'overtaking_capture',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 2, y: 0 },
      captureTarget: undefined,
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    // Test both orderings to cover both branches
    const selected1 = chooseLocalMoveFromCandidates(
      1,
      state as any,
      [captureMove, moveWithoutTarget],
      rng,
      0
    );

    const rng2 = makeSeededRng(1);
    const selected2 = chooseLocalMoveFromCandidates(
      1,
      state as any,
      [moveWithoutTarget, captureMove],
      rng2,
      0
    );

    // Both should select something and sorting should be deterministic
    expect(selected1).not.toBeNull();
    expect(selected2).not.toBeNull();
    expect(selected1?.id).toBe(selected2?.id);
  });

  it('handles build_stack and move_stack types', () => {
    const state = makeBaseGameState();
    state.currentPhase = 'movement';
    (state.board as any).stacks = {};

    const now = new Date();
    const buildMove: Move = {
      id: 'build-1',
      type: 'build_stack',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 0 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const moveRing: Move = {
      id: 'move-ring-1',
      type: 'move_stack',
      player: 1,
      from: { x: 1, y: 1 },
      to: { x: 2, y: 2 },
      timestamp: now,
      thinkTime: 0,
      moveNumber: 1,
    };

    const rng = makeSeededRng(1);
    const selected = chooseLocalMoveFromCandidates(1, state as any, [buildMove, moveRing], rng, 0);

    // Should select one of the simple movement types
    expect(['build_stack', 'move_stack']).toContain(selected?.type);
  });
});
