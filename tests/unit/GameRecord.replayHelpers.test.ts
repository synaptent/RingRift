// Use legacy replay helper for recorded games that may have phase transitions
// differing from current canonical spec. Canonical replay is tested elsewhere.
// TODO: Create separate test suite for canonical-only replays once golden games
// are regenerated with canonical Python engine.
import { reconstructStateAtMoveLegacy as reconstructStateAtMove } from '../../src/shared/engine/legacy/legacyReplayHelpers';
import type { GameRecord } from '../../src/shared/types/gameRecord';

describe('reconstructStateAtMove (legacy compatibility)', () => {
  // FSM-compatible test data: uses actual move_stack instead of no_movement_action
  // because FSM rejects no_movement_action when valid moves exist.
  const baseRecord: GameRecord = {
    id: 'test-game',
    boardType: 'square8',
    numPlayers: 2,
    rngSeed: 123,
    isRated: false,
    players: [
      {
        playerNumber: 1,
        username: 'P1',
        playerType: 'ai',
      },
      {
        playerNumber: 2,
        username: 'P2',
        playerType: 'ai',
      },
    ],
    winner: undefined,
    outcome: 'draw',
    finalScore: {
      ringsEliminated: {},
      territorySpaces: {},
      ringsRemaining: {},
    },
    startedAt: new Date().toISOString(),
    endedAt: new Date().toISOString(),
    totalMoves: 5,
    totalDurationMs: 0,
    // Per RR-CANON-R075: All phases must be visited with explicit moves.
    // After place_ring, the turn uses an actual move, then line/territory no-ops.
    moves: [
      {
        moveNumber: 1,
        player: 1,
        type: 'place_ring',
        from: undefined,
        to: { x: 3, y: 3 },
        thinkTimeMs: 0,
      },
      {
        moveNumber: 2,
        player: 1,
        type: 'move_stack',
        from: { x: 3, y: 3 },
        to: { x: 3, y: 4 },
        thinkTimeMs: 0,
      },
      {
        moveNumber: 3,
        player: 1,
        type: 'no_line_action',
        from: undefined,
        to: { x: 0, y: 0 },
        thinkTimeMs: 0,
      },
      {
        moveNumber: 4,
        player: 1,
        type: 'no_territory_action',
        from: undefined,
        to: { x: 0, y: 0 },
        thinkTimeMs: 0,
      },
      {
        moveNumber: 5,
        player: 2,
        type: 'place_ring',
        from: undefined,
        to: { x: 4, y: 4 },
        thinkTimeMs: 0,
      },
    ],
    metadata: {
      recordVersion: '1.0',
      createdAt: new Date().toISOString(),
      source: 'self_play',
      tags: [],
    },
  };

  it('returns initial state when moveIndex is 0', () => {
    const state = reconstructStateAtMove(baseRecord, 0);
    expect(state.currentPhase).toBe('ring_placement');
  });

  it('applies moves up to the requested index', () => {
    // moveIndex 1 = after place_ring at (3,3) by player 1
    const stateAfterFirst = reconstructStateAtMove(baseRecord, 1);
    // moveIndex 2 = after move_stack to (3,4) by player 1
    const stateAfterMove = reconstructStateAtMove(baseRecord, 2);
    // moveIndex 5 = after place_ring by player 2 (move indices 3-4 are no-op phase moves)
    const stateAfterSecond = reconstructStateAtMove(baseRecord, 5);

    const positionsAfterFirst = Array.from(stateAfterFirst.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );
    const positionsAfterMove = Array.from(stateAfterMove.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );
    const positionsAfterSecond = Array.from(stateAfterSecond.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );

    expect(positionsAfterFirst.sort()).toEqual(['3,3'].sort());
    expect(positionsAfterMove.sort()).toEqual(['3,4'].sort()); // Stack moved to 3,4
    expect(positionsAfterSecond.sort()).toEqual(['3,4', '4,4'].sort());
  });

  it('clamps moveIndex beyond the number of moves', () => {
    const state = reconstructStateAtMove(baseRecord, 10);
    const positions = Array.from(state.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );
    expect(positions.sort()).toEqual(['3,4', '4,4'].sort()); // Stack is at 3,4 after movement
  });

  it('throws for negative moveIndex', () => {
    expect(() => reconstructStateAtMove(baseRecord, -1)).toThrow();
  });
});
