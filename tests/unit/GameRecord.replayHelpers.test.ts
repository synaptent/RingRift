import { reconstructStateAtMove } from '../../src/shared/engine';
import type { GameRecord } from '../../src/shared/types/gameRecord';
import { isFSMOrchestratorActive } from '../../src/shared/utils/envFlags';

describe('reconstructStateAtMove', () => {
  // TODO: FSM validation is stricter - rejects no_movement_action when valid moves exist.
  // These tests use synthetic move sequences that don't satisfy FSM guard conditions.
  // Update test data to use valid FSM-compliant sequences.
  if (isFSMOrchestratorActive()) {
    it.skip('Skipping - Test data uses FSM-invalid move sequences', () => {});
    return;
  }
  const baseRecord: GameRecord = {
    id: 'test-game',
    boardType: 'square8',
    numPlayers: 2,
    rngSeed: 123,
    isRated: false,
    players: [
      {
        number: 1,
        name: 'P1',
        type: 'ai',
        rating: undefined,
        tags: [],
      },
      {
        number: 2,
        name: 'P2',
        type: 'ai',
        rating: undefined,
        tags: [],
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
    totalMoves: 2,
    totalDurationMs: 0,
    // Per RR-CANON-R075: All phases must be visited with explicit moves.
    // After place_ring, the turn progresses through movement, line_processing,
    // and territory_processing with explicit no-op moves before next player.
    moves: [
      {
        moveNumber: 0,
        player: 1,
        type: 'place_ring',
        from: undefined,
        to: { x: 3, y: 3 },
        placementCount: 1,
        thinkTimeMs: 0,
      },
      {
        moveNumber: 1,
        player: 1,
        type: 'no_movement_action',
        from: undefined,
        to: undefined,
        thinkTimeMs: 0,
      },
      {
        moveNumber: 2,
        player: 1,
        type: 'no_line_action',
        from: undefined,
        to: undefined,
        thinkTimeMs: 0,
      },
      {
        moveNumber: 3,
        player: 1,
        type: 'no_territory_action',
        from: undefined,
        to: undefined,
        thinkTimeMs: 0,
      },
      {
        moveNumber: 4,
        player: 2,
        type: 'place_ring',
        from: undefined,
        to: { x: 4, y: 4 },
        placementCount: 1,
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
    // moveIndex 1 = after place_ring by player 1
    const stateAfterFirst = reconstructStateAtMove(baseRecord, 1);
    // moveIndex 5 = after place_ring by player 2 (move indices 2-4 are no-op phase moves)
    const stateAfterSecond = reconstructStateAtMove(baseRecord, 5);

    const positionsAfterFirst = Array.from(stateAfterFirst.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );
    const positionsAfterSecond = Array.from(stateAfterSecond.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );

    expect(positionsAfterFirst.sort()).toEqual(['3,3'].sort());
    expect(positionsAfterSecond.sort()).toEqual(['3,3', '4,4'].sort());
  });

  it('clamps moveIndex beyond the number of moves', () => {
    const state = reconstructStateAtMove(baseRecord, 10);
    const positions = Array.from(state.board.stacks.values()).map(
      (s) => `${s.position.x},${s.position.y}`
    );
    expect(positions.sort()).toEqual(['3,3', '4,4'].sort());
  });

  it('throws for negative moveIndex', () => {
    expect(() => reconstructStateAtMove(baseRecord, -1)).toThrow();
  });
});
