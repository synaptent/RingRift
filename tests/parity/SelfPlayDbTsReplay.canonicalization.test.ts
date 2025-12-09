import type { Move } from '../../src/shared/types/game';
import type { SelfPlayMove } from '../../src/server/services/SelfPlayGameService';
import { buildCanonicalMoveFromSelfPlayRecord } from '../../scripts/selfplay-db-ts-replay';

describe('buildCanonicalMoveFromSelfPlayRecord', () => {
  it('preserves payload when moveType and raw move.type match for interactive moves', () => {
    const record: SelfPlayMove = {
      moveNumber: 3,
      turnNumber: 1,
      player: 1,
      phase: 'movement',
      moveType: 'move_stack',
      move: {
        id: 'm3',
        type: 'move_stack',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 0, y: 1 },
        timestamp: new Date('2025-01-01T00:00:00Z'),
        thinkTime: 123,
        moveNumber: 3,
      } as Move,
      thinkTimeMs: 123,
      engineEval: null,
    };

    const move = buildCanonicalMoveFromSelfPlayRecord(record, 3);

    expect(move.type).toBe('move_stack');
    expect(move.player).toBe(1);
    expect(move.from).toEqual({ x: 0, y: 0 });
    expect(move.to).toEqual({ x: 0, y: 1 });
    expect(move.moveNumber).toBe(3);
    expect(move.thinkTime).toBe(123);
  });

  it('uses canonical moveType and sanitizes payload for no_territory_action with stale geometry', () => {
    const record: SelfPlayMove = {
      moveNumber: 4,
      turnNumber: 1,
      player: 1,
      phase: 'territory_processing',
      moveType: 'no_territory_action',
      move: {
        // Legacy/stale payload that still claims to be a placement.
        id: 'legacy-m4',
        type: 'place_ring',
        player: 1,
        to: { x: 5, y: 5 },
        // Intentionally omit thinkTime to exercise fallback to thinkTimeMs.
        timestamp: '2025-01-01T00:00:04Z',
        moveNumber: 0,
      } as any,
      thinkTimeMs: 999,
      engineEval: null,
    };

    const move = buildCanonicalMoveFromSelfPlayRecord(record, 4);

    expect(move.type).toBe('no_territory_action');
    expect(move.player).toBe(1);
    // Geometry for no-op moves should be stripped / neutralised.
    expect(move.from).toBeUndefined();
    // captureTarget is not in Move interface but exists at runtime on some payloads.
    expect((move as any).captureTarget).toBeUndefined();
    expect(move.to).toEqual({ x: 0, y: 0 });
    expect(move.moveNumber).toBe(4);
    expect(move.thinkTime).toBe(999);
  });

  it('preserves forced_elimination move type and payload', () => {
    const record: SelfPlayMove = {
      moveNumber: 10,
      turnNumber: 3,
      player: 2,
      phase: 'forced_elimination',
      moveType: 'forced_elimination',
      move: {
        id: 'fe-10',
        type: 'forced_elimination',
        player: 2,
        to: { x: 3, y: 4 },
        eliminatedRings: [{ player: 2, count: 1 }],
        timestamp: new Date('2025-01-01T00:00:10Z'),
        thinkTime: 50,
        moveNumber: 10,
      } as any,
      thinkTimeMs: 50,
      engineEval: null,
    };

    const move = buildCanonicalMoveFromSelfPlayRecord(record, 10);

    expect(move.type).toBe('forced_elimination');
    expect(move.player).toBe(2);
    expect(move.to).toEqual({ x: 3, y: 4 });
    expect(move.eliminatedRings).toEqual([{ player: 2, count: 1 }]);
    expect(move.moveNumber).toBe(10);
    expect(move.thinkTime).toBe(50);
  });
});
