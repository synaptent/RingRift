import { GameEngine } from '../../src/server/game/GameEngine';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import {
  BoardType,
  GameState,
  Player,
  TimeControl,
  positionToString,
  Position,
} from '../../src/shared/types/game';

/**
 * Explicit S-invariant tests for a simple, hand-built position.
 *
 * Rules/reference:
 * - Compact rules §9 (progress invariant S = markers + collapsed + eliminated)
 * - `ringrift_compact_rules.md` §9 commentary
 *
 * These backend-focused tests complement the heavier, diagnostic
 * sandbox AI simulations by asserting that:
 *
 * 1. S is computed as M + C + E for a simple board.
 * 2. A canonical "marker → collapsed space + eliminated ring" style
 *    transition strictly increases S in a hand-constructed GameState.
 */

describe('ProgressSnapshot (S-invariant) – backend GameEngine (Rules §9)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createPlayers(): Player[] {
    return [
      {
        id: 'p1',
        username: 'P1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
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
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
  }

  it('Rules_9_SInvariant_basic_counts_backend', () => {
    const engine = new GameEngine('s-invariant-basic', boardType, createPlayers(), timeControl, false) as any;
    const state: GameState = engine.gameState as GameState;

    // Start from a clean board and construct a simple configuration:
    // - 1 marker for player 1
    // - 2 collapsed spaces
    // - totalRingsEliminated = 3
    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.eliminatedRings = {};
    (state as any).totalRingsEliminated = undefined;

    // One marker
    state.board.markers.set(key, { player: 1, position: pos, type: 'regular' });

    // Two collapsed spaces
    state.board.collapsedSpaces.set('1,0', 1);
    state.board.collapsedSpaces.set('2,0', 2);

    // Three eliminated rings via board summary
    state.board.eliminatedRings[1] = 1;
    state.board.eliminatedRings[2] = 2;

    const snap = computeProgressSnapshot(state);
    expect(snap.markers).toBe(1);
    expect(snap.collapsed).toBe(2);
    expect(snap.eliminated).toBe(3);
    expect(snap.S).toBe(1 + 2 + 3);
  });

  it('Rules_9_SInvariant_marker_collapse_increases_S_backend', () => {
    const engine = new GameEngine('s-invariant-collapse', boardType, createPlayers(), timeControl, false) as any;
    const state: GameState = engine.gameState as GameState;

    // Construct a tiny board position where:
    // - Player 1 has a marker at (0,0)
    // - No collapsed spaces
    // - No eliminated rings
    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.eliminatedRings = {};
    state.players.forEach((p) => {
      p.eliminatedRings = 0;
    });
    (state as any).totalRingsEliminated = undefined;

    state.board.markers.set(key, { player: 1, position: pos, type: 'regular' });

    const before = computeProgressSnapshot(state);
    expect(before).toEqual({ markers: 1, collapsed: 0, eliminated: 0, S: 1 });

    // Simulate a canonical progress step where the marker is collapsed
    // into territory and one ring is eliminated for player 1. This mirrors
    // the sort of structural progress made during line/territory processing.
    state.board.markers.delete(key);
    state.board.collapsedSpaces.set(key, 1);

    state.board.eliminatedRings[1] = 1;
    const p1 = state.players.find((p) => p.playerNumber === 1)!;
    p1.eliminatedRings = 1;
    (state as any).totalRingsEliminated = 1;

    const after = computeProgressSnapshot(state);
    expect(after.S).toBeGreaterThan(before.S);
    expect(after).toEqual({ markers: 0, collapsed: 1, eliminated: 1, S: 2 });
  });
});
