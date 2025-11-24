import {
  BoardType,
  Player,
  TimeControl,
  GameState,
  Position,
  Territory,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  enumerateTerritoryEliminationMoves,
  applyEliminateRingsFromStackDecision,
} from '../../src/shared/engine/territoryDecisionHelpers';

describe('territoryDecisionHelpers – shared territory decision enumeration and application', () => {
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
      ringsInHand: 0,
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
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createEmptyState(id: string): GameState {
    const state = createInitialGameState(
      id,
      boardType,
      players,
      timeControl
    ) as unknown as GameState;

    state.currentPlayer = 1;
    state.board.stacks.clear();
    state.board.markers.clear();
    state.board.collapsedSpaces.clear();
    state.board.territories = new Map();
    state.board.eliminatedRings = {};
    state.board.formedLines = [];
    state.totalRingsEliminated = 0;
    state.players = state.players.map((p) => ({
      ...p,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));
    return state;
  }

  function snapshotS(state: GameState): number {
    return computeProgressSnapshot(state as any).S;
  }

  it('enumerateProcessTerritoryRegionMoves filters regions by self-elimination prerequisite and encodes geometry', () => {
    const state = createEmptyState('territory-enum');
    const board = state.board;

    // Two stacks for player 1 inside a larger region; only the smaller
    // region that does not cover all stacks satisfies the self-elimination
    // prerequisite (must control at least one stack outside the region).
    const a: Position = { x: 0, y: 0 };
    const b: Position = { x: 1, y: 0 };
    const aKey = positionToString(a);
    const bKey = positionToString(b);

    board.stacks.set(aKey, {
      position: a,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as any);
    board.stacks.set(bKey, {
      position: b,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as any);

    const regionWithOutsideStack: Territory = {
      spaces: [a],
      controllingPlayer: 0,
      isDisconnected: true,
    };

    const regionWithoutOutsideStack: Territory = {
      spaces: [a, b],
      controllingPlayer: 0,
      isDisconnected: true,
    };

    const moves = enumerateProcessTerritoryRegionMoves(state, 1, {
      testOverrideRegions: [regionWithOutsideStack, regionWithoutOutsideStack],
    });

    expect(moves).toHaveLength(1);
    const move = moves[0];
    expect(move.type).toBe('process_territory_region');
    expect(move.player).toBe(1);
    expect(move.disconnectedRegions).toBeDefined();
    expect(move.disconnectedRegions!.length).toBe(1);

    const region = move.disconnectedRegions![0];
    expect(region.spaces.length).toBe(1);
    expect(region.spaces[0]).toEqual(a);
    expect(move.to).toEqual(a);
  });

  it('applyProcessTerritoryRegionDecision eliminates all internal rings, collapses region, credits gains, and sets pendingSelfElimination', () => {
    const state = createEmptyState('territory-apply-region');
    const board = state.board;

    // Region consists of two interior spaces; stacks inside the region may
    // belong to any player but all eliminations and territory gain are
    // credited to the acting player (player 1).
    const p1a: Position = { x: 0, y: 0 };
    const p2a: Position = { x: 1, y: 0 };
    const outside: Position = { x: 3, y: 3 };

    const p1aKey = positionToString(p1a);
    const p2aKey = positionToString(p2a);
    const outsideKey = positionToString(outside);

    // Two rings for player 1 inside the region.
    board.stacks.set(p1aKey, {
      position: p1a,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    // Three rings for player 2 inside the same region; these eliminations
    // are still credited to player 1 under the current rules.
    board.stacks.set(p2aKey, {
      position: p2a,
      rings: [2, 2, 2],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: 2,
    } as any);

    // Single stack for player 1 outside the region to satisfy the
    // self-elimination prerequisite.
    board.stacks.set(outsideKey, {
      position: outside,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as any);

    const region: Territory = {
      spaces: [p1a, p2a],
      controllingPlayer: 0,
      isDisconnected: true,
    };

    const move = {
      id: 'process-region-test',
      type: 'process_territory_region' as const,
      player: 1,
      to: p1a,
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const beforeS = snapshotS(state);
    const outcome = applyProcessTerritoryRegionDecision(state, move);
    const next = outcome.nextState;
    const afterS = snapshotS(next);

    expect(outcome.pendingSelfElimination).toBe(true);
    expect(outcome.processedRegion.spaces).toHaveLength(2);

    // All stacks inside the region have been eliminated.
    expect(next.board.stacks.has(p1aKey)).toBe(false);
    expect(next.board.stacks.has(p2aKey)).toBe(false);
    // Stack outside the region remains.
    expect(next.board.stacks.has(outsideKey)).toBe(true);

    // Territory gain: all interior spaces (no border markers in this setup).
    const p1After = next.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.territorySpaces).toBe(2);
    const p2After = next.players.find((p) => p.playerNumber === 2)!;
    expect(p2After.territorySpaces).toBe(0);

    // Eliminated rings: 2 (player 1) + 3 (player 2) credited entirely to player 1.
    expect(next.board.eliminatedRings[1]).toBe(5);
    expect(next.totalRingsEliminated).toBe(5);
    expect(p1After.eliminatedRings).toBe(5);

    // Region spaces are now collapsed territory for player 1.
    expect(next.board.collapsedSpaces.get(p1aKey)).toBe(1);
    expect(next.board.collapsedSpaces.get(p2aKey)).toBe(1);

    // S-invariant (markers + collapsed + eliminated) is non-decreasing.
    expect(afterS).toBeGreaterThanOrEqual(beforeS);
  });

  it('enumerateTerritoryEliminationMoves surfaces one elimination per eligible stack and applyEliminateRingsFromStackDecision updates caps and S-invariant', () => {
    const state = createEmptyState('territory-elim');
    const board = state.board;

    const a: Position = { x: 0, y: 0 };
    const b: Position = { x: 1, y: 0 };
    const c: Position = { x: 2, y: 0 };

    const aKey = positionToString(a);
    const bKey = positionToString(b);
    const cKey = positionToString(c);

    // Two stacks controlled by player 1 with non-zero caps.
    board.stacks.set(aKey, {
      position: a,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);

    board.stacks.set(bKey, {
      position: b,
      rings: [1, 2, 1],
      stackHeight: 3,
      capHeight: 1, // only the top ring belongs to player 1
      controllingPlayer: 1,
    } as any);

    // Opponent stack should not generate an elimination move for player 1.
    board.stacks.set(cKey, {
      position: c,
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    } as any);

    // Ensure board-level elimination counters start from a known baseline.
    state.board.eliminatedRings = { 1: 0, 2: 0 };
    state.totalRingsEliminated = 0;
    state.players = state.players.map((p) =>
      p.playerNumber === 1 ? { ...p, eliminatedRings: 0 } : p
    );

    const beforeS = snapshotS(state);
    const moves = enumerateTerritoryEliminationMoves(state, 1);

    // Exactly one elimination move per eligible stack controlled by player 1.
    expect(moves.length).toBe(2);
    const ids = moves.map((m) => m.id).sort();
    expect(ids).toEqual([`eliminate-${aKey}`, `eliminate-${bKey}`].sort());

    moves.forEach((m) => {
      expect(m.type).toBe('eliminate_rings_from_stack');
      expect(m.player).toBe(1);
      expect(m.to).toBeDefined();
      expect(m.eliminationFromStack).toBeDefined();
      const snapshot = m.eliminationFromStack!;
      expect(snapshot.capHeight).toBeGreaterThan(0);
      expect(snapshot.totalHeight).toBeGreaterThanOrEqual(snapshot.capHeight);
    });

    // Apply elimination from stack A and verify counters and geometry.
    const aMove = moves.find((m) => positionToString(m.to!) === aKey)!;
    const outcome = applyEliminateRingsFromStackDecision(state, aMove);
    const next = outcome.nextState;
    const afterS = snapshotS(next);

    // Stack at A is fully removed (pure cap).
    expect(next.board.stacks.has(aKey)).toBe(false);

    // Board and player elimination counters updated by the cap height (2).
    expect(next.board.eliminatedRings[1]).toBe(2);
    const p1After = next.players.find((p) => p.playerNumber === 1)!;
    expect(p1After.eliminatedRings).toBe(2);
    expect(next.totalRingsEliminated).toBe(2);

    // Other stacks remain untouched.
    expect(next.board.stacks.has(bKey)).toBe(true);
    expect(next.board.stacks.has(cKey)).toBe(true);

    // S-invariant is non-decreasing after elimination.
    expect(afterS).toBeGreaterThanOrEqual(beforeS);
  });
});
