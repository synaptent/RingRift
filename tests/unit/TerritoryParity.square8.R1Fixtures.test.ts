/**
 * @semantic-anchor TerritoryParity.square8.R1Fixtures
 * @rules-level-counterparts
 *   - tests/unit/territoryProcessing.shared.test.ts (territory region processing)
 *   - tests/unit/territoryDecisionHelpers.shared.test.ts (territory decisions)
 *   - tests/unit/territoryProcessing.branchCoverage.test.ts (eligibility logic)
 *   - RR-CANON-R070–R082 (territory processing rules)
 *   - RR-CANON-R082 (elimination cost: entire cap from eligible stack only)
 * @classification Rules-level parity with fixtures
 */
import {
  BoardType,
  TimeControl,
  Player,
  GameState,
  Position,
  Move,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import {
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
} from '../../src/shared/engine/aggregates/LineAggregate';

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

function makeEmptyState(id: string): GameState {
  const state = createInitialGameState(id, boardType, players, timeControl) as unknown as GameState;

  state.board.stacks = new Map();
  state.board.markers = new Map();
  state.board.collapsedSpaces = new Map();
  state.board.territories = new Map();
  state.board.eliminatedRings = {};
  state.board.formedLines = [];

  state.moveHistory = [];
  state.history = [];

  state.players = state.players.map((p) => ({
    ...p,
    ringsInHand: 0,
    eliminatedRings: 0,
    territorySpaces: 0,
  }));

  state.totalRingsEliminated = 0;

  return state;
}

function pos(x: number, y: number): Position {
  return { x, y };
}

function addStack(state: GameState, at: Position, rings: number[]): void {
  const key = positionToString(at);
  state.board.stacks.set(key, {
    position: at,
    rings,
    stackHeight: rings.length,
    capHeight: computeCapHeight(rings),
    controllingPlayer: rings[0],
  } as any);
}

function computeCapHeight(rings: number[]): number {
  if (rings.length === 0) return 0;
  const top = rings[0];
  let count = 0;
  for (const r of rings) {
    if (r === top) {
      count += 1;
    } else {
      break;
    }
  }
  return count;
}

function build707fK45PreProcessLineState(): GameState {
  const state = makeEmptyState('707fd580-k45');
  const board = state.board;

  // Collapsed spaces at k=45 (both engines agree).
  board.collapsedSpaces.set('1,5', 2);
  board.collapsedSpaces.set('5,0', 2);
  board.collapsedSpaces.set('6,0', 2);

  // Markers at k=45 (TS view from state bundle).
  const markers: Array<{ x: number; y: number; player: number }> = [
    { x: 0, y: 7, player: 2 },
    { x: 1, y: 0, player: 2 },
    { x: 2, y: 3, player: 1 },
    { x: 5, y: 4, player: 2 },
    { x: 5, y: 5, player: 2 },
    { x: 5, y: 7, player: 1 },
    { x: 6, y: 3, player: 2 },
    { x: 6, y: 4, player: 2 },
    { x: 6, y: 5, player: 2 },
    { x: 7, y: 0, player: 2 },
  ];
  for (const m of markers) {
    const p = pos(m.x, m.y);
    board.markers.set(positionToString(p), {
      player: m.player,
      position: p,
      type: 'regular',
    } as any);
  }

  // Stacks at k=45 (TS view from state bundle).
  addStack(state, pos(1, 1), [2]);
  addStack(state, pos(1, 6), [2, 1, 1]);
  addStack(state, pos(2, 6), [1, 2, 2]);
  addStack(state, pos(2, 7), [1, 1, 1]);
  addStack(state, pos(3, 7), [1, 1]);
  addStack(state, pos(6, 1), [2, 2, 2, 1, 1, 2, 2]);

  board.size = 8;
  board.type = 'square8';
  board.eliminatedRings = { 1: 1, 2: 3 };

  // Players at k=45 (TS view).
  state.players = state.players.map((p) => {
    if (p.playerNumber === 1) {
      return {
        ...p,
        ringsInHand: 7,
        eliminatedRings: 1,
        territorySpaces: 0,
      };
    }
    if (p.playerNumber === 2) {
      return {
        ...p,
        ringsInHand: 6,
        eliminatedRings: 3,
        territorySpaces: 0,
      };
    }
    return p;
  });

  state.currentPhase = 'line_processing';
  state.currentPlayer = 2;
  state.boardType = 'square8';
  state.totalRingsEliminated = 4;
  state.territoryVictoryThreshold = 33;
  state.victoryThreshold = 19;

  return state;
}

describe('TerritoryParity.square8.R1Fixtures – mini-line and reward collapse parity', () => {
  it('Game 707fd580… k=46: process_line collapses 6,3–6,5 into P2 territory like Python', () => {
    const state = build707fK45PreProcessLineState();

    const before = computeProgressSnapshot(state as any);
    const beforeS = before.S;

    const linePositions: Position[] = [pos(6, 3), pos(6, 4), pos(6, 5)];

    const move: Move = {
      id: 'process-line-0-6,3',
      type: 'process_line',
      player: 2,
      to: pos(6, 3),
      formedLines: [
        {
          positions: linePositions,
          player: 2,
          length: 3,
          direction: { x: 0, y: 1 },
        },
      ],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 46,
    };

    const { nextState, pendingLineRewardElimination } = applyProcessLineDecision(state, move);

    // Mini-line (length 3) on 2p square8: no mandatory elimination reward.
    expect(pendingLineRewardElimination).toBe(false);

    const collapsed = nextState.board.collapsedSpaces;
    // Pre-existing collapsed spaces remain.
    expect(collapsed.get('1,5')).toBe(2);
    expect(collapsed.get('5,0')).toBe(2);
    expect(collapsed.get('6,0')).toBe(2);

    // Line cells are newly collapsed to P2, matching Python k=46.
    expect(collapsed.get('6,3')).toBe(2);
    expect(collapsed.get('6,4')).toBe(2);
    expect(collapsed.get('6,5')).toBe(2);

    const markers = nextState.board.markers;
    expect(markers.has('6,3')).toBe(false);
    expect(markers.has('6,4')).toBe(false);
    expect(markers.has('6,5')).toBe(false);

    const p1 = nextState.players.find((p) => p.playerNumber === 1)!;
    const p2 = nextState.players.find((p) => p.playerNumber === 2)!;

    expect(p1.territorySpaces).toBe(0);
    expect(p2.territorySpaces).toBe(3);

    // Eliminated rings are unchanged by mini-line collapse.
    expect(nextState.board.eliminatedRings[1]).toBe(1);
    expect(nextState.board.eliminatedRings[2]).toBe(3);
    expect(p1.eliminatedRings).toBe(1);
    expect(p2.eliminatedRings).toBe(3);

    const after = computeProgressSnapshot(nextState as any);
    const afterS = after.S;

    // S-invariant (markers + collapsed + eliminated) is preserved.
    expect(afterS).toBe(beforeS);
  });

  it('Game 85ecc4fa… k=32: choose_line_reward with collapsedMarkers collapses only 6,1–6,3 like Python', () => {
    const state = makeEmptyState('85ecc4fa-k31');
    const board = state.board;

    // Pre-existing collapsed spaces at k=31 (3,3 and 5,6 are already territory for P1).
    board.collapsedSpaces.set('3,3', 1);
    board.collapsedSpaces.set('5,6', 1);

    // Markers forming a length-4 vertical line at x=6, y=0–3 for P1.
    const lineMarkers: Array<{ x: number; y: number }> = [
      { x: 6, y: 0 },
      { x: 6, y: 1 },
      { x: 6, y: 2 },
      { x: 6, y: 3 },
    ];
    for (const m of lineMarkers) {
      const p = pos(m.x, m.y);
      board.markers.set(positionToString(p), {
        player: 1,
        position: p,
        type: 'regular',
      } as any);
    }

    // Players at k=31: P1 has no territory yet, P2 none.
    state.players = state.players.map((p) => {
      if (p.playerNumber === 1) {
        return {
          ...p,
          ringsInHand: 0,
          eliminatedRings: 1,
          territorySpaces: 0,
        };
      }
      if (p.playerNumber === 2) {
        return {
          ...p,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        };
      }
      return p;
    });

    state.currentPhase = 'line_processing';
    state.currentPlayer = 1;
    state.boardType = 'square8';
    state.totalRingsEliminated = 1;
    state.territoryVictoryThreshold = 33;
    state.victoryThreshold = 19;

    const before = computeProgressSnapshot(state as any);
    const beforeS = before.S;

    // Exact-length line (length 4) for 2p square8; Python fixture at k=32
    // uses CHOOSE_LINE_REWARD with collapsed_markers = [6,1], [6,2], [6,3].
    const fullLine: Position[] = [pos(6, 0), pos(6, 1), pos(6, 2), pos(6, 3)];
    const collapsedSubset: Position[] = [pos(6, 1), pos(6, 2), pos(6, 3)];

    const move: Move = {
      id: 'choose-line-reward-0-6,0-min-1',
      type: 'choose_line_reward',
      player: 1,
      to: pos(6, 0),
      formedLines: [
        {
          positions: fullLine,
          player: 1,
          length: 4,
          direction: { x: 0, y: 1 },
        },
      ],
      collapsedMarkers: collapsedSubset,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 32,
    };

    const { nextState, pendingLineRewardElimination } = applyChooseLineRewardDecision(state, move);

    // Minimum-collapse variant must NOT force a self-elimination reward even
    // though the underlying line length equals the required threshold.
    expect(pendingLineRewardElimination).toBe(false);

    const collapsed = nextState.board.collapsedSpaces;
    // Pre-existing collapsed spaces remain.
    expect(collapsed.get('3,3')).toBe(1);
    expect(collapsed.get('5,6')).toBe(1);

    // Only the subset 6,1–6,3 is collapsed to territory for P1, matching Python.
    expect(collapsed.get('6,1')).toBe(1);
    expect(collapsed.get('6,2')).toBe(1);
    expect(collapsed.get('6,3')).toBe(1);
    // 6,0 remains a marker (not collapsed).
    expect(collapsed.has('6,0')).toBe(false);

    const markersAfter = nextState.board.markers;
    expect(markersAfter.has('6,0')).toBe(true);
    expect(markersAfter.has('6,1')).toBe(false);
    expect(markersAfter.has('6,2')).toBe(false);
    expect(markersAfter.has('6,3')).toBe(false);

    const p1 = nextState.players.find((p) => p.playerNumber === 1)!;
    const p2 = nextState.players.find((p) => p.playerNumber === 2)!;

    // P1 gains exactly 3 territory spaces from the subset, P2 unaffected.
    expect(p1.territorySpaces).toBe(3);
    expect(p2.territorySpaces).toBe(0);

    // Eliminated rings are unchanged by this line reward choice.
    expect(nextState.board.eliminatedRings[1]).toBeUndefined();
    expect(nextState.board.eliminatedRings[2]).toBeUndefined();
    expect(p1.eliminatedRings).toBe(1);
    expect(p2.eliminatedRings).toBe(0);

    const after = computeProgressSnapshot(nextState as any);
    const afterS = after.S;

    // S-invariant (markers + collapsed + eliminated) is preserved.
    expect(afterS).toBe(beforeS);
  });
});
