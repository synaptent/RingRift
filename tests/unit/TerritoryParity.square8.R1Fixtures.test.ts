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
import { applyProcessLineDecision } from '../../src/shared/engine/aggregates/LineAggregate';

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

describe('TerritoryParity.square8.R1Fixtures – mini-line collapse parity', () => {
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
});
