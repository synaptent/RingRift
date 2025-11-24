import fs from 'fs';
import path from 'path';

import type { GameState, BoardState, Player, RingStack } from '../../src/shared/types/game';
import {
  evaluateHeuristicState,
  getHeuristicWeightsTS,
} from '../../src/shared/engine/heuristicEvaluation';

interface HeuristicFixtureState {
  id: string;
  playerNumber: number;
  gameState: {
    boardType: GameState['boardType'];
    gameStatus: GameState['gameStatus'];
    currentPlayer: number;
    board: {
      type: BoardState['type'];
      size: number;
      stacks: {
        [key: string]: {
          rings: number[];
          stackHeight: number;
          capHeight: number;
          controllingPlayer: number;
        };
      };
    };
    players: Array<{
      playerNumber: number;
      type: Player['type'];
      ringsInHand: number;
      eliminatedRings: number;
      territorySpaces: number;
    }>;
  };
}

interface HeuristicFixture {
  version: string;
  description?: string;
  profileId?: string;
  states: HeuristicFixtureState[];
  orderings: Array<{
    profileId?: string;
    better: string;
    worse: string;
  }>;
}

function loadFixture(filename: string): HeuristicFixture {
  const fixturePath = path.resolve(__dirname, '../fixtures/heuristic/v1', filename);
  const raw = fs.readFileSync(fixturePath, 'utf8');
  return JSON.parse(raw) as HeuristicFixture;
}

function buildBoard(snapshot: HeuristicFixtureState['gameState']['board']): BoardState {
  const stacks = new Map<string, RingStack>();

  for (const [key, entry] of Object.entries(snapshot.stacks)) {
    const [xStr, yStr, zStr] = key.split(',');
    const x = Number(xStr);
    const y = Number(yStr);
    const z = zStr !== undefined ? Number(zStr) : undefined;

    const stack: RingStack = {
      position: z !== undefined ? { x, y, z } : { x, y },
      rings: [...entry.rings],
      stackHeight: entry.stackHeight,
      capHeight: entry.capHeight,
      controllingPlayer: entry.controllingPlayer,
    } as RingStack;

    stacks.set(key, stack);
  }

  return {
    type: snapshot.type,
    size: snapshot.size,
    stacks,
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
  } as unknown as BoardState;
}

function buildPlayers(snapshot: HeuristicFixtureState['gameState']): Player[] {
  const players: Player[] = snapshot.players.map((p) => {
    const base: Player = {
      id: `p${p.playerNumber}`,
      username: `p${p.playerNumber}`,
      type: p.type,
      playerNumber: p.playerNumber,
      isReady: true,
      timeRemaining: 0,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
    } as unknown as Player;

    return base;
  });

  return players.sort((a, b) => a.playerNumber - b.playerNumber);
}

function buildGameState(entry: HeuristicFixtureState): GameState {
  const snap = entry.gameState;
  const board = buildBoard(snap.board);
  const players = buildPlayers(snap);

  const base: GameState = {
    id: entry.id,
    boardType: snap.boardType,
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: snap.currentPlayer,
    moveHistory: [],
    history: [],
    timeControl: { initialTime: 0, increment: 0, type: 'rapid' },
    spectators: [],
    gameStatus: snap.gameStatus,
    createdAt: new Date(0),
    lastMoveAt: new Date(0),
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0,
  } as unknown as GameState;

  return base;
}

/**
 * Cross-language heuristic parity scaffold for TS side.
 *
 * For each (better, worse) pair in the shared fixture, assert that the
 * TS-side evaluateHeuristicState gives a strictly higher score to the
 * `better` state for the indicated playerNumber under the requested
 * heuristic profile. Python tests load the same fixture and assert the
 * same ordering for HeuristicAI.
 */
describe('heuristic parity fixtures – TS evaluator ordering', () => {
  it('respects ordering constraints from square8_2p_simple_stacks.v1.json', () => {
    const fixture = loadFixture('square8_2p_simple_stacks.v1.json');

    const statesById = new Map<string, HeuristicFixtureState>();
    for (const s of fixture.states) {
      statesById.set(s.id, s);
    }

    for (const ordering of fixture.orderings) {
      const better = statesById.get(ordering.better);
      const worse = statesById.get(ordering.worse);

      if (!better || !worse) {
        throw new Error(
          `Unknown state id in heuristic fixture: ${ordering.better} or ${ordering.worse}`
        );
      }

      const betterState = buildGameState(better);
      const worseState = buildGameState(worse);
      const playerNumber = better.playerNumber;

      const profileId = ordering.profileId ?? fixture.profileId ?? 'heuristic_v1_balanced';
      const weights = getHeuristicWeightsTS(profileId);

      const betterScore = evaluateHeuristicState(betterState, playerNumber, weights);
      const worseScore = evaluateHeuristicState(worseState, playerNumber, weights);

      expect(betterScore).toBeGreaterThan(worseScore);
    }
  });
});
