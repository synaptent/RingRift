import { readFileSync } from 'fs';

import { createInitialGameState } from '../../src/shared/engine/initialState';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { applyMoveForReplay } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { BoardType, GameState, Move, Player, TimeControl } from '../../src/shared/types/game';
import { snapshotFromGameState } from '../utils/stateSnapshots';

interface RawCase {
  caseId: string;
  boardType: BoardType;
  numPlayers: number;
  moves: Array<Record<string, unknown>>;
}

interface InputPayload {
  cases: RawCase[];
}

function makePlayers(numPlayers: number): Player[] {
  const players: Player[] = [];
  for (let i = 1; i <= numPlayers; i += 1) {
    players.push({
      id: `p${i}`,
      username: `AI ${i}`,
      type: 'ai',
      playerNumber: i,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
      aiDifficulty: 10,
    });
  }
  return players;
}

function makeTimeControl(): TimeControl {
  return {
    type: 'blitz',
    initialTime: 600,
    increment: 0,
  };
}

function activateState(state: GameState): GameState {
  return {
    ...state,
    gameStatus: 'active',
  };
}

function reviveMove(rawMove: Record<string, unknown>, index: number): Move {
  const timestamp =
    typeof rawMove.timestamp === 'string' ? new Date(rawMove.timestamp) : new Date(0);

  return {
    ...(rawMove as unknown as Move),
    id: typeof rawMove.id === 'string' ? rawMove.id : `ts-trace-${index + 1}`,
    timestamp,
  };
}

function runCase(rawCase: RawCase) {
  const rulesOptions = rawCase.numPlayers === 2 ? { swapRuleEnabled: false } : undefined;
  let state = activateState(
    createInitialGameState(
      rawCase.caseId,
      rawCase.boardType,
      makePlayers(rawCase.numPlayers),
      makeTimeControl(),
      false,
      1234,
      rulesOptions
    )
  );

  rawCase.moves.forEach((rawMove, index) => {
    const move = reviveMove(rawMove, index);
    state = applyMoveForReplay(state, move).nextState;
  });

  const progress = computeProgressSnapshot(state);
  return {
    caseId: rawCase.caseId,
    boardType: rawCase.boardType,
    numPlayers: rawCase.numPlayers,
    snapshot: snapshotFromGameState(rawCase.caseId, state),
    progress: {
      markers: progress.markers,
      collapsed: progress.collapsed,
      eliminated: progress.eliminated,
      S: progress.S,
    },
  };
}

function main(): void {
  const inputPath = process.argv[2];
  if (!inputPath) {
    throw new Error('Usage: ts-node ts_rules_config_trace_parity.ts <payload.json>');
  }

  const payload = JSON.parse(readFileSync(inputPath, 'utf8')) as InputPayload;
  const results = payload.cases.map(runCase);
  process.stdout.write(JSON.stringify({ results }));
}

try {
  main();
} catch (error) {
  const message = error instanceof Error ? (error.stack ?? error.message) : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
}
