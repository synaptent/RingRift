#!/usr/bin/env ts-node
/**
 * Export a single GameReplayDB game as a TypeScript GameRecord JSON fixture.
 *
 * This bridges Python self-play / golden GameReplayDBs into the TS golden
 * replay suite by constructing a best-effort GameRecord from the recorded
 * metadata and canonical Move JSON.
 *
 * Usage (from repo root):
 *
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-golden-game-record.ts \
 *     --db <path-to-canonical-db> \
 *     --game-id 121f8003-b363-4be1-92f2-3fbb247c0772 \
 *     --output tests/fixtures/golden-games/golden_square8_2p_121f8.json
 */

import fs from 'fs';
import path from 'path';

import type { BoardType, Move } from '../src/shared/types/game';
import type {
  GameRecord,
  GameRecordMetadata,
  FinalScore,
  PlayerRecordInfo,
} from '../src/shared/types/gameRecord';
import { moveToMoveRecord } from '../src/shared/types/gameRecord';
import { getSelfPlayGameService } from '../src/server/services/SelfPlayGameService';

interface CliArgs {
  dbPath: string;
  gameId: string;
  outputPath: string;
}

function printUsage(): void {
  // eslint-disable-next-line no-console
  console.log(
    [
      'Usage:',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-golden-game-record.ts \\',
      '    --db <path/to/GameReplayDB.db> \\',
      '    --game-id <game_id> \\',
      '    --output tests/fixtures/golden-games/<name>.json',
      '',
      'Example:',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-golden-game-record.ts \\',
      '    --db <path-to-canonical-db> \\',
      '    --game-id 121f8003-b363-4be1-92f2-3fbb247c0772 \\',
      '    --output tests/fixtures/golden-games/golden_square8_2p_121f8.json',
    ].join('\n')
  );
}

function parseArgs(argv: string[]): CliArgs | null {
  let dbPath: string | undefined;
  let gameId: string | undefined;
  let outputPath: string | undefined;

  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw.startsWith('--')) continue;

    const [flag, inlineValue] = raw.split('=', 2);
    const next = argv[i + 1];
    const value = inlineValue ?? (next && !next.startsWith('--') ? next : undefined);

    switch (flag) {
      case '--db':
        if (!value) {
          console.error('Missing value for --db');
          return null;
        }
        dbPath = value;
        if (!inlineValue && next === value) i += 1;
        break;
      case '--game-id':
        if (!value) {
          console.error('Missing value for --game-id');
          return null;
        }
        gameId = value;
        if (!inlineValue && next === value) i += 1;
        break;
      case '--output':
        if (!value) {
          console.error('Missing value for --output');
          return null;
        }
        outputPath = value;
        if (!inlineValue && next === value) i += 1;
        break;
      default:
        console.warn(`Ignoring unknown flag: ${flag}`);
        if (!inlineValue && next && !next.startsWith('--')) {
          i += 1;
        }
    }
  }

  if (!dbPath || !gameId || !outputPath) {
    return null;
  }

  return { dbPath, gameId, outputPath };
}

function buildFinalScoreFromPlayers(
  players: {
    playerNumber: number;
    finalEliminatedRings: number | null;
    finalTerritorySpaces: number | null;
  }[]
): FinalScore {
  const ringsEliminated: Record<number, number> = {};
  const territorySpaces: Record<number, number> = {};
  const ringsRemaining: Record<number, number> = {};

  for (const player of players) {
    const playerNumber = player.playerNumber;
    ringsEliminated[playerNumber] = player.finalEliminatedRings ?? 0;
    territorySpaces[playerNumber] = player.finalTerritorySpaces ?? 0;
    // We do not currently have final rings-in-hand from GameReplayDB; leave as 0.
    ringsRemaining[playerNumber] = 0;
  }

  return { ringsEliminated, territorySpaces, ringsRemaining };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  if (!args) {
    printUsage();
    process.exitCode = 1;
    return;
  }

  const dbPath = path.isAbsolute(args.dbPath) ? args.dbPath : path.join(process.cwd(), args.dbPath);
  const outputPath = path.isAbsolute(args.outputPath)
    ? args.outputPath
    : path.join(process.cwd(), args.outputPath);

  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  const service = getSelfPlayGameService();
  const detail = service.getGame(dbPath, args.gameId);

  if (!detail) {
    console.error(`Game ${args.gameId} not found in DB ${dbPath}`);
    process.exitCode = 1;
    return;
  }

  const players: PlayerRecordInfo[] = detail.players.map((p) => {
    const player: PlayerRecordInfo = {
      playerNumber: p.playerNumber,
      username: `P${p.playerNumber + 1}`,
      playerType: p.playerType === 'human' ? 'human' : 'ai',
    };

    if (p.aiDifficulty !== null && p.aiDifficulty !== undefined) {
      player.aiDifficulty = p.aiDifficulty;
    }
    if (p.aiType !== null && p.aiType !== undefined) {
      player.aiType = p.aiType;
    }

    return player;
  });

  const moves = detail.moves.map((m) => moveToMoveRecord(m.move as Move));

  const finalScore = buildFinalScoreFromPlayers(
    detail.players.map((p) => ({
      playerNumber: p.playerNumber,
      finalEliminatedRings: p.finalEliminatedRings,
      finalTerritorySpaces: p.finalTerritorySpaces,
    }))
  );

  const nowIso = new Date().toISOString();

  const metadata: GameRecordMetadata = {
    recordVersion: '1.0.0',
    createdAt: nowIso,
    source: 'self_play',
    // We treat the DB path + gameId as provenance tags for debugging.
    tags: ['golden_candidate', `db:${path.basename(dbPath)}`, `game_id:${detail.gameId}`],
  };

  const record: GameRecord = {
    id: detail.gameId,
    boardType: detail.boardType as BoardType,
    numPlayers: detail.numPlayers,
    isRated: false,
    players,
    ...(detail.winner !== null && detail.winner !== undefined && { winner: detail.winner }),
    // Best-effort outcome; golden replay tests only assert winner consistency.
    outcome: detail.winner !== null && detail.winner !== undefined ? 'ring_elimination' : 'draw',
    finalScore,
    startedAt: detail.createdAt,
    endedAt: detail.completedAt ?? detail.createdAt,
    totalMoves: detail.totalMoves,
    totalDurationMs: detail.durationMs ?? 0,
    moves,
    metadata,
  };

  fs.writeFileSync(outputPath, JSON.stringify(record, null, 2), 'utf-8');

  // eslint-disable-next-line no-console
  console.log(
    `[export-golden-game-record] Wrote GameRecord fixture for game ${detail.gameId} from ${dbPath} -> ${outputPath}`
  );
}

void main();
