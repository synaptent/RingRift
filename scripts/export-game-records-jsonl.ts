#!/usr/bin/env ts-node
/**
 * Export completed Game records from Postgres as JSONL GameRecord lines.
 *
 * This is the TS/Node companion to the Python-side training exporters:
 * it lets you pull canonical GameRecord JSONL directly from the online
 * games database for use in analysis or training pipelines.
 *
 * Usage:
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \\
 *     --output data/game_records.jsonl
 *
 *   # Filter by board type
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \\
 *     --output data/square8_records.jsonl --board-type square8
 */

import fs from 'fs';
import path from 'path';

import { BoardType } from '@prisma/client';

import {
  gameRecordRepository,
  type GameRecordFilter,
} from '../src/server/services/GameRecordRepository';
import { connectDatabase, disconnectDatabase } from '../src/server/database/connection';

export interface CliArgs {
  output?: string;
  boardType?: BoardType;
  limit?: number;
  ratedOnly?: boolean;
  since?: Date;
  until?: Date;
}

function printUsage(): void {
  // eslint-disable-next-line no-console
  console.log(
    [
      'Usage: export-game-records-jsonl.ts [--output <path>] [--board-type square8|square19|hexagonal]',
      '                                      [--limit <n>] [--rated-only] [--since <iso>] [--until <iso>]',
      '',
      'Examples:',
      '  # Basic export with default output location',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts',
      '',
      '  # Filter by board type and rated games only',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \\',
      '    --output data/square8_rated.jsonl --board-type square8 --rated-only --limit 100',
    ].join('\n')
  );
}

export function parseArgs(argv: string[]): CliArgs | null {
  let output: string | undefined;
  let boardType: BoardType | undefined;
  let limit: number | undefined;
  let ratedOnly = false;
  let since: Date | undefined;
  let until: Date | undefined;

  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw.startsWith('--')) continue;

    const [flag, valueMaybe] = raw.split('=', 2);
    const next = argv[i + 1];
    const value = valueMaybe ?? (next && !next.startsWith('--') ? next : undefined);

    switch (flag) {
      case '--output':
        if (!value) {
          console.error('Missing value for --output');
          return null;
        }
        output = value;
        if (!valueMaybe && next === value) {
          i += 1;
        }
        break;
      case '--board-type':
        if (!value) {
          console.error('Missing value for --board-type');
          return null;
        }
        if (value !== 'square8' && value !== 'square19' && value !== 'hexagonal') {
          console.error(`Invalid --board-type value: ${value}`);
          return null;
        }
        boardType = value as BoardType;
        if (!valueMaybe && next === value) {
          i += 1;
        }
        break;
      case '--limit':
        if (!value) {
          console.error('Missing value for --limit');
          return null;
        }
        {
          const parsed = Number.parseInt(value, 10);
          if (!Number.isFinite(parsed) || parsed <= 0) {
            console.error(`Invalid --limit value: ${value}`);
            return null;
          }
          limit = parsed;
        }
        if (!valueMaybe && next === value) {
          i += 1;
        }
        break;
      case '--rated-only':
        if (valueMaybe || (next && !next.startsWith('--'))) {
          const rawVal = value ?? next;
          if (rawVal === 'false' || rawVal === '0') {
            ratedOnly = false;
          } else {
            ratedOnly = true;
          }
          if (!valueMaybe && next === rawVal) {
            i += 1;
          }
        } else {
          ratedOnly = true;
        }
        break;
      case '--since':
      case '--until':
        if (!value) {
          console.error(`Missing value for ${flag}`);
          return null;
        }
        {
          const date = new Date(value);
          if (Number.isNaN(date.getTime())) {
            console.error(`Invalid ${flag} datetime: ${value}`);
            return null;
          }
          if (flag === '--since') {
            since = date;
          } else {
            until = date;
          }
        }
        if (!valueMaybe && next === value) {
          i += 1;
        }
        break;
      default:
        console.warn(`Ignoring unknown flag: ${flag}`);
        if (!valueMaybe && next && !next.startsWith('--')) {
          i += 1;
        }
    }
  }

  const args: CliArgs = {
    ratedOnly,
  };

  if (output !== undefined) {
    args.output = output;
  }
  if (boardType !== undefined) {
    args.boardType = boardType;
  }
  if (limit !== undefined) {
    args.limit = limit;
  }
  if (since !== undefined) {
    args.since = since;
  }
  if (until !== undefined) {
    args.until = until;
  }

  return args;
}

function buildDefaultOutputPath(): string {
  const iso = new Date().toISOString();
  const timestamp = iso.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  const relative = path.join('results', 'game-records', `game_records_${timestamp}.jsonl`);
  return path.join(process.cwd(), relative);
}

export function resolveOutputPath(output: string | undefined): string {
  const resolved = output ?? buildDefaultOutputPath();
  return path.isAbsolute(resolved) ? resolved : path.join(process.cwd(), resolved);
}

export function buildFilterFromCliArgs(args: CliArgs): GameRecordFilter {
  const filter: GameRecordFilter = {};
  if (args.boardType) {
    filter.boardType = args.boardType;
  }
  if (args.ratedOnly) {
    filter.isRated = true;
  }
  if (typeof args.limit === 'number') {
    filter.limit = args.limit;
  }
  if (args.since) {
    filter.fromDate = args.since;
  }
  if (args.until) {
    filter.toDate = args.until;
  }
  return filter;
}

export async function runExport(filter: GameRecordFilter, outputPath: string): Promise<number> {
  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  const writeStream = fs.createWriteStream(outputPath, { flags: 'w' });

  let count = 0;

  for await (const line of gameRecordRepository.exportAsJsonl(filter)) {
    writeStream.write(line);
    writeStream.write('\n');
    count += 1;
  }

  await new Promise<void>((resolve, reject) => {
    writeStream.end((err: NodeJS.ErrnoException | null) => {
      if (err) reject(err);
      else resolve();
    });
  });

  return count;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  if (!args) {
    printUsage();
    process.exitCode = 1;
    return;
  }

  const outputPath = resolveOutputPath(args.output);
  const filter = buildFilterFromCliArgs(args);

  // eslint-disable-next-line no-console
  console.log(
    `[export-game-records-jsonl] Exporting game records to ${outputPath}` +
      `${filter.boardType ? ` (boardType=${filter.boardType})` : ''}` +
      `${filter.isRated ? ' [rated-only]' : ''}` +
      `${filter.limit ? ` [limit=${filter.limit}]` : ''}` +
      `${filter.fromDate ? ` [since=${filter.fromDate.toISOString()}]` : ''}` +
      `${filter.toDate ? ` [until=${filter.toDate.toISOString()}]` : ''}...`
  );

  try {
    await connectDatabase();
  } catch (err) {
    console.error('[export-game-records-jsonl] Failed to connect to database:', err);
    process.exitCode = 1;
    return;
  }

  try {
    const count = await runExport(filter, outputPath);
    console.log(`[export-game-records-jsonl] Done. Wrote ${count} record(s) to ${outputPath}.`);
  } finally {
    await disconnectDatabase().catch(() => undefined);
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error('[export-game-records-jsonl] Fatal error:', err);
    process.exitCode = 1;
  });
}
