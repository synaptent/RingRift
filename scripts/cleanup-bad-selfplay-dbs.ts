/**
 * Maintenance script: detect and optionally delete self-play databases
 * whose initial_state_json already contains a non-empty moveHistory.
 *
 * These databases typically come from multi-start CMA-ES / GA evaluation
 * runs where the "initial" GameState is a mid/late-game snapshot from a
 * state pool rather than a true start-of-game state. For sandbox replay
 * and TS↔Python parity debugging we generally want databases whose
 * initial_state_json.moveHistory is empty.
 *
 * This scans the same directories as SelfPlayGameService:
 *   - data/games
 *   - ai-service/logs/cmaes
 *   - ai-service/data/games
 *
 * For each *.db (or games.db) file, it:
 *   - Samples up to SAMPLE_GAMES_PER_DB games from the games table.
 *   - For each sampled game, loads game_initial_state.initial_state_json,
 *     decompresses it when compressed=1, and parses the JSON.
 *   - If any sampled game's initial_state_json.moveHistory has length > 0,
 *     the DB is marked as "bad" (likely mid-snapshot recordings).
 *
 * When invoked with --delete, it deletes all such DBs plus typical SQLite
 * sidecar files (-wal/-shm). Without --delete it only prints a report.
 *
 * Usage:
 *   # Dry-run: list suspect DBs
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/cleanup-bad-selfplay-dbs.ts
 *
 *   # Delete suspect DBs
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/cleanup-bad-selfplay-dbs.ts --delete
 */

import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';
import { gunzipSync } from 'zlib';

interface BadDbRecord {
  path: string;
  reason: string;
  sampleGameId?: string;
}

const SAMPLE_GAMES_PER_DB = 25;

function isCandidateDbFile(fileName: string): boolean {
  return fileName === 'games.db' || fileName.endsWith('.db');
}

function hasMidgameInitialState(dbPath: string): BadDbRecord | null {
  let db: Database.Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });

    const gameIds = db
      .prepare('SELECT game_id FROM games ORDER BY created_at DESC LIMIT ?')
      .all(SAMPLE_GAMES_PER_DB) as Array<{ game_id: string }>;

    for (const row of gameIds) {
      const gameId = row.game_id;
      const stateRow = db
        .prepare(
          `
          SELECT initial_state_json, compressed
          FROM game_initial_state
          WHERE game_id = ?
        `
        )
        .get(gameId) as { initial_state_json: string | Buffer; compressed: number } | undefined;

      if (!stateRow) continue;

      let jsonPayload = stateRow.initial_state_json;
      if (stateRow.compressed && Buffer.isBuffer(jsonPayload)) {
        jsonPayload = gunzipSync(jsonPayload).toString('utf-8');
      }
      const state =
        typeof jsonPayload === 'string'
          ? (JSON.parse(jsonPayload) as any)
          : (JSON.parse(jsonPayload.toString()) as any);

      const moveHistory = Array.isArray(state?.moveHistory) ? state.moveHistory : [];
      const stacks = state?.board && typeof state.board === 'object' ? state.board.stacks : null;
      const stackKeys =
        stacks && typeof stacks === 'object' ? Object.keys(stacks as Record<string, unknown>) : [];

      // Heuristic: treat a DB as suspect if the recorded "initial" state
      // already contains either:
      //   - a non-empty moveHistory array, or
      //   - any stacks on the board (mid-game snapshot rather than empty).
      if (moveHistory.length > 0 || stackKeys.length > 0) {
        const reasonParts: string[] = [];
        if (moveHistory.length > 0) {
          reasonParts.push(`moveHistory length ${moveHistory.length}`);
        }
        if (stackKeys.length > 0) {
          reasonParts.push(`board.stacks has ${stackKeys.length} entries`);
        }

        return {
          path: dbPath,
          reason: `initial_state_json indicates mid-game snapshot for game ${gameId} (${reasonParts.join(
            ', '
          )})`,
          sampleGameId: gameId,
        };
      }
    }
  } catch (err) {
    console.warn(
      `[cleanup-bad-selfplay-dbs] Skipping DB due to error: ${dbPath}`,
      err instanceof Error ? err.message : String(err)
    );
    return null;
  } finally {
    if (db) {
      try {
        db.close();
      } catch {
        // ignore
      }
    }
  }

  return null;
}

function findBadSelfPlayDbs(rootDir: string): BadDbRecord[] {
  const results: BadDbRecord[] = [];

  const searchPaths = [
    path.join(rootDir, 'data', 'games'),
    path.join(rootDir, 'ai-service', 'logs', 'cmaes'),
    path.join(rootDir, 'ai-service', 'data', 'games'),
  ];

  const visited = new Set<string>();

  const walk = (dir: string, depth: number) => {
    if (depth <= 0) return;
    const realDir = path.resolve(dir);
    if (visited.has(realDir)) return;
    visited.add(realDir);

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(realDir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(realDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath, depth - 1);
        continue;
      }

      if (!entry.isFile() || !isCandidateDbFile(entry.name)) {
        continue;
      }

      const badRecord = hasMidgameInitialState(fullPath);
      if (badRecord) {
        results.push(badRecord);
      }
    }
  };

  for (const searchPath of searchPaths) {
    if (fs.existsSync(searchPath)) {
      walk(searchPath, 7);
    }
  }

  return results;
}

function deleteBadDbs(records: BadDbRecord[]): void {
  for (const record of records) {
    const dbPath = record.path;
    // eslint-disable-next-line no-console
    console.log(
      `[cleanup-bad-selfplay-dbs] Deleting suspect self-play DB: ${dbPath} (${record.reason})`
    );

    try {
      fs.unlinkSync(dbPath);
    } catch (err) {
      console.error(
        `[cleanup-bad-selfplay-dbs] Failed to delete DB file: ${dbPath}`,
        err instanceof Error ? err.message : String(err)
      );
      continue;
    }

    for (const suffix of ['-wal', '-shm']) {
      const sidecar = `${dbPath}${suffix}`;
      try {
        if (fs.existsSync(sidecar)) {
          fs.unlinkSync(sidecar);
          // eslint-disable-next-line no-console
          console.log(`[cleanup-bad-selfplay-dbs] Deleted sidecar file: ${sidecar}`);
        }
      } catch (err) {
        console.warn(
          `[cleanup-bad-selfplay-dbs] Failed to delete sidecar file: ${sidecar}`,
          err instanceof Error ? err.message : String(err)
        );
      }
    }
  }
}

function main(): void {
  const rootDir = process.cwd();
  const args = process.argv.slice(2);
  const shouldDelete = args.includes('--delete');

  // eslint-disable-next-line no-console
  console.log(
    `[cleanup-bad-selfplay-dbs] Scanning for self-play DBs with non-empty initial moveHistory under ${rootDir}`
  );

  const badDbs = findBadSelfPlayDbs(rootDir);

  if (badDbs.length === 0) {
    // eslint-disable-next-line no-console
    console.log('[cleanup-bad-selfplay-dbs] No suspect self-play databases found.');
    return;
  }

  // eslint-disable-next-line no-console
  console.log(`[cleanup-bad-selfplay-dbs] Found ${badDbs.length} suspect self-play database(s):`);
  for (const rec of badDbs) {
    // eslint-disable-next-line no-console
    console.log(
      `  - ${rec.path} (${rec.reason}${
        rec.sampleGameId ? `, sample game_id=${rec.sampleGameId}` : ''
      })`
    );
  }

  if (!shouldDelete) {
    // eslint-disable-next-line no-console
    console.log(
      '[cleanup-bad-selfplay-dbs] Dry run only. Re-run with --delete to remove these databases.'
    );
    return;
  }

  deleteBadDbs(badDbs);
}

main();
