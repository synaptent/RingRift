/**
 * Maintenance script: delete self-play game databases that contain zero games.
 *
 * This scans the same directories used by SelfPlayGameService:
 *   - data/games
 *   - ai-service/logs/cmaes
 *   - ai-service/data/games
 *
 * For each *.db (or games.db) file, it:
 *   - Opens the SQLite database in read-only mode.
 *   - Executes `SELECT COUNT(*) AS count FROM games`.
 *   - If the query succeeds and count === 0, deletes the database file.
 *
 * Databases without a `games` table, or where the query fails, are left
 * untouched and logged for manual inspection.
 *
 * Usage:
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/cleanup-empty-selfplay-dbs.ts
 */

import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';

interface EmptyDbRecord {
  path: string;
  reason: string;
}

function isCandidateDbFile(fileName: string): boolean {
  return fileName === 'games.db' || fileName.endsWith('.db');
}

function findEmptyGameDbs(rootDir: string): EmptyDbRecord[] {
  const results: EmptyDbRecord[] = [];

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
      // Skip unreadable directories
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

      try {
        const db = new Database(fullPath, { readonly: true });
        try {
          const row = db.prepare('SELECT COUNT(*) AS count FROM games').get() as
            | { count?: number }
            | undefined;

          const count = typeof row?.count === 'number' ? row.count : NaN;

          if (Number.isFinite(count) && count === 0) {
            results.push({
              path: fullPath,
              reason: 'games table exists with 0 rows',
            });
          }
        } finally {
          db.close();
        }
      } catch (err) {
        // If we cannot open the DB or query the games table, log but do not delete.

        console.warn(
          `[cleanup-empty-selfplay-dbs] Skipping candidate DB due to error: ${fullPath}`,
          err instanceof Error ? err.message : String(err)
        );
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

function deleteEmptyDbs(emptyDbs: EmptyDbRecord[]): void {
  for (const record of emptyDbs) {
    const dbPath = record.path;
    // eslint-disable-next-line no-console
    console.log(
      `[cleanup-empty-selfplay-dbs] Deleting empty self-play DB: ${dbPath} (${record.reason})`
    );

    try {
      fs.unlinkSync(dbPath);
    } catch (err) {
      console.error(
        `[cleanup-empty-selfplay-dbs] Failed to delete DB file: ${dbPath}`,
        err instanceof Error ? err.message : String(err)
      );
      continue;
    }

    // Best-effort cleanup of typical SQLite sidecar files.
    for (const suffix of ['-wal', '-shm']) {
      const sidecar = `${dbPath}${suffix}`;
      try {
        if (fs.existsSync(sidecar)) {
          fs.unlinkSync(sidecar);
          // eslint-disable-next-line no-console
          console.log(`[cleanup-empty-selfplay-dbs] Deleted sidecar file: ${sidecar}`);
        }
      } catch (err) {
        console.warn(
          `[cleanup-empty-selfplay-dbs] Failed to delete sidecar file: ${sidecar}`,
          err instanceof Error ? err.message : String(err)
        );
      }
    }
  }
}

function main() {
  const rootDir = process.cwd();
  // eslint-disable-next-line no-console
  console.log(
    `[cleanup-empty-selfplay-dbs] Scanning for empty self-play databases under ${rootDir}`
  );

  const emptyDbs = findEmptyGameDbs(rootDir);

  if (emptyDbs.length === 0) {
    // eslint-disable-next-line no-console
    console.log('[cleanup-empty-selfplay-dbs] No empty self-play databases found.');
    return;
  }

  // eslint-disable-next-line no-console
  console.log(`[cleanup-empty-selfplay-dbs] Found ${emptyDbs.length} empty self-play database(s).`);

  deleteEmptyDbs(emptyDbs);
}

main();
