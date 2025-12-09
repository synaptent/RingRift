#!/usr/bin/env npx ts-node -T
/**
 * FSM Active Mode Validation Script
 *
 * Validates that FSM validation works correctly across multiple game databases
 * before enabling active mode in production.
 *
 * Usage:
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/validate-fsm-active-mode.ts
 *
 * Options:
 *   --db <path>       Validate specific database (can be repeated)
 *   --limit <n>       Max games per database (default: 50)
 *   --mode <mode>     'shadow' or 'active' (default: both)
 *   --verbose         Show per-game results
 *   --fail-fast       Stop on first error
 */

import * as fs from 'fs';
import * as path from 'path';
import Database from 'better-sqlite3';

// Set FSM validation mode for testing
process.env.RINGRIFT_FSM_VALIDATION_MODE = 'shadow';

import { CanonicalReplayEngine } from '../src/shared/replay/CanonicalReplayEngine';

interface ValidationResult {
  db: string;
  gameId: string;
  mode: 'shadow' | 'active';
  success: boolean;
  movesApplied: number;
  totalMoves: number;
  error?: string;
  divergences: number;
}

interface DatabaseStats {
  db: string;
  gamesValidated: number;
  gamesPassed: number;
  gamesFailed: number;
  totalMoves: number;
  totalDivergences: number;
  errors: string[];
}

// Capture console logs to detect divergences
let divergenceCount = 0;
let capturedLogs: string[] = [];
const originalConsoleLog = console.log;

function captureConsoleLogs() {
  divergenceCount = 0;
  capturedLogs = [];
  console.log = (...args: unknown[]) => {
    const msg = args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
    capturedLogs.push(msg);
    if (msg.includes('DIVERGENCE') || msg.includes('[FSM_SHADOW_VALIDATION]')) {
      divergenceCount++;
    }
    // Still log to console in verbose mode
    if (process.argv.includes('--verbose')) {
      originalConsoleLog(...args);
    }
  };
}

function restoreConsoleLogs() {
  console.log = originalConsoleLog;
}

async function validateGame(
  dbPath: string,
  gameId: string,
  mode: 'shadow' | 'active'
): Promise<ValidationResult> {
  // Set mode
  process.env.RINGRIFT_FSM_VALIDATION_MODE = mode;

  captureConsoleLogs();

  try {
    const db = new Database(dbPath, { readonly: true });

    // Get game config
    const gameRow = db.prepare('SELECT * FROM games WHERE game_id = ?').get(gameId) as {
      board_type: string;
      num_players: number;
    };

    if (!gameRow) {
      return {
        db: dbPath,
        gameId,
        mode,
        success: false,
        movesApplied: 0,
        totalMoves: 0,
        error: 'Game not found',
        divergences: 0,
      };
    }

    // Get moves
    const moves = db
      .prepare('SELECT move_json FROM game_moves WHERE game_id = ? ORDER BY move_number')
      .all(gameId) as { move_json: string }[];

    db.close();

    // Create replay engine
    const engine = new CanonicalReplayEngine({
      boardType: gameRow.board_type as 'square8' | 'hexagonal',
      numPlayers: gameRow.num_players,
    });

    // Apply moves
    let appliedCount = 0;
    for (const moveRow of moves) {
      try {
        const move = JSON.parse(moveRow.move_json);
        engine.applyMove(move);
        appliedCount++;
      } catch (err) {
        restoreConsoleLogs();
        return {
          db: dbPath,
          gameId,
          mode,
          success: false,
          movesApplied: appliedCount,
          totalMoves: moves.length,
          error: err instanceof Error ? err.message : String(err),
          divergences: divergenceCount,
        };
      }
    }

    restoreConsoleLogs();

    return {
      db: dbPath,
      gameId,
      mode,
      success: true,
      movesApplied: appliedCount,
      totalMoves: moves.length,
      divergences: divergenceCount,
    };
  } catch (err) {
    restoreConsoleLogs();
    return {
      db: dbPath,
      gameId,
      mode,
      success: false,
      movesApplied: 0,
      totalMoves: 0,
      error: err instanceof Error ? err.message : String(err),
      divergences: divergenceCount,
    };
  }
}

async function validateDatabase(
  dbPath: string,
  limit: number,
  modes: ('shadow' | 'active')[],
  failFast: boolean,
  verbose: boolean
): Promise<DatabaseStats> {
  const stats: DatabaseStats = {
    db: dbPath,
    gamesValidated: 0,
    gamesPassed: 0,
    gamesFailed: 0,
    totalMoves: 0,
    totalDivergences: 0,
    errors: [],
  };

  if (!fs.existsSync(dbPath)) {
    stats.errors.push(`Database not found: ${dbPath}`);
    return stats;
  }

  const db = new Database(dbPath, { readonly: true });

  // Get game IDs
  const games = db.prepare('SELECT game_id FROM games LIMIT ?').all(limit) as { game_id: string }[];
  db.close();

  originalConsoleLog(`\n📁 Validating ${dbPath} (${games.length} games)`);

  for (const { game_id } of games) {
    for (const mode of modes) {
      const result = await validateGame(dbPath, game_id, mode);

      stats.gamesValidated++;
      stats.totalMoves += result.movesApplied;
      stats.totalDivergences += result.divergences;

      if (result.success && result.divergences === 0) {
        stats.gamesPassed++;
        if (verbose) {
          originalConsoleLog(`  ✓ ${game_id} [${mode}]: ${result.movesApplied} moves`);
        }
      } else {
        stats.gamesFailed++;
        const reason = result.error || `${result.divergences} divergences`;
        stats.errors.push(`${game_id} [${mode}]: ${reason}`);
        originalConsoleLog(`  ✗ ${game_id} [${mode}]: ${reason}`);

        if (failFast) {
          return stats;
        }
      }
    }
  }

  return stats;
}

async function main() {
  const args = process.argv.slice(2);

  // Parse arguments
  const databases: string[] = [];
  let limit = 50;
  let modes: ('shadow' | 'active')[] = ['shadow', 'active'];
  const verbose = args.includes('--verbose');
  const failFast = args.includes('--fail-fast');

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--db' && args[i + 1]) {
      databases.push(args[++i]);
    } else if (args[i] === '--limit' && args[i + 1]) {
      limit = parseInt(args[++i], 10);
    } else if (args[i] === '--mode' && args[i + 1]) {
      const m = args[++i];
      if (m === 'shadow' || m === 'active') {
        modes = [m];
      }
    }
  }

  // Default databases if none specified
  if (databases.length === 0) {
    const defaultDbs = [
      'ai-service/data/games/selfplay.db',
      'ai-service/data/games/canonical_square8.db',
      'ai-service/data/games/coverage_selfplay.db',
    ];
    for (const db of defaultDbs) {
      if (fs.existsSync(db)) {
        databases.push(db);
      }
    }
  }

  if (databases.length === 0) {
    originalConsoleLog('No databases found to validate.');
    process.exit(1);
  }

  originalConsoleLog('═══════════════════════════════════════════════════════════');
  originalConsoleLog('  FSM Active Mode Validation');
  originalConsoleLog('═══════════════════════════════════════════════════════════');
  originalConsoleLog(`  Databases: ${databases.length}`);
  originalConsoleLog(`  Limit per DB: ${limit}`);
  originalConsoleLog(`  Modes: ${modes.join(', ')}`);
  originalConsoleLog(`  Fail fast: ${failFast}`);
  originalConsoleLog('═══════════════════════════════════════════════════════════');

  const allStats: DatabaseStats[] = [];

  for (const dbPath of databases) {
    const stats = await validateDatabase(dbPath, limit, modes, failFast, verbose);
    allStats.push(stats);

    if (failFast && stats.gamesFailed > 0) {
      break;
    }
  }

  // Summary
  originalConsoleLog('\n═══════════════════════════════════════════════════════════');
  originalConsoleLog('  VALIDATION SUMMARY');
  originalConsoleLog('═══════════════════════════════════════════════════════════\n');

  let totalPassed = 0;
  let totalFailed = 0;
  let totalMoves = 0;
  let totalDivergences = 0;

  for (const stats of allStats) {
    totalPassed += stats.gamesPassed;
    totalFailed += stats.gamesFailed;
    totalMoves += stats.totalMoves;
    totalDivergences += stats.totalDivergences;

    const status = stats.gamesFailed === 0 ? '✓' : '✗';
    originalConsoleLog(
      `  ${status} ${path.basename(stats.db)}: ${stats.gamesPassed}/${stats.gamesValidated} passed, ${stats.totalMoves} moves`
    );
  }

  originalConsoleLog('\n───────────────────────────────────────────────────────────');
  originalConsoleLog(`  Total: ${totalPassed}/${totalPassed + totalFailed} games passed`);
  originalConsoleLog(`  Moves validated: ${totalMoves}`);
  originalConsoleLog(`  Divergences: ${totalDivergences}`);
  originalConsoleLog('───────────────────────────────────────────────────────────\n');

  if (totalFailed === 0 && totalDivergences === 0) {
    originalConsoleLog('✅ VALIDATION PASSED - Safe to enable active mode\n');
    process.exit(0);
  } else {
    originalConsoleLog('❌ VALIDATION FAILED - Do not enable active mode\n');

    if (totalDivergences > 0) {
      originalConsoleLog(`   ${totalDivergences} divergences detected in shadow mode`);
    }
    if (totalFailed > 0) {
      originalConsoleLog(`   ${totalFailed} games failed validation`);
    }

    originalConsoleLog('\n   Review errors above and fix before proceeding.\n');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Validation failed:', err);
  process.exit(1);
});
