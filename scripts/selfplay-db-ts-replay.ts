#!/usr/bin/env ts-node
/**
 * Diagnostic & admin script for self-play SQLite GameReplayDBs.
 *
 * Modes:
 * - Replay (default): replay a recorded self-play game into the
 *   ClientSandboxEngine and log per-move TS state summaries.
 * - Import: batch-import completed self-play games as canonical GameRecords
 *   into Postgres for training/export pipelines.
 *
 * Usage (from repo root):
 *
 *   # Replay a single game (existing behaviour)
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts \\
 *     --db /absolute/or/relative/path/to/games.db \\
 *     --game 7f031908-655b-49af-ad05-f330e9d07488
 *
 *   # Import completed games into canonical GameRecord rows
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts \\
 *     --mode import --db /path/to/games.db --limit 100 --tag exp:P4.3-1
 *
 * The replay mode mirrors the /sandbox self-play replay path (SelfPlayBrowser +
 * SandboxGameHost) but runs headless under Node so you can compare the TS
 * engine’s state sequence against Python’s GameReplayDB.get_state_at_move.
 *
 * The import mode is an offline/admin-only tool for populating canonical
 * GameRecord rows from self-play episodes.
 */

import * as path from 'path';
import * as fs from 'fs';

import {
  getSelfPlayGameService,
  importSelfPlayGameAsGameRecord,
} from '../src/server/services/SelfPlayGameService';
import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../src/client/sandbox/ClientSandboxEngine';
import type { BoardType, GameState, Move, Position } from '../src/shared/types/game';
import { hashGameStateSHA256, getEffectiveLineLengthThreshold } from '../src/shared/engine';
import { serializeGameState } from '../src/shared/engine/contracts/serialization';
import { connectDatabase, disconnectDatabase } from '../src/server/database/connection';
import type { PrismaClient } from '@prisma/client';

type Mode = 'replay' | 'import';

interface BaseCliArgs {
  mode: Mode;
  dbPath: string;
}

interface ReplayCliArgs extends BaseCliArgs {
  mode: 'replay';
  gameId: string;
  /** Move number(s) at which to dump full TS state JSON for debugging */
  dumpStateAt?: number[];
}

interface ImportCliArgs extends BaseCliArgs {
  mode: 'import';
  limit?: number;
  tags: string[];
  dryRun: boolean;
}

export type CliArgs = ReplayCliArgs | ImportCliArgs;

function printUsage(): void {
  console.error(
    [
      'Usage:',
      '  # Replay a single self-play game into the TS sandbox engine',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts \\',
      '    --db /path/to/games.db --game <gameId> [--dump-state-at <k1,k2,...>]',
      '',
      '  # Dump full TS state JSON at specific move numbers for parity debugging',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts \\',
      '    --db /path/to/games.db --game <gameId> --dump-state-at 50,51,52',
      '',
      '  # Import completed self-play games as canonical GameRecords',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/selfplay-db-ts-replay.ts \\',
      '    --mode import --db /path/to/games.db [--limit <n>] [--tag <tag>] [--dry-run]',
    ].join('\n')
  );
}

/**
 * Parse CLI arguments.
 *
 * For replay mode (default):
 *   --db /path/to/games.db --game <gameId>
 *
 * For import mode:
 *   --mode import --db /path/to/games.db [--limit N] [--tag TAG] [--dry-run]
 */
export function parseArgs(argv: string[]): CliArgs | null {
  let mode: Mode = 'replay';
  let dbPath = '';
  let gameId = '';
  let limit: number | undefined;
  const tags: string[] = [];
  let dryRun = false;
  const dumpStateAt: number[] = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    if ((arg === '--db' || arg === '--dbPath') && argv[i + 1]) {
      dbPath = argv[i + 1];
      i += 1;
    } else if ((arg === '--game' || arg === '--gameId') && argv[i + 1]) {
      gameId = argv[i + 1];
      i += 1;
    } else if (arg === '--mode' && argv[i + 1]) {
      const value = argv[i + 1];
      if (value === 'replay' || value === 'import') {
        mode = value;
      } else {
        console.error(`Unknown --mode value: ${value}`);
        return null;
      }
      i += 1;
    } else if (arg === '--limit' && argv[i + 1]) {
      const raw = argv[i + 1];
      const parsed = Number.parseInt(raw, 10);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        console.error(`Invalid --limit value: ${raw}`);
        return null;
      }
      limit = parsed;
      i += 1;
    } else if (arg === '--tag' && argv[i + 1]) {
      tags.push(argv[i + 1]);
      i += 1;
    } else if (arg === '--dry-run') {
      dryRun = true;
    } else if ((arg === '--dump-state-at' || arg === '--dump-at') && argv[i + 1]) {
      const raw = argv[i + 1];
      // Support comma-separated list: --dump-state-at 10,20,30
      const parts = raw.split(',').map((s) => s.trim());
      for (const part of parts) {
        const parsed = Number.parseInt(part, 10);
        if (Number.isFinite(parsed) && parsed >= 0) {
          dumpStateAt.push(parsed);
        } else {
          console.error(`Invalid --dump-state-at value: ${part}`);
          return null;
        }
      }
      i += 1;
    }
  }

  if (!dbPath) {
    console.error('Missing required --db path');
    return null;
  }

  const resolvedDb = path.isAbsolute(dbPath) ? dbPath : path.resolve(process.cwd(), dbPath);

  if (mode === 'replay') {
    if (!gameId) {
      console.error('Missing required --game for replay mode');
      return null;
    }

    return {
      mode: 'replay',
      dbPath: resolvedDb,
      gameId,
      ...(dumpStateAt.length > 0 && { dumpStateAt }),
    };
  }

  // Import mode
  return {
    mode: 'import' as const,
    dbPath: resolvedDb,
    ...(limit !== undefined && { limit }),
    tags,
    dryRun,
  };
}

/**
 * Normalize a recorded move from the self-play database into the canonical
 * Move surface expected by the sandbox engine.
 *
 * This mirrors SelfPlayBrowser.normalizeRecordedMove but is defined locally
 * to avoid a React dependency in this Node script.
 */
function normalizeRecordedMove(rawMove: Move, fallbackMoveNumber: number): Move {
  const anyMove = rawMove as any;

  const type: Move['type'] =
    anyMove.type === 'forced_elimination' ? 'eliminate_rings_from_stack' : anyMove.type;

  const timestampRaw = anyMove.timestamp;
  const timestamp: Date =
    timestampRaw instanceof Date
      ? timestampRaw
      : typeof timestampRaw === 'string'
        ? new Date(timestampRaw)
        : new Date();

  const from: Position | undefined =
    anyMove.from && typeof anyMove.from === 'object' ? anyMove.from : undefined;

  const moveNumber =
    typeof anyMove.moveNumber === 'number' && Number.isFinite(anyMove.moveNumber)
      ? anyMove.moveNumber
      : fallbackMoveNumber;

  const thinkTime =
    typeof anyMove.thinkTime === 'number'
      ? anyMove.thinkTime
      : typeof anyMove.thinkTimeMs === 'number'
        ? anyMove.thinkTimeMs
        : 0;

  return {
    ...anyMove,
    type,
    from,
    timestamp,
    thinkTime,
    moveNumber,
  } as Move;
}

/**
 * Replay-only compatibility shim for legacy line-processing moves.
 *
 * Older self-play databases sometimes record overlength line rewards as a
 * single `process_line` Move with `formedLines[0].length > L` (where L is the
 * effective line length). Modern semantics express overlength rewards via
 * `choose_line_reward` moves instead, and the shared
 * `applyProcessLineDecision()` helper treats such overlength `process_line`
 * moves as a no-op.
 *
 * For replay/parity purposes we want to preserve the original behaviour
 * encoded in the DB: collapsing the full line as if Option‑1 had been chosen.
 * To do this without changing shared engine semantics, we rewrite those
 * legacy `process_line` moves into `choose_line_reward` moves with the same
 * `formedLines` payload and no `collapsedMarkers` (collapse-all sentinel)
 * before feeding them into the sandbox.
 */
function rewriteLegacyProcessLineMoves(
  moves: Move[],
  boardType: BoardType,
  numPlayers: number
): Move[] {
  const requiredLength = getEffectiveLineLengthThreshold(boardType, numPlayers, undefined);

  return moves.map((move) => {
    if (move.type !== 'process_line') {
      return move;
    }

    const anyMove = move as any;
    const formed = anyMove.formedLines as { positions?: Position[]; length?: number }[] | undefined;
    const line0 = formed && formed[0];
    if (!line0) {
      return move;
    }

    const length =
      typeof line0.length === 'number'
        ? line0.length
        : Array.isArray(line0.positions)
          ? line0.positions.length
          : 0;

    if (length <= requiredLength) {
      // Exact-length or shorter: leave as process_line and let the shared
      // helpers handle it according to modern semantics.
      return move;
    }

    // Overlength legacy process_line: rewrite as a collapse-all
    // choose_line_reward for replay purposes.
    const rewritten: Move = {
      ...(move as any),
      type: 'choose_line_reward',
      // Keep formedLines; omit / clear collapsedMarkers so that the shared
      // helper interprets this as a collapse-all variant.
      collapsedMarkers: undefined,
    };
    return rewritten;
  });
}

function summarizeState(label: string, state: GameState): Record<string, unknown> {
  return {
    label,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
    moveHistoryLength: state.moveHistory.length,
    // Use the compact hash that matches Python's _compute_state_hash for
    // cross-engine parity checks and GameReplayDB history entries.
    stateHash: hashGameStateSHA256(state),
  };
}

async function runReplayMode(args: ReplayCliArgs): Promise<void> {
  const { dbPath, gameId, dumpStateAt } = args;

  const service = getSelfPlayGameService();
  const detail = service.getGame(dbPath, gameId);

  if (!detail) {
    console.error(`Game ${gameId} not found in DB ${dbPath}`);
    process.exitCode = 1;
    return;
  }

  // Sanitize initial state in the same way SelfPlayBrowser does before
  // passing it into /sandbox, so this script matches sandbox behaviour.
  const rawState = detail.initialState as any;
  const sanitizedState = rawState && typeof rawState === 'object' ? { ...rawState } : rawState;
  if (sanitizedState && Array.isArray(sanitizedState.moveHistory)) {
    sanitizedState.moveHistory = [];
  }
  if (sanitizedState && Array.isArray(sanitizedState.history)) {
    sanitizedState.history = [];
  }

  const config: SandboxConfig = {
    boardType: detail.boardType as BoardType,
    numPlayers: detail.numPlayers,
    playerKinds: Array.from({ length: detail.numPlayers }, () => 'human'),
  };

  const interactionHandler: SandboxInteractionHandler = {
    async requestChoice(choice: any) {
      const options = (choice?.options as any[]) ?? [];
      const selectedOption = options.length > 0 ? options[0] : undefined;
      return {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        choiceType: choice.type,
        selectedOption,
      } as any;
    },
  };

  // Optional debug configuration: dump TS GameState JSON at specified move numbers.
  // Supports both CLI args (--dump-state-at 10,20,30) and env var (legacy).
  // Dumps are written to RINGRIFT_TS_REPLAY_DUMP_DIR or ./ts-replay-dumps.
  const dumpKRaw = process.env.RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K;
  const envDumpK = dumpKRaw ? Number.parseInt(dumpKRaw, 10) : NaN;
  const dumpKSet = new Set<number>(dumpStateAt ?? []);
  if (Number.isFinite(envDumpK)) {
    dumpKSet.add(envDumpK);
  }
  const shouldDumpState = dumpKSet.size > 0;
  const dumpDirEnv = process.env.RINGRIFT_TS_REPLAY_DUMP_DIR;
  const dumpDir =
    dumpDirEnv && dumpDirEnv.length > 0 ? dumpDirEnv : path.join(process.cwd(), 'ts-replay-dumps');

  const engine = new ClientSandboxEngine({
    config,
    interactionHandler,
    // Enable traceMode so the sandbox replay path uses strict, parity-oriented
    // semantics (no extra auto line/territory processing beyond recorded moves).
    traceMode: true,
  });

  engine.initFromSerializedState(sanitizedState, config.playerKinds, interactionHandler);

  const normalizedMoves: Move[] = detail.moves.map((m) =>
    normalizeRecordedMove(m.move as Move, m.moveNumber)
  );

  // Replay-only compatibility pass for legacy overlength process_line moves.
  const recordedMoves: Move[] = rewriteLegacyProcessLineMoves(
    normalizedMoves,
    detail.boardType as BoardType,
    detail.numPlayers
  );

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify({
      kind: 'ts-replay-initial',
      dbPath,
      gameId,
      totalRecordedMoves: recordedMoves.length,
      summary: summarizeState('initial', engine.getGameState()),
    })
  );

  let applied = 0;
  for (let i = 0; i < recordedMoves.length; i++) {
    const move = recordedMoves[i];
    const nextMove = i + 1 < recordedMoves.length ? recordedMoves[i + 1] : null;
    applied += 1;

    await engine.applyCanonicalMoveForReplay(move, nextMove);
    const state = engine.getGameState();

    // Optional debug dump for parity investigation
    if (shouldDumpState && dumpKSet.has(applied)) {
      try {
        fs.mkdirSync(dumpDir, { recursive: true });
        const fileName = `${path.basename(dbPath)}__${gameId}__k${applied}.ts_state.json`;
        const outPath = path.join(dumpDir, fileName);
        // Use serializeGameState to properly convert Map objects to plain objects
        fs.writeFileSync(outPath, JSON.stringify(serializeGameState(state), null, 2), 'utf-8');

        console.error(
          `[selfplay-db-ts-replay] Dumped TS state for ${gameId} @ k=${applied} to ${outPath}`
        );
      } catch (err) {
        console.error(
          `[selfplay-db-ts-replay] Failed to dump TS state for ${gameId} @ k=${applied}:`,
          err
        );
      }
    }
    // eslint-disable-next-line no-console
    console.log(
      JSON.stringify({
        kind: 'ts-replay-step',
        k: applied,
        moveType: move.type,
        movePlayer: move.player,
        moveNumber: move.moveNumber,
        summary: summarizeState(`after_move_${applied}`, state),
      })
    );
  }

  const finalState = engine.getGameState();
  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify({
      kind: 'ts-replay-final',
      appliedMoves: applied,
      summary: summarizeState('final', finalState),
    })
  );
}

async function hasExistingSelfPlayImport(
  prisma: PrismaClient,
  selfPlayGameId: string
): Promise<boolean> {
  try {
    // Use JSON-path filters on recordMetadata to detect existing imports with
    // matching (source='self_play', sourceId=selfPlayGameId). This keeps the
    // check local to this admin script without changing repository helpers.
    const existing = await prisma.game.findFirst({
      where: {
        AND: [
          {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            recordMetadata: { path: ['source'], equals: 'self_play' } as any,
          },
          {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            recordMetadata: { path: ['sourceId'], equals: selfPlayGameId } as any,
          },
        ],
      } as any,
    });
    return Boolean(existing);
  } catch {
    // On any error, fall back to treating the game as not-yet-imported so the
    // caller can decide how to handle failures.
    return false;
  }
}

async function runImportMode(args: ImportCliArgs): Promise<void> {
  const { dbPath, limit, tags, dryRun } = args;

  const service = getSelfPlayGameService();
  const allGames = service.listGames(dbPath);
  const completed = allGames.filter((g) => g.completedAt !== null);

  const sortedCompleted = completed
    .slice()
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime());

  const candidates =
    typeof limit === 'number' && limit >= 0 ? sortedCompleted.slice(0, limit) : sortedCompleted;

  // eslint-disable-next-line no-console
  console.log(
    `[selfplay-import] Found ${completed.length} completed game(s) in ${dbPath}; ` +
      `processing ${candidates.length} candidate(s)` +
      `${typeof limit === 'number' ? ` (limit=${limit})` : ''}` +
      `${dryRun ? ' [dry-run]' : ''}` +
      `${tags.length ? ` [tags=${tags.join(',')}]` : ''}.`
  );

  if (dryRun) {
    for (const g of candidates) {
      // eslint-disable-next-line no-console
      console.log(`[selfplay-import] Would import self-play game ${g.gameId} from ${dbPath}`);
    }
    // eslint-disable-next-line no-console
    console.log(
      `[selfplay-import] Dry-run complete: ${candidates.length} completed game(s) would be imported from ${dbPath}.`
    );
    return;
  }

  let prisma: PrismaClient;
  try {
    prisma = await connectDatabase();
  } catch (err) {
    console.error('[selfplay-import] Failed to connect to database:', err);
    process.exitCode = 1;
    return;
  }

  let importedCount = 0;
  let skippedExisting = 0;
  let failedCount = 0;

  for (const g of candidates) {
    const alreadyImported = await hasExistingSelfPlayImport(prisma, g.gameId);
    if (alreadyImported) {
      skippedExisting += 1;
      // eslint-disable-next-line no-console
      console.log(
        `[selfplay-import] Skipping self-play game ${g.gameId} from ${dbPath}: already imported`
      );
      continue;
    }

    try {
      await importSelfPlayGameAsGameRecord({
        dbPath,
        gameId: g.gameId,
        source: 'self_play',
        tags,
      });
      importedCount += 1;
      // eslint-disable-next-line no-console
      console.log(`[selfplay-import] Imported self-play game ${g.gameId} from ${dbPath}`);
    } catch (err) {
      failedCount += 1;

      console.error(
        `[selfplay-import] Failed to import self-play game ${g.gameId} from ${dbPath}:`,
        err
      );
    }
  }

  // eslint-disable-next-line no-console
  console.log(
    `[selfplay-import] Import summary for ${dbPath}: ` +
      `completed=${completed.length}, ` +
      `attempted=${candidates.length}, ` +
      `imported=${importedCount}, ` +
      `skipped_existing=${skippedExisting}, ` +
      `failed=${failedCount}.`
  );

  if (failedCount > 0) {
    process.exitCode = 1;
  }

  await disconnectDatabase().catch(() => undefined);
}

export async function main(argv: string[] = process.argv.slice(2)): Promise<void> {
  const args = parseArgs(argv);
  if (!args) {
    printUsage();
    process.exitCode = 1;
    return;
  }

  if (args.mode === 'import') {
    await runImportMode(args);
  } else {
    await runReplayMode(args);
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error('[selfplay-db-ts-replay] Fatal error:', err);
    process.exitCode = 1;
  });
}
