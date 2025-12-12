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
import type { SelfPlayMove } from '../src/server/services/SelfPlayGameService';
import type { BoardType, GameState, GamePhase, Move, Position } from '../src/shared/types/game';
import {
  hashGameStateSHA256,
  getEffectiveLineLengthThreshold,
  evaluateVictory,
  hasAnyRealAction,
  playerHasMaterial,
} from '../src/shared/engine';
import {
  createLpsTrackingState,
  updateLpsTracking,
  evaluateLpsVictory,
  isLpsActivePhase,
  type LpsTrackingState,
} from '../src/shared/engine/lpsTracking';
import { serializeGameState } from '../src/shared/engine/contracts/serialization';
import { validateMoveWithFSM, computeFSMOrchestration } from '../src/shared/engine/fsm/FSMAdapter';
import { getValidMoves } from '../src/shared/engine/orchestration/turnOrchestrator';
import { isANMState } from '../src/shared/engine/globalActions';
import {
  hasAnyPlacementForPlayer,
  hasAnyMovementForPlayer,
  hasAnyCaptureForPlayer,
} from '../src/shared/engine/turnDelegateHelpers';
import type { PerTurnState } from '../src/shared/engine/turnLogic';
import { connectDatabase, disconnectDatabase } from '../src/server/database/connection';
import type { PrismaClient } from '@prisma/client';
import { CanonicalReplayEngine } from '../src/shared/replay/CanonicalReplayEngine';

// Known-bad recordings that should be skipped until regenerated.
const DEFAULT_SKIP_GAME_IDS = new Set<string>([
  // canonical_square8.db: territory -> no_placement_action off-phase, missing initial state (non-canonical).
  '151ba34a-b7bf-4845-a779-5232221f592e',
  // selfplay.db: chain_capture interrupted by no_line_action bookkeeping (legacy recording bug).
  '6b8b1145-7078-476b-a72f-75a35faecb5e',
]);

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
  /** Game IDs to skip (non-canonical/known-bad recordings) */
  skipGameIds?: Set<string>;
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
  const skipGameIds: Set<string> = new Set();

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
    } else if (arg === '--skip-game' && argv[i + 1]) {
      skipGameIds.add(argv[i + 1]);
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
      ...(skipGameIds.size > 0 && { skipGameIds }),
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
 * Build a canonical Move from a SelfPlayMove record stored in a GameReplayDB.
 *
 * The GameReplayDB stores:
 * - Canonical (phase, move_type) columns that have been validated and,
 *   for canonical DBs, canonicalised by the Python writer.
 * - A legacy/raw move_json payload whose `type` field may be stale.
 *
 * For parity purposes, the canonical move_type field is the single source of
 * truth for Move.type. The legacy move_json payload is treated as a best-effort
 * source of geometry and diagnostics only.
 */
export function buildCanonicalMoveFromSelfPlayRecord(
  record: SelfPlayMove,
  fallbackMoveNumber: number
): Move {
  const raw = (record.move ?? {}) as any;

  // Derive type from the canonical DB column, never from raw.type.
  const canonicalType = record.moveType as Move['type'];

  const timestampRaw = raw.timestamp;
  const timestamp: Date =
    timestampRaw instanceof Date
      ? timestampRaw
      : typeof timestampRaw === 'string'
        ? new Date(timestampRaw)
        : new Date(0);

  const moveNumber =
    typeof raw.moveNumber === 'number' && Number.isFinite(raw.moveNumber) && raw.moveNumber > 0
      ? raw.moveNumber
      : Number.isFinite(record.moveNumber) && record.moveNumber > 0
        ? record.moveNumber
        : fallbackMoveNumber;

  const thinkTime =
    typeof raw.thinkTime === 'number'
      ? raw.thinkTime
      : typeof raw.thinkTimeMs === 'number'
        ? raw.thinkTimeMs
        : typeof record.thinkTimeMs === 'number'
          ? record.thinkTimeMs
          : 0;

  const id =
    typeof raw.id === 'string' && raw.id.length > 0
      ? raw.id
      : `db-${record.moveNumber}-${canonicalType}`;

  // Start from the raw payload so that geometry/effect fields for interactive
  // moves are preserved, but always override the discriminant and metadata
  // fields with canonical values from the DB row.
  const canonical: Move = {
    ...(raw as Partial<Move>),
    id,
    type: canonicalType,
    player: record.player,
    timestamp,
    thinkTime,
    moveNumber,
  } as Move;

  // Ensure `to` is always present – required by the Move contract. For moves
  // without meaningful geometry we will overwrite this with a benign sentinel.
  if (!isValidPosition((canonical as any).to)) {
    (canonical as any).to = { x: 0, y: 0 };
  }

  // Sanitize bookkeeping / forced no-op moves so they don't carry stale
  // geometry or effect payloads from legacy move_json blobs.
  if (
    canonicalType === 'no_placement_action' ||
    canonicalType === 'no_movement_action' ||
    canonicalType === 'no_line_action' ||
    canonicalType === 'no_territory_action'
  ) {
    delete (canonical as any).from;
    delete (canonical as any).captureTarget;
    delete (canonical as any).formedLines;
    delete (canonical as any).capturedStacks;
    delete (canonical as any).captureChain;
    delete (canonical as any).overtakenRings;
    delete (canonical as any).claimedTerritory;
    delete (canonical as any).disconnectedRegions;
    delete (canonical as any).eliminatedRings;
    (canonical as any).to = { x: 0, y: 0 };
  } else if (
    canonicalType === 'skip_placement' ||
    canonicalType === 'skip_capture' ||
    canonicalType === 'skip_territory_processing'
  ) {
    // Voluntary skips: there is no meaningful geometry; ensure we don't
    // accidentally interpret stale coordinates as real board actions.
    delete (canonical as any).from;
    delete (canonical as any).captureTarget;
    if (!isValidPosition((canonical as any).to)) {
      (canonical as any).to = { x: 0, y: 0 };
    }
  } else {
    // Interactive moves: normalise from/to so they are either valid Positions
    // or undefined. This keeps downstream consumers resilient to incomplete
    // legacy payloads without changing semantics for canonical recordings.
    if (!isValidPosition((canonical as any).from)) {
      (canonical as any).from = undefined;
    }
    if (!isValidPosition((canonical as any).to)) {
      (canonical as any).to = { x: 0, y: 0 };
    }
  }

  return canonical;
}

/**
 * Runtime guard for Position-like objects.
 */
function isValidPosition(value: unknown): value is Position {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const pos = value as { x?: unknown; y?: unknown };
  return typeof pos.x === 'number' && typeof pos.y === 'number';
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

/**
 * Phase order for determining which phases are skipped during replay.
 * Turn ends after territory_processing (or forced_elimination), then next player
 * starts at ring_placement.
 */
const PHASE_ORDER: GamePhase[] = [
  'ring_placement',
  'movement',
  'capture',
  'chain_capture',
  'line_processing',
  'territory_processing',
  'forced_elimination',
];

/**
 * Get the bookkeeping move type for a phase that has no recorded action.
 *
 * For movement, this consults the same phase-local movement surface used by
 * the shared engine and Python GameEngine when deciding whether
 * NO_MOVEMENT_ACTION is required: getValidMoves in a synthetic MOVEMENT
 * state for the target player. This keeps the replay bridge aligned with
 * canonical ANM semantics and the FSM guard on NO_MOVEMENT_ACTION.
 */
function getBookkeepingMoveType(
  phase: GamePhase,
  state: GameState,
  player: number
): Move['type'] | null {
  switch (phase) {
    case 'ring_placement':
      return 'no_placement_action';
    case 'movement': {
      // Only synthesize a forced no_movement_action bridge when there are no
      // legal movement/capture/recovery moves for this player in MOVEMENT.
      // If any such moves exist, the movement phase must be played out via
      // explicit DB moves and we must not auto-skip it via bookkeeping.
      const syntheticState: GameState = {
        ...state,
        currentPhase: 'movement',
        currentPlayer: player,
      };
      const validMoves = getValidMoves(syntheticState);
      const movementLike = validMoves.filter(
        (m) =>
          m.player === player &&
          (m.type === 'move_stack' ||
            m.type === 'move_ring' ||
            m.type === 'overtaking_capture' ||
            m.type === 'continue_capture_segment' ||
            m.type === 'recovery_slide')
      );
      if (movementLike.length > 0) {
        return null;
      }
      return 'no_movement_action';
    }
    case 'line_processing':
      return 'no_line_action';
    case 'territory_processing':
      return 'no_territory_action';
    default:
      return null;
  }
}

/**
 * Check if a move type is valid for a given phase.
 */
function isMoveValidInPhase(moveType: Move['type'], phase: GamePhase): boolean {
  const validMoves: Record<GamePhase, Move['type'][]> = {
    ring_placement: ['place_ring', 'skip_placement', 'no_placement_action'],
    movement: [
      'move_stack',
      'move_ring',
      'overtaking_capture',
      'continue_capture_segment',
      'no_movement_action',
      'recovery_slide',
    ],
    capture: ['overtaking_capture', 'continue_capture_segment', 'skip_capture'],
    chain_capture: ['overtaking_capture', 'continue_capture_segment'],
    line_processing: ['process_line', 'choose_line_reward', 'no_line_action'],
    territory_processing: [
      'process_territory_region',
      'eliminate_rings_from_stack',
      'skip_territory_processing',
      'no_territory_action',
    ],
    forced_elimination: ['forced_elimination'],
    game_over: [],
  };
  return validMoves[phase]?.includes(moveType) ?? false;
}

/**
 * Synthesize bookkeeping moves to bridge between current state and next recorded move.
 *
 * This is needed for legacy recordings that don't include explicit bookkeeping
 * moves (no_line_action, no_territory_action) for phases that had no action.
 * The modern engine requires explicit bookkeeping for phase invariant enforcement.
 *
 * @param currentPhase The engine's current phase
 * @param currentPlayer The engine's current player
 * @param nextMove The next recorded move to apply
 * @param moveNumberBase Base move number for synthesized moves
 * @returns Array of bookkeeping moves to inject before nextMove
 */
function synthesizeBookkeepingMoves(
  currentState: GameState,
  nextMove: Move,
  moveNumberBase: number
): Move[] {
  const synthesized: Move[] = [];

  const currentPhase = currentState.currentPhase;
  const currentPlayer = currentState.currentPlayer;

  // Don't bridge if game is over
  if (currentPhase === 'game_over') {
    return synthesized;
  }

  // If the next move is already valid in the current phase, no bridging needed
  if (isMoveValidInPhase(nextMove.type, currentPhase)) {
    return synthesized;
  }

  // If the next move is for a different player, we need to complete the current
  // player's turn by injecting bookkeeping for remaining phases, AND also
  // synthesize empty turns for any intermediate players without turn-material.
  if (nextMove.player !== currentPlayer) {
    // Find current phase index
    const currentIdx = PHASE_ORDER.indexOf(currentPhase);
    if (currentIdx === -1) return synthesized;

    // Inject bookkeeping for phases after current until turn ends
    for (let i = currentIdx; i < PHASE_ORDER.length; i++) {
      const phase = PHASE_ORDER[i];
      const bookkeepingType = getBookkeepingMoveType(phase, currentState, currentPlayer);
      if (bookkeepingType) {
        synthesized.push({
          id: `synthesized-${bookkeepingType}-${moveNumberBase + synthesized.length}`,
          type: bookkeepingType,
          player: currentPlayer,
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: moveNumberBase + synthesized.length,
        });
      }
    }
  } else {
    // Same player - check if we need to bridge phases
    // Find the expected phase for the next move type
    const nextMovePhase = getMovePhase(nextMove.type);
    if (!nextMovePhase) return synthesized;

    const currentIdx = PHASE_ORDER.indexOf(currentPhase);
    const nextIdx = PHASE_ORDER.indexOf(nextMovePhase);

    // If next move is for a later phase, inject bookkeeping for skipped phases
    if (nextIdx > currentIdx) {
      for (let i = currentIdx; i < nextIdx; i++) {
        const phase = PHASE_ORDER[i];
        const bookkeepingType = getBookkeepingMoveType(phase, currentState, currentPlayer);
        if (bookkeepingType) {
          synthesized.push({
            id: `synthesized-${bookkeepingType}-${moveNumberBase + synthesized.length}`,
            type: bookkeepingType,
            player: currentPlayer,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: moveNumberBase + synthesized.length,
          });
        }
      }
    }
  }

  return synthesized;
}

/**
 * Get the phase a move type belongs to.
 */
function getMovePhase(moveType: Move['type']): GamePhase | null {
  switch (moveType) {
    case 'place_ring':
    case 'skip_placement':
    case 'no_placement_action':
      return 'ring_placement';
    case 'move_stack':
    case 'move_ring':
    case 'no_movement_action':
    case 'recovery_slide':
      return 'movement';
    case 'overtaking_capture':
    case 'continue_capture_segment':
    case 'skip_capture':
      return 'capture';
    case 'process_line':
    case 'choose_line_reward':
    case 'no_line_action':
      return 'line_processing';
    case 'process_territory_region':
    case 'skip_territory_processing':
    case 'no_territory_action':
    case 'eliminate_rings_from_stack':
      return 'territory_processing';
    case 'forced_elimination':
      return 'forced_elimination';
    default:
      return null;
  }
}

function summarizeState(label: string, state: GameState): Record<string, unknown> {
  return {
    label,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
    moveHistoryLength: state.moveHistory.length,
    // Per-state ANM classification for the active player. This mirrors the
    // Python is_anm_state() helper and is consumed by the TS↔Python replay
    // parity harness when available, but remains purely additive metadata for
    // other tools that consume these summaries.
    is_anm: isANMState(state),
    // Use the compact hash that matches Python's _compute_state_hash for
    // cross-engine parity checks and GameReplayDB history entries.
    stateHash: hashGameStateSHA256(state),
  };
}

async function runReplayMode(args: ReplayCliArgs): Promise<void> {
  const { dbPath, gameId, dumpStateAt, skipGameIds } = args;
  const skipSet = new Set<string>([...DEFAULT_SKIP_GAME_IDS, ...(skipGameIds ?? [])]);

  const service = getSelfPlayGameService();
  const detail = service.getGame(dbPath, gameId);

  if (!detail) {
    console.error(`Game ${gameId} not found in DB ${dbPath}`);
    process.exitCode = 1;
    return;
  }

  if (skipSet.has(gameId)) {
    console.error(`[ts-replay] Skipping game ${gameId} (known-bad or listed in --skip-game)`);
    return;
  }

  // Optional debug configuration: dump TS GameState JSON at specified move numbers.
  // Supports both CLI args (--dump-state-at 10,20,30) and env var (legacy).
  // Parse env var which may be comma-separated (e.g., "50,51,52")
  const dumpKRaw = process.env.RINGRIFT_TS_REPLAY_DUMP_STATE_AT_K;
  const dumpKSet = new Set<number>(dumpStateAt ?? []);
  if (dumpKRaw) {
    const parts = dumpKRaw.split(',').map((s) => s.trim());
    for (const part of parts) {
      if (part === 'all') {
        // Special value to dump at every step (handled in the loop)
        dumpKSet.add(-1); // Sentinel for "dump all"
      } else {
        const parsed = Number.parseInt(part, 10);
        if (Number.isFinite(parsed)) {
          dumpKSet.add(parsed);
        }
      }
    }
  }
  const shouldDumpState = dumpKSet.size > 0;
  const dumpAll = dumpKSet.has(-1);
  const dumpDirEnv = process.env.RINGRIFT_TS_REPLAY_DUMP_DIR;
  const dumpDir =
    dumpDirEnv && dumpDirEnv.length > 0 ? dumpDirEnv : path.join(process.cwd(), 'ts-replay-dumps');

  // Sanitize initial state (clear history arrays to ensure clean replay)
  let initialState: unknown = undefined;
  const rawState = detail.initialState as any;
  if (rawState && typeof rawState === 'object') {
    const sanitized = { ...rawState };
    if (Array.isArray(sanitized.moveHistory)) {
      sanitized.moveHistory = [];
    }
    if (Array.isArray(sanitized.history)) {
      sanitized.history = [];
    }
    initialState = sanitized;
  } else {
    console.error(
      `[selfplay-db-ts-replay] No initial state found for ${gameId}; creating fresh state for ${detail.boardType} with ${detail.numPlayers} players`
    );
  }

  // Create CanonicalReplayEngine - coercion-free replay via TurnEngineAdapter
  const engine = new CanonicalReplayEngine({
    gameId,
    boardType: detail.boardType as BoardType,
    numPlayers: detail.numPlayers,
    initialState,
  });

  const normalizedMoves: Move[] = detail.moves.map((m) =>
    buildCanonicalMoveFromSelfPlayRecord(m, m.moveNumber)
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
      summary: { ...engine.summarize('initial'), label: 'initial' },
    })
  );

  let applied = 0;
  let synthesizedCount = 0;
  let fsmValidationFailures = 0;
  // Track the last DB move index we've emitted a "complete" state for
  let lastEmittedDbMoveComplete = -1;

  // LPS (Last-Player-Standing) tracking for R172 victory condition.
  // This must be kept in sync across move applications to detect when one
  // player has been the exclusive real-action holder for 2 consecutive rounds.
  const lpsState = createLpsTrackingState();

  // Helper to build action availability delegates for hasAnyRealAction
  const buildActionDelegates = (state: GameState) => {
    const turnForPlayer = (pn: number): PerTurnState => ({
      hasPlacedThisTurn: pn === state.currentPlayer && Boolean(state.mustMoveFromStackKey),
      mustMoveFromStackKey:
        pn === state.currentPlayer ? (state.mustMoveFromStackKey ?? undefined) : undefined,
    });

    return {
      hasPlacement: (pn: number) => hasAnyPlacementForPlayer(state, pn),
      hasMovement: (pn: number) => hasAnyMovementForPlayer(state, pn, turnForPlayer(pn)),
      hasCapture: (pn: number) => hasAnyCaptureForPlayer(state, pn, turnForPlayer(pn)),
    };
  };

  /**
   * Unified victory evaluation combining base victory conditions with LPS tracking.
   *
   * This helper consolidates the two-step victory check pattern:
   * 1. evaluateVictory() - Ring elimination, territory control, bare-board stalemate
   * 2. evaluateLpsVictory() - Round-based Last-Player-Standing (R172)
   *
   * Returns a combined result indicating whether the game has ended and the winner.
   *
   * @param state - Current game state
   * @param lps - LPS tracking state (round-based counters)
   * @returns Victory evaluation result with isGameOver, winner, and reason
   */
  const evaluateVictoryWithLps = (
    state: GameState,
    lps: LpsTrackingState
  ): { isGameOver: boolean; winner?: number; reason?: string; isLpsVictory?: boolean } => {
    // First check base victory conditions (ring elimination, territory, stalemate)
    const baseVerdict = evaluateVictory(state);
    if (baseVerdict.isGameOver && baseVerdict.winner !== undefined) {
      return {
        isGameOver: true,
        winner: baseVerdict.winner,
        reason: baseVerdict.reason,
        isLpsVictory: false,
      };
    }

    // If game is still active, check LPS victory (R172)
    if (state.gameStatus === 'active') {
      const delegates = buildActionDelegates(state);
      const lpsVerdict = evaluateLpsVictory({
        gameState: state,
        lps,
        hasAnyRealAction: (pn) => hasAnyRealAction(state, pn, delegates),
        hasMaterial: (pn) => playerHasMaterial(state, pn),
      });

      if (lpsVerdict.isVictory && lpsVerdict.winner !== undefined) {
        return {
          isGameOver: true,
          winner: lpsVerdict.winner,
          reason: 'last_player_standing',
          isLpsVictory: true,
        };
      }
    }

    return { isGameOver: false };
  };

  for (let i = 0; i < recordedMoves.length; i++) {
    const move = recordedMoves[i];

    // Check if we need to synthesize bookkeeping moves to bridge phases
    const currentState = engine.getState() as GameState;
    const prePhase = currentState.currentPhase;

    // Per RR-CANON-R073: ALL players start in ring_placement without exception.
    // NO PHASE SKIPPING - both TS and Python now always start in ring_placement.
    // This skip logic is retained for backwards compatibility with old game databases
    // that were recorded before the no-phase-skipping fix (2025-12).
    if (move.type === 'no_placement_action' && currentState.currentPhase === 'movement') {
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          kind: 'ts-replay-skip-redundant',
          db_move_index: i,
          moveType: move.type,
          currentPhase: currentState.currentPhase,
          reason: 'Legacy DB: already in movement phase, no_placement_action is redundant',
        })
      );
      continue;
    }

    // Skip redundant no_line_action moves recorded during phases where line processing
    // doesn't occur (e.g., chain_capture). These are legacy recording artifacts.
    // IMPORTANT: Only skip if the move is for the SAME player as current state.
    // If the move is for a different player, we need to apply it to trigger turn rotation.
    if (
      move.type === 'no_line_action' &&
      move.player === currentState.currentPlayer &&
      currentState.currentPhase !== 'line_processing'
    ) {
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          kind: 'ts-replay-skip-redundant',
          db_move_index: i,
          moveType: move.type,
          currentPhase: currentState.currentPhase,
          reason: 'no_line_action is only valid in line_processing phase',
        })
      );
      continue;
    }

    // Skip redundant no_movement_action moves recorded during phases where movement
    // doesn't occur (e.g., territory_processing, line_processing). These are legacy
    // recording artifacts from older Python versions.
    // IMPORTANT: Only skip if the move is for the SAME player as current state.
    // If the move is for a different player, we need to apply it to trigger turn rotation.
    if (
      move.type === 'no_movement_action' &&
      move.player === currentState.currentPlayer &&
      currentState.currentPhase !== 'movement' &&
      currentState.currentPhase !== 'capture' &&
      currentState.currentPhase !== 'chain_capture'
    ) {
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          kind: 'ts-replay-skip-redundant',
          db_move_index: i,
          moveType: move.type,
          currentPhase: currentState.currentPhase,
          reason: 'no_movement_action is only valid in movement/capture phases',
        })
      );
      continue;
    }

    // Stop processing if the game is over (legacy recordings may have trailing moves)
    if (currentState.currentPhase === 'game_over' || currentState.gameStatus === 'completed') {
      // Emit db-move-complete for the previous move before stopping
      if (i > 0 && lastEmittedDbMoveComplete < i - 1) {
        // eslint-disable-next-line no-console
        console.log(
          JSON.stringify({
            kind: 'ts-replay-db-move-complete',
            db_move_index: i - 1,
            summary: {
              ...summarizeState(`db_move_${i - 1}_complete`, currentState),
              // view: 'post_bridge' – state after closing out canonical DB move
              // db_move_index, including any synthesized bookkeeping moves
              // required before the next recorded move can be applied.
              view: 'post_bridge',
            },
          })
        );
        lastEmittedDbMoveComplete = i - 1;
      }
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          kind: 'ts-replay-game-ended',
          appliedMoves: applied,
          remainingRecordedMoves: recordedMoves.length - i,
          summary: summarizeState('game_ended', currentState),
        })
      );
      break;
    }
    const bridgeMoves = synthesizeBookkeepingMoves(currentState, move, applied + 1);

    // Apply any synthesized bookkeeping moves first
    for (const bridgeMove of bridgeMoves) {
      synthesizedCount += 1;

      // FSM parity validation for bridge moves
      const bridgeStateBeforeMove = engine.getState() as GameState;
      const bridgeFsmValidation = validateMoveWithFSM(bridgeStateBeforeMove, bridgeMove);
      if (!bridgeFsmValidation.valid) {
        fsmValidationFailures += 1;
        console.error(
          JSON.stringify({
            kind: 'ts-replay-fsm-bridge-validation-warning',
            k: applied,
            bridgeMove: {
              type: bridgeMove.type,
              player: bridgeMove.player,
            },
            fsmPhase: bridgeFsmValidation.currentPhase,
            gamePhase: bridgeStateBeforeMove.currentPhase,
            errorCode: bridgeFsmValidation.errorCode,
            reason: bridgeFsmValidation.reason,
          })
        );
        // Do not apply this bridge move or any subsequent ones for this DB move.
        // Falling back to a non-bridge path avoids desynchronising TS phase/player
        // relative to the DB history when the FSM rejects a synthesized move.
        break;
      }

      const bridgeResult = await engine.applyMove(bridgeMove);
      if (!bridgeResult.success) {
        console.error(
          JSON.stringify({
            kind: 'ts-replay-bridge-error',
            k: applied,
            bridgeMove,
            error: bridgeResult.error,
            targetMove: move,
            stateSummary: summarizeState('bridge-error', engine.getState() as GameState),
          })
        );
        // Abort replay for this game to avoid applying further DB moves from a
        // desynchronised state after a failed bridge application.
        return;
      }

      // FSM orchestration trace for bridge moves
      const bridgeState = engine.getState() as GameState;
      const bridgeFsmOrch = computeFSMOrchestration(bridgeState, bridgeMove, {
        postMoveStateForChainCheck: bridgeState,
      });

      const dbMoveIndexForBridge = applied > 0 ? applied - 1 : 0;
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          kind: 'ts-replay-bridge',
          db_move_index: dbMoveIndexForBridge,
          synthesizedMoveType: bridgeMove.type,
          synthesizedPlayer: bridgeMove.player,
          summary: {
            ...summarizeState('after_bridge', bridgeState),
            // view: 'bridge' – state after a synthesized bookkeeping move
            // associated with db_move_index, bridging phases between canonical
            // DB moves without changing the underlying engine/rules semantics.
            view: 'bridge',
          },
          // FSM action trace for bridge move
          fsm: {
            success: bridgeFsmOrch.success,
            nextPhase: bridgeFsmOrch.nextPhase,
            nextPlayer: bridgeFsmOrch.nextPlayer,
            actions: bridgeFsmOrch.actions.map((a) => a.type),
            pendingDecisionType: bridgeFsmOrch.pendingDecisionType,
            ...(bridgeFsmOrch.error && { error: bridgeFsmOrch.error }),
          },
        })
      );
    }

    // After applying bridges (but before applying this move), emit the "complete" state
    // for the PREVIOUS DB move. This state includes all bridges that were needed to
    // transition from move i-1's phase to a phase where move i is valid.
    // This matches Python's get_state_at_move(i-1) which includes internal phase transitions.
    if (i > 0 && lastEmittedDbMoveComplete < i - 1) {
      const stateForPreviousMove = engine.getState() as GameState;
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          kind: 'ts-replay-db-move-complete',
          db_move_index: i - 1,
          summary: {
            ...summarizeState(`db_move_${i - 1}_complete`, stateForPreviousMove),
            // view: 'post_bridge' – state after closing out canonical DB move
            // db_move_index, including any synthesized bookkeeping moves
            // required before the next recorded move can be applied.
            view: 'post_bridge',
          },
        })
      );
      lastEmittedDbMoveComplete = i - 1;
    }

    applied += 1;

    // FSM parity validation: validate move against FSM before applying
    // This catches any divergence between the recorded move and FSM rules
    const stateBeforeMove = engine.getState() as GameState;
    const fsmValidation = validateMoveWithFSM(stateBeforeMove, move);
    if (!fsmValidation.valid) {
      fsmValidationFailures += 1;
      // Log FSM validation failure but continue - the move may still apply
      // via the engine's own validation (FSM may be more restrictive)
      console.error(
        JSON.stringify({
          kind: 'ts-replay-fsm-validation-warning',
          k: applied,
          db_move_index: i,
          move: {
            type: move.type,
            player: move.player,
            from: move.from,
            to: move.to,
          },
          fsmPhase: fsmValidation.currentPhase,
          gamePhase: stateBeforeMove.currentPhase,
          errorCode: fsmValidation.errorCode,
          reason: fsmValidation.reason,
          validEventTypes: fsmValidation.validEventTypes,
        })
      );
    }

    const result = await engine.applyMove(move);

    if (!result.success) {
      const state = engine.getState();
      console.error(
        JSON.stringify(
          {
            kind: 'ts-replay-error',
            k: applied,
            move,
            error: result.error,
            stateSummary: summarizeState('error', state as GameState),
          },
          null,
          2
        )
      );
      throw new Error(result.error);
    }

    const state = engine.getState();
    const stateTyped = state as GameState;

    // Check for immediate victory conditions (ring elimination, territory, Early LPS)
    // This must happen after every move to match Python's apply_move behavior.
    // Python's GameEngine.apply_move calls _check_victory after EVERY move and
    // terminates the game immediately when conditions are met.
    const verdict = evaluateVictory(stateTyped);
    const earlyVictoryDetected =
      verdict.isGameOver && verdict.winner !== undefined && stateTyped.gameStatus !== 'completed';

    if (earlyVictoryDetected) {
      console.log(
        JSON.stringify({
          kind: 'ts-replay-early-victory',
          k: applied,
          winner: verdict.winner,
          reason: verdict.reason,
        })
      );
      // Update the engine state to match Python's behavior - terminate immediately
      // We need to cast to mutable and set the fields directly since ClientSandboxEngine
      // doesn't expose a direct way to end the game.
      const mutableState = stateTyped as any;
      mutableState.gameStatus = 'completed';
      mutableState.winner = verdict.winner;
      mutableState.currentPhase = 'game_over';
      // Python's _check_victory sets current_player to winner for TS parity
      mutableState.currentPlayer = verdict.winner;
    }

    // Update LPS tracking for R172 only at the START of an interactive turn.
    // We detect turn starts by a rotation to a new currentPlayer in an
    // interactive phase (ring_placement / movement / capture / chain_capture).
    if (
      stateTyped.gameStatus === 'active' &&
      stateTyped.currentPhase === 'ring_placement' &&
      prePhase !== 'ring_placement' &&
      isLpsActivePhase(stateTyped.currentPhase)
    ) {
      const activePlayers = stateTyped.players
        .filter((p) => playerHasMaterial(stateTyped, p.playerNumber))
        .map((p) => p.playerNumber);
      const delegates = buildActionDelegates(stateTyped);
      const hasReal = hasAnyRealAction(stateTyped, stateTyped.currentPlayer, delegates);

      updateLpsTracking(lpsState, {
        currentPlayer: stateTyped.currentPlayer,
        activePlayers,
        hasRealAction: hasReal,
      });

      if (process.env.RINGRIFT_TRACE_LPS === '1') {
        // eslint-disable-next-line no-console
        console.log(
          JSON.stringify({
            kind: 'ts-replay-lps-update',
            k: applied,
            currentPlayer: stateTyped.currentPlayer,
            activePlayers,
            hasRealAction: hasReal,
            lpsState: {
              roundIndex: lpsState.roundIndex,
              currentRoundFirstPlayer: lpsState.currentRoundFirstPlayer,
              exclusivePlayerForCompletedRound: lpsState.exclusivePlayerForCompletedRound,
              consecutiveExclusiveRounds: lpsState.consecutiveExclusiveRounds,
              consecutiveExclusivePlayer: lpsState.consecutiveExclusivePlayer,
            },
          })
        );
      }

      // Check for LPS victory after updating tracking
      const lpsResult = evaluateLpsVictory({
        gameState: stateTyped,
        lps: lpsState,
        hasAnyRealAction: (pn) => hasAnyRealAction(stateTyped, pn, delegates),
        hasMaterial: (pn) => playerHasMaterial(stateTyped, pn),
      });

      if (lpsResult.isVictory && lpsResult.winner !== undefined) {
        // LPS victory detected - update the engine state to reflect game over
        // This keeps parity with Python's LPS detection
        console.log(
          JSON.stringify({
            kind: 'ts-replay-lps-victory',
            k: applied,
            winner: lpsResult.winner,
            lpsState: {
              consecutiveExclusiveRounds: lpsState.consecutiveExclusiveRounds,
              consecutiveExclusivePlayer: lpsState.consecutiveExclusivePlayer,
              roundIndex: lpsState.roundIndex,
            },
          })
        );
        const mutableState = stateTyped as any;
        mutableState.gameStatus = 'completed';
        mutableState.winner = lpsResult.winner;
        mutableState.currentPhase = 'game_over';
        mutableState.currentPlayer = lpsResult.winner;
      }
    }

    // FSM orchestration trace: compute what the FSM would do after this move
    // This provides action traces for parity analysis and debugging
    const fsmOrchestration = computeFSMOrchestration(state as GameState, move, {
      postMoveStateForChainCheck: state as GameState,
    });

    // Optional debug dump for parity investigation
    if (shouldDumpState && (dumpAll || dumpKSet.has(applied))) {
      try {
        fs.mkdirSync(dumpDir, { recursive: true });
        const fileName = `${path.basename(dbPath)}__${gameId}__k${applied}.ts_state.json`;
        const outPath = path.join(dumpDir, fileName);
        // Use serializeGameState to properly convert Map objects to plain objects
        fs.writeFileSync(
          outPath,
          JSON.stringify(serializeGameState(state as GameState), null, 2),
          'utf-8'
        );

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

    const stepDbMoveIndex = applied > 0 ? applied - 1 : null;
    // eslint-disable-next-line no-console
    console.log(
      JSON.stringify({
        kind: 'ts-replay-step',
        k: applied,
        db_move_index: stepDbMoveIndex,
        moveType: move.type,
        movePlayer: move.player,
        moveNumber: move.moveNumber,
        // view: 'post_move' – state immediately after applying DB move
        // db_move_index, before any synthesized bookkeeping for subsequent
        // phases. This is the canonical TS post_move[N] view used for
        // TS↔Python parity in post_move mode.
        summary: {
          ...summarizeState(`after_move_${applied}`, state as GameState),
          view: 'post_move',
        },
        // FSM action trace: what the FSM computed for this move
        fsm: {
          success: fsmOrchestration.success,
          nextPhase: fsmOrchestration.nextPhase,
          nextPlayer: fsmOrchestration.nextPlayer,
          actions: fsmOrchestration.actions.map((a) => a.type),
          pendingDecisionType: fsmOrchestration.pendingDecisionType,
          ...(fsmOrchestration.error && { error: fsmOrchestration.error }),
        },
      })
    );

    // If early victory was detected, return immediately after emitting the step
    // This ensures the ts-replay-step is recorded before we exit the loop
    if (earlyVictoryDetected) {
      console.log(
        JSON.stringify({
          kind: 'ts-replay-game-ended',
          appliedMoves: applied,
          remainingRecordedMoves: recordedMoves.length - applied,
          summary: summarizeState('game_ended', engine.getState() as GameState),
        })
      );

      return {
        state: engine.getState() as GameState,
        appliedMoves: applied,
        synthesizedMoves: 0,
        fsmValidationFailures: fsmValidationFailures,
      };
    }
  }

  // Emit db-move-complete for the final recorded move (no bridges needed since there's no next move)
  if (recordedMoves.length > 0 && lastEmittedDbMoveComplete < recordedMoves.length - 1) {
    const finalDbMoveIndex = recordedMoves.length - 1;
    const stateForFinalMove = engine.getState() as GameState;
    // eslint-disable-next-line no-console
    console.log(
      JSON.stringify({
        kind: 'ts-replay-db-move-complete',
        db_move_index: finalDbMoveIndex,
        summary: {
          ...summarizeState(`db_move_${finalDbMoveIndex}_complete`, stateForFinalMove),
          // view: 'post_bridge' – state after closing out canonical DB move
          // db_move_index, including any synthesized bookkeeping moves
          // required before the next recorded move can be applied.
          view: 'post_bridge',
        },
      })
    );
  }

  // Recompute terminal victory per canonical rules, regardless of any
  // terminal metadata that may have been recorded in the DB. This ensures
  // bare-board stalemate tie-breakers (territory → eliminated → markers →
  // last actor) are applied even when legacy logs omit a winner, and keeps
  // parity with the shared VictoryAggregate semantics.
  //
  // Uses evaluateVictoryWithLps() to combine:
  // 1. evaluateVictory() - Ring elimination, territory control, bare-board stalemate
  // 2. evaluateLpsVictory() - Round-based Last-Player-Standing (R172)
  let finalState = engine.getState() as GameState;
  const terminalVerdict = evaluateVictoryWithLps(finalState, lpsState);
  const finalWinner: number | undefined = terminalVerdict.winner;
  const finalIsGameOver = terminalVerdict.isGameOver;

  // Log LPS victory if that's how the game ended
  if (terminalVerdict.isLpsVictory && terminalVerdict.winner !== undefined) {
    console.log(
      JSON.stringify({
        kind: 'ts-replay-lps-victory-final',
        winner: terminalVerdict.winner,
        lpsState: {
          consecutiveExclusiveRounds: lpsState.consecutiveExclusiveRounds,
          consecutiveExclusivePlayer: lpsState.consecutiveExclusivePlayer,
          roundIndex: lpsState.roundIndex,
        },
      })
    );
  }

  if (finalIsGameOver) {
    finalState = {
      ...finalState,
      gameStatus: 'completed',
      winner: finalWinner,
      currentPhase: 'game_over',
    };
  }

  // If the final k was requested for dumping, overwrite it with the recomputed
  // terminal state so parity bundles see the canonical phase/winner.
  if (shouldDumpState && (dumpAll || dumpKSet.has(applied))) {
    try {
      fs.mkdirSync(dumpDir, { recursive: true });
      const fileName = `${path.basename(dbPath)}__${gameId}__k${applied}.ts_state.json`;
      const outPath = path.join(dumpDir, fileName);
      fs.writeFileSync(outPath, JSON.stringify(serializeGameState(finalState), null, 2), 'utf-8');
    } catch (err) {
      console.error(
        `[selfplay-db-ts-replay] Failed to dump FINAL TS state for ${gameId} @ k=${applied}:`,
        err
      );
    }
  }

  // Emit a final step summary with the recomputed terminal state. The parity
  // parser keeps the last summary for a given k, so this overrides the
  // pre-recompute summary for the last move.
  const finalStepDbMoveIndex = recordedMoves.length > 0 ? recordedMoves.length - 1 : null;
  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify({
      kind: 'ts-replay-step',
      k: applied,
      db_move_index: finalStepDbMoveIndex,
      moveType: recordedMoves.length > 0 ? recordedMoves[recordedMoves.length - 1].type : undefined,
      movePlayer:
        recordedMoves.length > 0 ? recordedMoves[recordedMoves.length - 1].player : undefined,
      moveNumber:
        recordedMoves.length > 0 ? recordedMoves[recordedMoves.length - 1].moveNumber : undefined,
      summary: {
        ...summarizeState(`after_move_${applied}_final`, finalState),
        // view: 'post_move' – terminal state after the final DB move's
        // effects have been applied and victory has been recomputed
        // canonically. This overrides the earlier post_move summary at the
        // same k for parity diagnostics.
        view: 'post_move',
      },
    })
  );
  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify({
      kind: 'ts-replay-final',
      appliedMoves: applied,
      synthesizedMoves: synthesizedCount,
      fsmValidationFailures,
      summary: summarizeState('final', finalState),
    })
  );

  // Emit FSM parity summary if there were any validation failures
  if (fsmValidationFailures > 0) {
    console.error(
      `[selfplay-db-ts-replay] FSM parity warning: ${fsmValidationFailures} move(s) failed FSM validation for game ${gameId}`
    );
  }
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
