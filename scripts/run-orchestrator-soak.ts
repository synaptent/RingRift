#!/usr/bin/env ts-node
/**
 * Orchestrator soak & invariant harness.
 *
 * Runs many random self-play games against the shared TS orchestrator via
 * real hosts (backend GameEngine and/or ClientSandboxEngine) and checks
 * basic safety/correctness invariants. Produces a machine-readable JSON
 * summary suitable for local inspection or CI/monitoring ingestion.
 */

import fs from 'fs';
import path from 'path';

import { GameEngine } from '../src/server/game/GameEngine';
import { getMetricsService } from '../src/server/services/MetricsService';

import type { BoardType, GameState, Move, Player, TimeControl } from '../src/shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../src/shared/types/game';

import { SeededRNG } from '../src/shared/utils/rng';
import { computeProgressSnapshot, isANMState } from '../src/shared/engine';
import { validateMove as orchestratorValidateMove } from '../src/shared/engine/orchestration/turnOrchestrator';

type HostMode = 'backend';
type HarnessMode = 'backend';

type SoakProfileId = 'ci-short' | 'local-deep';

interface SoakProfile {
  id: SoakProfileId;
  description: string;
  boardTypes: BoardType[];
  gamesPerBoard: number;
  maxTurns: number;
  /**
   * Root RNG seed used to derive per-game seeds. Given the same profile +
   * randomSeed, the soak run is fully deterministic.
   */
  randomSeed: number;
}

interface SoakConfig {
  /**
   * Optional named profile controlling default board types, gamesPerBoard,
   * maxTurns, and randomSeed. CLI flags can still override these values.
   *
   * Supported profile ids:
   * - "ci-short"   – CI-safe multi-board short soak.
   * - "local-deep" – deeper multi-board soak for local or scheduled runs.
   */
  profile?: SoakProfileId;
  boardTypes: BoardType[];
  gamesPerBoard: number;
  maxTurns: number;
  randomSeed: number;
  mode: HarnessMode;
  outputPath: string;
  /**
   * When true, the harness will exit with a non-zero status code if any
   * invariant violations are detected. This is useful when wiring the
   * script into automated test runners or CI gates.
   */
  failOnViolation: boolean;
  /**
   * When true, the harness forces RINGRIFT_TRACE_DEBUG=1 for the duration
   * of the run so that strict S-invariant and elimination bookkeeping logs
   * from GameEngine/TurnEngineAdapter are emitted without requiring callers
   * to remember the environment flag.
   */
  enableTraceDebug: boolean;
}

const SOAK_PROFILES: Record<SoakProfileId, SoakProfile> = {
  'ci-short': {
    id: 'ci-short',
    description:
      'CI short profile: multi-board (square8, square19, hexagonal) with modest gamesPerBoard and bounded maxTurns.',
    boardTypes: ['square8', 'square19', 'hexagonal'],
    gamesPerBoard: 5,
    maxTurns: 300,
    randomSeed: 123456789,
  },
  'local-deep': {
    id: 'local-deep',
    description:
      'Local deep profile: multi-board coverage with more games per board and higher maxTurns, intended for manual or scheduled runs.',
    boardTypes: ['square8', 'square19', 'hexagonal'],
    gamesPerBoard: 20,
    maxTurns: 400,
    randomSeed: 987654321,
  },
};

interface GameRunResult {
  boardType: BoardType;
  hostMode: HostMode;
  gameIndex: number;
  seed: number;
  turns: number;
  completed: boolean;
  hitMaxTurns: boolean;
  invariantViolations: InvariantViolation[];
}

interface BucketStats {
  boardType: BoardType;
  hostMode: HostMode;
  gameCount: number;
  completedCount: number;
  maxTurnsCount: number;
  invariantViolationCount: number;
  turnCounts: number[];
  /**
   * Per-violation-id counts within this (boardType, hostMode) bucket. This
   * provides invariant-level visibility without changing the existing
   * aggregated fields.
   */
  invariantViolationsById: Record<string, number>;
}

interface InvariantViolation {
  id: string;
  message: string;
  boardType: BoardType;
  hostMode: HostMode;
  gameId: string;
  gameIndex: number;
  turnIndex: number;
  seed: number;
  gameStatus: string;
  currentPlayer: number;
  currentPhase: string;
  markers: number;
  collapsed: number;
  sInvariant: number;
  eliminatedRings: number;
  totalRingsEliminated: number;
  movesTail: Move[];
}

interface ParsedArgs {
  [key: string]: string | boolean | undefined;
}

function parseArgs(argv: string[]): SoakConfig {
  const args: ParsedArgs = {};
  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw.startsWith('--')) {
      continue;
    }
    const eqIndex = raw.indexOf('=');
    let key: string;
    let value: string | boolean | undefined;
    if (eqIndex !== -1) {
      key = raw.slice(2, eqIndex);
      value = raw.slice(eqIndex + 1);
    } else {
      key = raw.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        value = next;
        i += 1;
      } else {
        value = true;
      }
    }
    args[key] = value;
  }

  const profileArg = args.profile as string | undefined;
  let profileId: SoakProfileId | undefined;
  if (profileArg) {
    if (profileArg in SOAK_PROFILES) {
      profileId = profileArg as SoakProfileId;
    } else {
      // eslint-disable-next-line no-console
      console.warn(
        `Unknown orchestrator soak profile "${profileArg}". Falling back to explicit CLI arguments.`
      );
    }
  }
  const profile = profileId ? SOAK_PROFILES[profileId] : undefined;

  const boardTypesArg =
    (args.boardTypes as string | undefined) ??
    (profile ? profile.boardTypes.join(',') : 'square8,square19,hexagonal');
  const boardTypes = boardTypesArg
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0) as BoardType[];

  const gamesPerBoardRaw =
    args.gamesPerBoard !== undefined
      ? Number(args.gamesPerBoard)
      : profile
      ? profile.gamesPerBoard
      : 50;
  const maxTurnsRaw =
    args.maxTurns !== undefined
      ? Number(args.maxTurns)
      : profile
      ? profile.maxTurns
      : 500;
  const randomSeedRaw =
    args.randomSeed !== undefined
      ? Number(args.randomSeed)
      : profile
      ? profile.randomSeed
      : Date.now() & 0x7fffffff;

  const mode: HarnessMode = 'backend';
  const outputPath =
    (args.outputPath as string | undefined) ?? 'results/orchestrator_soak_summary.json';

  const failOnViolation =
    args.failOnViolation === true ||
    args.failOnViolation === 'true' ||
    args.failOnInvariantViolation === true ||
    args.failOnInvariantViolation === 'true';

  const enableTraceDebug =
    args.debug === true ||
    args.debug === 'true' ||
    args.traceDebug === true ||
    args.traceDebug === 'true';

  const gamesPerBoard =
    Number.isFinite(gamesPerBoardRaw) && gamesPerBoardRaw > 0 ? gamesPerBoardRaw : 50;
  const maxTurns =
    Number.isFinite(maxTurnsRaw) && maxTurnsRaw > 0 ? maxTurnsRaw : 500;
  const randomSeed = Number.isFinite(randomSeedRaw)
    ? randomSeedRaw
    : Date.now() & 0x7fffffff;

  const config: SoakConfig = {
    boardTypes,
    gamesPerBoard,
    maxTurns,
    randomSeed,
    mode,
    outputPath,
    failOnViolation,
    enableTraceDebug,
  };

  if (profileId) {
    config.profile = profileId;
  }

  return config;
}

const DEFAULT_TIME_CONTROL: TimeControl = {
  initialTime: 600,
  increment: 0,
  type: 'blitz',
};

function createDefaultPlayers(boardType: BoardType): Player[] {
  const ringsPerPlayer = BOARD_CONFIGS[boardType].ringsPerPlayer;
  return [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: DEFAULT_TIME_CONTROL.initialTime * 1000,
      ringsInHand: ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: DEFAULT_TIME_CONTROL.initialTime * 1000,
      ringsInHand: ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

function createBackendHost(gameId: string, boardType: BoardType, gameSeed: number): GameEngine {
  const players = createDefaultPlayers(boardType);
  const engine = new GameEngine(
    gameId,
    boardType,
    players,
    DEFAULT_TIME_CONTROL,
    false,
    undefined,
    gameSeed
  );

  // Ensure orchestrator adapter is enabled regardless of config flags.
  engine.enableOrchestratorAdapter();

  // Mark all players ready and seed rng for reproducibility.
  const engineAny = engine as any;
  if (engineAny.gameState && Array.isArray(engineAny.gameState.players)) {
    engineAny.gameState.players.forEach((p: any) => {
      p.isReady = true;
    });
    engineAny.gameState.rngSeed = gameSeed;
  }

  if (!engine.startGame()) {
    throw new Error(`Failed to start GameEngine for gameId=${gameId}`);
  }

  return engine;
}

// NOTE: Sandbox host support is intentionally disabled in this harness for now.
// Pulling in the client-side sandbox engine under ts-node introduces strict
// type-checking errors that are orthogonal to orchestrator soak behaviour.
// The harness currently exercises the orchestrator via the backend GameEngine
// only. Future work can reintroduce a sandbox-backed soak entrypoint if needed.

function toEngineMove(move: Move): Omit<Move, 'id' | 'timestamp' | 'moveNumber'> {
  const { id, timestamp, moveNumber, ...rest } = move as any;
  return {
    ...rest,
    thinkTime: 0,
  };
}

function makeViolation(
  id: string,
  message: string,
  state: GameState,
  boardType: BoardType,
  hostMode: HostMode,
  context: {
    gameId: string;
    gameIndex: number;
    turnIndex: number;
    seed: number;
    lastMoves: Move[];
  },
  snapshot: {
    markers: number;
    collapsed: number;
    S: number;
    eliminated: number;
  },
  totalRingsEliminated: number
): InvariantViolation {
  const violation: InvariantViolation = {
    id,
    message,
    boardType,
    hostMode,
    gameId: context.gameId,
    gameIndex: context.gameIndex,
    turnIndex: context.turnIndex,
    seed: context.seed,
    gameStatus: state.gameStatus,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    markers: snapshot.markers,
    collapsed: snapshot.collapsed,
    sInvariant: snapshot.S,
    eliminatedRings: snapshot.eliminated,
    totalRingsEliminated,
    movesTail: context.lastMoves.slice(-10),
  };

  // Best-effort emission of orchestrator invariant metrics so that soak
  // runs contribute to the same per-invariant counters used by runtime
  // hosts. Metrics failures must never break the harness.
  try {
    const metrics = getMetricsService();
    metrics.recordOrchestratorInvariantViolation(violation.id);
  } catch {
    // Ignore metrics errors in soak runs.
  }

  return violation;
}

function checkStructuralInvariants(
  state: GameState,
  boardType: BoardType,
  hostMode: HostMode,
  context: {
    gameId: string;
    gameIndex: number;
    turnIndex: number;
    seed: number;
    lastMoves: Move[];
  }
): InvariantViolation[] {
  const violations: InvariantViolation[] = [];
  const snapshot = computeProgressSnapshot(state as any);
  const totalEliminated = state.totalRingsEliminated ?? 0;

  // Board stack invariants.
  for (const stack of state.board.stacks.values()) {
    if (stack.stackHeight < 0) {
      violations.push(
        makeViolation(
          'NEGATIVE_STACK_HEIGHT',
          `Stack height < 0 at ${positionToString(stack.position)}`,
          state,
          boardType,
          hostMode,
          context,
          snapshot,
          totalEliminated
        )
      );
      break;
    }
    if (stack.rings.length !== stack.stackHeight) {
      violations.push(
        makeViolation(
          'STACK_HEIGHT_MISMATCH',
          `stackHeight (${stack.stackHeight}) !== rings.length (${stack.rings.length}) at ${positionToString(stack.position)}`,
          state,
          boardType,
          hostMode,
          context,
          snapshot,
          totalEliminated
        )
      );
      break;
    }
    if (stack.capHeight < 0 || stack.capHeight > stack.stackHeight) {
      violations.push(
        makeViolation(
          'INVALID_CAP_HEIGHT',
          `capHeight (${stack.capHeight}) out of range at ${positionToString(stack.position)}`,
          state,
          boardType,
          hostMode,
          context,
          snapshot,
          totalEliminated
        )
      );
      break;
    }
  }

  // Eliminated-rings map non-negative.
  for (const [playerKey, count] of Object.entries(state.board.eliminatedRings ?? {})) {
    if (count < 0) {
      violations.push(
        makeViolation(
          'NEGATIVE_ELIMINATED_RINGS',
          `Negative eliminatedRings for player ${playerKey}: ${count}`,
          state,
          boardType,
          hostMode,
          context,
          snapshot,
          totalEliminated
        )
      );
      break;
    }
  }

  return violations;
}

async function runSingleGame(
  boardType: BoardType,
  hostMode: HostMode,
  gameIndex: number,
  seed: number,
  maxTurns: number
): Promise<GameRunResult> {
  const gameId = `soak-${boardType}-${hostMode}-${gameIndex}`;
  const moveRng = new SeededRNG(seed);

  const engine = createBackendHost(gameId, boardType, seed);

  let turns = 0;
  let completed = false;
  let hitMaxTurns = false;
  const invariantViolations: InvariantViolation[] = [];
  const maxViolationsPerGame = 8;
  const recentMoves: Move[] = [];

  let lastS: number | null = null;
  let lastTotalEliminated: number | null = null;

  for (; turns < maxTurns; turns += 1) {
    const state = (engine as GameEngine).getGameState();

    const snapshot = computeProgressSnapshot(state as any);
    const currentS = snapshot.S;
    const totalEliminated = state.totalRingsEliminated ?? 0;

    const contextBase = {
      gameId,
      gameIndex,
      turnIndex: turns,
      seed,
      lastMoves: recentMoves,
    };

    // Monotonic S-invariant.
    if (lastS !== null && currentS < lastS && invariantViolations.length < maxViolationsPerGame) {
      invariantViolations.push(
        makeViolation(
          'S_INVARIANT_DECREASED',
          `S-invariant decreased from ${lastS} to ${currentS}`,
          state,
          boardType,
          hostMode,
          contextBase,
          snapshot,
          totalEliminated
        )
      );
      break;
    }

    // Monotonic totalRingsEliminated.
    if (
      lastTotalEliminated !== null &&
      totalEliminated < lastTotalEliminated &&
      invariantViolations.length < maxViolationsPerGame
    ) {
      invariantViolations.push(
        makeViolation(
          'TOTAL_RINGS_ELIMINATED_DECREASED',
          `totalRingsEliminated decreased from ${lastTotalEliminated} to ${totalEliminated}`,
          state,
          boardType,
          hostMode,
          contextBase,
          snapshot,
          totalEliminated
        )
      );
      break;
    }

    lastS = currentS;
    lastTotalEliminated = totalEliminated;

    // Basic board-structure invariants.
    const structural = checkStructuralInvariants(state, boardType, hostMode, contextBase);
    if (structural.length > 0) {
      invariantViolations.push(
        ...structural.slice(0, maxViolationsPerGame - invariantViolations.length)
      );
      break;
    }

    // Termination check.
    if (state.gameStatus === 'completed') {
      completed = true;
      break;
    }
    if (state.gameStatus !== 'active') {
      if (invariantViolations.length < maxViolationsPerGame) {
        invariantViolations.push(
          makeViolation(
            'UNEXPECTED_GAME_STATUS',
            `Encountered non-active, non-completed gameStatus=${state.gameStatus}`,
            state,
            boardType,
            hostMode,
            contextBase,
            snapshot,
            totalEliminated
          )
        );
      }
      break;
    }

    const currentPlayer = state.currentPlayer;
    const moves = (engine as GameEngine).getValidMoves(currentPlayer);

    // After calling into the host's getValidMoves, the backend may have
    // auto-resolved blocked interactive states (forced elimination / skips)
    // and advanced the turn. Re-snapshot the state that the moves were
    // actually generated from so that orchestrator validation uses the same
    // phase/currentPlayer/board geometry as the host.
    const validationState = (engine as GameEngine).getGameState();
    const validationSnapshot = computeProgressSnapshot(validationState as any);
    const validationTotalEliminated = validationState.totalRingsEliminated ?? 0;

    // Strict active-no-move invariant, evaluated against the post-enumeration
    // state. This uses the shared-engine ANM predicate so that global
    // placements and forced elimination are treated as legal actions even
    // when the current micro-phase exposes no local moves.
    if (validationState.gameStatus === 'active' && isANMState(validationState as GameState)) {
      if (invariantViolations.length < maxViolationsPerGame) {
        invariantViolations.push(
          makeViolation(
            'ACTIVE_NO_MOVES',
            'GameStatus is active but getValidMoves returned 0 moves',
            validationState,
            boardType,
            hostMode,
            contextBase,
            validationSnapshot,
            validationTotalEliminated
          )
        );
      }
      break;
    }

    // Orchestrator validation invariant: every enumerated move must validate.
    for (const move of moves) {
      const validation = orchestratorValidateMove(validationState, move);
      if (!validation.valid) {
        if (invariantViolations.length < maxViolationsPerGame) {
          invariantViolations.push(
            makeViolation(
              'ORCHESTRATOR_VALIDATE_MOVE_FAILED',
              `orchestrator validateMove rejected move of type ${move.type} for player ${move.player} in phase ${validationState.currentPhase}: ${validation.reason ?? 'no reason provided'}`,
              validationState,
              boardType,
              hostMode,
              contextBase,
              validationSnapshot,
              validationTotalEliminated
            )
          );
        }
        break;
      }
    }
    if (invariantViolations.length > 0) {
      break;
    }

    // Prefer "real" actions when available (placements, movements, captures).
    const realMoveTypes: Move['type'][] = [
      'place_ring',
      'skip_placement',
      'move_ring',
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
    ];
    const realMoves = moves.filter((m) => realMoveTypes.includes(m.type));
    const candidateMoves = realMoves.length > 0 ? realMoves : moves;

    // Defensive: if there are no candidate moves to select from, do not attempt
    // to sample a move. This can occur if getValidMoves advanced the game out
    // of an active state and returned an empty list.
    if (candidateMoves.length === 0) {
      if (validationState.gameStatus !== 'active') {
        // Treat this as a terminal/non-interactive state reached as a
        // side-effect of getValidMoves. Mark completion when appropriate and
        // exit the turn loop.
        if (validationState.gameStatus === 'completed') {
          completed = true;
        }
        break;
      }

      // If the game is still reported as active but we have no candidate
      // moves, record an explicit invariant violation and stop this game.
      if (invariantViolations.length < maxViolationsPerGame) {
        invariantViolations.push(
          makeViolation(
            'ACTIVE_NO_CANDIDATE_MOVES',
            'GameStatus is active but no candidate moves were available for selection',
            validationState,
            boardType,
            hostMode,
            contextBase,
            validationSnapshot,
            validationTotalEliminated
          )
        );
      }
      break;
    }

    const selectedIndex = moveRng.nextInt(0, candidateMoves.length);
    const selectedMove = candidateMoves[selectedIndex];

    recentMoves.push(selectedMove);
    if (recentMoves.length > 16) {
      recentMoves.shift();
    }

    // Apply the move via backend host entrypoint.
    try {
      const engineMove = toEngineMove(selectedMove);
      const result = await (engine as GameEngine).makeMove(engineMove);
      if (!result.success) {
        if (invariantViolations.length < maxViolationsPerGame) {
          invariantViolations.push(
            makeViolation(
              'HOST_REJECTED_MOVE',
              `GameEngine.makeMove rejected move of type ${selectedMove.type}: ${result.error ?? 'no error message'}`,
              state,
              boardType,
              hostMode,
              contextBase,
              snapshot,
              totalEliminated
            )
          );
        }
        break;
      }
    } catch (err) {
      if (invariantViolations.length < maxViolationsPerGame) {
        const message = err instanceof Error ? err.message : String(err);
        invariantViolations.push(
          makeViolation(
            'UNHANDLED_EXCEPTION',
            `Exception while applying move of type ${selectedMove.type}: ${message}`,
            state,
            boardType,
            hostMode,
            contextBase,
            snapshot,
            totalEliminated
          )
        );
      }
      break;
    }
  }

  if (turns >= maxTurns && !completed && invariantViolations.length === 0) {
    hitMaxTurns = true;
  }

  return {
    boardType,
    hostMode,
    gameIndex,
    seed,
    turns,
    completed,
    hitMaxTurns,
    invariantViolations,
  };
}

function computeLengthStats(turnCounts: number[]): {
  min: number;
  max: number;
  median: number;
  p95: number;
} {
  if (turnCounts.length === 0) {
    return { min: 0, max: 0, median: 0, p95: 0 };
  }
  const sorted = [...turnCounts].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const median = sorted[Math.floor(sorted.length / 2)];
  const p95Index = Math.floor(0.95 * (sorted.length - 1));
  const p95 = sorted[p95Index];
  return { min, max, median, p95 };
}

async function run(): Promise<void> {
  const config = parseArgs(process.argv);

  if (config.enableTraceDebug) {
    const current = (process as any).env?.RINGRIFT_TRACE_DEBUG;
    if (!current) {
      // eslint-disable-next-line no-console
      console.log('Enabling RINGRIFT_TRACE_DEBUG=1 for orchestrator soak run (--debug).');
      (process as any).env.RINGRIFT_TRACE_DEBUG = '1';
    }
  }

  const hostModes: HostMode[] = ['backend'];

  console.log('Orchestrator soak harness starting with config:');
  console.log(JSON.stringify(config, null, 2));
  console.log('');

  const rng = new SeededRNG(config.randomSeed);

  const bucketMap = new Map<string, BucketStats>();
  const allViolations: InvariantViolation[] = [];
  const maxViolationTraces = 50;
  const overallViolationsById: Record<string, number> = {};

  const totalGamesPlanned = config.boardTypes.length * config.gamesPerBoard * hostModes.length;
  let gameCounter = 0;

  for (const boardType of config.boardTypes) {
    for (let gameIndex = 0; gameIndex < config.gamesPerBoard; gameIndex += 1) {
      const seed = rng.nextInt(1, 0x7fffffff);

      for (const hostMode of hostModes) {
        gameCounter += 1;
        // eslint-disable-next-line no-console
        console.log(
          `[${gameCounter}/${totalGamesPlanned}] boardType=${boardType} host=${hostMode} gameIndex=${gameIndex} seed=${seed}`
        );

        const result = await runSingleGame(boardType, hostMode, gameIndex, seed, config.maxTurns);

        const bucketKey = `${boardType}:${hostMode}`;
        let bucket = bucketMap.get(bucketKey);
        if (!bucket) {
          bucket = {
            boardType,
            hostMode,
            gameCount: 0,
            completedCount: 0,
            maxTurnsCount: 0,
            invariantViolationCount: 0,
            turnCounts: [],
            invariantViolationsById: {},
          };
          bucketMap.set(bucketKey, bucket);
        }

        bucket.gameCount += 1;
        bucket.turnCounts.push(result.turns);
        if (result.completed) {
          bucket.completedCount += 1;
        }
        if (result.hitMaxTurns) {
          bucket.maxTurnsCount += 1;
        }
        if (result.invariantViolations.length > 0) {
          bucket.invariantViolationCount += result.invariantViolations.length;

          for (const violation of result.invariantViolations) {
            bucket.invariantViolationsById[violation.id] =
              (bucket.invariantViolationsById[violation.id] ?? 0) + 1;
            overallViolationsById[violation.id] =
              (overallViolationsById[violation.id] ?? 0) + 1;
          }

          if (allViolations.length < maxViolationTraces) {
            const remaining = maxViolationTraces - allViolations.length;
            allViolations.push(...result.invariantViolations.slice(0, remaining));
          }
        }
      }
    }
  }

  // Build summary structure.
  const buckets = Array.from(bucketMap.values()).map((bucket) => {
    const stats = computeLengthStats(bucket.turnCounts);
    return {
      boardType: bucket.boardType,
      hostMode: bucket.hostMode,
      totalGames: bucket.gameCount,
      completedGames: bucket.completedCount,
      maxTurnsGames: bucket.maxTurnsCount,
      invariantViolations: bucket.invariantViolationCount,
      invariantViolationsById: bucket.invariantViolationsById,
      gameLength: stats,
    };
  });

  const overall = buckets.reduce(
    (acc, bucket) => {
      acc.totalGames += bucket.totalGames;
      acc.completedGames += bucket.completedGames;
      acc.maxTurnsGames += bucket.maxTurnsGames;
      acc.invariantViolations += bucket.invariantViolations;
      return acc;
    },
    {
      totalGames: 0,
      completedGames: 0,
      maxTurnsGames: 0,
      invariantViolations: 0,
      invariantViolationsById: {} as Record<string, number>,
    }
  );
  overall.invariantViolationsById = overallViolationsById;

  const summary = {
    timestamp: new Date().toISOString(),
    config,
    overall,
    buckets,
    violations: allViolations,
  };

  if (allViolations.length > 0) {
    // eslint-disable-next-line no-console
    console.log('');
    // eslint-disable-next-line no-console
    console.log('Invariant violations detected during orchestrator soak:');
    const preview = allViolations.slice(0, 5);
    for (const v of preview) {
      // eslint-disable-next-line no-console
      console.log(
        `- [${v.id}] board=${v.boardType} host=${v.hostMode} game=${v.gameId} ` +
          `turn=${v.turnIndex} S=${v.sInvariant} status=${v.gameStatus} :: ${v.message}`
      );
    }
    if (allViolations.length > preview.length) {
      // eslint-disable-next-line no-console
      console.log(
        `  (+ ${allViolations.length - preview.length} additional violation(s) in summary JSON)`
      );
    }
  }

  const outputPath = path.resolve(config.outputPath);
  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2), 'utf8');

  console.log('');
  console.log('Soak run complete.');
  console.log(
    `Total games=${overall.totalGames}, completed=${overall.completedGames}, maxTurns=${overall.maxTurnsGames}, invariantViolations=${overall.invariantViolations}`
  );
  console.log('');
  console.log(`Summary written to: ${outputPath}`);

  if (config.failOnViolation && overall.invariantViolations > 0) {
    // Signal failure to automated harnesses without throwing; we rely on
    // process.exitCode so any finally/cleanup logic above still runs.
    // eslint-disable-next-line no-console
    console.log('Failing with non-zero exit code due to invariant violations.');

    process.exitCode = 1;
  }
}

run().catch((err) => {
  console.error('Fatal error in orchestrator soak harness:', err);
  process.exitCode = 1;
});
