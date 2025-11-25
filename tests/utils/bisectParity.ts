import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace } from './traces';
import {
  backendAdapter,
  sandboxAdapter,
  findFirstMismatchIndex,
  compareEnginesAtPrefix,
  replayPrefixOnEngine,
} from './traceReplayer';
import { snapshotFromGameState, ComparableSnapshot } from './stateSnapshots';
import { computeProgressSnapshot, hashGameState } from '../../src/shared/engine/core';

/**
 * Shared helper for backend vs sandbox snapshot-parity bisect harnesses.
 *
 * Responsibilities:
 *   1) Generate a sandbox AI trace for a given boardType / numPlayers / seed.
 *   2) Binary-search (via traceReplayer) for the smallest prefix length at
 *      which backend and sandbox snapshots diverge when replaying the
 *      canonical move list from the common initial state.
 *   3) When a mismatch is found, compute backend vs sandbox snapshots at the
 *      earliest mismatching prefix so callers can log focused diagnostics.
 *
 * This keeps the core bisect mechanics in one place so specialised tests
 * (seed-specific diagnostics, board-type variations, etc.) can focus on
 * expectations and logging rather than duplicating trace + replay logic.
 */

export interface BisectConfig {
  boardType: BoardType;
  numPlayers: number;
  seed: number;
  maxSteps: number;
}

export interface BisectOutcome {
  /** Canonical move list extracted from the sandbox trace. */
  moves: Move[];
  /** Whether backend and sandbox snapshots were equal for the full move list. */
  allEqual: boolean;
  /** Index of first mismatching prefix in [0, moves.length], or moves.length when none. */
  firstMismatchIndex: number;
  /**
   * Backend snapshot at the earliest mismatching prefix, when a mismatch is
   * found. Undefined when allEqual === true.
   */
  backendSnapAtMismatch?: ComparableSnapshot;
  /**
   * Sandbox snapshot at the earliest mismatching prefix, when a mismatch is
   * found. Undefined when allEqual === true.
   */
  sandboxSnapAtMismatch?: ComparableSnapshot;
  /** S-invariant at the earliest mismatching prefix (backend). */
  backendSAtMismatch?: number;
  /** S-invariant at the earliest mismatching prefix (sandbox). */
  sandboxSAtMismatch?: number;
  /** State hash at the earliest mismatching prefix (backend). */
  backendHashAtMismatch?: string;
  /** State hash at the earliest mismatching prefix (sandbox). */
  sandboxHashAtMismatch?: string;
  /** Initial GameState from which both engines are constructed. */
  initialState: GameState;
}

export async function runBackendVsSandboxBisect(config: BisectConfig): Promise<BisectOutcome> {
  const { boardType, numPlayers, seed, maxSteps } = config;

  const trace = await runSandboxAITrace(boardType, numPlayers, seed, maxSteps);
  const moves: Move[] = trace.entries.map((e) => e.action as Move);

  const { allEqual, firstMismatchIndex } = await findFirstMismatchIndex(
    backendAdapter,
    sandboxAdapter,
    trace.initialState,
    moves
  );

  let backendSnapAtMismatch: ComparableSnapshot | undefined;
  let sandboxSnapAtMismatch: ComparableSnapshot | undefined;
  let backendSAtMismatch: number | undefined;
  let sandboxSAtMismatch: number | undefined;
  let backendHashAtMismatch: string | undefined;
  let sandboxHashAtMismatch: string | undefined;

  if (!allEqual) {
    const prefix = firstMismatchIndex;

    let backendStateAtMismatch: GameState;
    let sandboxStateAtMismatch: GameState;

    // For prefix 0 we simply compare the initial state; for k > 0, reuse the
    // shared compareEnginesAtPrefix helper and also replay prefixes to obtain
    // concrete GameState instances for S/hash diagnostics.
    if (prefix === 0) {
      const snap = snapshotFromGameState('initial-state', trace.initialState);
      backendSnapAtMismatch = snap;
      sandboxSnapAtMismatch = snap;
      backendStateAtMismatch = trace.initialState;
      sandboxStateAtMismatch = trace.initialState;
    } else {
      const { backendSnap, sandboxSnap } = await compareEnginesAtPrefix(
        backendAdapter,
        sandboxAdapter,
        trace.initialState,
        moves,
        prefix
      );
      backendSnapAtMismatch = backendSnap;
      sandboxSnapAtMismatch = sandboxSnap;

      const backendReplay = await replayPrefixOnEngine(
        backendAdapter,
        trace.initialState,
        moves,
        prefix
      );
      const sandboxReplay = await replayPrefixOnEngine(
        sandboxAdapter,
        trace.initialState,
        moves,
        prefix
      );
      backendStateAtMismatch = backendReplay.finalState;
      sandboxStateAtMismatch = sandboxReplay.finalState;
    }

    const backendProgress = computeProgressSnapshot(backendStateAtMismatch);
    const sandboxProgress = computeProgressSnapshot(sandboxStateAtMismatch);
    backendSAtMismatch = backendProgress.S;
    sandboxSAtMismatch = sandboxProgress.S;
    backendHashAtMismatch = hashGameState(backendStateAtMismatch);
    sandboxHashAtMismatch = hashGameState(sandboxStateAtMismatch);
  }

  return {
    moves,
    allEqual,
    firstMismatchIndex,
    backendSnapAtMismatch,
    sandboxSnapAtMismatch,
    backendSAtMismatch,
    sandboxSAtMismatch,
    backendHashAtMismatch,
    sandboxHashAtMismatch,
    initialState: trace.initialState,
  };
}
