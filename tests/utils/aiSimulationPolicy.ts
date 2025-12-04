import { BoardType, GameState } from '../../src/shared/types/game';
import { computeProgressSnapshot, hashGameState } from '../../src/shared/engine/core';

/**
 * Canonical per-step snapshot used when classifying AI simulation runs.
 *
 * Each snapshot captures:
 * - before/after GameState
 * - S-invariant progress snapshots for both states
 * - canonical state hashes
 *
 * Callers should construct these via createSimulationStepSnapshot to ensure
 * consistent hashing and S semantics across all harnesses.
 */
export type ProgressSnapshot = ReturnType<typeof computeProgressSnapshot>;

export interface SimulationStepSnapshot {
  before: GameState;
  after: GameState;
  beforeProgress: ProgressSnapshot;
  afterProgress: ProgressSnapshot;
  beforeHash: string;
  afterHash: string;
}

export interface StallWindow {
  /** Zero-based index of the first action in the stall window within the run. */
  startAction: number;
  /** Number of consecutive stagnant actions in this window. */
  length: number;
}

/**
 * Per-seed classification for an AI-vs-AI simulation run under the shared
 * S-invariant + stall policy.
 *
 * Semantics:
 * - sViolations: any per-step decrease in S (after.S < before.S).
 * - stallWindows: any window of STALL_WINDOW_STEPS or more consecutive actions
 *   where gameStatus remains 'active' and hashGameState(before) === hashGameState(after).
 * - nonTerminating: the run exceeded MAX_AI_ACTIONS_PER_GAME while the final
 *   state remained active and at least one state change occurred during the run.
 */
export interface SeedRunClassification {
  boardType: BoardType;
  numPlayers: number;
  seed: number;
  /** Total number of AI actions taken in this run. */
  totalActions: number;
  /** Per-step S-invariant violations (after.S < before.S). */
  sViolations: SimulationStepSnapshot[];
  /** One or more stagnant windows detected during the run. */
  stallWindows: StallWindow[];
  /** True when the run exhausted the action budget while still active. */
  nonTerminating: boolean;
}

/**
 * Canonical stall window (in AI actions) used by sandbox and backend harnesses.
 *
 * A stall is N consecutive AI actions where:
 *   - gameStatus === 'active'
 *   - hashGameState(before) === hashGameState(after)
 */
export const STALL_WINDOW_STEPS = 8;

/**
 * Canonical per-game AI action budget used by simulation harnesses.
 *
 * Runs that reach or exceed this many AI actions while the game remains
 * active are considered non-terminating (subject to some state changes
 * having occurred during the run).
 */
export const MAX_AI_ACTIONS_PER_GAME = 400;

/**
 * Canonical classification labels for a seed run.
 *
 * - 'ok':           no S violations, no stall windows, and terminating within budget.
 * - 's_violation':  at least one per-step S decrease.
 * - 'stall':        at least one stall window of STALL_WINDOW_STEPS or more.
 * - 'non_terminating': exhausted the action budget while still active.
 */
export type SeedRunStatus = 'ok' | 's_violation' | 'stall' | 'non_terminating';

/**
 * Build a SimulationStepSnapshot from raw before/after states using the shared
 * computeProgressSnapshot and hashGameState helpers.
 *
 * This keeps all step-level instrumentation aligned across harnesses.
 */
export function createSimulationStepSnapshot(
  before: GameState,
  after: GameState
): SimulationStepSnapshot {
  const beforeProgress = computeProgressSnapshot(before);
  const afterProgress = computeProgressSnapshot(after);
  const beforeHash = hashGameState(before);
  const afterHash = hashGameState(after);

  return {
    before,
    after,
    beforeProgress,
    afterProgress,
    beforeHash,
    afterHash,
  };
}

/**
 * Classify a single seeded AI simulation run according to the shared S-invariant
 * and stall policy.
 *
 * Callers are expected to:
 *   - Advance the game via AI actions up to either:
 *       - a terminal state, or
 *       - MAX_AI_ACTIONS_PER_GAME actions (or a stricter budget of their choice),
 *   - Collect SimulationStepSnapshot entries built via createSimulationStepSnapshot
 *     for each applied AI action,
 *   - Pass the final GameState and run metadata into this function.
 */
export function classifySeedRun(
  snapshots: SimulationStepSnapshot[],
  finalState: GameState,
  boardType: BoardType,
  numPlayers: number,
  seed: number,
  opts?: {
    /**
     * Optional override for the action budget used to flag non-terminating runs.
     * Defaults to MAX_AI_ACTIONS_PER_GAME when omitted.
     */
    maxActionsPerGame?: number;
    /**
     * Optional override for stall window length (in actions). Defaults to
     * STALL_WINDOW_STEPS when omitted. Primarily intended for experimental
     * diagnostics; test semantics should normally use the canonical window.
     */
    stallWindowSteps?: number;
  }
): SeedRunClassification {
  const maxActionsPerGame = opts?.maxActionsPerGame ?? MAX_AI_ACTIONS_PER_GAME;
  const stallWindowSteps = opts?.stallWindowSteps ?? STALL_WINDOW_STEPS;

  const sViolations: SimulationStepSnapshot[] = [];
  const stallWindows: StallWindow[] = [];

  let stagnantCount = 0;
  let currentStallStart: number | null = null;

  snapshots.forEach((step, index) => {
    // 1. Enforce per-step non-decreasing S and collect violations.
    if (step.afterProgress.S < step.beforeProgress.S) {
      sViolations.push(step);
    }

    // 2. Track consecutive stagnant steps for stall detection.
    const isStagnant = step.beforeHash === step.afterHash && step.after.gameStatus === 'active';

    if (isStagnant) {
      if (stagnantCount === 0) {
        currentStallStart = index;
      }
      stagnantCount += 1;

      if (stagnantCount === stallWindowSteps) {
        // First time we reach the window size: start a new stall window record.
        stallWindows.push({
          startAction: currentStallStart ?? index - stallWindowSteps + 1,
          length: stallWindowSteps,
        });
      } else if (stagnantCount > stallWindowSteps) {
        // Extend the last window in-place to cover the longer streak.
        const last = stallWindows[stallWindows.length - 1];
        if (last && last.startAction === (currentStallStart ?? index - stagnantCount + 1)) {
          last.length = stagnantCount;
        }
      }
    } else {
      // Streak broken; reset counters.
      stagnantCount = 0;
      currentStallStart = null;
    }
  });

  const totalActions = snapshots.length;

  // 3. Non-termination: we exceeded the action budget while the game remained
  //    active and at least one state change occurred during the run. Pure
  //    frozen stalls are captured separately via stallWindows.
  const hadAnyStateChange = snapshots.some((s) => s.beforeHash !== s.afterHash);
  const nonTerminating =
    finalState.gameStatus === 'active' && totalActions >= maxActionsPerGame && hadAnyStateChange;

  return {
    boardType,
    numPlayers,
    seed,
    totalActions,
    sViolations,
    stallWindows,
    nonTerminating,
  };
}

/**
 * Derive a coarse classification label from a SeedRunClassification.
 *
 * Priority order:
 *   1. s_violation (any S decrease is treated as the most severe issue),
 *   2. stall (one or more stall windows),
 *   3. non_terminating (budget exhausted while still active),
 *   4. ok (no issues found).
 */
export function getSeedRunStatus(classification: SeedRunClassification): SeedRunStatus {
  if (classification.sViolations.length > 0) {
    return 's_violation';
  }
  if (classification.stallWindows.length > 0) {
    return 'stall';
  }
  if (classification.nonTerminating) {
    return 'non_terminating';
  }
  return 'ok';
}
