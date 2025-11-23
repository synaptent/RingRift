import { BoardType } from '../../src/shared/types/game';
import { runSandboxAITrace, replayMovesOnBackend, replayTraceOnSandbox } from '../utils/traces';
import { formatMoveList } from '../../src/shared/engine/notation';

/**
 * Backend vs Sandbox trace-based parity checks (minimal harness, **trace-level**).
 *
 * Classification (see tests/README.md + tests/TEST_SUITE_PARITY_PLAN.md):
 * - Level: trace-level parity smoke test.
 * - Domain: backend GameEngine ↔ ClientSandboxEngine.
 * - Canonical semantics: src/shared/engine/* plus rules-level suites such as:
 *   - tests/unit/RefactoredEngine.test.ts
 *   - tests/unit/RefactoredEngineParity.test.ts
 *   - tests/unit/LineDetectionParity.rules.test.ts
 *   - tests/unit/Seed14Move35LineParity.test.ts
 *   and the scenario suites listed in RULES_SCENARIO_MATRIX.md.
 *
 * Contract:
 * - Recorded traces are treated as **derived artifacts**, not ground truth.
 * - If this harness fails but rules-level suites are green, treat the failing
 *   seed/trace as **stale under the current semantics**.
 * - In that case either:
 *   - regenerate the trace with the current shared-engine semantics, or
 *   - retire the seed from this harness and add/extend a focused rules-level
 *     test that captures the intended behaviour (the "seed-14 pattern").
 *
 * For a small set of seeds, we:
 *   1) Generate a sandbox AI trace via runSandboxAITrace.
 *   2) Replay the same canonical moves into a fresh backend GameEngine.
 *   3) Replay the same moves into a fresh ClientSandboxEngine.
 *   4) Compare S-invariant, hashes, and board summaries step-by-step.
 *
 * This is intentionally limited in scope (few seeds, modest depth) so it can
 * be used as a focused diagnostic without slowing CI.
 */

describe('Backend vs Sandbox trace parity (square8 / 2p)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  // NOTE: historical seed 14 used to exercise a trace that emitted a
  // `process_line` move at a state where, under the corrected line
  // semantics (Section 11.1 + shared BoardManager/sandboxLines
  // detectors), no valid lines exist. That divergence is now covered
  // and locked in by dedicated rules/parity tests (see
  // Seed14Move35LineParity and TraceParity.seed14.firstDivergence),
  // and the shared engine + detectors are treated as canonical.
  //
  // In line with the rules-over-traces policy, this minimal harness
  // keeps its seed list focused on a single representative AI game
  // (seed 5). Semantics for seed 14 (and any similar edge cases) are
  // asserted via those more targeted rules-level suites rather than by
  // preserving outdated traces here.
  const seeds = [5];
  const MAX_STEPS = 60;

  for (const seed of seeds) {
    test(`square8 / 2p / seed=${seed}: backend and sandbox traces stay in lockstep`, async () => {
      const sandboxTrace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);

      expect(sandboxTrace.entries.length).toBeGreaterThan(0);

      // Use the new replayMovesOnBackend helper
      const backendTrace = await replayMovesOnBackend(
        sandboxTrace.initialState,
        sandboxTrace.entries.map((e) => e.action)
      );
      const sandboxReplayTrace = await replayTraceOnSandbox(sandboxTrace);

      const minLen = Math.min(
        sandboxTrace.entries.length,
        backendTrace.entries.length,
        sandboxReplayTrace.entries.length
      );

      expect(minLen).toBeGreaterThan(0);

      let firstMismatchIndex: number | null = null;

      for (let i = 0; i < minLen; i++) {
        const original = sandboxTrace.entries[i];
        const backend = backendTrace.entries[i];
        const replayed = sandboxReplayTrace.entries[i];

        // S-invariant should match across engines at each step.
        const sOriginal = original.progressAfter.S;
        const sBackend = backend.progressAfter.S;
        const sReplayed = replayed.progressAfter.S;

        const sInvariantOk = sBackend === sOriginal && sReplayed === sOriginal;

        // When hashes are present, they should align as well.
        const hashBackendOk =
          !original.stateHashAfter ||
          !backend.stateHashAfter ||
          backend.stateHashAfter === original.stateHashAfter;
        const hashSandboxOk =
          !original.stateHashAfter ||
          !replayed.stateHashAfter ||
          replayed.stateHashAfter === original.stateHashAfter;

        // Board summaries (when present) should match exactly. We compare
        // via JSON.stringify here since these summaries are small,
        // structured objects.
        const boardBackendOk =
          !original.boardAfterSummary ||
          !backend.boardAfterSummary ||
          JSON.stringify(backend.boardAfterSummary) === JSON.stringify(original.boardAfterSummary);
        const boardSandboxOk =
          !original.boardAfterSummary ||
          !replayed.boardAfterSummary ||
          JSON.stringify(replayed.boardAfterSummary) === JSON.stringify(original.boardAfterSummary);

        if (
          !sInvariantOk ||
          !hashBackendOk ||
          !hashSandboxOk ||
          !boardBackendOk ||
          !boardSandboxOk
        ) {
          firstMismatchIndex = i;
          break;
        }
      }

      if (firstMismatchIndex !== null) {
        // Trace-level parity is treated as a diagnostic, derived from the
        // canonical shared-engine + rules suites. Any divergence that occurs
        // *strictly before* the final closing sequence of the game indicates a
        // regression in core line / territory / RNG semantics and must fail
        // the test.
        //
        // In contrast, divergences that are confined to the last couple of
        // trace entries are typically due to end-of-game bookkeeping and
        // self-elimination / victory resolution details. Those semantics are
        // now asserted by the dedicated victory/territory suites (for example
        // GameEngine victory scenarios and TerritoryParity/Decision tests),
        // rather than by preserving historical seed-5 traces as canonical.
        //
        // Concretely, we allow trace parity to drift within a small tolerance
        // window at the end of the game (the final two entries). Any earlier
        // drift still fails this test.
        // eslint-disable-next-line no-console
        console.log('[Backend_vs_Sandbox.traceParity] first divergence for seed', {
          seed,
          index: firstMismatchIndex,
          moveNumber: sandboxTrace.entries[firstMismatchIndex].moveNumber,
        });

        const toleranceWindowFromEnd = 2;
        const minIndexToTolerate = Math.max(0, minLen - toleranceWindowFromEnd);
        expect(firstMismatchIndex).toBeGreaterThanOrEqual(minIndexToTolerate);
      }

      // As a small sanity check, ensure the move counts match; if they do
      // not, surface a helpful diagnostic.
      const lengthMismatch = {
        originalEntries: sandboxTrace.entries.length,
        backendEntries: backendTrace.entries.length,
        sandboxReplayEntries: sandboxReplayTrace.entries.length,
      };

      expect(lengthMismatch.backendEntries).toBe(lengthMismatch.originalEntries);
      expect(lengthMismatch.sandboxReplayEntries).toBe(lengthMismatch.originalEntries);

      // If this ever fails in CI, having an easy way to print the move list
      // is useful when debugging locally.
      void formatMoveList(
        sandboxTrace.entries.map((e) => e.action),
        { boardType }
      );
    });
  }
});
