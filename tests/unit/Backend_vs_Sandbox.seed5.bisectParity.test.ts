import { BoardType, Move } from '../../src/shared/types/game';
import { runBackendVsSandboxBisect } from '../utils/bisectParity';
import { diffSnapshots, ComparableSnapshot } from '../utils/stateSnapshots';

/**
 * Binary-search parity harness for the known seed-5 backend vs sandbox mismatch.
 *
 * This test:
 *   1) Generates a sandbox AI trace for square8 / 2p / seed=5.
 *   2) Uses the shared traceReplayer helpers to binary-search for the smallest
 *      prefix length at which backend and sandbox snapshots diverge, replaying
 *      canonical moves via the common EngineAdapter model.
 *   3) Logs a concise snapshot/marker diff at the earliest mismatching prefix
 *      so targeted micro-tests can be added around the offending move.
 *
 * It is intentionally diagnostic and remains skipped; when investigating
 * parity issues you can temporarily enable it locally.
 */
/**
 * TODO-BISECT-PARITY: This binary-search parity harness tests seed-5
 * backend vs sandbox divergence at prefix lengths. It requires complex
 * trace infrastructure and performs extensive state comparisons.
 * Skipped pending investigation of seed-5 parity issues.
 */
describe('Backend vs Sandbox snapshot parity bisect (square8 / 2p / seed=5)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 60;

  /**
   * Summarise marker and collapsed-space key differences between two
   * snapshots in a compact, key-focused form. This is used by the
   * seed-5 bisect harness so logs clearly identify which coordinates
   * diverge, rather than dumping entire marker/collapsed arrays.
   */
  function summariseMarkerAndCollapsedDiff(
    backend: ComparableSnapshot,
    sandbox: ComparableSnapshot
  ): {
    markerKeysOnlyInBackend: string[];
    markerKeysOnlyInSandbox: string[];
    collapsedKeysOnlyInBackend: string[];
    collapsedKeysOnlyInSandbox: string[];
  } {
    const backendMarkerKeys = new Set(backend.markers.map((m) => m.key));
    const sandboxMarkerKeys = new Set(sandbox.markers.map((m) => m.key));
    const backendCollapsedKeys = new Set(backend.collapsedSpaces.map((c) => c.key));
    const sandboxCollapsedKeys = new Set(sandbox.collapsedSpaces.map((c) => c.key));

    const markerKeysOnlyInBackend: string[] = [];
    const markerKeysOnlyInSandbox: string[] = [];
    const collapsedKeysOnlyInBackend: string[] = [];
    const collapsedKeysOnlyInSandbox: string[] = [];

    for (const key of backendMarkerKeys) {
      if (!sandboxMarkerKeys.has(key)) {
        markerKeysOnlyInBackend.push(key);
      }
    }
    for (const key of sandboxMarkerKeys) {
      if (!backendMarkerKeys.has(key)) {
        markerKeysOnlyInSandbox.push(key);
      }
    }

    for (const key of backendCollapsedKeys) {
      if (!sandboxCollapsedKeys.has(key)) {
        collapsedKeysOnlyInBackend.push(key);
      }
    }
    for (const key of sandboxCollapsedKeys) {
      if (!backendCollapsedKeys.has(key)) {
        collapsedKeysOnlyInSandbox.push(key);
      }
    }

    markerKeysOnlyInBackend.sort();
    markerKeysOnlyInSandbox.sort();
    collapsedKeysOnlyInBackend.sort();
    collapsedKeysOnlyInSandbox.sort();

    return {
      markerKeysOnlyInBackend,
      markerKeysOnlyInSandbox,
      collapsedKeysOnlyInBackend,
      collapsedKeysOnlyInSandbox,
    };
  }

  test('binary search finds earliest backend vs sandbox snapshot divergence for seed=5', async () => {
    const {
      moves,
      allEqual,
      firstMismatchIndex,
      backendSnapAtMismatch,
      sandboxSnapAtMismatch,
      backendSAtMismatch,
      sandboxSAtMismatch,
      backendHashAtMismatch,
      sandboxHashAtMismatch,
    } = await runBackendVsSandboxBisect({
      boardType,
      numPlayers,
      seed,
      maxSteps: MAX_STEPS,
    });

    expect(moves.length).toBeGreaterThan(0);

    if (!allEqual) {
      if (!backendSnapAtMismatch || !sandboxSnapAtMismatch) {
        throw new Error(
          'runBackendVsSandboxBisect reported a mismatch but did not return snapshots at the mismatching prefix'
        );
      }

      const diff = diffSnapshots(backendSnapAtMismatch, sandboxSnapAtMismatch);
      const markerSummary = summariseMarkerAndCollapsedDiff(
        backendSnapAtMismatch,
        sandboxSnapAtMismatch
      );

      // eslint-disable-next-line no-console
      console.error('[Backend_vs_Sandbox.seed5.bisectParity] Snapshot mismatch at prefix', {
        seed,
        maxSteps: MAX_STEPS,
        totalMoves: moves.length,
        firstMismatchIndex,
        allEqual,
        backendSAtMismatch,
        sandboxSAtMismatch,
        backendHashAtMismatch,
        sandboxHashAtMismatch,
        diff,
        markerSummary,
      });
    }

    // eslint-disable-next-line no-console
    console.log('[Backend_vs_Sandbox.seed5.bisectParity] result', {
      seed,
      maxSteps: MAX_STEPS,
      totalMoves: moves.length,
      firstMismatchIndex,
      allEqual,
      backendSAtMismatch,
      sandboxSAtMismatch,
      backendHashAtMismatch,
      sandboxHashAtMismatch,
    });

    // Expectations (zero tolerance enforced):
    // - Backend and sandbox must remain in full parity for the entire trace.
    // - We expect allEqual === true and firstMismatchIndex === moves.length.
    // - Any divergence at any point in the game is a failure.
    expect(allEqual).toBe(true);
    expect(firstMismatchIndex).toBe(moves.length);
  });
});
