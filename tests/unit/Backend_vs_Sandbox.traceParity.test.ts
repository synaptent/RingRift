import { BoardType } from '../../src/shared/types/game';
import { runSandboxAITrace, replayMovesOnBackend, replayTraceOnSandbox } from '../utils/traces';
import { formatMoveList } from '../../src/shared/engine/notation';

/**
 * Backend vs Sandbox trace-based parity checks (minimal harness).
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
  const seeds = [5, 14];
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

      for (let i = 0; i < minLen; i++) {
        const original = sandboxTrace.entries[i];
        const backend = backendTrace.entries[i];
        const replayed = sandboxReplayTrace.entries[i];

        // S-invariant should match across engines at each step.
        expect(backend.progressAfter.S).toBe(original.progressAfter.S);
        expect(replayed.progressAfter.S).toBe(original.progressAfter.S);

        // When hashes are present, they should align as well.
        if (original.stateHashAfter && backend.stateHashAfter) {
          expect(backend.stateHashAfter).toBe(original.stateHashAfter);
        }
        if (original.stateHashAfter && replayed.stateHashAfter) {
          expect(replayed.stateHashAfter).toBe(original.stateHashAfter);
        }

        // Board summaries (when present) should match exactly.
        if (original.boardAfterSummary && backend.boardAfterSummary) {
          expect(backend.boardAfterSummary).toEqual(original.boardAfterSummary);
        }
        if (original.boardAfterSummary && replayed.boardAfterSummary) {
          expect(replayed.boardAfterSummary).toEqual(original.boardAfterSummary);
        }
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
