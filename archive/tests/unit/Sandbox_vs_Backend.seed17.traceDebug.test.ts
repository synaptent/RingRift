/**
 * Archived diagnostic suite: Sandbox vs Backend trace debug for seed 17.
 *
 * This file was moved from tests/unit/Sandbox_vs_Backend.seed17.traceDebug.test.ts
 * and is retained for historical/debugging reference only. It is not part of
 * the canonical rules or CI gating suites.
 */
import { BoardType, GameHistoryEntry } from '../../../src/shared/types/game';
import {
  runSandboxAITrace,
  replayTraceOnBackend,
  replayTraceOnSandbox,
} from '../../../tests/utils/traces';

const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Sandbox vs Backend trace debug: square8 / 2p / seed=17',
  () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const seed = 17;
    const MAX_STEPS = 80;

    test('finds first board-summary divergence between sandbox, backend replay, and sandbox replay', async () => {
      const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
      expect(trace.entries.length).toBeGreaterThan(0);

      const backendTrace = await replayTraceOnBackend(trace);
      const sandboxReplayTrace = await replayTraceOnSandbox(trace);

      const originalEntries = trace.entries;
      const backendEntries = backendTrace.entries;
      const sandboxReplayEntries = sandboxReplayTrace.entries;

      const minLength = Math.min(
        originalEntries.length,
        backendEntries.length,
        sandboxReplayEntries.length
      );

      let divergenceIndex = -1;

      for (let i = 0; i < minLength; i++) {
        const o = originalEntries[i];
        const b = backendEntries[i];
        const s = sandboxReplayEntries[i];

        const sameBoardAfter =
          JSON.stringify(o.boardAfterSummary) === JSON.stringify(b.boardAfterSummary) &&
          JSON.stringify(o.boardAfterSummary) === JSON.stringify(s.boardAfterSummary);

        const sameBoardBefore =
          JSON.stringify(o.boardBeforeSummary) === JSON.stringify(b.boardBeforeSummary) &&
          JSON.stringify(o.boardBeforeSummary) === JSON.stringify(s.boardBeforeSummary);

        if (!sameBoardBefore || !sameBoardAfter) {
          divergenceIndex = i;
          break;
        }
      }

      expect(divergenceIndex).toBeGreaterThanOrEqual(0);
    });
  }
);
