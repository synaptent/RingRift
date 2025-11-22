import { BoardType, GameHistoryEntry } from '../../src/shared/types/game';
import { runSandboxAITrace, replayTraceOnBackend } from '../utils/traces';

/**
 * Minimal per-step geometry parity test for the known heuristic-coverage
 * scenario: square8 / 2 players / seed=17.
 *
 * This focuses **only** on board summaries (stacks/markers/collapsedSpaces)
 * for the original sandbox trace vs a backend replay of the same canonical
 * moves. It locates the earliest index where geometry diverges and fails
 * with a compact diff describing which components differ.
 */
describe('Seed17 geometry parity: square8 / 2p / backend vs sandbox trace', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 17;
  const MAX_STEPS = 80;

  function firstArrayDiff(a: string[] = [], b: string[] = []) {
    const maxLen = Math.max(a.length, b.length);
    for (let i = 0; i < maxLen; i++) {
      const av = a[i];
      const bv = b[i];
      if (av !== bv) {
        return { index: i, sandbox: av, backend: bv };
      }
    }
    return null as {
      index: number;
      sandbox: string | undefined;
      backend: string | undefined;
    } | null;
  }

  test('per-step geometry parity against original sandbox trace', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    const backendTrace = await replayTraceOnBackend(trace);

    const originalEntries: GameHistoryEntry[] = trace.entries;
    const backendEntries: GameHistoryEntry[] = backendTrace.entries;

    const minLength = Math.min(originalEntries.length, backendEntries.length);

    let firstMismatchIndex = -1;
    let mismatchDetail: any = null;

    for (let i = 0; i < minLength; i++) {
      const o = originalEntries[i];
      const b = backendEntries[i];

      const oBefore = o.boardBeforeSummary;
      const bBefore = b.boardBeforeSummary;
      const oAfter = o.boardAfterSummary;
      const bAfter = b.boardAfterSummary;

      const stacksBeforeDiff = firstArrayDiff(oBefore?.stacks ?? [], bBefore?.stacks ?? []);
      const markersBeforeDiff = firstArrayDiff(oBefore?.markers ?? [], bBefore?.markers ?? []);
      const collapsedBeforeDiff = firstArrayDiff(
        oBefore?.collapsedSpaces ?? [],
        bBefore?.collapsedSpaces ?? []
      );

      const stacksAfterDiff = firstArrayDiff(oAfter?.stacks ?? [], bAfter?.stacks ?? []);
      const markersAfterDiff = firstArrayDiff(oAfter?.markers ?? [], bAfter?.markers ?? []);
      const collapsedAfterDiff = firstArrayDiff(
        oAfter?.collapsedSpaces ?? [],
        bAfter?.collapsedSpaces ?? []
      );

      const anyDiff =
        stacksBeforeDiff ||
        markersBeforeDiff ||
        collapsedBeforeDiff ||
        stacksAfterDiff ||
        markersAfterDiff ||
        collapsedAfterDiff;

      if (anyDiff) {
        firstMismatchIndex = i;

        mismatchDetail = {
          index: i,
          sandboxMoveNumber: o.moveNumber,
          backendMoveNumber: b.moveNumber,
          differingComponents: {
            stacksBefore: stacksBeforeDiff,
            markersBefore: markersBeforeDiff,
            collapsedBefore: collapsedBeforeDiff,
            stacksAfter: stacksAfterDiff,
            markersAfter: markersAfterDiff,
            collapsedAfter: collapsedAfterDiff,
          },
        };

        break;
      }
    }

    if (firstMismatchIndex !== -1) {
      throw new Error(
        '[seed17-geometry] First backend vs sandbox geometry mismatch found at index ' +
          firstMismatchIndex +
          ' (see differingComponents for first mismatching stack/marker/collapsed key)\n' +
          JSON.stringify(mismatchDetail, null, 2)
      );
    }
  });
});
