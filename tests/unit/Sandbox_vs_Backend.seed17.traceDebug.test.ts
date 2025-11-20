import { BoardType, GameHistoryEntry } from '../../src/shared/types/game';
import { runSandboxAITrace, replayTraceOnBackend, replayTraceOnSandbox } from '../utils/traces';

/**
 * Focused trace-based debug harness for the known heuristic-coverage scenario:
 *   square8 / 2 players / seed=17
 *
 * Workflow:
 *   1. Generate a sandbox AI trace under the deterministic PRNG.
 *   2. Replay the same canonical moves onto a fresh backend GameEngine and a
 *      fresh ClientSandboxEngine.
 *   3. Find the first index where board summaries (and/or state hashes)
 *      diverge between:
 *        - the original sandbox trace,
 *        - the backend replay trace,
 *        - the sandbox replay trace.
 *   4. Fail with a rich diagnostic payload describing per-engine snapshots at
 *      the offending step so that rules can be checked directly against the
 *      rules document.
 */
describe('Sandbox vs Backend trace debug: square8 / 2p / seed=17', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 17;
  const MAX_STEPS = 80; // Enough to cover the opening region where divergence is observed.

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

      const sameHashes =
        o.stateHashBefore === b.stateHashBefore &&
        o.stateHashAfter === b.stateHashAfter &&
        o.stateHashBefore === s.stateHashBefore &&
        o.stateHashAfter === s.stateHashAfter;

      const sameBoardAfter =
        JSON.stringify(o.boardAfterSummary) === JSON.stringify(b.boardAfterSummary) &&
        JSON.stringify(o.boardAfterSummary) === JSON.stringify(s.boardAfterSummary);

      const sameBoardBefore =
        JSON.stringify(o.boardBeforeSummary) === JSON.stringify(b.boardBeforeSummary) &&
        JSON.stringify(o.boardBeforeSummary) === JSON.stringify(s.boardBeforeSummary);

      if (!sameHashes || !sameBoardBefore || !sameBoardAfter) {
        divergenceIndex = i;
        break;
      }
    }

    if (divergenceIndex === -1) {
      // No divergence; backend and sandbox replay stayed in lockstep for this trace.
      return;
    }

    const originalEntriesLength = originalEntries.length;
    const backendEntriesLength = backendEntries.length;
    const sandboxReplayEntriesLength = sandboxReplayEntries.length;

    const lengthMismatch = {
      originalEntries: originalEntriesLength,
      backendEntries: backendEntriesLength,
      sandboxReplayEntries: sandboxReplayEntriesLength
    };

    const originalEntry: GameHistoryEntry | undefined = originalEntries[divergenceIndex];
    const backendEntry: GameHistoryEntry | undefined = backendEntries[divergenceIndex];
    const sandboxReplayEntry: GameHistoryEntry | undefined = sandboxReplayEntries[divergenceIndex];

        // Additional diagnostics focused on the opening move and the board state
        // immediately after it. For the known seed-17 divergence, divergenceIndex
        // is 1 (move 2), so entry[0] corresponds to "move 1" and
        // originalEntry.boardBeforeSummary reflects the board after move 1.
        const firstOriginalEntry: GameHistoryEntry | undefined = originalEntries[0];
        const firstBackendEntry: GameHistoryEntry | undefined = backendEntries[0];
        const firstSandboxReplayEntry: GameHistoryEntry | undefined = sandboxReplayEntries[0];
    
        const markerKey = '1,7';
        const hasMarkerAt = (markers: string[] | undefined): boolean => {
          if (!markers) return false;
          return markers.some((m) => m.startsWith(`${markerKey}:`));
        };
    
        const firstOriginalAfter = firstOriginalEntry?.boardAfterSummary;
        const firstBackendAfter = firstBackendEntry?.boardAfterSummary;
        const firstSandboxReplayAfter = firstSandboxReplayEntry?.boardAfterSummary;
        const originalBefore = originalEntry?.boardBeforeSummary;
    
        const firstMoveDebug =
          firstOriginalEntry &&
          firstOriginalAfter && {
            moveNumber: firstOriginalEntry.moveNumber,
            action: firstOriginalEntry.action,
            phaseBefore: firstOriginalEntry.phaseBefore,
            phaseAfter: firstOriginalEntry.phaseAfter,
            statusBefore: firstOriginalEntry.statusBefore,
            statusAfter: firstOriginalEntry.statusAfter,
            sandboxTraceBoardAfter: {
              stacks: firstOriginalAfter.stacks,
              markers: firstOriginalAfter.markers,
              collapsedSpaces: firstOriginalAfter.collapsedSpaces,
              hasMarkerAt1_7: hasMarkerAt(firstOriginalAfter.markers),
            },
            backendReplayBoardAfter:
              firstBackendEntry &&
              firstBackendAfter && {
                stacks: firstBackendAfter.stacks,
                markers: firstBackendAfter.markers,
                collapsedSpaces: firstBackendAfter.collapsedSpaces,
                hasMarkerAt1_7: hasMarkerAt(firstBackendAfter.markers),
              },
            sandboxReplayBoardAfter:
              firstSandboxReplayEntry &&
              firstSandboxReplayAfter && {
                stacks: firstSandboxReplayAfter.stacks,
                markers: firstSandboxReplayAfter.markers,
                collapsedSpaces: firstSandboxReplayAfter.collapsedSpaces,
                hasMarkerAt1_7: hasMarkerAt(firstSandboxReplayAfter.markers),
              },
            // Board state immediately before the divergent move index; when
            // divergenceIndex === 1 this corresponds to "after move 1" for each
            // engine, which is where the unexpected marker at 1,7 appears in the
            // sandbox traces but not in the backend.
            sandboxTraceBeforeDivergentMove:
              originalBefore && {
                stacks: originalBefore.stacks,
                markers: originalBefore.markers,
                collapsedSpaces: originalBefore.collapsedSpaces,
                hasMarkerAt1_7: hasMarkerAt(originalBefore.markers),
              },
          };

    const context: any = {
      seed,
      maxSteps: MAX_STEPS,
      divergenceIndex,
      lengthMismatch,
      firstMoveDebug,
      original: originalEntry && {
        moveNumber: originalEntry.moveNumber,
        actor: originalEntry.actor,
        action: originalEntry.action,
        phaseBefore: originalEntry.phaseBefore,
        phaseAfter: originalEntry.phaseAfter,
        statusBefore: originalEntry.statusBefore,
        statusAfter: originalEntry.statusAfter,
        progressBefore: originalEntry.progressBefore,
        progressAfter: originalEntry.progressAfter,
        stateHashBefore: originalEntry.stateHashBefore,
        stateHashAfter: originalEntry.stateHashAfter,
        boardBeforeSummary: originalEntry.boardBeforeSummary,
        boardAfterSummary: originalEntry.boardAfterSummary,
      },
      backend: backendEntry && {
        moveNumber: backendEntry.moveNumber,
        actor: backendEntry.actor,
        action: backendEntry.action,
        phaseBefore: backendEntry.phaseBefore,
        phaseAfter: backendEntry.phaseAfter,
        statusBefore: backendEntry.statusBefore,
        statusAfter: backendEntry.statusAfter,
        progressBefore: backendEntry.progressBefore,
        progressAfter: backendEntry.progressAfter,
        stateHashBefore: backendEntry.stateHashBefore,
        stateHashAfter: backendEntry.stateHashAfter,
        boardBeforeSummary: backendEntry.boardBeforeSummary,
        boardAfterSummary: backendEntry.boardAfterSummary,
      },
      sandboxReplay: sandboxReplayEntry && {
        moveNumber: sandboxReplayEntry.moveNumber,
        actor: sandboxReplayEntry.actor,
        action: sandboxReplayEntry.action,
        phaseBefore: sandboxReplayEntry.phaseBefore,
        phaseAfter: sandboxReplayEntry.phaseAfter,
        statusBefore: sandboxReplayEntry.statusBefore,
        statusAfter: sandboxReplayEntry.statusAfter,
        progressBefore: sandboxReplayEntry.progressBefore,
        progressAfter: sandboxReplayEntry.progressAfter,
        stateHashBefore: sandboxReplayEntry.stateHashBefore,
        stateHashAfter: sandboxReplayEntry.stateHashAfter,
        boardBeforeSummary: sandboxReplayEntry.boardBeforeSummary,
        boardAfterSummary: sandboxReplayEntry.boardAfterSummary,
      },
    };

    throw new Error(
      'Board divergence detected in sandbox AI trace for square8 / 2p / seed=17. ' +
        'Inspect the context for per-engine snapshots at the first differing step:\n' +
        JSON.stringify(context, null, 2)
    );
  });
});