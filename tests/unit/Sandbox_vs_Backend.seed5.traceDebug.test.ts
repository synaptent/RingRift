import { BoardType, GameHistoryEntry } from '../../src/shared/types/game';
import { runSandboxAITrace, replayTraceOnBackend, replayTraceOnSandbox } from '../utils/traces';

/**
 * Focused trace-based debug harness for the known S-drop scenario:
 *   square8 / 2 players / seed=5
 *
 * Workflow:
 *   1. Generate a sandbox AI trace under the deterministic PRNG.
 *   2. Scan for the first entry where S_after < S_before.
 *   3. Replay the same canonical moves onto a fresh backend GameEngine
 *      and a fresh ClientSandboxEngine.
 *   4. If an S-drop is found, fail the test with rich diagnostic
 *      context showing per-engine progress snapshots and board
 *      summaries at the offending step.
 *
 * Once the underlying bug is fixed, this test should pass simply by
 * finding no S-drop in the sandbox trace (dropIndex === -1).
 */

describe('Sandbox vs Backend trace debug: square8 / 2p / seed=5', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 60; // Enough to cover the known-problematic opening sequence.

  test('S-invariant should never decrease within a sandbox AI trace', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);

    expect(trace.entries.length).toBeGreaterThan(0);

    // Locate the first step where S_after < S_before.
    let dropIndex = -1;
    for (let i = 0; i < trace.entries.length; i++) {
      const entry = trace.entries[i];
      if (entry.progressAfter.S < entry.progressBefore.S) {
        dropIndex = i;
        break;
      }
    }

    // If no S-drop is found, the invariant holds for this scenario and the
    // test passes. This is the desired long-term behaviour once the bug is
    // fixed.
    if (dropIndex === -1) {
      return;
    }

    // Otherwise, we are in a failing/broken state. Replay the same canonical
    // move list onto backend and a fresh sandbox to gather comparative
    // diagnostics.
    const backendTrace = await replayTraceOnBackend(trace);
    const sandboxReplayTrace = await replayTraceOnSandbox(trace);

    // Ensure traces are comparable in length; if not, surface that first.
    const lengthMismatch = {
      originalEntries: trace.entries.length,
      backendEntries: backendTrace.entries.length,
      sandboxReplayEntries: sandboxReplayTrace.entries.length
    };

    const originalEntry: GameHistoryEntry | undefined = trace.entries[dropIndex];
    const backendEntry: GameHistoryEntry | undefined = backendTrace.entries[dropIndex];
    const sandboxReplayEntry: GameHistoryEntry | undefined = sandboxReplayTrace.entries[dropIndex];

    const context = {
      seed,
      maxSteps: MAX_STEPS,
      dropIndex,
      lengthMismatch,
      original: originalEntry && {
        moveNumber: originalEntry.moveNumber,
        actor: originalEntry.actor,
        phaseBefore: originalEntry.phaseBefore,
        phaseAfter: originalEntry.phaseAfter,
        statusBefore: originalEntry.statusBefore,
        statusAfter: originalEntry.statusAfter,
        progressBefore: originalEntry.progressBefore,
        progressAfter: originalEntry.progressAfter,
        stateHashBefore: originalEntry.stateHashBefore,
        stateHashAfter: originalEntry.stateHashAfter,
        boardAfterSummary: originalEntry.boardAfterSummary
      },
      backend: backendEntry && {
        moveNumber: backendEntry.moveNumber,
        actor: backendEntry.actor,
        phaseBefore: backendEntry.phaseBefore,
        phaseAfter: backendEntry.phaseAfter,
        statusBefore: backendEntry.statusBefore,
        statusAfter: backendEntry.statusAfter,
        progressBefore: backendEntry.progressBefore,
        progressAfter: backendEntry.progressAfter,
        stateHashBefore: backendEntry.stateHashBefore,
        stateHashAfter: backendEntry.stateHashAfter,
        boardAfterSummary: backendEntry.boardAfterSummary
      },
      sandboxReplay: sandboxReplayEntry && {
        moveNumber: sandboxReplayEntry.moveNumber,
        actor: sandboxReplayEntry.actor,
        phaseBefore: sandboxReplayEntry.phaseBefore,
        phaseAfter: sandboxReplayEntry.phaseAfter,
        statusBefore: sandboxReplayEntry.statusBefore,
        statusAfter: sandboxReplayEntry.statusAfter,
        progressBefore: sandboxReplayEntry.progressBefore,
        progressAfter: sandboxReplayEntry.progressAfter,
        stateHashBefore: sandboxReplayEntry.stateHashBefore,
        stateHashAfter: sandboxReplayEntry.stateHashAfter,
        boardAfterSummary: sandboxReplayEntry.boardAfterSummary
      }
    };

    // Additional check: does the S-drop reproduce when we replay the trace
    // into a fresh sandbox using applyCanonicalMove? This helps ensure the
    // bug is deterministic (not RNG-dependent) and localized to the
    // reducers/state transitions rather than to AI choice randomness.
    let replayDropIndex = -1;
    for (let i = 0; i < sandboxReplayTrace.entries.length; i++) {
      const entry = sandboxReplayTrace.entries[i];
      if (entry.progressAfter.S < entry.progressBefore.S) {
        replayDropIndex = i;
        break;
      }
    }

    (context as any).sandboxReplayDropIndex = replayDropIndex;

    // Fail with a rich diagnostic payload so that the underlying bug can be
    // localised to the appropriate sandbox reducer.
    throw new Error(
      'S-invariant violation detected in sandbox AI trace for square8 / 2p / seed=5. ' +
        'See attached context for per-engine snapshots at the offending step:\n' +
        JSON.stringify(context, null, 2)
    );
  });
});
