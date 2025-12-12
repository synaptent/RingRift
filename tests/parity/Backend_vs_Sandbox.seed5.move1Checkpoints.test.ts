import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import { CanonicalReplayEngine } from '../../src/shared/replay';
import {
  snapshotFromGameState,
  snapshotsEqual,
  diffSnapshots,
  ComparableSnapshot,
} from '../utils/stateSnapshots';

/**
 * Checkpoint-based parity harness for the first divergent move in the
 * seed-5 trace (move index 0).
 *
 * This test:
 *   1) Generates the sandbox AI trace for square8 / 2p / seed=5.
 *   2) Rebuilds fresh backend + sandbox engines from the trace initial state.
 *   3) Attaches debug checkpoint hooks to both engines.
 *   4) Applies only the first move in the trace to each engine:
 *        - Backend via getValidMoves + makeMove
 *        - Sandbox via applyCanonicalMove
 *   5) Compares backend vs sandbox checkpoints label-by-label using
 *      snapshotFromGameState and diffSnapshots.
 *
 * When parity is fully restored, all shared checkpoint labels should have
 * identical snapshots and this test will pass. Until then, it will fail
 * and log structured diffs to guide debugging.
 */
describe('Backend vs Sandbox checkpoint parity for first move (square8 / 2p / seed=5)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 60;

  function createReplayEngineFromInitial(initial: GameState): CanonicalReplayEngine {
    return new CanonicalReplayEngine({
      gameId: initial.id,
      boardType: initial.boardType,
      numPlayers: initial.players.length,
      initialState: initial,
    });
  }

  test('backend and sandbox checkpoints for move 1 remain in parity', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    const moves: Move[] = trace.entries.map((e) => e.action as Move);
    const targetMove = moves[0];
    expect(targetMove).toBeDefined();

    // --- Backend: set up engine + checkpoint hook and apply first move ---
    const backendEngine = createBackendEngineFromInitialState(trace.initialState);
    const backendCheckpoints: Array<{ label: string; snapshot: ComparableSnapshot }> = [];

    backendEngine.setDebugCheckpointHook((label, state) => {
      backendCheckpoints.push({
        label,
        snapshot: snapshotFromGameState(`backend-${label}`, state),
      });
    });

    {
      const backendStateBefore = backendEngine.getGameState();
      const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
      const matching = findMatchingBackendMove(targetMove, backendValidMoves);

      if (!matching) {
        console.error('[Seed5 Move1 Checkpoints] No matching backend move for sandbox move 1', {
          sandboxMove: {
            moveNumber: targetMove.moveNumber,
            type: targetMove.type,
            player: targetMove.player,
            from: targetMove.from,
            to: targetMove.to,
            captureTarget: targetMove.captureTarget,
          },
          backendCurrentPlayer: backendStateBefore.currentPlayer,
          backendCurrentPhase: backendStateBefore.currentPhase,
          backendValidMovesCount: backendValidMoves.length,
        });
        throw new Error(
          `No matching backend move for first sandbox move moveNumber=${targetMove.moveNumber}`
        );
      }

      const { id, timestamp, moveNumber, ...payload } = matching as any;
      const backendResult = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!backendResult.success) {
        throw new Error(
          `Backend makeMove failed for first move (backend moveNumber=${(matching as any).moveNumber}): ${backendResult.error}`
        );
      }
    }

    // --- Sandbox: set up engine + checkpoint hook and apply first move ---
    const sandboxEngine = createReplayEngineFromInitial(trace.initialState);
    const sandboxCheckpoints: Array<{ label: string; snapshot: ComparableSnapshot }> = [];

    sandboxEngine.setDebugCheckpointHook((label, state) => {
      sandboxCheckpoints.push({
        label,
        snapshot: snapshotFromGameState(`sandbox-${label}`, state),
      });
    });

    const sandboxResult = await sandboxEngine.applyMove(targetMove);
    if (!sandboxResult.success) {
      throw new Error(
        `Replay applyMove failed for first move (moveNumber=${targetMove.moveNumber}): ${sandboxResult.error}`
      );
    }

    // --- Compare checkpoints by label ---
    const backendLabels = backendCheckpoints.map((c) => c.label);
    const sandboxLabels = sandboxCheckpoints.map((c) => c.label);

    const sharedLabels = Array.from(
      new Set(backendLabels.filter((l) => sandboxLabels.includes(l)))
    ).sort();

    // eslint-disable-next-line no-console
    console.log('[Seed5 Move1 Checkpoints] labels', {
      backendLabels,
      sandboxLabels,
      sharedLabels,
    });

    let allEqual = true;

    for (const label of sharedLabels) {
      const backendEntry = backendCheckpoints.find((c) => c.label === label)!;
      const sandboxEntry = sandboxCheckpoints.find((c) => c.label === label)!;

      if (!snapshotsEqual(backendEntry.snapshot, sandboxEntry.snapshot)) {
        allEqual = false;

        console.error('[Seed5 Move1 Checkpoints] snapshot mismatch at checkpoint', {
          label,
          diff: diffSnapshots(backendEntry.snapshot, sandboxEntry.snapshot),
        });
      }
    }

    if (!allEqual) {
      throw new Error(
        'Backend vs Sandbox checkpoint mismatch for seed=5, move 1. ' +
          'See logged diffs for per-checkpoint discrepancies.'
      );
    }

    expect(allEqual).toBe(true);
  });
});
