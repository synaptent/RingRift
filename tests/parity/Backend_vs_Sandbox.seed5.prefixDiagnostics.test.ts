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
 * Generic prefix diagnostics harness for seed=5 trace parity.
 *
 * This test:
 *   1) Generates the sandbox AI trace for square8 / 2p / seed=5.
 *   2) Rebuilds fresh backend + sandbox engines from the trace initial state.
 *   3) For each move index i, replays the canonical move into both engines:
 *        - Backend via getValidMoves + findMatchingBackendMove + makeMove.
 *        - Sandbox via applyCanonicalMove.
 *   4) At each step logs:
 *        - The canonical move and index.
 *        - Compact backend vs sandbox internal flags
 *          (pendingTerritorySelfElimination, pendingLineRewardElimination,
 *           hasPlacedThisTurn, mustMoveFromStackKey).
 *        - Snapshot equality and, for the first mismatch, a structured diff
 *          of board/players/markers/collapsed spaces.
 *
 * This is intentionally diagnostic: it does not currently fail on snapshot
 * mismatches, serving as a step-by-step replay log that narrows down the
 * exact index and shape of backend vs sandbox divergence.
 */
describe('Backend vs Sandbox prefix diagnostics (square8 / 2p / seed=5)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 60;

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

  function createReplayEngineFromInitial(initial: GameState): CanonicalReplayEngine {
    return new CanonicalReplayEngine({
      gameId: initial.id,
      boardType: initial.boardType,
      numPlayers: initial.players.length,
      initialState: initial,
    });
  }

  test('logs per-prefix backend vs sandbox state and first snapshot divergence for seed=5', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    const moves: Move[] = trace.entries.map((e) => e.action as Move);

    const backendEngine = createBackendEngineFromInitialState(trace.initialState);
    const replayEngine = createReplayEngineFromInitial(trace.initialState);

    let firstMismatchIndex = -1;

    for (let i = 0; i < moves.length; i++) {
      const move = moves[i];

      // Snapshot + internal flags BEFORE applying move i
      const backendStateBefore = backendEngine.getGameState();
      const sandboxStateBefore = replayEngine.getState();
      const backendAny: any = backendEngine;

      // eslint-disable-next-line no-console
      console.log('[Seed5 PrefixDiagnostics] before step', {
        index: i,
        moveNumber: move.moveNumber,
        type: move.type,
        player: move.player,
        backend: {
          currentPlayer: backendStateBefore.currentPlayer,
          currentPhase: backendStateBefore.currentPhase,
          gameStatus: backendStateBefore.gameStatus,
          totalRingsEliminated: backendStateBefore.totalRingsEliminated,
          pendingTerritorySelfElimination: backendAny.pendingTerritorySelfElimination === true,
          pendingLineRewardElimination: backendAny.pendingLineRewardElimination === true,
          hasPlacedThisTurn: backendAny.hasPlacedThisTurn === true,
          mustMoveFromStackKey: backendAny.mustMoveFromStackKey,
        },
        sandbox: {
          currentPlayer: sandboxStateBefore.currentPlayer,
          currentPhase: sandboxStateBefore.currentPhase,
          gameStatus: sandboxStateBefore.gameStatus,
          totalRingsEliminated: sandboxStateBefore.totalRingsEliminated,
        },
      });

      // Backend: map sandbox move to canonical backend move
      const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
      const matching = findMatchingBackendMove(move, backendValidMoves);

      if (!matching) {
        console.error('[Seed5 PrefixDiagnostics] no matching backend move', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          backendCurrentPlayer: backendStateBefore.currentPlayer,
          backendCurrentPhase: backendStateBefore.currentPhase,
          backendValidMovesCount: backendValidMoves.length,
        });
        firstMismatchIndex = i;
        break;
      }

      const { id, timestamp, moveNumber, ...payload } = matching as any;
      const backendResult = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!backendResult.success) {
        console.error('[Seed5 PrefixDiagnostics] backend makeMove failed', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          backendMoveNumber: (matching as any).moveNumber,
          error: backendResult.error,
        });
        firstMismatchIndex = i;
        break;
      }

      // Canonical replay engine: apply canonical move directly.
      const replayResult = await replayEngine.applyMove(move);
      if (!replayResult.success) {
        console.error('[Seed5 PrefixDiagnostics] replay applyMove failed', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          error: replayResult.error,
        });
        firstMismatchIndex = i;
        break;
      }

      // AFTER applying move i, compare snapshots.
      const backendAfter = backendEngine.getGameState();
      const sandboxAfter = replayEngine.getState();

      const backendSnap = snapshotFromGameState(`backend-step-${i}`, backendAfter);
      const sandboxSnap = snapshotFromGameState(`sandbox-step-${i}`, sandboxAfter);

      if (!snapshotsEqual(backendSnap, sandboxSnap)) {
        firstMismatchIndex = i;
        const diff = diffSnapshots(backendSnap, sandboxSnap);
        const markerSummary = summariseMarkerAndCollapsedDiff(backendSnap, sandboxSnap);

        console.error('[Seed5 PrefixDiagnostics] snapshot mismatch after step', {
          index: i,
          moveNumber: move.moveNumber,
          type: move.type,
          player: move.player,
          diff,
          markerSummary,
        });
        break;
      }
    }

    // eslint-disable-next-line no-console
    console.log('[Seed5 PrefixDiagnostics] result', {
      seed,
      totalMoves: moves.length,
      firstMismatchIndex,
    });

    // Diagnostic harness: ensure we at least exercised the trace; do not
    // fail the test on snapshot mismatches here.
    expect(moves.length).toBeGreaterThan(0);
  });
});
