import { BoardType } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { hashGameState, summarizeBoard } from '../../src/shared/engine/core';
import { Move } from '../../src/shared/types/game';
import { findMatchingBackendMove } from '../utils/moveMatching';

/**
 * Helper/debug test: locate the FIRST move index in the seed-14 trace where
 * backend and sandbox hashes diverge when we replay the sandbox canonical
 * moves into a fresh backend GameEngine using the same matching logic as
 * replayMovesOnBackend.
 *
 * This mirrors the seed-5 helper but targets the known seed-14 parity
 * failures (e.g. around move 31 where a placement appears legal in the
 * sandbox but not on the backend).
 */

/**
 * TODO-SEED14-DIVERGENCE: This diagnostic test locates the first move
 * index where backend and sandbox hash diverge for seed 14. It is a
 * helper for investigating known seed-14 parity failures around move 31.
 * Skipped as it is a diagnostic helper, not a regression test.
 */
describe.skip('Trace parity first-divergence helper: square8 / 2p / seed=14', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 14;
  const MAX_STEPS = 60;

  test('log first backend vs sandbox hash/phase divergence for seed 14', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    const engine = createBackendEngineFromInitialState(trace.initialState);

    let firstMismatchIndex = -1;

    for (let i = 0; i < trace.entries.length; i++) {
      const entry = trace.entries[i];
      const move = entry.action as Move;

      // Always advance the backend through automatic bookkeeping phases so we
      // only compare interactive phases (placement/movement/capture).
      engine.stepAutomaticPhasesForTesting();

      const backendBefore = engine.getGameState();
      const backendMoves = engine.getValidMoves(backendBefore.currentPlayer);
      const matching = findMatchingBackendMove(move, backendMoves as Move[]);

      if (!matching) {
        firstMismatchIndex = i;
        console.log('FIRST MOVE MATCH FAILURE at index', i, 'moveNumber', move.moveNumber);
        console.log('Sandbox Move:', JSON.stringify(move, null, 2));
        console.log(
          'Backend State Summary (before move):',
          JSON.stringify(summarizeBoard(backendBefore.board), null, 2)
        );
        console.log('Backend State Hash (before move):', hashGameState(backendBefore));
        console.log('Backend currentPlayer/currentPhase BEFORE move:', {
          currentPlayer: backendBefore.currentPlayer,
          currentPhase: backendBefore.currentPhase,
          gameStatus: backendBefore.gameStatus,
        });
        console.log('Sandbox phase/status BEFORE move (from trace entry):', {
          actor: entry.actor,
          phaseBefore: entry.phaseBefore,
          statusBefore: entry.statusBefore,
        });
        break;
      }

      const { id, timestamp, moveNumber, ...payload } = matching as Move;
      const result = await engine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );
      if (!result.success) {
        firstMismatchIndex = i;
        console.log(
          'BACKEND makeMove failure at index',
          i,
          'moveNumber',
          move.moveNumber,
          'error:',
          result.error
        );
        break;
      }

      const backendAfter = engine.getGameState();
      const backendHashAfter = hashGameState(backendAfter);
      const sandboxHashAfter = entry.stateHashAfter;

      // --- Phase / player orchestration diagnostics ---
      const backendPhaseAfter = backendAfter.currentPhase;
      const backendPlayerAfter = backendAfter.currentPlayer;
      const sandboxPhaseAfter = entry.phaseAfter;
      const sandboxActor = entry.actor;

      if (sandboxPhaseAfter && backendPhaseAfter !== sandboxPhaseAfter) {
        firstMismatchIndex = i;
        console.log('FIRST PHASE DIVERGENCE at index', i, 'moveNumber', move.moveNumber);
        console.log('Sandbox phases/status (from trace entry):', {
          actor: sandboxActor,
          phaseBefore: entry.phaseBefore,
          phaseAfter: entry.phaseAfter,
          statusBefore: entry.statusBefore,
          statusAfter: entry.statusAfter,
        });
        console.log('Backend phases/status AFTER move (from live state):', {
          currentPlayer: backendPlayerAfter,
          currentPhase: backendPhaseAfter,
          gameStatus: backendAfter.gameStatus,
        });

        const backendHistoryLast = backendAfter.history[backendAfter.history.length - 1];
        console.log(
          'Backend last history entry (if any):',
          backendHistoryLast && {
            actor: backendHistoryLast.actor,
            phaseBefore: backendHistoryLast.phaseBefore,
            phaseAfter: backendHistoryLast.phaseAfter,
            statusBefore: backendHistoryLast.statusBefore,
            statusAfter: backendHistoryLast.statusAfter,
          }
        );

        console.log(
          'Sandbox State Summary AFTER move:',
          JSON.stringify(entry.boardAfterSummary, null, 2)
        );
        console.log(
          'Backend State Summary AFTER move:',
          JSON.stringify(summarizeBoard(backendAfter.board), null, 2)
        );
        break;
      }

      // --- Hash-based divergence diagnostics ---
      if (sandboxHashAfter && backendHashAfter && backendHashAfter !== sandboxHashAfter) {
        firstMismatchIndex = i;
        console.log('FIRST HASH DIVERGENCE at index', i, 'moveNumber', move.moveNumber);
        console.log(
          'Sandbox State Summary AFTER move:',
          JSON.stringify(entry.boardAfterSummary, null, 2)
        );
        console.log('Sandbox State Hash (after move):', sandboxHashAfter);
        console.log(
          'Backend State Summary AFTER move:',
          JSON.stringify(summarizeBoard(backendAfter.board), null, 2)
        );
        console.log('Backend State Hash (after move):', backendHashAfter);
        console.log('Backend currentPlayer/currentPhase AFTER move:', {
          currentPlayer: backendPlayerAfter,
          currentPhase: backendPhaseAfter,
          gameStatus: backendAfter.gameStatus,
        });
        console.log('Sandbox phase/status AFTER move (from trace entry):', {
          actor: sandboxActor,
          phaseAfter: entry.phaseAfter,
          statusAfter: entry.statusAfter,
        });
        break;
      }
    }

    if (firstMismatchIndex === -1) {
      console.log('No hash/phase divergence found for seed 14 up to maxSteps', MAX_STEPS);
    } else {
      throw new Error(
        `Backend vs Sandbox trace parity divergence for seed 14 at index ${firstMismatchIndex}. ` +
          'See earlier console diagnostics for phase/hash mismatch details.'
      );
    }

    // Sanity check: we still expect a non-empty trace for this scenario.
    expect(trace.entries.length).toBeGreaterThan(0);
  });
});
