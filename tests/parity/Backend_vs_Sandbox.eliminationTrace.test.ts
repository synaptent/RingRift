/**
 * @semantic-anchor Backend_vs_Sandbox.eliminationTrace
 * @rules-level-counterparts
 *   - tests/unit/territoryProcessing.shared.test.ts (elimination bookkeeping)
 *   - tests/unit/victory.shared.test.ts (elimination thresholds)
 *   - RR-CANON-R022 (line elimination cost: 1 ring from any stack)
 *   - RR-CANON-R082 (territory elimination cost: entire cap from eligible stack)
 * @classification Trace-level parity / diagnostic
 */
import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace, createBackendEngineFromInitialState } from '../utils/traces';
import { findMatchingBackendMove } from '../utils/moveMatching';
import { CanonicalReplayEngine } from '../../src/shared/replay';

/**
 * Diagnostic test that traces totalRingsEliminated at every step to find
 * where backend and sandbox first diverge.
 */
describe('Backend vs Sandbox elimination count trace (square8 / 2p / seed=5)', () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 5;
  const MAX_STEPS = 70;

  function createReplayEngineFromInitial(initial: GameState): CanonicalReplayEngine {
    return new CanonicalReplayEngine({
      gameId: initial.id,
      boardType: initial.boardType,
      numPlayers: initial.players.length,
      initialState: initial,
    });
  }

  test('trace totalRingsEliminated step by step to find first divergence', async () => {
    const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
    expect(trace.entries.length).toBeGreaterThan(0);

    const moves: Move[] = trace.entries.map((e) => e.action as Move);

    const backendEngine = createBackendEngineFromInitialState(trace.initialState);
    const replayEngine = createReplayEngineFromInitial(trace.initialState);

    const results: Array<{
      index: number;
      moveType: string;
      movePlayer: number;
      backendEliminated: number;
      sandboxEliminated: number;
      backendPlayerSum: number;
      diff: number;
      backendCurrentPlayer: number;
      sandboxCurrentPlayer: number;
      backendPhase: string;
      sandboxPhase: string;
      backendStatus: string;
      sandboxStatus: string;
    }> = [];

    let firstDivergenceIndex = -1;

    for (let i = 0; i < moves.length; i++) {
      const move = moves[i];

      // Apply to backend
      const backendStateBefore = backendEngine.getGameState();
      const backendValidMoves = backendEngine.getValidMoves(backendStateBefore.currentPlayer);
      const matching = findMatchingBackendMove(move, backendValidMoves);

      if (!matching) {
        console.error(`[eliminationTrace] No matching backend move at index ${i}`, {
          sandboxMove: {
            type: move.type,
            player: move.player,
            from: move.from,
            to: move.to,
          },
          backendCurrentPlayer: backendStateBefore.currentPlayer,
          backendPhase: backendStateBefore.currentPhase,
        });
        break;
      }

      const { id, timestamp, moveNumber, ...payload } = matching as any;
      const backendResult = await backendEngine.makeMove(
        payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
      );

      if (!backendResult.success) {
        console.error(
          `[eliminationTrace] Backend makeMove failed at index ${i}`,
          backendResult.error
        );
        break;
      }

      // Apply to canonical replay engine
      const replayResult = await replayEngine.applyMove(move);
      if (!replayResult.success) {
        console.error(`[eliminationTrace] Replay applyMove failed at index ${i}`, {
          error: replayResult.error,
          moveNumber: move.moveNumber,
          type: move.type,
        });
        break;
      }

      // Compare states
      const backendState = backendEngine.getGameState();
      const sandboxState = replayEngine.getState();

      const diff = sandboxState.totalRingsEliminated - backendState.totalRingsEliminated;

      // Check if backend's totalRingsEliminated matches player sum
      const backendPlayerSum = backendState.players.reduce((sum, p) => sum + p.eliminatedRings, 0);
      const backendInternal = backendState.totalRingsEliminated !== backendPlayerSum;

      results.push({
        index: i,
        moveType: move.type,
        movePlayer: move.player,
        backendEliminated: backendState.totalRingsEliminated,
        sandboxEliminated: sandboxState.totalRingsEliminated,
        backendPlayerSum,
        diff,
        backendCurrentPlayer: backendState.currentPlayer,
        sandboxCurrentPlayer: sandboxState.currentPlayer,
        backendPhase: backendState.currentPhase,
        sandboxPhase: sandboxState.currentPhase,
        backendStatus: backendState.gameStatus,
        sandboxStatus: sandboxState.gameStatus,
      });

      // Report backend internal inconsistency
      if (backendInternal) {
        console.log(`[eliminationTrace] BACKEND INCONSISTENCY at index ${i}:`, {
          move: { type: move.type, player: move.player },
          totalRingsEliminated: backendState.totalRingsEliminated,
          playerSum: backendPlayerSum,
          mismatch: backendPlayerSum - backendState.totalRingsEliminated,
        });
      }

      if (diff !== 0 && firstDivergenceIndex === -1) {
        firstDivergenceIndex = i;
        console.log(`[eliminationTrace] FIRST DIVERGENCE at index ${i}:`, {
          move: {
            type: move.type,
            player: move.player,
            from: move.from,
            to: move.to,
            captureTarget: move.captureTarget,
          },
          backendEliminated: backendState.totalRingsEliminated,
          sandboxEliminated: sandboxState.totalRingsEliminated,
        });

        // Log player-level elimination counts
        console.log('[eliminationTrace] Player stats at divergence:', {
          backend: backendState.players.map((p) => ({
            player: p.playerNumber,
            eliminated: p.eliminatedRings,
            inHand: p.ringsInHand,
          })),
          sandbox: sandboxState.players.map((p) => ({
            player: p.playerNumber,
            eliminated: p.eliminatedRings,
            inHand: p.ringsInHand,
          })),
        });
      }

      // Stop if game completes
      if (backendState.gameStatus === 'completed' || sandboxState.gameStatus === 'completed') {
        console.log(`[eliminationTrace] Game completed at index ${i}:`, {
          backendStatus: backendState.gameStatus,
          sandboxStatus: sandboxState.gameStatus,
        });
        break;
      }
    }

    // Output summary
    console.log('[eliminationTrace] === SUMMARY ===');
    console.log(`Total moves processed: ${results.length}`);
    console.log(`First divergence index: ${firstDivergenceIndex}`);

    // Show last 10 results
    console.log('[eliminationTrace] Last 10 moves:');
    const last10 = results.slice(-10);
    for (const r of last10) {
      const marker = r.diff !== 0 ? '*** DIFF ***' : '';
      console.log(
        `  [${r.index}] ${r.moveType} by P${r.movePlayer}: ` +
          `eliminated=${r.backendEliminated}/${r.sandboxEliminated} ` +
          `player=${r.backendCurrentPlayer}/${r.sandboxCurrentPlayer} ` +
          `phase=${r.backendPhase}/${r.sandboxPhase} ${marker}`
      );
    }

    // Assert: for now, just report - we expect divergence
    if (firstDivergenceIndex !== -1) {
      console.log(`[eliminationTrace] Divergence found at index ${firstDivergenceIndex}`);
    } else {
      console.log('[eliminationTrace] No divergence found in elimination counts!');
    }

    expect(results.length).toBeGreaterThan(0);
  });
});
