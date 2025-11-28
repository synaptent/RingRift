import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Focused debug harness for a specific backend AI-style simulation seed.
 *
 * This mirrors tests/unit/GameEngine.aiSimulation.debug.test.ts but targets
 * the scenario that fails in the fuzzy AI simulation harness:
 *   - boardType = 'square8'
 *   - numPlayers = 2
 *   - seed = 10 (corresponding to run=9 for square8/2p in the fuzzy test)
 *
 * It logs rich diagnostic information when we reach an "active game with no
 * legal moves" state or when the S-invariant (progress metric) appears to be
 * violated, so we can inspect the exact board/turn state.
 *
 * NOTE: Skipped when ORCHESTRATOR_ADAPTER_ENABLED=true because AI simulation diverges (intentional)
 */

// Skip this test suite when orchestrator adapter is enabled - AI simulation diverges
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'GameEngine AI simulation debug: square8 with 2 AI players, seed=10',
  () => {
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const MAX_MOVES_PER_GAME = 500; // smaller cap for debug

    function createEngineWithPlayers(bt: BoardType, playersCount: number): GameEngine {
      const boardConfig = BOARD_CONFIGS[bt];

      const players: Player[] = Array.from({ length: playersCount }, (_, idx) => {
        const playerNumber = idx + 1;
        return {
          id: `p${playerNumber}`,
          username: `Player${playerNumber}`,
          type: 'ai',
          playerNumber,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: boardConfig.ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        } as Player;
      });

      const engine = new GameEngine('backend-ai-sim-debug-seed10', bt, players, timeControl, false);
      const started = engine.startGame();
      if (!started) {
        throw new Error('Failed to start GameEngine for AI simulation debug (seed 10)');
      }
      return engine;
    }

    /**
     * Progress / termination invariant helper based on the rules-level S metric:
     *   S = M + C + E
     * where
     *   - M = number of markers on the board,
     *   - C = number of collapsed spaces (territory),
     *   - E = total eliminated rings over all players.
     */
    function computeProgressMetric(state: GameState): {
      markers: number;
      collapsed: number;
      eliminated: number;
      S: number;
    } {
      const markers = state.board.markers.size;
      const collapsed = state.board.collapsedSpaces.size;

      const eliminatedFromBoard = Object.values(state.board.eliminatedRings ?? {}).reduce(
        (sum, value) => sum + value,
        0
      );
      const eliminated =
        (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ??
        eliminatedFromBoard;

      const S = markers + collapsed + eliminated;
      return { markers, collapsed, eliminated, S };
    }

    function makePrng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function chooseRandomMove(
      engine: GameEngine,
      state: GameState,
      rng: () => number
    ): Move | null {
      const currentPlayer = state.currentPlayer;
      const moves = engine.getValidMoves(currentPlayer);

      if (!moves.length) {
        return null;
      }

      const idx = Math.floor(rng() * moves.length);
      return moves[Math.min(idx, moves.length - 1)];
    }

    test('logs detailed state for square8 / 2 AI players / seed=10 when assumptions fail', async () => {
      const seed = 10;
      const rng = makePrng(seed);

      const engine = createEngineWithPlayers(boardType, numPlayers);

      let lastProgress = computeProgressMetric(engine.getGameState());

      for (let i = 0; i < MAX_MOVES_PER_GAME; i++) {
        const before = engine.getGameState();
        const beforeProgress = computeProgressMetric(before);

        // S must be globally non-decreasing.
        expect(beforeProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
        lastProgress = beforeProgress;

        if (before.gameStatus !== 'active') {
          // Game ended naturally.
          // eslint-disable-next-line no-console
          console.log('[DEBUG seed10] Game ended naturally', {
            step: i,
            seed,
            gameStatus: before.gameStatus,
            winner: before.winner,
            currentPlayer: before.currentPlayer,
            currentPhase: before.currentPhase,
            progress: beforeProgress,
          });
          return;
        }

        if (
          before.currentPhase === 'line_processing' ||
          before.currentPhase === 'territory_processing'
        ) {
          const autoBefore = before;
          const autoBeforeProgress = beforeProgress;

          engine.stepAutomaticPhasesForTesting();
          const afterAuto = engine.getGameState();
          const afterAutoProgress = computeProgressMetric(afterAuto);

          // eslint-disable-next-line no-console
          console.log('[DEBUG seed10] stepAutomaticPhasesForTesting', {
            step: i,
            phase: autoBefore.currentPhase,
            beforeProgress: autoBeforeProgress,
            afterProgress: afterAutoProgress,
          });

          expect(afterAutoProgress.S).toBeGreaterThanOrEqual(autoBeforeProgress.S);
          lastProgress = afterAutoProgress;
          continue;
        }

        const moves = engine.getValidMoves(before.currentPlayer);

        if (!moves.length) {
          // eslint-disable-next-line no-console
          console.log('[DEBUG seed10] No legal moves for active game (pre-resolve)', {
            step: i,
            seed,
            boardType,
            numPlayers,
            currentPlayer: before.currentPlayer,
            currentPhase: before.currentPhase,
            gameStatus: before.gameStatus,
            progress: beforeProgress,
            players: before.players.map((p) => ({
              playerNumber: p.playerNumber,
              ringsInHand: p.ringsInHand,
              eliminatedRings: p.eliminatedRings,
              territorySpaces: p.territorySpaces,
            })),
            stacks: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
              key,
              controllingPlayer: stack.controllingPlayer,
              stackHeight: stack.stackHeight,
              capHeight: stack.capHeight,
            })),
            markers: Array.from(before.board.markers.entries()).map(([key, owner]) => ({
              key,
              owner,
            })),
            collapsed: Array.from(before.board.collapsedSpaces.entries()).map(([key, owner]) => ({
              key,
              owner,
            })),
          });

          engine.resolveBlockedStateForCurrentPlayerForTesting();
          const afterResolve = engine.getGameState();
          const afterResolveProgress = computeProgressMetric(afterResolve);

          // eslint-disable-next-line no-console
          console.log('[DEBUG seed10] After resolveBlockedStateForCurrentPlayerForTesting', {
            step: i,
            seed,
            boardType,
            numPlayers,
            currentPlayer: afterResolve.currentPlayer,
            currentPhase: afterResolve.currentPhase,
            gameStatus: afterResolve.gameStatus,
            progress: afterResolveProgress,
          });

          expect(afterResolveProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
          lastProgress = afterResolveProgress;

          if (afterResolve.gameStatus !== 'active') {
            return;
          }

          const movesAfter = engine.getValidMoves(afterResolve.currentPlayer);
          if (!movesAfter.length) {
            // eslint-disable-next-line no-console
            console.log('[DEBUG seed10] Still no legal moves after resolver', {
              step: i,
              seed,
              boardType,
              numPlayers,
              currentPlayer: afterResolve.currentPlayer,
              currentPhase: afterResolve.currentPhase,
              gameStatus: afterResolve.gameStatus,
              progress: afterResolveProgress,
            });

            expect(movesAfter.length).toBeGreaterThan(0);
            return;
          }

          continue;
        }

        const move = chooseRandomMove(engine, before, rng)!;
        const { id, timestamp, moveNumber, ...payload } = move;

        const result = await engine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        expect(result.success).toBe(true);

        const after = engine.getGameState();
        const afterProgress = computeProgressMetric(after);

        // Log per-move evolution for early steps to understand how S evolves.
        if (i <= 12) {
          // eslint-disable-next-line no-console
          console.log('[DEBUG seed10] Move applied', {
            step: i,
            seed,
            phase: before.currentPhase,
            movePlayer: move.player,
            moveType: move.type,
            from: move.from ? positionToString(move.from) : undefined,
            to: move.to ? positionToString(move.to) : undefined,
            beforeProgress,
            afterProgress,
          });
        }

        // For movement / capture phases, S should strictly increase.
        if (before.currentPhase === 'movement' || before.currentPhase === 'capture') {
          if (!(afterProgress.S > beforeProgress.S)) {
            // eslint-disable-next-line no-console
            console.log('[DEBUG seed10] S-invariant violation (debug harness)', {
              step: i,
              seed,
              phase: before.currentPhase,
              currentPlayer: before.currentPlayer,
              moveType: move.type,
              movePlayer: move.player,
              moveFrom: move.from ? positionToString(move.from) : undefined,
              moveTo: move.to ? positionToString(move.to) : undefined,
              beforeProgress,
              afterProgress,
              players: before.players.map((p) => ({
                playerNumber: p.playerNumber,
                ringsInHand: p.ringsInHand,
                eliminatedRings: p.eliminatedRings,
                territorySpaces: p.territorySpaces,
              })),
              stacksBefore: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
                key,
                controllingPlayer: stack.controllingPlayer,
                stackHeight: stack.stackHeight,
                capHeight: stack.capHeight,
              })),
              stacksAfter: Array.from(after.board.stacks.entries()).map(([key, stack]) => ({
                key,
                controllingPlayer: stack.controllingPlayer,
                stackHeight: stack.stackHeight,
                capHeight: stack.capHeight,
              })),
              markersBefore: Array.from(before.board.markers.entries()).map(([key, marker]) => ({
                key,
                player: marker.player,
              })),
              markersAfter: Array.from(after.board.markers.entries()).map(([key, marker]) => ({
                key,
                player: marker.player,
              })),
              collapsedBefore: Array.from(before.board.collapsedSpaces.entries()).map(
                ([key, owner]) => ({ key, owner })
              ),
              collapsedAfter: Array.from(after.board.collapsedSpaces.entries()).map(
                ([key, owner]) => ({ key, owner })
              ),
            });

            expect(afterProgress.S).toBeGreaterThan(beforeProgress.S);
            return;
          }
        }

        lastProgress = afterProgress;
      }

      const finalState = engine.getGameState();
      const finalProgress = computeProgressMetric(finalState);
      // eslint-disable-next-line no-console
      console.log('[DEBUG seed10] Reached MAX_MOVES_PER_GAME without termination', {
        seed,
        boardType,
        numPlayers,
        finalGameStatus: finalState.gameStatus,
        finalCurrentPlayer: finalState.currentPlayer,
        finalPhase: finalState.currentPhase,
        finalProgress,
      });

      expect(finalState.gameStatus).not.toBe('active');
    });
  }
);
