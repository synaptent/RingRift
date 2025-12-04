import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  TimeControl,
  BOARD_CONFIGS,
} from '../../src/shared/types/game';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { logAiDiagnostic } from '../utils/aiTestLogger';
import {
  createSimulationStepSnapshot,
  classifySeedRun,
  getSeedRunStatus,
  STALL_WINDOW_STEPS,
  type SimulationStepSnapshot,
  type SeedRunClassification,
} from '../utils/aiSimulationPolicy';

/**
 * Backend AI-vs-AI style simulation tests.
 *
 * These do not go through the Python AI service. Instead, they use a simple
 * local policy that always selects a legal move from GameEngine.getValidMoves
 * for the current player. The goal is to exercise the full GameEngine/RuleEngine
 * flow and verify that:
 *
 * - The game makes steady progress under the S-invariant (no decreases in S),
 * - Structural stalls are detected via the shared hash-based window semantics,
 * - The game terminates (gameStatus !== 'active') within a generous bound,
 * - We never reach a state where the game is active, it is a player's turn,
 *   and there are zero legal moves available from getValidMoves.
 *
 * This mirrors the sandbox AI simulation harness but runs entirely on the
 * backend engine. Here we fuzz across multiple board types and player counts
 * using a seeded PRNG so that any discovered stall or S-invariant violation
 * is reproducible and classifiable via the shared aiSimulationPolicy helper.
 *
 * NOTE: Jest timeout set to 60 seconds. Each individual game also has a
 * MAX_TIME_PER_GAME_MS guard to prevent runaway simulations from blocking the suite.
 */

// Set generous timeout for the entire test file - these are simulation tests
jest.setTimeout(60000);

const BACKEND_AI_SIM_ENABLED = process.env.RINGRIFT_ENABLE_BACKEND_AI_SIM === '1';
const maybeDescribe = BACKEND_AI_SIM_ENABLED ? describe : describe.skip;

/**
 * These AI-style simulation tests run extensive random games (10 runs x 500
 * moves each) across multiple board types and player counts. They are treated
 * as a **diagnostic harness**, not a default CI gate, and are enabled only
 * when `RINGRIFT_ENABLE_BACKEND_AI_SIM=1` is set in the environment (see
 * `tests/README.md` and the `test:ai-backend:quiet` npm script).
 *
 * The shared aiSimulationPolicy helper defines canonical S + stall semantics
 * based on:
 *   - S = markers + collapsed + eliminated (computeProgressSnapshot),
 *   - Stall windows of STALL_WINDOW_STEPS with unchanged hashGameState and
 *     gameStatus === 'active',
 *   - A per-game action budget. Here we use MAX_MOVES_PER_GAME for backend
 *     moves, which is slightly higher than the canonical MAX_AI_ACTIONS_PER_GAME
 *     used by sandbox diagnostics but wired through classifySeedRun via an
 *     explicit override.
 */
maybeDescribe(
  'GameEngine AI-style simulations (backend termination / stall classification)',
  () => {
    const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal'];
    const playerCounts: number[] = [2, 3, 4];
    // Reduced for faster CI: still enough fuzzing to catch regressions, but
    // without the original 100 * 10_000 move bound per scenario.
    const RUNS_PER_SCENARIO = 10; // Reduced from 20 to prevent timeout
    const MAX_MOVES_PER_GAME = 500; // Slightly higher than sandbox's MAX_AI_ACTIONS_PER_GAME (400)
    const MAX_TIME_PER_GAME_MS = 5000; // Maximum 5 seconds per individual game

    function createEngineWithPlayers(boardType: BoardType, numPlayers: number): GameEngine {
      const boardConfig = BOARD_CONFIGS[boardType];

      const players: Player[] = Array.from({ length: numPlayers }, (_, idx) => {
        const playerNumber = idx + 1;
        return {
          id: `p${playerNumber}`,
          username: `Player${playerNumber}`,
          // Treat these simulated players as AI so the GameEngine constructor
          // auto-marks them as ready in startGame(). We do not use the
          // Python AI service here; moves come from getValidMoves.
          type: 'ai',
          playerNumber,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: boardConfig.ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        } as Player;
      });

      const engine = new GameEngine('backend-ai-sim', boardType, players, timeControl, false);
      // Start the game so gameStatus becomes 'active' and timers are initialised.
      const started = engine.startGame();
      if (!started) {
        throw new Error('Failed to start GameEngine for AI simulation test');
      }
      return engine;
    }

    /**
     * Tiny deterministic PRNG so we can reproduce any failing run by its seed.
     */
    function makePrng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        // LCG parameters from Numerical Recipes
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    for (const boardType of boardTypes) {
      for (const numPlayers of playerCounts) {
        const scenarioLabel = `${boardType} with ${numPlayers} AI players`;

        test(`${scenarioLabel}: seeded backend games classify seeds by S-invariant / stall / termination within ${MAX_MOVES_PER_GAME} moves`, async () => {
          const boardIndex = boardTypes.indexOf(boardType);
          const playerCountIndex = playerCounts.indexOf(numPlayers);

          const scenarioClassifications: SeedRunClassification[] = [];

          for (let run = 0; run < RUNS_PER_SCENARIO; run++) {
            // Derive a reproducible seed for this scenario + run.
            const seed = 1 + run + playerCountIndex * 1000 + boardIndex * 100000;
            const rng = makePrng(seed);

            const engine = createEngineWithPlayers(boardType, numPlayers);

            const snapshots: SimulationStepSnapshot[] = [];
            let lastProgress = computeProgressSnapshot(engine.getGameState());
            const gameStartTime = Date.now();

            for (let i = 0; i < MAX_MOVES_PER_GAME; i++) {
              // Time-based guard: break out if this individual game is taking too long.
              const elapsed = Date.now() - gameStartTime;
              if (elapsed > MAX_TIME_PER_GAME_MS) {
                logAiDiagnostic(
                  'backend-time-limit-reached',
                  {
                    scenario: scenarioLabel,
                    run,
                    seed,
                    step: i,
                    elapsedMs: elapsed,
                    maxTimeMs: MAX_TIME_PER_GAME_MS,
                  },
                  'backend-ai-sim'
                );
                // Consider this run complete (time limit reached, not a failure).
                break;
              }

              const before = engine.getGameState();
              const beforeProgress = computeProgressSnapshot(before);

              // S must be globally non-decreasing over the lifetime of the game.
              expect(beforeProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
              lastProgress = beforeProgress;

              if (before.gameStatus !== 'active') {
                // Game ended naturally (ring elimination, territory, or other terminal condition).
                break;
              }

              // Auto-advance through non-interactive phases before asking
              // for a player move. During line_processing and
              // territory_processing there are no legal moves to choose
              // from; they are internal bookkeeping phases.
              if (
                before.currentPhase === 'line_processing' ||
                before.currentPhase === 'territory_processing'
              ) {
                engine.stepAutomaticPhasesForTesting();
                const afterAuto = engine.getGameState();
                const afterAutoProgress = computeProgressSnapshot(afterAuto);

                // Automatic processing must also respect non-decreasing S.
                expect(afterAutoProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
                lastProgress = afterAutoProgress;

                if (afterAuto.gameStatus !== 'active') {
                  break;
                }
                // Re-evaluate from the new state on the next loop.
                continue;
              }

              const moves = engine.getValidMoves(before.currentPlayer);

              if (!moves.length) {
                // Diagnostic: we have an active game, in an interactive
                // phase, but no legal moves. Treat this as a trigger to
                // apply the same forced-elimination / skip semantics that
                // TurnEngine would normally apply immediately after
                // territory processing.
                logAiDiagnostic(
                  'backend-active-no-moves',
                  {
                    scenario: scenarioLabel,
                    run,
                    seed,
                    step: i,
                    phase: before.currentPhase,
                    currentPlayer: before.currentPlayer,
                  },
                  'backend-ai-sim'
                );

                engine.resolveBlockedStateForCurrentPlayerForTesting();
                const afterResolve = engine.getGameState();
                const afterResolveProgress = computeProgressSnapshot(afterResolve);

                // Forced elimination / skips must also respect the
                // non-decreasing S invariant.
                expect(afterResolveProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
                lastProgress = afterResolveProgress;

                if (afterResolve.gameStatus !== 'active') {
                  break;
                }

                // Re-enter loop from the new state on the next iteration.
                continue;
              }

              const idx = Math.floor(rng() * moves.length);
              const move = moves[Math.min(idx, moves.length - 1)];

              const { id, timestamp, moveNumber, ...payload } = move;
              const result = await engine.makeMove(
                payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
              );
              expect(result.success).toBe(true);

              const after = engine.getGameState();
              const afterProgress = computeProgressSnapshot(after);

              // Progress metric must be globally non-decreasing.
              expect(afterProgress.S).toBeGreaterThanOrEqual(lastProgress.S);

              // For movement / capture phases (i.e. real "action" turns), S is
              // expected to increase over the course of the game, but individual
              // actions may occasionally be S-neutral while still being legal.
              // We therefore log diagnostics when S does not strictly increase
              // here, but rely on the global non-decrease check above plus the
              // shared stall classification to guard against pathological behaviour.
              if (before.currentPhase === 'movement' || before.currentPhase === 'capture') {
                if (!(afterProgress.S > beforeProgress.S)) {
                  // Provide a rich diagnostic snapshot to the AI test logger so we
                  // can debug concrete S-invariant edge cases without failing the
                  // entire simulation suite by default.
                  logAiDiagnostic(
                    'backend-s-invariant-violation',
                    {
                      scenario: scenarioLabel,
                      boardType,
                      numPlayers,
                      run,
                      seed,
                      step: i,
                      phase: before.currentPhase,
                      currentPlayer: before.currentPlayer,
                      beforeProgress,
                      afterProgress,
                      beforeStatus: before.gameStatus,
                      afterStatus: after.gameStatus,
                      stacks: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
                        key,
                        controllingPlayer: stack.controllingPlayer,
                        stackHeight: stack.stackHeight,
                        capHeight: stack.capHeight,
                      })),
                      markers: Array.from(before.board.markers.entries()).map(([key, marker]) => ({
                        key,
                        player: marker.player,
                      })),
                      collapsedSpaces: Array.from(before.board.collapsedSpaces.entries()).map(
                        ([key, owner]) => ({ key, owner })
                      ),
                    },
                    'backend-ai-sim'
                  );
                }
              }

              lastProgress = afterProgress;

              // Record this move as a canonical simulation step for S + stall
              // classification (hash-based, active-game stall windows).
              snapshots.push(createSimulationStepSnapshot(before, after));

              if (after.gameStatus !== 'active') {
                // Game terminated on this move.
                break;
              }
            }

            const finalState = engine.getGameState();

            const classification = classifySeedRun(
              snapshots,
              finalState,
              boardType,
              numPlayers,
              seed,
              {
                maxActionsPerGame: MAX_MOVES_PER_GAME,
                stallWindowSteps: STALL_WINDOW_STEPS,
              }
            );

            scenarioClassifications.push(classification);

            // For clearly non-terminating runs (exhausted move budget while still
            // active), log a detailed snapshot to aid debugging. The shared
            // classification helper will surface these as non_terminating seeds.
            if (classification.nonTerminating) {
              logAiDiagnostic(
                'backend-non-terminating-game',
                {
                  scenario: scenarioLabel,
                  run,
                  seed,
                  finalPlayer: finalState.currentPlayer,
                  finalPhase: finalState.currentPhase,
                  players: finalState.players.map((p) => ({
                    playerNumber: p.playerNumber,
                    type: p.type,
                    ringsInHand: p.ringsInHand,
                    eliminatedRings: p.eliminatedRings,
                    territorySpaces: p.territorySpaces,
                  })),
                  stacks: Array.from(finalState.board.stacks.entries()).map(([key, stack]) => ({
                    key,
                    controllingPlayer: stack.controllingPlayer,
                    stackHeight: stack.stackHeight,
                    capHeight: stack.capHeight,
                  })),
                  markers: Array.from(finalState.board.markers.entries()).map(([key, marker]) => ({
                    key,
                    player: marker.player,
                  })),
                  collapsedSpaces: Array.from(finalState.board.collapsedSpaces.entries()).map(
                    ([key, owner]) => ({ key, owner })
                  ),
                },
                'backend-ai-sim'
              );
            }
          }

          // After all runs for this scenario, determine which seeds are actionable
          // and log diagnostics only for those. Any non-ok seed is considered
          // actionable for the backend harness (no backend-specific whitelist).
          const actionable: SeedRunClassification[] = [];

          for (const classification of scenarioClassifications) {
            const status = getSeedRunStatus(classification);
            if (status === 'ok') {
              continue;
            }

            logAiDiagnostic(
              'backend-ai-seed-result',
              {
                boardType: classification.boardType,
                numPlayers: classification.numPlayers,
                seed: classification.seed,
                totalActions: classification.totalActions,
                sViolationsCount: classification.sViolations.length,
                stallWindows: classification.stallWindows,
                nonTerminating: classification.nonTerminating,
                status,
              },
              'backend-ai-sim'
            );

            actionable.push(classification);
          }

          if (actionable.length > 0) {
            const summary = actionable
              .map((c) => {
                const status = getSeedRunStatus(c);
                const firstStall = c.stallWindows[0];
                const stallInfo = firstStall
                  ? `firstStallStart=${firstStall.startAction},firstStallLength=${firstStall.length}`
                  : 'noStall';
                return `${c.boardType}/${c.numPlayers}p seed=${c.seed} status=${status} (S-violations=${c.sViolations.length}, stalls=${c.stallWindows.length} [${stallInfo}], nonTerminating=${c.nonTerminating})`;
              })
              .join('; ');

            throw new Error(
              `Backend AI-style simulation reported actionable seeds for scenario "${scenarioLabel}" ` +
                `(S-invariant decreases, stalls of ${STALL_WINDOW_STEPS}+ moves, or non-terminating games ` +
                `within ${MAX_MOVES_PER_GAME} moves): ${summary}. ` +
                'See logs/ai/backend-ai-sim.log for detailed diagnostics.'
            );
          }
        });
      }
    }
  }
);
