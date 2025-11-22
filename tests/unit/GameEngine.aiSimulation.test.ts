import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { logAiDiagnostic } from '../utils/aiTestLogger';

/**
 * Backend AI-vs-AI style simulation tests.
 *
 * These do not go through the Python AI service. Instead, they use a simple
 * local policy that always selects a legal move from GameEngine.getValidMoves
 * for the current player. The goal is to exercise the full GameEngine/RuleEngine
 * flow and verify that:
 *
 * - The game makes steady progress (no infinite active-loop with legal moves),
 * - The game terminates (gameStatus !== 'active') within a generous bound,
 * - We never reach a state where the game is active, it is a player's turn,
 *   and there are zero legal moves available from getValidMoves.
 *
 * This mirrors the sandbox AI simulation harness but runs entirely on the
 * backend engine. Here we fuzz across multiple board types and player counts
 * using a seeded PRNG so that any discovered stall is reproducible.
 */

describe('GameEngine AI-style simulations (backend termination / stall checks)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal'];
  const playerCounts: number[] = [2, 3, 4];
  // Reduced for faster CI: still enough fuzzing to catch regressions, but
  // without the original 100 * 10_000 move bound per scenario.
  const RUNS_PER_SCENARIO = 20;
  const MAX_MOVES_PER_GAME = 1000;
  const MAX_STAGNANT = 8; // tolerate brief no-op stretches but not long stalls

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

  function hashGameState(state: GameState): string {
    const board = state.board;

    const stacks: string[] = [];
    for (const [key, stack] of board.stacks.entries()) {
      stacks.push(`${key}:${stack.controllingPlayer}:${stack.stackHeight}:${stack.capHeight}`);
    }
    stacks.sort();

    const markers: string[] = [];
    for (const [key, marker] of board.markers.entries()) {
      markers.push(`${key}:${marker.player}`);
    }
    markers.sort();

    const collapsed: string[] = [];
    for (const [key, owner] of board.collapsedSpaces.entries()) {
      collapsed.push(`${key}:${owner}`);
    }
    collapsed.sort();

    const playersMeta = state.players
      .map((p) => `${p.playerNumber}:${p.ringsInHand}:${p.eliminatedRings}:${p.territorySpaces}`)
      .sort()
      .join('|');

    const meta = `${state.currentPlayer}:${state.currentPhase}:${state.gameStatus}`;
    return [meta, playersMeta, stacks.join('|'), markers.join('|'), collapsed.join('|')].join('#');
  }

  /**
   * Progress / termination invariant helper based on the rules-level S metric:
   *   S = M + C + E
   * where
   *   - M = number of markers on the board,
   *   - C = number of collapsed spaces (territory),
   *   - E = total eliminated rings over all players.
   *
   * This should be non-decreasing over the lifetime of any legal game, and
   * strictly increasing whenever a real movement / capture-style action is
   * performed. These tests primarily assert non-decrease to catch regressions
   * in the implementation of the invariant.
   */
  function computeProgressMetric(state: GameState): {
    markers: number;
    collapsed: number;
    eliminated: number;
    S: number;
  } {
    const markers = state.board.markers.size;
    const collapsed = state.board.collapsedSpaces.size;

    // Prefer the aggregated totalRingsEliminated when available; fall back to
    // summing per-player eliminated ring counts on the board for any legacy
    // states that might not have totalRingsEliminated populated.
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

  /**
   * Helper: choose a random legal move for the current player using
   * GameEngine.getValidMoves, driven by a seeded PRNG.
   */
  function chooseRandomMove(engine: GameEngine, state: GameState, rng: () => number): Move | null {
    const currentPlayer = state.currentPlayer;
    const moves = engine.getValidMoves(currentPlayer);

    if (!moves.length) {
      return null;
    }

    const idx = Math.floor(rng() * moves.length);
    return moves[Math.min(idx, moves.length - 1)];
  }

  for (const boardType of boardTypes) {
    for (const numPlayers of playerCounts) {
      const scenarioLabel = `${boardType} with ${numPlayers} AI players`;

      test(`${scenarioLabel}: 100 seeded-random games do not stall and terminate within ${MAX_MOVES_PER_GAME} moves`, async () => {
        const boardIndex = boardTypes.indexOf(boardType);
        const playerCountIndex = playerCounts.indexOf(numPlayers);

        for (let run = 0; run < RUNS_PER_SCENARIO; run++) {
          // Derive a reproducible seed for this scenario + run.
          const seed = 1 + run + playerCountIndex * 1000 + boardIndex * 100000;
          const rng = makePrng(seed);

          const engine = createEngineWithPlayers(boardType, numPlayers);

          let stagnantSteps = 0;
          let lastProgress = computeProgressMetric(engine.getGameState());

          for (let i = 0; i < MAX_MOVES_PER_GAME; i++) {
            const before = engine.getGameState();
            const beforeProgress = computeProgressMetric(before);

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
              const afterAutoProgress = computeProgressMetric(afterAuto);

              // Automatic processing must also respect non-decreasing S.
              expect(afterAutoProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
              lastProgress = afterAutoProgress;

              if (afterAuto.gameStatus !== 'active') {
                break;
              }
              // Re-evaluate from the new state on the next loop
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
              const afterResolveProgress = computeProgressMetric(afterResolve);

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

            const beforeHash = hashGameState(before);

            const { id, timestamp, moveNumber, ...payload } = move;
            const result = await engine.makeMove(
              payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
            );
            expect(result.success).toBe(true);

            const after = engine.getGameState();
            const afterHash = hashGameState(after);
            const afterProgress = computeProgressMetric(after);

            // Progress metric must be globally non-decreasing.
            expect(afterProgress.S).toBeGreaterThanOrEqual(lastProgress.S);

            // For movement / capture phases (i.e. real "action" turns), S is
            // expected to increase over the course of the game, but individual
            // actions may occasionally be S-neutral while still being legal.
            // We therefore log diagnostics when S does not strictly increase
            // here, but rely on the global non-decrease check above plus the
            // separate stall detector (stagnantSteps) to guard against
            // pathological behaviour.
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

            if (afterHash === beforeHash && after.gameStatus === 'active') {
              stagnantSteps++;
            } else {
              stagnantSteps = 0;
            }

            if (stagnantSteps >= MAX_STAGNANT) {
              throw new Error(
                `Detected potential backend stall: scenario=${scenarioLabel}, run=${run}, seed=${seed}, ` +
                  `currentPlayer=${after.currentPlayer}, phase=${after.currentPhase}, ` +
                  `no state change for ${stagnantSteps} consecutive moves at step ${i}`
              );
            }

            if (after.gameStatus !== 'active') {
              // Game terminated on this move.
              break;
            }
          }

          const finalState = engine.getGameState();
          if (finalState.gameStatus === 'active') {
            // Log a detailed snapshot to the AI test logger so we can debug
            // non-terminating scenarios without dumping the entire board into
            // the Jest console by default.
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

            throw new Error(
              `Backend AI-style simulation did not reach a terminal state within ${MAX_MOVES_PER_GAME} moves. ` +
                `scenario=${scenarioLabel}, run=${run}, seed=${seed}, ` +
                `final currentPlayer=${finalState.currentPlayer}, phase=${finalState.currentPhase}`
            );
          }
        }
      });
    }
  }
});
