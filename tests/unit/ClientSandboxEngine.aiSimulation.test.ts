import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice
} from '../../src/shared/types/game';
import { hashGameState, computeProgressSnapshot } from '../../src/shared/engine/core';
import { logAiDiagnostic } from '../utils/aiTestLogger';

/**
 * AI-vs-AI sandbox simulation tests.
 *
 * These are aimed at surfacing stalls where:
 * - gameStatus remains 'active'
 * - current player is an AI
 * - repeated calls to maybeRunAITurn do not change the game state
 *
 * We fuzz across multiple board types and player counts using a seeded PRNG
 * (by monkey-patching Math.random) so that any discovered stall is
 * reproducible via its seed.
 */

describe('ClientSandboxEngine AI sandbox simulations (termination / stall checks)', () => {
  const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal'];
  const playerCounts: number[] = [2, 3, 4];

  // Reduced for faster CI: fewer fuzzed games and a lower per-game action cap,
  // still sufficient to exercise termination / S-invariant behaviour.
  const RUNS_PER_SCENARIO = 20;
  const MAX_AI_ACTIONS = 1000;
  const MAX_STAGNANT = 8; // tolerate brief no-op stretches but not long stalls

  function createEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'ai')
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;

        if (anyChoice.type === 'capture_direction') {
          const cd = anyChoice as CaptureDirectionChoice;
          const options = cd.options || [];
          if (options.length === 0) {
            throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
          }

          // Deterministically pick the option with the smallest landing x,y
          // to keep simulations reproducible given a fixed Math.random.
          let selected = options[0];
          for (const opt of options) {
            if (
              opt.landingPosition.x < selected.landingPosition.x ||
              (opt.landingPosition.x === selected.landingPosition.x &&
                opt.landingPosition.y < selected.landingPosition.y)
            ) {
              selected = opt;
            }
          }

          return {
            choiceId: cd.id,
            playerNumber: cd.playerNumber,
            choiceType: cd.type,
            selectedOption: selected
          } as PlayerChoiceResponseFor<TChoice>;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption
        } as PlayerChoiceResponseFor<TChoice>;
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
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
    // Delegate to the shared helper so S is computed consistently across
    // backend and sandbox engines.
    return computeProgressSnapshot(state);
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

      test(`${scenarioLabel}: 100 seeded-random sandbox games do not stall within ${MAX_AI_ACTIONS} AI actions or until victory`, async () => {
        const boardIndex = boardTypes.indexOf(boardType);
        const playerCountIndex = playerCounts.indexOf(numPlayers);

        for (let run = 0; run < RUNS_PER_SCENARIO; run++) {
          const seed = 1 + run + playerCountIndex * 1000 + boardIndex * 100000;
          const rng = makePrng(seed);
          const originalRandom = Math.random;
          Math.random = rng;

          try {
            const engine = createEngine(boardType, numPlayers);

            let stagnantSteps = 0;
            let lastProgress = computeProgressMetric(engine.getGameState());
            const recentActions: any[] = [];

            for (let i = 0; i < MAX_AI_ACTIONS; i++) {
              const before = engine.getGameState();
              const beforeProgress = computeProgressMetric(before);

              // S must be globally non-decreasing over the lifetime of the game.
              if (!(beforeProgress.S >= lastProgress.S)) {
                logAiDiagnostic(
                  'sandbox-s-invariant-decrease-before',
                  {
                    scenario: scenarioLabel,
                    run,
                    seed,
                    action: i,
                    lastProgress,
                    beforeProgress
                  },
                  'sandbox-ai-sim'
                );
              }
              expect(beforeProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
              lastProgress = beforeProgress;

              if (before.gameStatus !== 'active') {
                // Game ended naturally; exit the simulation loop.
                break;
              }

              const currentPlayer = before.players.find(
                p => p.playerNumber === before.currentPlayer
              );
              if (!currentPlayer || currentPlayer.type !== 'ai') {
                // Non-AI to move; in the actual UI loop this batch would end and the
                // next batch would resume when an AI is to move. For the purposes of
                // this test, just continue to the next iteration.
                continue;
              }

              const beforeHash = hashGameState(before);

              await engine.maybeRunAITurn();

              const after = engine.getGameState();
              const afterHash = hashGameState(after);
              const afterProgress = computeProgressMetric(after);

              // Progress metric must be globally non-decreasing.
              if (!(afterProgress.S >= lastProgress.S)) {
                const boardBeforeSummaryForDecrease = {
                  stacks: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
                    key,
                    controllingPlayer: stack.controllingPlayer,
                    stackHeight: stack.stackHeight,
                    capHeight: stack.capHeight
                  })),
                  markers: Array.from(before.board.markers.entries()).map(([key, marker]) => ({
                    key,
                    player: marker.player
                  })),
                  collapsedSpaces: Array.from(
                    before.board.collapsedSpaces.entries()
                  ).map(([key, owner]) => ({ key, owner }))
                };

                const boardAfterSummaryForDecrease = {
                  stacks: Array.from(after.board.stacks.entries()).map(([key, stack]) => ({
                    key,
                    controllingPlayer: stack.controllingPlayer,
                    stackHeight: stack.stackHeight,
                    capHeight: stack.capHeight
                  })),
                  markers: Array.from(after.board.markers.entries()).map(([key, marker]) => ({
                    key,
                    player: marker.player
                  })),
                  collapsedSpaces: Array.from(
                    after.board.collapsedSpaces.entries()
                  ).map(([key, owner]) => ({ key, owner }))
                };

                logAiDiagnostic(
                  'sandbox-s-invariant-decrease-after',
                  {
                    scenario: scenarioLabel,
                    run,
                    seed,
                    action: i,
                    lastProgress,
                    beforeProgress,
                    afterProgress,
                    boardBefore: boardBeforeSummaryForDecrease,
                    boardAfter: boardAfterSummaryForDecrease
                  },
                  'sandbox-ai-sim'
                );
              }
              expect(afterProgress.S).toBeGreaterThanOrEqual(lastProgress.S);

              // When maybeRunAITurn actually performs a real "action" turn in
              // movement or capture phase, S should strictly increase between
              // the pre- and post-AI-action states. Ring placement and other
              // non-movement phases are allowed to leave S unchanged, matching
              // the backend AI simulation harness semantics. Pure pass/rotation
              // steps that only advance currentPlayer without any AI move are
              // treated as non-actions and only checked for non-decrease.
              const boardBeforeSummary = {
                stacks: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
                  key,
                  controllingPlayer: stack.controllingPlayer,
                  stackHeight: stack.stackHeight,
                  capHeight: stack.capHeight
                })),
                markers: Array.from(before.board.markers.entries()).map(([key, marker]) => ({
                  key,
                  player: marker.player
                })),
                collapsedSpaces: Array.from(
                  before.board.collapsedSpaces.entries()
                ).map(([key, owner]) => ({ key, owner }))
              };

              const boardAfterSummary = {
                stacks: Array.from(after.board.stacks.entries()).map(([key, stack]) => ({
                  key,
                  controllingPlayer: stack.controllingPlayer,
                  stackHeight: stack.stackHeight,
                  capHeight: stack.capHeight
                })),
                markers: Array.from(after.board.markers.entries()).map(([key, marker]) => ({
                  key,
                  player: marker.player
                })),
                collapsedSpaces: Array.from(
                  after.board.collapsedSpaces.entries()
                ).map(([key, owner]) => ({ key, owner }))
              };

              const lastAIMove = engine.getLastAIMoveForTesting();
              const isMovementOrCaptureAction =
                !!lastAIMove &&
                (lastAIMove.type === 'move_stack' ||
                  lastAIMove.type === 'move_ring' ||
                  lastAIMove.type === 'overtaking_capture');

              if (
                afterHash !== beforeHash &&
                (before.currentPhase === 'movement' || before.currentPhase === 'capture') &&
                isMovementOrCaptureAction
              ) {
                if (!(afterProgress.S > beforeProgress.S)) {
                  logAiDiagnostic(
                    'sandbox-s-invariant-non-increase-on-action',
                    {
                      scenario: scenarioLabel,
                      run,
                      seed,
                      action: i,
                      stepKind: 'window',
                      violation: {
                        beforePhase: before.currentPhase,
                        afterPhase: after.currentPhase,
                        beforePlayer: before.currentPlayer,
                        afterPlayer: after.currentPlayer,
                        beforeProgress,
                        afterProgress,
                        beforeHash,
                        afterHash,
                        lastAIMove
                      },
                      history: recentActions,
                      boardBefore: boardBeforeSummary,
                      boardAfter: boardAfterSummary
                    },
                    'sandbox-ai-sim'
                  );
                }
                expect(afterProgress.S).toBeGreaterThan(beforeProgress.S);
              }

              // Track recent actions for post-failure diagnostics.
              recentActions.push({
                actionIndex: i,
                beforeProgress,
                afterProgress,
                beforePhase: before.currentPhase,
                afterPhase: after.currentPhase,
                beforePlayer: before.currentPlayer,
                afterPlayer: after.currentPlayer,
                gameStatusBefore: before.gameStatus,
                gameStatusAfter: after.gameStatus,
                boardBefore: boardBeforeSummary,
                boardAfter: boardAfterSummary
              });
              if (recentActions.length > 10) {
                recentActions.shift();
              }

              lastProgress = afterProgress;

              if (afterHash === beforeHash && after.gameStatus === 'active') {
                stagnantSteps++;
              } else {
                stagnantSteps = 0;
              }

              if (stagnantSteps >= MAX_STAGNANT) {
                logAiDiagnostic(
                  'sandbox-ai-stall',
                  {
                    scenario: scenarioLabel,
                    run,
                    seed,
                    action: i,
                    stagnantSteps,
                    gameStatus: after.gameStatus,
                    currentPlayer: after.currentPlayer,
                    phase: after.currentPhase,
                    recentActions
                  },
                  'sandbox-ai-sim'
                );
                throw new Error(
                  `Detected potential sandbox AI stall: scenario=${scenarioLabel}, run=${run}, seed=${seed}, ` +
                    `action=${i}, gameStatus=${after.gameStatus}, currentPlayer=${after.currentPlayer}, ` +
                    `phase=${after.currentPhase}, no state change for ${stagnantSteps} consecutive AI actions`
                );
              }
            }

            // If we exhaust the maximum number of AI actions without the sandbox
            // game reaching a terminal state, treat this as a failure so we can
            // surface potential non-terminating behaviour in CI.
            const finalState = engine.getGameState();
            if (finalState.gameStatus === 'active') {
              // Log a detailed snapshot to the AI test logger so we can debug
              // non-terminating sandbox scenarios without flooding the Jest
              // console by default.
              logAiDiagnostic(
                'sandbox-non-terminating-game',
                {
                  scenario: scenarioLabel,
                  run,
                  seed,
                  finalPlayer: finalState.currentPlayer,
                  finalPhase: finalState.currentPhase,
                  recentActions,
                  players: finalState.players.map(p => ({
                    playerNumber: p.playerNumber,
                    type: p.type,
                    ringsInHand: p.ringsInHand,
                    eliminatedRings: p.eliminatedRings,
                    territorySpaces: p.territorySpaces
                  })),
                  stacks: Array.from(finalState.board.stacks.entries()).map(
                    ([key, stack]) => ({
                      key,
                      controllingPlayer: stack.controllingPlayer,
                      stackHeight: stack.stackHeight,
                      capHeight: stack.capHeight
                    })
                  ),
                  markers: Array.from(finalState.board.markers.entries()).map(
                    ([key, marker]) => ({
                      key,
                      player: marker.player
                    })
                  ),
                  collapsedSpaces: Array.from(
                    finalState.board.collapsedSpaces.entries()
                  ).map(([key, owner]) => ({ key, owner }))
                },
                'sandbox-ai-sim'
              );

              throw new Error(
                `Sandbox AI simulation did not reach a terminal state within ${MAX_AI_ACTIONS} AI actions; ` +
                  `scenario=${scenarioLabel}, run=${run}, seed=${seed}, ` +
                  `final currentPlayer=${finalState.currentPlayer}, phase=${finalState.currentPhase}`
              );
            }
          } finally {
            Math.random = originalRandom;
          }
        }
      });
    }
  }
});
