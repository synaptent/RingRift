import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import { logAiDiagnostic } from '../utils/aiTestLogger';
import {
  createSimulationStepSnapshot,
  classifySeedRun,
  getSeedRunStatus,
  STALL_WINDOW_STEPS,
  MAX_AI_ACTIONS_PER_GAME,
  type SimulationStepSnapshot,
  type SeedRunClassification,
} from '../utils/aiSimulationPolicy';

/**
 * AI-vs-AI sandbox simulation tests.
 *
 * These are aimed at surfacing seeds where the sandbox AI:
 *   - violates the S-invariant (S decreases across an AI action),
 *   - exhibits structural stalls (hashGameState unchanged for a window of
 *     STALL_WINDOW_STEPS or more consecutive AI actions while gameStatus
 *     remains 'active'), or
 *   - fails to terminate within MAX_AI_ACTIONS_PER_GAME AI actions while
 *     still making some state changes.
 *
 * The suite fuzzes across multiple board types and player counts using a
 * seeded PRNG (by monkey-patching Math.random) so that any discovered issue
 * is reproducible via its seed and scenario.
 *
 * Per seed, we classify runs as:
 *   - ok
 *   - s_violation
 *   - stall
 *   - non_terminating
 *
 * Only seeds with non-ok status are logged and treated as actionable;
 * known, already-covered edge cases are whitelisted so they remain purely
 * diagnostic.
 */

const AI_SIM_ENABLED = process.env.RINGRIFT_ENABLE_SANDBOX_AI_SIM === '1';

/**
 * Known edge-case seeds that already have dedicated regression coverage
 * elsewhere and are therefore treated as diagnostic-only here unless they
 * regress beyond their historical behaviour.
 *
 * Keys are of the form `${boardType}:${numPlayers}:${seed}`.
 */
const KNOWN_EDGE_CASES = new Set<string>([
  // Historical square8 / 2 AI plateau seed used by stall regression + scenario tests.
  'square8:2:1',
  // Historical square8 / 2 AI ring_placement stall seed with dedicated regression + debug harness.
  'square8:2:18',
]);

describe('ClientSandboxEngine AI sandbox simulations (termination / stall classification)', () => {
  const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal'];
  const playerCounts: number[] = [2, 3, 4];

  // Reduced for faster CI: fewer fuzzed games and a canonical per-game action
  // budget aligned with the shared aiSimulationPolicy helper.
  const RUNS_PER_SCENARIO = 20;
  const MAX_AI_ACTIONS = MAX_AI_ACTIONS_PER_GAME;

  function createEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
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
            selectedOption: selected,
          } as PlayerChoiceResponseFor<TChoice>;
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
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

  // This suite is relatively heavy and, as of P1.4 in KNOWN_ISSUES.md,
  // is considered a diagnostic harness rather than a hard CI gate. By
  // default it is skipped unless RINGRIFT_ENABLE_SANDBOX_AI_SIM=1 is
  // set in the environment.
  const maybeTest = AI_SIM_ENABLED ? test : test.skip;

  for (const boardType of boardTypes) {
    for (const numPlayers of playerCounts) {
      const scenarioLabel = `${boardType} with ${numPlayers} AI players`;

      maybeTest(
        `${scenarioLabel}: seeded sandbox games classify seeds by S-invariant / stall / termination within ${MAX_AI_ACTIONS} AI actions`,
        async () => {
          const boardIndex = boardTypes.indexOf(boardType);
          const playerCountIndex = playerCounts.indexOf(numPlayers);

          const runClassifications: SeedRunClassification[] = [];

          for (let run = 0; run < RUNS_PER_SCENARIO; run++) {
            const seed = 1 + run + playerCountIndex * 1000 + boardIndex * 100000;
            const rng = makePrng(seed);
            const originalRandom = Math.random;
            Math.random = rng;

            try {
              const engine = createEngine(boardType, numPlayers);
              const engineAny = engine as any;

              const snapshots: SimulationStepSnapshot[] = [];

              for (let actionIndex = 0; actionIndex < MAX_AI_ACTIONS; actionIndex++) {
                const before = engine.getGameState();

                if (before.gameStatus !== 'active') {
                  // Game ended naturally; stop collecting AI actions for this seed.
                  break;
                }

                const currentPlayer = before.players.find(
                  (p) => p.playerNumber === before.currentPlayer
                );
                if (!currentPlayer || currentPlayer.type !== 'ai') {
                  // Non-AI to move; in the actual UI loop this batch would end and the
                  // next batch would resume when an AI is to move. For the purposes of
                  // this harness, only count concrete AI actions.
                  continue;
                }

                await engine.maybeRunAITurn();

                const after = engine.getGameState();

                // Assert board invariants after each AI action so any
                // illegal stack/marker/territory coexistence is surfaced
                // immediately in diagnostic runs.
                if (engineAny && typeof engineAny.assertBoardInvariants === 'function') {
                  engineAny.assertBoardInvariants(
                    `sandbox-ai-sim:${scenarioLabel}:run=${run}:action=${actionIndex}`
                  );
                }

                const stepSnapshot = createSimulationStepSnapshot(before, after);
                snapshots.push(stepSnapshot);

                if (after.gameStatus !== 'active') {
                  // Game terminated on this AI action; no further actions for this seed.
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
                  maxActionsPerGame: MAX_AI_ACTIONS,
                  stallWindowSteps: STALL_WINDOW_STEPS,
                }
              );

              runClassifications.push(classification);
            } finally {
              Math.random = originalRandom;
            }
          }

          // After all runs for this scenario, determine which seeds are actionable
          // and log diagnostics only for those.
          const actionable: SeedRunClassification[] = [];

          for (const classification of runClassifications) {
            const status = getSeedRunStatus(classification);
            if (status === 'ok') {
              continue;
            }

            const key = `${classification.boardType}:${classification.numPlayers}:${classification.seed}`;
            const isKnownEdgeCase = KNOWN_EDGE_CASES.has(key);

            // Log diagnostics for any non-ok seed; this keeps logs informative
            // without spamming benign seeds.
            logAiDiagnostic(
              'sandbox-ai-seed-result',
              {
                boardType: classification.boardType,
                numPlayers: classification.numPlayers,
                seed: classification.seed,
                totalActions: classification.totalActions,
                sViolationsCount: classification.sViolations.length,
                stallWindows: classification.stallWindows,
                nonTerminating: classification.nonTerminating,
                status,
                knownEdgeCase: isKnownEdgeCase,
              },
              'sandbox-ai-simulation'
            );

            if (!isKnownEdgeCase) {
              actionable.push(classification);
            }
          }

          if (actionable.length > 0) {
            const summary = actionable
              .map((c) => {
                const status = getSeedRunStatus(c);
                return `${c.boardType}/${c.numPlayers}p seed=${c.seed} status=${status} (S-violations=${c.sViolations.length}, stalls=${c.stallWindows.length}, nonTerminating=${c.nonTerminating})`;
              })
              .join('; ');

            throw new Error(
              `Sandbox AI simulation reported actionable seeds for scenario "${scenarioLabel}" ` +
                `(S-invariant violations, stalls of \${STALL_WINDOW_STEPS}+ actions, or non-terminating games ` +
                `within ${MAX_AI_ACTIONS} AI actions): ${summary}. ` +
                'See logs/ai/sandbox-ai-simulation.log for detailed diagnostics.'
            );
          }
        }
      );
    }
  }
});
