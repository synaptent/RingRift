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
import { hashGameState, computeProgressSnapshot } from '../../src/shared/engine/core';
import { STALL_WINDOW_STEPS, MAX_AI_ACTIONS_PER_GAME } from '../utils/aiSimulationPolicy';

/**
 * Focused regression for the historical sandbox AI stall observed in the
 * square8 / 2-player / seed=18 scenario.
 *
 * This test is intentionally much lighter than the single-seed debug
 * harness: it simply asserts that, from a fresh game with that seed,
 * repeated calls to maybeRunAITurn eventually terminate the game
 * (gameStatus !== 'active') within a reasonable action budget and do
 * not get stuck in a long run of true no-op AI turns.
 *
 * The heavy diagnostics and invariant checks remain in
 * ClientSandboxEngine.aiSingleSeedDebug.test.ts; this file serves as a
 * fast, CI-friendly guard against reintroducing a non-terminating
 * ring_placement control-flow bug.
 */

test('Sandbox AI regression: square8 / 2p / seed=18 terminates without long no-op stall', async () => {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;
  const seed = 18;

  // Action budget and no-op window are deliberately smaller than the
  // single-seed debug harness, since this test is meant to be a quick
  // regression rather than a full diagnostic run. The window length is
  // aligned with the canonical STALL_WINDOW_STEPS used by the shared
  // aiSimulationPolicy tests so that "no-op stall" semantics match the
  // S-invariant + hash-based stall classification.
  const MAX_AI_ACTIONS = MAX_AI_ACTIONS_PER_GAME;
  const MAX_STAGNANT = STALL_WINDOW_STEPS;

  function makePrng(seedValue: number): () => number {
    let s = seedValue >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }

  function createEngine(bt: BoardType, players: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType: bt,
      numPlayers: players,
      playerKinds: Array.from({ length: players }, () => 'ai'),
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

  const rng = makePrng(seed);
  const originalRandom = Math.random;
  Math.random = rng;

  try {
    const engine = createEngine(boardType, numPlayers);

    let stagnantSteps = 0;
    let lastHash = hashGameState(engine.getGameState());
    let lastProgress = computeProgressSnapshot(engine.getGameState());

    for (let i = 0; i < MAX_AI_ACTIONS; i++) {
      const before = engine.getGameState();
      const beforeProgress = computeProgressSnapshot(before);

      // Global S-invariant: non-decreasing over time.
      expect(beforeProgress.S).toBeGreaterThanOrEqual(lastProgress.S);
      lastProgress = beforeProgress;

      if (before.gameStatus !== 'active') {
        break;
      }

      await engine.maybeRunAITurn();
      const after = engine.getGameState();
      const afterHash = hashGameState(after);
      const afterProgress = computeProgressSnapshot(after);

      // Per-action S-invariant: S must not decrease across an AI action.
      expect(afterProgress.S).toBeGreaterThanOrEqual(beforeProgress.S);

      if (afterHash === lastHash && after.gameStatus === 'active') {
        stagnantSteps++;
      } else {
        stagnantSteps = 0;
      }

      lastHash = afterHash;
      lastProgress = afterProgress;

      // Guard: we should never see a long run of true no-op AI turns
      // in this scenario now that ring_placement control flow correctly
      // forces progress (placement, movement, or elimination). The stall
      // window is aligned with the canonical STALL_WINDOW_STEPS used by
      // the shared aiSimulationPolicy classification.
      if (stagnantSteps >= MAX_STAGNANT) {
        throw new Error(
          `Sandbox AI regression: detected ${stagnantSteps} consecutive no-op AI turns ` +
            `for square8/2p/seed=18 at action=${i}`
        );
      }
    }

    const finalState = engine.getGameState();
    expect(finalState.gameStatus).not.toBe('active');
  } finally {
    Math.random = originalRandom;
  }
});
