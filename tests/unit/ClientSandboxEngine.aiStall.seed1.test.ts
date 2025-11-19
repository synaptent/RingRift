import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import { hashGameState, computeProgressSnapshot } from '../../src/shared/engine/core';

/**
 * Regression guard for a previously observed sandbox AI stall on square8 with
 * 2 AI players and seed=1. This is intentionally wired behind an env flag
 * so it does not run in normal CI:
 *
 *   RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO=1 npm test -- ClientSandboxEngine.aiStall.seed1
 *
 * The test is now expected to PASS only when no stall is present, i.e. it
 * never observes MAX_STAGNANT or more consecutive AI actions with no state
 * change. If a future change reintroduces a stall for this seed, this test
 * will begin to fail again.
 */

const STALL_REPRO_ENABLED = process.env.RINGRIFT_ENABLE_SANDBOX_AI_STALL_REPRO === '1';

const maybeTest = STALL_REPRO_ENABLED ? test : test.skip;

const BOARD_TYPE: BoardType = 'square8';
const NUM_PLAYERS = 2;
const SEED = 1;

// Keep these aligned with the main aiSimulation harness so that we
// reproduce the same behaviour reported there.
const MAX_AI_ACTIONS = 100; // enough to reach the first reported stall
const MAX_STAGNANT = 8;

/**
 * Tiny deterministic PRNG so we can reproduce the failing run by its seed.
 * Same LCG parameters as used in ClientSandboxEngine.aiSimulation.test.ts.
 */
function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    // LCG parameters from Numerical Recipes
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

class SeedStallInteractionHandler implements SandboxInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;

    if (anyChoice.type === 'capture_direction') {
      const cd = anyChoice as CaptureDirectionChoice;
      const options = cd.options || [];
      if (options.length === 0) {
        throw new Error('SeedStallInteractionHandler: no options for capture_direction');
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
  }
}

function createEngine(): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType: BOARD_TYPE,
    numPlayers: NUM_PLAYERS,
    playerKinds: Array.from({ length: NUM_PLAYERS }, () => 'ai'),
  };

  return new ClientSandboxEngine({
    config,
    interactionHandler: new SeedStallInteractionHandler(),
  });
}

maybeTest('sandbox AI seed=1 (square8, 2 AI) does not exhibit a movement-phase stall', async () => {
  const rng = makePrng(SEED);
  const originalRandom = Math.random;
  Math.random = rng;

  try {
    const engine = createEngine();

    let stagnantSteps = 0;
    let lastState: GameState = engine.getGameState();
    let lastHash = hashGameState(lastState);

    for (let i = 0; i < MAX_AI_ACTIONS; i += 1) {
      const before = engine.getGameState();
      const beforeHash = hashGameState(before);
      const beforeProgress = computeProgressSnapshot(before);

      // Only care about AI turns while the game is active.
      if (before.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = before.players.find((p) => p.playerNumber === before.currentPlayer);
      if (!currentPlayer || currentPlayer.type !== 'ai') {
        continue;
      }

      await engine.maybeRunAITurn();

      const after = engine.getGameState();
      const afterHash = hashGameState(after);
      const afterProgress = computeProgressSnapshot(after);

      const stateUnchanged = afterHash === beforeHash && after.gameStatus === 'active';

      if (stateUnchanged) {
        stagnantSteps += 1;
      } else {
        stagnantSteps = 0;
      }

      // Record the last seen state so debugging this test is easier.
      lastState = after;
      lastHash = afterHash;

      if (stagnantSteps >= MAX_STAGNANT) {
        break;
      }
    }

    // This test is intentionally written to assert that no stall is present
    // for this seed: we should never see MAX_STAGNANT or more consecutive
    // AI turns with no state change.
    expect(stagnantSteps).toBeLessThan(MAX_STAGNANT);

    // Sanity check: ensure we actually executed at least one AI action.
    expect(lastHash).toBeDefined();
    expect(lastState.gameStatus).toBeDefined();
  } finally {
    Math.random = originalRandom;
  }
});
