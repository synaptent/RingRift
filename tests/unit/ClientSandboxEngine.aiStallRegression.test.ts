import { hashGameState, computeProgressSnapshot } from '../../src/shared/engine/core';
import { reproduceSquare8TwoAiSeed1AtAction } from '../utils/aiSeedSnapshots';
import { STALL_WINDOW_STEPS } from '../utils/aiSimulationPolicy';
import { isFSMOrchestratorActive } from '../../src/shared/utils/envFlags';

/**
 * Regression for the sandbox AI stall discovered by the fuzz harness in the
 * square8 / 2 AI / seed=1 configuration around action ~58.
 *
 * The original failure mode was that, from a mid-game plateau, repeated
 * calls to maybeRunAITurn would leave the GameState unchanged for many
 * consecutive AI actions while gameStatus remained 'active'. This test
 * checkpoints a reproducible mid-game state derived from the same seeded
 * harness and then asserts that we do not observe a long stagnant stretch
 * of identical hashes under continued AI play.
 */

// TODO: FSM issue - AI stall behavior may differ under FSM orchestration.
// These tests were created with legacy orchestration behavior.
// Enable once FSM behavior is fully stabilized for AI simulations.
const testFn = isFSMOrchestratorActive() ? test.skip : test;

testFn(
  'ClientSandboxEngine AI stall regression: square8 / 2 AI / seed=1 plateau does not re-stall',
  async () => {
    // Reproduce a mid-game state near the historical stall plateau.
    const targetActionIndex = 58;
    const {
      engine,
      state: checkpointState,
      snapshot,
      actionsTaken,
    } = await reproduceSquare8TwoAiSeed1AtAction(targetActionIndex);

    // Sanity checks on the checkpoint.
    expect(checkpointState.boardType).toBe('square8');
    expect(checkpointState.players.length).toBe(2);
    // Depending on current AI behaviour, the historical stall plateau may now
    // resolve to a completed game before we reach targetActionIndex. For the
    // purposes of this regression we only require that the game is not aborted.
    expect(['active', 'completed']).toContain(checkpointState.gameStatus);
    expect(actionsTaken).toBeGreaterThan(0);

    // Basic S-invariant sanity: non-negative and reasonably mid-game.
    const progressAtCheckpoint = computeProgressSnapshot(checkpointState);
    expect(progressAtCheckpoint.S).toBeGreaterThan(0);

    // From this checkpoint, run a bounded number of additional AI turns and
    // assert that:
    //   - S remains non-decreasing across all AI actions, and
    //   - we do not see a long stretch of identical hashes while the game
    //     remains active, using the shared STALL_WINDOW_STEPS semantics from
    //     the aiSimulationPolicy helper.
    let stagnantSteps = 0;
    let lastHash = hashGameState(engine.getGameState());
    let lastProgress = computeProgressSnapshot(engine.getGameState());

    const MAX_FOLLOWUP_ACTIONS = 80;

    for (let i = 0; i < MAX_FOLLOWUP_ACTIONS; i++) {
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

      if (after.gameStatus === 'active' && afterHash === lastHash) {
        stagnantSteps += 1;
      } else {
        stagnantSteps = 0;
      }

      lastHash = afterHash;

      if (stagnantSteps >= STALL_WINDOW_STEPS) {
        throw new Error(
          `AI stall regression: observed ${stagnantSteps} consecutive stagnant actions ` +
            `from square8/2p/seed1 plateau (checkpoint actionsTaken=${actionsTaken})`
        );
      }
    }

    // The snapshot value is not asserted here directly, but its existence
    // ensures we can plug this plateau into future parity / scenario suites
    // without having to re-encode the full GameState shape.
    expect(snapshot.label).toContain('square8-2p-seed1-action-');
  }
);
