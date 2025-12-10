import {
  computeGlobalLegalActionsSummary,
  isANMState,
  evaluateVictory,
} from '../../src/shared/engine';
import { makeAnmScen06_GlobalStalemateBareBoard } from '../fixtures/anmFixtures';

/**
 * Victory / ANM chain scenarios (TS-side only, light)
 *
 * These tests focus on:
 * - ANM-SCEN-06 – Global stalemate on a bare board with rings in hand.
 *
 * They tie together:
 * - The R200 global legal action summary surface, and
 * - The shared victory evaluator (evaluateVictory).
 *
 * Canonical intent:
 * - In the bare-board global stalemate shape, no player has any global actions:
 *   - No placements (all spaces are collapsed),
 *   - No movement/capture (no stacks),
 *   - No forced elimination (no stacks).
 * - Victory is resolved structurally via the stalemate ladder (§13.4):
 *   - Here, by treating rings in hand as eliminated for tie-break purposes.
 */
describe('Victory / ANM chain scenarios – global stalemate and structural termination', () => {
  /**
   * ANM-SCEN-06 – Global stalemate (no actions for any player).
   *
   * Fixture shape (makeAnmScen06_GlobalStalemateBareBoard):
   * - Board type: square8.
   * - No stacks on the board.
   * - Every cell is collapsed territory.
   * - Both players still have rings in hand (P1 > P2).
   *
   * Expectations:
   * - For each player:
   *   - hasGlobalPlacementAction === false.
   *   - hasPhaseLocalInteractiveMove === false.
   *   - hasForcedEliminationAction === false.
   * - evaluateVictory reports a terminal stalemate resolved via the
   *   ring-elimination tie-break (hand → eliminated), with Player 1
   *   winning and handCountsAsEliminated === true.
   *
   * Note: The ANM predicate (isANMState) is intentionally *not* asserted
   * here. Canonically, this is the only permitted global “no actions for
   * anyone” shape, and termination is handled by evaluateVictory. Future
   * engine work may refine isANMState semantics around structural
   * terminality, but these tests focus on victory behaviour.
   */
  test('ANM-SCEN-06: bare-board global stalemate is resolved by victory logic, not stalled as ANM', () => {
    const state = makeAnmScen06_GlobalStalemateBareBoard();

    // Shape sanity checks.
    expect(state.board.stacks.size).toBe(0);
    expect(state.players.length).toBeGreaterThanOrEqual(2);
    expect(state.gameStatus).toBe('active');

    const anyRingsInHand = state.players.some((p) => p.ringsInHand > 0);
    expect(anyRingsInHand).toBe(true);

    // Global legal action summary: no placements, no movement/capture, no FE
    // for any player. Turn-material remains true via ringsInHand.
    for (const player of state.players) {
      const summary = computeGlobalLegalActionsSummary(state, player.playerNumber);

      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasGlobalPlacementAction).toBe(false);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
      expect(summary.hasForcedEliminationAction).toBe(false);
    }

    // Victory logic must treat this as a structural terminal position and
    // resolve via the stalemate ladder (§13.4), converting rings in hand
    // to eliminated rings for tie-breaking.
    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    expect(result.reason).toBe('ring_elimination');
    // Fixture gives Player 1 strictly more rings in hand than Player 2.
    expect(result.winner).toBe(1);
    expect(result.handCountsAsEliminated).toBe(true);

    // For debugging/telemetry purposes we still ensure the ANM predicate is
    // a well-defined boolean on this shape, but we do not assert its exact
    // value here; termination is enforced via evaluateVictory.
    const anmFlag = isANMState(state);
    expect(typeof anmFlag).toBe('boolean');
  });
});
