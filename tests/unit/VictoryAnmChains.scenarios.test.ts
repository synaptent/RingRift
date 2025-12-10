import type { GameState } from '../../src/shared/types/game';
import { computeGlobalLegalActionsSummary, isANMState } from '../../src/shared/engine/globalActions';
import { evaluateVictory } from '../../src/shared/engine/victoryLogic';
import { makeAnmScen06_GlobalStalemateBareBoard } from '../fixtures/anmFixtures';

/**
 * Victory / ANM chain scenarios (TS-side)
 *
 * These tests exercise how ANM-style global no-action states interact with
 * the canonical victory ladder in victoryLogic, focusing on:
 *
 * - ANM-SCEN-06 – Global stalemate on a bare board (no legal actions for any player).
 *
 * The key requirement is that structurally terminal bare-board positions are
 * resolved by victory logic (stalemate ladder) rather than being treated as
 * ANM violations for the active player.
 */
describe('Victory / ANM chains – global stalemate and LPS', () => {
  /**
   * ANM-SCEN-06 – Global stalemate (bare board, no legal actions).
   *
   * Fixture shape (see tests/fixtures/anmFixtures.ts):
   * - No stacks on the board.
   * - All spaces are collapsed territory.
   * - Players still have rings in hand, but:
   *   - hasGlobalPlacementAction(state, player) === false for every player.
   *   - hasForcedEliminationAction(state, player) === false for every player.
   *
   * Victory expectations (per victoryLogic.evaluateVictory):
   * - Game is terminal via the stalemate ladder.
   * - Winner is the player with the highest effective elimination score
   *   (including rings in hand treated as eliminated).
   * - reason === 'ring_elimination' and handCountsAsEliminated === true.
   *
   * ANM expectations:
   * - isANMState(state) === false for the synthetic ACTIVE snapshot; hosts
   *   must apply victory evaluation instead of treating this as an ANM turn.
   */
  test('ANM-SCEN-06: bare-board global stalemate is resolved by victory logic, not ANM', () => {
    const state: GameState = makeAnmScen06_GlobalStalemateBareBoard();

    // Sanity-check bare-board geometry.
    expect(state.board.stacks.size).toBe(0);
    expect(state.board.collapsedSpaces.size).toBe(state.board.size * state.board.size);
    expect(state.gameStatus).toBe('active');

    // For every player, the global action surface should be empty:
    // - no placements,
    // - no phase-local interactive moves (currentPlayer is a non-participant),
    // - no forced elimination.
    for (const player of state.players) {
      const summary = computeGlobalLegalActionsSummary(state, player.playerNumber);

      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasGlobalPlacementAction).toBe(false);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
      expect(summary.hasForcedEliminationAction).toBe(false);
    }

    // Victory logic should detect a terminal stalemate and select a winner
    // via the §13.4 stalemate ladder (territory → eliminated+hand → markers → last actor).
    const result = evaluateVictory(state);

    expect(result.isGameOver).toBe(true);
    // Fixture is constructed so that player 1 wins on effective elimination score
    // once rings in hand are treated as eliminated.
    expect(result.winner).toBe(1);
    expect(result.reason).toBe('ring_elimination');
    expect(result.handCountsAsEliminated).toBe(true);

    // The synthetic ACTIVE snapshot used for global stalemate evaluation is
    // not considered an ANM state for the (non-participating) currentPlayer.
    expect(isANMState(state)).toBe(false);
  });
});
