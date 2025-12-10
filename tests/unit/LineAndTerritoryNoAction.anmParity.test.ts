import type { GameState } from '../../src/shared/types/game';
import {
  hasPhaseLocalInteractiveMove,
  computeGlobalLegalActionsSummary,
  isANMState,
} from '../../src/shared/engine/globalActions';
import { enumerateProcessLineMoves } from '../../src/shared/engine/aggregates/LineAggregate';
import {
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
} from '../../src/shared/engine/aggregates/TerritoryAggregate';
import {
  makeAnmScen04_TerritoryNoRemainingDecisions,
  makeAnmScen05_LineProcessingNoRemainingDecisions,
} from '../fixtures/anmFixtures';

/**
 * Line & Territory ANM / no_*_action parity tests.
 *
 * These tests complement MovementNoAction.anmParity.test.ts by covering:
 *
 * - ANM-SCEN-04 – Territory processing with no remaining decisions.
 * - ANM-SCEN-05 – Line processing with no remaining decisions.
 *
 * In both scenarios:
 * - Phase-local decision surfaces for the current phase are empty.
 * - Other global actions (placements, future movement, etc.) still exist.
 * - isANMState(state) must therefore return false.
 */
describe('Line and Territory ANM / no_*_action parity', () => {
  /**
   * ANM-SCEN-04 – Territory processing with no remaining decisions.
   *
   * Shape:
   * - gameStatus == 'active'
   * - currentPhase == 'territory_processing'
   * - No disconnected regions for currentPlayer.
   * - No pending territory eliminations for currentPlayer.
   * - Player still has global legal placements available.
   *
   * Expectations:
   * - enumerateProcessTerritoryRegionMoves(...) === [].
   * - enumerateTerritoryEliminationMoves(...) === [].
   * - hasPhaseLocalInteractiveMove(...) === false for territory_processing.
   * - Global placements exist so isANMState(state) === false.
   */
  test('ANM-SCEN-04: territory_processing with no remaining decisions is not ANM', () => {
    const state: GameState = makeAnmScen04_TerritoryNoRemainingDecisions();
    const player = state.currentPlayer;

    expect(state.currentPhase).toBe('territory_processing');
    expect(state.gameStatus).toBe('active');

    const regionMoves = enumerateProcessTerritoryRegionMoves(state, player);
    const elimMoves = enumerateTerritoryEliminationMoves(state, player);

    expect(regionMoves.length).toBe(0);
    expect(elimMoves.length).toBe(0);

    // Phase-local decision surface for territory_processing is empty.
    expect(hasPhaseLocalInteractiveMove(state, player)).toBe(false);

    const summary = computeGlobalLegalActionsSummary(state, player);

    // Player still has turn-material via rings in hand and can place globally.
    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasGlobalPlacementAction).toBe(true);
    expect(summary.hasForcedEliminationAction).toBe(false);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);

    // INV-ACTIVE-NO-MOVES: state is not ANM because global placements exist.
    expect(isANMState(state)).toBe(false);
  });

  /**
   * ANM-SCEN-05 – Line processing with no remaining decisions.
   *
   * Shape:
   * - gameStatus == 'active'
   * - currentPhase == 'line_processing'
   * - No lines/choices remain for currentPlayer.
   * - Player still has global placements available.
   *
   * Expectations:
   * - enumerateProcessLineMoves(...) === [].
   * - hasPhaseLocalInteractiveMove(...) === false for line_processing.
   * - Global placements exist so isANMState(state) === false.
   */
  test('ANM-SCEN-05: line_processing with no remaining decisions is not ANM', () => {
    const state: GameState = makeAnmScen05_LineProcessingNoRemainingDecisions();
    const player = state.currentPlayer;

    expect(state.currentPhase).toBe('line_processing');
    expect(state.gameStatus).toBe('active');

    const lineMoves = enumerateProcessLineMoves(state, player, {
      detectionMode: 'detect_now',
    });

    expect(lineMoves.length).toBe(0);
    expect(hasPhaseLocalInteractiveMove(state, player)).toBe(false);

    const summary = computeGlobalLegalActionsSummary(state, player);

    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasGlobalPlacementAction).toBe(true);
    expect(summary.hasForcedEliminationAction).toBe(false);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);

    expect(isANMState(state)).toBe(false);
  });
});
