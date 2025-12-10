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
 * Line / Territory ANM parity tests.
 *
 * These tests exercise the TS-side global-actions helpers for:
 * - ANM-SCEN-04 – Territory processing with no remaining decisions.
 * - ANM-SCEN-05 – Line processing with no remaining decisions.
 *
 * For both scenarios we assert:
 * - Phase-local interactive surfaces in the current phase are empty.
 * - Global action surface G(state, currentPlayer) is non-empty via placements.
 * - isANMState(state) === false (these are not ANM states).
 */
describe('Line & Territory ANM parity', () => {
  /**
   * ANM-SCEN-04 – territory_processing with no remaining decisions.
   *
   * Expectations for the currentPlayer:
   * - enumerateProcessTerritoryRegionMoves(...) === []
   * - enumerateTerritoryEliminationMoves(...) === []
   * - hasPhaseLocalInteractiveMove(...) === false
   * - hasTurnMaterial === true (stack + rings in hand)
   * - hasGlobalPlacementAction === true
   * - hasForcedEliminationAction === false
   * - isANMState(state) === false
   */
  test('ANM-SCEN-04: territory_processing with no remaining decisions is not ANM', () => {
    const state: GameState = makeAnmScen04_TerritoryNoRemainingDecisions();
    const player = state.currentPlayer;

    expect(state.currentPhase).toBe('territory_processing');
    expect(state.gameStatus).toBe('active');

    const regionMoves = enumerateProcessTerritoryRegionMoves(state, player);
    const elimMoves = enumerateTerritoryEliminationMoves(state, player);

    // No local territory decisions remain.
    expect(regionMoves.length).toBe(0);
    expect(elimMoves.length).toBe(0);

    // Phase-local interactive surface is empty for the current phase.
    expect(hasPhaseLocalInteractiveMove(state, player)).toBe(false);

    const summary = computeGlobalLegalActionsSummary(state, player);

    // Player still has material and global placements available.
    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasGlobalPlacementAction).toBe(true);
    // No forced elimination and no phase-local moves in this phase.
    expect(summary.hasForcedEliminationAction).toBe(false);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);

    // Territory no-op state is not ANM because placements exist globally.
    expect(isANMState(state)).toBe(false);
  });

  /**
   * ANM-SCEN-05 – line_processing with no remaining decisions.
   *
   * Expectations for the currentPlayer:
   * - enumerateProcessLineMoves(...) === []
   * - hasPhaseLocalInteractiveMove(...) === false
   * - hasTurnMaterial === true (rings in hand)
   * - hasGlobalPlacementAction === true
   * - hasForcedEliminationAction === false
   * - isANMState(state) === false
   */
  test('ANM-SCEN-05: line_processing with no remaining decisions is not ANM', () => {
    const state: GameState = makeAnmScen05_LineProcessingNoRemainingDecisions();
    const player = state.currentPlayer;

    expect(state.currentPhase).toBe('line_processing');
    expect(state.gameStatus).toBe('active');

    const lineMoves = enumerateProcessLineMoves(state, player, {
      detectionMode: 'detect_now',
    });

    // No line/choice decisions remain for the current player.
    expect(lineMoves.length).toBe(0);

    // Phase-local interactive surface is empty for line_processing.
    expect(hasPhaseLocalInteractiveMove(state, player)).toBe(false);

    const summary = computeGlobalLegalActionsSummary(state, player);

    // Player still has rings in hand and thus placements available.
    expect(summary.hasTurnMaterial).toBe(true);
    expect(summary.hasGlobalPlacementAction).toBe(true);
    // No forced elimination and no local decisions in this phase.
    expect(summary.hasForcedEliminationAction).toBe(false);
    expect(summary.hasPhaseLocalInteractiveMove).toBe(false);

    // Line-processing no-op state is not ANM because placements exist globally.
    expect(isANMState(state)).toBe(false);
  });
});
