import {
  hasTurnMaterial,
  hasGlobalPlacementAction,
  hasPhaseLocalInteractiveMove,
  hasForcedEliminationAction,
  applyForcedEliminationForPlayer,
  computeGlobalLegalActionsSummary,
  isANMState,
  computeSMetric,
  computeTMetric,
} from '../../src/shared/engine';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  pos,
} from '../utils/fixtures';

describe('globalActions – R200/R2xx global legal actions and ANM semantics', () => {
  describe('hasTurnMaterial', () => {
    it('is true when player has rings in hand', () => {
      const state = createTestGameState({
        players: [createTestPlayer(1, { ringsInHand: 5 }), createTestPlayer(2)],
      });

      expect(hasTurnMaterial(state, 1)).toBe(true);
      expect(hasTurnMaterial(state, 2)).toBe(true);
    });

    it('is true when player controls at least one stack even with no rings in hand', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 2);

      const state = createTestGameState({
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0 }),
          createTestPlayer(2, { ringsInHand: 0 }),
        ],
      });

      expect(hasTurnMaterial(state, 1)).toBe(true);
      expect(hasTurnMaterial(state, 2)).toBe(false);
    });

    it('is false when player has no rings in hand and no stacks', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [
          createTestPlayer(1, { ringsInHand: 0 }),
          createTestPlayer(2, { ringsInHand: 0 }),
        ],
      });

      expect(hasTurnMaterial(state, 1)).toBe(false);
      expect(hasTurnMaterial(state, 2)).toBe(false);
    });
  });

  describe('hasGlobalPlacementAction and hasPhaseLocalInteractiveMove', () => {
    it('reports global placements and phase-local moves in a normal ring_placement state', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
      });

      expect(hasGlobalPlacementAction(state, 1)).toBe(true);
      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(true);

      // Non-active players never have phase-local interactive moves.
      expect(hasPhaseLocalInteractiveMove(state, 2)).toBe(false);
    });

    it('hasGlobalPlacementAction is false when ringsInHand == 0', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [
          createTestPlayer(1, { ringsInHand: 0 }),
          createTestPlayer(2, { ringsInHand: 0 }),
        ],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
      });

      expect(hasGlobalPlacementAction(state, 1)).toBe(false);
      expect(hasGlobalPlacementAction(state, 2)).toBe(false);
    });
  });

  // INV-S-MONOTONIC / INV-ELIMINATION-MONOTONIC (R191, R207)
  // INV-ACTIVE-NO-MOVES (R200–R203) – forced elimination counts as a global action
  describe('hasForcedEliminationAction and applyForcedEliminationForPlayer', () => {
    it('detects forced-elimination opportunities when a player is blocked with stacks but no placements or moves', () => {
      // Use a 1×1 square board so that no non-capture movement or capture is
      // possible from the single cell.
      const board = createTestBoard('square8');
      // Override size for this test to force a 1×1 effective grid.
      (board as any).size = 1;
      addStack(board, pos(0, 0), 1, 2);

      const state = createTestGameState({
        board,
        players: [
          createTestPlayer(1, { ringsInHand: 0 }),
          createTestPlayer(2, { ringsInHand: 0 }),
        ],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        totalRingsEliminated: 0,
      });

      // Preconditions: player 1 has material on board but no placements and
      // no legal moves/captures from the single stack.
      expect(hasTurnMaterial(state, 1)).toBe(true);
      expect(hasGlobalPlacementAction(state, 1)).toBe(false);

      expect(hasForcedEliminationAction(state, 1)).toBe(true);

      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasGlobalPlacementAction).toBe(false);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
      expect(summary.hasForcedEliminationAction).toBe(true);

      const beforeS = computeSMetric(state);
      const beforeT = computeTMetric(state);
      const outcome = applyForcedEliminationForPlayer(state, 1);
      expect(outcome).toBeDefined();
      if (!outcome) {
        return;
      }

      expect(outcome.eliminatedPlayer).toBe(1);
      expect(outcome.eliminatedFrom).toEqual(pos(0, 0));
      expect(outcome.eliminatedCount).toBeGreaterThanOrEqual(1);

      const afterS = computeSMetric(outcome.nextState);
      const afterT = computeTMetric(outcome.nextState);
      expect(afterS).toBeGreaterThanOrEqual(beforeS);
      expect(afterT).toBeGreaterThan(beforeT);

      // By definition, states where forced elimination is available are not ANM.
      expect(isANMState(state)).toBe(false);
      expect(isANMState(outcome.nextState)).toBe(false);
    });
  });

  // INV-ACTIVE-NO-MOVES (R200–R203) and INV-ANM-TURN-MATERIAL-SKIP (R201)
  describe('isANMState (ACTIVE-no-moves predicate)', () => {
    it('returns false when game is not active', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        gameStatus: 'finished',
        currentPlayer: 1,
      });

      expect(isANMState(state)).toBe(false);
    });

    it('returns false when current player has no turn material', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        currentPlayer: 1,
        gameStatus: 'active',
        players: [
          createTestPlayer(1, { ringsInHand: 0 }),
          createTestPlayer(2, { ringsInHand: 0 }),
        ],
      });

      expect(hasTurnMaterial(state, 1)).toBe(false);
      expect(isANMState(state)).toBe(false);
    });

    it('returns false when global placements or phase-local moves exist', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
      });

      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasGlobalPlacementAction).toBe(true);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(true);
      expect(summary.hasForcedEliminationAction).toBe(false);

      expect(isANMState(state)).toBe(false);
    });

    it('treats placements-only movement states as non-ANM (ANM-SCEN-02)', () => {
      // Movement phase with no movement/capture actions but legal placements
      // available on the next ring_placement phase.
      const board = createTestBoard('square8');
      const state = createTestGameState({
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
      });

      // No stacks on the board ⇒ no phase-local movement/capture options.
      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);

      // But global placements are still available and must count as legal
      // actions for ANM purposes (RR-CANON-R200 / ANM-SCEN-02).
      expect(summary.hasGlobalPlacementAction).toBe(true);
      expect(summary.hasForcedEliminationAction).toBe(false);

      expect(isANMState(state)).toBe(false);
    });

    it('returns true for an artificial ANM candidate with turn-material but no global actions (unreachable shape)', () => {
      const board = createTestBoard('square8');
      // Shrink the playable area so that no placements or movements exist.
      (board as any).size = 0;
      board.stacks.clear();
      board.markers.clear();
      board.collapsedSpaces.clear();

      const state = createTestGameState({
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 1 }), createTestPlayer(2)],
      });

      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasGlobalPlacementAction).toBe(false);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
      expect(summary.hasForcedEliminationAction).toBe(false);

      // INV-ACTIVE-NO-MOVES: ANM(state) holds for this synthetic candidate.
      expect(isANMState(state)).toBe(true);
    });
  });
});
