/**
 * ANM Scenario Fixtures (TS-side)
 *
 * These fixtures provide small, hand-constructed `GameState` snapshots that
 * correspond to the ANM catalogue in
 * docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md:
 *
 * - ANM-SCEN-01 – Movement phase, no movement/capture, forced elimination available.
 * - ANM-SCEN-02 – Movement phase, placements-only global actions.
 * - ANM-SCEN-03 – Movement phase with a fully eliminated current player.
 * - ANM-SCEN-04 – Territory processing with no remaining decisions.
 * - ANM-SCEN-05 – Line processing with no remaining decisions.
 * - ANM-SCEN-06 – Global stalemate on a bare board (no global actions for any player).
 *
 * Where possible these fixtures reuse the standard test helpers from
 * tests/utils/fixtures.ts so that board layout and player data stay aligned
 * with other shared-engine tests.
 */

import type { BoardState, GameState, Position } from '../../src/shared/types/game';
import { positionToString } from '../../src/shared/types/game';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
} from '../utils/fixtures';

/**
 * Utility: collapse every board space except the specified position.
 */
function collapseAllExcept(board: BoardState, except: Position, owner: number): void {
  const exceptKey = positionToString(except);

  for (let x = 0; x < board.size; x += 1) {
    for (let y = 0; y < board.size; y += 1) {
      const key = positionToString({ x, y });
      if (key === exceptKey) continue;
      board.collapsedSpaces.set(key, owner);
    }
  }
}

/**
 * Utility: collapse every board space (used for global bare-board stalemate).
 */
function collapseEntireBoard(board: BoardState, owner: number): void {
  for (let x = 0; x < board.size; x += 1) {
    for (let y = 0; y < board.size; y += 1) {
      const key = positionToString({ x, y });
      board.collapsedSpaces.set(key, owner);
    }
  }
}

/**
 * ANM-SCEN-01 – Movement phase, no moves but forced elimination available.
 *
 * Shape:
 * - gameStatus == 'active'
 * - currentPhase == 'movement'
 * - currentPlayer controls exactly one stack on a square8 board
 * - Every other space is collapsed territory
 *   - No legal movement or capture from the stack
 *   - No legal placements (no-dead-placement on a fully collapsed board)
 * - Forced-elimination preconditions hold:
 *   - hasTurnMaterial == true (stack on board)
 *   - hasGlobalPlacementAction == false
 *   - hasAnyGlobalMovementOrCapture == false
 *
 * This state is used to exercise:
 * - Forced-elimination detection in globalActions.hasForcedEliminationAction
 * - ANM classification via isANMState (should be false – FE is a global action)
 * - Movement FSM no_movement_action acceptance (no phase-local moves)
 */
export function makeAnmScen01_MovementNoMovesButFEAvailable(): GameState {
  const board = createTestBoard('square8');
  const origin: Position = { x: 3, y: 3 };

  // Single stack for player 1 at origin.
  addStack(board, origin, 1, 1);

  // Collapse every other space so that:
  // - No simple movement or captures are possible from origin.
  // - No placements are possible anywhere (no-dead-placement on collapsed spaces).
  collapseAllExcept(board, origin, 1);

  const players = [
    createTestPlayer(1, { ringsInHand: 0 }),
    createTestPlayer(2, { ringsInHand: 0 }),
  ];

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'movement',
  });
}

/**
 * ANM-SCEN-02 – Movement phase, placements-only global actions.
 *
 * Shape:
 * - gameStatus == 'active'
 * - currentPhase == 'movement'
 * - currentPlayer has rings in hand but no stacks on the board
 * - get_valid_moves (movement/capture) returns []
 * - hasGlobalPlacementAction == true on an otherwise empty square8 board
 * - No forced elimination (player has no stacks)
 *
 * This fixture encodes a clean placements-only state: the global action
 * surface is non-empty via placements, but MOVEMENT has no phase-local
 * interactive actions.
 */
export function makeAnmScen02_MovementPlacementsOnly(): GameState {
  const board = createTestBoard('square8');

  // No stacks for any player; placements are the only global actions.
  const players = [
    createTestPlayer(1, { ringsInHand: 3 }),
    createTestPlayer(2, { ringsInHand: 3 }),
  ];

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'movement',
  });
}

/**
 * ANM-SCEN-03 – Movement phase with a fully eliminated current player.
 *
 * Shape:
 * - gameStatus == 'active'
 * - currentPhase == 'movement'
 * - currentPlayer P has:
 *   - no stacks (no stack with controllingPlayer == P)
 *   - ringsInHand[P] == 0
 * - Another player Q still has stacks and/or rings in hand
 *
 * Under RR-CANON this is an invalid ACTIVE state: turn rotation must skip
 * fully-eliminated players. From the globalActions perspective we need to
 * ensure:
 * - hasTurnMaterial(state, P) == false
 * - isANMState(state) == false (ANM is only defined for players with material)
 */
export function makeAnmScen03_MovementCurrentPlayerFullyEliminated(): GameState {
  const board = createTestBoard('square8');

  // Give player 2 a single stack so someone still has material.
  addStack(board, { x: 4, y: 4 }, 2, 1);

  const players = [
    // Player 1: fully eliminated for turn rotation (no stacks, no rings).
    createTestPlayer(1, { ringsInHand: 0 }),
    // Player 2: still has material.
    createTestPlayer(2, { ringsInHand: 2 }),
  ];

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'movement',
  });
}

/**
 * ANM-SCEN-04 – Territory processing with no remaining decisions.
 *
 * Shape (TS-side simplification):
 * - gameStatus == 'active'
 * - currentPhase == 'territory_processing'
 * - No disconnected regions for the current player
 * - No pending territory eliminations
 * - Player 1 still has legal placements available
 *
 * This matches the global intent that:
 * - enumerateProcessTerritoryRegionMoves(...) === []
 * - enumerateTerritoryEliminationMoves(...) === []
 * - hasPhaseLocalInteractiveMove(...) === false for territory_processing
 * - G(state, currentPlayer) is non-empty via placements, so isANMState(state) == false
 */
export function makeAnmScen04_TerritoryNoRemainingDecisions(): GameState {
  const board = createTestBoard('square8');

  // One simple stack for player 1 outside any curated territory geometry.
  addStack(board, { x: 0, y: 0 }, 1, 1);

  const players = [
    createTestPlayer(1, { ringsInHand: 2 }),
    createTestPlayer(2, { ringsInHand: 2 }),
  ];

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'territory_processing',
  });
}

/**
 * ANM-SCEN-05 – Line processing with no remaining decisions.
 *
 * Shape (TS-side simplification):
 * - gameStatus == 'active'
 * - currentPhase == 'line_processing'
 * - No lines requiring processing or reward choice for currentPlayer
 * - Player 1 still has legal placements available
 *
 * This ensures:
 * - enumerateProcessLineMoves(...) === []
 * - hasPhaseLocalInteractiveMove(...) === false for line_processing
 * - Global action surface is non-empty (placements), so isANMState(state) == false.
 */
export function makeAnmScen05_LineProcessingNoRemainingDecisions(): GameState {
  const board = createTestBoard('square8');

  // No markers/lines; only placements are available globally.
  const players = [
    createTestPlayer(1, { ringsInHand: 2 }),
    createTestPlayer(2, { ringsInHand: 2 }),
  ];

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'line_processing',
  });
}

/**
 * ANM-SCEN-06 – Global stalemate (no actions for any player).
 *
 * Shape:
 * - gameStatus == 'active' (pre-termination snapshot)
 * - No stacks on the board
 * - All board spaces are collapsed territory
 * - Both players still have rings in hand, but:
 *   - hasGlobalPlacementAction(...) == false for every player
 *   - hasForcedEliminationAction(...) == false for every player
 *
 * This is the bare-board structural stalemate used by victoryLogic.evaluateVictory
 * to apply the §13.4 stalemate ladder (territory → eliminated+hand → markers → last actor).
 *
 * NOTE: From the ANM invariant perspective, hosts / RuleEngine should invoke
 * victory evaluation before treating this as an ANM violation; the fixture is
 * intended for TS-side victory + global-actions tests.
 */
export function makeAnmScen06_GlobalStalemateBareBoard(): GameState {
  const board = createTestBoard('square8');

  // Ensure there are no stacks or markers at all.
  board.stacks.clear();
  board.markers.clear();

  // Collapse every space on the board so that no legal placement exists.
  collapseEntireBoard(board, 1);

  const players = [
    // Player 1: more rings in hand for stalemate ladder tie-break.
    createTestPlayer(1, { ringsInHand: 3, eliminatedRings: 0, territorySpaces: 0 }),
    createTestPlayer(2, { ringsInHand: 1, eliminatedRings: 0, territorySpaces: 0 }),
  ];

  // Make sure eliminatedRings map is aligned with players for victory logic.
  board.eliminatedRings[1] = 0;
  board.eliminatedRings[2] = 0;

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    // Use a non-participating currentPlayer so that isANMState(state) is false
    // (hasTurnMaterial(state, currentPlayer) === false) while the stalemate
    // ladder in victoryLogic still evaluates over players[1..N].
    currentPlayer: 3,
    currentPhase: 'ring_placement',
  });
}

/**
 * Helper fixture for the mustMoveFromStackKey edge case:
 *
 * - Two stacks for the current player:
 *   - A "stuck" stack at (3,3) surrounded by collapsed spaces (no legal moves).
 *   - A "free" stack at (6,6) with open space around it.
 * - mustMoveFromStackKey is set to the stuck stack's key.
 *
 * This ensures:
 * - hasAnyGlobalMovementOrCapture(state, player) == false when constrained
 *   by mustMoveFromStackKey, even though:
 * - hasAnyGlobalMovementOrCapture(stateWithoutConstraint, player) == true
 *   (the free stack could move if unconstrained).
 *
 * Global ANM semantics are still non-empty because placements remain available.
 */
export function makeMovementMustMoveFromStackKeyConstrainedState(): GameState {
  const board = createTestBoard('square8');

  const stuck: Position = { x: 3, y: 3 };
  const free: Position = { x: 6, y: 6 };

  addStack(board, stuck, 1, 1);
  addStack(board, free, 1, 1);

  // Surround the stuck stack with collapsed spaces so it has no legal movement/capture.
  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      if (dx === 0 && dy === 0) continue;
      const nx = stuck.x + dx;
      const ny = stuck.y + dy;
      if (nx < 0 || nx >= board.size || ny < 0 || ny >= board.size) {
        continue;
      }
      const key = positionToString({ x: nx, y: ny });
      board.collapsedSpaces.set(key, 1);
    }
  }

  const players = [
    // Player 1: has rings in hand so placements are globally available.
    createTestPlayer(1, { ringsInHand: 2 }),
    createTestPlayer(2, { ringsInHand: 2 }),
  ];

  const mustMoveFromStackKey = positionToString(stuck);

  return createTestGameState({
    boardType: 'square8',
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'movement',
    // Constrain movement for this turn to the stuck stack only.
    mustMoveFromStackKey,
  });
}

/**
 * ANM-SCEN-07 / ANM-SCEN-08 sequence-style fixtures are optional for this TS
 * subtask. They are stubbed here as empty sequences to make the exports
 * available to future parity / LPS tasks without committing to a specific
 * board geometry in this change.
 *
 * TODO (ANM-SCEN-07/08): Populate these sequences with concrete GameState
 * trajectories once the TS↔Python LPS parity harness is wired up.
 */
export function makeAnmScen07_LpsRealActionsOnly(): GameState[] {
  return [];
}

export function makeAnmScen08_MultiPlayerRotationEliminatedPlayers(): GameState[] {
  return [];
}
