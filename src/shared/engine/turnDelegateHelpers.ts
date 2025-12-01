import type { GameState, BoardType, Position } from '../types/game';
import { BOARD_CONFIGS } from '../types/game';
import type { PerTurnState, TurnLogicDelegates } from './turnLogic';
import { countRingsOnBoardForPlayer } from './core';
import { validatePlacementOnBoard } from './validators/PlacementValidator';

/**
 * Shared helpers and factory for {@link TurnLogicDelegates} used by the
 * phase/turn sequencer in [`turnLogic.ts`](src/shared/engine/turnLogic.ts:1).
 *
 * The goal of this module is to:
 *
 * - Document a **canonical** way to answer the three action-availability
 *   questions used by the sequencer:
 *   - "Does this player have any legal placement?"
 *   - "Does this player have any legal non-capturing movement?"
 *   - "Does this player have any legal overtaking capture?"
 * - Provide a small factory that backend and sandbox hosts can use to build
 *   {@link TurnLogicDelegates} instances that share the same underlying
 *   reachability semantics instead of each host maintaining its own
 *   `hasValidMovements` / `hasValidCaptures` implementations.
 *
 * For P0 Task #21 these functions are intentionally left as **design-time
 * stubs**: they encode the intended signatures and semantics in docstrings
 * but are not yet wired into any host engine, so they do not change runtime
 * behaviour. Later tasks (P0 #7–#9) will port the concrete logic from:
 *
 * - Backend:
 *   - `GameEngine.hasValidPlacements`,
 *   - `GameEngine.hasValidMovements`,
 *   - `GameEngine.hasValidCaptures`, and
 *   - `TurnEngine.advanceGameForCurrentPlayer` forced-elimination helpers.
 * - Sandbox:
 *   - `sandboxTurnEngine.hasAnyPlacementForCurrentPlayer`,
 *   - `sandboxTurnEngine.hasAnyMovementForCurrentPlayer`,
 *   - `sandboxTurnEngine.hasAnyCaptureForCurrentPlayer`, and
 *   - `sandboxTurnEngine.maybeProcessForcedEliminationForCurrentPlayer`.
 */

/**
 * Canonical "has any placement?" predicate for the shared turn/phase
 * sequencer.
 *
 * Target semantics (to be implemented in a future task):
 *
 * - Only placements that satisfy the no-dead-placement rule encoded by
 *   [`validatePlacementOnBoard`](src/shared/engine/validators/PlacementValidator.ts:76)
 *   are counted as "legal".
 * - Ring supply and per-player caps (BOARD_CONFIGS.ringsPerPlayer) are
 *   honoured.
 * - The current phase is **not** considered here; hosts remain responsible
 *   for calling this only when placement is relevant (usually from
 *   `'ring_placement'` or at turn start).
 *
 * This helper is pure and must not mutate `state`.
 */
export function hasAnyPlacementForPlayer(state: GameState, player: number): boolean {
  const playerObj = state.players.find((p) => p.playerNumber === player);
  if (!playerObj || playerObj.ringsInHand <= 0) {
    return false;
  }

  const board = state.board;
  const boardConfig = BOARD_CONFIGS[board.type as BoardType];

  if (!boardConfig) {
    return false;
  }

  // Global own-colour supply cap: if the player has no remaining capacity
  // under ringsPerPlayer, no placements are possible regardless of board
  // geometry.
  const ringsOnBoard = countRingsOnBoardForPlayer(board as any, player);
  const remainingByCap = boardConfig.ringsPerPlayer - ringsOnBoard;
  const remainingBySupply = playerObj.ringsInHand;
  const maxAvailableGlobal = Math.min(remainingByCap, remainingBySupply);

  if (maxAvailableGlobal <= 0) {
    return false;
  }

  const baseContext = {
    boardType: board.type as BoardType,
    player,
    ringsInHand: playerObj.ringsInHand,
    ringsPerPlayerCap: boardConfig.ringsPerPlayer,
    ringsOnBoard,
    maxAvailableGlobal,
  };

  const hasPlacementAt = (pos: Position): boolean => {
    const result = validatePlacementOnBoard(board as any, pos, 1, baseContext);
    return result.valid;
  };

  if (board.type === 'square8' || board.type === 'square19') {
    const size = board.size;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const pos: Position = { x, y };
        if (hasPlacementAt(pos)) {
          return true;
        }
      }
    }
    return false;
  }

  if (board.type === 'hexagonal') {
    // For hex boards, iterate over all cube coordinates within the board
    // radius. validatePlacementOnBoard will discard off-board positions.
    const radius = board.size - 1;
    for (let q = -radius; q <= radius; q += 1) {
      for (let r = -radius; r <= radius; r += 1) {
        const s = -q - r;
        const pos: Position = { x: q, y: r, z: s };
        if (hasPlacementAt(pos)) {
          return true;
        }
      }
    }
    return false;
  }

  // Unknown board type: conservatively report no placements.
  return false;
}

/**
 * Canonical "has any non-capturing movement?" predicate for the shared
 * turn/phase sequencer.
 *
 * Target semantics:
 *
 * - Returns true if at least one legal **non-capturing** stack move exists
 *   for `player` in `state`, respecting:
 *   - board geometry and collapsed spaces,
 *   - stack ownership and height rules, and
 *   - any per-turn constraints encoded in {@link PerTurnState}
 *     (e.g. must-move-from-stack requirements).
 * - Capture opportunities are **not** considered here; those are handled by
 *   {@link hasAnyCaptureForPlayer}.
 *
 * At implementation time this helper is expected to:
 *
 * - Build a lightweight {@code MovementBoardView} from `state.board`, and
 * - Delegate the actual reachability checks to either:
 *   - [`enumerateSimpleMoveTargetsFromStack`](src/shared/engine/movementLogic.ts:55), or
 *   - [`hasAnyLegalMoveOrCaptureFromOnBoard`](src/shared/engine/core.ts:367) with capture
 *     reachability disabled.
 */
export function hasAnyMovementForPlayer(
  state: GameState,
  player: number,
  turn: PerTurnState
): boolean {
  void turn;
  throw new Error(
    'TODO(P0-HELPERS): hasAnyMovementForPlayer is a design-time stub. ' +
      'See P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md for intended semantics.'
  );
}

/**
 * Canonical "has any overtaking capture?" predicate for the shared
 * turn/phase sequencer.
 *
 * Target semantics:
 *
 * - Returns true if at least one legal overtaking capture (initial or
 *   continuation) exists for `player` in `state`, respecting any per-turn
 *   constraints encoded in {@link PerTurnState}.
 * - Builds on the same capture geometry used everywhere else via
 *   [`enumerateCaptureMoves`](src/shared/engine/captureLogic.ts:26) and
 *   [`validateCaptureSegmentOnBoard`](src/shared/engine/core.ts:202).
 *
 * As with {@link hasAnyMovementForPlayer}, this helper is pure and must not
 * mutate `state`.
 */
export function hasAnyCaptureForPlayer(
  state: GameState,
  player: number,
  turn: PerTurnState
): boolean {
  void turn;
  throw new Error(
    'TODO(P0-HELPERS): hasAnyCaptureForPlayer is a design-time stub. ' +
      'See P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md for intended semantics.'
  );
}

/**
 * Configuration required to build a {@link TurnLogicDelegates} instance using
 * the shared action-availability helpers defined in this module.
 *
 * Hosts remain responsible for:
 *
 * - how player numbers advance in seat order, and
 * - how forced-elimination is applied (which stack to eliminate from,
 *   whether to allow elimination-from-hand, when to run victory checks, etc.).
 */
export interface DefaultTurnDelegatesConfig {
  /**
   * Host-specific implementation of the delegate used by
   * [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135) to rotate
   * the active player number in seat order.
   */
  getNextPlayerNumber: TurnLogicDelegates['getNextPlayerNumber'];

  /**
   * Host-specific implementation of forced elimination for a blocked player.
   * This is threaded through without modification so that:
   *
   * - the backend can continue using its existing elimination logic wrapped
   *   around `GameEngine.eliminatePlayerRingOrCap`, and
   * - the sandbox can keep its UI-/history-specific elimination behaviour.
   */
  applyForcedElimination: TurnLogicDelegates['applyForcedElimination'];
}

/**
 * Build a {@link TurnLogicDelegates} instance that:
 *
 * - Answers `hasAnyPlacement`, `hasAnyMovement`, and `hasAnyCapture` via the
 *   shared predicates declared in this module, and
 * - Defers `getNextPlayerNumber` and `applyForcedElimination` to host
 *   implementations supplied via {@link DefaultTurnDelegatesConfig}.
 *
 * This factory is intended as the long-term entry point for both:
 *
 * - the backend TurnEngine (wrapping its existing next-player / forced-
 *   elimination semantics), and
 * - the sandbox turn engine, once both are refactored to drive the shared
 *   [`advanceTurnAndPhase`](src/shared/engine/turnLogic.ts:135) helper.
 *
 * For P0 Task #21 the returned delegates will still throw at runtime because
 * they call the stubbed action-availability helpers above. Hosts are **not**
 * yet wired to use this factory; doing so is part of later refactor tasks.
 */
export function createDefaultTurnLogicDelegates(
  config: DefaultTurnDelegatesConfig
): TurnLogicDelegates {
  return {
    getPlayerStacks: (state: GameState, player: number) => {
      // Minimal adapter returning just the fields required by the sequencer.
      const stacks: Array<{ position: any; stackHeight: number }> = [];
      for (const stack of state.board.stacks.values()) {
        if (stack.controllingPlayer !== player) continue;
        stacks.push({ position: stack.position, stackHeight: stack.stackHeight });
      }
      return stacks;
    },
    hasAnyPlacement: (state: GameState, player: number) => hasAnyPlacementForPlayer(state, player),
    hasAnyMovement: (state: GameState, player: number, turn: PerTurnState) =>
      hasAnyMovementForPlayer(state, player, turn),
    hasAnyCapture: (state: GameState, player: number, turn: PerTurnState) =>
      hasAnyCaptureForPlayer(state, player, turn),
    applyForcedElimination: config.applyForcedElimination,
    getNextPlayerNumber: config.getNextPlayerNumber,
  };
}
