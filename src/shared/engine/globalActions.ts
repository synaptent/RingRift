import type { GameState, GamePhase, BoardType, Position, Move } from '../types/game';
import { positionToString } from '../types/game';
import {
  MovementBoardView,
  hasAnyLegalMoveOrCaptureFromOnBoard,
  computeProgressSnapshot,
} from './core';
import {
  enumeratePlacementPositions,
  evaluateSkipPlacementEligibility,
  SkipPlacementEligibilityResult,
} from './aggregates/PlacementAggregate';
import { enumerateProcessLineMoves } from './aggregates/LineAggregate';
import {
  enumerateProcessTerritoryRegionMoves,
  enumerateTerritoryEliminationMoves,
} from './aggregates/TerritoryAggregate';
import { applyEliminateRingsFromStackDecision } from './territoryDecisionHelpers';
import { isValidPosition } from './validators/utils';

/**
 * Summary of the global legal action surface G(state, P) for a player.
 *
 * This is a thin TS mirror of the RR-CANON R200–R203 definitions used by the
 * invariants framework:
 *
 * - hasTurnMaterial          – player is not fully eliminated (R201).
 * - hasGlobalPlacementAction – at least one legal ring placement exists,
 *                              ignoring currentPhase (R200).
 * - hasPhaseLocalInteractiveMove – at least one interactive move is legal in
 *                                  the current phase (placement/skip, movement/
 *                                  capture, line or territory decisions) (R200).
 * - hasForcedEliminationAction – R205/R072/R100 preconditions hold: player is
 *                                blocked with stacks, no placements or
 *                                movement/capture exist, and host-level forced
 *                                elimination is available.
 */
export interface GlobalLegalActionsSummary {
  hasTurnMaterial: boolean;
  hasGlobalPlacementAction: boolean;
  hasPhaseLocalInteractiveMove: boolean;
  hasForcedEliminationAction: boolean;
}

/**
 * True when the specified player still has turn-material in the sense of
 * RR-CANON-R201: at least one controlled stack or at least one ring in hand.
 */
export function hasTurnMaterial(state: GameState, player: number): boolean {
  const playerState = state.players.find((p) => p.playerNumber === player);
  if (!playerState) {
    return false;
  }

  if (playerState.ringsInHand > 0) {
    return true;
  }

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer === player && stack.stackHeight > 0) {
      return true;
    }
  }

  return false;
}

/**
 * True when at least one legal ring placement exists for `player` in `state`,
 * respecting:
 *
 * - own-colour supply caps (ringsPerPlayer),
 * - ringsInHand,
 * - board geometry and collapsed spaces, and
 * - the no-dead-placement rule via validatePlacementOnBoard.
 *
 * This helper is **phase-agnostic**: it ignores state.currentPhase and can be
 * used from movement/territory/victory code as required by R200.
 */
export function hasGlobalPlacementAction(state: GameState, player: number): boolean {
  const positions = enumeratePlacementPositions(state, player);
  return positions.length > 0;
}

/**
 * Internal helper: build a MovementBoardView over the current GameState so we
 * can reuse hasAnyLegalMoveOrCaptureFromOnBoard without duplicating geometry.
 */
function createMovementBoardView(state: GameState): MovementBoardView {
  const board = state.board;
  const boardType = board.type as BoardType;
  const size = board.size;

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) {
        return undefined;
      }
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const key = positionToString(pos);
      const marker = board.markers.get(key);
      return marker?.player;
    },
  };
}

/**
 * Internal helper: true when **any** non-capture movement or overtaking
 * capture exists for `player` somewhere on the board, irrespective of
 * state.currentPhase. This is used when evaluating forced-elimination
 * preconditions (R072/R100/R205).
 */
function hasAnyGlobalMovementOrCapture(state: GameState, player: number): boolean {
  const boardType = state.board.type as BoardType;
  const view = createMovementBoardView(state);

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }

    if (hasAnyLegalMoveOrCaptureFromOnBoard(boardType, stack.position, player, view)) {
      return true;
    }
  }

  return false;
}

/**
 * True when the **active** player has at least one phase-local interactive
 * move in the current phase (R200, R204):
 *
 * - ring_placement:
 *   - place_ring anywhere that validatePlacementOnBoard accepts, or
 *   - skip_placement when the aggregate reports it as legal.
 * - movement / capture / chain_capture:
 *   - at least one legal non-capturing move or overtaking capture from any
 *     controlled stack.
 * - line_processing:
 *   - at least one process_line or choose_line_reward decision.
 * - territory_processing:
 *   - at least one process_territory_region or eliminate_rings_from_stack
 *     decision.
 *
 * For non-active players this helper always returns false.
 */
export function hasPhaseLocalInteractiveMove(state: GameState, player: number): boolean {
  if (state.gameStatus !== 'active') {
    return false;
  }
  if (player !== state.currentPlayer) {
    return false;
  }

  const phase: GamePhase = state.currentPhase;

  switch (phase) {
    case 'ring_placement': {
      if (hasGlobalPlacementAction(state, player)) {
        return true;
      }
      const eligibility: SkipPlacementEligibilityResult = evaluateSkipPlacementEligibility(
        state,
        player
      );
      return eligibility.eligible;
    }

    case 'movement':
    case 'capture':
    case 'chain_capture': {
      return hasAnyGlobalMovementOrCapture(state, player);
    }

    case 'line_processing': {
      const moves = enumerateProcessLineMoves(state, player, {
        detectionMode: 'detect_now',
      });
      return moves.length > 0;
    }

    case 'territory_processing': {
      const regionMoves = enumerateProcessTerritoryRegionMoves(state, player);
      const elimMoves = enumerateTerritoryEliminationMoves(state, player);
      return regionMoves.length > 0 || elimMoves.length > 0;
    }

    default:
      return false;
  }
}

/**
 * True when RR-CANON forced-elimination preconditions hold for `player` in
 * `state` (R072, R100, R205):
 *
 * - Game is ACTIVE.
 * - Player controls at least one stack (has material on board).
 * - Player has **no** legal placements (hasGlobalPlacementAction === false).
 * - Player has **no** legal non-capture movement or overtaking capture from
 *   any controlled stack (hasAnyGlobalMovementOrCapture === false).
 *
 * This predicate is intentionally phase-agnostic and may be used by hosts at
 * start-of-turn, during decision-phase exits, or by invariants/soaks.
 * It does **not** perform the elimination; hosts should call a shared
 * elimination helper to actually remove the chosen cap.
 */
export function hasForcedEliminationAction(state: GameState, player: number): boolean {
  if (state.gameStatus !== 'active') {
    return false;
  }

  const playerState = state.players.find((p) => p.playerNumber === player);
  if (!playerState) {
    return false;
  }

  // Must control at least one stack on the board.
  let hasStacks = false;
  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer === player && stack.stackHeight > 0) {
      hasStacks = true;
      break;
    }
  }
  if (!hasStacks) {
    return false;
  }

  const anyPlacement = hasGlobalPlacementAction(state, player);
  const anyMoveOrCapture = hasAnyGlobalMovementOrCapture(state, player);

  return !anyPlacement && !anyMoveOrCapture;
}

/**
 * A single stack that is eligible for forced elimination.
 */
export interface ForcedEliminationOption {
  /** Position of the stack. */
  position: Position;
  /** Height of the current player's cap on this stack. */
  capHeight: number;
  /** Total stack height. */
  stackHeight: number;
  /** Move ID for the corresponding eliminate_rings_from_stack move. */
  moveId: string;
}

/**
 * Enumerate all stacks that are eligible for forced elimination for a player.
 *
 * When a player is blocked (has stacks but no legal placement, movement, or
 * capture actions), they must eliminate a cap from one of their stacks. This
 * function returns all eligible stacks so hosts can present a choice to the
 * player (for human players) or auto-select (for AI).
 *
 * Returns an empty array if the player is not in a forced elimination state.
 */
export function enumerateForcedEliminationOptions(
  state: GameState,
  player: number
): ForcedEliminationOption[] {
  if (!hasForcedEliminationAction(state, player)) {
    return [];
  }

  const options: ForcedEliminationOption[] = [];

  for (const stack of state.board.stacks.values()) {
    if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
      continue;
    }
    const capHeight: number = typeof stack.capHeight === 'number' ? stack.capHeight : 0;

    options.push({
      position: stack.position,
      capHeight,
      stackHeight: stack.stackHeight,
      moveId: `forced-elim-${positionToString(stack.position)}`,
    });
  }

  return options;
}

/**
 * Canonical host-level forced-elimination operator aligned with RR-CANON-R205.
 *
 * This helper is intentionally small and purely functional so that:
 * - turn/phase orchestrators (shared or host-specific) can invoke a single
 *   implementation when a player is blocked with stacks (R072/R100/R205), and
 * - invariant/soak harnesses can apply or simulate forced elimination without
 *   re-deriving stack-selection or elimination bookkeeping.
 *
 * Preconditions (mirroring hasForcedEliminationAction):
 * - state.gameStatus === 'active'
 * - player controls at least one stack on the board
 * - player has no legal placements and no legal movement/capture actions
 *
 * The implementation:
 * - If targetPosition is provided, uses that stack for elimination.
 * - Otherwise, selects a stack controlled by `player`, preferring the smallest
 *   positive capHeight and falling back to the first stack when no caps exist.
 * - Eliminates that stack's cap via applyEliminateRingsFromStackDecision; and
 * - Returns the updated GameState along with basic accounting metadata so
 *   callers can assert INV-ELIMINATION-MONOTONIC / INV-S-MONOTONIC.
 *
 * @param state - Current game state
 * @param player - Player who must eliminate
 * @param targetPosition - Optional: specific stack position chosen by player.
 *                         If not provided, auto-selects (smallest cap heuristic).
 */
export interface ForcedEliminationOutcome {
  /** Updated state after applying forced elimination. */
  nextState: GameState;
  /** Player who paid the elimination cost (always `player`). */
  eliminatedPlayer: number;
  /** Position of the stack that was chosen for elimination, if any. */
  eliminatedFrom?: Position;
  /** Number of rings eliminated from the chosen stack. */
  eliminatedCount: number;
}

export function applyForcedEliminationForPlayer(
  state: GameState,
  player: number,
  targetPosition?: Position
): ForcedEliminationOutcome | undefined {
  if (!hasForcedEliminationAction(state, player)) {
    return undefined;
  }

  let chosenStack:
    | {
        position: Position;
        capHeight: number;
        stackHeight: number;
      }
    | undefined;

  // If a specific target position was provided (player made a choice), use it
  if (targetPosition) {
    const targetKey = positionToString(targetPosition);
    const stack = state.board.stacks.get(targetKey);
    if (stack && stack.controllingPlayer === player && stack.stackHeight > 0) {
      const capHeight: number = typeof stack.capHeight === 'number' ? stack.capHeight : 0;
      chosenStack = {
        position: stack.position,
        capHeight,
        stackHeight: stack.stackHeight,
      };
    }
  }

  // Auto-select if no target provided or target was invalid
  if (!chosenStack) {
    let smallestCap = Number.POSITIVE_INFINITY;

    for (const stack of state.board.stacks.values()) {
      if (stack.controllingPlayer !== player || stack.stackHeight <= 0) {
        continue;
      }
      const capHeight: number = typeof stack.capHeight === 'number' ? stack.capHeight : 0;

      if (capHeight > 0 && capHeight < smallestCap) {
        smallestCap = capHeight;
        chosenStack = {
          position: stack.position,
          capHeight,
          stackHeight: stack.stackHeight,
        };
      } else if (!chosenStack) {
        // Fallback: remember the first stack so that we always have a target
        // even when all caps are zero in legacy or degenerate states.
        chosenStack = {
          position: stack.position,
          capHeight,
          stackHeight: stack.stackHeight,
        };
      }
    }
  }

  if (!chosenStack) {
    return undefined;
  }

  const move: Move = {
    id: `forced-elim-${positionToString(chosenStack.position)}`,
    type: 'eliminate_rings_from_stack',
    player,
    to: chosenStack.position,
    eliminatedRings: [{ player, count: Math.max(1, chosenStack.capHeight || 0) }],
    eliminationFromStack: {
      position: chosenStack.position,
      capHeight: chosenStack.capHeight,
      totalHeight: chosenStack.stackHeight,
    },
    // Deterministic placeholders – callers should not rely on these fields.
    timestamp: new Date(0),
    thinkTime: 0,
    moveNumber: 0,
  } as Move;

  const beforeTotal =
    (state as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ?? 0;

  const { nextState } = applyEliminateRingsFromStackDecision(state, move);

  const afterTotal =
    (nextState as GameState & { totalRingsEliminated?: number }).totalRingsEliminated ?? 0;

  const eliminatedDelta =
    afterTotal > beforeTotal ? afterTotal - beforeTotal : Math.max(1, chosenStack.capHeight || 0);

  return {
    nextState,
    eliminatedPlayer: player,
    eliminatedFrom: chosenStack.position,
    eliminatedCount: eliminatedDelta,
  };
}

/**
 * Compute the R200 global legal action summary for a given player.
 */
export function computeGlobalLegalActionsSummary(
  state: GameState,
  player: number
): GlobalLegalActionsSummary {
  const material = hasTurnMaterial(state, player);
  const placements = hasGlobalPlacementAction(state, player);
  const phaseMoves = hasPhaseLocalInteractiveMove(state, player);
  const forcedElim = hasForcedEliminationAction(state, player);

  return {
    hasTurnMaterial: material,
    hasGlobalPlacementAction: placements,
    hasPhaseLocalInteractiveMove: phaseMoves,
    hasForcedEliminationAction: forcedElim,
  };
}

/**
 * ANM(state) predicate for the active player, mirroring RR-CANON-R202/R203:
 *
 * ANM(state) holds iff:
 * - state.gameStatus === 'active', and
 * - currentPlayer has turn-material (R201), and
 * - G(state, currentPlayer) is empty:
 *   - no global placements,
 *   - no phase-local interactive moves, and
 *   - no forced-elimination action available.
 *
 * Hosts and invariant harnesses can use this helper to enforce
 * INV-ACTIVE-NO-MOVES and INV-ANM-TURN-MATERIAL-SKIP.
 */
export function isANMState(state: GameState): boolean {
  if (state.gameStatus !== 'active') {
    return false;
  }

  const player = state.currentPlayer;
  if (!hasTurnMaterial(state, player)) {
    return false;
  }

  const summary = computeGlobalLegalActionsSummary(state, player);

  return (
    !summary.hasGlobalPlacementAction &&
    !summary.hasPhaseLocalInteractiveMove &&
    !summary.hasForcedEliminationAction
  );
}

/**
 * Convenience wrappers for S/T-style progress metrics used by the
 * invariants framework. S matches computeProgressSnapshot(state).S; T is
 * currently defined as "collapsed + eliminated" so that:
 *
 * - S(state) = markers + collapsed + eliminated (INV-S-MONOTONIC), and
 * - T(state) = collapsed + eliminated (useful for decision-phase chains
 *   that may leave markers unchanged).
 *
 * These helpers are intentionally small so that invariant tests and soaks
 * do not have to re-derive metric definitions.
 */
export function computeSMetric(state: GameState): number {
  return computeProgressSnapshot(state).S;
}

export function computeTMetric(state: GameState): number {
  const snapshot = computeProgressSnapshot(state);
  return snapshot.collapsed + snapshot.eliminated;
}
