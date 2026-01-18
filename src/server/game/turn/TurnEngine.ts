import type {
  GameState,
  Move,
  GameResult,
  Position,
  PerTurnState as SharedPerTurnState,
  TurnLogicDelegates,
  LpsTrackingState,
} from '../../../shared/engine';
import {
  advanceTurnAndPhase,
  applyForcedEliminationForPlayer,
  evaluateLpsVictory,
  updateLpsTracking,
  // Shared action-availability predicates (canonical implementations)
  hasAnyPlacementForPlayer,
  hasAnyMovementForPlayer,
  hasAnyCaptureForPlayer,
  playerHasAnyRings,
} from '../../../shared/engine';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';
import { flagEnabled, debugLog } from '../../../shared/utils/envFlags';

/**
 * Dependencies required for turn/phase orchestration. This keeps the
 * turn engine decoupled from the concrete GameEngine class while still
 * allowing it to inspect board geometry and rules.
 */
export interface TurnEngineDeps {
  boardManager: BoardManager;
  ruleEngine: RuleEngine;
}

/**
 * Internal per-turn state for the backend engine. This is a thin alias
 * of the shared engine PerTurnState so that backend GameEngine and the
 * shared turnLogic helper stay in sync.
 */
export type PerTurnState = SharedPerTurnState;

/**
 * Hooks that let the turn engine delegate elimination and game-end
 * side effects back to the owning GameEngine without depending on its
 * concrete class shape.
 */
export interface TurnEngineHooks {
  eliminatePlayerRingOrCap: (playerNumber: number, stackPosition?: Position) => void;
  endGame: (winner?: number, reason?: string) => { success: boolean; gameResult: GameResult };
  /** Accessors for LPS tracking state (host-owned, e.g., GameEngine) */
  getLpsState: () => LpsTrackingState;
  setLpsState: (next: LpsTrackingState) => void;
  /** LPS helpers from host */
  hasAnyRealActionForPlayer: (playerNumber: number) => boolean;
  hasMaterialForPlayer: (playerNumber: number) => boolean;
}

/**
 * Update internal per-turn placement/movement bookkeeping after a move
 * has been applied. This keeps the must-move origin in sync with the
 * stack that was placed or moved, mirroring the sandbox engine’s
 * behaviour while keeping these details off of GameState.
 *
 * This is a direct extraction of GameEngine.updatePerTurnStateAfterMove
 * rewritten in functional style.
 */
export function updatePerTurnStateAfterMove(turnState: PerTurnState, move: Move): PerTurnState {
  let { hasPlacedThisTurn, mustMoveFromStackKey } = turnState;

  // When a ring is placed, mark that we have placed this turn and
  // record which stack must be moved. The updated stack always
  // resides at move.to (either an empty cell or an existing stack).
  if (move.type === 'place_ring' && move.to) {
    hasPlacedThisTurn = true;
    mustMoveFromStackKey = positionToStringLocal(move.to);
    return { hasPlacedThisTurn, mustMoveFromStackKey };
  }

  // For movement/capture moves, track the landing position so subsequent
  // captures in this turn are restricted to originate from that stack.
  // RR-CANON-R093: "you may choose to initiate an Overtaking capture only
  // from the stack that just moved"
  if (
    move.from &&
    move.to &&
    (move.type === 'move_stack' ||
      move.type === 'overtaking_capture' ||
      move.type === 'continue_capture_segment')
  ) {
    const fromKey = positionToStringLocal(move.from);
    // If mustMoveFromStackKey was set (from placement), only update if
    // this move originates from that stack. Otherwise, always set it to
    // the landing position to enforce capture eligibility per RR-CANON-R093.
    if (!mustMoveFromStackKey || fromKey === mustMoveFromStackKey) {
      mustMoveFromStackKey = positionToStringLocal(move.to);
    }
  }

  return { hasPlacedThisTurn, mustMoveFromStackKey };
}

/**
 * Advance game through phases according to RingRift rules for the
 * current player.
 *
 * This wrapper delegates the core phase/turn sequencing to the shared
 * engine helper {@link advanceTurnAndPhase} so that backend GameEngine,
 * the shared reference engine, and the client sandbox all use the same
 * canonical state machine. Backend-specific concerns (forced
 * elimination details and victory evaluation) remain here via the
 * delegates.
 */
export function advanceGameForCurrentPlayer(
  gameState: GameState,
  turnState: PerTurnState,
  deps: TurnEngineDeps,
  hooks: TurnEngineHooks
): PerTurnState {
  const delegates: TurnLogicDelegates = {
    getPlayerStacks: (state, player) => deps.boardManager.getPlayerStacks(state.board, player),
    // Use shared canonical predicates from turnDelegateHelpers.ts
    hasAnyPlacement: (state, player) => hasAnyPlacementForPlayer(state, player),
    hasAnyMovement: (state, player, turn) => hasAnyMovementForPlayer(state, player, turn),
    hasAnyCapture: (state, player, turn) => hasAnyCaptureForPlayer(state, player, turn),
    applyForcedElimination: (state, player) => {
      // Reuse the existing forced-elimination helper and backend victory
      // evaluator so that shared turnLogic observes exactly the same
      // semantics as the legacy TurnEngine branch.
      processForcedElimination(state, deps, hooks, player);

      const gameEndCheck = deps.ruleEngine.checkGameEnd(state);
      if (gameEndCheck.isGameOver) {
        hooks.endGame(gameEndCheck.winner, gameEndCheck.reason || 'forced_elimination');
      }

      return state;
    },
    getNextPlayerNumber: (state, current) => {
      const currentIndex = state.players.findIndex((p) => p.playerNumber === current);
      const nextIndex = (currentIndex + 1) % state.players.length;
      return state.players[nextIndex].playerNumber;
    },
    // Use shared canonical predicate from globalActions.ts
    playerHasAnyRings: (state, player) => playerHasAnyRings(state, player),
  };

  const beforeSnapshot = {
    currentPlayer: gameState.currentPlayer,
    currentPhase: gameState.currentPhase,
    gameStatus: gameState.gameStatus,
  };

  const { nextState, nextTurn } = advanceTurnAndPhase(gameState, turnState, delegates);

  const afterSnapshot = {
    currentPlayer: nextState.currentPlayer,
    currentPhase: nextState.currentPhase,
    gameStatus: nextState.gameStatus,
  };

  debugLog(flagEnabled('RINGRIFT_TRACE_DEBUG'), '[TurnTrace.backend.advanceGameForCurrentPlayer]', {
    decision: 'advanceGameForCurrentPlayer',
    reason: 'advanceTurnAndPhase',
    before: beforeSnapshot,
    after: afterSnapshot,
  });

  // Mutate the provided GameState reference in-place so callers that
  // hold onto `gameState` (notably backend GameEngine) observe the
  // updated phase/player fields and any forced-elimination effects
  // without replacing their internal pointer.
  Object.assign(gameState, nextState);

  // RR-PARITY-FIX-2025-12-20: LPS timing parity with Python.
  // Python evaluates LPS at turn START:
  //   1. _update_lps_round_tracking_for_current_player() - updates tracking FIRST
  //   2. _maybe_apply_lps_victory_at_turn_start() - THEN evaluates LPS victory
  // We must update tracking BEFORE evaluating victory to ensure the
  // consecutiveExclusiveRounds counter is current when checking the threshold.
  //
  // This matches ClientSandboxEngine.handleStartOfInteractiveTurn() which also
  // calls updateLpsRoundTrackingForCurrentPlayer() before maybeEndGameByLastPlayerStanding().

  // Update LPS tracking state first
  const lpsState = hooks.getLpsState();
  updateLpsTracking(lpsState, {
    currentPlayer: gameState.currentPlayer,
    activePlayers: gameState.players.map((p) => p.playerNumber),
    hasRealAction: hooks.hasAnyRealActionForPlayer(gameState.currentPlayer),
  });
  // State was mutated in place - persist it back
  hooks.setLpsState(lpsState);

  // Now evaluate Last Player Standing (R172) using shared helpers.
  const lpsResult = evaluateLpsVictory({
    gameState,
    lps: hooks.getLpsState(),
    hasAnyRealAction: (pn) => hooks.hasAnyRealActionForPlayer(pn),
    hasMaterial: (pn) => hooks.hasMaterialForPlayer(pn),
  });

  if (lpsResult.isVictory) {
    hooks.endGame(lpsResult.winner, 'last_player_standing');
  }

  return nextTurn as PerTurnState;
}

// Local action-availability helpers removed - TurnEngine now uses shared canonical
// predicates from turnDelegateHelpers.ts:
// - hasValidCaptures → hasAnyCaptureForPlayer
// - hasValidPlacements → hasAnyPlacementForPlayer
// - hasValidMovements → hasAnyMovementForPlayer
// - hasValidActions, hasValidRecovery, nextPlayer → removed (dead code after migration)

/**
 * Force player to eliminate a cap when blocked with no valid moves.
 *
 * This is now a thin wrapper around the shared
 * {@link applyForcedEliminationForPlayer} helper (RR-CANON R205), which:
 *
 * - Checks the formal forced-elimination preconditions (R072/R100/R205).
 * - Selects a stack controlled by the player, preferring the smallest
 *   positive capHeight and falling back to the first stack when no caps
 *   exist.
 * - Eliminates that stack's cap via {@code eliminate_rings_from_stack},
 *   updating board.eliminatedRings, players[].eliminatedRings, and
 *   totalRingsEliminated in a way that satisfies INV-ELIMINATION-MONOTONIC
 *   and contributes to INV-S-MONOTONIC.
 *
 * TurnEngine mutates the provided {@link gameState} reference in-place so
 * that backend GameEngine callers observe the updated state without
 * changing their object identity contracts.
 */
function processForcedElimination(
  gameState: GameState,
  _deps: TurnEngineDeps,
  _hooks: TurnEngineHooks,
  playerNumber: number
): void {
  const outcome = applyForcedEliminationForPlayer(gameState, playerNumber);
  if (!outcome) {
    // Preconditions not satisfied (no stacks or actions); nothing to do.
    return;
  }

  // applyForcedEliminationForPlayer returns a new GameState instance; merge
  // it back into the caller-owned reference so downstream logic (including
  // advanceTurnAndPhase and GameEngine) continues to see mutations in-place.
  Object.assign(gameState, outcome.nextState);
}

// Local positionToString helper to avoid depending on the shared
// string-based serialization directly; behaviour matches
// shared/types/game.positionToString for the coordinates used here.
function positionToStringLocal(pos: Position): string {
  return pos.z !== undefined ? `${pos.x},${pos.y},${pos.z}` : `${pos.x},${pos.y}`;
}
