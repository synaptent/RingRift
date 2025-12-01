import type {
  GameState,
  Move,
  GameResult,
  Position,
  PerTurnState as SharedPerTurnState,
  TurnLogicDelegates,
} from '../../../shared/engine';
import {
  advanceTurnAndPhase,
  enumerateAllCaptureMoves as enumerateAllCaptureMovesAggregate,
  applyForcedEliminationForPlayer,
} from '../../../shared/engine';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';

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

  // For movement/capture moves originating from the must-move stack,
  // advance the tracked key to the new landing position so that any
  // subsequent phase (e.g. capture / chain_capture) references the same stack.
  if (
    mustMoveFromStackKey &&
    move.from &&
    move.to &&
    (move.type === 'move_stack' ||
      move.type === 'move_ring' ||
      move.type === 'build_stack' ||
      move.type === 'overtaking_capture' ||
      move.type === 'continue_capture_segment')
  ) {
    const fromKey = positionToStringLocal(move.from);
    if (fromKey === mustMoveFromStackKey) {
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
    hasAnyPlacement: (state, player) => hasValidPlacements(state, deps, player),
    hasAnyMovement: (state, player, turn) =>
      hasValidMovements(state, turn as PerTurnState, deps, player),
    hasAnyCapture: (state, player, turn) =>
      hasValidCaptures(state, turn as PerTurnState, deps, player),
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

  if (
    typeof process !== 'undefined' &&
    (process as any).env &&
    (process as any).env.NODE_ENV === 'test'
  ) {
    // eslint-disable-next-line no-console
    console.log('[TurnTrace.backend.advanceGameForCurrentPlayer]', {
      decision: 'advanceGameForCurrentPlayer',
      reason: 'advanceTurnAndPhase',
      before: beforeSnapshot,
      after: afterSnapshot,
    });
  }

  // Mutate the provided GameState reference in-place so callers that
  // hold onto `gameState` (notably backend GameEngine) observe the
  // updated phase/player fields and any forced-elimination effects
  // without replacing their internal pointer.
  Object.assign(gameState, nextState);

  return nextTurn as PerTurnState;
}

/**
 * Check if player has any valid capture moves available
 * Rule Reference: Section 10.1
 */
function hasValidCaptures(
  gameState: GameState,
  turnState: PerTurnState,
  _deps: TurnEngineDeps,
  playerNumber: number
): boolean {
  // Use the shared CaptureAggregate global enumerator so that the decision to
  // enter the capture phase stays in sync with the canonical capture surface
  // used by the sandbox and shared engine.
  const tempState: GameState = {
    ...gameState,
    currentPlayer: playerNumber,
    currentPhase: 'capture',
  };

  let moves = enumerateAllCaptureMovesAggregate(tempState, playerNumber);

  // Respect per-turn must-move constraints: if a stack was just placed or
  // updated this turn, only captures originating from that stack are
  // considered when deciding whether to enter the capture phase. This keeps
  // TurnEngine's gating semantics aligned with GameEngine.getValidMoves,
  // which applies the same restriction.
  const { mustMoveFromStackKey } = turnState;
  if (mustMoveFromStackKey) {
    moves = moves.filter((m) => {
      if ((m.type !== 'overtaking_capture' && m.type !== 'continue_capture_segment') || !m.from) {
        return false;
      }

      const fromKey = positionToStringLocal(m.from);
      return fromKey === mustMoveFromStackKey;
    });
  }

  return moves.length > 0;
}

/**
 * Check if player has any valid actions available
 * Rule Reference: Section 4.4
 */
function hasValidActions(
  gameState: GameState,
  turnState: PerTurnState,
  deps: TurnEngineDeps,
  playerNumber: number
): boolean {
  return (
    hasValidPlacements(gameState, deps, playerNumber) ||
    hasValidMovements(gameState, turnState, deps, playerNumber) ||
    hasValidCaptures(gameState, turnState, deps, playerNumber)
  );
}

/**
 * Check if player has any valid placement moves
 * Rule Reference: Section 4.1, 6.1-6.3
 */
function hasValidPlacements(
  gameState: GameState,
  deps: TurnEngineDeps,
  playerNumber: number
): boolean {
  const player = gameState.players.find((p) => p.playerNumber === playerNumber);
  if (!player || player.ringsInHand === 0) {
    return false; // No rings in hand to place
  }

  const { ruleEngine } = deps;

  // Ask RuleEngine for actual placement moves in a lightweight view
  // where this player is active and the phase is forced to
  // 'ring_placement'. This keeps forced-elimination gating in sync
  // with real placement availability.
  const tempState: GameState = {
    ...gameState,
    currentPlayer: playerNumber,
    currentPhase: 'ring_placement',
  };

  const moves = ruleEngine.getValidMoves(tempState);
  // Treat only actual place_ring actions as evidence of a real placement
  // option. skip_placement is a bookkeeping-only move and should not
  // prevent forced elimination or LPS tracking from considering the
  // player "blocked" for placement purposes.
  return moves.some((m) => m.type === 'place_ring');
}

/**
 * Check if player has any valid movement moves
 * Rule Reference: Section 8.1, 8.2
 */
function hasValidMovements(
  gameState: GameState,
  turnState: PerTurnState,
  deps: TurnEngineDeps,
  playerNumber: number
): boolean {
  const { ruleEngine } = deps;

  // Construct a movement-phase view for this player and delegate to
  // RuleEngine so movement availability is determined by the same
  // rules used for actual move generation.
  const tempState: GameState = {
    ...gameState,
    currentPlayer: playerNumber,
    currentPhase: 'movement',
  };

  let moves = ruleEngine.getValidMoves(tempState);

  // Respect per-turn must-move constraints: if a stack was just
  // placed or updated this turn, only movements originating from that
  // stack are considered when deciding whether to enter the movement
  // phase. This keeps TurnEngine's gating semantics aligned with
  // GameEngine.getValidMoves, which applies the same restriction.
  const { mustMoveFromStackKey } = turnState;
  if (mustMoveFromStackKey) {
    moves = moves.filter((m) => {
      const isMovementType =
        m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack';

      if (!isMovementType || !m.from) {
        return false;
      }

      const fromKey = positionToStringLocal(m.from);
      return fromKey === mustMoveFromStackKey;
    });
  }

  return moves.some(
    (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
  );
}

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

/**
 * Advance to the next player in turn order.
 */
function nextPlayer(gameState: GameState): void {
  const currentIndex = gameState.players.findIndex(
    (p) => p.playerNumber === gameState.currentPlayer
  );
  const nextIndex = (currentIndex + 1) % gameState.players.length;
  gameState.currentPlayer = gameState.players[nextIndex].playerNumber;
}

/**
 * Internal no-op hook to keep selected helper methods referenced so that
 * ts-node/TypeScript with noUnusedLocals can compile backend entrypoints
 * (including orchestrator soak harnesses) without treating them as dead code.
 * This has no behavioural impact.
 */
function _debugUseInternalTurnEngineHelpers(): void {
  void hasValidActions;
  void nextPlayer;
}
// Invoke once at module load so the helpers are marked as used.
_debugUseInternalTurnEngineHelpers();

// Local positionToString helper to avoid depending on the shared
// string-based serialization directly; behaviour matches
// shared/types/game.positionToString for the coordinates used here.
function positionToStringLocal(pos: Position): string {
  return pos.z !== undefined ? `${pos.x},${pos.y},${pos.z}` : `${pos.x},${pos.y}`;
}
