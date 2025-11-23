import { GameState, Move, GameResult, Position } from '../../../shared/types/game';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';
import {
  advanceTurnAndPhase,
  PerTurnState as SharedPerTurnState,
  TurnLogicDelegates,
} from '../../../shared/engine/turnLogic';

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

  const { nextState, nextTurn } = advanceTurnAndPhase(gameState, turnState, delegates);

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
  deps: TurnEngineDeps,
  playerNumber: number
): boolean {
  const { ruleEngine } = deps;

  // Delegate to RuleEngine for capture generation so that the
  // decision to enter the capture phase stays in sync with the
  // actual overtaking_capture semantics. We construct a lightweight
  // view of the current state with phase forced to 'capture' for the
  // specified player and ask RuleEngine for valid moves.
  const tempState: GameState = {
    ...gameState,
    currentPlayer: playerNumber,
    currentPhase: 'capture',
  };

  let moves = ruleEngine.getValidMoves(tempState);

  // Respect per-turn must-move constraints: if a stack was just
  // placed or updated this turn, only captures originating from that
  // stack are considered when deciding whether to enter the capture
  // phase. This keeps TurnEngine's gating semantics aligned with
  // GameEngine.getValidMoves, which applies the same restriction.
  const { mustMoveFromStackKey } = turnState;
  if (mustMoveFromStackKey) {
    moves = moves.filter((m) => {
      const isMovementOrCaptureType =
        m.type === 'move_stack' ||
        m.type === 'move_ring' ||
        m.type === 'build_stack' ||
        m.type === 'overtaking_capture' ||
        m.type === 'continue_capture_segment';

      if (!isMovementOrCaptureType || !m.from) {
        return false;
      }

      const fromKey = positionToStringLocal(m.from);
      return fromKey === mustMoveFromStackKey;
    });
  }

  return moves.some((m) => m.type === 'overtaking_capture');
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
  return moves.some((m) => m.type === 'place_ring' || m.type === 'skip_placement');
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
 * Force player to eliminate a cap when blocked with no valid moves
 * Rule Reference: Section 4.4 - Forced Elimination When Blocked
 */
function processForcedElimination(
  gameState: GameState,
  deps: TurnEngineDeps,
  hooks: TurnEngineHooks,
  playerNumber: number
): void {
  const { boardManager } = deps;
  const playerStacks = boardManager.getPlayerStacks(gameState.board, playerNumber);

  if (playerStacks.length === 0) {
    // No stacks to eliminate from - player forfeits turn
    return;
  }

  // Auto-execute elimination: Select the best stack to eliminate.
  // Strategy:
  // 1. Prefer stacks with capHeight > 0 (actual caps).
  // 2. Among those, prefer the one with the SMALLEST capHeight to minimize loss.
  // 3. If no caps (e.g. test fixtures), pick the first available stack.

  let bestStack = playerStacks[0];
  let minCapHeight = Infinity;

  for (const stack of playerStacks) {
    const capHeight = (stack as any).capHeight;
    if (typeof capHeight === 'number' && capHeight > 0) {
      if (capHeight < minCapHeight) {
        minCapHeight = capHeight;
        bestStack = stack;
      }
    }
  }

  // If we found a stack with a cap, bestStack is the one with the smallest cap.
  // If we didn't find any (minCapHeight is still Infinity), bestStack remains the first stack.
  // This handles both real game scenarios and simplified test fixtures.

  hooks.eliminatePlayerRingOrCap(playerNumber, bestStack.position);
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

// Local positionToString helper to avoid depending on the shared
// string-based serialization directly; behaviour matches
// shared/types/game.positionToString for the coordinates used here.
function positionToStringLocal(pos: Position): string {
  return pos.z !== undefined ? `${pos.x},${pos.y},${pos.z}` : `${pos.x},${pos.y}`;
}
