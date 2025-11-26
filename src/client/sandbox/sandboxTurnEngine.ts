import type {
  GamePhase,
  GameResult,
  GameState,
  Position,
  BoardState,
  RingStack,
  PerTurnState as SharedPerTurnState,
  TurnLogicDelegates,
} from '../../shared/engine';
import { BOARD_CONFIGS, positionToString, advanceTurnAndPhase } from '../../shared/engine';

/**
 * Internal per-turn state for the sandbox engine. Mirrors
 * ClientSandboxEngine._hasPlacedThisTurn and
 * ClientSandboxEngine._mustMoveFromStackKey but is kept here so turn
 * logic can be exercised in isolation. This is a thin alias of the
 * shared PerTurnState used by src/shared/engine/turnLogic so sandbox,
 * backend, and shared engines all share the same per-turn shape.
 */
export type SandboxTurnState = SharedPerTurnState;

/**
 * Hooks that the sandbox turn engine uses to delegate concrete board
 * operations back to ClientSandboxEngine. This keeps the turn engine
 * free of direct dependencies on the class implementation while still
 * allowing forced elimination and victory checks.
 */
export interface SandboxTurnHooks {
  /**
   * Enumerate legal ring placement positions for the given player,
   * enforcing the sandbox no-dead-placement rule.
   */
  enumerateLegalRingPlacements: (state: GameState, playerNumber: number) => Position[];

  /**
   * Check whether a stack at `from` has any legal move or capture on
   * the provided board. Analogue of
   * ClientSandboxEngine.hasAnyLegalMoveOrCaptureFrom.
   */
  hasAnyLegalMoveOrCaptureFrom: (
    state: GameState,
    from: Position,
    playerNumber: number,
    board: BoardState
  ) => boolean;

  /**
   * Get all stacks controlled by the specified player on the given
   * board.
   */
  getPlayerStacks: (state: GameState, playerNumber: number, board: BoardState) => RingStack[];

  /**
   * Eliminate one cap from the specified player's stacks, updating
   * board and player elimination counters.
   */
  forceEliminateCap: (state: GameState, playerNumber: number) => GameState;

  /**
   * Apply ring-elimination and territory-control victory checks and
   * return an updated state if the game has ended.
   */
  checkAndApplyVictory: (state: GameState) => GameState;
}

/**
 * Shared turn/phase progression helper for sandbox mode. This delegates
 * to the shared advanceTurnAndPhase helper so that backend, shared
 * engine, and sandbox all rotate turns and handle forced elimination
 * using the same canonical state machine.
 *
 * Callers are expected to invoke this after completing all automatic
 * post-move consequences for the current player (lines, territory,
 * victory checks) and, when modelling a full turn boundary, with
 * state.currentPhase set conceptually to 'territory_processing'.
 */
export function advanceTurnAndPhaseForCurrentPlayerSandbox(
  state: GameState,
  turnState: SandboxTurnState,
  hooks: SandboxTurnHooks
): { state: GameState; turnState: SandboxTurnState } {
  const hasAnyMovementForPlayer = (
    s: GameState,
    playerNumber: number,
    sharedTurn: SharedPerTurnState
  ): boolean => {
    const board = s.board;
    const stacks = hooks.getPlayerStacks(s, playerNumber, board);
    if (stacks.length === 0) {
      return false;
    }

    const mustKey = sharedTurn.mustMoveFromStackKey;
    const isMovementPhase = s.currentPhase === 'movement';

    if (mustKey && isMovementPhase) {
      const mustStack = stacks.find((stack) => positionToString(stack.position) === mustKey);
      if (mustStack) {
        return hooks.hasAnyLegalMoveOrCaptureFrom(s, mustStack.position, playerNumber, board);
      }
    }

    return stacks.some((stack) =>
      hooks.hasAnyLegalMoveOrCaptureFrom(s, stack.position, playerNumber, board)
    );
  };

  const delegates: TurnLogicDelegates = {
    getPlayerStacks: (s, playerNumber) => hooks.getPlayerStacks(s, playerNumber, s.board),
    hasAnyPlacement: (s, playerNumber) => {
      const player = s.players.find((p) => p.playerNumber === playerNumber);
      if (!player || player.ringsInHand <= 0) {
        return false;
      }

      const boardConfig = BOARD_CONFIGS[s.boardType];
      const stacks = hooks.getPlayerStacks(s, playerNumber, s.board);
      const ringsOnBoard = stacks.reduce((sum, stack) => sum + stack.rings.length, 0);
      const perPlayerCap = boardConfig.ringsPerPlayer;
      const remainingByCap = perPlayerCap - ringsOnBoard;
      const atOrAboveCap = remainingByCap <= 0;
      if (atOrAboveCap) {
        return false;
      }

      const placements = hooks.enumerateLegalRingPlacements(s, playerNumber);
      return placements.length > 0;
    },
    hasAnyMovement: (s, playerNumber, sharedTurn) =>
      hasAnyMovementForPlayer(s, playerNumber, sharedTurn as SharedPerTurnState),
    hasAnyCapture: (s, playerNumber, sharedTurn) =>
      hasAnyMovementForPlayer(s, playerNumber, sharedTurn as SharedPerTurnState),
    applyForcedElimination: (s, playerNumber) => {
      const afterElimination = hooks.forceEliminateCap(s, playerNumber);
      return hooks.checkAndApplyVictory(afterElimination);
    },
    getNextPlayerNumber: (s, current) => getNextPlayerNumberSandbox(s, current),
  };

  const { nextState, nextTurn } = advanceTurnAndPhase(
    state,
    turnState as SharedPerTurnState,
    delegates
  );

  if (
    typeof process !== 'undefined' &&
    (process as any).env &&
    (process as any).env.NODE_ENV === 'test'
  ) {
    // eslint-disable-next-line no-console
    console.log('[SandboxTurnEngine.advanceTurnAndPhaseForCurrentPlayerSandbox.after]', {
      before: {
        currentPlayer: state.currentPlayer,
        currentPhase: state.currentPhase,
      },
      after: {
        currentPlayer: nextState.currentPlayer,
        currentPhase: nextState.currentPhase,
      },
    });
  }

  return {
    state: nextState,
    turnState: nextTurn as SandboxTurnState,
  };
}

/**
 * Decide the starting phase for the current player at the beginning of
 * their turn and apply forced elimination when they are completely
 * blocked with no rings in hand.
 *
 * This is a functional extraction of
 * ClientSandboxEngine.startTurnForCurrentPlayer.
 */
export function startTurnForCurrentPlayerSandbox(
  state: GameState,
  turnState: SandboxTurnState,
  hooks: SandboxTurnHooks
): { state: GameState; turnState: SandboxTurnState } {
  // Before starting a new turn, re-check victory conditions in case the
  // prior player's movement/territory/line processing produced a
  // terminal state.
  state = hooks.checkAndApplyVictory(state);
  if (state.gameStatus !== 'active') {
    return { state, turnState };
  }

  // Reset per-turn flags at the beginning of a player's turn.
  turnState = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

  // We may need to advance through multiple players if some are forced
  // to eliminate a cap and immediately lose their turn. Guard with a
  // safety counter to avoid pathological loops.
  for (let safety = 0; safety < state.players.length; safety++) {
    const current = state.currentPlayer;
    const player = state.players.find((p) => p.playerNumber === current);
    if (!player) {
      return { state, turnState };
    }

    const eliminatedResult = maybeProcessForcedEliminationForCurrentPlayerSandbox(
      state,
      turnState,
      hooks
    );
    state = eliminatedResult.state;
    turnState = eliminatedResult.turnState;

    if (eliminatedResult.eliminated) {
      // Continue loop with the (updated) current player.
      continue;
    }

    // Determine starting phase for the current player. To stay aligned
    // with backend GameEngine.hasValidPlacements semantics (which only
    // gates on ringsInHand > 0), we do not use the no-dead-placement
    // enumerator here: if the player has rings in hand, they begin in
    // ring_placement; otherwise they begin in movement.
    const hasRings = player.ringsInHand > 0;
    const nextPhase: GamePhase = hasRings ? 'ring_placement' : 'movement';

    state = {
      ...state,
      currentPhase: nextPhase,
    };
    return { state, turnState };
  }

  return { state, turnState };
}

/**
 * If the current player has stacks on the board, no rings in hand, and
 * no legal moves or captures from any of their stacks, perform a forced
 * elimination and advance to the next player. This mirrors
 * ClientSandboxEngine.maybeProcessForcedEliminationForCurrentPlayer,
 * including the must-move-stack semantics during the movement phase.
 */
export function maybeProcessForcedEliminationForCurrentPlayerSandbox(
  state: GameState,
  turnState: SandboxTurnState,
  hooks: SandboxTurnHooks
): { state: GameState; turnState: SandboxTurnState; eliminated: boolean } {
  const current = state.currentPlayer;
  const player = state.players.find((p) => p.playerNumber === current);
  if (!player) {
    return { state, turnState, eliminated: false };
  }

  const board = state.board;
  const stacks = hooks.getPlayerStacks(state, current, board);
  if (stacks.length === 0) {
    // When the player has no stacks and no rings in hand, they are
    // completely out of pieces. In this case, advance turn order to
    // the next player instead of leaving the game stuck on an
    // unplayable turn.
    if (player.ringsInHand <= 0) {
      const nextPlayer = getNextPlayerNumberSandbox(state, current);
      state = {
        ...state,
        currentPlayer: nextPlayer,
      };

      const nextTurnState: SandboxTurnState = {
        hasPlacedThisTurn: false,
        mustMoveFromStackKey: undefined,
      };

      // Treat this as an "eliminated" turn for control-flow purposes
      // so callers such as startTurnForCurrentPlayerSandbox will
      // continue advancing turn order.
      return { state, turnState: nextTurnState, eliminated: true };
    }

    // If the player still has rings in hand but no stacks, they may be
    // able to act again once they re-enter ring_placement; do not
    // advance turn order here.
    return { state, turnState, eliminated: false };
  }

  // Approximate per-player ring cap usage in the same way as the
  // sandbox AI ring_placement policy: count all rings in stacks
  // controlled by this player and compare against BOARD_CONFIGS.
  // This keeps forced-elimination gating aligned with the AI's
  // maxAvailableGlobal calculation so that situations where the AI
  // treats the player as "at cap" do not get stuck because the
  // turn engine still believes placements are available.
  const boardConfig = BOARD_CONFIGS[state.boardType];
  const ringsOnBoard = stacks.reduce((sum, stack) => sum + stack.rings.length, 0);
  const perPlayerCap = boardConfig.ringsPerPlayer;
  const remainingByCap = perPlayerCap - ringsOnBoard;
  const atOrAboveCap = remainingByCap <= 0;

  // Determine whether this player has any legal non-capture moves or
  // captures available. When a must-move stack is being tracked during
  // the movement phase, we first check that specific stack; only if it
  // no longer exists do we fall back to a global "any stack" check and
  // clear the must-move constraint.
  const mustKey = turnState.mustMoveFromStackKey;
  let hasAnyAction: boolean;
  let nextTurnState = { ...turnState };

  if (mustKey && state.currentPhase === 'movement') {
    const mustStack = stacks.find((s) => positionToString(s.position) === mustKey);

    if (mustStack) {
      hasAnyAction = hooks.hasAnyLegalMoveOrCaptureFrom(state, mustStack.position, current, board);
    } else {
      // The must-move stack has been removed (e.g. via capture, lines,
      // or territory effects). In this case the constraint is no longer
      // meaningful; clear it and defer to the global reachability check.
      nextTurnState.mustMoveFromStackKey = undefined;
      hasAnyAction = stacks.some((stack) =>
        hooks.hasAnyLegalMoveOrCaptureFrom(state, stack.position, current, board)
      );
    }
  } else {
    hasAnyAction = stacks.some((stack) =>
      hooks.hasAnyLegalMoveOrCaptureFrom(state, stack.position, current, board)
    );
  }

  // Also check whether the player has any legal ring placements that
  // satisfy the no-dead-placement rule. When the player has reached
  // or exceeded their per-player ring cap (as approximated by
  // ringsOnBoard above), we treat placement as unavailable even if
  // enumerateLegalRingPlacements reports potential positions. This
  // keeps forced elimination aligned with the sandbox AI, which uses
  // the same cap approximation to decide when placement is no longer
  // an option.
  const hasAnyPlacement = (() => {
    if (player.ringsInHand <= 0) {
      return false;
    }
    if (atOrAboveCap) {
      return false;
    }
    const placements = hooks.enumerateLegalRingPlacements(state, current);
    return placements.length > 0;
  })();

  if (hasAnyAction || hasAnyPlacement) {
    return { state, turnState: nextTurnState, eliminated: false };
  }

  // At this point, the player controls stacks but has no legal moves,
  // captures, or placements. Apply forced elimination and advance to
  // the next player.
  state = hooks.forceEliminateCap(state, current);

  const nextPlayer = getNextPlayerNumberSandbox(state, current);
  state = {
    ...state,
    currentPlayer: nextPlayer,
  };

  // After forced elimination, clear per-turn must-move state for the
  // eliminated player; the next player will get a fresh turn state.
  nextTurnState = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

  return { state, turnState: nextTurnState, eliminated: true };
}

/**
 * Utility: get the next player's number in turn order for the given
 * state, wrapping around at the end.
 */
export function getNextPlayerNumberSandbox(state: GameState, current: number): number {
  const players = state.players;
  const idx = players.findIndex((p) => p.playerNumber === current);
  const nextIdx = (idx + 1) % players.length;
  return players[nextIdx].playerNumber;
}
