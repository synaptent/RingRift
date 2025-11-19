import { GameState, Move, GameResult, Position } from '../../../shared/types/game';
import { BoardManager } from '../BoardManager';
import { RuleEngine } from '../RuleEngine';
import { getMovementDirectionsForBoardType } from '../../../shared/engine/core';

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
 * Internal per-turn state for the backend engine. This mirrors the
 * fields kept on GameEngine but lives here so turn logic can be
 * exercised in isolation.
 */
export interface PerTurnState {
  hasPlacedThisTurn: boolean;
  mustMoveFromStackKey?: string | undefined;
}

/**
 * Hooks that let the turn engine delegate elimination and game-end
 * side effects back to the owning GameEngine without depending on its
 * concrete class shape.
 */
export interface TurnEngineHooks {
  eliminatePlayerRingOrCap: (playerNumber: number) => void;
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
  // subsequent phase (e.g. capture) references the same stack.
  if (
    mustMoveFromStackKey &&
    move.from &&
    move.to &&
    (move.type === 'move_stack' ||
      move.type === 'move_ring' ||
      move.type === 'build_stack' ||
      move.type === 'overtaking_capture')
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
 * This is a direct extraction of GameEngine.advanceGame and its
 * helper methods (hasValidCaptures / hasValidActions /
 * hasValidPlacements / hasValidMovements / processForcedElimination /
 * nextPlayer), rewritten in functional style but preserving
 * semantics.
 */
export function advanceGameForCurrentPlayer(
  gameState: GameState,
  turnState: PerTurnState,
  deps: TurnEngineDeps,
  hooks: TurnEngineHooks
): PerTurnState {
  switch (gameState.currentPhase) {
    case 'ring_placement': {
      // After placing a ring (or skipping), the active player must
      // take an action if any are available. Depending on the board
      // state and must-move constraints, that action might be a
      // movement *or* an overtaking capture. Both of these are chosen
      // from the movement phase in the current backend engine
      // (captures are exposed as overtaking_capture moves alongside
      // simple movements), so we only need to decide whether any
      // action exists at all.

      const canMove = hasValidMovements(gameState, turnState, deps, gameState.currentPlayer);
      const canCapture = hasValidCaptures(gameState, turnState, deps, gameState.currentPlayer);

      if (canMove || canCapture) {
        // At least one legal movement or capture exists (respecting
        // any must-move origin). Start the interactive part of the
        // turn in the movement phase so the player/AI can choose
        // between simple moves and overtaking_capture.
        gameState.currentPhase = 'movement';
      } else {
        // Defensive fallback: no actions remain despite a
        // ring_placement phase. In well-formed games this should be
        // prevented by placement validation, but if it occurs we
        // proceed to bookkeeping rather than leaving the game stuck
        // in an interactive phase with no moves.
        gameState.currentPhase = 'line_processing';
      }

      break;
    }

    case 'movement': {
      // After the interactive movement step (which may be either a
      // simple move_stack/move_ring or an initial overtaking_capture),
      // all mandatory capture chaining is driven internally by
      // GameEngine via its chainCaptureState loop. By the time control
      // returns here, either the chain has been fully resolved or no
      // captures were taken at all, and the next step is always to
      // run post-move bookkeeping (lines, territory, etc.).
      //
      // Therefore we skip the legacy "enter capture phase if any
      // captures exist" behaviour and advance directly to
      // line_processing.
      gameState.currentPhase = 'line_processing';
      break;
    }

    case 'capture': {
      // After captures complete, proceed to line processing
      // Rule Reference: Section 4.3, 4.5
      gameState.currentPhase = 'line_processing';
      break;
    }

    case 'line_processing': {
      // After processing lines, proceed to territory processing
      // Rule Reference: Section 4.5
      gameState.currentPhase = 'territory_processing';
      break;
    }

    case 'territory_processing': {
      // After processing territory, turn is complete
      // Check if player still has rings/stacks or needs to place
      // Rule Reference: Section 4, Section 4.1
      nextPlayer(gameState);
      // New player starts with a fresh per-turn state; any previous
      // must-move constraint applies only to the player who just
      // moved.
      turnState = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };

      const { boardManager, ruleEngine } = deps;

      // Determine starting phase for next player
      const playerStacks = boardManager.getPlayerStacks(gameState.board, gameState.currentPlayer);
      const currentPlayer = gameState.players.find(
        (p) => p.playerNumber === gameState.currentPlayer
      );

      // Rule Reference: Section 4.4 - Forced Elimination When Blocked
      // Check if player has no valid actions but controls stacks
      if (
        playerStacks.length > 0 &&
        !hasValidActions(gameState, turnState, deps, gameState.currentPlayer)
      ) {
        // Player is blocked with stacks - must eliminate a cap
        processForcedElimination(gameState, deps, hooks, gameState.currentPlayer);

        // After forced elimination, check victory conditions
        const gameEndCheck = ruleEngine.checkGameEnd(gameState);
        if (gameEndCheck.isGameOver) {
          // Game ended due to forced elimination
          hooks.endGame(gameEndCheck.winner, gameEndCheck.reason || 'forced_elimination');
          // Reset per-turn placement state for the next turn (if any).
          return { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };
        }

        // Continue to next player after forced elimination
        nextPlayer(gameState);

        // Re-evaluate starting phase for the actual next player, with
        // the same skip-over-dead-players semantics used in the normal
        // progression path below.
        const MAX_SKIPS = gameState.players.length;
        let skips = 0;

        while (skips < MAX_SKIPS) {
          const stacksForCurrent = boardManager.getPlayerStacks(
            gameState.board,
            gameState.currentPlayer
          );
          const currentPlayerState = gameState.players.find(
            (p) => p.playerNumber === gameState.currentPlayer
          );

          if (!currentPlayerState) {
            break;
          }

          if (stacksForCurrent.length === 0 && currentPlayerState.ringsInHand === 0) {
            // This player has no rings on the board and none in hand;
            // they cannot take any actions this turn. Skip them and
            // advance to the next player. Global terminal states
            // (e.g. all players out of material) are handled by
            // RuleEngine.checkGameEnd at the GameEngine level.
            nextPlayer(gameState);
            skips++;
            continue;
          }

          if (stacksForCurrent.length === 0 && currentPlayerState.ringsInHand > 0) {
            // No rings on board but has rings in hand - must place
            gameState.currentPhase = 'ring_placement';
          } else if (currentPlayerState.ringsInHand > 0) {
            // Has rings in hand and on board - can optionally place
            gameState.currentPhase = 'ring_placement';
          } else {
            // No rings in hand or all rings placed - go directly to movement
            gameState.currentPhase = 'movement';
          }

          break;
        }
      } else {
        // Normal turn progression. In addition to the standard
        // placement-vs-movement choice, we must also handle players
        // who have *no* material at all (no stacks and no rings in
        // hand). Such players can never take actions again and should
        // be skipped entirely so the game can continue for the
        // remaining players.
        const MAX_SKIPS = gameState.players.length;
        let skips = 0;

        while (skips < MAX_SKIPS) {
          const stacksForCurrent = boardManager.getPlayerStacks(
            gameState.board,
            gameState.currentPlayer
          );
          const currentPlayerState = gameState.players.find(
            (p) => p.playerNumber === gameState.currentPlayer
          );

          if (!currentPlayerState) {
            break;
          }

          if (stacksForCurrent.length === 0 && currentPlayerState.ringsInHand === 0) {
            // This player has no rings on the board and none in hand;
            // they cannot take any actions this turn. Skip them and
            // advance to the next player. Global terminal states
            // (e.g. all players out of material) are handled by
            // RuleEngine.checkGameEnd at the GameEngine level.
            nextPlayer(gameState);
            skips++;
            continue;
          }

          if (stacksForCurrent.length === 0 && currentPlayerState.ringsInHand > 0) {
            // No rings on board but has rings in hand - must place
            gameState.currentPhase = 'ring_placement';
          } else if (currentPlayerState.ringsInHand > 0) {
            // Has rings in hand and on board - can optionally place
            gameState.currentPhase = 'ring_placement';
          } else {
            // No rings in hand or all rings placed - go directly to movement
            gameState.currentPhase = 'movement';
          }

          break;
        }
      }

      // Reset per-turn placement state for the newly active player.
      turnState = { hasPlacedThisTurn: false, mustMoveFromStackKey: undefined };
      break;
    }
  }

  return turnState;
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
        m.type === 'overtaking_capture';

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

  // TODO: In full implementation, player should choose which stack.
  // For now, eliminate from first stack with a valid cap.
  for (const stack of playerStacks) {
    if (stack.capHeight > 0) {
      hooks.eliminatePlayerRingOrCap(playerNumber);
      return;
    }
  }
}

/**
 * Get all movement directions based on board type.
 */
function getAllDirections(
  boardType: GameState['boardType']
): { x: number; y: number; z?: number }[] {
  return getMovementDirectionsForBoardType(boardType);
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
