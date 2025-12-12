/**
 * Player State Helpers
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Shared helpers for checking player material and action availability.
 * Used by GameEngine and ClientSandboxEngine for LPS (Last Player Standing)
 * tracking and turn validation.
 *
 * Rule Reference: RR-CANON R172 (Last Player Standing Victory)
 *
 * @module playerStateHelpers
 */

import type { GameState, BoardState } from '../types/game';
import { countRingsInPlayForPlayer } from './core';

/**
 * Check if a player has any material remaining in the game.
 *
 * A player has material if they have at least one ring of their colour
 * still in play (on the board in any stack) or in their hand.
 *
 * This is used for LPS tracking to determine which players are still
 * active participants in the game.
 *
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @returns True if the player has any rings in play or hand
 */
export function playerHasMaterial(state: GameState, playerNumber: number): boolean {
  return countRingsInPlayForPlayer(state, playerNumber) > 0;
}

/**
 * Check if a player controls any stacks on the board.
 *
 * A player controls a stack when their ring is on top (the controlling player).
 * This is different from having material - a player might have rings buried
 * in opponent-controlled stacks but control no stacks themselves.
 *
 * @param board - Current board state
 * @param playerNumber - Player to check
 * @returns True if the player controls at least one stack
 */
export function playerControlsAnyStack(board: BoardState, playerNumber: number): boolean {
  for (const stack of board.stacks.values()) {
    if (stack.controllingPlayer === playerNumber) {
      return true;
    }
  }
  return false;
}

/**
 * Check if a player has any active material (controlled stacks or rings in hand).
 *
 * This is a stricter check than playerHasMaterial - it requires the player
 * to either control a stack OR have rings in hand. Rings buried in opponent
 * stacks don't count.
 *
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @returns True if the player controls stacks or has rings in hand
 */
export function playerHasActiveMaterial(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return false;
  }

  if (player.ringsInHand > 0) {
    return true;
  }

  return playerControlsAnyStack(state.board, playerNumber);
}

/**
 * Delegates for action availability checks.
 *
 * Both GameEngine and ClientSandboxEngine have different ways to enumerate
 * valid moves. This interface allows the shared hasAnyRealAction helper
 * to work with either engine's enumeration methods.
 */
export interface ActionAvailabilityDelegates {
  /** Check if player can place a ring */
  hasPlacement: (playerNumber: number) => boolean;
  /** Check if player can make a non-capture movement */
  hasMovement: (playerNumber: number) => boolean;
  /** Check if player can make an overtaking capture */
  hasCapture: (playerNumber: number) => boolean;
}

/**
 * Check if a player has any "real" action available.
 *
 * Real actions are:
 * - Ring placement (place_ring)
 * - Non-capture stack movement (move_stack, move_ring, build_stack)
 * - Overtaking capture (overtaking_capture)
 *
 * NOT real actions (for LPS purposes):
 * - Recovery slide (recovery_slide) - player can still use recovery but it
 *   does not count as a real action for LPS (only placement/movement/capture do).
 * - Forced elimination (eliminate_rings_from_stack when stack > 5)
 * - Line processing decisions
 * - Territory processing decisions
 * - Skip placement (pass)
 *
 * This is used for R172 Last Player Standing tracking.
 *
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @param delegates - Engine-specific move enumeration functions
 * @returns True if the player has at least one real action available
 */
export function hasAnyRealAction(
  state: GameState,
  playerNumber: number,
  delegates: ActionAvailabilityDelegates
): boolean {
  if (state.gameStatus !== 'active') {
    return false;
  }

  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return false;
  }

  // Check placement first (if player has rings in hand)
  if (player.ringsInHand > 0 && delegates.hasPlacement(playerNumber)) {
    return true;
  }

  // Check non-capture movement
  if (delegates.hasMovement(playerNumber)) {
    return true;
  }

  // Check overtaking capture
  if (delegates.hasCapture(playerNumber)) {
    return true;
  }

  // NOTE: Recovery is intentionally NOT checked here.
  // Recovery moves are still valid actions a player can take, but they don't
  // count as "real actions" for LPS purposes. This means recovery alone does
  // not prevent Last-Player-Standing outcomes.

  return false;
}

/**
 * Check if a player owns any markers on the board.
 *
 * @param board - Current board state
 * @param playerNumber - Player to check
 * @returns True if the player owns at least one marker
 */
export function playerHasMarkers(board: BoardState, playerNumber: number): boolean {
  for (const marker of board.markers.values()) {
    if (marker.player === playerNumber) {
      return true;
    }
  }
  return false;
}

/**
 * Count buried rings for a player.
 *
 * A buried ring is a ring of the player's colour that is in an opponent-
 * controlled stack (not the top ring). These are used for recovery action
 * costs per RR-CANON-R113.
 *
 * @param board - Current board state
 * @param playerNumber - Player to count buried rings for
 * @returns Number of buried rings
 */
export function countBuriedRings(board: BoardState, playerNumber: number): number {
  let count = 0;

  for (const stack of board.stacks.values()) {
    // Only count rings in opponent-controlled stacks
    if (stack.controllingPlayer === playerNumber) continue;

    // Count rings belonging to this player (excluding top ring)
    // rings[0] is the top ring (controlling player), so start from index 1
    for (let i = 1; i < stack.rings.length; i++) {
      if (stack.rings[i] === playerNumber) {
        count++;
      }
    }
  }

  return count;
}

/**
 * Check if a player is eligible for recovery action.
 *
 * A player is eligible per RR-CANON-R110 if:
 * - They control no stacks
 * - They own at least one marker
 * - They have at least one buried ring
 *
 * Note: Recovery eligibility is independent of rings in hand.
 * Players with rings may choose recovery over placement.
 *
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @returns True if the player is eligible for recovery action
 */
export function isEligibleForRecovery(state: GameState, playerNumber: number): boolean {
  const player = state.players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return false;
  }

  // Must control no stacks
  if (playerControlsAnyStack(state.board, playerNumber)) {
    return false;
  }

  // Must own at least one marker
  let hasMarker = false;
  for (const marker of state.board.markers.values()) {
    if (marker.player === playerNumber) {
      hasMarker = true;
      break;
    }
  }
  if (!hasMarker) {
    return false;
  }

  // Must have at least one buried ring
  if (countBuriedRings(state.board, playerNumber) < 1) {
    return false;
  }

  return true;
}
