/**
 * @fileoverview Sandbox Movement Helpers - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This module is an **adapter** over the canonical shared engine.
 * It provides movement enumeration utilities for sandbox/offline games.
 *
 * Canonical SSoT:
 * - Movement logic: `src/shared/engine/aggregates/MovementAggregate.ts`
 * - Movement helpers: `src/shared/engine/aggregates/movementHelpers.ts`
 * - Recovery: `src/shared/engine/aggregates/RecoveryAggregate.ts`
 *
 * This adapter:
 * - Enumerates simple move targets via shared `enumerateSimpleMoveTargetsFromStack`
 * - Wraps recovery eligibility checks via shared helpers
 * - Provides marker path helper re-exports for sandbox use
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import type {
  BoardState,
  BoardType,
  Position,
  MovementBoardView,
  GameState,
} from '../../shared/engine';
import { positionToString, enumerateSimpleMoveTargetsFromStack } from '../../shared/engine';
import {
  enumerateRecoverySlideTargets,
  RecoverySlideTarget,
} from '../../shared/engine/aggregates/RecoveryAggregate';
import { isEligibleForRecovery as isEligibleHelper } from '../../shared/engine/playerStateHelpers';

// Re-export marker-path helpers: MarkerPathHelpers is a TypeScript-only
// interface, so we export it as a type to avoid creating a runtime import
// that bundlers expect from core.ts. The function helper remains a normal
// value export.
export type { MarkerPathHelpers } from '../../shared/engine';
export { applyMarkerEffectsAlongPathOnBoard } from '../../shared/engine';

export interface SimpleLanding {
  fromKey: string;
  to: Position;
}

/**
 * Enumerate simple, non-capturing movement options for the given player on
 * the provided board. This is a thin adapter over the shared movement
 * reachability helper so that sandbox movement semantics stay aligned with
 * the backend RuleEngine and shared GameEngine.
 */
export function enumerateSimpleMovementLandings(
  boardType: BoardType,
  board: BoardState,
  playerNumber: number,
  isValidPosition: (pos: Position) => boolean
): SimpleLanding[] {
  const results: SimpleLanding[] = [];

  const view: MovementBoardView = {
    isValidPosition: (pos: Position) => isValidPosition(pos),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) return undefined;
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

  for (const stack of board.stacks.values()) {
    if (stack.controllingPlayer !== playerNumber || stack.stackHeight <= 0) continue;

    const targets = enumerateSimpleMoveTargetsFromStack(
      boardType,
      stack.position,
      playerNumber,
      view
    );

    for (const target of targets) {
      results.push({
        fromKey: positionToString(target.from),
        to: target.to,
      });
    }
  }

  return results;
}

/**
 * Check if a player is eligible for recovery action in sandbox context.
 *
 * @param state - Current game state
 * @param playerNumber - Player to check
 * @returns True if the player can perform recovery actions
 */
export function isPlayerEligibleForRecovery(state: GameState, playerNumber: number): boolean {
  return isEligibleHelper(state, playerNumber);
}

/**
 * Enumerate recovery slide targets for the sandbox UI.
 *
 * Recovery is available when a player:
 * - Controls no stacks
 * - Has zero rings in hand
 * - Has at least one marker on the board
 * - Has at least one buried ring
 *
 * @param state - Current game state
 * @param playerNumber - Player to enumerate recovery moves for
 * @returns Array of recovery slide targets with metadata
 */
export function enumerateRecoverySlideLandings(
  state: GameState,
  playerNumber: number
): RecoverySlideTarget[] {
  return enumerateRecoverySlideTargets(state, playerNumber);
}

// Re-export the RecoverySlideTarget type for sandbox components
export type { RecoverySlideTarget };
