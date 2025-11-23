import { BoardState, BoardType, Position, positionToString } from '../../shared/types/game';
import { MovementBoardView } from '../../shared/engine/core';
import { enumerateSimpleMoveTargetsFromStack } from '../../shared/engine/movementLogic';

// Re-export marker-path helpers: MarkerPathHelpers is a TypeScript-only
// interface, so we export it as a type to avoid creating a runtime import
// that bundlers expect from core.ts. The function helper remains a normal
// value export.
export type { MarkerPathHelpers } from '../../shared/engine/core';
export { applyMarkerEffectsAlongPathOnBoard } from '../../shared/engine/core';

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
