import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  Position,
  RingStack,
  positionToString,
} from '../../shared/types/game';
import {
  calculateDistance,
  getMovementDirectionsForBoardType,
  getPathPositions,
} from '../../shared/engine/core';

export interface SimpleLanding {
  fromKey: string;
  to: Position;
}

/**
 * Enumerate simple, non-capturing movement options for the given player on
 * the provided board. This is a pure helper that mirrors the path/occupancy
 * checks used by ClientSandboxEngine for simple moves.
 */
export function enumerateSimpleMovementLandings(
  boardType: BoardType,
  board: BoardState,
  playerNumber: number,
  isValidPosition: (pos: Position) => boolean
): SimpleLanding[] {
  const results: SimpleLanding[] = [];
  const directions = getMovementDirectionsForBoardType(boardType);
  const config = BOARD_CONFIGS[boardType];

  for (const stack of board.stacks.values()) {
    if (stack.controllingPlayer !== playerNumber) continue;
    const from = stack.position;
    const fromKey = positionToString(from);

    if (fromKey === '2,7') {
      // eslint-disable-next-line no-console
      console.log(
        '[Sandbox Movement Debug] Checking moves for 2,7. Board stacks keys:',
        Array.from(board.stacks.keys())
      );
    }

    for (const dir of directions) {
      let step = 1;
      for (;;) {
        const to: Position = {
          x: from.x + dir.x * step,
          y: from.y + dir.y * step,
          ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step }),
        };

        const toKey = positionToString(to);

        // Stop once we move off the logical board or far beyond its bounds.
        if (
          !isValidPosition(to) ||
          Math.abs(to.x) > config.size * 2 ||
          Math.abs(to.y) > config.size * 2
        ) {
          if (fromKey === '2,7') {
            // eslint-disable-next-line no-console
            console.log(`[Sandbox Movement Debug] Off board or invalid at ${toKey} from 2,7`);
          }
          break;
        }

        // Collapsed spaces block movement and cannot be landed on.
        if (board.collapsedSpaces.has(toKey)) {
          if (fromKey === '2,7') {
            // eslint-disable-next-line no-console
            console.log(`[Sandbox Movement Debug] Collapsed space at ${toKey} from 2,7`);
          }
          break;
        }

        // Check that the path between from and to is unobstructed (no stacks
        // or collapsed spaces) excluding the landing cell itself.
        const path = getPathPositions(from, to).slice(1, -1);
        let blocked = false;
        for (const pos of path) {
          const pathKey = positionToString(pos);
          const hasStack = board.stacks.has(pathKey);
          const hasCollapsed = board.collapsedSpaces.has(pathKey);

          if (fromKey === '2,7' && toKey === '4,7') {
            // eslint-disable-next-line no-console
            console.log(
              `[Sandbox Movement Debug] Checking path pos '${pathKey}' for 2,7->4,7. Stack: ${hasStack}, Collapsed: ${hasCollapsed}`
            );
            if (pathKey === '3,7' && !hasStack) {
              const keys = Array.from(board.stacks.keys());
              const matching = keys.find((k) => k === '3,7');
              // eslint-disable-next-line no-console
              console.log('[Sandbox Movement Debug] Key check:', {
                pathKey,
                pathKeyLen: pathKey.length,
                keysIncludePathKey: keys.includes(pathKey),
                matchingKeyInMap: matching,
                matchingKeyLen: matching?.length,
              });
            }
          }

          if (hasCollapsed || hasStack) {
            blocked = true;
            break;
          }
        }
        if (blocked) {
          if (fromKey === '2,7') {
            // eslint-disable-next-line no-console
            console.log(`[Sandbox Movement Debug] Blocked path from 2,7 to ${toKey}`);
          }
          break;
        }

        const destinationStack = board.stacks.get(toKey);
        const marker = board.markers.get(toKey);
        const markerOwner = marker?.player;

        // Rule alignment with backend validateStackMovement and
        // hasAnyLegalMoveOrCaptureFromOnBoard:
        // - Can land on empty or same-color marker.
        // - Cannot land on opponent marker, but opponent markers do NOT
        //   block further movement along the ray.
        if (!destinationStack || destinationStack.stackHeight === 0) {
          if (markerOwner !== undefined && markerOwner !== stack.controllingPlayer) {
            // Opponent marker: this cell is not a legal landing square,
            // but the ray continues past it.
            if (fromKey === '2,7') {
              // eslint-disable-next-line no-console
              console.log(`[Sandbox Movement Debug] Opponent marker at ${toKey} from 2,7`);
            }
            step += 1;
            continue;
          }

          const distance = calculateDistance(boardType, from, to);
          if (distance >= stack.stackHeight) {
            if (fromKey === '2,7') {
              // eslint-disable-next-line no-console
              console.log(`[Sandbox Movement Debug] Adding move from 2,7 to ${toKey}`, {
                fromKey,
                toKey,
                pushedFromKey: fromKey,
              });
            }
            results.push({ fromKey, to });
            if (fromKey === '2,7') {
              // eslint-disable-next-line no-console
              console.log(
                `[Sandbox Movement Debug] Results length after push: ${results.length}. Last item fromKey: ${results[results.length - 1].fromKey}`
              );
            }
          } else if (fromKey === '2,7') {
            // eslint-disable-next-line no-console
            console.log(`[Sandbox Movement Debug] Distance too short from 2,7 to ${toKey}`, {
              distance,
              stackHeight: stack.stackHeight,
            });
          }

          // Empty/own-marker-only spaces do not block further exploration;
          // keep walking along this ray.
          step += 1;
          continue;
        }

        // Landing on a stack (own or opponent) is allowed for simple merging
        // movement, but we cannot move beyond that stack.
        const distance = calculateDistance(boardType, from, to);
        if (distance >= stack.stackHeight) {
          if (fromKey === '2,7') {
            // eslint-disable-next-line no-console
            console.log(`[Sandbox Movement Debug] Adding move (merge) from 2,7 to ${toKey}`, {
              fromKey,
              toKey,
              pushedFromKey: fromKey,
            });
          }
          results.push({ fromKey, to });
        } else if (fromKey === '2,7') {
          // eslint-disable-next-line no-console
          console.log(`[Sandbox Movement Debug] Distance too short (merge) from 2,7 to ${toKey}`, {
            distance,
            stackHeight: stack.stackHeight,
          });
        }

        // Stacks block further positions along this ray.
        break;
      }
    }
  }

  if (
    results.length === 0 &&
    board.stacks.has('2,7') &&
    playerNumber === board.stacks.get('2,7')?.controllingPlayer
  ) {
    // eslint-disable-next-line no-console
    console.log('[Sandbox Movement Debug] Returning 0 results for player', playerNumber);
  }
  if (board.stacks.has('2,7') && playerNumber === board.stacks.get('2,7')?.controllingPlayer) {
    const movesFrom27 = results.filter((r) => r.fromKey === '2,7');
    // eslint-disable-next-line no-console
    console.log(
      `[Sandbox Movement Debug] Returning ${results.length} results. Moves from 2,7: ${movesFrom27.length}`
    );
    if (movesFrom27.length === 0 && results.length > 0) {
      // eslint-disable-next-line no-console
      console.log(
        '[Sandbox Movement Debug] All fromKeys:',
        results.map((r) => r.fromKey)
      );
    }
  }
  return results;
}

export interface MarkerPathHelpers {
  setMarker(position: Position, playerNumber: number, board: BoardState): void;
  collapseMarker(position: Position, playerNumber: number, board: BoardState): void;
  flipMarker(position: Position, playerNumber: number, board: BoardState): void;
}

/**
 * Apply marker effects for a move or capture segment from `from` to `to` on
 * the given board, using the provided helper callbacks.
 *
 * By default this mirrors the backend movement behaviour:
 *   - Leave a marker on the true departure space.
 *   - Process intermediate markers (collapse/flip).
 *   - Remove a same-colour marker on the landing space.
 *
 * Callers that need finer-grained control (e.g. capture segments that want
 * to avoid placing a departure marker on an intermediate stack such as the
 * capture target) can pass options to disable the departure marker while
 * still reusing the intermediate/landing semantics.
 */
export function applyMarkerEffectsAlongPathOnBoard(
  board: BoardState,
  from: Position,
  to: Position,
  playerNumber: number,
  helpers: MarkerPathHelpers,
  options?: { leaveDepartureMarker?: boolean }
): void {
  const path = getPathPositions(from, to);
  if (path.length === 0) return;

  const leaveDepartureMarker = options?.leaveDepartureMarker !== false;

  const fromKey = positionToString(from);
  // Leave a marker on the departure space if it isn't already collapsed.
  if (leaveDepartureMarker && !board.collapsedSpaces.has(fromKey)) {
    const existing = board.markers.get(fromKey);
    if (!existing) {
      helpers.setMarker(from, playerNumber, board);
    }
  }

  // Process intermediate positions (excluding endpoints)
  const intermediate = path.slice(1, -1);
  for (const pos of intermediate) {
    const key = positionToString(pos);
    if (board.collapsedSpaces.has(key)) {
      continue;
    }
    const marker = board.markers.get(key);
    if (!marker) {
      continue;
    }
    if (marker.player === playerNumber) {
      // Own marker collapses to territory
      helpers.collapseMarker(pos, playerNumber, board);
    } else {
      // Opponent marker flips to mover's color
      helpers.flipMarker(pos, playerNumber, board);
    }
  }

  // Landing: remove own marker if present
  const landingKey = positionToString(to);
  const landingMarker = board.markers.get(landingKey);
  if (landingMarker && landingMarker.player === playerNumber) {
    board.markers.delete(landingKey);
  }
}
