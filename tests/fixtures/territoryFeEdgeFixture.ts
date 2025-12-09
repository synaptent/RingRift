/**
 * Territory / Forced-Elimination Edge Fixture (canonical_square8_regen k90)
 *
 * Derived from the python_states["89"].board slice in
 * canonical_square8_regen__221a3037-3ca1-4d83-bccf-32d32edf0885__k90.state_bundle.json.
 *
 * This snapshot represents a late-game square‑8 position where:
 * - Most spaces are already collapsed to territory,
 * - Player 1 controls a tall mixed stack at (6,1),
 * - Several opponent markers form a narrow border near the eastern edge.
 *
 * The fixture is used to exercise territory-processing semantics (internal
 * eliminations + border collapse) against a realistic, dense board geometry
 * taken from a real self-play parity failure.
 */

import type { BoardState, Territory } from '../../src/shared/types/game';
import type { SerializedBoardState } from '../../src/shared/engine/contracts/serialization';
import { deserializeBoardState } from '../../src/shared/engine/contracts/serialization';

/**
 * Serialized board state reconstructed from the parity bundle's python_states["89"].board.
 *
 * Notes:
 * - Only board-level fields needed for territory processing are included
 *   (stacks, markers, collapsedSpaces, eliminatedRings, size, type).
 * - formedLines is empty because line processing is not the focus of this
 *   fixture.
 */
const territoryFeEdgeSerializedBoard: SerializedBoardState = {
  type: 'square8',
  size: 8,
  stacks: {
    '0,7': {
      position: { x: 0, y: 7 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    },
    '1,6': {
      position: { x: 1, y: 6 },
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    },
    '1,7': {
      position: { x: 1, y: 7 },
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    },
    '2,3': {
      position: { x: 2, y: 3 },
      // Mixed stack: bottom [1, 1], top [2, 2], cap belongs to player 2.
      rings: [1, 1, 2, 2],
      stackHeight: 4,
      capHeight: 2,
      controllingPlayer: 2,
    },
    '3,3': {
      position: { x: 3, y: 3 },
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    },
    '4,7': {
      position: { x: 4, y: 7 },
      // Mixed stack with player 2 cap over player 1 base.
      rings: [2, 2, 1, 1],
      stackHeight: 4,
      capHeight: 2,
      controllingPlayer: 1,
    },
    '5,4': {
      position: { x: 5, y: 4 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    },
    '6,1': {
      position: { x: 6, y: 1 },
      // Tall mixed stack used to exercise internal eliminations inside a region.
      rings: [2, 2, 2, 1, 1],
      stackHeight: 5,
      capHeight: 2,
      controllingPlayer: 1,
    },
    '6,5': {
      position: { x: 6, y: 5 },
      rings: [1, 1, 1],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: 1,
    },
  },
  markers: {
    '0,5': {
      position: { x: 0, y: 5 },
      player: 2,
      type: 'regular',
    },
    '2,7': {
      position: { x: 2, y: 7 },
      player: 2,
      type: 'regular',
    },
    '3,2': {
      position: { x: 3, y: 2 },
      player: 1,
      type: 'regular',
    },
    '3,7': {
      position: { x: 3, y: 7 },
      player: 2,
      type: 'regular',
    },
    '4,3': {
      position: { x: 4, y: 3 },
      player: 1,
      type: 'regular',
    },
    '5,2': {
      position: { x: 5, y: 2 },
      player: 1,
      type: 'regular',
    },
    '6,0': {
      position: { x: 6, y: 0 },
      player: 2,
      type: 'regular',
    },
    '7,2': {
      position: { x: 7, y: 2 },
      player: 1,
      type: 'regular',
    },
  },
  collapsedSpaces: {
    '0,0': 1,
    '0,1': 1,
    '0,2': 1,
    '1,0': 1,
    '1,1': 1,
    '1,2': 1,
    '1,3': 1,
    '2,0': 2,
    '2,1': 1,
    '2,2': 1,
    '2,5': 2,
    '2,6': 1,
    '3,0': 2,
    '3,1': 2,
    '3,4': 1,
    '3,5': 2,
    '3,6': 1,
    '4,0': 2,
    '4,5': 2,
    '4,6': 1,
    '5,3': 2,
    '5,5': 1,
    '5,6': 2,
    '5,7': 1,
    '6,6': 1,
    '6,7': 1,
    '7,3': 2,
    '7,4': 2,
    '7,5': 2,
    '7,6': 1,
    '7,7': 1,
  },
  eliminatedRings: {
    1: 5,
    2: 9,
  },
  formedLines: [],
};

/**
 * Reconstruct the late-game BoardState used by the territory/FE edge tests.
 */
export function createTerritoryFeEdgeBoard(): BoardState {
  return deserializeBoardState(territoryFeEdgeSerializedBoard);
}

/**
 * Disconnected territory region near the eastern edge that:
 * - Includes the tall mixed stack at (6,1) for player 1,
 * - Is bordered by markers at (6,0), (5,2), and (7,2),
 * - Has stacks for player 1 outside the region (e.g., at (0,7), (4,7), (5,4), (6,5)),
 *   so the self-elimination prerequisite (FAQ Q23 / §12.2) is satisfied.
 *
 * This region is not produced by the detector directly in tests; instead it
 * is a curated slice over the canonical board geometry, allowing us to probe
 * applyTerritoryRegion and canProcessTerritoryRegion on a realistic layout.
 */
export const territoryFeEdgeRegionForPlayer1: Territory = {
  spaces: [
    { x: 6, y: 1 },
    { x: 6, y: 2 },
    { x: 6, y: 3 },
  ],
  controllingPlayer: 1,
  isDisconnected: true,
};
