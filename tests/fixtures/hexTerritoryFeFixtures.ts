/**
 * Hex Territory / Forced-Elimination Fixtures (HX-A)
 *
 * These fixtures mirror the square8 territory/FE edge fixtures in
 * {@link territoryFeEdgeFixture.ts} but for a hexagonal board. They are
 * intentionally small, board-slice-oriented snapshots that can be reused
 * across engine, orchestrator, and parity tests without embedding full
 * GameState histories.
 *
 * HX-A (hexTerritoryFeRegionHxAForPlayer1):
 *
 * - Board type: hexagonal (radius 12, size 13).
 * - Region spaces: a compact three-cell region around the origin:
 *     (0,0,0), (1,-1,0), (0,-1,1)
 * - Internal stacks on the serialized board:
 *     - Player 1 stack at (0,0,0): [1]
 *     - Player 2 stacks at (1,-1,0): [2, 2] and (0,-1,1): [2]
 * - Outside stack for Player 1:
 *     - (2,-2,0): [1]
 *
 * This gives:
 * - A small, contiguous hex region with mixed internal stacks (P1 + P2),
 * - At least one P1 stack outside the region, so the self-elimination
 *   prerequisite (FAQ Q23 / §12.2) is satisfied,
 * - No pre-collapsed territory or border markers, so tests can focus on
 *   canProcessTerritoryRegion and applyTerritoryRegion geometry and
 *   crediting semantics in isolation.
 */

import type { BoardState, Territory } from '../../src/shared/types/game';
import type { SerializedBoardState } from '../../src/shared/engine/contracts/serialization';
import { deserializeBoardState } from '../../src/shared/engine/contracts/serialization';

/**
 * Serialized hex board for HX-A. Only fields relevant to territory
 * processing are included: stacks, markers, collapsedSpaces,
 * eliminatedRings, size, and type.
 */
const hexTerritoryFeHxASerializedBoard: SerializedBoardState = {
  type: 'hexagonal',
  // Radius 12 hex board uses size 13 in the TS engine (see BOARD_CONFIGS.hexagonal).
  size: 13,
  stacks: {
    // Internal P1 stack inside the region.
    '0,0,0': {
      position: { x: 0, y: 0, z: 0 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    },
    // Internal P2 stack inside the region.
    '1,-1,0': {
      position: { x: 1, y: -1, z: 0 },
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    },
    // Internal P2 stack inside the region.
    '0,-1,1': {
      position: { x: 0, y: -1, z: 1 },
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    },
    // Outside P1 stack used for mandatory self-elimination and to satisfy
    // the self-elimination prerequisite (must have a stack outside).
    '2,-2,0': {
      position: { x: 2, y: -2, z: 0 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    },
  },
  markers: {},
  collapsedSpaces: {},
  eliminatedRings: {
    1: 0,
    2: 0,
  },
  formedLines: [],
};

/**
 * Reconstruct the HX-A hex BoardState used by hex territory/FE tests.
 */
export function createHexTerritoryFeBoardHxA(): BoardState {
  return deserializeBoardState(hexTerritoryFeHxASerializedBoard);
}

/**
 * HX-A curated hex territory region for player 1:
 *
 * - Spaces: compact three-cell region around the origin.
 * - controllingPlayer: 1 (acting player).
 * - isDisconnected: true (represents a disconnected region per detector).
 *
 * This region is not produced by the detector directly in tests; it is a
 * curated slice over the hex geometry, analogous to the SQ8-A fixture,
 * allowing us to probe applyTerritoryRegion and canProcessTerritoryRegion
 * on a compact, mixed-colour hex territory patch.
 */
export const hexTerritoryFeRegionHxAForPlayer1: Territory = {
  spaces: [
    { x: 0, y: 0, z: 0 },
    { x: 1, y: -1, z: 0 },
    { x: 0, y: -1, z: 1 },
  ],
  controllingPlayer: 1,
  isDisconnected: true,
};