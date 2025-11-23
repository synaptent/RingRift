import type { BoardState, Position } from './types';
import type { AdjacencyType } from '../types/game';
import { BOARD_CONFIGS, positionToString } from '../types/game';
import { SQUARE_MOORE_DIRECTIONS } from './core';

export type TerritoryBorderMode = 'ts_legacy' | 'rust_aligned';

export interface TerritoryBorderOptions {
  mode?: TerritoryBorderMode;
}

/****************************************************************************
 * Shared border-marker helper
 *
 * This module centralises the geometry used to expand a disconnected
 * territory region's marker border. Both the backend BoardManager and the
 * client sandbox delegate to this helper so that border enumeration is
 * defined in exactly one place.
 *
 * Behaviour (rust_aligned / ts_legacy):
 *   1) Seed border markers using the board's territory adjacency
 *      (BOARD_CONFIGS[board.type].territoryAdjacency) by looking for
 *      marker-owning spaces directly adjacent to any region space.
 *   2) For square boards, expand that seed set across connected markers
 *      using Moore (8-direction) adjacency, never stepping into the
 *      region itself.
 *   3) For hex boards, the effective behaviour of the current backend
 *      implementation is seed-only (no additional expansion), so the
 *      shared helper preserves that semantics.
 *
 * The returned list is de-duplicated and sorted in a stable
 * (row-major / cube-lexicographic) order to make tests deterministic.
 ****************************************************************************/

export function getBorderMarkerPositionsForRegion(
  board: BoardState,
  regionSpaces: Position[],
  opts?: TerritoryBorderOptions
): Position[] {
  const mode = opts?.mode ?? 'rust_aligned';

  switch (mode) {
    case 'ts_legacy':
    case 'rust_aligned':
    default:
      return getBorderMarkersRustAligned(board, regionSpaces);
  }
}

function getBorderMarkersRustAligned(board: BoardState, regionSpaces: Position[]): Position[] {
  if (regionSpaces.length === 0) {
    return [];
  }

  const regionSet = new Set(regionSpaces.map((p) => positionToString(p)));
  const config = BOARD_CONFIGS[board.type];
  const territoryAdjacency: AdjacencyType = config.territoryAdjacency;

  // Step 1: seed border markers = direct territory-adjacent markers.
  const seedMap = new Map<string, Position>();

  for (const space of regionSpaces) {
    const neighbors = getTerritoryNeighbors(board, space, territoryAdjacency);
    for (const neighbor of neighbors) {
      const key = positionToString(neighbor);
      if (regionSet.has(key)) continue;
      const marker = board.markers.get(key);
      if (marker && !seedMap.has(key)) {
        seedMap.set(key, neighbor);
      }
    }
  }

  if (seedMap.size === 0) {
    return [];
  }

  // Step 2: expand across connected markers using Moore adjacency on square
  // boards. For hex boards, the effective backend behaviour is seed-only, so
  // we intentionally skip any further expansion to preserve parity.
  const borderMarkers = new Map<string, Position>(seedMap);

  if (board.type !== 'hexagonal') {
    const queue: Position[] = Array.from(seedMap.values());
    const visited = new Set<string>(seedMap.keys());

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = getMooreNeighbors(board, current);

      for (const neighbor of neighbors) {
        const key = positionToString(neighbor);
        if (visited.has(key)) continue;
        if (regionSet.has(key)) continue; // never step into region

        const marker = board.markers.get(key);
        if (marker) {
          visited.add(key);
          borderMarkers.set(key, neighbor);
          queue.push(neighbor);
        }
      }
    }
  }

  const result = Array.from(borderMarkers.values());
  result.sort(comparePositionsStable);
  return result;
}

function getTerritoryNeighbors(
  board: BoardState,
  position: Position,
  adjacencyType: AdjacencyType
): Position[] {
  const neighbors: Position[] = [];
  const { x, y, z } = position;

  if (adjacencyType === 'hexagonal') {
    const directions = [
      { x: 1, y: 0, z: -1 },
      { x: 1, y: -1, z: 0 },
      { x: 0, y: -1, z: 1 },
      { x: -1, y: 0, z: 1 },
      { x: -1, y: 1, z: 0 },
      { x: 0, y: 1, z: -1 },
    ];
    for (const dir of directions) {
      const neighbor: Position = {
        x: x + dir.x,
        y: y + dir.y,
        z: (z ?? 0) + dir.z,
      };
      if (isValidPositionOnBoard(board, neighbor)) {
        neighbors.push(neighbor);
      }
    }
    return neighbors;
  }

  if (adjacencyType === 'von_neumann') {
    const directions = [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: -1, y: 0 },
    ];
    for (const dir of directions) {
      const neighbor: Position = { x: x + dir.x, y: y + dir.y };
      if (isValidPositionOnBoard(board, neighbor)) {
        neighbors.push(neighbor);
      }
    }
    return neighbors;
  }

  // Fallback: Moore adjacency on square boards (used only for any legacy
  // callers that configure territoryAdjacency as 'moore').
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      if (dx === 0 && dy === 0) continue;
      const neighbor: Position = { x: x + dx, y: y + dy };
      if (isValidPositionOnBoard(board, neighbor)) {
        neighbors.push(neighbor);
      }
    }
  }

  return neighbors;
}

function getMooreNeighbors(board: BoardState, position: Position): Position[] {
  const neighbors: Position[] = [];

  // Moore adjacency is only meaningful on square boards; for hex boards we
  // deliberately return an empty set so that border expansion remains
  // seed-only, matching the effective backend behaviour.
  if (board.type === 'hexagonal') {
    return neighbors;
  }

  for (const dir of SQUARE_MOORE_DIRECTIONS) {
    const neighbor: Position = {
      x: position.x + dir.x,
      y: position.y + dir.y,
    };
    if (isValidPositionOnBoard(board, neighbor)) {
      neighbors.push(neighbor);
    }
  }

  return neighbors;
}

function isValidPositionOnBoard(board: BoardState, position: Position): boolean {
  const size = board.size;

  if (board.type === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z ?? -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  }

  return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
}

function comparePositionsStable(a: Position, b: Position): number {
  // For pure square boards, sort row-major (y, then x).
  if (a.z === undefined && b.z === undefined) {
    return a.y - b.y || a.x - b.x;
  }

  // Cube-lexicographic ordering for hex coordinates (x, then y, then z).
  const az = a.z ?? -a.x - a.y;
  const bz = b.z ?? -b.x - b.y;
  return a.x - b.x || a.y - b.y || az - bz;
}
