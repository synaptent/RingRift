import {
  BoardState,
  Player,
  Position,
  Territory,
  BOARD_CONFIGS,
  positionToString,
  RingStack
} from '../../shared/types/game';
import { forceEliminateCapOnBoard, ForcedEliminationResult } from './sandboxElimination';

/**
 * Sandbox territory helpers
 *
 * These are browser-safe, pure counterparts to the server-side
 * BoardManager.findDisconnectedRegions and GameEngine territory
 * processing logic. They operate directly on BoardState/Player
 * structures and use BOARD_CONFIGS for geometry.
 */

function isValidPositionOnBoard(board: BoardState, pos: Position): boolean {
  const config = BOARD_CONFIGS[board.type];

  if (board.type === 'hexagonal') {
    const radius = config.size - 1;
    const x = pos.x;
    const y = pos.y;
    const z = pos.z !== undefined ? pos.z : -x - y;
    const dist = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
    return dist <= radius;
  }

  return (
    pos.x >= 0 &&
    pos.x < config.size &&
    pos.y >= 0 &&
    pos.y < config.size
  );
}

function getTerritoryNeighbors(board: BoardState, pos: Position): Position[] {
  const config = BOARD_CONFIGS[board.type];
  const adj = config.territoryAdjacency;
  const result: Position[] = [];

  if (adj === 'hexagonal') {
    const directions = [
      { x: 1, y: -1, z: 0 },
      { x: 1, y: 0, z: -1 },
      { x: 0, y: 1, z: -1 },
      { x: -1, y: 1, z: 0 },
      { x: -1, y: 0, z: 1 },
      { x: 0, y: -1, z: 1 }
    ];

    for (const d of directions) {
      const n: Position = {
        x: pos.x + d.x,
        y: pos.y + d.y,
        z: (pos.z || 0) + d.z
      };
      if (isValidPositionOnBoard(board, n)) {
        result.push(n);
      }
    }
    return result;
  }

  if (adj === 'von_neumann') {
    const dirs = [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: -1, y: 0 }
    ];
    for (const d of dirs) {
      const n: Position = { x: pos.x + d.x, y: pos.y + d.y };
      if (isValidPositionOnBoard(board, n)) {
        result.push(n);
      }
    }
    return result;
  }

  // Moore adjacency (used for some legacy territory exploration on square boards)
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      if (dx === 0 && dy === 0) continue;
      const n: Position = { x: pos.x + dx, y: pos.y + dy };
      if (isValidPositionOnBoard(board, n)) {
        result.push(n);
      }
    }
  }

  return result;
}

function getAllPositions(board: BoardState): Position[] {
  const config = BOARD_CONFIGS[board.type];

  if (board.type === 'hexagonal') {
    const radius = config.size - 1;
    const positions: Position[] = [];
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        positions.push({ x: q, y: r, z: s });
      }
    }
    return positions;
  }

  const res: Position[] = [];
  for (let x = 0; x < config.size; x++) {
    for (let y = 0; y < config.size; y++) {
      res.push({ x, y });
    }
  }
  return res;
}

function isCollapsedSpace(board: BoardState, pos: Position): boolean {
  const key = positionToString(pos);
  return board.collapsedSpaces.has(key);
}

function getMarkerOwner(board: BoardState, pos: Position): number | undefined {
  const key = positionToString(pos);
  const marker = board.markers.get(key);
  return marker?.player;
}

function getStackAt(board: BoardState, pos: Position) {
  const key = positionToString(pos);
  return board.stacks.get(key);
}

function getRepresentedPlayers(regionSpaces: Position[], board: BoardState): Set<number> {
  const represented = new Set<number>();
  for (const space of regionSpaces) {
    const stack = getStackAt(board, space);
    if (stack) {
      represented.add(stack.controllingPlayer);
    }
  }
  return represented;
}

function exploreRegionWithBorderColor(
  board: BoardState,
  start: Position,
  borderColor: number,
  visitedGlobal: Set<string>
): Position[] {
  const region: Position[] = [];
  const queue: Position[] = [start];
  const localVisited = new Set<string>();

  while (queue.length > 0) {
    const current = queue.shift()!;
    const key = positionToString(current);
    if (localVisited.has(key)) continue;
    localVisited.add(key);
    visitedGlobal.add(key);

    if (isCollapsedSpace(board, current)) continue;

    const marker = getMarkerOwner(board, current);
    if (marker === borderColor) continue;

    region.push(current);

    const neighbors = getTerritoryNeighbors(board, current);
    for (const n of neighbors) {
      const nk = positionToString(n);
      if (!localVisited.has(nk) && isValidPositionOnBoard(board, n)) {
        queue.push(n);
      }
    }
  }

  return region;
}

function exploreRegionWithoutMarkerBorder(
  board: BoardState,
  start: Position,
  visitedGlobal: Set<string>
): Position[] {
  const region: Position[] = [];
  const queue: Position[] = [start];
  const localVisited = new Set<string>();

  while (queue.length > 0) {
    const current = queue.shift()!;
    const key = positionToString(current);
    if (localVisited.has(key)) continue;
    localVisited.add(key);
    visitedGlobal.add(key);

    if (isCollapsedSpace(board, current)) continue;

    region.push(current);

    const neighbors = getTerritoryNeighbors(board, current);
    for (const n of neighbors) {
      const nk = positionToString(n);
      if (!localVisited.has(nk) && isValidPositionOnBoard(board, n)) {
        queue.push(n);
      }
    }
  }

  return region;
}

function isRegionBorderedByCollapsedOnly(regionSpaces: Position[], board: BoardState): boolean {
  const regionSet = new Set(regionSpaces.map(positionToString));

  for (const space of regionSpaces) {
    const neighbors = getTerritoryNeighbors(board, space);
    for (const n of neighbors) {
      const nk = positionToString(n);
      if (regionSet.has(nk)) continue;

      if (!isValidPositionOnBoard(board, n)) {
        // Board edge is acceptable as a border.
        continue;
      }

      if (isCollapsedSpace(board, n)) {
        // Collapsed spaces are acceptable as border.
        continue;
      }

      // If there is a marker at the border, region is not "collapsed-only".
      if (getMarkerOwner(board, n) !== undefined) {
        return false;
      }

      // Empty or stack-bearing neighbors break the collapsed-only condition.
      return false;
    }
  }

  return true;
}

function findRegionsWithBorderColor(
  board: BoardState,
  borderColor: number,
  activePlayers: Set<number>
): Territory[] {
  const disconnected: Territory[] = [];
  const visited = new Set<string>();

  for (const pos of getAllPositions(board)) {
    const key = positionToString(pos);
    if (visited.has(key)) continue;

    if (isCollapsedSpace(board, pos)) {
      visited.add(key);
      continue;
    }

    const marker = getMarkerOwner(board, pos);
    if (marker === borderColor) {
      visited.add(key);
      continue;
    }

    const regionSpaces = exploreRegionWithBorderColor(board, pos, borderColor, visited);
    if (regionSpaces.length === 0) continue;

    const represented = getRepresentedPlayers(regionSpaces, board);
    if (represented.size < activePlayers.size) {
      disconnected.push({
        spaces: regionSpaces,
        controllingPlayer: 0,
        isDisconnected: true
      });
    }
  }

  return disconnected;
}

function findRegionsWithoutMarkerBorder(
  board: BoardState,
  activePlayers: Set<number>
): Territory[] {
  const disconnected: Territory[] = [];
  const visited = new Set<string>();

  for (const pos of getAllPositions(board)) {
    const key = positionToString(pos);
    if (visited.has(key)) continue;

    if (isCollapsedSpace(board, pos)) {
      visited.add(key);
      continue;
    }

    const regionSpaces = exploreRegionWithoutMarkerBorder(board, pos, visited);
    if (regionSpaces.length === 0) continue;

    if (!isRegionBorderedByCollapsedOnly(regionSpaces, board)) {
      continue;
    }

    const represented = getRepresentedPlayers(regionSpaces, board);
    if (represented.size < activePlayers.size) {
      disconnected.push({
        spaces: regionSpaces,
        controllingPlayer: 0,
        isDisconnected: true
      });
    }
  }

  return disconnected;
}

/**
 * Compute all disconnected regions on the board according to the compact
 * rules:
 * - Physically disconnected by a single marker color + edges/collapsed
 *   spaces, or by collapsed spaces + edges alone.
 * - Color-disconnected when at least one active player has no stack
 *   inside the region.
 */
export function findDisconnectedRegionsOnBoard(board: BoardState): Territory[] {
  const disconnected: Territory[] = [];

  const activePlayers = new Set<number>();
  for (const stack of board.stacks.values()) {
    activePlayers.add(stack.controllingPlayer);
  }

  if (activePlayers.size === 0) {
    return [];
  }

  const markerColors = new Set<number>();
  for (const marker of board.markers.values()) {
    markerColors.add(marker.player);
  }

  for (const borderColor of markerColors) {
    const regions = findRegionsWithBorderColor(board, borderColor, activePlayers);
    disconnected.push(...regions);
  }

  const collapsedOnlyRegions = findRegionsWithoutMarkerBorder(board, activePlayers);
  disconnected.push(...collapsedOnlyRegions);

  return disconnected;
}

/**
 * Identify all marker positions that form the border around a disconnected
 * region. This mirrors BoardManager.getBorderMarkerPositions and is used
 * when collapsing a region to determine which border markers should become
 * territory for the moving player.
 */
export function getBorderMarkerPositionsForRegion(
  board: BoardState,
  regionSpaces: Position[]
): Position[] {
  const regionSet = new Set(regionSpaces.map(positionToString));
  const config = BOARD_CONFIGS[board.type];

  // Step 1: seed border markers = direct territory-adjacent markers.
  const borderSeedMap = new Map<string, Position>();

  for (const space of regionSpaces) {
    const neighbors = getTerritoryNeighbors(board, space);
    for (const n of neighbors) {
      const nk = positionToString(n);
      if (regionSet.has(nk)) continue;
      const markerOwner = getMarkerOwner(board, n);
      if (markerOwner !== undefined && !borderSeedMap.has(nk)) {
        borderSeedMap.set(nk, n);
      }
    }
  }

  if (borderSeedMap.size === 0) {
    return [];
  }

  // Step 2: expand across connected markers using Moore adjacency for
  // square boards and hex adjacency for hex boards.
  const borderMarkers = new Map<string, Position>(borderSeedMap);
  const queue: Position[] = Array.from(borderSeedMap.values());
  const visited = new Set<string>(borderSeedMap.keys());

  const mooreDirs = [
    { x: -1, y: -1 },
    { x: -1, y: 0 },
    { x: -1, y: 1 },
    { x: 0, y: -1 },
    { x: 0, y: 1 },
    { x: 1, y: -1 },
    { x: 1, y: 0 },
    { x: 1, y: 1 }
  ];

  const hexDirs = [
    { x: 1, y: -1, z: 0 },
    { x: 1, y: 0, z: -1 },
    { x: 0, y: 1, z: -1 },
    { x: -1, y: 1, z: 0 },
    { x: -1, y: 0, z: 1 },
    { x: 0, y: -1, z: 1 }
  ];

  while (queue.length > 0) {
    const current = queue.shift()!;
    const dirs = config.type === 'hexagonal' ? hexDirs : mooreDirs;

    for (const d of dirs) {
      const n: Position = config.type === 'hexagonal'
        ? { x: current.x + d.x, y: current.y + d.y, z: (current.z || 0) + (d as any).z }
        : { x: current.x + (d as any).x, y: current.y + (d as any).y };

      const nk = positionToString(n);
      if (visited.has(nk)) continue;
      if (regionSet.has(nk)) continue; // never step into region
      if (!isValidPositionOnBoard(board, n)) continue;

      const markerOwner = getMarkerOwner(board, n);
      if (markerOwner !== undefined) {
        visited.add(nk);
        borderMarkers.set(nk, n);
        queue.push(n);
      }
    }
  }

  return Array.from(borderMarkers.values());
}

/**
 * Process a single disconnected region for the moving player, applying the
 * collapse and elimination rules described in Section 6 of the compact
 * rules. This function is pure with respect to GameState: callers must
 * update totalRingsEliminated themselves using the returned delta.
 */
export function processDisconnectedRegionOnBoard(
  board: BoardState,
  players: Player[],
  movingPlayer: number,
  regionSpaces: Position[]
): { board: BoardState; players: Player[]; totalRingsEliminatedDelta: number } {
  let nextBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: { ...board.eliminatedRings }
  };

  let nextPlayers: Player[] = players.map(p => ({ ...p }));

  // 1. Determine border markers that participate in this region's boundary.
  const borderMarkers = getBorderMarkerPositionsForRegion(nextBoard, regionSpaces);

  // 2. Eliminate all rings inside the region before collapsing spaces.
  let internalRingsEliminated = 0;
  for (const pos of regionSpaces) {
    const key = positionToString(pos);
    const stack = nextBoard.stacks.get(key);
    if (stack) {
      internalRingsEliminated += stack.stackHeight;
      nextBoard.stacks.delete(key);
    }
  }

  // 3. Collapse all spaces in the region to the moving player's color.
  for (const pos of regionSpaces) {
    const key = positionToString(pos);
    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, movingPlayer);
  }

  // 4. Collapse all border markers to the moving player's color.
  for (const pos of borderMarkers) {
    const key = positionToString(pos);
    nextBoard.markers.delete(key);
    nextBoard.stacks.delete(key);
    nextBoard.collapsedSpaces.set(key, movingPlayer);
  }

  // Update territorySpaces for moving player.
  const territoryGain = regionSpaces.length + borderMarkers.length;
  nextPlayers = nextPlayers.map(p =>
    p.playerNumber === movingPlayer
      ? { ...p, territorySpaces: p.territorySpaces + territoryGain }
      : p
  );

  // 5. Credit all internal eliminations to the moving player.
  if (internalRingsEliminated > 0) {
    const newEliminated = { ...nextBoard.eliminatedRings };
    newEliminated[movingPlayer] =
      (newEliminated[movingPlayer] || 0) + internalRingsEliminated;
    nextBoard = { ...nextBoard, eliminatedRings: newEliminated };

    nextPlayers = nextPlayers.map(p =>
      p.playerNumber === movingPlayer
        ? { ...p, eliminatedRings: p.eliminatedRings + internalRingsEliminated }
        : p
    );
  }

  // 6. Mandatory self-elimination: eliminate one cap from a moving-player stack
  // outside the region using the shared forceEliminateCapOnBoard helper.
  const movingStacks: RingStack[] = [];
  for (const stack of nextBoard.stacks.values()) {
    if (stack.controllingPlayer === movingPlayer) {
      movingStacks.push(stack);
    }
  }

  const elimResult: ForcedEliminationResult = forceEliminateCapOnBoard(
    nextBoard,
    nextPlayers,
    movingPlayer,
    movingStacks
  );

  const totalDelta = internalRingsEliminated + elimResult.totalRingsEliminatedDelta;

  return {
    board: elimResult.board,
    players: elimResult.players,
    totalRingsEliminatedDelta: totalDelta
  };
}
