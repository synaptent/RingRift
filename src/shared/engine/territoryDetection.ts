import {
  BoardState,
  Territory,
  Position,
  RingStack,
  BOARD_CONFIGS,
  positionToString,
  stringToPosition,
} from '../types/game';

/** Minimal board shape needed for adjacency/position generation. */
type MinimalBoardConfig = Pick<BoardState, 'type' | 'size'>;

const adjacencyCache = new Map<string, Map<string, string[]>>();

function getAdjacencyGraph(boardType: string): Map<string, string[]> {
  const cached = adjacencyCache.get(boardType);
  if (cached) return cached;

  const graph = new Map<string, string[]>();
  const config = BOARD_CONFIGS[boardType as keyof typeof BOARD_CONFIGS];
  const adjType = config.territoryAdjacency;

  // Generate valid positions based on config
  const minimalBoard: MinimalBoardConfig = {
    type: boardType as BoardState['type'],
    size: config.size,
  };
  const positions = generateValidPositions(minimalBoard as BoardState);

  for (const posStr of positions) {
    const pos = stringToPosition(posStr);
    const neighbors = getNeighbors(pos, adjType, minimalBoard as BoardState);
    graph.set(posStr, neighbors.map(positionToString));
  }

  adjacencyCache.set(boardType, graph);
  return graph;
}

/**
 * Find all disconnected regions on the board
 * Rule Reference: Section 12.2 - Territory Disconnection
 */
export function findDisconnectedRegions(board: BoardState): Territory[] {
  const disconnectedRegions: Territory[] = [];

  // Get all active players (those with stacks on board)
  const activePlayers = new Set<number>();
  for (const [, stack] of board.stacks) {
    if (stack) {
      activePlayers.add(stack.controllingPlayer);
    }
  }

  // PARITY FIX: If there is only one or zero active players, there is no
  // meaningful notion of disconnection. One player cannot be "cut off" from
  // themselves. This matches Python's find_disconnected_regions behavior.
  if (activePlayers.size <= 1) {
    return [];
  }

  // Get all marker colors present on board
  const markerColors = new Set<number>();
  for (const [, marker] of board.markers) {
    markerColors.add(marker.player);
  }

  // Check for disconnection with respect to each marker color
  for (const borderColor of markerColors) {
    const regions = findRegionsWithBorderColor(board, borderColor, activePlayers);
    disconnectedRegions.push(...regions);
  }

  // Also check for regions surrounded by only collapsed spaces and edges (no marker borders)
  const regionsWithoutMarkerBorder = findRegionsWithoutMarkerBorder(board, activePlayers);
  disconnectedRegions.push(...regionsWithoutMarkerBorder);

  return disconnectedRegions;
}

function findRegionsWithBorderColor(
  board: BoardState,
  borderColor: number,
  activePlayers: Set<number>
): Territory[] {
  const disconnectedRegions: Territory[] = [];
  const visited = new Set<string>();
  const validPositions = generateValidPositions(board);

  for (const posStr of validPositions) {
    if (visited.has(posStr)) continue;

    const position = stringToPosition(posStr);

    // Skip if this is a border (collapsed or borderColor marker)
    if (isCollapsedSpace(position, board)) {
      visited.add(posStr);
      continue;
    }

    const marker = getMarker(position, board);
    if (marker === borderColor) {
      visited.add(posStr);
      continue;
    }

    // Find connected region using flood fill
    const region = exploreRegionWithBorderColor(position, board, borderColor, visited);

    if (region.length === 0) continue;

    // Check representation
    const representedPlayers = getRepresentedPlayers(region, board);

    if (representedPlayers.size < activePlayers.size) {
      // Region is bounded by borderColor markers; attribute control to that player.
      disconnectedRegions.push({
        spaces: region,
        controllingPlayer: borderColor,
        isDisconnected: true,
      });
    }
  }

  return disconnectedRegions;
}

function findRegionsWithoutMarkerBorder(
  board: BoardState,
  activePlayers: Set<number>
): Territory[] {
  const disconnectedRegions: Territory[] = [];
  const visited = new Set<string>();
  const validPositions = generateValidPositions(board);

  for (const posStr of validPositions) {
    if (visited.has(posStr)) continue;

    const position = stringToPosition(posStr);

    if (isCollapsedSpace(position, board)) {
      visited.add(posStr);
      continue;
    }

    const region = exploreRegionWithoutMarkerBorder(position, board, visited);

    if (region.length === 0) continue;

    if (!isRegionBorderedByCollapsedOnly(region, board)) {
      continue;
    }

    const representedPlayers = getRepresentedPlayers(region, board);

    if (representedPlayers.size < activePlayers.size) {
      // If exactly one player is represented inside, attribute control to that player.
      // Otherwise skip this ambiguous region to avoid non-canonical neutral territories.
      if (representedPlayers.size === 1) {
        const [solePlayer] = Array.from(representedPlayers);
        disconnectedRegions.push({
          spaces: region,
          controllingPlayer: solePlayer,
          isDisconnected: true,
        });
      }
    }
  }

  return disconnectedRegions;
}

function exploreRegionWithBorderColor(
  startPosition: Position,
  board: BoardState,
  borderColor: number,
  visited: Set<string>
): Position[] {
  const region: Position[] = [];
  const startKey = positionToString(startPosition);
  const queue: string[] = [startKey];
  const localVisited = new Set<string>();
  const adjacencyGraph = getAdjacencyGraph(board.type);

  while (queue.length > 0) {
    const currentKey = queue.shift();
    if (!currentKey) continue;

    if (localVisited.has(currentKey)) continue;
    localVisited.add(currentKey);
    visited.add(currentKey);

    // Check if this is a border
    // Optimization: check directly against board maps using string key
    if (board.collapsedSpaces.has(currentKey)) continue;

    const marker = board.markers.get(currentKey);
    if (marker?.player === borderColor) continue;

    // This space is part of the region
    region.push(stringToPosition(currentKey));

    // Explore neighbors using cached graph
    const neighbors = adjacencyGraph.get(currentKey);
    if (neighbors) {
      for (const neighborKey of neighbors) {
        if (!localVisited.has(neighborKey)) {
          queue.push(neighborKey);
        }
      }
    }
  }

  return region;
}

function exploreRegionWithoutMarkerBorder(
  startPosition: Position,
  board: BoardState,
  visited: Set<string>
): Position[] {
  const region: Position[] = [];
  const startKey = positionToString(startPosition);
  const queue: string[] = [startKey];
  const localVisited = new Set<string>();
  const adjacencyGraph = getAdjacencyGraph(board.type);

  while (queue.length > 0) {
    const currentKey = queue.shift();
    if (!currentKey) continue;

    if (localVisited.has(currentKey)) continue;
    localVisited.add(currentKey);
    visited.add(currentKey);

    // Check if this is a border (only collapsed spaces)
    if (board.collapsedSpaces.has(currentKey)) continue;

    // This space is part of the region
    region.push(stringToPosition(currentKey));

    // Explore neighbors using cached graph
    const neighbors = adjacencyGraph.get(currentKey);
    if (neighbors) {
      for (const neighborKey of neighbors) {
        if (!localVisited.has(neighborKey)) {
          queue.push(neighborKey);
        }
      }
    }
  }

  return region;
}

function isRegionBorderedByCollapsedOnly(regionSpaces: Position[], board: BoardState): boolean {
  const regionSet = new Set(regionSpaces.map(positionToString));
  const adjacencyType = BOARD_CONFIGS[board.type].territoryAdjacency;

  for (const space of regionSpaces) {
    const neighbors = getNeighbors(space, adjacencyType, board);
    for (const neighbor of neighbors) {
      const neighborKey = positionToString(neighbor);

      if (regionSet.has(neighborKey)) continue;

      if (!isValidPosition(neighbor, board)) continue; // Edge is OK

      if (isCollapsedSpace(neighbor, board)) continue; // Collapsed is OK

      if (getMarker(neighbor, board) !== undefined) {
        return false;
      }

      return false;
    }
  }

  return true;
}

function getRepresentedPlayers(regionSpaces: Position[], board: BoardState): Set<number> {
  const represented = new Set<number>();

  for (const space of regionSpaces) {
    const stack = getStack(space, board);
    if (stack) {
      represented.add(stack.controllingPlayer);
    }
  }

  return represented;
}

// Helpers

function generateValidPositions(board: BoardState): Set<string> {
  const positions = new Set<string>();
  const size = board.size;

  if (board.type === 'hexagonal') {
    const radius = size - 1;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        positions.add(positionToString({ x: q, y: r, z: s }));
      }
    }
  } else {
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        positions.add(positionToString({ x, y }));
      }
    }
  }
  return positions;
}

function isValidPosition(position: Position, board: BoardState): boolean {
  const size = board.size;
  if (board.type === 'hexagonal') {
    const radius = size - 1;
    const q = position.x;
    const r = position.y;
    const s = position.z || -q - r;
    return (
      Math.abs(q) <= radius && Math.abs(r) <= radius && Math.abs(s) <= radius && q + r + s === 0
    );
  } else {
    return position.x >= 0 && position.x < size && position.y >= 0 && position.y < size;
  }
}

function getNeighbors(position: Position, adjacencyType: string, board: BoardState): Position[] {
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
      const neighbor = { x: x + dir.x, y: y + dir.y, z: (z || 0) + dir.z };
      if (isValidPosition(neighbor, board)) neighbors.push(neighbor);
    }
  } else if (adjacencyType === 'von_neumann') {
    const directions = [
      { x: 0, y: 1 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: -1, y: 0 },
    ];
    for (const dir of directions) {
      const neighbor = { x: x + dir.x, y: y + dir.y };
      if (isValidPosition(neighbor, board)) neighbors.push(neighbor);
    }
  } else if (adjacencyType === 'moore') {
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;
        const neighbor = { x: x + dx, y: y + dy };
        if (isValidPosition(neighbor, board)) neighbors.push(neighbor);
      }
    }
  }
  return neighbors;
}

function getMarker(position: Position, board: BoardState): number | undefined {
  const posKey = positionToString(position);
  const marker = board.markers.get(posKey);
  return marker?.player;
}

function isCollapsedSpace(position: Position, board: BoardState): boolean {
  const posKey = positionToString(position);
  return board.collapsedSpaces.has(posKey);
}

function getStack(position: Position, board: BoardState): RingStack | undefined {
  const posKey = positionToString(position);
  return board.stacks.get(posKey);
}
