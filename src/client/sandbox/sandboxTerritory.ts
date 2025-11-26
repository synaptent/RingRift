/**
 * Legacy sandbox territory helpers.
 *
 * **IMPORTANT:** This file exists primarily as a thin adapter layer between
 * the sandbox engine (ClientSandboxEngine) and the canonical shared territory
 * helpers. The actual territory semantics (region detection, collapse logic,
 * border enumeration) are defined in:
 *
 * - {@link ../../shared/engine/territoryDetection.ts} - Region detection
 * - {@link ../../shared/engine/territoryProcessing.ts} - Collapse logic
 * - {@link ../../shared/engine/territoryBorders.ts} - Border markers
 * - {@link ../../shared/engine/territoryDecisionHelpers.ts} - Decision helpers
 *
 * **Exported functions** in this file are wrappers that delegate to those
 * shared modules and remain in use by ClientSandboxEngine and sandboxTerritoryEngine.
 *
 * **Internal helpers** marked with `@deprecated` are DEAD CODE that was
 * superseded when the shared helpers were introduced. They are retained
 * temporarily for reference but will be removed in a future cleanup pass.
 *
 * Do NOT add new territory processing logic here; extend the shared helpers instead.
 *
 * @module client/sandbox/sandboxTerritory
 */
import type {
  BoardState,
  BoardType,
  Player,
  Position,
  Territory,
  RingStack,
  TerritoryProcessingContext,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  findDisconnectedRegions as findDisconnectedRegionsShared,
  getBorderMarkerPositionsForRegion as getSharedBorderMarkers,
  applyTerritoryRegion,
} from '../../shared/engine';
import { forceEliminateCapOnBoard, ForcedEliminationResult } from './sandboxElimination';

const TERRITORY_TRACE_DEBUG =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

const adjacencyCache = new Map<string, Map<string, string[]>>();

function getAdjacencyGraph(boardType: BoardType): Map<string, string[]> {
  if (!adjacencyCache.has(boardType)) {
    const graph = new Map<string, string[]>();
    const config = BOARD_CONFIGS[boardType];

    const positions: Position[] = [];
    if (boardType === 'hexagonal') {
      const radius = config.size - 1;
      for (let q = -radius; q <= radius; q++) {
        const r1 = Math.max(-radius, -q - radius);
        const r2 = Math.min(radius, -q + radius);
        for (let r = r1; r <= r2; r++) {
          const s = -q - r;
          positions.push({ x: q, y: r, z: s });
        }
      }
    } else {
      for (let x = 0; x < config.size; x++) {
        for (let y = 0; y < config.size; y++) {
          positions.push({ x, y });
        }
      }
    }

    // Dummy board for isValidPositionOnBoard check inside getTerritoryNeighbors
    const dummyBoard = { type: boardType } as BoardState;

    for (const pos of positions) {
      const posStr = positionToString(pos);
      const neighbors = getTerritoryNeighbors(dummyBoard, pos);
      graph.set(posStr, neighbors.map(positionToString));
    }

    adjacencyCache.set(boardType, graph);
  }
  return adjacencyCache.get(boardType)!;
}

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

  return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
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
      { x: 0, y: -1, z: 1 },
    ];

    for (const d of directions) {
      const n: Position = {
        x: pos.x + d.x,
        y: pos.y + d.y,
        z: (pos.z || 0) + d.z,
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
      { x: -1, y: 0 },
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

/**
 * @deprecated This internal helper is DEAD CODE and is not called anywhere.
 * Region detection now delegates to `findDisconnectedRegions` from
 * `src/shared/engine/territoryDetection.ts`.
 *
 * This function is retained temporarily for reference but will be removed
 * in a future cleanup pass.
 *
 * @internal
 */
function exploreRegionWithBorderColor(
  board: BoardState,
  start: Position,
  borderColor: number,
  visitedGlobal: Set<string>
): Position[] {
  const region: Position[] = [];
  const startKey = positionToString(start);
  const queue: string[] = [startKey];
  const localVisited = new Set<string>();
  const adjacencyGraph = getAdjacencyGraph(board.type);

  while (queue.length > 0) {
    const currentKey = queue.shift()!;
    if (localVisited.has(currentKey)) continue;
    localVisited.add(currentKey);
    visitedGlobal.add(currentKey);

    // Optimization: check directly against board maps using string key
    if (board.collapsedSpaces.has(currentKey)) continue;

    const marker = board.markers.get(currentKey);
    if (marker?.player === borderColor) continue;

    // This space is part of the region
    // Only convert to Position when adding to result
    // We need to parse it because the result expects Position[]
    // But we can optimize by parsing only once at the end?
    // No, we push to region array inside loop.
    // We can parse here.
    // Wait, we can use a helper to parse or just stringToPosition if available?
    // sandboxTerritory doesn't import stringToPosition?
    // It imports positionToString.
    // It doesn't import stringToPosition.
    // I need to check imports.
    // It imports positionToString from shared/types/game.
    // I should import stringToPosition too.
    // But wait, I can just parse it manually or import it.
    // Let's check imports again.
    // It imports positionToString.
    // I'll add stringToPosition to imports.

    // For now, I'll assume I can add it.
    // But wait, I can't change imports in this block easily without another block.
    // I'll use a local helper or just parse it.
    // Or I can just use the existing `getAllPositions` logic which returns Positions.
    // But here I have a key.
    // I'll add stringToPosition to imports in a separate block first.
    // Actually, I can just parse it: "x,y" or "x,y,z".
    const parts = currentKey.split(',').map(Number);
    const currentPos =
      parts.length === 3 ? { x: parts[0], y: parts[1], z: parts[2] } : { x: parts[0], y: parts[1] };

    region.push(currentPos);

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

/**
 * @deprecated This internal helper is DEAD CODE and is not called anywhere.
 * Region detection now delegates to `findDisconnectedRegions` from
 * `src/shared/engine/territoryDetection.ts`.
 *
 * This function is retained temporarily for reference but will be removed
 * in a future cleanup pass.
 *
 * @internal
 */
function exploreRegionWithoutMarkerBorder(
  board: BoardState,
  start: Position,
  visitedGlobal: Set<string>
): Position[] {
  const region: Position[] = [];
  const startKey = positionToString(start);
  const queue: string[] = [startKey];
  const localVisited = new Set<string>();
  const adjacencyGraph = getAdjacencyGraph(board.type);

  while (queue.length > 0) {
    const currentKey = queue.shift()!;
    if (localVisited.has(currentKey)) continue;
    localVisited.add(currentKey);
    visitedGlobal.add(currentKey);

    if (board.collapsedSpaces.has(currentKey)) continue;

    const parts = currentKey.split(',').map(Number);
    const currentPos =
      parts.length === 3 ? { x: parts[0], y: parts[1], z: parts[2] } : { x: parts[0], y: parts[1] };

    region.push(currentPos);

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

/**
 * @deprecated This internal helper is DEAD CODE and is not called anywhere.
 * Region detection now delegates to `findDisconnectedRegions` from
 * `src/shared/engine/territoryDetection.ts`.
 *
 * This function is retained temporarily for reference but will be removed
 * in a future cleanup pass.
 *
 * @internal
 */
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
        isDisconnected: true,
      });
    }
  }

  return disconnected;
}

/**
 * @deprecated This internal helper is DEAD CODE and is not called anywhere.
 * Region detection now delegates to `findDisconnectedRegions` from
 * `src/shared/engine/territoryDetection.ts`.
 *
 * This function is retained temporarily for reference but will be removed
 * in a future cleanup pass.
 *
 * @internal
 */
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
        isDisconnected: true,
      });
    }
  }

  return disconnected;
}

/**
 * Compute all disconnected regions on the board.
 *
 * Sandbox now delegates to the shared detector in
 * src/shared/engine/territoryDetection.ts so that region geometry is
 * defined in exactly one place for backend and client engines.
 */
export function findDisconnectedRegionsOnBoard(board: BoardState): Territory[] {
  return findDisconnectedRegionsShared(board);
}

/**
 * Identify all marker positions that form the border around a disconnected
 * region. This delegates to the shared helper in
 * src/shared/engine/territoryBorders.ts so backend and sandbox cannot
 * drift.
 */
export function getBorderMarkerPositionsForRegion(
  board: BoardState,
  regionSpaces: Position[]
): Position[] {
  return getSharedBorderMarkers(board, regionSpaces, { mode: 'rust_aligned' });
}

/**
 * Process a single disconnected region for the moving player, applying the
 * collapse and elimination rules described in Section 6 of the compact
 * rules. This function is pure with respect to GameState: callers must
 * update totalRingsEliminated themselves using the returned delta.
 */
export function processDisconnectedRegionCoreOnBoard(
  board: BoardState,
  players: Player[],
  movingPlayer: number,
  regionSpaces: Position[]
): { board: BoardState; players: Player[]; totalRingsEliminatedDelta: number } {
  // Delegate the geometric core (internal eliminations + region/border
  // collapse) to the shared engine helper so that backend, sandbox, and
  // rules-layer tests all share a single source of truth for territory
  // semantics.
  const region: Territory = {
    spaces: regionSpaces,
    controllingPlayer: movingPlayer,
    isDisconnected: true,
  };

  const ctx: TerritoryProcessingContext = { player: movingPlayer };
  const outcome = applyTerritoryRegion(board, region, ctx);

  let nextPlayers: Player[] = players.map((p) => ({ ...p }));

  // Update territorySpaces for moving player based on the shared outcome.
  const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
  if (territoryGain > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === movingPlayer
        ? { ...p, territorySpaces: p.territorySpaces + territoryGain }
        : p
    );
  }

  // Credit all internal eliminations to the moving player at the Player[]
  // level. BoardState-level eliminatedRings has already been updated by
  // applyTerritoryRegion.
  const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
  if (internalElims > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === movingPlayer
        ? { ...p, eliminatedRings: p.eliminatedRings + internalElims }
        : p
    );
  }

  return {
    board: outcome.board,
    players: nextPlayers,
    totalRingsEliminatedDelta: internalElims,
  };
}

export function processDisconnectedRegionOnBoard(
  board: BoardState,
  players: Player[],
  movingPlayer: number,
  regionSpaces: Position[]
): { board: BoardState; players: Player[]; totalRingsEliminatedDelta: number } {
  // Legacy / non-move-driven helper: apply the core geometric consequences of
  // processing a disconnected region and then immediately perform the mandatory
  // self-elimination using forceEliminateCapOnBoard. This preserves the
  // original sandbox semantics used by processDisconnectedRegionsForCurrentPlayerEngine.
  const coreResult = processDisconnectedRegionCoreOnBoard(
    board,
    players,
    movingPlayer,
    regionSpaces
  );

  const nextBoard = coreResult.board;
  const nextPlayers = coreResult.players;

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

  const totalDelta = coreResult.totalRingsEliminatedDelta + elimResult.totalRingsEliminatedDelta;

  return {
    board: elimResult.board,
    players: elimResult.players,
    totalRingsEliminatedDelta: totalDelta,
  };
}
