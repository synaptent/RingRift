import type {
  Position,
  BoardType,
  BoardState,
  RingStack,
  Territory,
  LineInfo,
  AdjacencyType,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  positionToString,
  stringToPosition,
  findDisconnectedRegions as findDisconnectedRegionsShared,
  findAllLines as findAllLinesShared,
  findLinesForPlayer as findLinesForPlayerShared,
} from '../../shared/engine';
import { getBorderMarkerPositionsForRegion as getSharedBorderMarkers } from '../../shared/engine/territoryBorders';
import { flagEnabled, isTestEnvironment, debugLog } from '../../shared/utils/envFlags';

const TERRITORY_TRACE_DEBUG = flagEnabled('RINGRIFT_TRACE_DEBUG');

// When true, backend BoardManager will treat board invariant violations as
// hard errors (throwing in tests) rather than best-effort diagnostics. This
// mirrors the ClientSandboxEngine assertBoardInvariants helper and is used
// by rules/parity tests to surface exclusivity bugs early.
//
// Test-only override: allows specific tests to simulate non-strict mode.
let _testStrictModeOverride: boolean | undefined = undefined;

/**
 * Test-only helper: allows tests to temporarily override the strict mode
 * setting. Pass `undefined` to restore default behaviour.
 */
export function __testSetStrictModeOverride(value: boolean | undefined): void {
  _testStrictModeOverride = value;
}

function isBoardInvariantsStrict(): boolean {
  if (_testStrictModeOverride !== undefined) {
    return _testStrictModeOverride;
  }
  return isTestEnvironment() || flagEnabled('RINGRIFT_ENABLE_BACKEND_BOARD_INVARIANTS');
}

export class BoardManager {
  private boardType: BoardType;
  private size: number;
  private config: (typeof BOARD_CONFIGS)[BoardType];
  private validPositions: Set<string>;
  private adjacencyGraph: Map<string, string[]> = new Map();

  // Internal counter used to track how many "repair" actions this
  // BoardManager instance has performed while attempting to restore
  // board invariants. A non-zero value is always a defect signal in
  // tests: legal trajectories must never rely on repairs.
  private repairCount: number = 0;

  /**
   * Internal helper used by invariant checks to record when we have to
   * "repair" an illegal board state (for example, stack+marker overlap).
   * This keeps the corrective behaviour itself unchanged while making
   * every repair observable to tests and debug tooling.
   */
  private logRepair(kind: string, details?: unknown): void {
    this.repairCount++;

    // Keep runtime logging behaviour very similar to the previous
    // console.error calls so existing diagnostics remain useful, but
    // funnel everything through a single structured hook.
    console.error('[BoardManager] board repair', {
      kind,
      details,
    });
  }

  /**
   * Test-only helper: expose the number of repairs performed by this
   * BoardManager instance so invariant tests can assert that legal
   * trajectories never trigger repairs.
   */
  public getRepairCountForTesting(): number {
    return this.repairCount;
  }

  constructor(boardType: BoardType) {
    this.boardType = boardType;
    this.config = BOARD_CONFIGS[boardType];
    this.size = this.config.size;
    this.validPositions = this.generateValidPositions();
    this.buildAdjacencyGraph();
  }

  /**
   * Internal board invariants used for defensive checks in tests and
   * strict/dev modes. Each cell must be in exactly one of the following
   * categories:
   *   - empty
   *   - occupied by a stack
   *   - occupied by a marker
   *   - collapsed territory
   *
   * In particular:
   *   1) No stacks may exist on collapsed territory.
   *   2) A cell may not host both a stack and a marker.
   *   3) A cell may not host both a marker and collapsed territory.
   */
  private assertBoardInvariants(board: BoardState, context: string): void {
    const errors: string[] = [];

    // Defensive repair pass: ensure that stacks and markers never coexist on
    // the same cell, and that markers never coexist with collapsed territory.
    // If we detect such a state, we log a diagnostic and repair the board
    // before running the stricter invariant checks below. This keeps the
    // runtime board geometry consistent even if an earlier rule path produced
    // a borderline state, while still surfacing anomalies in test logs.
    for (const key of board.stacks.keys()) {
      if (board.markers.has(key)) {
        this.logRepair('stack_marker_overlap', {
          context,
          key,
          source: 'assertBoardInvariants',
        });
        board.markers.delete(key);
      }
    }

    for (const key of board.markers.keys()) {
      if (board.collapsedSpaces.has(key)) {
        this.logRepair('marker_on_collapsed_space', {
          context,
          key,
          source: 'assertBoardInvariants',
        });
        board.markers.delete(key);
      }
    }

    // Invariant 1 + 2: stack vs collapsed / marker exclusivity
    for (const key of board.stacks.keys()) {
      if (board.collapsedSpaces.has(key)) {
        errors.push(`stack present on collapsed space at ${key}`);
      }
      if (board.markers.has(key)) {
        errors.push(`stack and marker coexist at ${key}`);
      }
    }

    // Invariant 3: marker vs collapsed exclusivity
    for (const key of board.markers.keys()) {
      if (board.collapsedSpaces.has(key)) {
        errors.push(`marker present on collapsed space at ${key}`);
      }
    }

    if (errors.length === 0) {
      return;
    }

    const message = `[BoardManager] invariant violation (${context}):` + '\n' + errors.join('\n');

    console.error(message);

    if (isBoardInvariantsStrict()) {
      throw new Error(message);
    }
  }

  private buildAdjacencyGraph(): void {
    // Pre-calculate neighbors for territory adjacency (used in hot path)
    const adjType = this.config.territoryAdjacency;
    for (const posStr of this.validPositions) {
      const pos = stringToPosition(posStr);
      const neighbors = this.getNeighbors(pos, adjType);
      this.adjacencyGraph.set(posStr, neighbors.map(positionToString));
    }
  }

  createBoard(): BoardState {
    // Starting from a fresh board state, also reset the internal repair
    // counter so that tests can reason about repairs on a per-board basis.
    this.repairCount = 0;

    return {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: this.size,
      type: this.boardType,
    };
  }

  private generateValidPositions(): Set<string> {
    const positions = new Set<string>();

    if (this.boardType === 'hexagonal') {
      // Generate hexagonal board positions
      // size=13 means radius 12 (positions from -12 to 12)
      // This gives 3*12^2 + 3*12 + 1 = 469 positions
      const radius = this.size - 1;
      for (let q = -radius; q <= radius; q++) {
        const r1 = Math.max(-radius, -q - radius);
        const r2 = Math.min(radius, -q + radius);
        for (let r = r1; r <= r2; r++) {
          const s = -q - r;
          // Convert cube coordinates to axial coordinates
          positions.add(positionToString({ x: q, y: r, z: s }));
        }
      }
    } else {
      // Generate square board positions
      for (let x = 0; x < this.size; x++) {
        for (let y = 0; y < this.size; y++) {
          positions.add(positionToString({ x, y }));
        }
      }
    }

    return positions;
  }

  isValidPosition(position: Position): boolean {
    return this.validPositions.has(positionToString(position));
  }

  // Get neighbors based on adjacency type
  getNeighbors(position: Position, adjacencyType?: AdjacencyType): Position[] {
    const adjType = adjacencyType || this.config.movementAdjacency;

    if (this.boardType === 'hexagonal' || adjType === 'hexagonal') {
      return this.getHexagonalNeighbors(position);
    } else if (adjType === 'moore') {
      return this.getMooreNeighbors(position);
    } else if (adjType === 'von_neumann') {
      return this.getVonNeumannNeighbors(position);
    }

    return [];
  }

  private getHexagonalNeighbors(position: Position): Position[] {
    const neighbors: Position[] = [];
    // Hexagonal neighbors (6 directions)
    const directions = [
      { x: 1, y: 0, z: -1 }, // East
      { x: 1, y: -1, z: 0 }, // Northeast
      { x: 0, y: -1, z: 1 }, // Northwest
      { x: -1, y: 0, z: 1 }, // West
      { x: -1, y: 1, z: 0 }, // Southwest
      { x: 0, y: 1, z: -1 }, // Southeast
    ];

    for (const dir of directions) {
      const neighbor: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y,
        z: (position.z || 0) + dir.z,
      };

      if (this.isValidPosition(neighbor)) {
        neighbors.push(neighbor);
      }
    }

    return neighbors;
  }

  private getMooreNeighbors(position: Position): Position[] {
    const neighbors: Position[] = [];
    // Moore neighborhood (8 directions)
    const directions = [
      { x: -1, y: -1 },
      { x: -1, y: 0 },
      { x: -1, y: 1 },
      { x: 0, y: -1 },
      { x: 0, y: 1 },
      { x: 1, y: -1 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
    ];

    for (const dir of directions) {
      const neighbor: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y,
      };

      if (this.isValidPosition(neighbor)) {
        neighbors.push(neighbor);
      }
    }

    return neighbors;
  }

  private getVonNeumannNeighbors(position: Position): Position[] {
    const neighbors: Position[] = [];
    // Von Neumann neighborhood (4 directions)
    const directions = [
      { x: -1, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: 0, y: 1 },
    ];

    for (const dir of directions) {
      const neighbor: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y,
      };

      if (this.isValidPosition(neighbor)) {
        neighbors.push(neighbor);
      }
    }

    return neighbors;
  }

  calculateDistance(pos1: Position, pos2: Position): number {
    if (this.boardType === 'hexagonal') {
      // Hexagonal distance using cube coordinates
      const dx = Math.abs(pos1.x - pos2.x);
      const dy = Math.abs(pos1.y - pos2.y);
      const dz = Math.abs((pos1.z || 0) - (pos2.z || 0));
      return Math.max(dx, dy, dz);
    } else {
      // Chebyshev distance for square boards (Moore adjacency)
      return Math.max(Math.abs(pos1.x - pos2.x), Math.abs(pos1.y - pos2.y));
    }
  }

  // Marker manipulation methods - Section 8.3 of rules

  /**
   * Find all marker lines on the board (4+ for 8x8, 4+ for 19x19/hex)
   * Rule Reference: Section 11.1 - Line Formation Rules
   */
  setMarker(position: Position, player: number, board: BoardState): void {
    const posKey = positionToString(position);

    // Markers must never coexist with stacks or collapsed territory. When a
    // marker is placed on a space that currently hosts a stack (for example,
    // a departure cell during movement/capture), we treat the marker as
    // replacing that stack so the resulting cell is a pure marker space.
    //
    // Likewise, markers are not meaningful on collapsed territory; if a caller
    // attempts to place a marker on a collapsed space, we ignore the request.
    // Any residual inconsistencies are surfaced via assertBoardInvariants.
    if (board.collapsedSpaces.has(posKey)) {
      // Do not place markers on collapsed territory.
      return;
    }

    // Ensure stack+marker exclusivity by removing any stack that might exist
    // at this position before setting the marker. This mirrors the invariant
    // that stacks and markers are mutually exclusive and prevents transient
    // stack+marker overlaps at departure cells.
    if (board.stacks.has(posKey)) {
      board.stacks.delete(posKey);
    }

    board.markers.set(posKey, {
      player,
      position,
      type: 'regular',
    });

    this.assertBoardInvariants(board, 'setMarker');
  }

  /**
   * Gets marker at position (returns undefined if no marker or if space is collapsed)
   */
  getMarker(position: Position, board: BoardState): number | undefined {
    const posKey = positionToString(position);
    const marker = board.markers.get(posKey);
    return marker?.player;
  }

  /**
   * Removes a marker from the board
   * Rule Reference: Section 8.2 - Landing on same-color marker removes it
   */
  removeMarker(position: Position, board: BoardState): void {
    const posKey = positionToString(position);
    board.markers.delete(posKey);
  }

  /**
   * Flips an opponent marker to the moving player's color
   * Rule Reference: Section 8.3 - Opponent markers flip to your color
   */
  flipMarker(position: Position, newPlayer: number, board: BoardState): void {
    const posKey = positionToString(position);
    const existingMarker = board.markers.get(posKey);
    if (existingMarker && existingMarker.player !== newPlayer) {
      board.markers.set(posKey, {
        player: newPlayer,
        position,
        type: 'regular',
      });
    }
  }

  /**
   * Collapses a marker into claimed territory
   * Rule Reference: Section 8.3 - Your own markers become collapsed territory
   */
  collapseMarker(position: Position, player: number, board: BoardState): void {
    const posKey = positionToString(position);
    // Remove from markers
    board.markers.delete(posKey);
    // When a marker collapses to territory, the cell becomes exclusive
    // territory: no stacks or markers may remain.
    board.stacks.delete(posKey);
    // Add to collapsed spaces
    board.collapsedSpaces.set(posKey, player);

    this.assertBoardInvariants(board, 'collapseMarker');
  }

  /**
   * Gets the controlling player of a collapsed space (undefined if not collapsed)
   */
  getCollapsedSpace(position: Position, board: BoardState): number | undefined {
    const posKey = positionToString(position);
    return board.collapsedSpaces.get(posKey);
  }

  /**
   * Sets a position as collapsed territory
   * Rule Reference: Section 11.2, 12.2 - Spaces collapse to player's color
   */
  setCollapsedSpace(position: Position, player: number, board: BoardState): void {
    const posKey = positionToString(position);
    // Remove any marker that might exist
    board.markers.delete(posKey);
    // Remove any stack that might exist
    board.stacks.delete(posKey);
    // Mark as collapsed
    board.collapsedSpaces.set(posKey, player);

    this.assertBoardInvariants(board, 'setCollapsedSpace');
  }

  /**
   * Checks if a position is collapsed territory (cannot be moved through or occupied)
   */
  isCollapsedSpace(position: Position, board: BoardState): boolean {
    const posKey = positionToString(position);
    return board.collapsedSpaces.has(posKey);
  }

  // Stack manipulation methods
  getStack(position: Position, board: BoardState): RingStack | undefined {
    return board.stacks.get(positionToString(position));
  }

  setStack(position: Position, stack: RingStack, board: BoardState): void {
    const posKey = positionToString(position);

    // Invariant guard: stacks and markers should not coexist on the same
    // space. If we ever see both at once, log a diagnostic and repair the
    // state by removing the marker. This mirrors the client sandbox
    // semantics where stacks and markers are mutually exclusive.
    if (board.markers.has(posKey)) {
      const existingMarker = board.markers.get(posKey);

      this.logRepair('setStack_on_marker', {
        posKey,
        stack,
        existingMarker,
      });

      board.markers.delete(posKey);
    }

    board.stacks.set(posKey, stack);

    this.assertBoardInvariants(board, 'setStack');
  }

  removeStack(position: Position, board: BoardState): void {
    board.stacks.delete(positionToString(position));
  }

  // Check if position has a stack controlled by player
  hasPlayerStack(position: Position, playerId: number, board: BoardState): boolean {
    const stack = this.getStack(position, board);
    return stack?.controllingPlayer === playerId;
  }

  // Get all positions with stacks controlled by player
  getPlayerStackPositions(playerId: number, board: BoardState): Position[] {
    const positions: Position[] = [];
    for (const [posStr, stack] of board.stacks) {
      if (stack.controllingPlayer === playerId) {
        positions.push(stringToPosition(posStr));
      }
    }
    return positions;
  }

  // Get all stacks controlled by player
  getPlayerStacks(board: BoardState, player: number): RingStack[] {
    const stacks: RingStack[] = [];
    for (const [, stack] of board.stacks) {
      if (stack.controllingPlayer === player) {
        stacks.push(stack);
      }
    }
    return stacks;
  }

  // Find territories controlled by player
  findPlayerTerritories(board: BoardState, player: number): Territory[] {
    return this.findAllTerritories(player, board);
  }

  // Get all positions with stacks (any player)
  getAllStackPositions(board: BoardState): Position[] {
    return Array.from(board.stacks.keys()).map(stringToPosition);
  }

  // Territory analysis methods
  findConnectedTerritory(
    startPosition: Position,
    playerId: number,
    board: BoardState
  ): Set<string> {
    const territory = new Set<string>();
    const visited = new Set<string>();
    const queue = [startPosition];

    while (queue.length > 0) {
      const current = queue.shift();
      if (!current) continue;
      const currentKey = positionToString(current);

      if (visited.has(currentKey)) continue;
      visited.add(currentKey);

      // Check if this position is controlled by the player
      const stack = this.getStack(current, board);
      if (stack?.controllingPlayer === playerId) {
        territory.add(currentKey);

        // Add neighbors for territory expansion (using territory adjacency)
        const neighbors = this.getNeighbors(current, this.config.territoryAdjacency);
        for (const neighbor of neighbors) {
          const neighborKey = positionToString(neighbor);
          if (!visited.has(neighborKey)) {
            queue.push(neighbor);
          }
        }
      }
    }

    return territory;
  }

  // Find all territories for all players
  // Find all territories for all players
  findAllTerritoriesForAllPlayers(board: BoardState): Territory[] {
    const allTerritories: Territory[] = [];
    const allPlayers = new Set<number>();

    // Find all players who have stacks on the board
    for (const [, stack] of board.stacks) {
      if (stack) {
        allPlayers.add(stack.controllingPlayer);
      }
    }

    // Get territories for each player
    for (const playerId of allPlayers) {
      const playerTerritories = this.findAllTerritories(playerId, board);
      allTerritories.push(...playerTerritories);
    }

    return allTerritories;
  }
  // Find all territories for a player
  findAllTerritories(playerId: number, board: BoardState): Territory[] {
    const territories: Territory[] = [];
    const visited = new Set<string>();
    const playerPositions = this.getPlayerStackPositions(playerId, board);

    for (const position of playerPositions) {
      const posKey = positionToString(position);
      if (!visited.has(posKey)) {
        const territoryPositions = this.findConnectedTerritory(position, playerId, board);

        // Mark all positions in this territory as visited
        for (const pos of territoryPositions) {
          visited.add(pos);
        }

        if (territoryPositions.size > 0) {
          territories.push({
            spaces: Array.from(territoryPositions).map(stringToPosition),
            controllingPlayer: playerId,
            isDisconnected: false,
          });
        }
      }
    }

    return territories;
  }

  // Line detection methods - CRITICAL: Lines are formed by MARKERS, not stacks
  // Rule Reference: Section 11.1 - Line Formation Rules
  findLinesFromPosition(position: Position, board: BoardState): LineInfo[] {
    // Delegate geometry to the shared lineDetection helper so backend and
    // sandbox share identical line sets. We restrict to lines owned by the
    // marker on this position and then filter to those that actually include
    // the queried cell.
    const markerOwner = this.getMarker(position, board);
    if (markerOwner === undefined) {
      return [];
    }

    const playerLines = findLinesForPlayerShared(board, markerOwner);
    const key = positionToString(position);

    return playerLines.filter((line) => line.positions.some((p) => positionToString(p) === key));
  }

  /**
   * Debug helper used in rules/parity tests to inspect backend line
   * detection behaviour. This intentionally logs only high-level keys
   * so it remains stable across refactors.
   */
  debugFindAllLines(board: BoardState): { keys: string[] } {
    const lines = this.findAllLines(board);
    const keys = lines
      .map((l) =>
        l.positions
          .map((p) => positionToString(p))
          .sort()
          .join('|')
      )
      .sort();

    debugLog(flagEnabled('RINGRIFT_TRACE_DEBUG'), '[BoardManager.debugFindAllLines]', {
      boardType: this.boardType,
      markerCount: board.markers.size,
      stackCount: board.stacks.size,
      collapsedCount: board.collapsedSpaces.size,
      lineCount: lines.length,
      keys,
    });

    return { keys };
  }

  /**
   * Find all marker lines on the board for all players.
   *
   * NOTE: Geometry is delegated to src/shared/engine/lineDetection.findAllLines
   * so that backend, sandbox, and shared GameEngine share a single
   * implementation of Section 11.1 line rules.
   */
  findAllLines(board: BoardState): LineInfo[] {
    return findAllLinesShared(board);
  }

  // Pathfinding with obstacles
  findPath(from: Position, to: Position, obstacles: Set<string>): Position[] | null {
    // A* pathfinding algorithm
    const openSet = new Set<string>([positionToString(from)]);
    const closedSet = new Set<string>(); // Track visited nodes to prevent infinite loops
    const cameFrom = new Map<string, string>();
    const gScore = new Map<string, number>();
    const fScore = new Map<string, number>();

    gScore.set(positionToString(from), 0);
    fScore.set(positionToString(from), this.calculateDistance(from, to));

    while (openSet.size > 0) {
      // Find node with lowest fScore
      let current = '';
      let lowestF = Infinity;
      for (const node of openSet) {
        const f = fScore.get(node) || Infinity;
        if (f < lowestF) {
          lowestF = f;
          current = node;
        }
      }

      if (current === positionToString(to)) {
        // Reconstruct path
        const path: Position[] = [];
        let currentPos = current;
        while (currentPos) {
          path.unshift(stringToPosition(currentPos));
          currentPos = cameFrom.get(currentPos) || '';
        }
        return path;
      }

      openSet.delete(current);
      closedSet.add(current); // Mark as visited
      const currentPosition = stringToPosition(current);
      const neighbors = this.getNeighbors(currentPosition, this.config.movementAdjacency);

      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);

        // Skip if already visited
        if (closedSet.has(neighborKey)) {
          continue;
        }

        // Skip if obstacle
        if (obstacles.has(neighborKey)) {
          continue;
        }

        const tentativeG = (gScore.get(current) || 0) + 1;

        if (!gScore.has(neighborKey) || tentativeG < (gScore.get(neighborKey) || Infinity)) {
          cameFrom.set(neighborKey, current);
          gScore.set(neighborKey, tentativeG);
          fScore.set(neighborKey, tentativeG + this.calculateDistance(neighbor, to));

          if (!openSet.has(neighborKey)) {
            openSet.add(neighborKey);
          }
        }
      }
    }

    return null; // No path found
  }

  // Utility methods
  getAllPositions(): Position[] {
    return Array.from(this.validPositions).map(stringToPosition);
  }

  getEdgePositions(): Position[] {
    const allPositions = this.getAllPositions();

    if (this.boardType === 'hexagonal') {
      // Hexagonal edge positions
      // size=13 means radius 12, so edge is at distance 12
      const radius = this.size - 1;
      return allPositions.filter((pos) => {
        const distance = Math.max(Math.abs(pos.x), Math.abs(pos.y), Math.abs(pos.z || 0));
        return distance === radius;
      });
    } else {
      // Square edge positions
      return allPositions.filter(
        (pos) => pos.x === 0 || pos.x === this.size - 1 || pos.y === 0 || pos.y === this.size - 1
      );
    }
  }

  getCenterPositions(): Position[] {
    const allPositions = this.getAllPositions();
    const center = Math.floor(this.size / 2);

    if (this.boardType === 'hexagonal') {
      // Hexagonal center positions
      return allPositions.filter((pos) => {
        const distance = Math.max(Math.abs(pos.x), Math.abs(pos.y), Math.abs(pos.z || 0));
        return distance <= 2; // Central area
      });
    } else {
      // Square center positions
      const centerRange = 2;
      return allPositions.filter(
        (pos) => Math.abs(pos.x - center) <= centerRange && Math.abs(pos.y - center) <= centerRange
      );
    }
  }

  isOnEdge(position: Position): boolean {
    if (this.boardType === 'hexagonal') {
      // size=13 means radius 12, so edge is at distance 12
      const radius = this.size - 1;
      const distance = Math.max(
        Math.abs(position.x),
        Math.abs(position.y),
        Math.abs(position.z || 0)
      );
      return distance === radius;
    } else {
      return (
        position.x === 0 ||
        position.x === this.size - 1 ||
        position.y === 0 ||
        position.y === this.size - 1
      );
    }
  }

  isInCenter(position: Position): boolean {
    const center = Math.floor(this.size / 2);

    if (this.boardType === 'hexagonal') {
      const distance = Math.max(
        Math.abs(position.x),
        Math.abs(position.y),
        Math.abs(position.z || 0)
      );
      return distance <= 2;
    } else {
      return Math.abs(position.x - center) <= 2 && Math.abs(position.y - center) <= 2;
    }
  }

  /**
   * Find all disconnected regions on the board.
   *
   * Backend now delegates to the shared territory-detection helper so that
   * disconnected-region geometry is defined in exactly one place
   * (src/shared/engine/territoryDetection.ts). This keeps backend,
   * sandbox, and rules-layer tests aligned and avoids subtle drift
   * between multiple independent implementations.
   *
   * The _movingPlayer parameter is retained for backward compatibility
   * with existing call sites but is no longer used; representation checks
   * are handled entirely by the shared helper.
   */
  findDisconnectedRegions(board: BoardState, _movingPlayer: number): Territory[] {
    return findDisconnectedRegionsShared(board);
  }

  /**
   * Get border marker positions for a disconnected region.
   *
   * Backend now delegates to the shared helper in
   * src/shared/engine/territoryBorders.ts so that border geometry is
   * defined in one place for both backend and sandbox engines. The
   * default mode ('rust_aligned') mirrors the Rust engine semantics.
   */
  getBorderMarkerPositions(regionSpaces: Position[], board: BoardState): Position[] {
    const positions = getSharedBorderMarkers(board, regionSpaces, { mode: 'rust_aligned' });

    if (TERRITORY_TRACE_DEBUG) {
      const regionSpacesSample = regionSpaces.slice(0, 12).map(positionToString);
      const borderMarkersSample = positions.slice(0, 12).map(positionToString);

      debugLog(TERRITORY_TRACE_DEBUG, '[BoardManager.getBorderMarkerPositions]', {
        boardType: board.type,
        regionSize: regionSpaces.length,
        borderCount: positions.length,
        regionSample: regionSpacesSample,
        borderSample: borderMarkersSample,
      });
    }

    return positions;
  }

  // Get board configuration
  getConfig() {
    return this.config;
  }

  // Get adjacent positions based on adjacency type
  getAdjacentPositions(position: Position, adjacencyType: AdjacencyType): Position[] {
    const adjacent: Position[] = [];
    const { x, y, z } = position;

    switch (adjacencyType) {
      case 'moore': // 8-direction for square boards
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            if (dx === 0 && dy === 0) continue;
            const newPos = { x: x + dx, y: y + dy };
            if (this.isValidPosition(newPos)) {
              adjacent.push(newPos);
            }
          }
        }
        break;

      case 'von_neumann': // 4-direction for square boards
        const vonNeumannDirections = [
          { x: 0, y: 1 },
          { x: 1, y: 0 },
          { x: 0, y: -1 },
          { x: -1, y: 0 },
        ];
        for (const dir of vonNeumannDirections) {
          const newPos = { x: x + dir.x, y: y + dir.y };
          if (this.isValidPosition(newPos)) {
            adjacent.push(newPos);
          }
        }
        break;

      case 'hexagonal': // 6-direction for hexagonal boards
        if (z !== undefined) {
          // Cube coordinates for hexagonal
          const hexDirections = [
            { x: 1, y: -1, z: 0 },
            { x: 1, y: 0, z: -1 },
            { x: 0, y: 1, z: -1 },
            { x: -1, y: 1, z: 0 },
            { x: -1, y: 0, z: 1 },
            { x: 0, y: -1, z: 1 },
          ];
          for (const dir of hexDirections) {
            const newPos = { x: x + dir.x, y: y + dir.y, z: z + dir.z };
            if (this.isValidPosition(newPos)) {
              adjacent.push(newPos);
            }
          }
        }
        break;
    }

    return adjacent;
  }
}
