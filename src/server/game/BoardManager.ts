import { 
  Position, 
  BoardType, 
  BoardState, 
  RingStack,
  Territory,
  LineInfo,
  AdjacencyType,
  BOARD_CONFIGS, 
  positionToString, 
  stringToPosition 
} from '../../shared/types/game';

export class BoardManager {
  private boardType: BoardType;
  private size: number;
  private config: typeof BOARD_CONFIGS[BoardType];
  private validPositions: Set<string>;

  constructor(boardType: BoardType) {
    this.boardType = boardType;
    this.config = BOARD_CONFIGS[boardType];
    this.size = this.config.size;
    this.validPositions = this.generateValidPositions();
  }

  createBoard(): BoardState {
    return {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: this.size,
      type: this.boardType
    };
  }

  private generateValidPositions(): Set<string> {
    const positions = new Set<string>();

    if (this.boardType === 'hexagonal') {
      // Generate hexagonal board positions
      // size=11 means radius 10 (positions from -10 to 10)
      // This gives 3*10^2 + 3*10 + 1 = 331 positions
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
      { x: 1, y: 0, z: -1 },   // East
      { x: 1, y: -1, z: 0 },   // Northeast
      { x: 0, y: -1, z: 1 },   // Northwest
      { x: -1, y: 0, z: 1 },   // West
      { x: -1, y: 1, z: 0 },   // Southwest
      { x: 0, y: 1, z: -1 }    // Southeast
    ];

    for (const dir of directions) {
      const neighbor: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y,
        z: (position.z || 0) + dir.z
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
      { x: -1, y: -1 }, { x: -1, y: 0 }, { x: -1, y: 1 },
      { x: 0, y: -1 },                   { x: 0, y: 1 },
      { x: 1, y: -1 },  { x: 1, y: 0 },  { x: 1, y: 1 }
    ];

    for (const dir of directions) {
      const neighbor: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y
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
      { x: -1, y: 0 }, { x: 1, y: 0 },
      { x: 0, y: -1 }, { x: 0, y: 1 }
    ];

    for (const dir of directions) {
      const neighbor: Position = {
        x: position.x + dir.x,
        y: position.y + dir.y
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
      return Math.max(
        Math.abs(pos1.x - pos2.x),
        Math.abs(pos1.y - pos2.y)
      );
    }
  }

  // Marker manipulation methods - Section 8.3 of rules
  
  /**
   * Sets a marker at the specified position
   * Rule Reference: Section 4.2.1 - Leave marker on departure space
   */
  setMarker(position: Position, player: number, board: BoardState): void {
    const posKey = positionToString(position);
    board.markers.set(posKey, {
      player,
      position,
      type: 'regular'
    });
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
        type: 'regular'
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
    // Add to collapsed spaces
    board.collapsedSpaces.set(posKey, player);
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
    board.stacks.set(positionToString(position), stack);
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
  findConnectedTerritory(startPosition: Position, playerId: number, board: BoardState): Set<string> {
    const territory = new Set<string>();
    const visited = new Set<string>();
    const queue = [startPosition];

    while (queue.length > 0) {
      const current = queue.shift()!;
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
            isDisconnected: false
          });
        }
      }
    }

    return territories;
  }

  // Line detection methods - CRITICAL: Lines are formed by MARKERS, not stacks
  // Rule Reference: Section 11.1 - Line Formation Rules
  findLinesFromPosition(position: Position, board: BoardState): LineInfo[] {
    const lines: LineInfo[] = [];
    
    // Check if this position has a marker (not a stack!)
    const marker = this.getMarker(position, board);
    if (marker === undefined) return lines;

    const playerId = marker;
    const directions = this.getLineDirections();

    for (const direction of directions) {
      const line = this.findLineInDirection(position, direction, playerId, board);
      if (line.length >= this.config.lineLength) {
        lines.push({
          positions: line,
          player: playerId,
          length: line.length,
          direction: direction
        });
      }
    }

    return lines;
  }

  private getLineDirections(): Position[] {
    if (this.boardType === 'hexagonal') {
      // 6 directions for hexagonal
      return [
        { x: 1, y: 0, z: -1 },   // East
        { x: 1, y: -1, z: 0 },   // Northeast
        { x: 0, y: -1, z: 1 },   // Northwest
      ];
    } else {
      // 8 directions for square (Moore adjacency for lines)
      return [
        { x: 1, y: 0 },   // East
        { x: 1, y: 1 },   // Southeast
        { x: 0, y: 1 },   // South
        { x: 1, y: -1 },  // Northeast
      ];
    }
  }

  /**
   * Find consecutive markers (not stacks!) in a direction
   * Rule Reference: Section 11.1 - Must consist of consecutive, contiguous, non-collapsed markers
   */
  private findLineInDirection(
    startPosition: Position, 
    direction: Position, 
    playerId: number, 
    board: BoardState
  ): Position[] {
    const line: Position[] = [startPosition];

    // Check forward direction
    let current = startPosition;
    while (true) {
      const next: Position = {
        x: current.x + direction.x,
        y: current.y + direction.y,
        z: (current.z || 0) + (direction.z || 0)
      };

      if (!this.isValidPosition(next)) break;
      
      // CRITICAL: Check for MARKER, not stack!
      // Line cannot be interrupted by empty spaces, collapsed spaces, or stacks
      const marker = this.getMarker(next, board);
      if (marker !== playerId) break;
      
      // Also check it's not a collapsed space or has a stack on it
      if (this.isCollapsedSpace(next, board) || this.getStack(next, board)) break;

      line.push(next);
      current = next;
    }

    // Check backward direction
    current = startPosition;
    while (true) {
      const prev: Position = {
        x: current.x - direction.x,
        y: current.y - direction.y,
        z: (current.z || 0) - (direction.z || 0)
      };

      if (!this.isValidPosition(prev)) break;
      
      // CRITICAL: Check for MARKER, not stack!
      const marker = this.getMarker(prev, board);
      if (marker !== playerId) break;
      
      // Also check it's not a collapsed space or has a stack on it
      if (this.isCollapsedSpace(prev, board) || this.getStack(prev, board)) break;

      line.unshift(prev);
      current = prev;
    }

    return line;
  }

  /**
   * Find all marker lines on the board (4+ for 8x8, 5+ for 19x19/hex)
   * Rule Reference: Section 11.1 - Line Formation Rules
   * CRITICAL: Lines are formed by MARKERS, not stacks!
   */
  findAllLines(board: BoardState): LineInfo[] {
    const lines: LineInfo[] = [];
    const processedLines = new Set<string>();

    // Iterate through all MARKERS (not stacks!)
    for (const [posStr, marker] of board.markers) {
      const position = stringToPosition(posStr);
      const directions = this.getLineDirections();

      for (const direction of directions) {
        const line = this.findLineInDirection(position, direction, marker.player, board);
        
        if (line.length >= this.config.lineLength) {
          // Create a unique key for this line (sorted positions to avoid duplicates)
          const lineKey = line
            .map(p => positionToString(p))
            .sort()
            .join('|');
          
          if (!processedLines.has(lineKey)) {
            processedLines.add(lineKey);
            lines.push({
              positions: line,
              player: marker.player,
              length: line.length,
              direction: direction
            });
          }
        }
      }
    }

    return lines;
  }

  // Pathfinding with obstacles
  findPath(from: Position, to: Position, obstacles: Set<string>): Position[] | null {
    // A* pathfinding algorithm
    const openSet = new Set<string>([positionToString(from)]);
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
      const currentPosition = stringToPosition(current);
      const neighbors = this.getNeighbors(currentPosition, this.config.movementAdjacency);

      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        
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
      // size=11 means radius 10, so edge is at distance 10
      const radius = this.size - 1;
      return allPositions.filter(pos => {
        const distance = Math.max(
          Math.abs(pos.x),
          Math.abs(pos.y),
          Math.abs(pos.z || 0)
        );
        return distance === radius;
      });
    } else {
      // Square edge positions
      return allPositions.filter(pos => 
        pos.x === 0 || pos.x === this.size - 1 || 
        pos.y === 0 || pos.y === this.size - 1
      );
    }
  }

  getCenterPositions(): Position[] {
    const allPositions = this.getAllPositions();
    const center = Math.floor(this.size / 2);
    
    if (this.boardType === 'hexagonal') {
      // Hexagonal center positions
      return allPositions.filter(pos => {
        const distance = Math.max(
          Math.abs(pos.x),
          Math.abs(pos.y),
          Math.abs(pos.z || 0)
        );
        return distance <= 2; // Central area
      });
    } else {
      // Square center positions
      const centerRange = 2;
      return allPositions.filter(pos => 
        Math.abs(pos.x - center) <= centerRange && 
        Math.abs(pos.y - center) <= centerRange
      );
    }
  }

  isOnEdge(position: Position): boolean {
    if (this.boardType === 'hexagonal') {
      // size=11 means radius 10, so edge is at distance 10
      const radius = this.size - 1;
      const distance = Math.max(
        Math.abs(position.x),
        Math.abs(position.y),
        Math.abs(position.z || 0)
      );
      return distance === radius;
    } else {
      return position.x === 0 || position.x === this.size - 1 || 
             position.y === 0 || position.y === this.size - 1;
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
      return Math.abs(position.x - center) <= 2 && 
             Math.abs(position.y - center) <= 2;
    }
  }

  /**
   * Find all disconnected regions on the board
   * Rule Reference: Section 12.2 - Territory Disconnection
   * 
   * A region is disconnected when:
   * 1. Physical Disconnection: Surrounded by collapsed spaces, board edges, or single-player marker border
   * 2. Representation: Lacks representation from at least one active player's ring stacks
   * 
   * Key Insight: Markers of a specific color can act as borders (treated like collapsed spaces).
   * We must check for disconnection with respect to each marker color separately.
   */
  findDisconnectedRegions(board: BoardState, _movingPlayer: number): Territory[] {
    const disconnectedRegions: Territory[] = [];
    
    // Get all active players (those with stacks on board)
    const activePlayers = new Set<number>();
    for (const [, stack] of board.stacks) {
      activePlayers.add(stack.controllingPlayer);
    }
    
    // Get all marker colors present on board
    const markerColors = new Set<number>();
    for (const [, marker] of board.markers) {
      markerColors.add(marker.player);
    }
    
    // Check for disconnection with respect to each marker color
    // (markers of that color act as borders along with collapsed spaces and edges)
    for (const borderColor of markerColors) {
      const regions = this.findRegionsWithBorderColor(board, borderColor, activePlayers);
      disconnectedRegions.push(...regions);
    }
    
    // Also check for regions surrounded by only collapsed spaces and edges (no marker borders)
    const regionsWithoutMarkerBorder = this.findRegionsWithoutMarkerBorder(board, activePlayers);
    disconnectedRegions.push(...regionsWithoutMarkerBorder);
    
    return disconnectedRegions;
  }


  /**
   * Find regions where markers of a specific color act as borders
   * Rule Reference: Section 12.2 - Markers of one color can form disconnecting borders
   */
  private findRegionsWithBorderColor(
    board: BoardState, 
    borderColor: number, 
    activePlayers: Set<number>
  ): Territory[] {
    const disconnectedRegions: Territory[] = [];
    const visited = new Set<string>();
    
    // Find all connected regions where borderColor markers act as borders
    for (const posStr of this.validPositions) {
      if (visited.has(posStr)) continue;
      
      const position = stringToPosition(posStr);
      
      // Skip if this is a border (collapsed or borderColor marker)
      if (this.isCollapsedSpace(position, board)) {
        visited.add(posStr);
        continue;
      }
      
      const marker = this.getMarker(position, board);
      if (marker === borderColor) {
        visited.add(posStr);
        continue;
      }
      
      // Find connected region using flood fill (borderColor markers act as borders)
      const region = this.exploreRegionWithBorderColor(position, board, borderColor, visited);
      
      if (region.length === 0) continue;
      
      // Check representation - must lack at least one active player's stacks
      const representedPlayers = this.getRepresentedPlayers(region, board);
      
      if (representedPlayers.size < activePlayers.size) {
        // This region is disconnected!
        disconnectedRegions.push({
          spaces: region,
          controllingPlayer: 0, // Will be set by caller
          isDisconnected: true
        });
      }
    }
    
    return disconnectedRegions;
  }

  /**
   * Find regions surrounded only by collapsed spaces and edges (no marker borders)
   */
  private findRegionsWithoutMarkerBorder(
    board: BoardState,
    activePlayers: Set<number>
  ): Territory[] {
    const disconnectedRegions: Territory[] = [];
    const visited = new Set<string>();
    
    // Find all connected regions where only collapsed spaces and edges form borders
    for (const posStr of this.validPositions) {
      if (visited.has(posStr)) continue;
      
      const position = stringToPosition(posStr);
      
      // Skip collapsed spaces
      if (this.isCollapsedSpace(position, board)) {
        visited.add(posStr);
        continue;
      }
      
      // Find connected region (only collapsed spaces/edges are borders)
      const region = this.exploreRegionWithoutMarkerBorder(position, board, visited);
      
      if (region.length === 0) continue;
      
      // Check if region is actually bordered by collapsed spaces/edges (no markers in border)
      if (!this.isRegionBorderedByCollapsedOnly(region, board)) {
        continue;
      }
      
      // Check representation
      const representedPlayers = this.getRepresentedPlayers(region, board);
      
      if (representedPlayers.size < activePlayers.size) {
        disconnectedRegions.push({
          spaces: region,
          controllingPlayer: 0,
          isDisconnected: true
        });
      }
    }
    
    return disconnectedRegions;
  }

  /**
   * Flood fill to find region where markers of borderColor act as borders
   */
  private exploreRegionWithBorderColor(
    startPosition: Position,
    board: BoardState,
    borderColor: number,
    visited: Set<string>
  ): Position[] {
    const region: Position[] = [];
    const queue = [startPosition];
    const localVisited = new Set<string>();
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentKey = positionToString(current);
      
      if (localVisited.has(currentKey)) continue;
      localVisited.add(currentKey);
      visited.add(currentKey);
      
      // Check if this is a border
      if (this.isCollapsedSpace(current, board)) continue;
      
      const marker = this.getMarker(current, board);
      if (marker === borderColor) continue;
      
      // This space is part of the region (empty, stack, or other-color marker)
      region.push(current);
      
      // Explore neighbors
      const neighbors = this.getNeighbors(current, this.config.territoryAdjacency);
      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        if (!localVisited.has(neighborKey) && this.isValidPosition(neighbor)) {
          queue.push(neighbor);
        }
      }
    }
    
    return region;
  }

  /**
   * Flood fill to find region where only collapsed spaces/edges are borders
   */
  private exploreRegionWithoutMarkerBorder(
    startPosition: Position,
    board: BoardState,
    visited: Set<string>
  ): Position[] {
    const region: Position[] = [];
    const queue = [startPosition];
    const localVisited = new Set<string>();
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentKey = positionToString(current);
      
      if (localVisited.has(currentKey)) continue;
      localVisited.add(currentKey);
      visited.add(currentKey);
      
      // Check if this is a border (only collapsed spaces)
      if (this.isCollapsedSpace(current, board)) continue;
      
      // This space is part of the region
      region.push(current);
      
      // Explore neighbors
      const neighbors = this.getNeighbors(current, this.config.territoryAdjacency);
      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        if (!localVisited.has(neighborKey) && this.isValidPosition(neighbor)) {
          queue.push(neighbor);
        }
      }
    }
    
    return region;
  }

  /**
   * Check if a region is bordered only by collapsed spaces and edges (no markers)
   */
  private isRegionBorderedByCollapsedOnly(regionSpaces: Position[], board: BoardState): boolean {
    const regionSet = new Set(regionSpaces.map(positionToString));
    
    // Check all border positions
    for (const space of regionSpaces) {
      const neighbors = this.getNeighbors(space, this.config.territoryAdjacency);
      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        
        // Skip if neighbor is in region
        if (regionSet.has(neighborKey)) continue;
        
        // Check if neighbor is valid position (board edge check)
        if (!this.isValidPosition(neighbor)) continue; // Edge is OK
        
        // Check if it's collapsed
        if (this.isCollapsedSpace(neighbor, board)) continue; // Collapsed is OK
        
        // If it has a marker, this region is NOT bordered by collapsed-only
        if (this.getMarker(neighbor, board) !== undefined) {
          return false;
        }
        
        // Empty spaces or stacks in border mean not a valid disconnection
        return false;
      }
    }
    
    return true;
  }

  // NOTE: Legacy region-exploration helpers (exploreRegion/analyzeRegionBorder)
  // were superseded by the more explicit territory-disconnection
  // implementations above (findRegionsWithBorderColor,
  // findRegionsWithoutMarkerBorder, isRegionBorderedByCollapsedOnly).
  //
  // Their previous implementations remain available in git history but
  // are intentionally removed from the compiled code to avoid unused-
  // symbol noise and keep the BoardManager focused on the current rules
  // interpretation.

  /**
   * Get all players represented in a region (by their ring stacks)
   * Rule Reference: Section 12.2 - Representation check
   */
  private getRepresentedPlayers(regionSpaces: Position[], board: BoardState): Set<number> {
    const represented = new Set<number>();
    
    for (const space of regionSpaces) {
      const stack = this.getStack(space, board);
      if (stack) {
        represented.add(stack.controllingPlayer);
      }
    }
    
    return represented;
  }

  /**
   * Get border marker positions for a disconnected region.
   *
   * This is the TS analogue of the Rust engine's boundary handling in
   * territory.rs / core_apply_disconnect_region:
   *   - We first find all marker neighbors of the region using
   *     territory adjacency (Von Neumann for square boards) to
   *     identify the "inner edge" of the border.
   *   - We then flood through MARKERS ONLY using Moore adjacency to
   *     capture the entire connected marker ring, including diagonal
   *     corners that are part of the border but not directly
   *     Von-Neumann-adjacent to interior spaces.
   *
   * This two-step approach keeps region discovery aligned with
   * territory adjacency while giving the border the same "Moore
   * continuity" treatment used in the Rust implementation.
   */
  getBorderMarkerPositions(regionSpaces: Position[], board: BoardState): Position[] {
    const regionSet = new Set(regionSpaces.map(positionToString));

    // Step 1: seed border markers = direct territory-adjacent markers
    const borderSeedMap = new Map<string, Position>();
    for (const space of regionSpaces) {
      const neighbors = this.getNeighbors(space, this.config.territoryAdjacency);
      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        if (regionSet.has(neighborKey)) continue;
        const marker = this.getMarker(neighbor, board);
        if (marker !== undefined && !borderSeedMap.has(neighborKey)) {
          borderSeedMap.set(neighborKey, neighbor);
        }
      }
    }

    // If there are no adjacent markers, there is no marker border.
    if (borderSeedMap.size === 0) {
      return [];
    }

    // Step 2: expand across connected markers using Moore adjacency to
    // capture the full border ring, including diagonal corners.
    const borderMarkers = new Map<string, Position>(borderSeedMap);
    const queue: Position[] = Array.from(borderSeedMap.values());
    const visited = new Set<string>(borderSeedMap.keys());

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = this.getMooreNeighbors(current);

      for (const neighbor of neighbors) {
        const neighborKey = positionToString(neighbor);
        if (visited.has(neighborKey)) continue;
        if (regionSet.has(neighborKey)) continue; // never step into region

        const marker = this.getMarker(neighbor, board);
        if (marker !== undefined) {
          visited.add(neighborKey);
          borderMarkers.set(neighborKey, neighbor);
          queue.push(neighbor);
        }
      }
    }

    return Array.from(borderMarkers.values());
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
          { x: 0, y: 1 }, { x: 1, y: 0 }, { x: 0, y: -1 }, { x: -1, y: 0 }
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
            { x: 1, y: -1, z: 0 }, { x: 1, y: 0, z: -1 }, { x: 0, y: 1, z: -1 },
            { x: -1, y: 1, z: 0 }, { x: -1, y: 0, z: 1 }, { x: 0, y: -1, z: 1 }
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
