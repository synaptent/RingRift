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
      const radius = this.size;
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

  // Line detection methods
  findLinesFromPosition(position: Position, board: BoardState): LineInfo[] {
    const lines: LineInfo[] = [];
    const stack = this.getStack(position, board);
    if (!stack) return lines;

    const playerId = stack.controllingPlayer;
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
      
      const stack = this.getStack(next, board);
      if (!stack || stack.controllingPlayer !== playerId) break;

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
      
      const stack = this.getStack(prev, board);
      if (!stack || stack.controllingPlayer !== playerId) break;

      line.unshift(prev);
      current = prev;
    }

    return line;
  }

  // Find all lines on the board
  findAllLines(board: BoardState): LineInfo[] {
    const lines: LineInfo[] = [];
    const processedPositions = new Set<string>();

    for (const [posStr] of board.stacks) {
      if (processedPositions.has(posStr)) continue;

      const position = stringToPosition(posStr);
      const positionLines = this.findLinesFromPosition(position, board);

      for (const line of positionLines) {
        // Mark all positions in this line as processed to avoid duplicates
        for (const pos of line.positions) {
          processedPositions.add(positionToString(pos));
        }
        lines.push(line);
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
      return allPositions.filter(pos => {
        const distance = Math.max(
          Math.abs(pos.x),
          Math.abs(pos.y),
          Math.abs(pos.z || 0)
        );
        return distance === this.size;
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
      const distance = Math.max(
        Math.abs(position.x),
        Math.abs(position.y),
        Math.abs(position.z || 0)
      );
      return distance === this.size;
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