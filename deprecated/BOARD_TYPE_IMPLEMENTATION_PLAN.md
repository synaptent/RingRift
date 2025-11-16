# Board Type Implementation Plan for RingRift

**Created:** November 13, 2025  
**Purpose:** Comprehensive plan for implementing Square (8x8, 19x19) and Hexagonal board support

---

## Executive Summary

RingRift supports three board types with **critical differences** in adjacency rules:

| Feature | Square Boards (8x8, 19x19) | Hexagonal Board |
|---------|---------------------------|-----------------|
| **Coordinate System** | Cartesian (x, y) | Cubic (x, y, z) |
| **Movement Adjacency** | Moore (8-direction) | Hexagonal (6-direction) |
| **Line Formation** | Moore (8-direction) | Hexagonal (3 axes, 6-direction) |
| **Territory Adjacency** | Von Neumann (4-direction) | Hexagonal (6-direction) |
| **Line Length** | 4+ (8x8), 5+ (19x19) | 5+ |
| **Board Size** | 64 (8x8), 361 (19x19) | 331 spaces |

**Critical Insight:** Square boards use **TWO different adjacency systems** (Moore for movement/lines, Von Neumann for territory), while hexagonal uses **ONE system** (Hexagonal for everything).

---

## Key Differences Requiring Implementation

### 1. **Adjacency Systems**

#### Moore Neighborhood (8-direction - Square boards only)
Used for: Movement and line formation on square boards
```
NW  N  NE
 \  |  /
W --*-- E
 /  |  \
SW  S  SE
```

Offsets from (x, y):
- N: (0, -1)
- NE: (1, -1)
- E: (1, 0)
- SE: (1, 1)
- S: (0, 1)
- SW: (-1, 1)
- W: (-1, 0)
- NW: (-1, -1)

#### Von Neumann Neighborhood (4-direction - Square boards only)
Used for: Territory disconnection on square boards
```
    N
    |
W --*-- E
    |
    S
```

Offsets from (x, y):
- N: (0, -1)
- E: (1, 0)
- S: (0, 1)
- W: (-1, 0)

#### Hexagonal Neighborhood (6-direction - Hexagonal board only)
Used for: Movement, lines, AND territory on hexagonal board

Cubic coordinates (x, y, z where x + y + z = 0):
```
     NW   NE
      \   /
   W --*-- E
      /   \
     SW   SE
```

Offsets from (x, y, z):
- E: (+1, -1, 0)
- W: (-1, +1, 0)
- NE: (+1, 0, -1)
- SW: (-1, 0, +1)
- NW: (0, +1, -1)
- SE: (0, -1, +1)

### 2. **Line Formation Differences**

#### Square Boards (8 directions)
Can form lines in 8 directions:
- 4 orthogonal: N, S, E, W
- 4 diagonal: NE, NW, SE, SW

#### Hexagonal Board (3 axes, 6 directions)
Can form lines along 3 main axes:
- Axis 1: E-W direction (x-axis)
- Axis 2: NE-SW direction
- Axis 3: NW-SE direction

Each axis has 2 directions, totaling 6 directions.

### 3. **Position Representations**

#### Square Boards
```typescript
interface Position {
  x: number;  // 0 to boardSize-1
  y: number;  // 0 to boardSize-1
}
```

#### Hexagonal Board
```typescript
interface Position {
  x: number;  // Cubic coordinate
  y: number;  // Cubic coordinate
  z: number;  // Cubic coordinate (x + y + z = 0)
}
```

---

## Implementation Strategy

### Phase 1: BoardManager Enhancements (Week 1)

#### 1.1: Abstract Adjacency System
```typescript
class BoardManager {
  private boardType: BoardType;
  
  // Get neighbors based on adjacency type
  getNeighbors(pos: Position, adjacencyType: AdjacencyType): Position[] {
    switch(adjacencyType) {
      case 'moore':
        return this.getMooreNeighbors(pos);
      case 'von_neumann':
        return this.getVonNeumannNeighbors(pos);
      case 'hexagonal':
        return this.getHexagonalNeighbors(pos);
    }
  }
  
  private getMooreNeighbors(pos: Position): Position[] {
    // 8-direction for square boards
    const offsets = [
      {x: 0, y: -1},   // N
      {x: 1, y: -1},   // NE
      {x: 1, y: 0},    // E
      {x: 1, y: 1},    // SE
      {x: 0, y: 1},    // S
      {x: -1, y: 1},   // SW
      {x: -1, y: 0},   // W
      {x: -1, y: -1}   // NW
    ];
    return offsets
      .map(off => ({x: pos.x + off.x, y: pos.y + off.y}))
      .filter(p => this.isValidPosition(p));
  }
  
  private getVonNeumannNeighbors(pos: Position): Position[] {
    // 4-direction for square board territory
    const offsets = [
      {x: 0, y: -1},   // N
      {x: 1, y: 0},    // E
      {x: 0, y: 1},    // S
      {x: -1, y: 0}    // W
    ];
    return offsets
      .map(off => ({x: pos.x + off.x, y: pos.y + off.y}))
      .filter(p => this.isValidPosition(p));
  }
  
  private getHexagonalNeighbors(pos: Position): Position[] {
    // 6-direction for hexagonal board
    if (!pos.z) throw new Error('Hexagonal position requires z coordinate');
    
    const offsets = [
      {x: 1, y: -1, z: 0},   // E
      {x: -1, y: 1, z: 0},   // W
      {x: 1, y: 0, z: -1},   // NE
      {x: -1, y: 0, z: 1},   // SW
      {x: 0, y: 1, z: -1},   // NW
      {x: 0, y: -1, z: 1}    // SE
    ];
    return offsets
      .map(off => ({
        x: pos.x + off.x, 
        y: pos.y + off.y, 
        z: (pos.z || 0) + off.z
      }))
      .filter(p => this.isValidPosition(p));
  }
}
```

#### 1.2: Movement Path Calculation
```typescript
// Must account for different movement rules
getMovementDirections(pos: Position): Position[] {
  const config = BOARD_CONFIGS[this.boardType];
  return this.getNeighbors(pos, config.movementAdjacency);
}

// Check if positions are in same line
areInSameLine(pos1: Position, pos2: Position): boolean {
  const config = BOARD_CONFIGS[this.boardType];
  
  if (config.type === 'square') {
    // 8 possible directions
    const dx = pos2.x - pos1.x;
    const dy = pos2.y - pos1.y;
    
    // Same row, column, or diagonal
    return dx === 0 || dy === 0 || Math.abs(dx) === Math.abs(dy);
  } else {
    // Hexagonal: 3 axes
    const dx = pos2.x - pos1.x;
    const dy = pos2.y - pos1.y;
    const dz = (pos2.z || 0) - (pos1.z || 0);
    
    // Must be along one of the 3 axes
    return (dx === 0) || (dy === 0) || (dz === 0);
  }
}
```

#### 1.3: Line Detection
```typescript
findAllLines(board: BoardState): LineInfo[] {
  const config = BOARD_CONFIGS[board.type];
  const lines: LineInfo[] = [];
  
  // Different line detection for square vs hex
  if (config.type === 'square') {
    lines.push(...this.findSquareBoardLines(board));
  } else {
    lines.push(...this.findHexagonalBoardLines(board));
  }
  
  return lines;
}

private findSquareBoardLines(board: BoardState): LineInfo[] {
  // Check 8 directions from each marker
  // Horizontal, vertical, and both diagonals
}

private findHexagonalBoardLines(board: BoardState): LineInfo[] {
  // Check 3 axes (6 directions total)
  // Each axis: constant x, constant y, or constant z
}
```

#### 1.4: Territory Disconnection Detection
```typescript
findDisconnectedRegions(board: BoardState, activePlayers: number[]): Territory[] {
  const config = BOARD_CONFIGS[board.type];
  
  // Critical: Use correct adjacency for territory
  const adjacencyType = config.territoryAdjacency;
  
  // Find all connected components using appropriate adjacency
  const regions = this.findConnectedComponents(board, adjacencyType);
  
  // Filter for disconnected regions (lack representation)
  return regions.filter(region => 
    this.lacksRepresentation(region, activePlayers, board)
  );
}
```

### Phase 2: Position System (Week 1)

#### 2.1: Update Position Utilities
```typescript
// Already in types/game.ts, but ensure proper usage:

export const positionToString = (pos: Position): string => {
  return pos.z !== undefined 
    ? `${pos.x},${pos.y},${pos.z}` 
    : `${pos.x},${pos.y}`;
};

export const stringToPosition = (str: string): Position => {
  const parts = str.split(',').map(Number);
  return parts.length === 3 
    ? { x: parts[0], y: parts[1], z: parts[2] }
    : { x: parts[0], y: parts[1] };
};

export const positionsEqual = (pos1: Position, pos2: Position): boolean => {
  return pos1.x === pos2.x && 
         pos1.y === pos2.y && 
         (pos1.z || 0) === (pos2.z || 0);
};

// Add new utilities for hex boards:
export const isValidHexPosition = (pos: Position): boolean => {
  if (pos.z === undefined) return false;
  // Cubic coordinates must sum to 0
  return (pos.x + pos.y + (pos.z || 0)) === 0;
};
```

#### 2.2: Board Initialization
```typescript
createBoard(): BoardState {
  const config = BOARD_CONFIGS[this.boardType];
  
  if (config.type === 'square') {
    return this.createSquareBoard(config.size);
  } else {
    return this.createHexagonalBoard(config.size);
  }
}

private createSquareBoard(size: number): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    size,
    type: this.boardType
  };
}

private createHexagonalBoard(radius: number): BoardState {
  // Hexagonal board with cubic coordinates
  // Initialize all valid hex positions
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    formedLines: [],
    size: radius,
    type: this.boardType
  };
}
```

### Phase 3: Movement Rules (Week 2)

#### 3.1: Implement Board-Aware Movement Validation
```typescript
validateMovement(from: Position, to: Position, board: BoardState): boolean {
  const config = BOARD_CONFIGS[board.type];
  
  // 1. Check if in same line
  if (!this.areInSameLine(from, to)) return false;
  
  // 2. Check minimum distance
  const distance = this.calculateDistance(from, to);
  const stack = this.getStack(from, board);
  if (!stack || distance < stack.stackHeight) return false;
  
  // 3. Check path is clear
  return this.isPathClear(from, to, board);
}

private calculateDistance(from: Position, to: Position): number {
  const config = BOARD_CONFIGS[this.boardType];
  
  if (config.type === 'square') {
    // Chebyshev distance (max of abs differences)
    return Math.max(
      Math.abs(to.x - from.x),
      Math.abs(to.y - from.y)
    );
  } else {
    // Hexagonal distance
    return Math.max(
      Math.abs(to.x - from.x),
      Math.abs(to.y - from.y),
      Math.abs((to.z || 0) - (from.z || 0))
    ) / 2;
  }
}
```

### Phase 4: Testing Strategy (Week 2-3)

#### 4.1: Unit Tests for Each Board Type
```typescript
describe('BoardManager - Square Boards', () => {
  describe('8x8 Board', () => {
    test('Moore adjacency returns 8 neighbors for center position', () => {
      // Test 8-direction movement
    });
    
    test('Von Neumann adjacency returns 4 neighbors for territory', () => {
      // Test 4-direction territory connectivity
    });
    
    test('Lines detected in 8 directions', () => {
      // Test horizontal, vertical, diagonal lines
    });
    
    test('Line length requirement is 4+', () => {
      // Test 8x8 specific line length
    });
  });
  
  describe('19x19 Board', () => {
    test('Uses same adjacency as 8x8', () => {
      // Moore for movement, Von Neumann for territory
    });
    
    test('Line length requirement is 5+', () => {
      // Test 19x19 specific line length
    });
  });
});

describe('BoardManager - Hexagonal Board', () => {
  test('Hexagonal adjacency returns 6 neighbors', () => {
    // Test 6-direction movement
  });
  
  test('Position validates cubic coordinate constraint', () => {
    // x + y + z = 0
  });
  
  test('Lines detected along 3 axes', () => {
    // Test constant x, y, or z
  });
  
  test('Territory uses same hexagonal adjacency', () => {
    // Unified 6-direction system
  });
  
  test('Line length requirement is 5+', () => {
    // Test hexagonal specific line length
  });
});
```

---

## Implementation Checklist

### Week 1: Core Infrastructure
- [x] Fix GamePhase type (remove 'main_game', add 'line_processing')
- [ ] Implement `getNeighbors(pos, adjacencyType)` method
- [ ] Implement Moore neighborhood (8-direction)
- [ ] Implement Von Neumann neighborhood (4-direction)
- [ ] Implement Hexagonal neighborhood (6-direction)
- [ ] Implement board-aware position validation
- [ ] Create square board initialization
- [ ] Create hexagonal board initialization
- [ ] Write position utility tests

### Week 2: Movement & Lines
- [ ] Implement `areInSameLine()` for both board types
- [ ] Implement `calculateDistance()` for both board types
- [ ] Implement `findSquareBoardLines()`
- [ ] Implement `findHexagonalBoardLines()`
- [ ] Implement movement path calculation
- [ ] Implement marker system (works for both board types)
- [ ] Write movement validation tests
- [ ] Write line detection tests

### Week 3: Territory & Integration
- [ ] Implement `findConnectedComponents()` with adjacency parameter
- [ ] Implement `findDisconnectedRegions()` for both board types
- [ ] Implement territory disconnection rules
- [ ] Update GameEngine to use board-aware methods
- [ ] Write territory disconnection tests
- [ ] Integration tests for complete game flow
- [ ] Test scenario from rules (both board types)

---

## Key Design Principles

1. **Polymorphism Through Configuration**
   - Use `BOARD_CONFIGS` to drive behavior
   - Don't hardcode board-specific logic
   - Let config specify which adjacency to use

2. **Type Safety**
   - Position type supports both square (x, y) and hex (x, y, z)
   - Validate hex positions have z coordinate
   - Validate cubic coordinate constraint (x + y + z = 0)

3. **Testability**
   - Each adjacency system tested independently
   - Board-specific logic tested separately
   - Integration tests verify correct system used

4. **Maintainability**
   - Clear separation between square and hex logic
   - Shared utilities where possible
   - Board type determined at runtime from config

---

## Risk Mitigation

### Risk 1: Hex Coordinate Confusion
**Mitigation:** 
- Comprehensive position validation
- Unit tests for all hex operations
- Clear documentation of cubic coordinates

### Risk 2: Adjacency System Misuse
**Mitigation:**
- Always use config-driven adjacency selection
- Never hardcode adjacency in game logic
- Test each adjacency type thoroughly

### Risk 3: Line Detection Complexity
**Mitigation:**
- Separate square and hex line detection
- Start with simple cases, add complexity
- Extensive test coverage

---

## Success Criteria

- [ ] All three board types initialize correctly
- [ ] Movement works on all board types with correct adjacency
- [ ] Line formation uses correct adjacency per board type
- [ ] Territory disconnection uses correct adjacency per board type
- [ ] All FAQ scenarios pass for applicable board types
- [ ] 90%+ test coverage on board-specific logic
- [ ] Can play complete games on all three board types

---

## Next Immediate Steps

1. **Implement adjacency methods** in BoardManager.ts
2. **Write unit tests** for each adjacency type
3. **Implement position validation** for hex boards
4. **Update board initialization** to support hex coordinates
5. **Test with simple scenarios** before complex ones

This plan ensures we build a solid foundation that correctly handles the fundamental differences between square and hexagonal boards.
