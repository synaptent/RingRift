import { RuleEngine } from '../../src/server/game/RuleEngine';
import {
  BoardState,
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  positionToString
} from '../../src/shared/types/game';

// Minimal BoardManager stub for placement tests
class FakeBoardManager {
  constructor(public boardType: BoardType) {}

  isValidPosition(_pos: Position): boolean {
    return true;
  }

  isCollapsedSpace(_pos: Position, _board: BoardState): boolean {
    return false;
  }

  getMarker(_pos: Position, _board: BoardState): number | undefined {
    return undefined;
  }

  getAllPositions(): Position[] {
    return [{ x: 0, y: 0 }];
  }

  findAllLines(_board: BoardState): Array<{ player: number; positions: Position[] }> {
    return [];
  }

  findAllTerritories(_player: number, _board: BoardState): any[] {
    return [];
  }
}

function createBaseGameState(boardType: BoardType = 'square8'): GameState {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'p2',
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: boardType === 'square8' ? 8 : boardType === 'square19' ? 19 : 11,
    type: boardType
  };

  const now = new Date();

  return {
    id: 'ruleengine-placement-test',
    boardType,
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    timeControl,
    spectators: [],
    gameStatus: 'active',
    createdAt: now,
    lastMoveAt: now,
    isRated: false,
    maxPlayers: players.length,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0
  };
}

/**
 * Helper to collapse all board positions except the specified ones.
 * This forces the engine to only consider placements at the target positions.
 *
 * NOTE: Per no-dead-placement rule, a newly placed stack must have at least one
 * legal move. For a height-N stack, it needs at least N spaces in a direction
 * to move to. Leave enough uncollapsed positions to satisfy this constraint.
 */
function collapseAllExcept(board: BoardState, keepPositions: Position[]): void {
  const size = board.size;
  const keepKeys = new Set(keepPositions.map(p => positionToString(p)));
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      const key = positionToString({ x, y });
      if (!keepKeys.has(key)) {
        board.collapsedSpaces.set(key, 1);
      }
    }
  }
}

describe('RuleEngine ring placement multi-ring semantics', () => {
  const boardType: BoardType = 'square8';

  function createEngineAndState() {
    const boardManager = new FakeBoardManager(boardType) as any;
    const engine = new RuleEngine(boardManager, boardType as any);
    const state = createBaseGameState(boardType);
    return { engine, state };
  }

  it('generates multi-count placements (1..3) on empties when ringsInHand >= 3', () => {
    const { engine, state } = createEngineAndState();

    // Ensure plenty of rings in hand
    state.players[0].ringsInHand = 10;
    state.currentPhase = 'ring_placement';

    // Target position for fresh placement
    const targetPos: Position = { x: 0, y: 0 };

    // Leave empty positions at distances 1, 2, 3 as movement targets.
    // This satisfies no-dead-placement for heights 1, 2, and 3.
    // (Height-N stack needs an empty position at distance N to have a legal move.)
    const movementTargets: Position[] = [
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 }
    ];

    // Collapse all positions except target and movement targets
    collapseAllExcept(state.board, [targetPos, ...movementTargets]);

    const moves = engine.getValidMoves(state).filter(m => m.type === 'place_ring');

    // Filter to only fresh placements at the target position
    const targetMoves = moves.filter(
      m => !m.placedOnStack && m.to.x === targetPos.x && m.to.y === targetPos.y
    );

    // Should generate exactly 3 placement moves for 1, 2, or 3 rings on empty space
    expect(targetMoves.length).toBe(3);

    const counts = targetMoves
      .map(m => m.placementCount ?? 1)
      .sort((a, b) => a - b);

    expect(counts).toEqual([1, 2, 3]);
    targetMoves.forEach(m => {
      expect(m.placedOnStack).toBe(false);
    });
  });

  it('caps multi-count placements on empties by ringsInHand', () => {
    const { engine, state } = createEngineAndState();

    // Only two rings available in hand
    state.players[0].ringsInHand = 2;
    state.currentPhase = 'ring_placement';

    // Target position for fresh placement
    const targetPos: Position = { x: 0, y: 0 };

    // Leave empty positions at distances 1, 2 as movement targets.
    // This satisfies no-dead-placement for heights 1 and 2.
    const movementTargets: Position[] = [
      { x: 1, y: 0 },
      { x: 2, y: 0 }
    ];

    // Collapse all positions except target and movement targets
    collapseAllExcept(state.board, [targetPos, ...movementTargets]);

    const moves = engine.getValidMoves(state).filter(m => m.type === 'place_ring');

    // Filter to only fresh placements at the target position
    const targetMoves = moves.filter(
      m => !m.placedOnStack && m.to.x === targetPos.x && m.to.y === targetPos.y
    );

    const counts = targetMoves
      .map(m => m.placementCount ?? 1)
      .sort((a, b) => a - b);

    // With only 2 rings in hand, can only place 1 or 2 rings
    expect(counts).toEqual([1, 2]);
  });

  it('generates exactly one 1-ring placement on existing stacks', () => {
    const { engine, state } = createEngineAndState();

    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    // Place player 1's stack at (0,0)
    state.board.stacks.set(key, {
      position: pos,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1
    });

    // After placing 1 ring on the existing stack, it becomes height 2.
    // The no-dead-placement rule requires the stack to have legal moves.
    // Leave empty positions at distance 1 (path) and 2 (target) for movement.
    const pathPos: Position = { x: 1, y: 0 };
    const movementTarget: Position = { x: 2, y: 0 };

    // Collapse all positions except the stack, path, and movement target
    collapseAllExcept(state.board, [pos, pathPos, movementTarget]);

    state.players[0].ringsInHand = 5;
    state.currentPhase = 'ring_placement';

    const moves = engine.getValidMoves(state).filter(m => m.type === 'place_ring');

    // Filter to only stacking placements at (0,0)
    const targetMoves = moves.filter(
      m => m.placedOnStack && m.to.x === pos.x && m.to.y === pos.y
    );

    // Per rules: placing on existing stack allows exactly 1 ring only
    expect(targetMoves.length).toBe(1);
    const move = targetMoves[0];
    expect(move.placedOnStack).toBe(true);
    expect(move.placementCount).toBe(1);
  });

  it('rejects multi-ring placements on stacks via validateMove', () => {
    const { engine, state } = createEngineAndState();

    const pos: Position = { x: 0, y: 0 };
    const key = positionToString(pos);

    state.board.stacks.set(key, {
      position: pos,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1
    });

    state.players[0].ringsInHand = 5;
    state.currentPhase = 'ring_placement';

    const move: Move = {
      id: 'stack-multi',
      type: 'place_ring',
      player: 1,
      to: pos,
      placedOnStack: true,
      placementCount: 2,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(false);
  });
});
