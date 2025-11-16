import { RuleEngine } from '../../src/server/game/RuleEngine';
import {
  BoardState,
  GameState,
  Move,
  Player,
  Position,
  BoardType,
  TimeControl,
  positionToString
} from '../../src/shared/types/game';

// Minimal fake BoardManager implementation with just the methods RuleEngine
// uses in these movement/capture tests.
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

  // Methods required by RuleEngine but not exercised in these tests
  getAllPositions(): Position[] {
    return [];
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
    id: 'ruleengine-test',
    boardType,
    board,
    players,
    currentPhase: 'movement',
    currentPlayer: 1,
    moveHistory: [],
    timeControl,
    spectators: [],
    gameStatus: 'active',
    createdAt: now,
    lastMoveAt: now,
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 0,
    territoryVictoryThreshold: 0
  };
}

describe('RuleEngine movement and capture validation (square8)', () => {
  const boardType: BoardType = 'square8';

  function createRuleEngineAndState() {
    const boardManager = new FakeBoardManager(boardType) as any;
    const engine = new RuleEngine(boardManager, boardType as any);
    const state = createBaseGameState(boardType);
    return { engine, state };
  }

  it('accepts a valid stack movement when distance >= stack height and path is clear', () => {
    const { engine, state } = createRuleEngineAndState();

    const from: Position = { x: 1, y: 1 };
    const to: Position = { x: 3, y: 1 }; // distance 2
    const fromKey = positionToString(from);

    state.board.stacks.set(fromKey, {
      position: from,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1
    });

    const move: Move = {
      id: 'm1',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(true);
  });

  it('rejects stack movement when distance < stack height', () => {
    const { engine, state } = createRuleEngineAndState();

    const from: Position = { x: 1, y: 1 };
    const to: Position = { x: 2, y: 1 }; // distance 1
    const fromKey = positionToString(from);

    state.board.stacks.set(fromKey, {
      position: from,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1
    });

    const move: Move = {
      id: 'm2',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(false);
  });

  it('rejects movement when landing on an opponent marker', () => {
    const boardManager = new FakeBoardManager(boardType) as any;
    // Override getMarker to simulate an opponent marker at the destination.
    boardManager.getMarker = jest.fn(() => 2);

    const engine = new RuleEngine(boardManager, boardType as any);
    const state = createBaseGameState(boardType);

    const from: Position = { x: 1, y: 1 };
    const to: Position = { x: 3, y: 1 };
    const fromKey = positionToString(from);

    state.board.stacks.set(fromKey, {
      position: from,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1
    });

    const move: Move = {
      id: 'm3',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(false);
  });

  it('accepts a simple overtaking capture when capHeight >= target cap and landing is beyond target', () => {
    const { engine, state } = createRuleEngineAndState();

    state.currentPhase = 'capture';

    const from: Position = { x: 1, y: 1 };
    const target: Position = { x: 2, y: 1 };
    const landing: Position = { x: 3, y: 1 };

    const fromKey = positionToString(from);
    const targetKey = positionToString(target);

    state.board.stacks.set(fromKey, {
      position: from,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1
    });

    state.board.stacks.set(targetKey, {
      position: target,
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2
    });

    const move: Move = {
      id: 'c1',
      type: 'overtaking_capture',
      player: 1,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(true);
  });

  it('rejects capture when attacker capHeight < target capHeight', () => {
    const { engine, state } = createRuleEngineAndState();

    state.currentPhase = 'capture';

    const from: Position = { x: 1, y: 1 };
    const target: Position = { x: 2, y: 1 };
    const landing: Position = { x: 3, y: 1 };

    const fromKey = positionToString(from);
    const targetKey = positionToString(target);

    // Attacker capHeight 1
    state.board.stacks.set(fromKey, {
      position: from,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1
    });

    // Target capHeight 2
    state.board.stacks.set(targetKey, {
      position: target,
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2
    });

    const move: Move = {
      id: 'c2',
      type: 'overtaking_capture',
      player: 1,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(false);
  });

  it('accepts overtaking own stacks when capHeight >= target capHeight (rule fix)', () => {
    const { engine, state } = createRuleEngineAndState();

    state.currentPhase = 'capture';

    const from: Position = { x: 1, y: 1 };
    const target: Position = { x: 2, y: 1 };
    const landing: Position = { x: 3, y: 1 };

    const fromKey = positionToString(from);
    const targetKey = positionToString(target);

    // Attacker: Player 1 with capHeight 2
    state.board.stacks.set(fromKey, {
      position: from,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1
    });

    // Target: Also Player 1 (own stack) with capHeight 1
    state.board.stacks.set(targetKey, {
      position: target,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1
    });

    const move: Move = {
      id: 'c3',
      type: 'overtaking_capture',
      player: 1,
      from,
      captureTarget: target,
      to: landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(true);
  });

  it('rejects ring placement that leaves no legal moves (rule fix)', () => {
    const { state } = createRuleEngineAndState();

    state.currentPhase = 'ring_placement';

    // Create a scenario where placing at (0,0) would leave no legal moves
    // Use a custom BoardManager that is restricted to a very small valid area
    const boardManager = new FakeBoardManager(boardType) as any;
    
    // Only positions (0,0) and (1,0) are valid, and (1,0) is collapsed
    boardManager.isValidPosition = jest.fn((pos: Position) => {
      return (pos.x === 0 && pos.y === 0) || (pos.x === 1 && pos.y === 0);
    });
    
    boardManager.isCollapsedSpace = jest.fn((pos: Position) => {
      // Block the only adjacent position
      return pos.x === 1 && pos.y === 0;
    });

    const customEngine = new RuleEngine(boardManager, boardType as any);

    const move: Move = {
      id: 'place1',
      type: 'place_ring',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = customEngine.validateMove(move, state);
    expect(valid).toBe(false);
  });

  it('accepts ring placement that has at least one legal move (rule fix)', () => {
    const { engine, state } = createRuleEngineAndState();

    state.currentPhase = 'ring_placement';

    // Place a ring at (3,3) which has plenty of open moves in all directions
    const move: Move = {
      id: 'place2',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    };

    const valid = engine.validateMove(move, state);
    expect(valid).toBe(true);
  });
});
