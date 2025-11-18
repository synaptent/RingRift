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
    // Treat all positions as valid for these tests
    return true;
  }

  isCollapsedSpace(_pos: Position, _board: BoardState): boolean {
    return false;
  }

  getMarker(_pos: Position, _board: BoardState): number | undefined {
    return undefined;
  }

  getAllPositions(): Position[] {
    // Small set of canonical positions for testing
    return [{ x: 0, y: 0 }];
  }

  // Unused in these tests, but required by RuleEngine constructor usage
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

    const moves = engine.getValidMoves(state).filter(m => m.type === 'place_ring');

    expect(moves.length).toBe(3);

    const counts = moves
      .map(m => m.placementCount ?? 1)
      .sort((a, b) => a - b);

    expect(counts).toEqual([1, 2, 3]);
    moves.forEach(m => {
      expect(m.placedOnStack).toBe(false);
    });
  });

  it('caps multi-count placements on empties by ringsInHand', () => {
    const { engine, state } = createEngineAndState();

    // Only two rings available in hand
    state.players[0].ringsInHand = 2;
    state.currentPhase = 'ring_placement';

    const moves = engine.getValidMoves(state).filter(m => m.type === 'place_ring');

    const counts = moves
      .map(m => m.placementCount ?? 1)
      .sort((a, b) => a - b);

    expect(counts).toEqual([1, 2]);
  });

  it('generates exactly one 1-ring placement on existing stacks', () => {
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

    const moves = engine.getValidMoves(state).filter(m => m.type === 'place_ring');

    expect(moves.length).toBe(1);
    const move = moves[0];
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
