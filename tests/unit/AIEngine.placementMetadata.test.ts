import { AIEngine } from '../../src/server/game/ai/AIEngine';
import { GameState, BoardState, Player, Move } from '../../src/shared/types/game';

function makeBaseGameState(overrides: Partial<GameState>): GameState {
  const board: BoardState = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8'
  } as any;

  const players: Player[] = [
    {
      id: 'p1',
      username: 'AI',
      type: 'ai',
      playerNumber: 1,
      rating: undefined,
      isReady: true,
      timeRemaining: 0,
      aiDifficulty: 5,
      aiProfile: { difficulty: 5, mode: 'service', aiType: 'random' },
      ringsInHand: 5,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  const base: GameState = {
    id: 'g1',
    boardType: 'square8',
    board,
    players,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    timeControl: { initialTime: 0, increment: 0, type: 'blitz' },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 1,
    totalRingsInPlay: 18,
    totalRingsEliminated: 0,
    victoryThreshold: 10,
    territoryVictoryThreshold: 33
  };

  return { ...base, ...overrides };
}

describe('AIEngine.normalizeServiceMove placement metadata', () => {
  it('sets placementCount = 1 and placedOnStack = true when placing on an existing stack', () => {
    const engine = new AIEngine();

    const board: BoardState = {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8'
    } as any;

    // Existing stack at (0,0)
    board.stacks.set('0,0', {
      position: { x: 0, y: 0 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1
    });

    const gameState = makeBaseGameState({ board });

    const serviceMove: Move = {
      id: 'svc-1',
      type: 'place_ring' as any,
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    } as Move;

    const normalized = (engine as any).normalizeServiceMove(serviceMove, gameState, 1) as Move | null;

    expect(normalized).not.toBeNull();
    expect(normalized!.type).toBe('place_ring');
    expect(normalized!.placedOnStack).toBe(true);
    expect(normalized!.placementCount).toBe(1);
  });

  it('clamps placementCount on empties to ringsInHand (and never below 1)', () => {
    const engine = new AIEngine();

    const board: BoardState = {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8'
    } as any;

    const players: Player[] = [
      {
        id: 'p1',
        username: 'AI',
        type: 'ai',
        playerNumber: 1,
        rating: undefined,
        isReady: true,
        timeRemaining: 0,
        aiDifficulty: 5,
        aiProfile: { difficulty: 5, mode: 'service', aiType: 'random' },
        ringsInHand: 2,
        eliminatedRings: 0,
        territorySpaces: 0
      }
    ];

    const gameState = makeBaseGameState({ board, players });

    const serviceMove: Move = {
      id: 'svc-2',
      type: 'place_ring' as any,
      player: 1,
      to: { x: 3, y: 3 },
      placementCount: 5, // service asks for more than available
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    } as Move;

    const normalized = (engine as any).normalizeServiceMove(serviceMove, gameState, 1) as Move | null;

    expect(normalized).not.toBeNull();
    expect(normalized!.type).toBe('place_ring');
    expect(normalized!.placedOnStack).toBe(false);
    // Cannot exceed ringsInHand = 2
    expect(normalized!.placementCount).toBeLessThanOrEqual(2);
    // And must be at least 1
    expect(normalized!.placementCount).toBeGreaterThanOrEqual(1);
  });

  it('chooses a placementCount in [1, min(3, ringsInHand)] on empty cells when service omits it', () => {
    const engine = new AIEngine();

    const board: BoardState = {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8'
    } as any;

    const players: Player[] = [
      {
        id: 'p1',
        username: 'AI',
        type: 'ai',
        playerNumber: 1,
        rating: undefined,
        isReady: true,
        timeRemaining: 0,
        aiDifficulty: 5,
        aiProfile: { difficulty: 5, mode: 'service', aiType: 'random' },
        ringsInHand: 5,
        eliminatedRings: 0,
        territorySpaces: 0
      }
    ];

    const gameState = makeBaseGameState({ board, players });

    const serviceMove: Move = {
      id: 'svc-3',
      type: 'place_ring' as any,
      player: 1,
      to: { x: 4, y: 4 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1
    } as Move;

    const normalized = (engine as any).normalizeServiceMove(serviceMove, gameState, 1) as Move | null;

    expect(normalized).not.toBeNull();
    expect(normalized!.type).toBe('place_ring');
    expect(normalized!.placedOnStack).toBe(false);

    const count = normalized!.placementCount;
    expect(typeof count).toBe('number');
    if (typeof count === 'number') {
      // With ringsInHand = 5, upper bound is min(3, 5) = 3
      expect(count).toBeGreaterThanOrEqual(1);
      expect(count).toBeLessThanOrEqual(3);
    }
  });
});
