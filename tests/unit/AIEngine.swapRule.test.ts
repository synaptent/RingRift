import { AIEngine, AIType as InternalAIType } from '../../src/server/game/ai/AIEngine';
import { GameState, Move, AIProfile } from '../../src/shared/types/game';

// Reuse the RuleEngine mock pattern from AIEngine.serviceClient tests so we can
// control the validMoves seen by AIEngine.getAIMove.
let mockRuleEngineMoves: Move[] = [];

jest.mock('../../src/server/game/RuleEngine', () => ({
  RuleEngine: jest.fn().mockImplementation(() => ({
    getValidMoves: () => mockRuleEngineMoves,
  })),
}));

jest.mock('../../src/server/game/BoardManager', () => ({
  BoardManager: jest.fn().mockImplementation(() => ({})),
}));

describe('AIEngine pie rule (swap_sides meta-move)', () => {
  beforeEach(() => {
    mockRuleEngineMoves = [];
  });

  it('surfaces swap_sides as an AI move when pie-rule conditions hold', async () => {
    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 5,
      mode: 'local_heuristic',
      aiType: 'heuristic',
    };

    // Configure AI for Player 2.
    engine.createAIFromProfile(2, profile);

    const now = new Date();

    const baseState: GameState = {
      id: 'swap-ai-test',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
        size: 8,
      } as any,
      players: [
        {
          id: 'p1',
          username: 'P1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'P2',
          type: 'ai',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 5,
        },
      ] as any,
      currentPhase: 'ring_placement',
      currentPlayer: 2,
      moveHistory: [
        {
          id: 'm1',
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          timestamp: now,
          thinkTime: 0,
          moveNumber: 1,
        } as Move,
      ],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 36,
      totalRingsEliminated: 0,
      victoryThreshold: 19,
      territoryVictoryThreshold: 33,
      rngSeed: 123,
      // Enable pie rule for this game.
      rulesOptions: { swapRuleEnabled: true },
    };

    // Simulate a position where the Python/TS rules engines see no normal
    // moves for Player 2 yet; AIEngine should still offer swap_sides as a
    // candidate via its internal gate.
    mockRuleEngineMoves = [];

    const move = await engine.getAIMove(2, baseState);

    expect(move).not.toBeNull();
    expect(move!.type).toBe('swap_sides');
    expect(move!.player).toBe(2);
  });
});
