import {
  AIEngine,
  AI_DIFFICULTY_PRESETS,
  AIType as InternalAIType,
} from '../../../src/server/game/ai/AIEngine';
import {
  getAIServiceClient,
  AIType as ServiceAIType,
  MoveResponse,
  RingEliminationChoiceResponsePayload,
  RegionOrderChoiceResponsePayload,
  LineOrderChoiceResponsePayload,
  CaptureDirectionChoiceResponsePayload,
} from '../../../src/server/services/AIServiceClient';
import {
  GameState,
  Move,
  AIProfile,
  RingEliminationChoice,
  RegionOrderChoice,
  LineOrderChoice,
  CaptureDirectionChoice,
} from '../../../src/shared/types/game';

jest.mock('../../../src/server/services/AIServiceClient');

// Local backing store for mocked RuleEngine valid moves used by getAIMove tests.
let mockRuleEngineMoves: Move[] = [];

// Mock RuleEngine and BoardManager so AIEngine.getAIMove does not depend on
// the full rules engine or board geometry when verifying service integration.
jest.mock('../../../src/server/game/RuleEngine', () => ({
  RuleEngine: jest.fn().mockImplementation(() => ({
    getValidMoves: () => mockRuleEngineMoves,
  })),
}));

jest.mock('../../../src/server/game/BoardManager', () => ({
  BoardManager: jest.fn().mockImplementation(() => ({})),
}));

describe('AIEngine service integration (profile-driven)', () => {
  it('getAIMove calls AIServiceClient.getAIMove with mapped ai_type and forwards the returned move', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeMove: Move = {
      id: 'svc-move-1',
      type: 'move_ring',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const fakeResponse: MoveResponse = {
      move: fakeMove,
      evaluation: 0.42,
      thinking_time_ms: 1234,
      ai_type: 'minimax',
      difficulty: 7,
    };

    const fakeClient = {
      getAIMove: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 7,
      mode: 'service',
      aiType: 'minimax',
    };

    engine.createAIFromProfile(1, profile);

    const gameState: GameState = {
      id: 'test-game',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: new Map(),
        size: 8,
      } as any,
      players: [
        {
          id: 'ai-player-1',
          username: 'AI Player 1',
          playerNumber: 1,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 7,
          aiProfile: profile,
        },
      ] as any,
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 123,
    };

    // Provide at least two valid moves so AIEngine does not early-return and
    // instead exercises the service-backed path.
    mockRuleEngineMoves = [
      fakeMove,
      {
        ...fakeMove,
        id: 'svc-move-2',
        to: { x: 2, y: 0 },
        moveNumber: 2,
      },
    ];

    const move = await engine.getAIMove(1, gameState);

    expect(fakeClient.getAIMove).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getAIMove.mock.calls[0];
    expect(callArgs[0]).toBe(gameState);
    expect(callArgs[1]).toBe(1);
    expect(callArgs[2]).toBe(7);
    expect(callArgs[3]).toBe(ServiceAIType.MINIMAX);

    // The move instance may not be the exact same reference, but it should be
    // structurally equal to the move returned by the service.
    expect(move).toEqual(fakeMove);
  });

  it('getAIMove uses IG_GMO service mapping when profile aiType is ig_gmo', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeMove: Move = {
      id: 'svc-move-ig-gmo',
      type: 'move_ring',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const fakeResponse: MoveResponse = {
      move: fakeMove,
      evaluation: 0.84,
      thinking_time_ms: 75,
      ai_type: 'ig_gmo',
      difficulty: 9,
    };

    const fakeClient = {
      getAIMove: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 9,
      mode: 'service',
      aiType: 'ig_gmo',
    };

    engine.createAIFromProfile(1, profile);

    const gameState: GameState = {
      id: 'test-game-ig-gmo',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: new Map(),
        size: 8,
      } as any,
      players: [
        {
          id: 'ai-player-1',
          username: 'AI Player 1',
          playerNumber: 1,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 9,
          aiProfile: profile,
        },
      ] as any,
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 101,
    };

    mockRuleEngineMoves = [
      fakeMove,
      {
        ...fakeMove,
        id: 'svc-move-ig-gmo-2',
        to: { x: 2, y: 0 },
        moveNumber: 2,
      },
    ];

    await engine.getAIMove(1, gameState);

    expect(fakeClient.getAIMove).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getAIMove.mock.calls[0];
    expect(callArgs[2]).toBe(9);
    expect(callArgs[3]).toBe(ServiceAIType.IG_GMO);
  });

  it('getRingEliminationChoice calls AIServiceClient.getRingEliminationChoice and returns the selected option', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const options: RingEliminationChoice['options'] = [
      {
        stackPosition: { x: 0, y: 0 },
        capHeight: 3,
        totalHeight: 5,
        moveId: 'opt-1',
      },
      {
        stackPosition: { x: 1, y: 1 },
        capHeight: 1,
        totalHeight: 4,
        moveId: 'opt-2',
      },
    ];

    const fakeResponse: RingEliminationChoiceResponsePayload = {
      selectedOption: options[1],
      aiType: 'heuristic',
      difficulty: 5,
    };

    const fakeClient = {
      getRingEliminationChoice: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    };

    engine.createAIFromProfile(2, profile);

    const selected = await engine.getRingEliminationChoice(2, null, options);

    expect(fakeClient.getRingEliminationChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getRingEliminationChoice.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(2);
    expect(callArgs[2]).toBe(5);
    expect(callArgs[3]).toBe(ServiceAIType.HEURISTIC);
    expect(callArgs[4]).toBe(options);

    expect(selected).toBe(options[1]);
  });

  it('getRegionOrderChoice calls AIServiceClient.getRegionOrderChoice and returns the selected option', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const options: RegionOrderChoice['options'] = [
      {
        regionId: 'small',
        size: 3,
        representativePosition: { x: 0, y: 0 },
        moveId: 'reg-1',
      },
      {
        regionId: 'large',
        size: 7,
        representativePosition: { x: 5, y: 5 },
        moveId: 'reg-2',
      },
    ];

    const fakeResponse: RegionOrderChoiceResponsePayload = {
      selectedOption: options[1],
      aiType: 'heuristic',
      difficulty: 6,
    };

    const fakeClient = {
      getRegionOrderChoice: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 6,
      mode: 'service',
      aiType: 'heuristic',
    };

    engine.createAIFromProfile(3, profile);

    const selected = await engine.getRegionOrderChoice(3, null, options);

    expect(fakeClient.getRegionOrderChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getRegionOrderChoice.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(3);
    expect(callArgs[2]).toBe(6);
    expect(callArgs[3]).toBe(ServiceAIType.HEURISTIC);
    expect(callArgs[4]).toBe(options);

    expect(selected).toBe(options[1]);
  });

  it('getLineOrderChoice calls AIServiceClient.getLineOrderChoice and returns the selected option', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const options: LineOrderChoice['options'] = [
      {
        moveId: 'm-short',
        lineId: 'short',
        markerPositions: [{ x: 0, y: 0 }],
      },
      {
        moveId: 'm-long',
        lineId: 'long',
        markerPositions: [
          { x: 1, y: 1 },
          { x: 2, y: 2 },
        ],
      },
    ];

    const fakeResponse: LineOrderChoiceResponsePayload = {
      selectedOption: options[0],
      aiType: 'heuristic',
      difficulty: 5,
    };

    const fakeClient = {
      getLineOrderChoice: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic',
    };

    engine.createAIFromProfile(4, profile);

    const selected = await engine.getLineOrderChoice(4, null, options);

    expect(fakeClient.getLineOrderChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getLineOrderChoice.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(4);
    expect(callArgs[2]).toBe(5);
    expect(callArgs[3]).toBe(ServiceAIType.HEURISTIC);
    expect(callArgs[4]).toBe(options);

    expect(selected).toBe(options[0]);
  });

  it('getCaptureDirectionChoice calls AIServiceClient.getCaptureDirectionChoice and returns the selected option', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const options: CaptureDirectionChoice['options'] = [
      {
        targetPosition: { x: 3, y: 3 },
        landingPosition: { x: 5, y: 5 },
        capturedCapHeight: 2,
      },
      {
        targetPosition: { x: 4, y: 4 },
        landingPosition: { x: 6, y: 6 },
        capturedCapHeight: 3,
      },
    ];

    const fakeResponse: CaptureDirectionChoiceResponsePayload = {
      selectedOption: options[1],
      aiType: 'heuristic',
      difficulty: 6,
    };

    const fakeClient = {
      getCaptureDirectionChoice: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 6,
      mode: 'service',
      aiType: 'heuristic',
    };

    engine.createAIFromProfile(5, profile);

    const selected = await engine.getCaptureDirectionChoice(5, null, options);

    expect(fakeClient.getCaptureDirectionChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getCaptureDirectionChoice.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(5);
    expect(callArgs[2]).toBe(6);
    expect(callArgs[3]).toBe(ServiceAIType.HEURISTIC);
    expect(callArgs[4]).toBe(options);

    expect(selected).toBe(options[1]);
  });

  it('AI_DIFFICULTY_PRESETS matches canonical mapping for difficulties 1–10', () => {
    const expected: Record<
      number,
      { aiType: InternalAIType; randomness: number; thinkTime: number }
    > = {
      1: { aiType: InternalAIType.RANDOM, randomness: 0.5, thinkTime: 150 },
      2: { aiType: InternalAIType.HEURISTIC, randomness: 0.3, thinkTime: 200 },
      3: { aiType: InternalAIType.MINIMAX, randomness: 0.15, thinkTime: 1800 },
      4: { aiType: InternalAIType.MINIMAX, randomness: 0.08, thinkTime: 2800 },
      5: { aiType: InternalAIType.MCTS, randomness: 0.05, thinkTime: 4000 },
      6: { aiType: InternalAIType.MCTS, randomness: 0.02, thinkTime: 5500 },
      7: { aiType: InternalAIType.MCTS, randomness: 0.0, thinkTime: 7500 },
      8: { aiType: InternalAIType.MCTS, randomness: 0.0, thinkTime: 9600 },
      9: { aiType: InternalAIType.DESCENT, randomness: 0.0, thinkTime: 12600 },
      10: { aiType: InternalAIType.DESCENT, randomness: 0.0, thinkTime: 16000 },
    };

    for (const [key, expectedPreset] of Object.entries(expected)) {
      const difficulty = Number(key);
      const preset = AI_DIFFICULTY_PRESETS[difficulty];

      expect(preset.aiType).toBe(expectedPreset.aiType);
      expect(preset.randomness).toBeCloseTo(expectedPreset.randomness);
      expect(preset.thinkTime).toBe(expectedPreset.thinkTime);
    }
  });

  it('createAIFromProfile uses preset aiType/randomness/thinkTime when aiType is omitted', () => {
    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 7,
      mode: 'service',
    };

    engine.createAIFromProfile(1, profile);
    const config = engine.getAIConfig(1);

    expect(config).toBeDefined();
    // Difficulty 7 should map to MCTS per the canonical ladder.
    expect(config!.aiType).toBe(InternalAIType.MCTS);
    expect(config!.randomness).toBeCloseTo(0.0);
    expect(config!.thinkTime).toBe(7500);
  });

  it('getAIMove uses canonical ladder aiType and service mapping for difficulty 8 (MCTS)', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeMove: Move = {
      id: 'svc-move-mcts',
      type: 'move_ring',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const fakeResponse: MoveResponse = {
      move: fakeMove,
      evaluation: 0.0,
      thinking_time_ms: 50,
      ai_type: 'mcts',
      difficulty: 8,
    };

    const fakeClient = {
      getAIMove: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 8,
      mode: 'service',
      // No explicit aiType → rely on canonical ladder mapping (MCTS).
    };

    engine.createAIFromProfile(1, profile);

    const gameState: GameState = {
      id: 'test-game-mcts',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: new Map(),
        size: 8,
      } as any,
      players: [
        {
          id: 'ai-player-1',
          username: 'AI Player 1',
          playerNumber: 1,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 8,
          aiProfile: profile,
        },
      ] as any,
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 42,
    };

    mockRuleEngineMoves = [
      fakeMove,
      {
        ...fakeMove,
        id: 'svc-move-mcts-2',
        to: { x: 2, y: 0 },
        moveNumber: 2,
      },
    ];

    await engine.getAIMove(1, gameState);

    expect(fakeClient.getAIMove).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getAIMove.mock.calls[0];
    expect(callArgs[2]).toBe(8);
    expect(callArgs[3]).toBe(ServiceAIType.MCTS);
  });

  it('getAIMove uses canonical ladder aiType and service mapping for difficulty 10 (Descent)', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeMove: Move = {
      id: 'svc-move-descent',
      type: 'move_ring',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const fakeResponse: MoveResponse = {
      move: fakeMove,
      evaluation: 0.0,
      thinking_time_ms: 50,
      ai_type: 'descent',
      difficulty: 10,
    };

    const fakeClient = {
      getAIMove: jest.fn().mockResolvedValue(fakeResponse),
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const engine = new AIEngine();

    const profile: AIProfile = {
      difficulty: 10,
      mode: 'service',
      // No explicit aiType → rely on canonical ladder mapping (Descent).
    };

    engine.createAIFromProfile(1, profile);

    const gameState: GameState = {
      id: 'test-game-descent',
      boardType: 'square8',
      board: {
        type: 'square8',
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: new Map(),
        size: 8,
      } as any,
      players: [
        {
          id: 'ai-player-1',
          username: 'AI Player 1',
          playerNumber: 1,
          type: 'ai',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 10,
          eliminatedRings: 0,
          territorySpaces: 0,
          aiDifficulty: 10,
          aiProfile: profile,
        },
      ] as any,
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
      rngSeed: 84,
    };

    mockRuleEngineMoves = [
      fakeMove,
      {
        ...fakeMove,
        id: 'svc-move-descent-2',
        to: { x: 2, y: 0 },
        moveNumber: 2,
      },
    ];

    await engine.getAIMove(1, gameState);

    expect(fakeClient.getAIMove).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getAIMove.mock.calls[0];
    expect(callArgs[2]).toBe(10);
    expect(callArgs[3]).toBe(ServiceAIType.DESCENT);
  });
});
