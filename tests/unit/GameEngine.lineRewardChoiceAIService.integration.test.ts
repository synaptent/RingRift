import { GameEngine } from '../../src/server/game/GameEngine';
import { PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';
import { AIInteractionHandler } from '../../src/server/game/ai/AIInteractionHandler';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';
import { getAIServiceClient, LineRewardChoiceResponsePayload } from '../../src/server/services/AIServiceClient';
import {
  BoardType,
  GameState,
  LineRewardChoice,
  Move,
  Player,
  Position,
  TimeControl,
  AIProfile
} from '../../src/shared/types/game';

jest.mock('../../src/server/services/AIServiceClient');

describe('GameEngine + AIInteractionHandler + AIServiceClient line_reward_option integration', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const players: Player[] = [
    {
      id: 'ai-red',
      username: 'AI Red',
      type: 'ai',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    },
    {
      id: 'blue',
      username: 'Blue',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  const makeStack = (playerNumber: number, height: number, position: Position) => {
    const rings = Array(height).fill(playerNumber);
    return {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber
    };
  };

  beforeEach(() => {
    jest.clearAllMocks();
    globalAIEngine.clearAll();
  });

  it('uses the AI service-selected line_reward_option when the service succeeds', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeResponse: LineRewardChoiceResponsePayload = {
      selectedOption: 'option_1_collapse_all_and_eliminate',
      aiType: 'heuristic',
      difficulty: 5
    };

    const fakeClient = {
      getLineRewardChoice: jest.fn().mockResolvedValue(fakeResponse)
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const profile: AIProfile = {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic'
    };

    globalAIEngine.createAIFromProfile(1, profile);

    const handler = new AIInteractionHandler();
    const interactionManager = new PlayerInteractionManager(handler);

    const engine = new GameEngine(
      'ai-line-reward-success',
      boardType,
      players,
      timeControl,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    gameState.currentPlayer = 1;

    const markerPositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 }
    ];

    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy.mockImplementationOnce(() => [
      {
        player: 1,
        positions: markerPositions
      }
    ]);
    findAllLinesSpy.mockImplementation(() => []);

    const stackPos: Position = { x: 0, y: 1 };
    boardManager.setStack(stackPos, makeStack(1, 3, stackPos), gameState.board);

    const initialEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;

    await engineAny.processLineFormations();

    expect(fakeClient.getLineRewardChoice).toHaveBeenCalledTimes(1);
    const callArgs = fakeClient.getLineRewardChoice.mock.calls[0];
    // gameState is currently passed as null from AIInteractionHandler → AIEngine
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]).toBe(1);
    expect(callArgs[2]).toBe(5);
    expect(callArgs[4]).toEqual(['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination']);

    for (const pos of markerPositions) {
      expect(boardManager.isCollapsedSpace(pos, gameState.board)).toBe(true);
    }

    const finalEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;
    expect(finalEliminated).toBeGreaterThan(initialEliminated);
  });

  it('falls back to local heuristic when the AI service fails', async () => {
    const mockedGetClient = getAIServiceClient as jest.MockedFunction<typeof getAIServiceClient>;

    const fakeClient = {
      getLineRewardChoice: jest.fn().mockRejectedValue(new Error('service down'))
    } as any;

    mockedGetClient.mockReturnValue(fakeClient);

    const profile: AIProfile = {
      difficulty: 5,
      mode: 'service',
      aiType: 'heuristic'
    };

    globalAIEngine.createAIFromProfile(1, profile);

    const handler = new AIInteractionHandler();
    const interactionManager = new PlayerInteractionManager(handler);

    const engine = new GameEngine(
      'ai-line-reward-fallback',
      boardType,
      players,
      timeControl,
      false,
      interactionManager
    );

    const engineAny: any = engine;
    const boardManager = engineAny.boardManager as any;
    const gameState = engineAny.gameState as GameState;

    gameState.currentPlayer = 1;

    const markerPositions: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 0 },
      { x: 3, y: 0 },
      { x: 4, y: 0 }
    ];

    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy.mockImplementationOnce(() => [
      {
        player: 1,
        positions: markerPositions
      }
    ]);
    findAllLinesSpy.mockImplementation(() => []);

    const stackPos: Position = { x: 0, y: 1 };
    boardManager.setStack(stackPos, makeStack(1, 3, stackPos), gameState.board);

    const initialEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;

    await engineAny.processLineFormations();

    expect(fakeClient.getLineRewardChoice).toHaveBeenCalledTimes(1);

    // Fallback path should behave like the local heuristic: choose
    // Option 2 (minimum collapse, no elimination). On an 8x8 board
    // this collapses only the first requiredLength=4 markers to
    // territory and leaves the overlong tail marker uncollapsed.
    expect(boardManager.isCollapsedSpace(markerPositions[0], gameState.board)).toBe(true);
    expect(boardManager.isCollapsedSpace(markerPositions[1], gameState.board)).toBe(true);
    expect(boardManager.isCollapsedSpace(markerPositions[2], gameState.board)).toBe(true);
    expect(boardManager.isCollapsedSpace(markerPositions[3], gameState.board)).toBe(true);
    expect(boardManager.isCollapsedSpace(markerPositions[4], gameState.board)).toBe(false);

    const finalEliminated = gameState.players.find(p => p.playerNumber === 1)!
      .eliminatedRings;
    expect(finalEliminated).toBe(initialEliminated);
  });
});
