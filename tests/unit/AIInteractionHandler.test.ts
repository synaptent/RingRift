import { AIInteractionHandler } from '../../src/server/game/ai/AIInteractionHandler';
import {
  CaptureDirectionChoice,
  LineOrderChoice,
  LineRewardChoice,
  PlayerChoice,
  PlayerChoiceResponse,
  Position,
  RegionOrderChoice,
  RingEliminationChoice,
} from '../../src/shared/types/game';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';
import { logger } from '../../src/server/utils/logger';

/**
 * Unit tests for AIInteractionHandler
 *
 * These tests validate that the handler:
 * - Implements the PlayerInteractionHandler contract
 * - Returns a valid selectedOption drawn from choice.options
 * - Applies simple, deterministic heuristics for each choice type
 */

jest.mock('../../src/server/game/ai/AIEngine', () => {
  const mockGlobalAIEngine = {
    getLineRewardChoice: jest.fn(),
    getRingEliminationChoice: jest.fn(),
    getRegionOrderChoice: jest.fn(),
    getLineOrderChoice: jest.fn(),
    getCaptureDirectionChoice: jest.fn(),
    // By default, behave as if the AI is in `service` mode so that
    // service-backed paths remain exercised unless a test overrides
    // this mock.
    getAIConfig: jest.fn(() => ({ difficulty: 5, aiType: 'heuristic', mode: 'service' })),
  };

  return { globalAIEngine: mockGlobalAIEngine };
});

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('AIInteractionHandler', () => {
  const handler = new AIInteractionHandler();

  const baseChoice = {
    id: 'choice-1',
    gameId: 'game-1',
    playerNumber: 1,
    prompt: 'Test choice',
  } as const;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns a PlayerChoiceResponse with matching choiceId and playerNumber', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const response: PlayerChoiceResponse<unknown> = await handler.requestChoice(
      choice as PlayerChoice
    );

    expect(response.choiceId).toBe(choice.id);
    expect(response.playerNumber).toBe(choice.playerNumber);
  });

  it('prefers longer lines for line_order choices', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    expect(['short', 'long']).toContain(selected.lineId);
    expect(selected.markerPositions.length).toBe(3);
    expect(selected.lineId).toBe('long');
  });

  it('prefers Option 2 for line_reward_option when available (falling back from service)', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    expect(selected).toBe('option_2_min_collapse_no_elimination');
  });

  it('chooses stack with smallest capHeight for ring_elimination choices', async () => {
    const choice: RingEliminationChoice = {
      ...baseChoice,
      type: 'ring_elimination',
      options: [
        {
          moveId: 'm-a',
          stackPosition: { x: 0, y: 0 },
          capHeight: 3,
          totalHeight: 5,
        },
        {
          moveId: 'm-b',
          stackPosition: { x: 1, y: 1 },
          capHeight: 1,
          totalHeight: 4,
        },
        {
          moveId: 'm-c',
          stackPosition: { x: 2, y: 2 },
          capHeight: 1,
          totalHeight: 6,
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RingEliminationChoice['options'][number];

    expect(selected.capHeight).toBe(1);
    // Among capHeight === 1, the option with smaller totalHeight (4) should win
    expect(selected.totalHeight).toBe(4);
    expect(selected.stackPosition).toEqual({ x: 1, y: 1 });
  });

  it('chooses largest region for region_order choices', async () => {
    const choice: RegionOrderChoice = {
      ...baseChoice,
      type: 'region_order',
      options: [
        {
          moveId: 'm-small',
          regionId: 'small',
          size: 3,
          representativePosition: { x: 0, y: 0 },
        },
        {
          moveId: 'm-large',
          regionId: 'large',
          size: 7,
          representativePosition: { x: 5, y: 5 },
        },
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as RegionOrderChoice['options'][number];

    expect(selected.regionId).toBe('large');
    expect(selected.size).toBe(7);
  });

  it('prefers higher capturedCapHeight for capture_direction choices, tie-breaking by distance to centre', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
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
      ],
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(3);
    expect(selected.targetPosition).toEqual({ x: 4, y: 4 });
  });

  it('uses AI service line_reward_option when it returns a valid option', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineRewardChoice: jest.Mock;
    };

    mockEngine.getLineRewardChoice.mockResolvedValue('option_1_collapse_all_and_eliminate');

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    expect(mockEngine.getLineRewardChoice).toHaveBeenCalledWith(
      choice.playerNumber,
      null,
      choice.options
    );
    // Service choice should override the local "prefer option 2" heuristic
    expect(selected).toBe('option_1_collapse_all_and_eliminate');
  });

  it('falls back to local heuristic and logs when AI service returns invalid line_reward_option', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineRewardChoice: jest.Mock;
    };

    mockEngine.getLineRewardChoice.mockResolvedValue('not_a_valid_option' as any);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineRewardChoice['options'][number];

    // Fallback heuristic still prefers option 2
    expect(selected).toBe('option_2_min_collapse_no_elimination');

    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI service returned invalid option for line_reward_option'),
      expect.objectContaining({
        gameId: choice.gameId,
        playerNumber: choice.playerNumber,
        choiceId: choice.id,
        choiceType: choice.type,
      })
    );
  });

  it('uses AI service line_order when it returns a valid option', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineOrderChoice: jest.Mock;
    };

    mockEngine.getLineOrderChoice.mockResolvedValue(choice.options[0]);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    expect(mockEngine.getLineOrderChoice).toHaveBeenCalledWith(
      choice.playerNumber,
      null,
      choice.options
    );
    expect(selected.lineId).toBe('short');
  });

  it('falls back to local line_order heuristic and logs when AI service returns invalid option', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 },
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { moveId: 'm-short', lineId: 'short', markerPositions: positionsB },
        { moveId: 'm-long', lineId: 'long', markerPositions: positionsA },
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getLineOrderChoice: jest.Mock;
    };

    mockEngine.getLineOrderChoice.mockResolvedValue({} as any);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as LineOrderChoice['options'][number];

    // Fallback heuristic should still pick the longer line ("long").
    expect(selected.lineId).toBe('long');

    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI service returned invalid option for line_order'),
      expect.objectContaining({
        gameId: choice.gameId,
        playerNumber: choice.playerNumber,
        choiceId: choice.id,
        choiceType: choice.type,
      })
    );
  });

  it('uses AI service capture_direction when it returns a valid option', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
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
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getCaptureDirectionChoice: jest.Mock;
    };

    mockEngine.getCaptureDirectionChoice.mockResolvedValue(choice.options[0]);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(mockEngine.getCaptureDirectionChoice).toHaveBeenCalledWith(
      choice.playerNumber,
      null,
      choice.options
    );
    expect(selected).toBe(choice.options[0]);
  });

  it('falls back to local capture_direction heuristic and logs when AI service returns invalid option', async () => {
    const choice: CaptureDirectionChoice = {
      ...baseChoice,
      type: 'capture_direction',
      options: [
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
      ],
    };

    const mockEngine = globalAIEngine as unknown as {
      getCaptureDirectionChoice: jest.Mock;
    };

    mockEngine.getCaptureDirectionChoice.mockResolvedValue({} as any);

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    // Fallback heuristic should still prefer the higher capturedCapHeight (3).
    expect(selected.capturedCapHeight).toBe(3);
    expect(logger.warn).toHaveBeenCalledWith(
      expect.stringContaining('AI service returned invalid option for capture_direction'),
      expect.objectContaining({
        gameId: choice.gameId,
        playerNumber: choice.playerNumber,
        choiceId: choice.id,
        choiceType: choice.type,
      })
    );
  });

  it('logs and throws when a generic choice has no options', async () => {
    const choice = {
      ...baseChoice,
      type: 'unknown_choice_type',
      options: [] as string[],
    } as unknown as PlayerChoice;

    await expect(handler.requestChoice(choice)).rejects.toThrow(
      'PlayerChoice[unknown_choice_type] must have at least one option'
    );

    expect(logger.error).toHaveBeenCalledWith(
      'AIInteractionHandler received choice with no options',
      expect.objectContaining({
        choiceId: baseChoice.id,
        playerNumber: baseChoice.playerNumber,
      })
    );
  });
});
