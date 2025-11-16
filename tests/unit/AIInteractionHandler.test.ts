import { AIInteractionHandler } from '../../src/server/game/ai/AIInteractionHandler';
import {
  CaptureDirectionChoice,
  LineOrderChoice,
  LineRewardChoice,
  PlayerChoice,
  PlayerChoiceResponse,
  Position,
  RegionOrderChoice,
  RingEliminationChoice
} from '../../src/shared/types/game';

/**
 * Unit tests for AIInteractionHandler
 *
 * These tests validate that the handler:
 * - Implements the PlayerInteractionHandler contract
 * - Returns a valid selectedOption drawn from choice.options
 * - Applies simple, deterministic heuristics for each choice type
 */

describe('AIInteractionHandler', () => {
  const handler = new AIInteractionHandler();

  const baseChoice = {
    id: 'choice-1',
    gameId: 'game-1',
    playerNumber: 1,
    prompt: 'Test choice'
  } as const;

  it('returns a PlayerChoiceResponse with matching choiceId and playerNumber', async () => {
    const choice: LineRewardChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: [
        'option_1_collapse_all_and_eliminate',
        'option_2_min_collapse_no_elimination'
      ]
    };

    const response: PlayerChoiceResponse<unknown> = await handler.requestChoice(choice as PlayerChoice);

    expect(response.choiceId).toBe(choice.id);
    expect(response.playerNumber).toBe(choice.playerNumber);
  });

  it('prefers longer lines for line_order choices', async () => {
    const positionsA: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 }
    ];
    const positionsB: Position[] = [
      { x: 3, y: 3 },
      { x: 4, y: 4 }
    ];

    const choice: LineOrderChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { lineId: 'short', markerPositions: positionsB },
        { lineId: 'long', markerPositions: positionsA }
      ]
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
      options: [
        'option_1_collapse_all_and_eliminate',
        'option_2_min_collapse_no_elimination'
      ]
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
          stackPosition: { x: 0, y: 0 },
          capHeight: 3,
          totalHeight: 5
        },
        {
          stackPosition: { x: 1, y: 1 },
          capHeight: 1,
          totalHeight: 4
        },
        {
          stackPosition: { x: 2, y: 2 },
          capHeight: 1,
          totalHeight: 6
        }
      ]
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
          regionId: 'small',
          size: 3,
          representativePosition: { x: 0, y: 0 }
        },
        {
          regionId: 'large',
          size: 7,
          representativePosition: { x: 5, y: 5 }
        }
      ]
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
          capturedCapHeight: 2
        },
        {
          targetPosition: { x: 4, y: 4 },
          landingPosition: { x: 6, y: 6 },
          capturedCapHeight: 3
        }
      ]
    };

    const response = await handler.requestChoice(choice as PlayerChoice);
    const selected = response.selectedOption as CaptureDirectionChoice['options'][number];

    expect(selected.capturedCapHeight).toBe(3);
    expect(selected.targetPosition).toEqual({ x: 4, y: 4 });
  });
});
