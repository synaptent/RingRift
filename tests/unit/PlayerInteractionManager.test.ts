import { PlayerChoice, PlayerChoiceResponse, Position } from '../../src/shared/types/game';
import { PlayerInteractionHandler, PlayerInteractionManager } from '../../src/server/game/PlayerInteractionManager';

describe('PlayerInteractionManager', () => {
  const baseChoice = {
    id: 'choice-1',
    gameId: 'game-1',
    playerNumber: 1,
    prompt: 'Test choice'
  } as const;

  const samplePositions: Position[] = [
    { x: 0, y: 0 },
    { x: 1, y: 1 }
  ];

  it('forwards choices to the handler and returns the typed selectedOption', async () => {
    const choice: PlayerChoice = {
      ...baseChoice,
      type: 'line_order',
      options: [
        { lineId: '0', markerPositions: samplePositions },
        { lineId: '1', markerPositions: samplePositions }
      ]
    };

    const handler: PlayerInteractionHandler = {
      requestChoice: async (incomingChoice) => {
        // echo the first option as the selectedOption
        const response: PlayerChoiceResponse<(typeof choice.options)[number]> = {
          choiceId: incomingChoice.id,
          playerNumber: incomingChoice.playerNumber,
          selectedOption: (choice.options as any)[0]
        };
        return response;
      }
    };

    const manager = new PlayerInteractionManager(handler);

    const response = await manager.requestChoice(choice);

    expect(response.choiceId).toBe(choice.id);
    expect(response.playerNumber).toBe(choice.playerNumber);
    expect(response.selectedOption.lineId).toBe('0');
    expect(response.selectedOption.markerPositions).toEqual(samplePositions);
  });

  it('throws if handler responds with a different playerNumber', async () => {
    const choice: PlayerChoice = {
      ...baseChoice,
      type: 'line_reward_option',
      options: [
        'option_1_collapse_all_and_eliminate',
        'option_2_min_collapse_no_elimination'
      ]
    };

    const handler: PlayerInteractionHandler = {
      requestChoice: async (incomingChoice) => {
        const response: PlayerChoiceResponse = {
          choiceId: incomingChoice.id,
          playerNumber: incomingChoice.playerNumber + 1, // wrong player
          selectedOption: choice.options[0]
        };
        return response;
      }
    };

    const manager = new PlayerInteractionManager(handler);

    await expect(manager.requestChoice(choice)).rejects.toThrow(
      /response.playerNumber .* does not match choice.playerNumber/
    );
  });
});
