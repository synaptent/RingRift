import { PlayerChoice, PlayerChoiceResponse, PlayerChoiceResponseFor } from '../../shared/types/game';

/**
 * PlayerInteractionHandler is the bridge between the pure game engine and the outside world.
 *
 * Implementations of this interface are responsible for:
 * - Emitting PlayerChoice requests to human clients (e.g. via WebSocket)
 * - Or routing them to AI (e.g. Python AI service) when the player is AI-controlled
 * - Returning a validated PlayerChoiceResponse
 *
 * The game engine itself should only depend on this abstraction, not on any
 * particular transport or UI technology.
 */
export interface PlayerInteractionHandler {
  requestChoice: (choice: PlayerChoice) => Promise<PlayerChoiceResponse<unknown>>;
}

/**
 * PlayerInteractionManager provides a type-safe facade over a PlayerInteractionHandler.
 *
 * It narrows the selectedOption type based on the concrete PlayerChoice.options element type,
 * so callers in GameEngine get strong typing for the response they receive.
 */
export class PlayerInteractionManager {
  constructor(private readonly handler: PlayerInteractionHandler) {}

  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const response = (await this.handler.requestChoice(choice)) as PlayerChoiceResponse<
      TChoice['options'][number]
    >;

    // Basic sanity check: ensure the responding player matches the choice target.
    if (response.playerNumber !== choice.playerNumber) {
      throw new Error(
        `PlayerInteractionManager: response.playerNumber (${response.playerNumber}) does not match choice.playerNumber (${choice.playerNumber})`
      );
    }

    // Ensure choiceType is always present and correctly narrowed for
    // downstream discriminated-union style handling.
    const enriched: PlayerChoiceResponseFor<TChoice> = {
      ...(response as PlayerChoiceResponse<TChoice['options'][number]>),
      choiceType: response.choiceType ?? choice.type
    } as PlayerChoiceResponseFor<TChoice>;

    return enriched;
  }
}
