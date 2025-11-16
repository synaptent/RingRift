import { PlayerChoice, PlayerChoiceResponse, PlayerType } from '../../shared/types/game';
import { PlayerInteractionHandler } from './PlayerInteractionManager';

/**
 * DelegatingInteractionHandler
 *
 * A composite PlayerInteractionHandler that routes choices to either a
 * human-facing handler (e.g. WebSocketInteractionHandler) or an AI-facing
 * handler (e.g. AIInteractionHandler) based on the player type.
 *
 * This keeps GameEngine dependent only on PlayerInteractionManager while
 * allowing different transport/decision mechanisms per player.
 */
export class DelegatingInteractionHandler implements PlayerInteractionHandler {
  constructor(
    private readonly humanHandler: PlayerInteractionHandler | null,
    private readonly aiHandler: PlayerInteractionHandler | null,
    private readonly getPlayerType: (playerNumber: number) => PlayerType
  ) {}

  async requestChoice(choice: PlayerChoice): Promise<PlayerChoiceResponse<unknown>> {
    const playerType = this.getPlayerType(choice.playerNumber);

    if (playerType === 'ai') {
      if (!this.aiHandler) {
        throw new Error(
          `DelegatingInteractionHandler: AI handler not configured for AI player ${choice.playerNumber}`
        );
      }
      return this.aiHandler.requestChoice(choice);
    }

    // Default path: human player. If a human handler is not available,
    // fall back to the AI handler as a defensive measure so that games
    // do not hang indefinitely; this mirrors the historic engine
    // behaviour of auto-selecting the first option when no interaction
    // system was wired.
    if (this.humanHandler) {
      return this.humanHandler.requestChoice(choice);
    }

    if (this.aiHandler) {
      return this.aiHandler.requestChoice(choice);
    }

    throw new Error('DelegatingInteractionHandler: no handlers configured for choices');
  }
}
