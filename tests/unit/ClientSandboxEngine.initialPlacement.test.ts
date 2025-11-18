import { ClientSandboxEngine, SandboxConfig } from '../../src/client/sandbox/ClientSandboxEngine';
import { BoardType, GameHistoryEntry } from '../../src/shared/types/game';
import { PlayerChoice, PlayerChoiceResponseFor } from '../../src/shared/types/game';

class NoopInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;
    const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
    return {
      choiceId: anyChoice.id,
      playerNumber: anyChoice.playerNumber,
      choiceType: anyChoice.type,
      selectedOption
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

/**
 * Diagnostic test: verify that a fresh sandbox AI game starts with
 * Player 1 in ring_placement on an empty board, and that the first
 * AI turn produces a canonical place_ring move by Player 1 recorded
 * in GameState.history.
 */
describe('ClientSandboxEngine initial AI placement behaviour', () => {
  const boardType: BoardType = 'square8';

  test('first AI turn from fresh game produces a P1 place_ring history entry', async () => {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['ai', 'ai']
    };

    const engine = new ClientSandboxEngine({ config, interactionHandler: new NoopInteractionHandler() });

    const initial = engine.getGameState();
    expect(initial.currentPlayer).toBe(1);
    expect(initial.currentPhase).toBe('ring_placement');
    expect(initial.board.stacks.size).toBe(0);
    expect(initial.players.find(p => p.playerNumber === 1)?.ringsInHand).toBeGreaterThan(0);

    await engine.maybeRunAITurn();

    const after = engine.getGameState();
    const history: GameHistoryEntry[] = after.history;

    // This test is intentionally strict so we can catch trace-completeness
    // issues early. If it fails, it means the sandbox is either skipping
    // P1's opening placement or not recording it in history.
    expect(history.length).toBeGreaterThan(0);

    const firstEntry = history[0];
    expect(firstEntry.actor).toBe(1);
    expect(firstEntry.action.type).toBe('place_ring');
    expect(firstEntry.action.player).toBe(1);
    expect(firstEntry.phaseBefore).toBe('ring_placement');
  });
});
