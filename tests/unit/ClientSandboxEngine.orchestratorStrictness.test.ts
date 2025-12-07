import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  BoardType,
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';

/**
 * Minimal interaction handler: always picks the first option.
 */
class TestInteractionHandler implements SandboxInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;
    const selectedOption = Array.isArray(anyChoice.options) ? anyChoice.options[0] : undefined;
    return {
      choiceId: anyChoice.id,
      playerNumber: anyChoice.playerNumber,
      choiceType: anyChoice.type,
      selectedOption,
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

function createEngine(boardType: BoardType = 'square8'): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers: 2,
    playerKinds: ['human', 'human'],
  };

  return new ClientSandboxEngine({
    config,
    interactionHandler: new TestInteractionHandler(),
    traceMode: true,
  });
}

describe('ClientSandboxEngine orchestrator strictness', () => {
  test('applyCanonicalMoveForReplay rejects mis-phased place_ring moves', async () => {
    const engine = createEngine();
    const handler = new TestInteractionHandler();

    // Construct a serialized state that is mid-line_processing but has no
    // recorded decision moves; applying a place_ring here should be rejected
    // by the orchestrator/phase invariant.
    const baseState = engine.getGameState();
    const invalidState: GameState = {
      ...baseState,
      currentPhase: 'line_processing',
    };

    engine.initFromSerializedState(invalidState as any, ['human', 'human'], handler);

    const badMove: Move = {
      id: 'bad-place-ring',
      type: 'place_ring',
      player: invalidState.currentPlayer,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    await expect(engine.applyCanonicalMoveForReplay(badMove)).rejects.toThrow(
      /processMove failed|Invalid move/i
    );
  });
});
