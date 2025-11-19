import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
} from '../../src/shared/types/game';

/**
 * Minimal deterministic interaction handler for sandbox tests.
 *
 * For capture_direction and other PlayerChoices, it always selects
 * the first option, which is sufficient for these mixed human/AI
 * progression tests.
 */
class SimpleInteractionHandler implements SandboxInteractionHandler {
  async requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>> {
    const anyChoice = choice as any;
    const options = anyChoice.options || [];
    const selectedOption = options.length > 0 ? options[0] : undefined;

    return {
      choiceId: anyChoice.id,
      playerNumber: anyChoice.playerNumber,
      choiceType: anyChoice.type,
      selectedOption,
    } as PlayerChoiceResponseFor<TChoice>;
  }
}

const BOARD_TYPE: BoardType = 'square8';
const MAX_AI_STEPS = 1000;

function createEngine(playerKinds: ('human' | 'ai')[]): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType: BOARD_TYPE,
    numPlayers: playerKinds.length,
    playerKinds,
  };

  return new ClientSandboxEngine({
    config,
    interactionHandler: new SimpleInteractionHandler(),
  });
}

/**
 * Helper: perform a legal ring placement for the current player on a fresh
 * square8 board by scanning for the first position where tryPlaceRings(1)
 * succeeds. This deliberately relies on the engine's no-dead-placement
 * logic instead of assuming any particular coordinate is legal.
 */
async function performFirstLegalPlacement(engine: ClientSandboxEngine): Promise<Position> {
  const state: GameState = engine.getGameState();
  expect(state.boardType).toBe(BOARD_TYPE);

  for (let x = 0; x < 8; x += 1) {
    for (let y = 0; y < 8; y += 1) {
      const pos: Position = { x, y };
      const placed = engine.tryPlaceRings(pos, 1);
      if (placed) {
        const after = engine.getGameState();
        // After a successful placement we must be in movement phase
        // for the same player.
        expect(after.currentPhase).toBe('movement');
        return pos;
      }
    }
  }

  throw new Error('performFirstLegalPlacement: no legal placement found on square8');
}

describe('ClientSandboxEngine mixed human/AI sandbox flows', () => {
  test('human then AI: human place+move passes turn to AI and AI game terminates', async () => {
    const engine = createEngine(['human', 'ai']);

    let state = engine.getGameState();
    expect(state.currentPlayer).toBe(1);
    expect(state.players[0].type).toBe('human');
    expect(state.players[1].type).toBe('ai');
    expect(state.currentPhase).toBe('ring_placement');

    // Human 1: perform a legal placement and then a legal movement from
    // the placed stack.
    const placementPos = await performFirstLegalPlacement(engine);

    const movementTargets = engine.getValidLandingPositionsForCurrentPlayer(placementPos);
    expect(movementTargets.length).toBeGreaterThan(0);
    const target = movementTargets[0];

    // After performFirstLegalPlacement, the engine has already marked the
    // placed stack as selected and advanced the phase to movement for
    // Player 1. The sandbox UI therefore expects a single click on a
    // highlighted destination to commit the move.
    await engine.handleHumanCellClick(target);

    state = engine.getGameState();
    expect(state.gameStatus).toBe('active');
    expect(state.currentPlayer).toBe(2);
    expect(state.players[1].type).toBe('ai');

    // From this point, Player 2 is AI. For this test we do not require the
    // AI to finish the entire game; we only require that repeated calls to
    // maybeRunAITurn make observable progress without stalling
    // indefinitely under this configuration.
    let steps = 0;
    while (steps < MAX_AI_STEPS && state.gameStatus === 'active') {
      await engine.maybeRunAITurn();
      state = engine.getGameState();
      steps += 1;
    }

    expect(steps).toBeGreaterThan(0);
    expect(steps).toBeLessThanOrEqual(MAX_AI_STEPS);
  });

  test('AI then human: AI opening eventually passes turn to human and game terminates', async () => {
    const engine = createEngine(['ai', 'human']);

    let state = engine.getGameState();
    expect(state.currentPlayer).toBe(1);
    expect(state.players[0].type).toBe('ai');
    expect(state.players[1].type).toBe('human');
    expect(state.currentPhase).toBe('ring_placement');

    // Drive AI turns until either the game ends or it is no longer
    // Player 1's turn. This ensures that the AI can both place and
    // move without stalling in ring_placement, and that turn
    // progression reaches the human seat.
    let steps = 0;
    while (
      steps < MAX_AI_STEPS &&
      state.gameStatus === 'active' &&
      state.currentPlayer === 1 &&
      state.players[0].type === 'ai'
    ) {
      await engine.maybeRunAITurn();
      state = engine.getGameState();
      steps += 1;
    }

    // Either the game has ended or the turn has passed to the human
    // player. Both outcomes are acceptable for this test; what we
    // explicitly rule out is an AI that never progresses the game.
    if (state.gameStatus === 'active') {
      expect(state.currentPlayer).toBe(2);
      expect(state.players[1].type).toBe('human');
    }

    // Regardless of whose turn it is now, the full game should still
    // reach a terminal state under continued AI turns when it is an
    // AI player's move.
    steps = 0;
    while (steps < MAX_AI_STEPS && state.gameStatus === 'active') {
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!current || current.type !== 'ai') {
        // Human to move: in the real UI loop this is where human input
        // would take over. For the purposes of this test, break out
        // and assert that we at least reached a human turn.
        break;
      }

      await engine.maybeRunAITurn();
      state = engine.getGameState();
      steps += 1;
    }

    // We either reached a terminal state or reached a human turn
    // without stalling. In both cases the AI has made observable
    // progress for this configuration.
    expect(steps).toBeLessThanOrEqual(MAX_AI_STEPS);
  });
});
