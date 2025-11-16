import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameResult,
  GameState,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice
} from '../../src/shared/types/game';
import { addStack, pos } from '../utils/fixtures';

/**
 * ClientSandboxEngine victory-condition tests.
 *
 * These mirror the ring-elimination and territory-control checks
 * performed by GameEngine but exercise the client-local sandbox
 * harness instead.
 */

describe('ClientSandboxEngine victory conditions (square19)', () => {
  const boardType: BoardType = 'square19';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human']
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends any>(choice: TChoice): Promise<PlayerChoiceResponseFor<any>> {
        const anyChoice = choice as CaptureDirectionChoice;
        const selectedOption = (anyChoice as any).options
          ? (anyChoice as any).options[0]
          : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption
        } as PlayerChoiceResponseFor<any>;
      }
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('ring-elimination victory when a player reaches victoryThreshold', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Sanity: victoryResult starts null and game is active.
    expect(engine.getVictoryResult()).toBeNull();
    expect(state.gameStatus).toBe('active');

    // Force player 1 to be just below their ring-elimination threshold,
    // then apply one more elimination via totalRingsEliminated.
    const p1 = state.players.find(p => p.playerNumber === 1)!;
    const threshold = state.victoryThreshold;
    p1.eliminatedRings = threshold - 1;

    // Simulate an additional elimination attributed to player 1.
    state.board.eliminatedRings[1] = threshold;
    p1.eliminatedRings = threshold;

    // Run the victory check directly.
    engineAny.checkAndApplyVictory();

    const result: GameResult | null = engine.getVictoryResult();
    const finalState = engine.getGameState();

    expect(result).not.toBeNull();
    expect(result!.reason).toBe('ring_elimination');
    expect(result!.winner).toBe(1);
    expect(finalState.gameStatus).toBe('completed');
    expect(finalState.winner).toBe(1);
  });

  test('territory-control victory when a player reaches territoryVictoryThreshold', () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;

    // Reset any existing victory state.
    engineAny.victoryResult = null;
    state.gameStatus = 'active';
    state.winner = undefined as any;

    const p1 = state.players.find(p => p.playerNumber === 1)!;
    const territoryThreshold = state.territoryVictoryThreshold;

    // Directly set player 1's territory to threshold and re-run checks.
    p1.territorySpaces = territoryThreshold;

    engineAny.checkAndApplyVictory();

    const result: GameResult | null = engine.getVictoryResult();
    const finalState = engine.getGameState();

    expect(result).not.toBeNull();
    expect(result!.reason).toBe('territory_control');
    expect(result!.winner).toBe(1);
    expect(finalState.gameStatus).toBe('completed');
    expect(finalState.winner).toBe(1);
  });
});
