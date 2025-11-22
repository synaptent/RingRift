import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameResult,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';
import { victoryRuleScenarios, VictoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → ClientSandboxEngine victory scenarios
 *
 * Data-driven sandbox checks for §13.1 (ring-elimination) and §13.2
 * (territory-control) using victoryRuleScenarios defined in rulesMatrix.ts.
 */

describe('RulesMatrix → ClientSandboxEngine victory scenarios (sandbox)', () => {
  const boardType: BoardType = 'square8';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;
        const options: any[] = (anyChoice.options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  const scenarios: VictoryRuleScenario[] = victoryRuleScenarios.filter(
    (s) =>
      s.ref.id === 'Rules_13_1_ring_elimination_threshold_square8' ||
      s.ref.id === 'Rules_13_2_territory_control_threshold_square8'
  );

  test.each<VictoryRuleScenario>(scenarios)(
    '%s → sandbox checkAndApplyVictory matches victory threshold semantics',
    (scenario) => {
      const engine = createEngine();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      // Reset any existing victory state.
      engineAny.victoryResult = null;
      state.gameStatus = 'active';
      (state as any).winner = undefined;

      const player1 = state.players.find((p) => p.playerNumber === 1)!;

      if (scenario.ref.id === 'Rules_13_1_ring_elimination_threshold_square8') {
        const threshold = state.victoryThreshold;
        player1.eliminatedRings = threshold;
        state.board.eliminatedRings[1] = threshold;
        state.totalRingsEliminated = threshold;

        engineAny.checkAndApplyVictory();

        const result: GameResult | null = engine.getVictoryResult();
        const finalState = engine.getGameState();

        expect(result).not.toBeNull();
        expect(result!.reason).toBe('ring_elimination');
        expect(result!.winner).toBe(1);
        expect(finalState.gameStatus).toBe('completed');
        expect(finalState.winner).toBe(1);
      } else if (scenario.ref.id === 'Rules_13_2_territory_control_threshold_square8') {
        const threshold = state.territoryVictoryThreshold;
        player1.territorySpaces = threshold;

        engineAny.checkAndApplyVictory();

        const result: GameResult | null = engine.getVictoryResult();
        const finalState = engine.getGameState();

        expect(result).not.toBeNull();
        expect(result!.reason).toBe('territory_control');
        expect(result!.winner).toBe(1);
        expect(finalState.gameStatus).toBe('completed');
        expect(finalState.winner).toBe(1);
      } else {
        throw new Error(`Unhandled victory scenario id: ${scenario.ref.id}`);
      }
    }
  );
});
