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

  function createThreePlayerEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
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
      s.ref.id === 'Rules_13_2_territory_control_threshold_square8' ||
      s.ref.id === 'Rules_13_3_last_player_standing_3p_unique_actor_square8'
  );

  test.each<VictoryRuleScenario>(scenarios)(
    '%s → sandbox victory semantics match rules/FAQ expectations',
    (scenario) => {
      if (scenario.ref.id === 'Rules_13_1_ring_elimination_threshold_square8') {
        const engine = createEngine();
        const engineAny: any = engine;
        const state: GameState = engineAny.gameState as GameState;

        // Reset any existing victory state.
        engineAny.victoryResult = null;
        state.gameStatus = 'active';
        (state as any).winner = undefined;

        const player1 = state.players.find((p) => p.playerNumber === 1)!;
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
        const engine = createEngine();
        const engineAny: any = engine;
        const state: GameState = engineAny.gameState as GameState;

        // Reset any existing victory state.
        engineAny.victoryResult = null;
        state.gameStatus = 'active';
        (state as any).winner = undefined;

        const player1 = state.players.find((p) => p.playerNumber === 1)!;
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
      } else if (scenario.ref.id === 'Rules_13_3_last_player_standing_3p_unique_actor_square8') {
        const engine = createThreePlayerEngine();
        const engineAny: any = engine;
        const state: GameState = engineAny.gameState as GameState;

        // Ensure LPS starts from an active game.
        engineAny.victoryResult = null;
        state.gameStatus = 'active';
        (state as any).winner = undefined;

        const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };
        engineAny.hasAnyRealActionForPlayer = jest.fn(
          (playerNumber: number) => !!realActionByPlayer[playerNumber]
        );

        const startInteractiveTurn = (playerNumber: number): GameResult | null => {
          state.currentPlayer = playerNumber;
          state.currentPhase = 'ring_placement';
          engineAny.handleStartOfInteractiveTurn();
          return engine.getVictoryResult();
        };

        // Round 1: P1, P2, P3 - only P1 has actions
        let result = startInteractiveTurn(1);
        expect(result).toBeNull();

        result = startInteractiveTurn(2);
        expect(result).toBeNull();

        result = startInteractiveTurn(3);
        expect(result).toBeNull();

        // Round 2: P1, P2, P3 - still only P1 has actions (2nd consecutive round)
        result = startInteractiveTurn(1);
        expect(result).toBeNull();

        result = startInteractiveTurn(2);
        expect(result).toBeNull();

        result = startInteractiveTurn(3);
        expect(result).toBeNull();

        // Round 3: P1, P2, P3 - still only P1 has actions (3rd consecutive round)
        result = startInteractiveTurn(1);
        expect(result).toBeNull();

        result = startInteractiveTurn(2);
        expect(result).toBeNull();

        result = startInteractiveTurn(3);
        expect(result).toBeNull();

        // Round 4 start: P1's turn - after 3 consecutive exclusive rounds, LPS triggers
        result = startInteractiveTurn(1);

        expect(result).not.toBeNull();
        expect(result!.reason).toBe('last_player_standing');
        expect(result!.winner).toBe(1);

        const finalState = engine.getGameState();
        expect(finalState.gameStatus).toBe('completed');
        expect(finalState.winner).toBe(1);
      } else {
        throw new Error(`Unhandled victory scenario id: ${scenario.ref.id}`);
      }
    }
  );
});
