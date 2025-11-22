import { GameEngine } from '../../src/server/game/GameEngine';
import { BoardType, GameState, Player, TimeControl } from '../../src/shared/types/game';
import { victoryRuleScenarios, VictoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → GameEngine victory scenarios
 *
 * Data-driven backend checks for §13.1 (ring-elimination) and §13.2
 * (territory-control) using victoryRuleScenarios defined in rulesMatrix.ts.
 */

describe('RulesMatrix → GameEngine victory scenarios (backend)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createPlayers(): Player[] {
    return [
      {
        id: 'p1',
        username: 'Player1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p2',
        username: 'Player2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
  }

  const scenarios: VictoryRuleScenario[] = victoryRuleScenarios.filter(
    (s) =>
      s.ref.id === 'Rules_13_1_ring_elimination_threshold_square8' ||
      s.ref.id === 'Rules_13_2_territory_control_threshold_square8'
  );

  test.each<VictoryRuleScenario>(scenarios)(
    '%s → backend RuleEngine.checkGameEnd matches victory threshold semantics',
    (scenario) => {
      const engine = new GameEngine(
        `rules-matrix-victory-${scenario.ref.id}`,
        boardType,
        createPlayers(),
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState: GameState = engineAny.gameState as GameState;

      const player1 = gameState.players.find((p) => p.playerNumber === 1)!;

      if (scenario.ref.id === 'Rules_13_1_ring_elimination_threshold_square8') {
        const threshold = gameState.victoryThreshold;
        player1.eliminatedRings = threshold;
        gameState.totalRingsEliminated = threshold;
        gameState.board.eliminatedRings[1] = threshold;

        const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);
        expect(endCheck.isGameOver).toBe(true);
        expect(endCheck.winner).toBe(1);
        expect(endCheck.reason).toBe('ring_elimination');
      } else if (scenario.ref.id === 'Rules_13_2_territory_control_threshold_square8') {
        const threshold = gameState.territoryVictoryThreshold;
        player1.territorySpaces = threshold;

        const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);
        expect(endCheck.isGameOver).toBe(true);
        expect(endCheck.winner).toBe(1);
        expect(endCheck.reason).toBe('territory_control');
      } else {
        throw new Error(`Unhandled victory scenario id: ${scenario.ref.id}`);
      }
    }
  );
});
