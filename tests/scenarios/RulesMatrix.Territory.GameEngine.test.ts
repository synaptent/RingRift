import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  Position,
  TimeControl,
  positionToString
} from '../../src/shared/types/game';
import { pos, addStack } from '../utils/fixtures';
import {
  territoryRuleScenarios,
  TerritoryRuleScenario
} from './rulesMatrix';

/**
 * RulesMatrix → GameEngine territory scenarios
 *
 * These tests replay Section 12 / FAQ Q23-style disconnected-region examples
 * using the data-only TerritoryRuleScenario definitions from rulesMatrix.ts.
 *
 * They complement tests/unit/GameEngine.territory.scenarios.test.ts by making
 * the self-elimination prerequisite explicit in the shared scenario matrix.
 */

describe('RulesMatrix → GameEngine territory scenarios (Section 12; FAQ Q23)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createPlayers(): Player[] {
    // Match the shape used in GameEngine.territory.scenarios.test.ts: three
    // human players with full ringsInHand and zero eliminated/territory.
    return [
      {
        id: 'p1',
        username: 'Player1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'p2',
        username: 'Player2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0
      },
      {
        id: 'p3',
        username: 'Player3',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0
      }
    ];
  }

  const q23Scenarios: TerritoryRuleScenario[] = territoryRuleScenarios.filter((s) =>
    s.ref.id.startsWith('Rules_12_2_Q23_')
  );

  test.each<TerritoryRuleScenario>(q23Scenarios)(
    '%s → backend GameEngine territory processing respects self-elimination prerequisite',
    async (scenario) => {
      const players = createPlayers();
      const engine = new GameEngine(
        'rules-matrix-territory',
        scenario.boardType as BoardType,
        players,
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState: GameState = engineAny.gameState as GameState;
      const board = gameState.board;
      const boardManager: any = engineAny.boardManager;

      gameState.currentPlayer = scenario.movingPlayer;

      const [region] = scenario.regions;
      const interiorCoords: Position[] = region.spaces.map((p) =>
        (p as any).z != null ? pos(p.x, p.y, (p as any).z) : pos(p.x, p.y)
      );

      // Place stacks for the victim player inside the disconnected region.
      for (const p of interiorCoords) {
        addStack(board, p, region.victimPlayer, 1);
      }

      if (region.movingPlayerHasOutsideStack) {
        // Give the moving player a single stack outside the region so they can
        // satisfy the self-elimination prerequisite.
        const outsidePos = pos(0, 1);
        addStack(board, outsidePos, scenario.movingPlayer, 2);

        const movingStacksOutside = boardManager.getPlayerStacks(board, scenario.movingPlayer);
        expect(movingStacksOutside.length).toBeGreaterThan(0);
      } else {
        // Ensure the moving player has no stacks anywhere on the board.
        const movingStacks = boardManager.getPlayerStacks(board, scenario.movingPlayer);
        expect(movingStacks.length).toBe(0);
      }

      const regionTerritory = {
        spaces: interiorCoords,
        controllingPlayer: region.controllingPlayer,
        isDisconnected: true
      };

      const findDisconnectedRegionsSpy = jest
        .spyOn(boardManager, 'findDisconnectedRegions')
        .mockImplementationOnce(() => [regionTerritory])
        .mockImplementation(() => []);

      const initialCollapsedCount = board.collapsedSpaces.size;
      const initialTotalEliminated = gameState.totalRingsEliminated;
      const initialMovingEliminated =
        gameState.players.find((p) => p.playerNumber === scenario.movingPlayer)?.eliminatedRings ?? 0;

      await (engineAny as any).processDisconnectedRegions();

      expect(findDisconnectedRegionsSpy).toHaveBeenCalled();

      if (!region.movingPlayerHasOutsideStack) {
        // Q23 negative case: because the moving player has no stacks outside
        // the region, it MUST NOT be processed.
        expect(board.collapsedSpaces.size).toBe(initialCollapsedCount);

        const stacksInRegion = Array.from(board.stacks.keys()).filter((key) => {
          return interiorCoords.some((p) => positionToString(p) === key);
        });
        expect(stacksInRegion.length).toBe(interiorCoords.length);

        const finalTotalEliminated = gameState.totalRingsEliminated;
        const finalMovingEliminated =
          gameState.players.find((p) => p.playerNumber === scenario.movingPlayer)?.eliminatedRings ?? 0;
        expect(finalTotalEliminated).toBe(initialTotalEliminated);
        expect(finalMovingEliminated).toBe(initialMovingEliminated);
      } else {
        // Q23 positive case: with at least one outside stack, the region MUST
        // be processed and the moving player must pay the self-elimination cost.
        for (const p of interiorCoords) {
          const key = positionToString(p);
          expect(board.collapsedSpaces.get(key)).toBe(region.controllingPlayer);
          expect(board.stacks.has(key)).toBe(false);
        }

        const finalCollapsedCount = board.collapsedSpaces.size;
        const finalTotalEliminated = gameState.totalRingsEliminated;
        const finalMovingEliminated =
          gameState.players.find((p) => p.playerNumber === scenario.movingPlayer)?.eliminatedRings ?? 0;

        expect(finalCollapsedCount).toBeGreaterThan(initialCollapsedCount);
        expect(finalTotalEliminated).toBeGreaterThan(initialTotalEliminated);
        expect(finalMovingEliminated).toBeGreaterThan(initialMovingEliminated);
      }
    }
  );
});