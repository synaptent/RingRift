import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  Position,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { addStack, pos } from '../utils/fixtures';
import { territoryRuleScenarios, TerritoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → GameEngine elimination decision scenarios
 *
 * Focused tests for explicit `eliminate_rings_from_stack` Moves driven
 * by the Q23-positive territory scenario in rulesMatrix.ts.
 */
describe('RulesMatrix &#8594; GameEngine eliminate_rings_from_stack (territory; Q23)', () => {
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
        ringsInHand: 36,
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
        ringsInHand: 36,
        eliminatedRings: 0,
        territorySpaces: 0,
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
        territorySpaces: 0,
      },
    ];
  }

  const eliminationScenarios: TerritoryRuleScenario[] = territoryRuleScenarios.filter(
    (s) => s.ref.id === 'Rules_12_2_Q23_region_processed_with_self_elimination_square19'
  );

  test.each<TerritoryRuleScenario>(eliminationScenarios)(
    '%s &#8594; explicit elimination Move removes cap and increases S',
    async (scenario) => {
      const players = createPlayers();
      const engine = new GameEngine(
        'rules-matrix-elimination',
        scenario.boardType as BoardType,
        players,
        timeControl,
        false
      );
      const engineAny: any = engine;
      const gameState: GameState = engineAny.gameState as GameState;
      const board = gameState.board;

      gameState.gameStatus = 'active';
      gameState.currentPlayer = scenario.movingPlayer;
      gameState.currentPhase = 'territory_processing';

      const [region] = scenario.regions;
      const interiorCoords: Position[] = region.spaces.map((p) =>
        (p as any).z != null ? pos(p.x, p.y, (p as any).z) : pos(p.x, p.y)
      );

      // Populate victim stacks inside the disconnected region.
      for (const p of interiorCoords) {
        addStack(board, p, region.victimPlayer, 1);
      }

      const outsidePos: Position = region.outsideStackPosition ?? pos(0, 1);
      const outsideHeight: number = region.selfEliminationStackHeight ?? 2;

      // Self-elimination stack for the moving player outside the region.
      addStack(board, outsidePos, scenario.movingPlayer, outsideHeight);

      const outsideKey = positionToString(outsidePos);
      const stackBefore = board.stacks.get(outsideKey);
      expect(stackBefore).toBeDefined();
      const capHeight = stackBefore!.capHeight;
      expect(capHeight).toBeGreaterThan(0);

      const playerBefore = gameState.players.find((p) => p.playerNumber === scenario.movingPlayer)!;
      const initialPlayerEliminated = playerBefore.eliminatedRings;
      const initialTotalEliminated = gameState.totalRingsEliminated;
      const progressBefore = computeProgressSnapshot(gameState);

      const move: Move = {
        id: '',
        type: 'eliminate_rings_from_stack',
        player: scenario.movingPlayer,
        to: outsidePos,
        eliminatedRings: [{ player: scenario.movingPlayer, count: capHeight }],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      };

      const result = await engine.makeMove({
        type: move.type,
        player: move.player,
        to: move.to,
        eliminatedRings: move.eliminatedRings,
      } as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);

      expect(result.success).toBe(true);
      const finalState = result.gameState as GameState;
      const finalBoard = finalState.board;
      const finalPlayer = finalState.players.find((p) => p.playerNumber === scenario.movingPlayer)!;

      const finalStack = finalBoard.stacks.get(outsideKey);
      // Because the stack was a pure cap for the moving player, the
      // entire stack should be gone after elimination.
      expect(finalStack).toBeUndefined();

      expect(finalPlayer.eliminatedRings).toBe(initialPlayerEliminated + capHeight);
      expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated + capHeight);

      const progressAfter = computeProgressSnapshot(finalState);
      expect(progressAfter.S).toBeGreaterThan(progressBefore.S);
    }
  );
});
