import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString
} from '../../src/shared/types/game';
import { lineRewardRuleScenarios, LineRewardRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → GameEngine backend scenarios
 *
 * This suite exercises a small set of high-value rules/FAQ-aligned scenarios
 * defined in `rulesMatrix.ts` against the real GameEngine implementation.
 *
 * The goal is to provide a canonical, data-driven pattern that can be reused
 * for additional rules clusters (movement, chain captures, territory, victory,
 * etc.) without duplicating scenario wiring logic in every test file.
 */

describe('RulesMatrix → GameEngine line-reward scenarios (backend)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
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
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0
    }
  ];

  function createEngine(boardType: BoardType): {
    engine: GameEngine;
    gameState: GameState;
    boardManager: any;
  } {
    const engine = new GameEngine('rules-matrix-lines', boardType, basePlayers, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: any = engineAny.boardManager;
    return { engine, gameState, boardManager };
  }

  function makeStack(
    boardManager: any,
    gameState: GameState,
    playerNumber: number,
    height: number,
    position: Position
  ) {
    const rings = Array(height).fill(playerNumber);
    const stack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber
    };
    boardManager.setStack(position, stack, gameState.board);
  }

  const scenarios: LineRewardRuleScenario[] = lineRewardRuleScenarios.filter(
    (s) =>
      s.ref.id === 'Rules_11_2_Q7_exact_length_line' ||
      s.ref.id === 'Rules_11_3_Q22_overlength_line_option2_default'
  );

  test.each<LineRewardRuleScenario>(scenarios)(
    '%s → backend line processing matches rules/FAQ expectations',
    async (scenario) => {
      const { engine, gameState, boardManager } = createEngine(scenario.boardType);
      const engineAny: any = engine;
      const engineState: GameState = gameState;
      const board = engineState.board;

      const requiredLength = BOARD_CONFIGS[scenario.boardType].lineLength;
      const totalLength = requiredLength + scenario.overlengthBy;

      engineState.currentPlayer = 1;

      // Clear any existing board state for a clean scenario.
      board.markers.clear();
      board.stacks.clear();
      board.collapsedSpaces.clear();

      // Synthetic horizontal line starting at x=0 on the configured row.
      const linePositions: Position[] = [];
      for (let i = 0; i < totalLength; i++) {
        linePositions.push({ x: i, y: scenario.rowIndex });
      }

      const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
      findAllLinesSpy
        .mockImplementationOnce(() => [
          {
            player: 1,
            positions: linePositions,
            length: linePositions.length,
            direction: { x: 1, y: 0 }
          }
        ])
        .mockImplementation(() => []);

      // Provide a stack for player 1 that can be used for elimination in the
      // exact-length case. For overlength/Option 2, this stack should remain.
      const stackPos: Position = { x: 7, y: 7 };
      makeStack(boardManager, engineState, 1, 2, stackPos);

      const player1Before = engineState.players.find((p) => p.playerNumber === 1)!;
      const initialTerritory = player1Before.territorySpaces;
      const initialEliminated = player1Before.eliminatedRings;
      const initialTotalEliminated = engineState.totalRingsEliminated;
      const initialCollapsed = board.collapsedSpaces.size;

      await engineAny.processLineFormations();

      const player1After = engineState.players.find((p) => p.playerNumber === 1)!;
      const collapsedKeysAfter = new Set<string>();
      for (const [key, owner] of board.collapsedSpaces) {
        if (owner === 1) collapsedKeysAfter.add(key);
      }

      const collapsedDelta = collapsedKeysAfter.size - initialCollapsed;
      const eliminatedDeltaPlayer1 = player1After.eliminatedRings - initialEliminated;
      const totalEliminatedDelta = engineState.totalRingsEliminated - initialTotalEliminated;
      const stackKey = positionToString(stackPos);
      const id = scenario.ref.id;

      const isExact =
        scenario.overlengthBy === 0 && id === 'Rules_11_2_Q7_exact_length_line';
      const isOption2Default =
        id === 'Rules_11_3_Q22_overlength_line_option2_default';
      const isOption1FullCollapse =
        id === 'Rules_11_3_Q22_overlength_line_option1_full_collapse_square19';

      if (isExact) {
        // Rules_11_2_Q7_exact_length_line: all markers in the line are
        // collapsed, one cap/ring is eliminated, and territory increases by
        // exactly the required line length.
        expect(collapsedDelta).toBe(requiredLength);
        expect(eliminatedDeltaPlayer1).toBeGreaterThan(0);
        expect(totalEliminatedDelta).toBeGreaterThan(0);
        expect(player1After.territorySpaces).toBe(initialTerritory + requiredLength);
        expect(board.stacks.get(stackKey)).toBeUndefined();
      } else if (isOption2Default) {
        // Rules_11_3_Q22_overlength_line_option2_default: overlength line
        // with no PlayerInteractionManager defaults to Option 2 - collapse the
        // minimum required markers, preserve one marker segment, and perform
        // NO elimination.
        expect(collapsedDelta).toBe(requiredLength);
        expect(eliminatedDeltaPlayer1).toBe(0);
        expect(totalEliminatedDelta).toBe(0);
        expect(player1After.territorySpaces).toBe(initialTerritory + requiredLength);
        expect(board.stacks.get(stackKey)).toBeDefined();
      } else if (isOption1FullCollapse) {
        // Rules_11_3_Q22_overlength_line_option1_full_collapse_square19:
        // overlength line where the moving player explicitly chooses Option 1,
        // collapsing the entire line and eliminating one of their rings/caps.
        expect(collapsedDelta).toBe(totalLength);
        expect(eliminatedDeltaPlayer1).toBeGreaterThan(0);
        expect(totalEliminatedDelta).toBeGreaterThan(0);
        expect(player1After.territorySpaces).toBe(initialTerritory + totalLength);
        expect(board.stacks.get(stackKey)).toBeUndefined();
      } else {
        throw new Error(`Unhandled line-reward scenario: ${id}`);
      }

      expect(findAllLinesSpy).toHaveBeenCalled();
    }
  );
});
