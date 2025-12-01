import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Move,
  Position,
  PlayerChoiceResponseFor,
  positionToString,
} from '../../src/shared/types/game';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { addStack, pos } from '../utils/fixtures';
import { territoryRuleScenarios, TerritoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → ClientSandboxEngine elimination decision scenarios
 *
 * Mirrors the backend elimination RulesMatrix test but runs entirely
 * through the sandbox canonical Move-applier. This asserts that
 * explicit `eliminate_rings_from_stack` Moves:
 *
 * - Remove the chosen cap/stack,
 * - Increase eliminatedRings and totalRingsEliminated appropriately,
 * - Increase the S-invariant.
 *
 * Test #2 sets _pendingTerritorySelfElimination flag to simulate the state
 * where the engine has determined that a self-elimination is required.
 */
describe('RulesMatrix → ClientSandboxEngine eliminate_rings_from_stack (territory; Q23)', () => {
  function createEngine(boardType: BoardType): { engine: ClientSandboxEngine; state: GameState } {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // Generic handler: always pick the first option for any choice so that
      // RegionOrderChoice / RingEliminationChoice can be satisfied if surfaced.
      async requestChoice(choice: any): Promise<PlayerChoiceResponseFor<any>> {
        const optionsArray = ((choice as any).options as any[]) ?? [];
        const selectedOption = optionsArray.length > 0 ? optionsArray[0] : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        } as PlayerChoiceResponseFor<any>;
      },
    };

    const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;
    return { engine, state };
  }

  const eliminationScenarios: TerritoryRuleScenario[] = territoryRuleScenarios.filter(
    (s) => s.ref.id === 'Rules_12_2_Q23_region_processed_with_self_elimination_square19'
  );

  test.each<TerritoryRuleScenario>(eliminationScenarios)(
    '%s → sandbox explicit elimination Move removes cap and increases S',
    async (scenario) => {
      const { engine, state } = createEngine(scenario.boardType as BoardType);
      const engineAny: any = engine;
      const board = state.board;

      state.currentPlayer = scenario.movingPlayer;
      state.currentPhase = 'territory_processing';

      const [region] = scenario.regions;
      const interiorCoords: Position[] = region.spaces.map((p) =>
        (p as any).z != null ? pos(p.x, p.y, (p as any).z) : pos(p.x, p.y)
      );

      // Victim stacks inside the disconnected region (not directly used
      // by this elimination Move, but representative of the Q23 layout).
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

      const movingPlayerBefore = state.players.find(
        (p) => p.playerNumber === scenario.movingPlayer
      )!;
      const initialPlayerEliminated = movingPlayerBefore.eliminatedRings;
      const initialTotalEliminated = state.totalRingsEliminated;
      const progressBefore = computeProgressSnapshot(state);

      const move: Move = {
        id: '',
        type: 'eliminate_rings_from_stack',
        player: scenario.movingPlayer,
        to: outsidePos,
        eliminatedRings: [{ player: scenario.movingPlayer, count: capHeight }],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      } as Move;

      await engine.applyCanonicalMove(move);

      const finalState: GameState = (engineAny.gameState as GameState) || state;
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

  test.each<TerritoryRuleScenario>(eliminationScenarios)(
    '%s → sandbox canonical elimination decision enumeration exposes self-elimination cap stack',
    (scenario) => {
      const { engine, state } = createEngine(scenario.boardType as BoardType);
      const engineAny: any = engine;
      const board = state.board;

      state.currentPlayer = scenario.movingPlayer;
      state.currentPhase = 'territory_processing';

      const [region] = scenario.regions;
      const interiorCoords: Position[] = region.spaces.map((p) =>
        (p as any).z != null ? pos(p.x, p.y, (p as any).z) : pos(p.x, p.y)
      );

      // Victim stacks inside the disconnected region. As in the explicit
      // elimination test, these do not affect the sandbox elimination
      // enumeration gating because there are no markers or collapsed
      // spaces, so sandboxTerritory.findDisconnectedRegionsOnBoard
      // returns no disconnected regions.
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

      // Set the pending elimination flag to simulate the state after territory
      // processing has determined that a self-elimination is required. Without
      // this flag, getValidEliminationDecisionMovesForCurrentPlayer() returns []
      // because the engine gates enumeration on having an outstanding debt.
      engineAny._pendingTerritorySelfElimination = true;

      const moves: Move[] = engineAny.getValidEliminationDecisionMovesForCurrentPlayer();
      const elimMoves = moves.filter((m) => m.type === 'eliminate_rings_from_stack');

      expect(elimMoves.length).toBeGreaterThan(0);

      const matching = elimMoves.find((m) => m.to && positionToString(m.to) === outsideKey);

      expect(matching).toBeDefined();

      if (matching) {
        // Diagnostic fields should reflect the underlying stack geometry.
        const diag = matching.eliminationFromStack;
        expect(diag).toBeDefined();
        if (diag) {
          expect(positionToString(diag.position)).toBe(outsideKey);
          expect(diag.capHeight).toBe(capHeight);
          expect(diag.totalHeight).toBe(stackBefore!.stackHeight);
        }

        const er = matching.eliminatedRings && matching.eliminatedRings[0];
        expect(er).toBeDefined();
        if (er) {
          expect(er.player).toBe(scenario.movingPlayer);
          expect(er.count).toBe(capHeight);
        }
      }
    }
  );
});
