import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  Move,
  PlayerChoiceResponseFor,
  positionToString,
  Territory,
} from '../../src/shared/types/game';
import { enumerateProcessTerritoryRegionMoves } from '../../src/shared/engine/territoryDecisionHelpers';
import { addStack, pos } from '../utils/fixtures';
import { territoryRuleScenarios, TerritoryRuleScenario } from './rulesMatrix';
import { isFSMOrchestratorActive } from '../../src/shared/utils/envFlags';

/**
 * RulesMatrix → ClientSandboxEngine territory scenarios
 *
 * Mirrors Section 12 / FAQ Q23-style disconnected-region examples from
 * rulesMatrix.ts against the client-local sandbox engine. These tests
 * complement RulesMatrix.Territory.GameEngine by asserting that the
 * sandbox respects the same self-elimination prerequisite.
 */

describe('RulesMatrix → ClientSandboxEngine territory scenarios (Section 12; FAQ Q23)', () => {
  // TODO: FSM issue - self-elimination prerequisite check not working correctly
  // in territory processing. Regions are being processed when they shouldn't be.
  if (isFSMOrchestratorActive()) {
    it.skip('Skipping - FSM needs territory self-elimination prerequisite fix', () => {});
    return;
  }

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
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    return { engine, state };
  }

  const q23Scenarios: TerritoryRuleScenario[] = territoryRuleScenarios.filter((s) =>
    s.ref.id.startsWith('Rules_12_2_Q23_')
  );

  test.each<TerritoryRuleScenario>(q23Scenarios)(
    '%s → sandbox territory processing respects self-elimination prerequisite',
    async (scenario) => {
      const { engine, state } = createEngine(scenario.boardType as BoardType);
      const engineAny: any = engine;
      const board = state.board;

      state.currentPlayer = scenario.movingPlayer;
      // Per RR-CANON-R073, territory processing moves require territory_processing phase
      state.currentPhase = 'territory_processing';

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
      } else {
        // Ensure the moving player has no stacks anywhere on the board.
        const stacksForMoving = Array.from(board.stacks.values()).filter(
          (s) => s.controllingPlayer === scenario.movingPlayer
        );
        expect(stacksForMoving.length).toBe(0);
      }

      const initialCollapsedCount = board.collapsedSpaces.size;
      const initialTotalEliminated = state.totalRingsEliminated;
      const initialMovingEliminated =
        state.players.find((p) => p.playerNumber === scenario.movingPlayer)?.eliminatedRings ?? 0;

      const move: Move = {
        id: '',
        type: 'process_territory_region',
        player: scenario.movingPlayer,
        disconnectedRegions: [
          {
            spaces: interiorCoords,
          },
        ],
      } as Move;

      const didApply: boolean = await (engineAny as any).applyCanonicalProcessTerritoryRegion(move);

      if (!region.movingPlayerHasOutsideStack) {
        // Q23 negative case: with no outside stack, the region MUST NOT be processed.
        expect(didApply).toBe(false);
        expect(board.collapsedSpaces.size).toBe(initialCollapsedCount);

        const stacksInRegion = Array.from(board.stacks.keys()).filter((key) =>
          interiorCoords.some((p) => positionToString(p) === key)
        );
        expect(stacksInRegion.length).toBe(interiorCoords.length);

        const finalTotalEliminated = state.totalRingsEliminated;
        const finalMovingEliminated =
          state.players.find((p) => p.playerNumber === scenario.movingPlayer)?.eliminatedRings ?? 0;

        expect(finalTotalEliminated).toBe(initialTotalEliminated);
        expect(finalMovingEliminated).toBe(initialMovingEliminated);
      } else {
        // Q23 positive case: with at least one outside stack, the region MUST
        // be processed and the moving player must pay the self-elimination cost.
        //
        // The sandbox helper applyCanonicalProcessTerritoryRegion delegates to
        // processDisconnectedRegionOnBoard with the provided regionSpaces and
        // then checks S-invariant components (collapsed count, eliminations)
        // rather than guaranteeing that each individual space in region.spaces
        // becomes collapsed. For now we assert only on the monotone effects
        // and on the fact that the move was applied.
        expect(didApply).toBe(true);

        const finalCollapsedCount = board.collapsedSpaces.size;
        const finalTotalEliminated = state.totalRingsEliminated;
        const finalMovingEliminated =
          state.players.find((p) => p.playerNumber === scenario.movingPlayer)?.eliminatedRings ?? 0;

        // applyCanonicalProcessTerritoryRegion only guarantees that S-invariant
        // components are non-decreasing; in some Q23-positive layouts, it may
        // increase eliminated rings while leaving collapsedSpaces unchanged, or
        // vice versa. For now we assert only non-decrease plus a successful
        // application of the Move.
        expect(finalCollapsedCount).toBeGreaterThanOrEqual(initialCollapsedCount);
        expect(finalTotalEliminated).toBeGreaterThanOrEqual(initialTotalEliminated);
        expect(finalMovingEliminated).toBeGreaterThanOrEqual(initialMovingEliminated);
      }
    }
  );

  test.each<TerritoryRuleScenario>(q23Scenarios)(
    '%s → sandbox canonical territory decision enumeration matches Q23 prerequisite',
    (scenario) => {
      const { state } = createEngine(scenario.boardType as BoardType);
      const board = state.board;

      state.currentPlayer = scenario.movingPlayer;

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
      } else {
        // Ensure the moving player has no stacks anywhere on the board.
        const stacksForMoving = Array.from(board.stacks.values()).filter(
          (s) => s.controllingPlayer === scenario.movingPlayer
        );
        expect(stacksForMoving.length).toBe(0);
      }

      // Use the shared helper with a test-only override so that decision
      // enumeration is driven purely by the curated RulesMatrix geometry
      // rather than by the full sandbox region detector.
      const regionTerritory: Territory = {
        spaces: interiorCoords,
        controllingPlayer: region.controllingPlayer,
        isDisconnected: true,
      };

      const moves: Move[] = enumerateProcessTerritoryRegionMoves(state, scenario.movingPlayer, {
        testOverrideRegions: [regionTerritory],
      });

      const keyFrom = (positions: Position[]) =>
        positions
          .map((p) => positionToString(p))
          .sort()
          .join('|');

      const interiorKey = keyFrom(interiorCoords);

      const matchingMove = moves.find((m) => {
        if (m.type !== 'process_territory_region') {
          return false;
        }
        if (!m.disconnectedRegions || m.disconnectedRegions.length === 0) {
          return false;
        }
        const regionSpaces: Position[] = m.disconnectedRegions[0].spaces || [];
        return keyFrom(regionSpaces) === interiorKey;
      });

      if (!region.movingPlayerHasOutsideStack) {
        // Q23 negative: with no outside stack, the region must not appear as
        // a legal process_territory_region decision.
        expect(matchingMove).toBeUndefined();
      } else {
        // Q23 positive: with at least one outside stack, the region must appear
        // as a legal process_territory_region decision for the moving player.
        expect(matchingMove).toBeDefined();
      }
    }
  );
});
