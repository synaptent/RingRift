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
  Territory,
  RegionOrderChoice,
  positionToString,
} from '../../src/shared/types/game';
import * as sandboxTerritory from '../../src/client/sandbox/sandboxTerritory';

/**
 * Sandbox RegionOrderChoice integration test.
 *
 * Mirrors GameEngine.regionOrderChoiceIntegration.test.ts but exercises
 * ClientSandboxEngine + SandboxInteractionHandler instead of the
 * server-side GameEngine + WebSocketInteractionHandler.
 */

// Skip with orchestrator adapter - this test spies on internal sandbox methods
// that are bypassed when the orchestrator adapter is enabled.
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'ClientSandboxEngine region order choice integration',
  () => {
    const boardType: BoardType = 'square8';

    function createEngineWithChoiceHandler(): {
      engine: ClientSandboxEngine;
      handler: SandboxInteractionHandler;
    } {
      const config: SandboxConfig = {
        boardType,
        numPlayers: 2,
        playerKinds: ['human', 'human'],
      };

      const handler: SandboxInteractionHandler = {
        async requestChoice<TChoice extends PlayerChoice>(
          choice: TChoice
        ): Promise<PlayerChoiceResponseFor<TChoice>> {
          // For RegionOrderChoice, deliberately select the SECOND option
          // to verify that the sandbox engine processes the chosen
          // region first.
          if (choice.type === 'region_order') {
            const regionChoice = choice as RegionOrderChoice;
            const options = regionChoice.options;
            const selectedOption = options[1] ?? options[0];

            return {
              choiceId: regionChoice.id,
              playerNumber: regionChoice.playerNumber,
              choiceType: regionChoice.type,
              selectedOption,
            } as PlayerChoiceResponseFor<TChoice>;
          }

          // For all other choices (e.g. capture_direction), just pick the
          // first option if present.
          const anyChoice = choice as any;
          const optionsArray: any[] = (anyChoice.options as any[]) ?? [];
          const selectedOption = optionsArray.length > 0 ? optionsArray[0] : undefined;

          return {
            choiceId: anyChoice.id,
            playerNumber: anyChoice.playerNumber,
            choiceType: anyChoice.type,
            selectedOption,
          } as PlayerChoiceResponseFor<TChoice>;
        },
      };

      const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
      return { engine, handler };
    }

    test('processDisconnectedRegionsForCurrentPlayer honors RegionOrderChoice selection', async () => {
      const { engine } = createEngineWithChoiceHandler();
      const engineAny: any = engine;
      const state: GameState = engineAny.gameState as GameState;

      state.currentPlayer = 1;

      // Two synthetic disconnected regions with distinct positions so we
      // can distinguish them easily.
      const regionA: Territory = {
        spaces: [
          { x: 1, y: 1 },
          { x: 1, y: 2 },
        ],
        controllingPlayer: 0,
        isDisconnected: true,
      };

      const regionB: Territory = {
        spaces: [
          { x: 5, y: 5 },
          { x: 5, y: 6 },
        ],
        controllingPlayer: 0,
        isDisconnected: true,
      };

      const findRegionsSpy = jest
        .spyOn(sandboxTerritory, 'findDisconnectedRegionsOnBoard')
        .mockImplementationOnce(() => [regionA, regionB])
        .mockImplementation(() => []);

      // For this integration test we focus purely on ordering, so stub
      // the self-elimination prerequisite to always return true.
      jest.spyOn(engineAny, 'canProcessDisconnectedRegion').mockReturnValue(true);

      // Spy on processDisconnectedRegionOnBoard to see which region is
      // processed first.
      const processRegionSpy = jest
        .spyOn(sandboxTerritory, 'processDisconnectedRegionOnBoard')
        .mockImplementation((board, players, movingPlayer, regionSpaces) => ({
          board,
          players,
          totalRingsEliminatedDelta: 0,
        }));

      await engineAny.processDisconnectedRegionsForCurrentPlayer();

      expect(findRegionsSpy).toHaveBeenCalled();
      expect(processRegionSpy).toHaveBeenCalled();

      const firstCallArgs = processRegionSpy.mock.calls[0] as [any, any, number, Position[]];
      const firstRegionSpaces = firstCallArgs[3];

      const toKeySet = (spaces: Position[]) => new Set(spaces.map((p) => positionToString(p)));
      const regionBKeys = toKeySet(regionB.spaces);
      const firstRegionKeys = toKeySet(firstRegionSpaces);

      // The first region processed by the sandbox should correspond to
      // regionB, because our handler selected the second option in the
      // RegionOrderChoice.
      expect(firstRegionKeys).toEqual(regionBKeys);
    });
  }
);
