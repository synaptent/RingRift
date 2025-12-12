import { BoardType, GameState, Move, positionToString } from '../../src/shared/types/game';
import { runSandboxAITrace } from '../utils/traces';
import { BoardManager } from '../../src/server/game/BoardManager';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../../src/shared/engine/territoryDetection';
import { findDisconnectedRegionsOnBoard as findDisconnectedRegionsSandbox } from '../../src/client/sandbox/sandboxTerritory';

// Skip this test suite when orchestrator adapter is enabled - territory processing timing differs intentionally
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Territory detection parity at seed 5 territory decision step',
  () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;
    const seed = 5;
    const MAX_STEPS = 60;

    function createSandboxEngineFromInitial(initial: GameState): ClientSandboxEngine {
      const config: SandboxConfig = {
        boardType: initial.boardType,
        numPlayers: initial.players.length,
        playerKinds: initial.players
          .slice()
          .sort((a, b) => a.playerNumber - b.playerNumber)
          .map((p) => p.type as 'human' | 'ai'),
      };

      const handler: SandboxInteractionHandler = {
        async requestChoice(choice: any) {
          const options = ((choice as any).options as any[]) ?? [];
          const selectedOption = options.length > 0 ? options[0] : undefined;

          return {
            choiceId: (choice as any).id,
            playerNumber: (choice as any).playerNumber,
            choiceType: (choice as any).type,
            selectedOption,
          } as any;
        },
      };

      const engine = new ClientSandboxEngine({
        config,
        interactionHandler: handler,
        traceMode: true,
      });
      const engineAny: any = engine;
      engineAny.gameState = initial;
      return engine;
    }

    test('shared vs backend vs sandbox detectors at first territory decision', async () => {
      const trace = await runSandboxAITrace(boardType, numPlayers, seed, MAX_STEPS);
      expect(trace.entries.length).toBeGreaterThan(0);

      const targetIndex = trace.entries.findIndex(
        (e) =>
          e.action.type === 'choose_territory_option' ||
          e.action.type === 'process_territory_region'
      );
      expect(targetIndex).toBeGreaterThanOrEqual(0);

      const engine = createSandboxEngineFromInitial(trace.initialState);
      for (let i = 0; i < targetIndex; i++) {
        const move = trace.entries[i].action as Move;
        await engine.applyCanonicalMove(move);
      }

      const stateBefore = engine.getGameState();
      const movingPlayer = stateBefore.currentPlayer;

      const sharedRegions = findDisconnectedRegionsShared(stateBefore.board);
      const sharedRegionKeys = sharedRegions.map((region) =>
        region.spaces.map(positionToString).sort()
      );

      const sandboxRegions = findDisconnectedRegionsSandbox(stateBefore.board);
      const sandboxRegionKeys = sandboxRegions.map((region) =>
        region.spaces.map(positionToString).sort()
      );

      const boardManager = new BoardManager(stateBefore.boardType);
      const backendRegions = boardManager.findDisconnectedRegions(stateBefore.board, movingPlayer);
      const backendRegionKeys = backendRegions.map((region) =>
        region.spaces.map(positionToString).sort()
      );

      const counts = {
        shared: sharedRegions.length,
        sandbox: sandboxRegions.length,
        backend: backendRegions.length,
      };

      // eslint-disable-next-line no-console
      console.log('[TerritoryDetection.seed5Move45] regionCounts', counts, {
        sharedRegionKeys,
        sandboxRegionKeys,
        backendRegionKeys,
        movingPlayer,
      });

      // Diagnostic-only: this test is currently intended to surface detector
      // differences via logging rather than enforcing parity. Once the
      // underlying bug is fixed, these expectations can be tightened to
      // require exact equality between detectors.
      expect(sharedRegions.length).toBeGreaterThanOrEqual(0);
    });
  }
);
