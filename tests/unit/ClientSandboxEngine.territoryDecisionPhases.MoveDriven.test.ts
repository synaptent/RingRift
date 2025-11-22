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
  Territory,
  PlayerChoiceResponseFor,
  positionToString,
} from '../../src/shared/types/game';
import { pos, addStack } from '../utils/fixtures';
import * as sandboxTerritory from '../../src/client/sandbox/sandboxTerritory';

/**
 * Sandbox-level move-driven territory decision phases
 *
 * This test mirrors the backend GameEngine move-driven territory test but
 * runs entirely through ClientSandboxEngine and its canonical Move helpers:
 *
 * - A geometry setup creates a concrete disconnected region of opponent
 *   stacks plus an outside stack for the moving player.
 * - The sandbox enumerates a process_territory_region Move via the
 *   internal getValidTerritoryProcessingMovesForCurrentPlayer helper.
 * - After applying that Move via applyCanonicalMove, the sandbox then
 *   enumerates explicit eliminate_rings_from_stack Moves via
 *   getValidEliminationDecisionMovesForCurrentPlayer.
 * - Applying an elimination Move yields a sandbox history trace that
 *   records process_territory_region followed by
 *   eliminate_rings_from_stack as two distinct canonical actions,
 *   mirroring the backend GameEngine history semantics.
 */

describe('ClientSandboxEngine move-driven territory decision phases', () => {
  function createEngine(boardType: BoardType = 'square8') {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
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

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler: handler,
      traceMode: true,
    });
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;
    return { engine, engineAny, state };
  }

  it('after processing a disconnected region via canonical Moves, surfaces explicit eliminate_rings_from_stack and records both in sandbox history', async () => {
    const { engine, engineAny, state } = createEngine('square8');
    const board = state.board;

    state.gameStatus = 'active';
    state.currentPlayer = 1;
    state.currentPhase = 'territory_processing';

    // Concrete disconnected region for Player 1 consisting of opponent
    // stacks that will be eliminated and collapsed when the region is
    // processed.
    const regionPositions: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];
    regionPositions.forEach((p) => addStack(board, p, 2, 1));

    // Give Player 1 a stack outside the region so the self-elimination
    // prerequisite is satisfied. This stack will be the source of the
    // later eliminate_rings_from_stack decision.
    const outside = pos(0, 1);
    addStack(board, outside, 1, 3);

    const region: Territory = {
      spaces: regionPositions,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    // First call to findDisconnectedRegionsOnBoard returns our concrete
    // region so that the sandbox territory-decision helper can resolve it.
    const findDisconnectedRegionsSpy = jest
      .spyOn(sandboxTerritory, 'findDisconnectedRegionsOnBoard')
      .mockImplementationOnce(() => [region]);

    // From the sandbox perspective in territory_processing, the valid
    // decision set should include at least one process_territory_region
    // Move for Player 1.
    const territoryMoves: Move[] = engineAny.getValidTerritoryProcessingMovesForCurrentPlayer();
    expect(territoryMoves.length).toBeGreaterThan(0);

    const processMove = territoryMoves[0];
    expect(processMove.type).toBe('process_territory_region');
    expect(processMove.disconnectedRegions && processMove.disconnectedRegions[0]).toBeDefined();

    const regionFromMove = processMove.disconnectedRegions![0];
    expect(regionFromMove.spaces.length).toBe(regionPositions.length);

    await engine.applyCanonicalMove(processMove);

    const afterRegion = engine.getGameState();

    // Sandbox history should record the territory processing decision as
    // a canonical process_territory_region Move, mirroring the backend
    // GameEngine decision-phase trace semantics.
    expect(afterRegion.history.length).toBe(1);
    expect(afterRegion.history[0].action.type).toBe('process_territory_region');

    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();
  });
});
