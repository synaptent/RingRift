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
import { pos, addStack, addMarker } from '../utils/fixtures';

/**
 * Sandbox-level move-driven territory decision phases
 *
 * This test mirrors the backend GameEngine move-driven territory test but
 * runs entirely through ClientSandboxEngine and its canonical Move helpers.
 * Geometry mirrors the RulesMatrix Q23 scenarios:
 * - Rules_12_2_Q23_region_not_processed_without_self_elimination_square19
 * - Rules_12_2_Q23_region_processed_with_self_elimination_square19
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

    // For move-driven territory decision tests we want to exercise the
    // sandbox's legacy decision helpers directly rather than routing
    // through the orchestrator adapter. This keeps the test focused on
    // ClientSandboxEngine's own gating and history behaviour while still
    // relying on the shared territoryDecisionHelpers for semantics.
    engine.disableOrchestratorAdapter();

    return { engine, engineAny, state };
  }

  it('after processing a disconnected region via canonical Moves, surfaces explicit eliminate_rings_from_stack and records both in sandbox history', async () => {
    const { engine, engineAny, state } = createEngine('square8');
    const board = state.board;

    state.gameStatus = 'active';
    state.currentPlayer = 1;
    state.currentPhase = 'territory_processing';

    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Concrete disconnected region for Player 1 consisting of opponent
    // stacks that will be eliminated and collapsed when the region is
    // processed. Geometry mirrors the Q23-style mini region used in
    // territoryProcessing.shared tests so that shared detection +
    // gating semantics are exercised end-to-end.
    const regionPositions: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    const internalStackHeight = 2;
    regionPositions.forEach((p) => addStack(board, p, 2, internalStackHeight));

    // Border markers for the moving player enclosing the region. These
    // markers ensure the shared territory detector identifies exactly
    // one disconnected region matching regionPositions.
    const borderCoords: Array<[number, number]> = [];
    for (let x = 1; x <= 4; x++) {
      borderCoords.push([x, 1]);
      borderCoords.push([x, 4]);
    }
    for (let y = 2; y <= 3; y++) {
      borderCoords.push([1, y]);
      borderCoords.push([4, y]);
    }
    borderCoords.forEach(([x, y]) => addMarker(board, pos(x, y), 1));

    // Give Player 1 a stack outside the region so the self-elimination
    // prerequisite is satisfied. This stack will be the source of the
    // later eliminate_rings_from_stack decision.
    const outside = pos(0, 0);
    addStack(board, outside, 1, 3);

    const p1Before = state.players.find((p) => p.playerNumber === 1)!;
    const eliminatedBefore = p1Before.eliminatedRings;
    const totalEliminatedBefore = state.totalRingsEliminated;
    const collapsedBefore = board.collapsedSpaces.size;

    // From the sandbox perspective in territory_processing, the valid
    // decision set should include at least one process_territory_region
    // Move for Player 1, driven by the shared
    // enumerateProcessTerritoryRegionMoves + canProcessTerritoryRegion
    // helpers under the hood.
    const territoryMoves: Move[] = engineAny.getValidTerritoryProcessingMovesForCurrentPlayer();
    expect(territoryMoves.length).toBeGreaterThan(0);

    const processMove = territoryMoves[0];
    expect(processMove.type).toBe('process_territory_region');
    expect(processMove.disconnectedRegions && processMove.disconnectedRegions[0]).toBeDefined();

    const regionFromMove = processMove.disconnectedRegions![0];
    expect(regionFromMove.spaces.length).toBe(regionPositions.length);

    await engine.applyCanonicalMove(processMove);

    const afterRegion = engine.getGameState();

    // Territory processing must increase S-invariant components
    // monotonically: collapsedSpaces and eliminated rings for the moving
    // player / total.
    expect(afterRegion.board.collapsedSpaces.size).toBeGreaterThanOrEqual(collapsedBefore);
    const p1AfterRegion = afterRegion.players.find((p) => p.playerNumber === 1)!;
    expect(p1AfterRegion.eliminatedRings).toBeGreaterThanOrEqual(eliminatedBefore);
    expect(afterRegion.totalRingsEliminated).toBeGreaterThanOrEqual(totalEliminatedBefore);

    // Sandbox history should record the territory processing decision as
    // a canonical process_territory_region Move, mirroring the backend
    // GameEngine decision-phase trace semantics.
    expect(afterRegion.history.length).toBe(1);
    expect(afterRegion.history[0].action.type).toBe('process_territory_region');

    // After processing a real disconnected region in trace-mode
    // territory_processing, the sandbox should surface explicit
    // eliminate_rings_from_stack decision Moves and defer self-
    // elimination until such a Move is applied.
    const eliminationMoves: Move[] = engineAny.getValidEliminationDecisionMovesForCurrentPlayer();
    expect(eliminationMoves.length).toBeGreaterThan(0);

    const outsideKey = positionToString(outside);
    const eliminationMove =
      eliminationMoves.find((m) => m.to && positionToString(m.to) === outsideKey) ??
      eliminationMoves[0];

    expect(eliminationMove.type).toBe('eliminate_rings_from_stack');
    expect(eliminationMove.to).toBeDefined();
    expect(eliminationMove.eliminationFromStack).toBeDefined();

    const capHeight = eliminationMove.eliminationFromStack?.capHeight ?? 0;
    expect(capHeight).toBeGreaterThan(0);

    await engine.applyCanonicalMove(eliminationMove);

    const afterElimination = engine.getGameState();

    // The outside stack should have been reduced by the cap height; in
    // this simple all-one-colour example the stack is fully removed.
    expect(afterElimination.board.stacks.has(outsideKey)).toBe(false);

    const finalP1 = afterElimination.players.find((p) => p.playerNumber === 1)!;
    expect(finalP1.eliminatedRings).toBeGreaterThanOrEqual(p1AfterRegion.eliminatedRings);
    expect(afterElimination.totalRingsEliminated).toBeGreaterThanOrEqual(
      afterRegion.totalRingsEliminated
    );

    // History should record the territory region processing and the
    // explicit self-elimination as two distinct canonical Moves.
    expect(afterElimination.history.length).toBe(2);
    expect(afterElimination.history[0].action.type).toBe('process_territory_region');
    expect(afterElimination.history[1].action.type).toBe('eliminate_rings_from_stack');
  });
});
