import {
  type BoardType,
  type BoardState,
  type GameState,
  type Move,
  type Player,
  type Position,
  type Territory,
  positionToString,
} from '../../src/shared/types/game';
import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
  enumerateTerritoryEliminationMoves,
} from '../../src/shared/engine/territoryDecisionHelpers';

/**
 * Tiny sandbox regression scenario derived from a live 8×8 AI game where
 * a surrounded disconnected territory was recognised (self-elimination
 * required) but the visual collapse and turn progression in the sandbox
 * UI were confusing.
 *
 * This test:
 * - Reconstructs the sandbox GameState at the start of the
 *   `territory_processing` phase from the provided fixture.
 * - Uses the shared `enumerateProcessTerritoryRegionMoves` helper to
 *   derive the canonical `process_territory_region` Move for the current
 *   player.
 * - Applies that Move via `ClientSandboxEngine.applyCanonicalMove`
 *   (orchestrator-backed path) and asserts that the region’s spaces are:
 *     - Marked as collapsed territory for the moving player,
 *     - Cleared of stacks and markers,
 *     - Reflected in the player’s `territorySpaces` / `eliminatedRings`
 *       counters in a way that matches the shared helper.
 */

function makeFixtureGameState(): GameState {
  const boardType: BoardType = 'square8';

  const rawStacks: Record<string, BoardState['stacks'][string]> = {
    '2,3': {
      position: { x: 2, y: 3 },
      rings: [2, 2, 1, 1, 1],
      stackHeight: 5,
      capHeight: 2,
      controllingPlayer: 2,
    },
    '1,6': {
      position: { x: 1, y: 6 },
      rings: [2, 2, 2, 2, 2],
      stackHeight: 5,
      capHeight: 5,
      controllingPlayer: 2,
    },
    '5,5': {
      position: { x: 5, y: 5 },
      rings: [2, 2, 2],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: 2,
    },
    '5,0': {
      position: { x: 5, y: 0 },
      rings: [2, 2, 2, 2, 2],
      stackHeight: 5,
      capHeight: 5,
      controllingPlayer: 2,
    },
    '0,7': {
      position: { x: 0, y: 7 },
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    },
  };

  const rawMarkers: Record<string, BoardState['markers'][string]> = {
    '7,3': {
      position: { x: 7, y: 3 },
      player: 1,
      type: 'regular',
    },
    '5,6': {
      position: { x: 5, y: 6 },
      player: 2,
      type: 'regular',
    },
    '1,2': {
      position: { x: 1, y: 2 },
      player: 2,
      type: 'regular',
    },
    '4,2': {
      position: { x: 4, y: 2 },
      player: 2,
      type: 'regular',
    },
    '7,0': {
      position: { x: 7, y: 0 },
      player: 2,
      type: 'regular',
    },
    '1,0': {
      position: { x: 1, y: 0 },
      player: 2,
      type: 'regular',
    },
  };

  const rawCollapsed: Record<string, number> = {
    '3,3': 2,
    '4,4': 1,
    '5,4': 1,
    '6,4': 1,
    '7,4': 1,
    '3,4': 1,
    '3,5': 1,
    '3,6': 1,
    '3,7': 1,
  };

  const board: BoardState = {
    type: boardType,
    size: 8,
    stacks: new Map(Object.entries(rawStacks)),
    markers: new Map(Object.entries(rawMarkers)),
    collapsedSpaces: new Map(Object.entries(rawCollapsed)),
    eliminatedRings: { 1: 2 } as any,
    territories: new Map<string, Territory>(),
    formedLines: [],
    pendingCaptureEvaluations: [],
  };

  const players: Player[] = [
    {
      playerNumber: 1,
      ringsInHand: 11,
      eliminatedRings: 2,
      territorySpaces: 8,
      isActive: true,
    } as Player,
    {
      playerNumber: 2,
      ringsInHand: 3,
      eliminatedRings: 0,
      territorySpaces: 0,
      isActive: true,
    } as Player,
  ];

  const state: GameState = {
    id: 'sandbox-local',
    boardType,
    board,
    players,
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    moveHistory: [],
    history: [],
    spectators: [],
    timeControl: { type: 'rapid', initialTime: 600, increment: 0 } as any,
    gameStatus: 'active',
    createdAt: new Date('2025-12-03T08:48:44.586Z'),
    lastMoveAt: new Date('2025-12-03T08:48:44.586Z'),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 36,
    totalRingsEliminated: 2,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
    rngSeed: 193054819,
  };

  return state;
}

describe('ClientSandboxEngine territory decision phases (Move-driven)', () => {
  function createSandboxEngineFromFixture(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType: 'square8',
      numPlayers: 2,
      playerKinds: ['human', 'ai'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice(choice: any): Promise<any> {
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

    (engine as any).gameState = makeFixtureGameState();
    return engine;
  }

  it('applies canonical process_territory_region so region spaces collapse and counts update', async () => {
    const engine = createSandboxEngineFromFixture();
    const before = engine.getGameState();

    expect(before.currentPhase).toBe('territory_processing');
    expect(before.currentPlayer).toBe(1);

    const movingPlayer = before.currentPlayer;

    // Derive the canonical process_territory_region Move from the shared helper.
    const territoryMoves = enumerateProcessTerritoryRegionMoves(before, movingPlayer);
    expect(territoryMoves.length).toBeGreaterThan(0);

    const regionMove = territoryMoves.find(
      (m) => m.type === 'process_territory_region' && m.disconnectedRegions?.length
    ) as Move | undefined;
    expect(regionMove).toBeDefined();

    const region = (regionMove!.disconnectedRegions ?? [])[0];
    expect(region?.spaces.length).toBeGreaterThan(0);

    const regionSpaces = region!.spaces;

    const beforeCollapsedCount = before.board.collapsedSpaces.size;
    const beforeTerritoryP1 = before.players.find((p) => p.playerNumber === 1)!.territorySpaces;
    const beforeElimsP1 = before.players.find((p) => p.playerNumber === 1)!.eliminatedRings;

    // Sanity-check the shared helper on a separate copy of the state so we
    // know the contract for this regionMove is internally consistent.
    const helperState: GameState = makeFixtureGameState();
    const helperOutcome = applyProcessTerritoryRegionDecision(helperState, regionMove!);
    const expected = helperOutcome.nextState;

    // Apply the same canonical Move via the sandbox orchestrator-backed path.
    await (engine as any).applyCanonicalMove(regionMove!);
    const after = engine.getGameState();

    // All spaces in the processed region should now be collapsed territory
    // for the moving player, with no stacks or markers remaining there.
    for (const pos of regionSpaces) {
      const key = positionToString(pos);
      expect(after.board.collapsedSpaces.get(key)).toBe(movingPlayer);
      expect(after.board.stacks.has(key)).toBe(false);
      expect(after.board.markers.has(key)).toBe(false);
    }

    // Collapsed-space count and player-1 territorySpaces must be at least
    // as large as before (monotone with respect to this decision).
    expect(after.board.collapsedSpaces.size).toBeGreaterThanOrEqual(beforeCollapsedCount);
    const afterTerritoryP1 = after.players.find((p) => p.playerNumber === 1)!.territorySpaces;
    expect(afterTerritoryP1).toBeGreaterThanOrEqual(beforeTerritoryP1);

    // If this region triggered a self-elimination requirement in the shared
    // helper, any additional eliminated rings for Player 1 should match.
    const afterElimsP1 = after.players.find((p) => p.playerNumber === 1)!.eliminatedRings;
    const expectedElimsP1 = expected.players.find((p) => p.playerNumber === 1)!.eliminatedRings;
    expect(afterElimsP1).toBeGreaterThanOrEqual(beforeElimsP1);
    expect(afterElimsP1).toBe(expectedElimsP1);
  });

  it('exposes territory_processing decisions via getValidMoves in parity with shared helper', async () => {
    const engine = createSandboxEngineFromFixture();
    const before = engine.getGameState();

    expect(before.currentPhase).toBe('territory_processing');
    const movingPlayer = before.currentPlayer;
    expect(movingPlayer).toBe(1);

    // Shared helper enumeration for the current player.
    const helperMoves = enumerateProcessTerritoryRegionMoves(before, movingPlayer);
    expect(helperMoves.length).toBeGreaterThan(0);

    // Host-level getValidMoves surface from the sandbox engine.
    const hostAllMoves = engine.getValidMoves(movingPlayer);
    const hostRegionMoves = hostAllMoves.filter((m) => m.type === 'process_territory_region');

    // In territory_processing phase, valid moves are process_territory_region,
    // skip_territory_processing, territory_elimination, and eliminate_rings_from_stack
    // (for self-elimination). We expect at least one region move.
    expect(hostRegionMoves.length).toBeGreaterThan(0);
    const validTerritoryMoveTypes = [
      'process_territory_region',
      'skip_territory_processing',
      'territory_elimination',
      'eliminate_rings_from_stack', // Self-elimination moves
    ];
    const unexpectedMoves = hostAllMoves.filter((m) => !validTerritoryMoveTypes.includes(m.type));
    expect(unexpectedMoves.map((m) => m.type)).toEqual([]);

    // Normalise moves by player + region geometry to compare helper vs host.
    const keyFromMove = (m: Move): string => {
      const region = (m.disconnectedRegions ?? [])[0];
      const parts = (region?.spaces ?? []).map((p) => positionToString(p)).sort();
      return `${m.player}:${parts.join('|')}`;
    };

    const helperKeys = helperMoves.map(keyFromMove).sort();
    const hostKeys = hostRegionMoves.map(keyFromMove).sort();

    expect(hostKeys).toEqual(helperKeys);
  });

  it('auto-applies mandatory self-elimination after region processing in traceMode (internal handling)', async () => {
    // The sandbox engine in traceMode handles self-elimination internally
    // rather than surfacing it through getValidMoves. This test verifies
    // that the elimination is correctly applied by checking ring counts.
    const engine = createSandboxEngineFromFixture();
    const base = engine.getGameState();

    expect(base.currentPhase).toBe('territory_processing');
    const movingPlayer = base.currentPlayer;
    expect(movingPlayer).toBe(1);

    // Use the shared helper to derive the canonical region-processing move.
    const helperRegionMoves = enumerateProcessTerritoryRegionMoves(base, movingPlayer);
    expect(helperRegionMoves.length).toBeGreaterThan(0);
    const regionMove = helperRegionMoves[0];

    // Apply territory processing via the shared helper on a cloned state
    // to determine the expected self-elimination requirement.
    const helperState = makeFixtureGameState();
    const helperOutcome = applyProcessTerritoryRegionDecision(helperState, regionMove);
    expect(helperOutcome.pendingSelfElimination).toBe(true);

    const beforeElimsP1 = base.players.find(
      (p) => p.playerNumber === movingPlayer
    )!.eliminatedRings;

    // Apply the region move through the sandbox engine. In traceMode,
    // the engine handles elimination internally rather than exposing it
    // through getValidMoves.
    await (engine as any).applyCanonicalMove(regionMove);
    const after = engine.getGameState();

    // Verify that the region spaces were collapsed.
    const regionSpaces = (regionMove.disconnectedRegions ?? [])[0]?.spaces ?? [];
    for (const pos of regionSpaces) {
      const key = positionToString(pos);
      expect(after.board.collapsedSpaces.get(key)).toBe(movingPlayer);
    }

    // Verify that elimination was applied (ring count increased).
    const afterElimsP1 = after.players.find(
      (p) => p.playerNumber === movingPlayer
    )!.eliminatedRings;
    expect(afterElimsP1).toBeGreaterThan(beforeElimsP1);
  });
});
