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
  LineInfo,
  RingStack,
  positionToString,
} from '../../src/shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine';
import * as sandboxTerritory from '../../src/client/sandbox/sandboxTerritory';
import * as territoryBorders from '../../src/shared/engine/territoryBorders';
import * as sharedLineDetection from '../../src/shared/engine/lineDetection';

/**
 * Hex-board variant of the combined line + territory post-processing test
 * for the sandbox engine.
 *
 * This mirrors GameEngine.territoryDisconnection.hex.test.ts but uses the
 * ClientSandboxEngine + sandboxTerritory helpers instead of the backend
 * GameEngine/BoardManager. We stub hex territory detection and line
 * enumeration so the test focuses on post-processing semantics:
 *
 * - Line collapses via processLinesForCurrentPlayer.
 * - Territory collapses via processDisconnectedRegionsForCurrentPlayer.
 * - Internal eliminations credited to the moving player.
 */

// Classification: legacy hex combined line+territory sandbox integration; semantics now
// primarily covered by shared helpers and square-board territory/line parity tests.
// Skip when orchestrator adapter is enabled - this test stubs sandboxLines/sandboxTerritory
// helpers which are bypassed by the orchestrator adapter's unified processing via
// shared engine helpers (lineDecisionHelpers, TerritoryAggregate).
const skipWithOrchestrator = process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true';

(skipWithOrchestrator ? describe.skip : describe)('ClientSandboxEngine territory + line processing (hexagonal)', () => {
  const boardType: BoardType = 'hexagonal';

  function createEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        // In this test, choices are irrelevant; always pick the first option.
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

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('line + territory consequences combine correctly on hex board (stubbed detection)', async () => {
    const engine = createEngine();
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;
    const board = state.board;

    state.currentPlayer = 1;

    // --- 1. Stub a small disconnected region for territory collapse ---
    const regionSpaces: Position[] = [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: -1, z: 0 },
      { x: 0, y: -1, z: 1 },
    ];

    // Place B stacks (player 2) in the region so internal eliminations occur.
    for (const p of regionSpaces) {
      const key = positionToString(p);
      const stack: RingStack = {
        position: p,
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      board.stacks.set(key, stack);
    }

    // Stub border markers returned by getBorderMarkerPositionsForRegion; we
    // don't rely on true geometry here, only on the fact that these spaces
    // are collapsed as part of territory processing.
    const borderPositions: Position[] = [
      { x: 2, y: -2, z: 0 },
      { x: -1, y: 1, z: 0 },
    ];

    // Stub border-marker computation at the canonical shared helper used by the
    // territory-processing pipeline so that our synthetic borderPositions are
    // treated as the marker border for this region. The sandboxTerritory
    // adapter's getBorderMarkerPositionsForRegion wrapper is no longer in the
    // active path for ClientSandboxEngine, so we target the shared territory
    // border helper directly.
    jest
      .spyOn(territoryBorders, 'getBorderMarkerPositionsForRegion')
      .mockImplementation(
        (_board: any, _regionSpaces: Position[], _opts?: any): Position[] => borderPositions
      );

    const territoryRegion: Territory = {
      spaces: regionSpaces,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    jest
      .spyOn(sandboxTerritory, 'findDisconnectedRegionsOnBoard')
      .mockImplementationOnce(() => [territoryRegion])
      .mockImplementation(() => []);

    // --- 2. Stub a hex line for player 1 ---
    const linePositions: Position[] = [
      { x: -2, y: 2, z: 0 },
      { x: -1, y: 1, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 1, y: -1, z: 0 },
      { x: 2, y: -2, z: 0 },
    ];

    const lineInfo: LineInfo = {
      positions: linePositions,
      player: 1,
      length: linePositions.length,
      direction: { x: 1, y: -1, z: 0 },
    };
 
    // Stub canonical line detection so that the shared lineDecisionHelpers
    // see our synthetic hex line for Player 1 regardless of marker layout.
    jest
      .spyOn(sharedLineDetection, 'findAllLines')
      .mockImplementation(() => [lineInfo]);

    // --- 3. Provide P1 stacks: one for line elimination and one for territory self-elim ---
    const lineStackPos: Position = { x: 3, y: -3, z: 0 };
    const territoryStackPos: Position = { x: -3, y: 3, z: 0 };

    const makeP1Stack = (pos: Position) => {
      const key = positionToString(pos);
      const stack: RingStack = {
        position: pos,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      };
      board.stacks.set(key, stack);
    };

    // Insert line stack first so default elimination logic picks it first.
    makeP1Stack(lineStackPos);
    makeP1Stack(territoryStackPos);

    // Sanity: no collapsed spaces and no eliminated rings yet.
    expect(board.collapsedSpaces.size).toBe(0);
    expect(state.board.eliminatedRings[1] || 0).toBe(0);
    expect(state.players[0].eliminatedRings).toBe(0);
    expect(state.totalRingsEliminated).toBe(0);

    // --- 4. Run combined post-move processing on hex board (no capture) ---
    engineAny.processLinesForCurrentPlayer();
    await engineAny.processDisconnectedRegionsForCurrentPlayer();

    const keysFrom = (positions: Position[]) => new Set(positions.map((p) => positionToString(p)));

    const interiorKeys = keysFrom(regionSpaces);
    const borderKeys = keysFrom(borderPositions);

    // 1. All interior region spaces collapsed for P1 and empty of stacks.
    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    for (const p of regionSpaces) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.stacks.get(key)).toBeUndefined();
    }

    // 2. All border positions collapsed for P1.
    for (const p of borderPositions) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. Line processing: at least the minimum-length subset of the line
    //    positions collapsed for P1. On hex boards the canonical threshold
    //    is given by getEffectiveLineLengthThreshold; the sandbox automatic
    //    pipeline defaults to the minimum-collapse reward (Option 2) for
    //    overlength lines, so we assert that at least that many cells from
    //    our synthetic line are collapsed for Player 1.
    const collapsedLinePositions = linePositions.filter((p) => {
      const key = positionToString(p);
      return finalBoard.collapsedSpaces.get(key) === 1;
    });
    const requiredLineLength = getEffectiveLineLengthThreshold(
      boardType,
      finalState.players.length,
      finalState.rulesOptions
    );
    expect(collapsedLinePositions.length).toBeGreaterThanOrEqual(requiredLineLength);

    // 4. All stacks inside the region should be eliminated.
    const stacksInRegion = Array.from(finalBoard.stacks.keys()).filter((k) => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    // 5. Territory accounting: P1's territorySpaces should match the
    //    number of collapsed spaces they own in this constructed
    //    scenario (union of region, border, and the actually-collapsed
    //    subset of line positions).
    const collapsedLineKeys = keysFrom(collapsedLinePositions);
    const allKeys = new Set<string>([
      ...Array.from(interiorKeys),
      ...Array.from(borderKeys),
      ...Array.from(collapsedLineKeys),
    ]);
    const collapsedForP1 = Array.from(finalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    // On hex boards, line and territory processing may collapse additional
    // markers beyond our synthetic region/line set (e.g., incidental
    // neighbours). Assert that P1 controls at least the union of our
    // constructed positions, and that their territorySpaces reflects at
    // least this union, even if more collapsed spaces exist.
    expect(collapsedForP1).toBeGreaterThanOrEqual(allKeys.size);
    expect(player1.territorySpaces).toBeGreaterThanOrEqual(allKeys.size);

    // 6. Eliminated ring counts should at least include the internal
    //    region stacks for B (3 rings). Depending on how additional
    //    line/territory eliminations are wired for hex boards, more
    //    rings may be attributed, but never fewer than the internal
    //    region eliminations.
    const minEliminatedForP1 = 3;
    expect(finalBoard.eliminatedRings[1]).toBeGreaterThanOrEqual(minEliminatedForP1);
    expect(player1.eliminatedRings).toBeGreaterThanOrEqual(minEliminatedForP1);
    expect(finalState.totalRingsEliminated).toBeGreaterThanOrEqual(minEliminatedForP1);
  });
});
