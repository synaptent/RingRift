import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { BoardType, GameState, Player, Position, TimeControl } from '../../src/shared/types/game';
import { summarizeBoard, computeProgressSnapshot } from '../../src/shared/engine/core';
import { addMarker, addStack, addCollapsedSpace, pos } from '../utils/fixtures';
import * as sandboxTerritory from '../../src/client/sandbox/sandboxTerritory';

/**
 * Territory parity harness: backend GameEngine vs ClientSandboxEngine.
 *
 * These tests focus on FAQ Q23-style disconnected-region scenarios:
 * - Positive cases where the moving player DOES satisfy the self-elimination
 *   prerequisite (at least one stack outside each disconnected region).
 * - Negative cases where the moving player has no stack outside the region
 *   and both engines must refuse to process it.
 * - A multi-region case where two disconnected regions are processed in
 *   sequence and both engines agree on the final territory + elimination
 *   accounting.
 */

describe('Territory parity: GameEngine vs ClientSandboxEngine (Q23 scenarios)', () => {
  const boardType: BoardType = 'square19';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createBackendPlayers(): Player[] {
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

  function createSandboxEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      // Default handler: always pick the first option for any choice.
      async requestChoice<TChoice>(choice: TChoice): Promise<any> {
        const optionsArray = ((choice as any).options as any[]) ?? [];
        const selectedOption = optionsArray.length > 0 ? optionsArray[0] : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        };
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  const keyFrom = (p: Position): string =>
    p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;

  test('Q23_disconnected_region_processed_when_self_elimination_available_parity', async () => {
    // Positive Q23 parity: both engines should process the region when
    // the moving player has at least one stack outside it.

    // --- Backend setup ---
    const backendPlayers = createBackendPlayers();
    const backendEngine = new GameEngine(
      'territory-q23-parity',
      boardType,
      backendPlayers,
      timeControl,
      false
    );
    const backendAny: any = backendEngine;
    const backendState: GameState = backendAny.gameState as GameState;
    const backendBoard = backendState.board;

    backendState.currentPlayer = 1;

    // --- Sandbox setup ---
    const sandboxEngine = createSandboxEngine();
    const sandboxAny: any = sandboxEngine;
    const sandboxState: GameState = sandboxAny.gameState as GameState;
    const sandboxBoard = sandboxState.board;

    sandboxState.currentPlayer = 1;

    // --- Shared geometry: canonical 3x3 interior region + border markers ---
    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        // Player 2 stacks inside the region on both boards.
        addStack(backendBoard, p, 2, 1);
        addStack(sandboxBoard, p, 2, 1);
      }
    }

    const borderCoords: Position[] = [];
    for (let x = 4; x <= 8; x++) {
      borderCoords.push(pos(x, 4));
      borderCoords.push(pos(x, 8));
    }
    for (let y = 5; y <= 7; y++) {
      borderCoords.push(pos(4, y));
      borderCoords.push(pos(8, y));
    }

    for (const p of borderCoords) {
      addMarker(backendBoard, p, 1);
      addMarker(sandboxBoard, p, 1);
    }

    // Player 1 stack outside the region (self-elimination prerequisite).
    const p1Outside = pos(1, 1);
    addStack(backendBoard, p1Outside, 1, 2);
    addStack(sandboxBoard, p1Outside, 1, 2);

    // Player 3 active elsewhere but not inside the region.
    const p3Outside = pos(0, 0);
    addStack(backendBoard, p3Outside, 3, 1);
    addStack(sandboxBoard, p3Outside, 3, 1);

    // Sanity: no collapsed spaces initially.
    expect(backendBoard.collapsedSpaces.size).toBe(0);
    expect(sandboxBoard.collapsedSpaces.size).toBe(0);

    const backendInitialTotalEliminated = backendState.totalRingsEliminated;
    const backendInitialP1 = backendState.players.find((p) => p.playerNumber === 1)!;
    const backendInitialP1Eliminated = backendInitialP1.eliminatedRings;

    const sandboxInitialTotalEliminated = sandboxState.totalRingsEliminated;
    const sandboxInitialP1 = sandboxState.players.find((p) => p.playerNumber === 1)!;
    const sandboxInitialP1Eliminated = sandboxInitialP1.eliminatedRings;

    // --- Run territory processing on both engines ---
    await backendAny.processDisconnectedRegions();
    await sandboxAny.processDisconnectedRegionsForCurrentPlayer();

    const backendFinalState = backendState;
    const backendFinalBoard = backendFinalState.board;
    const backendP1 = backendFinalState.players.find((p) => p.playerNumber === 1)!;

    const sandboxFinalState = sandboxEngine.getGameState();
    const sandboxFinalBoard = sandboxFinalState.board;
    const sandboxP1Final = sandboxFinalState.players.find((p) => p.playerNumber === 1)!;

    const interiorKeys = new Set(interiorCoords.map(keyFrom));

    // 1. Interior region spaces are collapsed for player 1 and empty of stacks on both engines.
    for (const p of interiorCoords) {
      const key = keyFrom(p);
      expect(backendFinalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(sandboxFinalBoard.collapsedSpaces.get(key)).toBe(1);

      expect(backendFinalBoard.stacks.get(key)).toBeUndefined();
      expect(sandboxFinalBoard.stacks.get(key)).toBeUndefined();
    }

    // 2. Border marker positions are collapsed for player 1 on both engines.
    for (const p of borderCoords) {
      const key = keyFrom(p);
      expect(backendFinalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(sandboxFinalBoard.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. Territory counts for player 1 match.
    const backendCollapsedForP1 = Array.from(backendFinalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    const sandboxCollapsedForP1 = Array.from(sandboxFinalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;

    expect(backendCollapsedForP1).toBe(sandboxCollapsedForP1);
    expect(backendP1.territorySpaces).toBe(sandboxP1Final.territorySpaces);

    // 4. All stacks inside the region are eliminated on both engines.
    const backendStacksInRegion = Array.from(backendFinalBoard.stacks.keys()).filter((k) =>
      interiorKeys.has(k)
    );
    const sandboxStacksInRegion = Array.from(sandboxFinalBoard.stacks.keys()).filter((k) =>
      interiorKeys.has(k)
    );

    expect(backendStacksInRegion.length).toBe(0);
    expect(sandboxStacksInRegion.length).toBe(0);

    // 5. Eliminated ring accounting matches across backend and sandbox.
    const backendFinalP1Eliminated = backendP1.eliminatedRings;
    const sandboxFinalP1Eliminated = sandboxP1Final.eliminatedRings;

    expect(backendFinalP1Eliminated - backendInitialP1Eliminated).toBe(
      sandboxFinalP1Eliminated - sandboxInitialP1Eliminated
    );

    const backendFinalTotalEliminated = backendFinalState.totalRingsEliminated;
    const sandboxFinalTotalEliminated = sandboxFinalState.totalRingsEliminated;

    expect(backendFinalTotalEliminated - backendInitialTotalEliminated).toBe(
      sandboxFinalTotalEliminated - sandboxInitialTotalEliminated
    );

    // Also ensure the board-level eliminatedRings bookkeeping matches for player 1.
    const backendBoardP1Elims = backendFinalBoard.eliminatedRings[1] || 0;
    const sandboxBoardP1Elims = sandboxFinalBoard.eliminatedRings[1] || 0;

    expect(backendBoardP1Elims).toBe(sandboxBoardP1Elims);
  });

  test('Q23_disconnected_region_illegal_when_no_self_elimination_available_parity', async () => {
    // Negative Q23 parity: both engines must refuse to process a disconnected
    // region when the moving player has no stacks outside that region.

    // --- Backend setup ---
    const backendPlayers = createBackendPlayers();
    const backendEngine = new GameEngine(
      'territory-q23-negative-parity',
      boardType,
      backendPlayers,
      timeControl,
      false
    );
    const backendAny: any = backendEngine;
    const backendState: GameState = backendAny.gameState as GameState;
    const backendBoard = backendState.board;
    const backendBoardManager: any = backendAny.boardManager;

    backendState.currentPlayer = 1;

    // --- Sandbox setup ---
    const sandboxEngine = createSandboxEngine();
    const sandboxAny: any = sandboxEngine;
    const sandboxState: GameState = sandboxAny.gameState as GameState;
    const sandboxBoard = sandboxState.board;

    sandboxState.currentPlayer = 1;

    // Synthetic 3x3 region as in other Q23 tests.
    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        // Place stacks for player 1 *inside* the region only on both boards.
        addStack(backendBoard, p, 1, 1);
        addStack(sandboxBoard, p, 1, 1);
      }
    }

    // Player 2 active elsewhere but not inside the region.
    const p2Outside = pos(0, 0);
    addStack(backendBoard, p2Outside, 2, 1);
    addStack(sandboxBoard, p2Outside, 2, 1);

    // Confirm player 1 has no stacks outside the region on either board.
    const isInRegion = (p: Position) => interiorCoords.some((q) => keyFrom(q) === keyFrom(p));

    const backendP1Stacks = Array.from(backendBoard.stacks.values()).filter(
      (s) => s.controllingPlayer === 1
    );
    const backendOutside = backendP1Stacks.filter((s) => !isInRegion(s.position));
    expect(backendOutside.length).toBe(0);

    const sandboxP1Stacks = Array.from(sandboxBoard.stacks.values()).filter(
      (s) => s.controllingPlayer === 1
    );
    const sandboxOutside = sandboxP1Stacks.filter((s) => !isInRegion(s.position));
    expect(sandboxOutside.length).toBe(0);

    const regionTerritory = {
      spaces: interiorCoords,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    // Stub disconnected-region detection on both engines so we focus purely
    // on the Q23 self-elimination prerequisite.
    const backendFindDisconnectedRegionsSpy = jest
      .spyOn(backendBoardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    const sandboxFindDisconnectedRegionsSpy = jest
      .spyOn(sandboxTerritory, 'findDisconnectedRegionsOnBoard')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    const backendInitialCollapsed = backendBoard.collapsedSpaces.size;
    const backendInitialTotalEliminated = backendState.totalRingsEliminated;
    const backendInitialP1Eliminated = backendState.players.find(
      (p) => p.playerNumber === 1
    )!.eliminatedRings;

    const sandboxInitialCollapsed = sandboxBoard.collapsedSpaces.size;
    const sandboxInitialTotalEliminated = sandboxState.totalRingsEliminated;
    const sandboxInitialP1Eliminated = sandboxState.players.find(
      (p) => p.playerNumber === 1
    )!.eliminatedRings;

    await backendAny.processDisconnectedRegions();
    await sandboxAny.processDisconnectedRegionsForCurrentPlayer();

    expect(backendFindDisconnectedRegionsSpy).toHaveBeenCalled();
    expect(sandboxFindDisconnectedRegionsSpy).toHaveBeenCalled();

    // Both engines must refuse to process the region:
    // - No new collapsed spaces.
    // - All interior stacks remain.
    // - No eliminations credited to player 1.
    expect(backendBoard.collapsedSpaces.size).toBe(backendInitialCollapsed);
    expect(sandboxBoard.collapsedSpaces.size).toBe(sandboxInitialCollapsed);

    const backendStacksInRegion = Array.from(backendBoard.stacks.keys()).filter((key) =>
      interiorCoords.some((p) => keyFrom(p) === key)
    );
    const sandboxStacksInRegion = Array.from(sandboxBoard.stacks.keys()).filter((key) =>
      interiorCoords.some((p) => keyFrom(p) === key)
    );

    expect(backendStacksInRegion.length).toBe(interiorCoords.length);
    expect(sandboxStacksInRegion.length).toBe(interiorCoords.length);

    const backendFinalTotalEliminated = backendState.totalRingsEliminated;
    const backendFinalP1Eliminated = backendState.players.find(
      (p) => p.playerNumber === 1
    )!.eliminatedRings;

    const sandboxFinalTotalEliminated = sandboxState.totalRingsEliminated;
    const sandboxFinalP1Eliminated = sandboxState.players.find(
      (p) => p.playerNumber === 1
    )!.eliminatedRings;

    expect(backendFinalTotalEliminated).toBe(backendInitialTotalEliminated);
    expect(backendFinalP1Eliminated).toBe(backendInitialP1Eliminated);

    expect(sandboxFinalTotalEliminated).toBe(sandboxInitialTotalEliminated);
    expect(sandboxFinalP1Eliminated).toBe(sandboxInitialP1Eliminated);
  });

  test('multi_region_disconnected_chain_reactions_parity_two_regions', async () => {
    // Multi-region Q23-positive parity: both engines process two disconnected
    // regions in sequence and agree on final territory and eliminated rings.

    // --- Backend setup ---
    const backendPlayers = createBackendPlayers();
    const backendEngine = new GameEngine(
      'territory-q23-multi-parity',
      boardType,
      backendPlayers,
      timeControl,
      false
    );
    const backendAny: any = backendEngine;
    const backendState: GameState = backendAny.gameState as GameState;
    const backendBoard = backendState.board;
    const backendBoardManager: any = backendAny.boardManager;

    backendState.currentPlayer = 1;

    // --- Sandbox setup ---
    const sandboxEngine = createSandboxEngine();
    const sandboxAny: any = sandboxEngine;
    const sandboxState: GameState = sandboxAny.gameState as GameState;
    const sandboxBoard = sandboxState.board;

    sandboxState.currentPlayer = 1;

    const makeInteriorBlock = (x0: number, y0: number): Position[] => {
      const coords: Position[] = [];
      for (let x = x0; x <= x0 + 2; x++) {
        for (let y = y0; y <= y0 + 2; y++) {
          const p = pos(x, y);
          coords.push(p);
          addStack(backendBoard, p, 2, 1);
          addStack(sandboxBoard, p, 2, 1);
        }
      }
      return coords;
    };

    const block1 = makeInteriorBlock(5, 5);
    const block2 = makeInteriorBlock(11, 5);

    const makeBorder = (x0: number, y0: number): Position[] => {
      const border: Position[] = [];
      for (let x = x0 - 1; x <= x0 + 3; x++) {
        border.push(pos(x, y0 - 1));
        border.push(pos(x, y0 + 3));
      }
      for (let y = y0; y <= y0 + 2; y++) {
        border.push(pos(x0 - 1, y));
        border.push(pos(x0 + 3, y));
      }
      border.forEach((p) => {
        addMarker(backendBoard, p, 1);
        addMarker(sandboxBoard, p, 1);
      });
      return border;
    };

    const border1 = makeBorder(5, 5);
    const border2 = makeBorder(11, 5);

    // Player 3 active elsewhere but not inside either region.
    const p3Outside = pos(0, 0);
    addStack(backendBoard, p3Outside, 3, 1);
    addStack(sandboxBoard, p3Outside, 3, 1);

    // Player 1 stacks outside both regions (for self-elimination across both).
    const outsideP1A = pos(1, 1);
    const outsideP1B = pos(15, 15);
    addStack(backendBoard, outsideP1A, 1, 1);
    addStack(backendBoard, outsideP1B, 1, 1);
    addStack(sandboxBoard, outsideP1A, 1, 1);
    addStack(sandboxBoard, outsideP1B, 1, 1);

    const region1 = {
      spaces: block1,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const region2 = {
      spaces: block2,
      controllingPlayer: 1,
      isDisconnected: true,
    };

    // Stub disconnected-region detection on both engines so we can control
    // the processing order: first region1+region2, then region2 only, then none.
    let backendCallCount = 0;
    const backendFindDisconnectedRegionsSpy = jest
      .spyOn(backendBoardManager, 'findDisconnectedRegions')
      .mockImplementation(() => {
        backendCallCount += 1;
        if (backendCallCount === 1) return [region1, region2];
        if (backendCallCount === 2) return [region2];
        return [];
      });

    let sandboxCallCount = 0;
    const sandboxFindDisconnectedRegionsSpy = jest
      .spyOn(sandboxTerritory, 'findDisconnectedRegionsOnBoard')
      .mockImplementation(() => {
        sandboxCallCount += 1;
        if (sandboxCallCount === 1) return [region1, region2];
        if (sandboxCallCount === 2) return [region2];
        return [];
      });

    const backendInitialTotalEliminated = backendState.totalRingsEliminated;
    const sandboxInitialTotalEliminated = sandboxState.totalRingsEliminated;

    const backendInitialP1 = backendState.players.find((p) => p.playerNumber === 1)!;
    const sandboxInitialP1 = sandboxState.players.find((p) => p.playerNumber === 1)!;

    const backendInitialP1Eliminated = backendInitialP1.eliminatedRings;
    const sandboxInitialP1Eliminated = sandboxInitialP1.eliminatedRings;

    await backendAny.processDisconnectedRegions();
    await sandboxAny.processDisconnectedRegionsForCurrentPlayer();

    expect(backendFindDisconnectedRegionsSpy).toHaveBeenCalled();
    expect(sandboxFindDisconnectedRegionsSpy).toHaveBeenCalled();

    const backendFinalState = backendState;
    const backendFinalBoard = backendFinalState.board;
    const backendP1Final = backendFinalState.players.find((p) => p.playerNumber === 1)!;

    const sandboxFinalState = sandboxEngine.getGameState();
    const sandboxFinalBoard = sandboxFinalState.board;
    const sandboxP1Final = sandboxFinalState.players.find((p) => p.playerNumber === 1)!;

    const interiorKeys1 = new Set(block1.map(keyFrom));
    const interiorKeys2 = new Set(block2.map(keyFrom));
    const borderKeys1 = new Set(border1.map(keyFrom));
    const borderKeys2 = new Set(border2.map(keyFrom));

    // 1. All interior spaces of both regions should be collapsed for P1 and empty of stacks.
    for (const p of [...block1, ...block2]) {
      const key = keyFrom(p);
      expect(backendFinalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(sandboxFinalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(backendFinalBoard.stacks.get(key)).toBeUndefined();
      expect(sandboxFinalBoard.stacks.get(key)).toBeUndefined();
    }

    // 2. All border markers for both regions should be collapsed for P1.
    for (const p of [...border1, ...border2]) {
      const key = keyFrom(p);
      expect(backendFinalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(sandboxFinalBoard.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. Player 1 territorySpaces and collapsed-space counts match.
    const backendCollapsedForP1 = Array.from(backendFinalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    const sandboxCollapsedForP1 = Array.from(sandboxFinalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;

    expect(backendCollapsedForP1).toBe(sandboxCollapsedForP1);
    expect(backendP1Final.territorySpaces).toBe(sandboxP1Final.territorySpaces);

    // 4. All stacks inside both regions are eliminated.
    const backendStacksInRegions = Array.from(backendFinalBoard.stacks.keys()).filter(
      (k) => interiorKeys1.has(k) || interiorKeys2.has(k)
    );
    const sandboxStacksInRegions = Array.from(sandboxFinalBoard.stacks.keys()).filter(
      (k) => interiorKeys1.has(k) || interiorKeys2.has(k)
    );

    expect(backendStacksInRegions.length).toBe(0);
    expect(sandboxStacksInRegions.length).toBe(0);

    // 5. Eliminated ring accounting matches across backend and sandbox.
    const backendFinalTotalEliminated = backendFinalState.totalRingsEliminated;
    const sandboxFinalTotalEliminated = sandboxFinalState.totalRingsEliminated;

    const backendFinalP1Eliminated = backendP1Final.eliminatedRings;
    const sandboxFinalP1Eliminated = sandboxP1Final.eliminatedRings;

    expect(backendFinalTotalEliminated - backendInitialTotalEliminated).toBe(
      sandboxFinalTotalEliminated - sandboxInitialTotalEliminated
    );

    expect(backendFinalP1Eliminated - backendInitialP1Eliminated).toBe(
      sandboxFinalP1Eliminated - sandboxInitialP1Eliminated
    );

    // Also ensure the board-level eliminatedRings bookkeeping matches for player 1.
    const backendBoardP1Elims = backendFinalBoard.eliminatedRings[1] || 0;
    const sandboxBoardP1Elims = sandboxFinalBoard.eliminatedRings[1] || 0;

    expect(backendBoardP1Elims).toBe(sandboxBoardP1Elims);

    // Sanity: the total territory gained for P1 matches the sum of both
    // regions' interior + border spaces.
    const expectedTerritory =
      interiorKeys1.size + interiorKeys2.size + borderKeys1.size + borderKeys2.size;

    expect(backendCollapsedForP1).toBe(expectedTerritory);
    expect(sandboxCollapsedForP1).toBe(expectedTerritory);
  });

  test('seed17_pre_final_board_territory_parity_square8', async () => {
    // Regression parity test for the pre-final seed17 board (square8 / 2p).
    //
    // We reconstruct the boardBeforeSummary for moveNumber 52 from
    // logs/seed17_trace_debug2.log and ensure that a single territory
    // processing pass on backend and sandbox produces identical results for
    // collapsedSpaces, totalRingsEliminated, and per-player
    // territorySpaces/eliminatedRings.

    const seedBoardType: BoardType = 'square8';
    const seedTimeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

    // --- Backend setup (square8 / 2 players) ---
    const backendPlayersSeed: Player[] = [
      {
        id: 'p1-seed17',
        username: 'P1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: seedTimeControl.initialTime * 1000,
        ringsInHand: 2, // from stateHashBefore: 1:2:2:0
        eliminatedRings: 2,
        territorySpaces: 0,
      },
      {
        id: 'p2-seed17',
        username: 'P2',
        type: 'human',
        playerNumber: 2,
        isReady: true,
        timeRemaining: seedTimeControl.initialTime * 1000,
        ringsInHand: 0, // from stateHashBefore: 2:0:5:6
        eliminatedRings: 5,
        territorySpaces: 6,
      },
    ];

    const backendEngineSeed = new GameEngine(
      'territory-seed17-pre-final-backend',
      seedBoardType,
      backendPlayersSeed,
      seedTimeControl,
      false
    );
    const backendAnySeed: any = backendEngineSeed;
    const backendStateSeed: GameState = backendAnySeed.gameState as GameState;
    const backendBoardSeed = backendStateSeed.board;

    backendStateSeed.currentPlayer = 2; // actor 2 moves at moveNumber 52
    backendStateSeed.totalRingsEliminated = 7; // from progressBefore.eliminated

    backendBoardSeed.stacks.clear();
    backendBoardSeed.markers.clear();
    backendBoardSeed.collapsedSpaces.clear();
    backendBoardSeed.eliminatedRings = { 1: 2, 2: 5 };

    const parsePos = (key: string): Position => {
      const [xStr, yStr] = key.split(',');
      return { x: parseInt(xStr, 10), y: parseInt(yStr, 10) };
    };

    const stackSpecsSeed = [
      '0,0:2:1:1',
      '1,3:1:5:5',
      '1,6:2:3:3',
      '2,0:2:5:5',
      '3,0:1:5:2',
      '5,3:1:2:2',
      '5,6:1:1:1',
      '6,4:2:5:1',
    ];

    for (const spec of stackSpecsSeed) {
      const [posKey, playerStr, heightStr] = spec.split(':');
      const pos = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      const height = parseInt(heightStr, 10);
      addStack(backendBoardSeed, pos, player, height);
    }

    const markerSpecsSeed = [
      '0,2:2',
      '0,3:1',
      '0,4:1',
      '0,6:2',
      '1,1:1',
      '1,7:1',
      '2,7:2',
      '3,6:1',
      '4,6:1',
      '4,7:1',
      '5,5:2',
      '7,4:1',
      '7,6:2',
    ];

    for (const spec of markerSpecsSeed) {
      const [posKey, playerStr] = spec.split(':');
      const pos = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      addMarker(backendBoardSeed, pos, player);
    }

    const collapsedSpecsSeed = [
      '1,5:2',
      '2,3:2',
      '3,4:1',
      '3,5:1',
      '4,2:2',
      '4,3:2',
      '5,1:2',
      '5,2:2',
      '6,0:2',
      '7,0:2',
      '7,1:2',
    ];

    for (const spec of collapsedSpecsSeed) {
      const [posKey, ownerStr] = spec.split(':');
      const pos = parsePos(posKey);
      const owner = parseInt(ownerStr, 10);
      addCollapsedSpace(backendBoardSeed, pos, owner);
    }

    // --- Sandbox setup mirroring the same geometry and counters ---
    const sandboxConfigSeed: SandboxConfig = {
      boardType: seedBoardType,
      numPlayers: 2,
      playerKinds: ['human', 'human'],
    };

    const sandboxHandlerSeed: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<any> {
        const optionsArray = ((choice as any).options as any[]) ?? [];
        const selectedOption = optionsArray.length > 0 ? optionsArray[0] : undefined;

        return {
          choiceId: (choice as any).id,
          playerNumber: (choice as any).playerNumber,
          choiceType: (choice as any).type,
          selectedOption,
        };
      },
    };

    const sandboxEngineSeed = new ClientSandboxEngine({
      config: sandboxConfigSeed,
      interactionHandler: sandboxHandlerSeed,
    });
    const sandboxAnySeed: any = sandboxEngineSeed;
    const sandboxStateSeed: GameState = sandboxAnySeed.gameState as GameState;
    const sandboxBoardSeed = sandboxStateSeed.board;

    sandboxStateSeed.currentPlayer = 2;
    sandboxStateSeed.totalRingsEliminated = 7;

    sandboxBoardSeed.stacks.clear();
    sandboxBoardSeed.markers.clear();
    sandboxBoardSeed.collapsedSpaces.clear();
    sandboxBoardSeed.eliminatedRings = { 1: 2, 2: 5 };

    const sp1 = sandboxStateSeed.players.find((p) => p.playerNumber === 1)!;
    sp1.ringsInHand = 2;
    sp1.eliminatedRings = 2;
    sp1.territorySpaces = 0;

    const sp2 = sandboxStateSeed.players.find((p) => p.playerNumber === 2)!;
    sp2.ringsInHand = 0;
    sp2.eliminatedRings = 5;
    sp2.territorySpaces = 6;

    for (const spec of stackSpecsSeed) {
      const [posKey, playerStr, heightStr] = spec.split(':');
      const pos = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      const height = parseInt(heightStr, 10);
      addStack(sandboxBoardSeed, pos, player, height);
    }

    for (const spec of markerSpecsSeed) {
      const [posKey, playerStr] = spec.split(':');
      const pos = parsePos(posKey);
      const player = parseInt(playerStr, 10);
      addMarker(sandboxBoardSeed, pos, player);
    }

    for (const spec of collapsedSpecsSeed) {
      const [posKey, ownerStr] = spec.split(':');
      const pos = parsePos(posKey);
      const owner = parseInt(ownerStr, 10);
      addCollapsedSpace(sandboxBoardSeed, pos, owner);
    }

    // Sanity: initial backend and sandbox geometry and S-invariants match.
    expect(summarizeBoard(backendBoardSeed)).toEqual(summarizeBoard(sandboxBoardSeed));

    const backendSnapBefore = computeProgressSnapshot(backendStateSeed);
    const sandboxSnapBefore = computeProgressSnapshot(sandboxStateSeed);
    expect(backendSnapBefore).toEqual(sandboxSnapBefore);

    const backendCollapsedBefore = backendBoardSeed.collapsedSpaces.size;
    const sandboxCollapsedBefore = sandboxBoardSeed.collapsedSpaces.size;
    expect(backendCollapsedBefore).toBe(sandboxCollapsedBefore);

    const backendTotalElimBefore = backendStateSeed.totalRingsEliminated;
    const sandboxTotalElimBefore = sandboxStateSeed.totalRingsEliminated;
    expect(backendTotalElimBefore).toBe(sandboxTotalElimBefore);

    // --- Run one round of territory processing on both engines ---
    await backendAnySeed.processDisconnectedRegions();
    await sandboxAnySeed.processDisconnectedRegionsForCurrentPlayer();

    const backendFinalStateSeed = backendStateSeed;
    const backendFinalBoardSeed = backendFinalStateSeed.board;

    const sandboxFinalStateSeed = sandboxEngineSeed.getGameState();
    const sandboxFinalBoardSeed = sandboxFinalStateSeed.board;

    // 1. Geometric parity: collapsed spaces, stacks, and markers.
    expect(summarizeBoard(backendFinalBoardSeed)).toEqual(summarizeBoard(sandboxFinalBoardSeed));

    // 2. S-invariant parity.
    const backendSnapAfter = computeProgressSnapshot(backendFinalStateSeed);
    const sandboxSnapAfter = computeProgressSnapshot(sandboxFinalStateSeed);
    expect(backendSnapAfter).toEqual(sandboxSnapAfter);

    // 3. Per-player territory and elimination accounting parity.
    const backendP1Seed = backendFinalStateSeed.players.find((p) => p.playerNumber === 1)!;
    const backendP2Seed = backendFinalStateSeed.players.find((p) => p.playerNumber === 2)!;

    const sandboxP1Seed = sandboxFinalStateSeed.players.find((p) => p.playerNumber === 1)!;
    const sandboxP2Seed = sandboxFinalStateSeed.players.find((p) => p.playerNumber === 2)!;

    expect(backendP1Seed.territorySpaces).toBe(sandboxP1Seed.territorySpaces);
    expect(backendP2Seed.territorySpaces).toBe(sandboxP2Seed.territorySpaces);

    expect(backendP1Seed.eliminatedRings).toBe(sandboxP1Seed.eliminatedRings);
    expect(backendP2Seed.eliminatedRings).toBe(sandboxP2Seed.eliminatedRings);

    // Board-level eliminatedRings and totalRingsEliminated must also match.
    expect(backendFinalStateSeed.totalRingsEliminated).toBe(
      sandboxFinalStateSeed.totalRingsEliminated
    );

    expect(backendFinalBoardSeed.eliminatedRings).toEqual(sandboxFinalBoardSeed.eliminatedRings);
  });
});
