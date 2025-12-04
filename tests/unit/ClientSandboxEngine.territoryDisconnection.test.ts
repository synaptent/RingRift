import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  Position,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
  positionToString,
} from '../../src/shared/types/game';
import { addMarker, addStack, pos } from '../utils/fixtures';

/**
 * ClientSandboxEngine territory-disconnection + region-collapse tests.
 *
 * These mirror the core 19x19 scenario from
 * GameEngine.territoryDisconnection.test.ts but exercise the
 * client-local sandbox engine instead of the backend GameEngine.
 *
 * Goals:
 * - Disconnected regions are detected for the moving player.
 * - All stacks inside the region are eliminated.
 * - Region + border markers collapse to the moving player's color.
 * - All eliminated rings (internal + self-elimination) are credited to
 *   the moving player.
 */

describe('ClientSandboxEngine territory disconnection (square19)', () => {
  const boardType: BoardType = 'square19';

  function createEngine(customHandler?: SandboxInteractionHandler): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = customHandler || {
      // Default handler: always pick the first option for any choice.
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

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  test('processDisconnectedRegionsForCurrentPlayer collapses a canonical 3x3 region and credits eliminations', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    const board = state.board;

    // Match the 19x19 territory scenario used by the backend tests:
    // - Player 2 (B) controls a 3x3 interior block at (5,5)–(7,7).
    // - Player 1 (A) forms a continuous marker border around this block.
    // - Player 3 (C) has a stack elsewhere, so C is active but not
    //   represented inside the region.
    // - Player 1 also has at least one stack outside the region to pay
    //   the self-elimination prerequisite.

    state.currentPlayer = 1;

    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        addStack(board, p, 2, 1); // B stacks (player 2)
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
    borderCoords.forEach((p) => addMarker(board, p, 1)); // A markers (player 1)

    // Player 1 stack outside the region (for self-elimination).
    const outsideP1 = pos(1, 1);
    addStack(board, outsideP1, 1, 1);

    // Player 3 active elsewhere but not inside the region.
    const outsideP3 = pos(0, 0);
    addStack(board, outsideP3, 3, 1);

    // No collapsed spaces initially.
    expect(board.collapsedSpaces.size).toBe(0);

    const initialTotalEliminated = state.totalRingsEliminated;
    const initialP1Eliminated = state.players.find((p) => p.playerNumber === 1)!.eliminatedRings;

    // Drive the sandbox territory processing loop directly.
    await engineAny.processDisconnectedRegionsForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    const interiorKeys = new Set(interiorCoords.map(positionToString));
    const borderKeys = new Set(borderCoords.map(positionToString));

    // 1. Interior region spaces should be collapsed to player 1 and free of stacks.
    interiorCoords.forEach((p) => {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.stacks.get(key)).toBeUndefined();
    });

    // 2. Border marker positions should be collapsed to player 1.
    borderCoords.forEach((p) => {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
    });

    // 3. Player 1's territorySpaces should be consistent with the number
    //    of collapsed spaces they own (region + border).
    const collapsedForP1 = Array.from(finalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    expect(player1.territorySpaces).toBe(collapsedForP1);

    // 4. All stacks inside the region should have been eliminated.
    const stacksInRegion = Array.from(finalBoard.stacks.keys()).filter((k) => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    // 5. Eliminated rings: 9 internal B stacks (one ring each) plus one
    //    self-elimination for player 1.
    const expectedDeltaForP1 = 10;
    const finalP1Eliminated = player1.eliminatedRings;
    expect(finalP1Eliminated).toBe(initialP1Eliminated + expectedDeltaForP1);
    expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated + expectedDeltaForP1);
  });

  test('processDisconnectedRegionsForCurrentPlayer processes multiple disconnected regions in sequence', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    const board = state.board;

    state.currentPlayer = 1;

    const makeInteriorBlock = (x0: number, y0: number): Position[] => {
      const coords: Position[] = [];
      for (let x = x0; x <= x0 + 2; x++) {
        for (let y = y0; y <= y0 + 2; y++) {
          const p = pos(x, y);
          coords.push(p);
          addStack(board, p, 2, 1); // B stacks (player 2)
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
      border.forEach((p) => addMarker(board, p, 1)); // A markers (player 1)
      return border;
    };

    const border1 = makeBorder(5, 5);
    const border2 = makeBorder(11, 5);

    // Player 3 (C) active elsewhere but not inside either region.
    addStack(board, pos(0, 0), 3, 1);

    // Player 1 has stacks outside both regions (for self-elimination across both).
    const outsideP1A = pos(1, 1);
    const outsideP1B = pos(15, 15);
    addStack(board, outsideP1A, 1, 1);
    addStack(board, outsideP1B, 1, 1);

    expect(board.collapsedSpaces.size).toBe(0);

    const initialTotalEliminated = state.totalRingsEliminated;
    const initialP1Eliminated = state.players.find((p) => p.playerNumber === 1)!.eliminatedRings;

    await engineAny.processDisconnectedRegionsForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    const keysFrom = (positions: Position[]) => new Set(positions.map(positionToString));

    const interiorKeys1 = keysFrom(block1);
    const interiorKeys2 = keysFrom(block2);
    const borderKeys1 = keysFrom(border1);
    const borderKeys2 = keysFrom(border2);

    // 1. All interior spaces of both regions should be collapsed for P1 and empty of stacks.
    for (const p of [...block1, ...block2]) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.stacks.get(key)).toBeUndefined();
    }

    // 2. All border markers for both regions should be collapsed for P1.
    for (const p of [...border1, ...border2]) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. Player 1's territorySpaces should equal the number of collapsed
    //    spaces owned by P1 in this scenario (two regions + borders).
    const collapsedForP1 = Array.from(finalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    const expectedTerritory =
      interiorKeys1.size + interiorKeys2.size + borderKeys1.size + borderKeys2.size;
    expect(collapsedForP1).toBe(expectedTerritory);
    expect(player1.territorySpaces).toBe(collapsedForP1);

    // 4. All stacks inside both regions should be eliminated.
    const stacksInRegions = Array.from(finalBoard.stacks.keys()).filter(
      (k) => interiorKeys1.has(k) || interiorKeys2.has(k)
    );
    expect(stacksInRegions.length).toBe(0);

    // 5. Eliminated ring counts: start with 18 B rings inside the two regions
    //    and two P1 rings outside. Each region collapse eliminates 9 internal
    //    rings plus one self-elim for P1, so total eliminated rings credited
    //    to P1 should be 20.
    const expectedEliminatedForP1 = 20;
    const finalP1Eliminated = player1.eliminatedRings;
    expect(finalP1Eliminated).toBe(initialP1Eliminated + expectedEliminatedForP1);
    expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated + expectedEliminatedForP1);
  });

  test('line + territory consequences combine in a single post-move cycle', async () => {
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    const board = state.board;

    state.currentPlayer = 1;

    // --- 1. Set up a canonical disconnected region for player 2 (B) ---
    // IMPORTANT: For square19 boards, lineLength=4, so we must use a 3x3 border (max 3 markers/row)
    // to avoid accidentally triggering line processing before territory processing!
    const interiorCoords: Position[] = [];
    // Single interior space at (5,5)
    const p = pos(5, 5);
    interiorCoords.push(p);
    addStack(board, p, 2, 1); // B stack (player 2)

    // 3x3 border: max 3 markers in a row (won't form a line on square19)
    const borderCoords: Position[] = [];
    // Top row: (4,4), (5,4), (6,4) = 3 markers
    for (let x = 4; x <= 6; x++) {
      borderCoords.push(pos(x, 4));
    }
    // Bottom row: (4,6), (5,6), (6,6) = 3 markers
    for (let x = 4; x <= 6; x++) {
      borderCoords.push(pos(x, 6));
    }
    // Left column: (4,5) = 1 marker
    borderCoords.push(pos(4, 5));
    // Right column: (6,5) = 1 marker
    borderCoords.push(pos(6, 5));
    // Total: 8 border markers
    borderCoords.forEach((p) => addMarker(board, p, 1)); // A markers (player 1)

    // Player 3 (C) active elsewhere but not inside the region.
    addStack(board, pos(0, 0), 3, 1);

    // --- 2. Set up a horizontal line of exactly 4 A markers (matches lineLength for square19) ---
    const lineCoords: Position[] = [];
    for (let x = 0; x < 4; x++) {
      const p = pos(x, 10);
      lineCoords.push(p);
      addMarker(board, p, 1);
    }

    // --- 3. Provide P1 stacks: one for line elimination, one for territory self-elim.
    const lineStackPos = pos(1, 1);
    const territoryStackPos = pos(15, 15);
    addStack(board, lineStackPos, 1, 1);
    addStack(board, territoryStackPos, 1, 1);

    // Sanity: no collapsed spaces and no eliminated rings yet.
    expect(board.collapsedSpaces.size).toBe(0);
    expect(state.board.eliminatedRings[1] || 0).toBe(0);
    expect(state.players.find((p) => p.playerNumber === 1)!.eliminatedRings).toBe(0);
    expect(state.totalRingsEliminated).toBe(0);

    // --- 4. Run the same post-move pipeline used by advanceAfterMovement.
    await engineAny.processLinesForCurrentPlayer();
    await engineAny.processDisconnectedRegionsForCurrentPlayer();

    const finalState = engine.getGameState();
    const finalBoard = finalState.board;
    const player1 = finalState.players.find((p) => p.playerNumber === 1)!;

    const keysFrom = (positions: Position[]) => new Set(positions.map(positionToString));

    const interiorKeys = keysFrom(interiorCoords);
    const borderKeys = keysFrom(borderCoords);
    const lineKeys = keysFrom(lineCoords);

    // 1. All interior region spaces should be collapsed for P1 and empty of stacks.
    for (const p of interiorCoords) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
      expect(finalBoard.stacks.get(key)).toBeUndefined();
    }

    // 2. All border markers should be collapsed for P1.
    for (const p of borderCoords) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. All line marker positions should be collapsed for P1 as a result of line processing.
    for (const p of lineCoords) {
      const key = positionToString(p);
      expect(finalBoard.collapsedSpaces.get(key)).toBe(1);
    }

    // 4. Player 1's territorySpaces should equal the number of collapsed spaces owned by P1.
    const collapsedForP1 = Array.from(finalBoard.collapsedSpaces.values()).filter(
      (v) => v === 1
    ).length;
    const expectedTerritory = interiorKeys.size + borderKeys.size + lineKeys.size;
    expect(collapsedForP1).toBe(expectedTerritory);
    expect(player1.territorySpaces).toBe(collapsedForP1);

    // 5. All stacks inside the region should be eliminated.
    const stacksInRegion = Array.from(finalBoard.stacks.keys()).filter((k) => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    // 6. Eliminated ring counts should combine line + territory contributions:
    //    - 1 internal B stack (one ring) eliminated when P1 processes territory
    //    - 1 ring from a P1 stack self-eliminated for the line
    //    - 1 ring from a P1 stack self-eliminated for territory processing
    //    Total: 3 rings attributed to player 1.
    const expectedEliminatedForP1 = 3;
    const finalP1Eliminated = player1.eliminatedRings;
    expect(finalP1Eliminated).toBe(expectedEliminatedForP1);
    expect(finalState.board.eliminatedRings[1]).toBe(expectedEliminatedForP1);
    expect(finalState.totalRingsEliminated).toBe(expectedEliminatedForP1);
  });

  test('Q23_disconnected_region_illegal_when_no_self_elimination_available_sandbox', async () => {
    // Sandbox parity for FAQ Q23: if the moving player has no stack outside
    // a disconnected region, that region must not be processed.
    const engine = createEngine();
    const engineAny = engine as any;
    const state: GameState = engineAny.gameState as GameState;
    const board = state.board;

    state.currentPlayer = 1;

    // Synthetic 3x3 region in the middle of the board.
    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        // Place stacks for player 1 *inside* the region only.
        addStack(board, p, 1, 1);
      }
    }

    // Player 2 is active elsewhere but not inside the region, to satisfy
    // the representation-based disconnection criteria.
    addStack(board, pos(0, 0), 2, 1);

    // Crucially, player 1 has NO stacks outside the region.
    const stacksP1 = Array.from(board.stacks.values()).filter((s) => s.controllingPlayer === 1);
    const outsideRegion = stacksP1.filter(
      (s) => !interiorCoords.some((p) => positionToString(p) === positionToString(s.position))
    );
    expect(outsideRegion.length).toBe(0);

    const initialCollapsedCount = board.collapsedSpaces.size;
    const initialTotalEliminated = state.totalRingsEliminated;
    const initialP1Eliminated = state.players.find((p) => p.playerNumber === 1)!.eliminatedRings;

    await engineAny.processDisconnectedRegionsForCurrentPlayer();

    // Region should NOT have been processed:
    // - No additional collapsed spaces.
    // - All interior stacks remain.
    // - No eliminations credited to player 1.
    expect(board.collapsedSpaces.size).toBe(initialCollapsedCount);

    const stacksInRegion = Array.from(board.stacks.keys()).filter((key) =>
      interiorCoords.some((p) => positionToString(p) === key)
    );
    expect(stacksInRegion.length).toBe(interiorCoords.length);

    const finalTotalEliminated = state.totalRingsEliminated;
    const finalP1Eliminated = state.players.find((p) => p.playerNumber === 1)!.eliminatedRings;
    expect(finalTotalEliminated).toBe(initialTotalEliminated);
    expect(finalP1Eliminated).toBe(initialP1Eliminated);
  });
});
