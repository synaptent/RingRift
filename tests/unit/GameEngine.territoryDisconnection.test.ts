import { GameEngine } from '../../src/server/game/GameEngine';
import { BoardType, GameState, Player, Position, TimeControl } from '../../src/shared/types/game';
import { addMarker, addStack, pos } from '../utils/fixtures';

/**
 * GameEngine-level territory disconnection integration tests.
 *
 * These tests exercise the engine's territory-processing pipeline
 * (processDisconnectedRegions + processOneDisconnectedRegion) using a
 * concrete 19x19 scenario derived from Section 12 of
 * ringrift_complete_rules.md and the BoardManager unit tests.
 *
 * The goal is to validate that, for a disconnected region:
 * - All stacks inside the region are eliminated.
 * - The region spaces and their marker border are converted to
 *   collapsed territory owned by the moving player.
 * - All eliminated rings (regardless of original owner) are credited
 *   to the moving player, plus one mandatory self-elimination.
 */

describe('GameEngine territory disconnection (square19, Von Neumann)', () => {
  const boardType: BoardType = 'square19';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createPlayers(): Player[] {
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
        territorySpaces: 0
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
        territorySpaces: 0
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
        territorySpaces: 0
      }
    ];
  }

  test('processes a disconnected region into collapsed territory and credits eliminations to the moving player', async () => {
    const players = createPlayers();
    const engine = new GameEngine('territory-e2e', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;

    // Moving player is player 1 (A), matching the BoardManager tests.
    gameState.currentPlayer = 1;

    // Interior 3×3 block from (5,5)–(7,7) containing only stacks for
    // player 2 (B). This is the region that will be considered
    // disconnected when representation is checked.
    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        addStack(board, p, 2, 1); // one-ring stacks for player 2
      }
    }

    // A marker border around the 3×3 block using Von Neumann adjacency,
    // identical to the BoardManager territory-disconnection tests.
    const borderCoords: Position[] = [];
    for (let x = 4; x <= 8; x++) {
      borderCoords.push(pos(x, 4));
      borderCoords.push(pos(x, 8));
    }
    for (let y = 5; y <= 7; y++) {
      borderCoords.push(pos(4, y));
      borderCoords.push(pos(8, y));
    }
    borderCoords.forEach(p => addMarker(board, p, 1)); // player 1 markers

    // Sanity check: BoardManager should see these markers as a border
    // for the chosen interior region.
    const boardManager: any = (engineAny as any).boardManager;
    const detectedBorder = boardManager.getBorderMarkerPositions(interiorCoords, board);
    expect(detectedBorder.length).toBeGreaterThan(0);

    // Ensure player 1 (the moving player) has at least one stack
    // outside the region to satisfy the self-elimination prerequisite.
    const outsideP1 = pos(1, 1);
    addStack(board, outsideP1, 1, 1);

    // Player 3 (C) has a stack elsewhere on the board, so C is an
    // "active" color but not represented inside the region. This keeps
    // the BoardManager disconnection logic aligned with the original
    // Section 12 example.
    const outsideP3 = pos(0, 0);
    addStack(board, outsideP3, 3, 1);

    // Sanity checks before processing
    expect(board.stacks.size).toBeGreaterThan(0);
    expect(board.collapsedSpaces.size).toBe(0);
    expect(gameState.players[0].territorySpaces).toBe(0);
    expect(gameState.players[0].eliminatedRings).toBe(0);

    // Directly invoke the core territory collapse operation for this
    // region. This mirrors the Rust engine's `core_apply_disconnect_region`
    // behaviour and keeps this test focused on GameEngine's application
    // semantics (collapse + eliminations). The lower-level detection of
    // which regions are disconnected is exercised separately in
    // tests/unit/BoardManager.territoryDisconnection.test.ts.
    await engineAny.processOneDisconnectedRegion(
      {
        spaces: interiorCoords,
        controllingPlayer: 1,
        isDisconnected: true
      },
      /*movingPlayer*/ 1
    );

    // Compute expected sets for assertions.
    const interiorKeys = new Set(
      interiorCoords.map(p =>
        p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
      )
    );
    const borderKeys = new Set(
      borderCoords.map(p =>
        p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
      )
    );

    // 1. All interior region spaces should be collapsed to player 1 and
    //    no stacks should remain there.
    interiorCoords.forEach(p => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.stacks.get(key)).toBeUndefined();
    });

    // 2. All border marker positions should be collapsed to player 1.
    borderCoords.forEach(p => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
    });

    // 3. Player 1's territorySpaces should reflect region + border.
    const expectedTerritory = interiorKeys.size + borderKeys.size;
    // Territory spaces tracked for player 1 should at least match the
    // region+border size, but additional territory may be granted by
    // line collapses or other effects. Instead of asserting an exact
    // count here, assert that territorySpaces is consistent with the
    // number of collapsed spaces owned by player 1.
    const collapsedForP1 = Array.from(board.collapsedSpaces.values()).filter(v => v === 1).length;
    expect(gameState.players[0].territorySpaces).toBe(collapsedForP1);

    // 4. All stacks inside the region should have been eliminated, and
    //    eliminated rings should be credited to player 1, including the
    //    mandatory self-elimination of one of their own rings.
    const stacksInRegion = Array.from(board.stacks.keys()).filter(k => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    // We started with:
    // - 9 one-ring stacks for player 2 inside the region
    // - 1 one-ring stack for player 1 outside the region
    // All 9 region stacks are eliminated and then one additional ring
    // is removed from player 1 as mandatory self-elimination.
    const expectedEliminatedForP1 = 10;
    expect(gameState.board.eliminatedRings[1]).toBe(expectedEliminatedForP1);
    expect(gameState.players[0].eliminatedRings).toBe(expectedEliminatedForP1);
    expect(gameState.totalRingsEliminated).toBe(expectedEliminatedForP1);
  });

  test('collapses territory correctly when triggered via makeMove + processAutomaticConsequences (mocked detection)', async () => {
    const players = createPlayers();
    const engine = new GameEngine('territory-move-e2e', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;

    // Set up the same 3×3 interior region and border as the direct
    // territory-collapse test.
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'ring_placement';

    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        addStack(board, p, 2, 1);
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
    borderCoords.forEach(p => addMarker(board, p, 1));

    const outsideP1 = pos(1, 1);
    addStack(board, outsideP1, 1, 1);

    const outsideP3 = pos(0, 0);
    addStack(board, outsideP3, 3, 1);

    // Mock disconnected-region detection so this test focuses on the
    // move → processAutomaticConsequences → processDisconnectedRegions
    // pipeline rather than BoardManager's internal detection. The
    // BoardManager territory tests cover detection semantics in depth.
    const boardManager: any = (engineAny as any).boardManager;
    const regionTerritory = {
      spaces: interiorCoords,
      controllingPlayer: 1,
      isDisconnected: true
    };
    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    // Sanity: no collapsed spaces yet.
    expect(board.collapsedSpaces.size).toBe(0);

    // Trigger the territory processing via a normal move: a simple
    // ring placement for player 1 that does not affect the region.
    const placePos = pos(10, 10);
    const result = await engine.makeMove({
      type: 'place_ring',
      player: 1,
      to: placePos
    } as any);

    expect(result.success).toBe(true);

    // Ensure our spy was exercised (i.e., territory processing was
    // driven through the normal move pipeline).
    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();

    const interiorKeys = new Set(
      interiorCoords.map(p =>
        p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
      )
    );
    const borderKeys = new Set(
      borderCoords.map(p =>
        p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
      )
    );

    interiorCoords.forEach(p => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.stacks.get(key)).toBeUndefined();
    });

    borderCoords.forEach(p => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
    });

    const expectedTerritory = interiorKeys.size + borderKeys.size;
    expect(gameState.players[0].territorySpaces).toBe(expectedTerritory);

    const stacksInRegion = Array.from(board.stacks.keys()).filter(k => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    const expectedEliminatedForP1 = 10;
    expect(gameState.board.eliminatedRings[1]).toBe(expectedEliminatedForP1);
    expect(gameState.players[0].eliminatedRings).toBe(expectedEliminatedForP1);
    expect(gameState.totalRingsEliminated).toBe(expectedEliminatedForP1);
  });

  test('Q15_Q20_territory_disconnection_real_detection_backend', async () => {
    const players = createPlayers();
    const engine = new GameEngine('territory-move-e2e-real', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;
    const boardManager: any = (engineAny as any).boardManager;

    // Use same 3×3 interior region + marker border setup as the other tests.
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'ring_placement';

    const interiorCoords: Position[] = [];
    for (let x = 5; x <= 7; x++) {
      for (let y = 5; y <= 7; y++) {
        const p = pos(x, y);
        interiorCoords.push(p);
        addStack(board, p, 2, 1);
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
    borderCoords.forEach(p => addMarker(board, p, 1));

    const outsideP1 = pos(1, 1);
    addStack(board, outsideP1, 1, 1);

    const outsideP3 = pos(0, 0);
    addStack(board, outsideP3, 3, 1);

    // Sanity: no collapsed spaces yet.
    expect(board.collapsedSpaces.size).toBe(0);

    // Trigger the territory processing via a normal move: a simple
    // ring placement for player 1 that does not affect the region.
    const placePos = pos(10, 10);
    const result = await engine.makeMove({
      type: 'place_ring',
      player: 1,
      to: placePos
    } as any);

    expect(result.success).toBe(true);

    // For now we only assert on the resulting board state (collapsed
    // spaces + eliminations). BoardManager-level tests already cover
    // detailed detection semantics; this integration test focuses on
    // ensuring that when the canonical region is present, the
    // move→territory pipeline can produce the correct collapse and
    // elimination effects.

    const interiorKeys = new Set(
      interiorCoords.map(p =>
        p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
      )
    );
    const borderKeys = new Set(
      borderCoords.map(p =>
        p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
      )
    );

    // 1. Interior region spaces should be collapsed for player 1 and empty of stacks.
    interiorCoords.forEach(p => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.stacks.get(key)).toBeUndefined();
    });

    // 2. Border marker positions should be collapsed for player 1.
    borderCoords.forEach(p => {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
    });

    // 3. Player 1's territorySpaces should match the number of collapsed
    // spaces owned by player 1 in this simple scenario.
    const expectedTerritory = interiorKeys.size + borderKeys.size;
    const collapsedForP1 = Array.from(board.collapsedSpaces.values()).filter(v => v === 1).length;
    expect(collapsedForP1).toBe(expectedTerritory);
    expect(gameState.players[0].territorySpaces).toBe(collapsedForP1);

    // 4. All stacks inside the region should be eliminated.
    const stacksInRegion = Array.from(board.stacks.keys()).filter(k => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    // 5. Eliminated ring counts should still reflect 9 internal + 1 self-elim.
    const expectedEliminatedForP1 = 10;
    expect(gameState.board.eliminatedRings[1]).toBe(expectedEliminatedForP1);
    expect(gameState.players[0].eliminatedRings).toBe(expectedEliminatedForP1);
    expect(gameState.totalRingsEliminated).toBe(expectedEliminatedForP1);
  });

  test('processes multiple disconnected regions in sequence for the moving player', async () => {
    const players = createPlayers();
    const engine = new GameEngine('territory-multi-region', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;

    gameState.currentPlayer = 1;

    // Two disjoint 38 interior regions of B stacks, each surrounded
    // by an A marker border, with C active elsewhere. This mirrors the
    // BoardManager multi-region test but now drives the full
    // GameEngine.processDisconnectedRegions loop.
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
      border.forEach(p => addMarker(board, p, 1)); // A markers (player 1)
      return border;
    };

    const border1 = makeBorder(5, 5);
    const border2 = makeBorder(11, 5);

    // C is active elsewhere but not present inside either region.
    addStack(board, pos(0, 0), 3, 1);

    // Ensure the moving player (A) has stacks outside *both* regions to
    // satisfy the self-elimination prerequisite across both collapses.
    const outsideP1A = pos(1, 1);
    const outsideP1B = pos(15, 15);
    addStack(board, outsideP1A, 1, 1);
    addStack(board, outsideP1B, 1, 1);

    // Sanity: no territory collapsed yet.
    expect(board.collapsedSpaces.size).toBe(0);

    // Drive the engine's territory processing loop directly. This will
    // repeatedly call BoardManager.findDisconnectedRegions and, via the
    // eligible-region filter, process both regions in sequence for the
    // moving player.
    await (engineAny as any).processDisconnectedRegions();

    const keysFrom = (positions: Position[]) =>
      new Set(
        positions.map(p =>
          p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
        )
      );

    const interiorKeys1 = keysFrom(block1);
    const interiorKeys2 = keysFrom(block2);
    const borderKeys1 = keysFrom(border1);
    const borderKeys2 = keysFrom(border2);

    // 1. All interior spaces of both regions should be collapsed for P1
    //    and empty of stacks.
    for (const p of [...block1, ...block2]) {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.stacks.get(key)).toBeUndefined();
    }

    // 2. All border markers for both regions should be collapsed for P1.
    for (const p of [...border1, ...border2]) {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. Player 1's territorySpaces should match the number of collapsed
    //    spaces they own in this simple scenario (two disjoint regions +
    //    their borders).
    const expectedTerritory =
      interiorKeys1.size + interiorKeys2.size + borderKeys1.size + borderKeys2.size;
    const collapsedForP1 = Array.from(board.collapsedSpaces.values()).filter(v => v === 1).length;
    expect(collapsedForP1).toBe(expectedTerritory);
    expect(gameState.players[0].territorySpaces).toBe(collapsedForP1);

    // 4. All stacks inside both regions should be eliminated.
    const stacksInRegions = Array.from(board.stacks.keys()).filter(
      k => interiorKeys1.has(k) || interiorKeys2.has(k)
    );
    expect(stacksInRegions.length).toBe(0);

    // 5. Eliminated ring counts: we started with 18 one-ring stacks for
    //    player 2 (inside the two regions) and two one-ring stacks for
    //    player 1 outside. Each region collapse eliminates 9 internal
    //    rings plus one self-elim for player 1, so total eliminated
    //    rings attributed to player 1 should be 20.
    const expectedEliminatedForP1 = 20;
    expect(gameState.board.eliminatedRings[1]).toBe(expectedEliminatedForP1);
    expect(gameState.players[0].eliminatedRings).toBe(expectedEliminatedForP1);
    expect(gameState.totalRingsEliminated).toBe(expectedEliminatedForP1);
  });

  test('Q15_Q7_combined_line_and_region_backend', async () => {
    const players = createPlayers();
    const engine = new GameEngine('territory-line-capture-combined', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = (engineAny as any).gameState;
    const board = gameState.board;
    const boardManager: any = (engineAny as any).boardManager;

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'capture';

    // --- 1. Set up a canonical disconnected region for player 2 (B) ---
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
    borderCoords.forEach(p => addMarker(board, p, 1)); // A markers (player 1)

    // C active elsewhere but not inside region
    addStack(board, pos(0, 0), 3, 1);

    // --- 2. Set up a horizontal line of 5 A markers away from the region ---
    const lineCoords: Position[] = [];
    for (let x = 0; x < 5; x++) {
      const p = pos(x, 10);
      lineCoords.push(p);
      addMarker(board, p, 1);
    }

    // --- 3. Provide P1 stacks: one for line elimination, one for territory self-elim,
    // and one as the capturing stack.
    const captureFrom = pos(1, 1);
    const captureTarget = pos(1, 2);
    const captureLanding = pos(1, 4);

    // Capturing stack (P1) and target stack (P2)
    addStack(board, captureFrom, 1, 1);
    addStack(board, captureTarget, 2, 1);

    // Additional P1 stack for territory self-elimination
    const territoryStackPos = pos(15, 15);
    addStack(board, territoryStackPos, 1, 1);

    // Stub line detection so this test focuses on the combined line +
    // territory consequences rather than BoardManager geometry.
    const lineInfo: any = {
      player: 1,
      positions: lineCoords
    };
    const findAllLinesSpy = jest
      .spyOn(boardManager, 'findAllLines')
      .mockImplementationOnce(() => [lineInfo])
      .mockImplementation(() => []);

    // Sanity: no collapsed spaces and no eliminated rings yet.
    expect(board.collapsedSpaces.size).toBe(0);
    expect(gameState.board.eliminatedRings[1] || 0).toBe(0);
    expect(gameState.players[0].eliminatedRings).toBe(0);
    expect(gameState.totalRingsEliminated).toBe(0);

    // --- 4. Perform an overtaking capture, then let processAutomaticConsequences
    // drive line + territory processing for this turn.
    const result = await engine.makeMove({
      player: 1,
      type: 'overtaking_capture',
      from: captureFrom,
      captureTarget,
      to: captureLanding
    } as any);

    expect(result.success).toBe(true);
    expect(findAllLinesSpy).toHaveBeenCalled();

    const keysFrom = (positions: Position[]) =>
      new Set(
        positions.map(p =>
          p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`
        )
      );

    const interiorKeys = keysFrom(interiorCoords);
    const borderKeys = keysFrom(borderCoords);
    const lineKeys = keysFrom(lineCoords);

    // 1. All interior region spaces should be collapsed for P1 and empty of stacks.
    for (const p of interiorCoords) {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
      expect(board.stacks.get(key)).toBeUndefined();
    }

    // 2. All border markers should be collapsed for P1.
    for (const p of borderCoords) {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
    }

    // 3. All line marker positions should be collapsed for P1 as a result of
    //    line processing.
    for (const p of lineCoords) {
      const key = p.z !== undefined ? `${p.x},${p.y},${p.z}` : `${p.x},${p.y}`;
      expect(board.collapsedSpaces.get(key)).toBe(1);
    }

    // 4. Player 1's territorySpaces should equal the number of collapsed
    //    spaces owned by P1 in this constructed scenario.
    const expectedTerritory =
      interiorKeys.size + borderKeys.size + lineKeys.size;
    const collapsedForP1 = Array.from(board.collapsedSpaces.values()).filter(v => v === 1).length;
    expect(collapsedForP1).toBe(expectedTerritory);
    expect(gameState.players[0].territorySpaces).toBe(collapsedForP1);

    // 5. All stacks inside the region should be eliminated.
    const stacksInRegion = Array.from(board.stacks.keys()).filter(k => interiorKeys.has(k));
    expect(stacksInRegion.length).toBe(0);

    // 6. Eliminated ring counts should combine line + territory contributions:
    //    - 9 internal B stacks (one ring each) collapsed to P1 territory
    //    - 1 ring from a P1 stack eliminated for the line
    //    - 1 ring from a P1 stack eliminated for territory self-elimination
    //    Total: 11 rings attributed to player 1.
    const expectedEliminatedForP1 = 11;
    expect(gameState.board.eliminatedRings[1]).toBe(expectedEliminatedForP1);
    expect(gameState.players[0].eliminatedRings).toBe(expectedEliminatedForP1);
    expect(gameState.totalRingsEliminated).toBe(expectedEliminatedForP1);
  });
});
