import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  BoardType as BoardTypeAlias,
  GameState,
  Player,
  Position,
  RingStack,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Scenario Tests: Line and Territory Interactions
 *
 * Covers combined scenarios from the rules/FAQ where a single turn can:
 * - Form a line (Section 11 / FAQ Q7, Q22)
 * - Then trigger territory disconnection processing (Section 12 / FAQ Q20, Q23)
 *
 * These tests are backend-focused and use the same internal helpers exercised
 * by the existing line/territory scenario suites, but compose them to ensure
 * ordering and combined effects behave as documented.
 */

describe('Scenario: Line and Territory Interactions (FAQ 7, 20, 22, 23; backend)', () => {
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Player1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 18,
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
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createEngine(boardType: BoardType = 'square8'): {
    engine: GameEngine;
    gameState: GameState;
    boardManager: any;
  } {
    const engine = new GameEngine(
      'scenario-line-territory',
      boardType as BoardTypeAlias,
      basePlayers,
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const boardManager: any = engineAny.boardManager;
    return { engine, gameState, boardManager };
  }

  function makeStack(
    boardManager: any,
    gameState: GameState,
    playerNumber: number,
    height: number,
    position: Position
  ) {
    const rings = Array(height).fill(playerNumber);
    const stack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    boardManager.setStack(position, stack, gameState.board);
  }

  const boardTypesUnderTest: BoardType[] = ['square8', 'square19', 'hexagonal'];

  test.each<BoardType>(boardTypesUnderTest)(
    'Q7_Q20_combined_line_and_territory_processing_order_backend_%s',
    async (boardType) => {
      // Rules reference:
      // - Section 11.2 / FAQ Q7, Q22: line formation & graduated rewards.
      // - Section 12.2 / 12.3 / FAQ Q20, Q23: territory disconnection with
      //   self-elimination prerequisite and chain reactions.
      //
      // Scenario shape (backend GameEngine, no PlayerInteractionManager wired):
      // - A single overlong line (length = requiredLength + 1) for Player 1
      //   exists on the board. With no interaction handler, backend defaults
      //   to Option 2: collapse exactly requiredLength markers and perform
      //   NO ring elimination (see GameEngine.lines.scenarios.test).
      // - A disconnected region for Player 1 also exists, containing a single
      //   Player 2 stack and satisfying the self-elimination prerequisite
      //   (Player 1 has a stack outside the region).
      // - We invoke the same internal helpers that the engine uses for
      //   automatic post-move processing: first processLineFormations, then
      //   processDisconnectedRegions.
      // - Expected on all board types (square8, square19, hexagonal):
      //   * Line collapse happens first, creating requiredLength collapsed
      //     spaces for Player 1 and NO ring elimination.
      //   * Territory processing then collapses the disconnected region,
      //     eliminates all rings inside it (Player 2), and forces a single
      //     self-elimination from Player 1, with all eliminations credited
      //     to Player 1.
 
    const { engine, gameState, boardManager } = createEngine(boardType);
    const engineAny: any = engine;
    const board = gameState.board;
    const requiredLength = BOARD_CONFIGS[gameState.boardType].lineLength;

    gameState.currentPlayer = 1;

    // Clear any existing board state for a clean scenario.
    board.markers.clear();
    board.stacks.clear();
    board.collapsedSpaces.clear();

    // Synthetic overlong line (length = requiredLength + 1) for Player 1.
    const linePositions: Position[] = [];
    for (let i = 0; i < requiredLength + 1; i++) {
      linePositions.push({ x: i, y: 0 });
    }

    // Stub BoardManager.findAllLines to return this single line once,
    // then no further lines. This mirrors the pattern used in
    // GameEngine.lines.scenarios.test.ts and isolates semantics of
    // line processing from geometric detection.
    const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
    findAllLinesSpy
      .mockImplementationOnce(() => [
        {
          player: 1,
          positions: linePositions,
          length: linePositions.length,
          direction: { x: 1, y: 0 },
        },
      ])
      .mockImplementation(() => []);

    // Territory region: a single-cell region at (5,5) containing a
    // Player 2 stack, plus a Player 1 stack outside the region so the
    // self-elimination prerequisite is satisfied.
    const regionPos: Position = { x: 5, y: 5 };
    const outsidePos: Position = { x: 7, y: 7 };

    makeStack(boardManager, gameState, 2, 1, regionPos); // victim inside region
    makeStack(boardManager, gameState, 1, 2, outsidePos); // P1 stack outside

    const regionTerritory = {
      spaces: [regionPos],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const findDisconnectedRegionsSpy = jest
      .spyOn(boardManager, 'findDisconnectedRegions')
      .mockImplementationOnce(() => [regionTerritory])
      .mockImplementation(() => []);

    // For simplicity, make border marker detection a no-op: this
    // scenario focuses on ordering and elimination/territory effects,
    // not the exact border geometry.
    const getBorderMarkersSpy = jest
      .spyOn(boardManager, 'getBorderMarkerPositions')
      .mockImplementation(() => []);

    const player1Before = gameState.players.find((p) => p.playerNumber === 1)!;
    const initialTerritory = player1Before.territorySpaces;
    const initialEliminated = player1Before.eliminatedRings;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const initialCollapsedCount = board.collapsedSpaces.size;

    // 1) Process line formations for the current player (Player 1).
    await engineAny.processLineFormations();

    const player1AfterLines = gameState.players.find((p) => p.playerNumber === 1)!;

    // Overlong line + no interaction manager → Option 2: exactly
    // requiredLength markers collapsed, no ring elimination.
    const collapsedKeysAfterLines = new Set<string>();
    for (const [key, owner] of board.collapsedSpaces) {
      if (owner === 1) collapsedKeysAfterLines.add(key);
    }

    expect(collapsedKeysAfterLines.size - initialCollapsedCount).toBe(requiredLength);
    expect(player1AfterLines.eliminatedRings).toBe(initialEliminated);
    expect(gameState.totalRingsEliminated).toBe(initialTotalEliminated);
    expect(player1AfterLines.territorySpaces).toBe(initialTerritory + requiredLength);

    // 2) Now process disconnected regions. This should collapse the
    // region at (5,5), eliminate the P2 stack there and one additional
    // ring/cap from P1 (self-elimination), with all eliminations
    // credited to Player 1.
    await engineAny.processDisconnectedRegions();

    const player1AfterTerritory = gameState.players.find((p) => p.playerNumber === 1)!;

    // Region space should now be a collapsed space for Player 1 and
    // the P2 stack there should be gone.
    const regionKey = positionToString(regionPos);
    expect(board.collapsedSpaces.get(regionKey)).toBe(1);
    expect(board.stacks.get(regionKey)).toBeUndefined();

    // Player 1's territory count should have grown by at least the
    // size of the region (1 space). Border markers are stubbed as []
    // so the exact increment is +1 beyond the line collapse.
    expect(player1AfterTerritory.territorySpaces).toBe(
      initialTerritory + requiredLength + regionTerritory.spaces.length
    );

    // Elimination accounting:
    // - Line processing: 0 rings eliminated (Option 2 default).
    // - Territory: 1 ring from P2 stack inside region + 1 ring (or cap)
    //   from P1 due to mandatory self-elimination.
    const eliminatedDeltaPlayer1 = player1AfterTerritory.eliminatedRings - initialEliminated;
    const totalEliminatedDelta = gameState.totalRingsEliminated - initialTotalEliminated;

    // In this concrete setup:
    // - Region contains a single-stack of height 1 (P2) ⇒ 1 ring eliminated.
    // - Player 1 has a single stack of height 2 outside the region; the
    //   default elimination path removes the entire cap (2 rings).
    // All three eliminated rings are credited to Player 1 (rules 12.2/9.2),
    // so we expect a delta of 3 here.
    expect(eliminatedDeltaPlayer1).toBe(3);
    expect(totalEliminatedDelta).toBe(3);

    // Sanity: ensure our spies were actually invoked.
    expect(findAllLinesSpy).toHaveBeenCalled();
    expect(findDisconnectedRegionsSpy).toHaveBeenCalled();
    expect(getBorderMarkersSpy).toHaveBeenCalled();
  });
});
