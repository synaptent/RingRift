import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BOARD_CONFIGS,
  BoardType,
  GameState,
  Player,
  TimeControl,
} from '../../src/shared/types/game';

describe('GameEngine pie rule (swap_sides meta-move)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  const basePlayers: Player[] = [
    {
      id: 'p1',
      username: 'Alice',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: BOARD_CONFIGS[boardType].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Bob',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: BOARD_CONFIGS[boardType].ringsPerPlayer,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  function createActiveGame(options: { swapRuleEnabled?: boolean } = {}): {
    engine: GameEngine;
    state: GameState;
  } {
    const { swapRuleEnabled = true } = options;

    const engine = new GameEngine(
      'swap-rule',
      boardType,
      basePlayers,
      timeControl,
      false,
      undefined,
      undefined,
      { swapRuleEnabled }
    );
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    // The GameEngine constructor sets isReady based on player type (AI vs human).
    // For human players, isReady defaults to false. We need to mark them ready
    // before calling startGame() so the game can transition to 'active'.
    state.players.forEach((p: Player) => {
      p.isReady = true;
    });

    // Properly start the game via the GameEngine API rather than manually
    // setting gameStatus. This ensures proper initialization for both
    // orchestrator and legacy paths.
    const started = engine.startGame();
    if (!started) {
      throw new Error('Failed to start game for swap rule test');
    }

    return { engine, state };
  }

  it('rejects swap_sides before Player 1 has moved', async () => {
    const { engine } = createActiveGame();

    const result = await engine.makeMove({
      type: 'swap_sides',
      player: 2,
      to: { x: 0, y: 0 },
    } as any);

    expect(result.success).toBe(false);
  });

  it('applies swap_sides by swapping player seats but not board geometry', async () => {
    const { engine } = createActiveGame();
    const engineAny: any = engine;
    const stateBefore: GameState = engineAny.gameState as GameState;

    // Give Player 1 a visible stack and some eliminated rings/territory so we
    // can verify stats move with the seat.
    const board = stateBefore.board;
    board.stacks.set('0,0', {
      position: { x: 0, y: 0 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as any);
    stateBefore.players[0].eliminatedRings = 2;
    stateBefore.players[0].territorySpaces = 3;

    // Give players distinct clocks so we can verify time budgets move with
    // the side (seat) rather than the original user id.
    stateBefore.players[0].timeRemaining = 12345;
    stateBefore.players[1].timeRemaining = 67890;

    // Complete Player 1's first turn with a trivial placement+movement.
    let p1Moves = engine.getValidMoves(1);
    const place = p1Moves.find((m) => m.type === 'place_ring');
    expect(place).toBeDefined();

    await engine.makeMove({
      type: 'place_ring',
      player: 1,
      to: place!.to,
      placementCount: place!.placementCount ?? 1,
    } as any);

    p1Moves = engine.getValidMoves(1);
    const move = p1Moves.find(
      (m) => m.type === 'move_stack' || m.type === 'move_stack' || m.type === 'overtaking_capture'
    );
    expect(move).toBeDefined();

    await engine.makeMove({
      type: move!.type,
      player: 1,
      from: move!.from,
      to: move!.to,
      captureTarget: (move as any).captureTarget,
    } as any);

    // Snapshot board geometry immediately before swap for comparison.
    const stacksBefore = Array.from(board.stacks.keys()).sort();
    const markersBefore = Array.from(board.markers.keys()).sort();
    const collapsedBefore = Array.from(board.collapsedSpaces.keys()).sort();

    // Player 2 turn: request swap_sides directly; engine should accept.
    const result = await engine.makeMove({
      type: 'swap_sides',
      player: 2,
      to: { x: 0, y: 0 },
    } as any);

    expect(result.success).toBe(true);
    const after = engine.getGameState();

    // Current player should remain Player 2 after swap (it is still P2's turn).
    expect(after.currentPlayer).toBe(2);

    // Player numbers 1 and 2 still exist, but their user ids/usernames are swapped.
    const p1After = after.players.find((p: Player) => p.playerNumber === 1)!;
    const p2After = after.players.find((p: Player) => p.playerNumber === 2)!;

    expect(p1After.username).toBe('Bob');
    expect(p2After.username).toBe('Alice');

    // Stats attached to seats move with the seat, not with the user id.
    expect(p1After.eliminatedRings).toBe(2);
    expect(p1After.territorySpaces).toBe(3);

    // Clocks move with the side: the player who takes over the opening
    // inherits its remaining time budget.
    expect(p1After.timeRemaining).toBe(12345);
    expect(p2After.timeRemaining).toBe(67890);

    // Board geometry should be unchanged by swap_sides.
    const stacksAfter = Array.from(after.board.stacks.keys()).sort();
    const markersAfter = Array.from(after.board.markers.keys()).sort();
    const collapsedAfter = Array.from(after.board.collapsedSpaces.keys()).sort();

    expect(stacksAfter).toEqual(stacksBefore);
    expect(markersAfter).toEqual(markersBefore);
    expect(collapsedAfter).toEqual(collapsedBefore);

    // A swap_sides move should be recorded as the last history entry.
    const lastMove = after.moveHistory[after.moveHistory.length - 1];
    expect(lastMove.type).toBe('swap_sides');
    expect(lastMove.player).toBe(2);
  });

  it('does not offer swap_sides when swapRuleEnabled is false', async () => {
    const { engine } = createActiveGame({ swapRuleEnabled: false });
    const engineAny: any = engine;
    const stateBefore: GameState = engineAny.gameState as GameState;

    // Complete Player 1's first turn with a trivial placement+movement.
    let p1Moves = engine.getValidMoves(1);
    const place = p1Moves.find((m) => m.type === 'place_ring');
    expect(place).toBeDefined();

    await engine.makeMove({
      type: 'place_ring',
      player: 1,
      to: place!.to,
      placementCount: place!.placementCount ?? 1,
    } as any);

    p1Moves = engine.getValidMoves(1);
    const move = p1Moves.find(
      (m) => m.type === 'move_stack' || m.type === 'move_stack' || m.type === 'overtaking_capture'
    );
    expect(move).toBeDefined();

    await engine.makeMove({
      type: move!.type,
      player: 1,
      from: move!.from,
      to: move!.to,
      captureTarget: (move as any).captureTarget,
    } as any);

    // With swap disabled, the internal gate and getValidMoves must not offer it.
    expect(stateBefore.rulesOptions?.swapRuleEnabled).toBe(false);
    expect(engineAny.shouldOfferSwapSidesMetaMove()).toBe(false);

    const p2Moves = engine.getValidMoves(2);
    expect(p2Moves.some((m) => m.type === 'swap_sides')).toBe(false);
  });

  it('offers swap_sides exactly once after P1 first turn in active 2p games', async () => {
    const { engine } = createActiveGame();
    const engineAny: any = engine;

    // Before any P1 move, pie rule must not be available.
    expect(engineAny.shouldOfferSwapSidesMetaMove()).toBe(false);

    // Complete Player 1's first turn with a trivial placement+movement.
    let p1Moves = engine.getValidMoves(1);
    const place = p1Moves.find((m) => m.type === 'place_ring');
    expect(place).toBeDefined();

    await engine.makeMove({
      type: 'place_ring',
      player: 1,
      to: place!.to,
      placementCount: place!.placementCount ?? 1,
    } as any);

    p1Moves = engine.getValidMoves(1);
    const move = p1Moves.find(
      (m) => m.type === 'move_stack' || m.type === 'move_stack' || m.type === 'overtaking_capture'
    );
    expect(move).toBeDefined();

    await engine.makeMove({
      type: move!.type,
      player: 1,
      from: move!.from,
      to: move!.to,
      captureTarget: (move as any).captureTarget,
    } as any);

    // Immediately after P1's first turn, in an active 2p game, swap_sides
    // should be offered exactly once to Player 2.
    expect(engineAny.shouldOfferSwapSidesMetaMove()).toBe(true);
    const p2MovesBefore = engine.getValidMoves(2);
    expect(p2MovesBefore.some((m) => m.type === 'swap_sides')).toBe(true);

    // Apply the swap once.
    const swapMove = p2MovesBefore.find((m) => m.type === 'swap_sides')!;
    const result = await engine.makeMove({
      type: 'swap_sides',
      player: 2,
      to: swapMove.to,
    } as any);
    expect(result.success).toBe(true);

    // After one successful swap, the meta-move must not be offered again.
    expect(engineAny.shouldOfferSwapSidesMetaMove()).toBe(false);
    const p2MovesAfter = engine.getValidMoves(2);
    expect(p2MovesAfter.some((m) => m.type === 'swap_sides')).toBe(false);
  });

  it('never offers swap_sides in non-active or non-2p games', () => {
    // Non-active game: even with 2 players and swapRuleEnabled, the meta-move
    // must not be surfaced.
    {
      const { engine, state } = createActiveGame();
      const engineAny: any = engine;
      state.gameStatus = 'completed';
      expect(engineAny.shouldOfferSwapSidesMetaMove()).toBe(false);
      const moves = engine.getValidMoves(2);
      expect(moves.some((m) => m.type === 'swap_sides')).toBe(false);
    }

    // 3-player game: swap_sides is undefined regardless of rulesOptions.
    {
      const threePlayers: Player[] = [
        {
          id: 'p1',
          username: 'P1',
          type: 'human',
          playerNumber: 1,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: BOARD_CONFIGS[boardType].ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: 'P2',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: BOARD_CONFIGS[boardType].ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p3',
          username: 'P3',
          type: 'human',
          playerNumber: 3,
          isReady: true,
          timeRemaining: timeControl.initialTime * 1000,
          ringsInHand: BOARD_CONFIGS[boardType].ringsPerPlayer,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const engine = new GameEngine(
        'swap-rule-3p',
        boardType,
        threePlayers,
        timeControl,
        false,
        undefined,
        undefined,
        { swapRuleEnabled: true }
      );
      const engineAny: any = engine;
      const state: GameState = (engineAny.gameState as GameState)!;

      // Mark all players as ready before calling startGame()
      state.players.forEach((p: Player) => {
        p.isReady = true;
      });

      // Properly start the game for 3-player test
      const started = engine.startGame();
      expect(started).toBe(true);

      // Set currentPlayer to 2 to test gating condition
      state.currentPlayer = 2;

      expect(engineAny.shouldOfferSwapSidesMetaMove()).toBe(false);
      const moves = engine.getValidMoves(2);
      expect(moves.some((m) => m.type === 'swap_sides')).toBe(false);
    }
  });
});
