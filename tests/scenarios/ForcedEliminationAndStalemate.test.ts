import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  Position,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';
import { computeProgressSnapshot } from '../../src/shared/engine/core';

/**
 * Scenario Tests: Forced Elimination & Stalemate (FAQ Q24, Q11)
 *
 * These scenarios exercise backend GameEngine helpers that mirror the
 * rules text for:
 *
 * - Q24: "What happens if I control stacks but have no valid placement,
 *   movement, or capture options on my turn?" (Forced elimination.)
 * - Q11 / Section 13.4: Stalemate with rings in hand when no moves or
 *   placements are possible and no stacks remain on the board.
 */

describe('Scenario: Forced Elimination & Stalemate (backend)', () => {
  const boardType: BoardType = 'square8';
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
        ringsInHand: 0,
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
        ringsInHand: 0,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
  }

  test('4.4.1_forced_elimination_when_blocked_with_stacks_backend', () => {
    // Rules reference:
    // - Section 4.4 / FAQ Q24: At the beginning of a player's turn, if
    //   they have no legal placement, movement, or capture but still
    //   control stacks, they must eliminate the entire cap of one of
    //   their stacks (forced elimination).
    // - Section 13.5: Progress invariant S = markers + collapsed + eliminated
    //   must be non-decreasing and strictly increases on forced elimination.
    //
    // Scenario:
    // - Player 1 is the current player, gameStatus is active.
    // - Player 1 controls a single stack at (0,0) with capHeight 2.
    // - All outward rays from (0,0) are blocked by collapsed spaces,
    //   so there are no legal moves or captures.
    // - Both players have ringsInHand = 0, so no placements are legal.
    // - resolveBlockedStateForCurrentPlayerForTesting() should apply
    //   forced elimination, removing Player 1's cap and increasing
    //   eliminatedRings and totalRingsEliminated, and increasing S.

    const players = createPlayers();
    const engine = new GameEngine('forced-elim-q24', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;
    const board = gameState.board;
    const boardManager: any = engineAny.boardManager;

    gameState.gameStatus = 'active';
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    const p2 = gameState.players.find((p) => p.playerNumber === 2)!;
    p1.ringsInHand = 0;
    p2.ringsInHand = 0;

    // Single Player 1 stack at (0,0) with capHeight == stackHeight.
    const stackPos: Position = { x: 0, y: 0 };
    const rings = [1, 1];
    const stack = {
      position: stackPos,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: 1,
    };
    boardManager.setStack(stackPos, stack, board);

    // Block all immediate outward rays so no legal movement/capture
    // exists from (0,0).
    const blockers: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    for (const pos of blockers) {
      board.collapsedSpaces.set(positionToString(pos), 0);
    }

    const initialP1Eliminated = p1.eliminatedRings;
    const initialTotalEliminated = gameState.totalRingsEliminated;
    const progressBefore = computeProgressSnapshot(gameState);

    engine.resolveBlockedStateForCurrentPlayerForTesting();

    const finalState = engine.getGameState();
    const finalP1 = finalState.players.find((p: Player) => p.playerNumber === 1)!;
    const progressAfter = computeProgressSnapshot(finalState);

    // Forced elimination must have increased P1's eliminatedRings and
    // the global total. In this setup we expect the entire cap (2 rings)
    // to be removed from the only stack.
    expect(finalP1.eliminatedRings).toBe(initialP1Eliminated + 2);
    expect(finalState.totalRingsEliminated).toBe(initialTotalEliminated + 2);

    // The original stack at (0,0) should be gone.
    const finalStack = finalState.board.stacks.get(positionToString(stackPos));
    expect(finalStack).toBeUndefined();

    // Progress invariant S must strictly increase when forced elimination fires.
    expect(progressAfter.S).toBeGreaterThan(progressBefore.S);
  });

  test('13.5.1_structural_stalemate_converts_rings_in_hand_to_eliminated_backend', () => {
    // Rules reference:
    // - Section 13.4 / FAQ Q11: In global stalemate when no legal
    //   actions (placements, movements, captures, forced eliminations)
    //   remain and no stacks are on the board, any rings remaining in
    //   hand are counted as eliminated rings for tie-breaking.
    // - Section 13.5: Progress & termination invariant – once no player
    //   has placements/moves/captures/forced eliminations, the game must
    //   be structurally terminal and S should increase as rings in hand
    //   are converted to eliminated rings.
    //
    // Scenario:
    // - No stacks on the board for any player.
    // - Both players have rings in hand, but we treat the state as
    //   structurally terminal via resolveBlockedStateForCurrentPlayerForTesting.
    // - The helper should convert ringsInHand → eliminatedRings for
    //   each player and mark the game as completed once checkGameEnd
    //   resolves the stalemate, with S strictly increasing.

    const players: Player[] = [
      {
        id: 'p1',
        username: 'Player1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 3,
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
        ringsInHand: 5,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const engine = new GameEngine('stalemate-q11', boardType, players, timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    gameState.gameStatus = 'active';
    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    // Ensure board has no stacks; we rely on the helper's branch that
    // treats stackless boards as structurally terminal. To also prevent
    // any legal placements, mark all board positions as collapsed so
    // RuleEngine.getValidMoves cannot generate place_ring moves.
    gameState.board.stacks.clear();
    for (let x = 0; x < 8; x++) {
      for (let y = 0; y < 8; y++) {
        gameState.board.collapsedSpaces.set(positionToString({ x, y }), 0);
      }
    }

    const progressBefore = computeProgressSnapshot(gameState);

    engine.resolveBlockedStateForCurrentPlayerForTesting();

    const finalState = engine.getGameState();
    const finalP1 = finalState.players.find((p: Player) => p.playerNumber === 1)!;
    const finalP2 = finalState.players.find((p: Player) => p.playerNumber === 2)!;
    const progressAfter = computeProgressSnapshot(finalState);

    // Rings in hand should have been converted to eliminated rings.
    expect(finalP1.ringsInHand).toBe(0);
    expect(finalP2.ringsInHand).toBe(0);
    expect(finalP1.eliminatedRings).toBe(3);
    expect(finalP2.eliminatedRings).toBe(5);

    // Global total should match the sum of all eliminated rings.
    expect(finalState.totalRingsEliminated).toBe(3 + 5);

    // The game should no longer be active after stalemate resolution.
    expect(finalState.gameStatus).toBe('completed');

    // Progress invariant S must strictly increase as rings in hand are
    // converted into eliminated rings at structural terminality.
    expect(progressAfter.S).toBeGreaterThan(progressBefore.S);
  });
});
