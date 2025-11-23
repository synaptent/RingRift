import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  TimeControl,
  positionToString,
} from '../../src/shared/types/game';

/**
 * Scenario Tests: GameEngine victory scenarios
 *
 * These backend-focused scenarios complement the client-local
 * `ClientSandboxEngine.victory` tests by exercising the RuleEngine
 * victory checks in a small, explicit way.
 *
 * Rules/FAQ references:
 * - ringrift_complete_rules.md §13.1–13.2 (ring-elimination, territory-control)
 * - Compact rules §7.1–7.2
 * - FAQ Q11, Q18, Q21 (high-level victory examples)
 */

describe('GameEngine victory scenarios (Section 13.1–13.2; FAQ 11, 18, 21)', () => {
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

  it('Rules_13_1_ring_elimination_victory_backend', () => {
    // Rules reference:
    // - §13.1 / compact §7.1: A player wins immediately when their
    //   eliminated ring total reaches victoryThreshold (strictly more
    //   than half of totalRingsInPlay).

    const engine = new GameEngine('victory-ring', boardType, createPlayers(), timeControl, false);
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    const threshold = gameState.victoryThreshold;

    // Simulate P1 having already eliminated exactly threshold rings.
    p1.eliminatedRings = threshold;
    gameState.totalRingsEliminated = threshold;
    gameState.board.eliminatedRings[1] = threshold;

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(1);
    // Reason string mirrors the compact rules 7.1 label and matches
    // the sandbox victory tests.
    expect(endCheck.reason).toBe('ring_elimination');
  });

  it('Rules_13_2_territory_control_victory_backend', () => {
    // Rules reference:
    // - §13.2 / compact §7.2: A player wins by territory-control when
    //   they control strictly more than half of the board spaces via
    //   collapsed territory (territoryVictoryThreshold).

    const engine = new GameEngine(
      'victory-territory',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    const threshold = gameState.territoryVictoryThreshold;

    // Directly set P1's territorySpaces to the threshold. In a real
    // game this would come from line/territory processing, but for the
    // purposes of this rules check we only need to satisfy the
    // invariant used by RuleEngine.
    p1.territorySpaces = threshold;

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(1);
    expect(endCheck.reason).toBe('territory_control');
  });

  it('Rules_13_3_last_player_standing_via_territory_tiebreak_backend', () => {
    // Rules reference (current implementation):
    // - §13.3: last-player-standing is effectively realised, when
    //   nobody has stacks or rings in hand, by the structural
    //   terminality tie-breakers (territory > eliminated rings).
    //
    // Scenario:
    // - No stacks remain and no rings are in hand.
    // - Player 1 controls some territory; Player 2 controls none.
    // - Nobody has reached strict victory thresholds, but
    //   RuleEngine.checkGameEnd should still award victory to Player 1
    //   via the fallback territory-control tie-break.

    const engine = new GameEngine(
      'victory-last-player-standing',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    // Structural terminality preconditions.
    gameState.board.stacks.clear();
    gameState.players.forEach((p) => {
      p.ringsInHand = 0;
      p.eliminatedRings = 0;
      p.territorySpaces = 0;
    });

    // Give Player 1 a small amount of territory; Player 2 remains at 0.
    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    p1.territorySpaces = 3;

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(1);
    expect(endCheck.reason).toBe('territory_control');
  });

  it('Rules_13_4_stalemate_tiebreak_eliminated_rings_backend', () => {
    // Rules reference (current implementation):
    // - §13.4–13.5: when structural terminality is reached and
    //   territory is tied, eliminated rings are the next tiebreaker.
    //
    // Scenario:
    // - No stacks and no rings in hand for any player.
    // - Territory is tied (0 for both players).
    // - Player 1 has eliminated more rings than Player 2.
    // - RuleEngine.checkGameEnd should award victory to Player 1 with
    //   reason 'ring_elimination'.

    const engine = new GameEngine(
      'victory-stalemate-tiebreak',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    gameState.board.stacks.clear();
    gameState.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    const p2 = gameState.players.find((p) => p.playerNumber === 2)!;

    p1.eliminatedRings = 4;
    p2.eliminatedRings = 2;

    gameState.totalRingsEliminated = p1.eliminatedRings + p2.eliminatedRings;

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(1);
    expect(endCheck.reason).toBe('ring_elimination');
  });

  it('Rules_13_5_stalemate_tiebreak_markers_backend', () => {
    // Rules reference (§13.4 / 16.9.4.5): when structural terminality is
    // reached and both territory and eliminated rings are tied, remaining
    // markers act as the next tiebreaker in the stalemate ladder.
    //
    // Scenario:
    // - No stacks and no rings in hand for any player.
    // - Territory is tied (0 for both players).
    // - Eliminated rings are tied.
    // - Player 1 has more markers on the board than Player 2.
    // - RuleEngine.checkGameEnd should award victory to Player 1 with
    //   reason 'last_player_standing'.

    const engine = new GameEngine(
      'victory-stalemate-markers',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    gameState.board.stacks.clear();
    gameState.board.markers.clear();

    gameState.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 2;
    });

    // Player 1: two markers; Player 2: one marker.
    gameState.board.markers.set('0,0', {
      player: 1,
      position: { x: 0, y: 0 },
      type: 'regular',
    });
    gameState.board.markers.set('1,0', {
      player: 1,
      position: { x: 1, y: 0 },
      type: 'regular',
    });
    gameState.board.markers.set('0,1', {
      player: 2,
      position: { x: 0, y: 1 },
      type: 'regular',
    });

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(1);
    expect(endCheck.reason).toBe('last_player_standing');
  });

  it('Rules_13_6_stalemate_last_actor_backend', () => {
    // Rules reference (§13.4 / 16.9.4.5): if territory, eliminated rings,
    // and markers are all tied at structural terminality, the last player
    // to complete a valid turn action wins as the final tiebreaker.
    //
    // Scenario:
    // - No stacks and no rings in hand for any player.
    // - Territory, eliminated rings, and markers are all tied at 0.
    // - With players [1,2] and currentPlayer = 1, the previous player in
    //   turn order (Player 2) is treated as the last actor.

    const engine = new GameEngine(
      'victory-stalemate-last-actor',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    gameState.board.stacks.clear();
    gameState.board.markers.clear();

    gameState.players.forEach((p) => {
      p.ringsInHand = 0;
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    // Ensure canonical ordering and currentPlayer for the fallback.
    gameState.currentPlayer = 1;

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(2);
    expect(endCheck.reason).toBe('last_player_standing');
  });

  it('Rules_13_7_bare_board_with_legal_reentry_not_terminal_backend', () => {
    // Scenario:
    // - No stacks on the board.
    // - Players still have rings in hand.
    // - At least one legal ring placement exists for some player under the
    //   no-dead-placement rule.
    //
    // Expected:
    // - RuleEngine.checkGameEnd should report that the game is *not*
    //   over, since re-entry into play is still possible.

    const engine = new GameEngine(
      'victory-bare-board-nonterminal',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    // Ensure structural bare-board preconditions.
    gameState.board.stacks.clear();

    // Make sure each player has at least one ring in hand and no prior
    // progress towards other victory conditions.
    gameState.players.forEach((p) => {
      if (p.ringsInHand <= 0) {
        p.ringsInHand = 1;
      }
      p.territorySpaces = 0;
      p.eliminatedRings = 0;
    });

    const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(false);
    expect(endCheck.winner).toBeUndefined();
    expect(endCheck.reason).toBeUndefined();
  });

  it('Rules_13_8_global_stalemate_bare_board_rings_in_hand_elimination_tiebreak_backend', () => {
    // Scenario:
    // - No stacks on the board, but both players still have rings in hand.
    // - The board is structurally terminal because *all* spaces are collapsed
    //   territory, so no legal placements exist for any player under the
    //   no-dead-placement rule.
    //
    // Under §13.4 / FAQ Q11, rings remaining in hand are then treated as
    // eliminated (hand → E) for tie-breaking purposes. The player with the
    // larger (eliminated + hand) total should win via ring_elimination.

    const engine = new GameEngine(
      'victory-bare-board-global-stalemate',
      boardType,
      createPlayers(),
      timeControl,
      false
    );
    const engineAny: any = engine;
    const gameState: GameState = engineAny.gameState as GameState;

    gameState.board.stacks.clear();
    gameState.board.markers.clear();

    const p1 = gameState.players.find((p) => p.playerNumber === 1)!;
    const p2 = gameState.players.find((p) => p.playerNumber === 2)!;

    // No territory or previously eliminated rings for either player so that
    // the elimination tiebreak is decided purely by (eliminated + hand).
    p1.territorySpaces = 0;
    p2.territorySpaces = 0;

    p1.eliminatedRings = 0;
    p2.eliminatedRings = 0;

    // Give Player 1 more rings in hand than Player 2.
    p1.ringsInHand = 3;
    p2.ringsInHand = 1;

    // Make every space on the board collapsed territory so that no legal
    // placements exist for any player under the shared no-dead-placement
    // rule. This models a true global bare-board stalemate that
    // evaluateVictory can detect from GameState alone.
    const board = gameState.board;
    for (let x = 0; x < board.size; x++) {
      for (let y = 0; y < board.size; y++) {
        const key = positionToString({ x, y });
        board.collapsedSpaces.set(key, 1);
      }
    }

    const ruleEngine: any = engineAny.ruleEngine;
    const endCheck = ruleEngine.checkGameEnd(gameState);

    expect(endCheck.isGameOver).toBe(true);
    expect(endCheck.winner).toBe(1);
    expect(endCheck.reason).toBe('ring_elimination');
  });
});
