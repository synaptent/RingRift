/**
 * Scenario Tests: Turn Sequence & Forced Elimination (Section 4, FAQ 15.2, FAQ 24)
 *
 * These tests exercise the extracted TurnEngine orchestration logic that backs
 * GameEngine.advanceGame. They focus on:
 *
 * - Handing the turn to the next player after territory_processing.
 * - Applying forced elimination when a player controls stacks but has no legal
 *   placements, movements, or captures (FAQ Q24).
 * - Skipping players who have no material at all (no stacks and no rings in hand)
 *   so the game can continue for remaining players.
 *
 * They deliberately use stubbed BoardManager / RuleEngine dependencies so that
 * the scenarios are isolated from low-level geometry and move generation.
 */
import {
  advanceGameForCurrentPlayer,
  PerTurnState,
  TurnEngineDeps,
  TurnEngineHooks,
} from '../../src/server/game/turn/TurnEngine';
import {
  GameState,
  BoardState,
  BoardType,
  TimeControl,
  Player,
  RingStack,
  positionToString,
  Position,
} from '../../src/shared/types/game';
import { createTestBoard, createTestPlayer } from '../utils/fixtures';

describe('GameEngine turn sequence & forced elimination scenarios (backend)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createGameState(players: Player[], board?: BoardState): GameState {
    const now = new Date();
    const boardState = board ?? createTestBoard(boardType);

    return {
      id: 'turn-sequence-scenario',
      boardType,
      board: boardState,
      players,
      currentPhase: 'territory_processing',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl,
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: false,
      maxPlayers: players.length,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
    };
  }

  test('Q24_forced_elimination_when_blocked_with_stacks_turn_engine', () => {
    // Player 2 controls a stack but has no legal placements, movements, or captures
    // (ringsInHand = 0, RuleEngine.getValidMoves = []). After territory_processing,
    // TurnEngine must invoke forced elimination on player 2 before handing the turn
    // to the next player with material.

    const player1 = createTestPlayer(1, { ringsInHand: 0 });
    const player2 = createTestPlayer(2, { ringsInHand: 0 });
    const players = [player1, player2];

    const board = createTestBoard(boardType);

    const stacksByPlayer: Record<number, RingStack[]> = {
      1: [],
      2: [
        ({
          position: { x: 0, y: 0 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        } as RingStack),
      ],
    };

    const boardManager: any = {
      getPlayerStacks: jest.fn((_board: BoardState, playerNumber: number) => {
        return stacksByPlayer[playerNumber] ?? [];
      }),
      isValidPosition: jest.fn(() => true),
      isCollapsedSpace: jest.fn(() => false),
      getMarker: jest.fn(() => undefined),
    };

    const ruleEngine: any = {
      // No placements, movements, or captures for any player in this scenario.
      getValidMoves: jest.fn(() => []),
      checkGameEnd: jest.fn(() => ({ isGameOver: false })),
    };

    // Seed the canonical BoardState with the same stack so that
    // applyForcedEliminationForPlayer can observe it and block all
    // outward rays so no legal movement/capture exists from (0,0).
    const p2Stack = stacksByPlayer[2][0];
    board.stacks.set(positionToString(p2Stack.position), p2Stack);

    const blockers: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    for (const pos of blockers) {
      board.collapsedSpaces.set(positionToString(pos), 0);
    }

    const deps: TurnEngineDeps = { boardManager, ruleEngine };

    const eliminatePlayerRingOrCap = jest.fn();
    const endGame: TurnEngineHooks['endGame'] = jest.fn((_winner?: number, _reason?: string) => {
      return {
        success: true,
        gameResult: {
          reason: 'game_completed',
          finalScore: {
            ringsEliminated: {},
            territorySpaces: {},
            ringsRemaining: {},
          },
        },
      };
    });

    const hooks: TurnEngineHooks = { eliminatePlayerRingOrCap, endGame };

    const gameState = createGameState(players, board);
    const initialTurnState: PerTurnState = {
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    };

    const afterTurnState = advanceGameForCurrentPlayer(gameState, initialTurnState, deps, hooks);

    // Forced elimination must have been applied to player 2 (FAQ Q24):
    // the only stack at (0,0) should have been removed and at least
    // one ring credited as eliminated.
    const finalBoard = gameState.board;
    const finalP2 = gameState.players.find((p) => p.playerNumber === 2)!;

    expect(finalBoard.stacks.get(positionToString({ x: 0, y: 0 }))).toBeUndefined();
    expect(finalP2.eliminatedRings).toBeGreaterThan(0);

    // After elimination and turn handoff, the next interactive turn belongs
    // to player 2 again (the only player with material) in the movement phase.
    expect(gameState.currentPlayer).toBe(2);
    expect(gameState.currentPhase).toBe('movement');

    // Per-turn placement bookkeeping is reset for the new turn.
    expect(afterTurnState).toEqual({
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    });
  });

  test('Players with no stacks and no rings are skipped when starting a new turn', () => {
    // Section 4 / FAQ 15.2: when advancing from territory_processing, players who
    // have no stacks and no rings in hand cannot take any actions and should be
    // skipped so the turn is given to a player who can act.

    const player1 = createTestPlayer(1, { ringsInHand: 0 });
    const player2 = createTestPlayer(2, { ringsInHand: 0 });
    const player3 = createTestPlayer(3, { ringsInHand: 5 });
    const players = [player1, player2, player3];

    const board = createTestBoard(boardType);

    const stacksByPlayer: Record<number, { x: number; y: number; z?: number }[]> = {
      1: [],
      2: [],
      3: [],
    };

    const boardManager: any = {
      getPlayerStacks: jest.fn((_board: BoardState, playerNumber: number) => {
        return stacksByPlayer[playerNumber] ?? [];
      }),
      isValidPosition: jest.fn(() => true),
      isCollapsedSpace: jest.fn(() => false),
      getMarker: jest.fn(() => undefined),
    };

    const ruleEngine: any = {
      getValidMoves: jest.fn(() => []),
      checkGameEnd: jest.fn(() => ({ isGameOver: false })),
    };

    const deps: TurnEngineDeps = { boardManager, ruleEngine };

    const eliminatePlayerRingOrCap = jest.fn();
    const endGame: TurnEngineHooks['endGame'] = jest.fn((_winner?: number, _reason?: string) => {
      return {
        success: true,
        gameResult: {
          reason: 'game_completed',
          finalScore: {
            ringsEliminated: {},
            territorySpaces: {},
            ringsRemaining: {},
          },
        },
      };
    });

    const hooks: TurnEngineHooks = { eliminatePlayerRingOrCap, endGame };

    const gameState = createGameState(players, board);
    const initialTurnState: PerTurnState = {
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    };

    const afterTurnState = advanceGameForCurrentPlayer(gameState, initialTurnState, deps, hooks);

    // Player 2 has no stacks and no rings in hand; they must be skipped.
    // Player 3 has rings in hand and no stacks, so their turn begins in
    // ring_placement phase.
    expect(gameState.currentPlayer).toBe(3);
    expect(gameState.currentPhase).toBe('ring_placement');

    // No forced elimination occurs in this scenario.
    expect(eliminatePlayerRingOrCap).not.toHaveBeenCalled();
    expect(afterTurnState).toEqual({
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    });
  });

  test('Rules_4_2_three_player_skip_and_forced_elimination_backend', () => {
    // Section 4 / FAQ 15.2, 24: when advancing from territory_processing
    // in a three-player game, players with no stacks and no rings are
    // skipped, and a blocked player who still controls stacks but has no
    // legal actions is subject to forced elimination before the turn
    // proceeds.

    const player1 = createTestPlayer(1, { ringsInHand: 0 });
    const player2 = createTestPlayer(2, { ringsInHand: 0 });
    const player3 = createTestPlayer(3, { ringsInHand: 0 });
    const players = [player1, player2, player3];

    const board = createTestBoard(boardType);

    const stacksByPlayer: Record<number, RingStack[]> = {
      1: [],
      2: [
        ({
          position: { x: 0, y: 0 },
          rings: [2],
          stackHeight: 1,
          capHeight: 1,
          controllingPlayer: 2,
        } as RingStack),
      ], // Player 2 controls a single stack but has no legal moves.
      3: [],
    };

    const boardManager: any = {
      getPlayerStacks: jest.fn((_board: BoardState, playerNumber: number) => {
        return stacksByPlayer[playerNumber] ?? [];
      }),
      isValidPosition: jest.fn(() => true),
      isCollapsedSpace: jest.fn(() => false),
      getMarker: jest.fn(() => undefined),
    };

    const ruleEngine: any = {
      // No legal placements, movements, or captures are available in
      // this simplified scenario for any player.
      getValidMoves: jest.fn(() => []),
      checkGameEnd: jest.fn(() => ({ isGameOver: false })),
    };

    // Seed the canonical BoardState with the same stack for Player 2
    // and block all immediate outward rays so player 2 has no legal
    // movements or captures from (0,0).
    const p2Stack = stacksByPlayer[2][0];
    board.stacks.set(positionToString(p2Stack.position), p2Stack);

    const blockers: Position[] = [
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ];
    for (const pos of blockers) {
      board.collapsedSpaces.set(positionToString(pos), 0);
    }

    const deps: TurnEngineDeps = { boardManager, ruleEngine };

    const eliminatePlayerRingOrCap = jest.fn();
    const endGame: TurnEngineHooks['endGame'] = jest.fn((_winner?: number, _reason?: string) => {
      return {
        success: true,
        gameResult: {
          reason: 'game_completed',
          finalScore: {
            ringsEliminated: {},
            territorySpaces: {},
            ringsRemaining: {},
          },
        },
      };
    });

    const hooks: TurnEngineHooks = { eliminatePlayerRingOrCap, endGame };

    const gameState = createGameState(players, board);
    const initialTurnState: PerTurnState = {
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    };

    const afterTurnState = advanceGameForCurrentPlayer(gameState, initialTurnState, deps, hooks);

    // Players 1 and 3 have no stacks and no rings, so they cannot act.
    // Player 2 controls a stack but has no legal actions and must be
    // forced to eliminate a ring/cap as per FAQ Q24. The stack at
    // (0,0) should have been reduced or removed for player 2, and at
    // least one ring should be credited as eliminated.
    const finalBoard = gameState.board;
    const finalP2 = gameState.players.find((p) => p.playerNumber === 2)!;

    expect(finalBoard.stacks.get(positionToString({ x: 0, y: 0 }))).toBeUndefined();
    expect(finalP2.eliminatedRings).toBeGreaterThan(0);

    // After forced elimination, the interactive turn remains with
    // player 2 in the movement phase, as in the two-player scenario.
    expect(gameState.currentPlayer).toBe(2);
    expect(gameState.currentPhase).toBe('movement');

    expect(afterTurnState).toEqual({
      hasPlacedThisTurn: false,
      mustMoveFromStackKey: undefined,
    });
  });
});
