import { GameState, BoardState, Player, RingStack } from '../../src/shared/types/game';
import {
  evaluateHeuristicState,
  HEURISTIC_WEIGHTS_V1_BALANCED,
} from '../../src/shared/engine/heuristicEvaluation';

/**
 * Unit tests for the minimal TS-side heuristic evaluator used by
 * fallback AI and sandbox. These tests focus on:
 *
 * - Stack control & effective height signals
 * - Simple territory advantage
 * - Local vulnerability to taller adjacent enemy stacks
 * - Determinism for a fixed state and weight profile
 */

describe('heuristicEvaluation.evaluateHeuristicState', () => {
  function makeEmptyBoardState(): BoardState {
    return {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: 8,
      type: 'square8',
    } as unknown as BoardState;
  }

  function makeBaseGameState(overrides: Partial<GameState> = {}): GameState {
    const board = (overrides.board as BoardState) ?? makeEmptyBoardState();

    const players: Player[] = (overrides.players as Player[]) ?? [
      {
        id: 'p1',
        username: 'P1',
        type: 'ai',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as unknown as Player,
      {
        id: 'p2',
        username: 'P2',
        type: 'ai',
        playerNumber: 2,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 10,
        eliminatedRings: 0,
        territorySpaces: 0,
      } as unknown as Player,
    ];

    const base: GameState = {
      id: 'heuristic-test',
      boardType: 'square8',
      board,
      players,
      currentPhase: 'movement',
      currentPlayer: 1,
      moveHistory: [],
      timeControl: { initialTime: 600, increment: 0, type: 'rapid' },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 0,
      totalRingsEliminated: 0,
      victoryThreshold: 10,
      territoryVictoryThreshold: 33,
    } as unknown as GameState;

    return { ...base, ...overrides, board, players };
  }

  it('rewards controlling more stacks and effective height', () => {
    const board = makeEmptyBoardState();

    // Player 1: two modest stacks
    const s1: RingStack = {
      position: { x: 0, y: 0 },
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as RingStack;
    const s2: RingStack = {
      position: { x: 2, y: 2 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as RingStack;

    // Player 2: one stack
    const s3: RingStack = {
      position: { x: 5, y: 5 },
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    } as RingStack;

    board.stacks.set('0,0', s1);
    board.stacks.set('2,2', s2);
    board.stacks.set('5,5', s3);

    const state = makeBaseGameState({ board });

    const scoreP1 = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const scoreP2 = evaluateHeuristicState(state, 2, HEURISTIC_WEIGHTS_V1_BALANCED);

    // Player 1 should evaluate better due to more stacks and similar total height.
    expect(scoreP1).toBeGreaterThan(scoreP2);
  });

  it('rewards territory advantage for the evaluating player', () => {
    const state = makeBaseGameState();

    // Give player 1 meaningful territory; player 2 none.
    state.players = state.players.map((p) =>
      p.playerNumber === 1 ? { ...p, territorySpaces: 10 } : { ...p, territorySpaces: 0 }
    ) as Player[];

    const scoreP1 = evaluateHeuristicState(state, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const scoreP2 = evaluateHeuristicState(state, 2, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(scoreP1).toBeGreaterThan(scoreP2);
    // Territory should be beneficial for the player who owns it.
    expect(scoreP1).toBeGreaterThan(0);
  });

  it('penalises locally vulnerable stacks near taller enemy stacks', () => {
    // Setup vulnerable position: Player 1's stack is threatened by taller Player 2 stack
    const vulnerableBoard = makeEmptyBoardState();

    const vulnerableStack: RingStack = {
      position: { x: 3, y: 3 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as RingStack;

    const threateningStack: RingStack = {
      position: { x: 4, y: 3 },
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    } as RingStack;

    vulnerableBoard.stacks.set('3,3', vulnerableStack);
    vulnerableBoard.stacks.set('4,3', threateningStack);

    const vulnerableState = makeBaseGameState({ board: vulnerableBoard });

    // Setup safe position: Same stacks but not adjacent (no vulnerability)
    const safeBoard = makeEmptyBoardState();

    const safeStack: RingStack = {
      position: { x: 3, y: 3 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    } as RingStack;

    const distantStack: RingStack = {
      position: { x: 6, y: 6 }, // Far away, not threatening
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    } as RingStack;

    safeBoard.stacks.set('3,3', safeStack);
    safeBoard.stacks.set('6,6', distantStack);

    const safeState = makeBaseGameState({ board: safeBoard });

    const vulnerableScore = evaluateHeuristicState(
      vulnerableState,
      1,
      HEURISTIC_WEIGHTS_V1_BALANCED
    );
    const safeScore = evaluateHeuristicState(safeState, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    // The vulnerable position should score LOWER than the safe position
    // because vulnerability penalty reduces the evaluation
    expect(vulnerableScore).toBeLessThan(safeScore);
  });

  it('is deterministic for a fixed state and weight profile', () => {
    const board = makeEmptyBoardState();

    const stackA: RingStack = {
      position: { x: 1, y: 1 },
      rings: [1, 1, 1],
      stackHeight: 3,
      capHeight: 3,
      controllingPlayer: 1,
    } as RingStack;
    board.stacks.set('1,1', stackA);

    const state = makeBaseGameState({ board });

    const w = { ...HEURISTIC_WEIGHTS_V1_BALANCED };

    const s1 = evaluateHeuristicState(state, 1, w);
    const s2 = evaluateHeuristicState(state, 1, w);
    const s3 = evaluateHeuristicState(state, 1, w);

    expect(s1).toBe(s2);
    expect(s2).toBe(s3);
  });

  it('assigns large positive/negative values for terminal wins/losses', () => {
    const base = makeBaseGameState();

    const winState: GameState = {
      ...base,
      gameStatus: 'finished',
      winner: 1,
    } as GameState;

    const lossState: GameState = {
      ...base,
      gameStatus: 'finished',
      winner: 2,
    } as GameState;

    const drawState: GameState = {
      ...base,
      gameStatus: 'finished',
      winner: undefined,
    } as GameState;

    const winScore = evaluateHeuristicState(winState, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const lossScore = evaluateHeuristicState(lossState, 1, HEURISTIC_WEIGHTS_V1_BALANCED);
    const drawScore = evaluateHeuristicState(drawState, 1, HEURISTIC_WEIGHTS_V1_BALANCED);

    expect(winScore).toBeGreaterThan(10_000);
    expect(lossScore).toBeLessThan(-10_000);
    expect(drawScore).toBe(0);
  });
});
