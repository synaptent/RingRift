import type { SandboxAIHooks } from '../../src/client/sandbox/sandboxAI';
import { maybeRunAITurnSandbox } from '../../src/client/sandbox/sandboxAI';
import type { GameState, Move, RingStack } from '../../src/shared/types/game';
import { createTestGameState } from '../utils/fixtures';

describe('sandboxAI forced_elimination phase', () => {
  it('applies a forced_elimination move when available for the current AI player', async () => {
    const baseState: GameState = createTestGameState({ boardType: 'square8' });

    let currentState: GameState = {
      ...baseState,
      players: baseState.players.map((p) =>
        p.playerNumber === 1 ? { ...p, type: 'ai' } : { ...p, type: 'human' }
      ),
      currentPlayer: 1,
      currentPhase: 'forced_elimination',
      gameStatus: 'active',
    };

    const forcedEliminationCandidate: Move = {
      id: 'fe-1',
      type: 'forced_elimination',
      player: 1,
      moveNumber: 1,
      timestamp: new Date(),
      thinkTime: 0,
    } as Move;

    const applyCanonicalMove = jest.fn(async (_move: Move) => {
      // No-op stub: this test only asserts that maybeRunAITurnSandbox
      // selects and attempts to apply a forced_elimination move.
    });

    let lastAIMove: Move | null = null;

    const hooks: SandboxAIHooks = {
      getPlayerStacks: () => [],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      getValidMovesForCurrentPlayer: () => [forcedEliminationCandidate],
      createHypotheticalBoardWithPlacement: (board) => board,
      tryPlaceRings: async () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {
        // no-op
      },
      appendHistoryEntry: () => {
        // no-op for this focused test
      },
      getGameState: () => currentState,
      setGameState: (state: GameState) => {
        currentState = state;
      },
      setLastAIMove: (move: Move | null) => {
        lastAIMove = move;
      },
      setSelectedStackKey: () => {
        // selection not relevant for this test
      },
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove,
      hasPendingTerritorySelfElimination: () => false,
      hasPendingLineRewardElimination: () => false,
      canCurrentPlayerSwapSides: () => false,
      applySwapSidesForCurrentPlayer: () => false,
    };

    const rng = () => 0.5;

    await maybeRunAITurnSandbox(hooks, rng);

    expect(applyCanonicalMove).toHaveBeenCalledTimes(1);
    const appliedMove = applyCanonicalMove.mock.calls[0][0] as Move;

    expect(appliedMove.type).toBe('forced_elimination');
    expect(appliedMove.player).toBe(1);

    expect(lastAIMove).not.toBeNull();
    if (!lastAIMove) {
      throw new Error('Expected lastAIMove to be non-null after forced_elimination move');
    }
    const lastMove = lastAIMove as Move;
    expect(lastMove.type).toBe('forced_elimination');
    expect(lastMove.player).toBe(1);
  });

  it('applies a forced_elimination move when available for currentPlayer 2 on a hex board', async () => {
    const baseState: GameState = createTestGameState({ boardType: 'hexagonal' });

    let currentState: GameState = {
      ...baseState,
      players: baseState.players.map((p) =>
        p.playerNumber === 2 ? { ...p, type: 'ai' } : { ...p, type: 'human' }
      ),
      currentPlayer: 2,
      currentPhase: 'forced_elimination',
      gameStatus: 'active',
    };

    const forcedEliminationCandidates: Move[] = [
      {
        id: 'fe-2a',
        type: 'forced_elimination',
        player: 2,
        moveNumber: 1,
        timestamp: new Date(),
        thinkTime: 0,
      } as Move,
      {
        id: 'fe-2b',
        type: 'forced_elimination',
        player: 2,
        moveNumber: 1,
        timestamp: new Date(),
        thinkTime: 0,
      } as Move,
    ];

    const applyCanonicalMove = jest.fn(async (_move: Move) => {
      // No-op stub: this test only asserts that maybeRunAITurnSandbox
      // selects and attempts to apply a forced_elimination move.
    });

    let lastAIMmoveForP2: Move | null = null;

    const hooksP2: SandboxAIHooks = {
      getPlayerStacks: () => [],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      getValidMovesForCurrentPlayer: () => forcedEliminationCandidates,
      createHypotheticalBoardWithPlacement: (board) => board,
      tryPlaceRings: async () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {
        // no-op
      },
      appendHistoryEntry: () => {
        // no-op for this focused test
      },
      getGameState: () => currentState,
      setGameState: (state: GameState) => {
        currentState = state;
      },
      setLastAIMove: (move: Move | null) => {
        lastAIMmoveForP2 = move;
      },
      setSelectedStackKey: () => {
        // selection not relevant for this test
      },
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove,
      hasPendingTerritorySelfElimination: () => false,
      hasPendingLineRewardElimination: () => false,
      canCurrentPlayerSwapSides: () => false,
      applySwapSidesForCurrentPlayer: () => false,
    };

    const rng = () => 0.5;

    await maybeRunAITurnSandbox(hooksP2, rng);

    expect(applyCanonicalMove).toHaveBeenCalledTimes(1);
    const appliedMove = applyCanonicalMove.mock.calls[0][0] as Move;

    expect(appliedMove.type).toBe('forced_elimination');
    expect(appliedMove.player).toBe(2);

    expect(lastAIMmoveForP2).not.toBeNull();
    if (!lastAIMmoveForP2) {
      throw new Error('Expected lastAIMmoveForP2 to be non-null after forced_elimination move');
    }
    const lastMove = lastAIMmoveForP2 as Move;
    expect(lastMove.type).toBe('forced_elimination');
    expect(lastMove.player).toBe(2);
  });
});

describe('sandboxAI line_processing pending elimination', () => {
  it('prefers eliminate_rings_from_stack when a line-reward elimination is pending', async () => {
    const now = new Date();
    const gameState: GameState = {
      id: 'line-elim',
      boardType: 'square8',
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: { 1: 0, 2: 0 },
        size: 8,
        type: 'square8',
      },
      players: [
        {
          id: 'p1',
          username: 'AI',
          type: 'ai',
          playerNumber: 1,
          isReady: true,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        } as any,
        {
          id: 'p2',
          username: 'Human',
          type: 'human',
          playerNumber: 2,
          isReady: true,
          timeRemaining: 0,
          ringsInHand: 0,
          eliminatedRings: 0,
          territorySpaces: 0,
        } as any,
      ],
      currentPhase: 'line_processing',
      currentPlayer: 1,
      gameStatus: 'active',
      moveHistory: [],
      history: [],
      rngSeed: 42,
      spectators: [],
      timeControl: { initialTime: 600, increment: 0, type: 'blitz' },
      createdAt: now,
      lastMoveAt: now,
      isRated: false,
      maxPlayers: 2,
      totalRingsInPlay: 3,
      totalRingsEliminated: 0,
      victoryThreshold: 0,
      territoryVictoryThreshold: 0,
    };

    const stacks: RingStack[] = [
      {
        position: { x: 0, y: 0 },
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      } as any,
      {
        position: { x: 1, y: 1 },
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      } as any,
    ];

    let lastAIMove: Move | null = null;
    const applyCanonicalMove = jest.fn(async (_move: Move) => {
      gameState.history.push({ ...(_move as any), player: _move.player } as any);
    });

    const hooks: SandboxAIHooks = {
      getPlayerStacks: () => stacks,
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      getValidMovesForCurrentPlayer: () => [],
      createHypotheticalBoardWithPlacement: (board) => board,
      tryPlaceRings: async () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {},
      appendHistoryEntry: () => {},
      getGameState: () => gameState,
      setGameState: (state: GameState) => {
        Object.assign(gameState, state);
      },
      setLastAIMove: (move: Move | null) => {
        lastAIMove = move;
      },
      setSelectedStackKey: () => {},
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove,
      hasPendingTerritorySelfElimination: () => false,
      hasPendingLineRewardElimination: () => true,
      canCurrentPlayerSwapSides: () => false,
      applySwapSidesForCurrentPlayer: () => false,
    };

    const rng = () => 0.4;

    await maybeRunAITurnSandbox(hooks, rng);

    expect(applyCanonicalMove).toHaveBeenCalledTimes(1);
    const appliedMove = applyCanonicalMove.mock.calls[0][0] as Move;
    expect(appliedMove.type).toBe('eliminate_rings_from_stack');
    expect(appliedMove.player).toBe(1);
    expect(lastAIMove).not.toBeNull();
    expect((lastAIMove as Move).type).toBe('eliminate_rings_from_stack');
  });
});
