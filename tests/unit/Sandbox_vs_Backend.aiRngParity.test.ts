import {
  GameState,
  BoardState,
  Player,
  Move,
  BoardType,
  BOARD_CONFIGS,
} from '../../src/shared/types/game';
import { hashGameState } from '../../src/shared/engine/core';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import { AIEngine } from '../../src/server/game/ai/AIEngine';
import { LocalAIRng } from '../../src/shared/engine/localAIMoveSelection';

// Ensure that any attempt by AIEngine to call the Python service during
// these RNG plumbing tests fails fast, forcing the local fallback path.
// This keeps the suite hermetic and guarantees that getAIMove exercises
// local heuristic selection when we explicitly test that wiring.
jest.mock('../../src/server/services/AIServiceClient', () => {
  return {
    AIType: {
      RANDOM: 'random',
      HEURISTIC: 'heuristic',
      MINIMAX: 'minimax',
      MCTS: 'mcts',
      DESCENT: 'descent',
    },
    getAIServiceClient: () => ({
      getAIMove: jest.fn().mockRejectedValue(new Error('AI service disabled in RNG tests')),
    }),
  };
});

// For the backend getAIMove RNG wiring test, stub RuleEngine.getValidMoves so
// that AIEngine sees a non-empty candidate set and actually exercises the
// local heuristic path instead of returning early when there are no moves.
jest.mock('../../src/server/game/RuleEngine', () => {
  const { BOARD_CONFIGS } = jest.requireActual('../../src/shared/types/game');
  const boardSize = BOARD_CONFIGS.square8.size;

  return {
    RuleEngine: jest.fn().mockImplementation(() => ({
      getValidMoves: () => {
        const now = new Date();
        const move1: Move = {
          id: 'backend-m1',
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
          placementCount: 1,
          timestamp: now,
          thinkTime: 0,
          moveNumber: 1,
        };
        const move2: Move = {
          id: 'backend-m2',
          type: 'place_ring',
          player: 1,
          to: { x: boardSize - 1, y: boardSize - 1 },
          placementCount: 1,
          timestamp: now,
          thinkTime: 0,
          moveNumber: 1,
        };
        return [move1, move2];
      },
    })),
  };
});

/**
 * RNG-parity / RNG-hook tests.
 *
 * These tests do NOT try to assert full end-to-end sandbox-vs-backend AI
 * behavioural parity (that is covered by existing trace + heuristic
 * coverage harnesses). Instead, they focus on the *RNG plumbing*:
 *
 *   - When an explicit RNG is provided, sandbox AI (`maybeRunAITurn`) and
 *     backend local AI selection (`AIEngine.chooseLocalMoveFromCandidates`)
 *     must use that RNG rather than falling back to global Math.random.
 *
 * This is the critical guarantee needed for trace-mode RNG hooks so that
 * higher-level parity harnesses can share a deterministic RNG stream across
 * sandbox and backend AI, while preserving the existing proportional
 * selection policy implemented in localAIMoveSelection.
 */

describe('Sandbox vs Backend AI RNG hooks', () => {
  /** Tiny deterministic PRNG (same LCG as other AI tests). */
  function makePrng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }

  function makeCountingRng(seed: number): { rng: LocalAIRng; getCallCount: () => number } {
    const base = makePrng(seed);
    let calls = 0;
    const rng: LocalAIRng = () => {
      calls += 1;
      return base();
    };
    return {
      rng,
      getCallCount: () => calls,
    };
  }

  function createSandboxEngine(boardType: BoardType, numPlayers: number): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers,
      playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice>(choice: TChoice): Promise<any> {
        const anyChoice = choice as any;

        // Mirror the deterministic handler used in trace + heuristic tests:
        // - For capture_direction, pick the lexicographically smallest landing.
        // - For all other choices, pick the first option.
        if (anyChoice.type === 'capture_direction') {
          const options = anyChoice.options || [];
          if (options.length === 0) {
            throw new Error('SandboxInteractionHandler: no options for capture_direction');
          }

          let selected = options[0];
          for (const opt of options) {
            if (
              opt.landingPosition.x < selected.landingPosition.x ||
              (opt.landingPosition.x === selected.landingPosition.x &&
                opt.landingPosition.y < selected.landingPosition.y)
            ) {
              selected = opt;
            }
          }

          return {
            choiceId: anyChoice.id,
            playerNumber: anyChoice.playerNumber,
            choiceType: anyChoice.type,
            selectedOption: selected,
          };
        }

        const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        };
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  function makeDummyGameStateForBackend(boardType: BoardType): GameState {
    const boardConfig = BOARD_CONFIGS[boardType];

    const board: BoardState = {
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: boardConfig.size,
      type: boardType,
    } as any;

    const players: Player[] = [
      {
        id: 'p1',
        username: 'AI',
        type: 'ai',
        playerNumber: 1,
        rating: undefined,
        isReady: true,
        timeRemaining: 0,
        aiDifficulty: 5,
        aiProfile: { difficulty: 5, mode: 'service', aiType: 'random' } as any,
        ringsInHand: boardConfig.ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    const base: GameState = {
      id: 'g-ai-rng-hooks',
      boardType,
      board,
      players,
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      timeControl: { initialTime: 0, increment: 0, type: 'blitz' },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 1,
      // For RNG-hook tests we only need a structurally valid GameState;
      // totalRingsInPlay is not inspected by localAIMoveSelection, so we
      // approximate it from ringsPerPlayer.
      totalRingsInPlay: boardConfig.ringsPerPlayer * players.length,
      totalRingsEliminated: 0,
      victoryThreshold: 10,
      territoryVictoryThreshold: 33,
    } as any;

    return base;
  }

  test('sandbox AI uses injected RNG instead of Math.random during maybeRunAITurn', async () => {
    const boardType: BoardType = 'square8';
    const numPlayers = 2;

    const sandbox = createSandboxEngine(boardType, numPlayers);

    const { rng, getCallCount } = makeCountingRng(42);

    const originalRandom = Math.random;
    Math.random = () => {
      throw new Error('Math.random should not be called when an explicit RNG is provided');
    };

    try {
      const before = sandbox.getGameState();
      const beforeHash = hashGameState(before);

      await sandbox.maybeRunAITurn(rng);

      const after = sandbox.getGameState();
      const afterHash = hashGameState(after);

      // Sanity: the AI should either change the state or at least record a
      // lastAIMove; this is not the primary assertion but helps catch wiring
      // regressions.
      const lastMove = sandbox.getLastAIMoveForTesting();
      expect(lastMove).not.toBeNull();
      expect(beforeHash).not.toBe(afterHash);

      // Core assertion: our injected RNG was used at least once, and
      // Math.random was never called.
      expect(getCallCount()).toBeGreaterThan(0);
    } finally {
      Math.random = originalRandom;
    }
  });

  test('backend getAIMove local fallback threads injected RNG through to local heuristic selection', async () => {
    const engine = new AIEngine();
    // Configure a simple AI profile so getAIMove has a config entry.
    engine.createAI(1, 5);

    const state = {
      id: 'g-ai-rng-fallback',
      boardType: 'square8',
      currentPhase: 'movement',
      currentPlayer: 1,
      board: {
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
        size: BOARD_CONFIGS.square8.size,
        type: 'square8',
      },
      players: [],
      moveHistory: [],
      timeControl: { initialTime: 0, increment: 0, type: 'blitz' },
      spectators: [],
      gameStatus: 'active',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated: false,
      maxPlayers: 2,
    } as unknown as GameState;

    const { rng } = makeCountingRng(321);

    const localSpy = jest
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .spyOn(engine as any, 'selectLocalHeuristicMove')
      // We do not care about the actual move here, only that the RNG is
      // propagated; stub out the implementation to avoid exercising the full
      // rules engine in this wiring test.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .mockImplementation((): any => null);

    const result = await engine.getAIMove(1, state, rng);

    // When selectLocalHeuristicMove returns null, the Level 3 random fallback
    // kicks in and picks a move from validMoves. We don't assert the specific
    // move, only that:
    // 1. selectLocalHeuristicMove was called with the correct RNG
    // 2. A move was returned (random fallback worked)
    expect(result).not.toBeNull();
    expect(localSpy).toHaveBeenCalledTimes(1);

    const [, , passedRng] = localSpy.mock.calls[0];
    expect(passedRng).toBe(rng);
  });

  test('backend local AI move selection uses injected RNG instead of Math.random', () => {
    const boardType: BoardType = 'square8';
    const state = makeDummyGameStateForBackend(boardType);

    // Construct a simple, non-empty candidate set. The exact moves are not
    // important; we just need localAIMoveSelection to be exercised.
    const candidates: Move[] = [
      {
        id: 'm1',
        type: 'place_ring' as any,
        player: 1,
        to: { x: 0, y: 0 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      } as Move,
      {
        id: 'm2',
        type: 'place_ring' as any,
        player: 1,
        to: { x: 1, y: 1 },
        placementCount: 1,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: 1,
      } as Move,
    ];

    const engine = new AIEngine();

    const { rng, getCallCount } = makeCountingRng(123);

    const originalRandom = Math.random;
    Math.random = () => {
      throw new Error('Math.random should not be called when an explicit RNG is provided');
    };

    try {
      const move = engine.chooseLocalMoveFromCandidates(1, state, candidates, rng);
      expect(move).not.toBeNull();

      // The shared selector should have used rng() internally at least once.
      expect(getCallCount()).toBeGreaterThan(0);
    } finally {
      Math.random = originalRandom;
    }
  });
});
