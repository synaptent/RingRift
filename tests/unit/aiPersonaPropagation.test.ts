/**
 * Tests for C2 (plan / issue #80) Phase 2: persona_id propagates from
 * AIProfile → AIConfig → AIServiceClient request payload.
 *
 * Phase 1 covered the Python /ai/move endpoint. Phase 3 is the client UI.
 * Here we lock in the TS-server middle layer so a personaId set on an
 * AIProfile at game creation time arrives at the Python service in the
 * MoveRequest as `persona_id`.
 */

import { AIEngine } from '../../src/server/game/ai/AIEngine';
import { getAIServiceClient, type PersonaId } from '../../src/server/services/AIServiceClient';
import type { GameState, Move, AIProfile } from '../../src/shared/types/game';

// Auto-mock the client for getAIMove control, but preserve real
// constants (ALLOWED_PERSONA_IDS) and type exports so the contract
// test against the Python _ALLOWED_PERSONA_IDS set still works.
jest.mock('../../src/server/services/AIServiceClient', () => {
  const actual = jest.requireActual('../../src/server/services/AIServiceClient');
  return {
    ...actual,
    getAIServiceClient: jest.fn(),
  };
});
jest.mock('../../src/server/utils/logger');

// Re-import ALLOWED_PERSONA_IDS from the real module via requireActual so
// the equality contract test below sees the live tuple, not an automock.
const { ALLOWED_PERSONA_IDS } = jest.requireActual('../../src/server/services/AIServiceClient');

// Shared mutable backing store for the mocked RuleEngine output.
let mockValidMoves: Move[] = [];
jest.mock('../../src/server/game/RuleEngine', () => ({
  RuleEngine: jest.fn().mockImplementation(() => ({
    getValidMoves: () => mockValidMoves,
  })),
}));

function makeMove(x = 0, y = 0): Move {
  return {
    id: `place-${x}-${y}`,
    type: 'place',
    player: 1,
    to: { x, y },
    ringsToPlace: 1,
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  } as unknown as Move;
}

function makeGameState(): GameState {
  return {
    id: 'test-game',
    status: 'active',
    boardType: 'square8',
    currentPlayer: 1,
    players: [
      { playerNumber: 1, name: 'p1' } as unknown as GameState['players'][0],
      { playerNumber: 2, name: 'p2' } as unknown as GameState['players'][0],
    ],
    currentPhase: 'placement',
    moveHistory: [],
    board: {
      type: 'square8',
      stacks: new Map(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
    } as any,
    rngSeed: 1,
  } as unknown as GameState;
}

describe('AI persona propagation (C2 phase 2)', () => {
  let engine: AIEngine;
  let mockClient: ReturnType<typeof getAIServiceClient>;

  beforeEach(() => {
    jest.clearAllMocks();
    engine = new AIEngine();
    mockClient = {
      getAIMove: jest.fn().mockResolvedValue({
        move: makeMove(),
        evaluation: 0,
        thinking_time_ms: 10,
        ai_type: 'heuristic',
        difficulty: 2,
      }),
      healthCheck: jest.fn(),
      clearCache: jest.fn(),
      getCircuitBreakerStatus: jest.fn(() => ({ isOpen: false, failureCount: 0 })),
    } as unknown as ReturnType<typeof getAIServiceClient>;
    (getAIServiceClient as jest.Mock).mockReturnValue(mockClient);
    mockValidMoves = [makeMove(0, 0), makeMove(1, 0)];
  });

  describe('ALLOWED_PERSONA_IDS export', () => {
    it('mirrors the Python _ALLOWED_PERSONA_IDS set', () => {
      expect([...ALLOWED_PERSONA_IDS].sort()).toEqual([
        'aggressive',
        'balanced',
        'defensive',
        'territorial',
      ]);
    });
  });

  describe('AIConfig stores validated personaId', () => {
    for (const persona of ['balanced', 'aggressive', 'territorial', 'defensive'] as const) {
      it(`accepts ${persona} from AIProfile and stores on AIConfig`, () => {
        const profile: AIProfile = { difficulty: 5, personaId: persona };
        engine.createAIFromProfile(1, profile);
        const cfg = engine.getAIConfig(1);
        expect(cfg?.personaId).toBe(persona);
      });
    }

    it('drops unknown persona strings (AIConfig has undefined)', () => {
      const profile: AIProfile = { difficulty: 5, personaId: 'unknown-persona' };
      engine.createAIFromProfile(1, profile);
      const cfg = engine.getAIConfig(1);
      expect(cfg?.personaId).toBeUndefined();
    });

    it('drops casing-variant persona strings', () => {
      const profile: AIProfile = { difficulty: 5, personaId: 'AGGRESSIVE' };
      engine.createAIFromProfile(1, profile);
      expect(engine.getAIConfig(1)?.personaId).toBeUndefined();
    });

    it('omitted personaId leaves AIConfig.personaId undefined', () => {
      const profile: AIProfile = { difficulty: 5 };
      engine.createAIFromProfile(1, profile);
      expect(engine.getAIConfig(1)?.personaId).toBeUndefined();
    });
  });

  describe('persona reaches AIServiceClient.getAIMove', () => {
    it('forwards personaId as the 7th positional arg when set', async () => {
      engine.createAIFromProfile(1, { difficulty: 5, personaId: 'aggressive' });
      await engine.getAIMove(1, makeGameState());
      const calls = (mockClient.getAIMove as jest.Mock).mock.calls;
      expect(calls.length).toBeGreaterThan(0);
      // Signature: (gameState, playerNumber, difficulty, aiType, seed, options, personaId)
      const personaArg = calls[0][6];
      expect(personaArg).toBe('aggressive');
    });

    it('forwards undefined when no persona is configured', async () => {
      engine.createAIFromProfile(1, { difficulty: 5 });
      await engine.getAIMove(1, makeGameState());
      const personaArg = (mockClient.getAIMove as jest.Mock).mock.calls[0][6];
      expect(personaArg).toBeUndefined();
    });

    it('forwards undefined when an unknown persona was attempted', async () => {
      engine.createAIFromProfile(1, { difficulty: 5, personaId: 'unknown' });
      await engine.getAIMove(1, makeGameState());
      const personaArg = (mockClient.getAIMove as jest.Mock).mock.calls[0][6];
      expect(personaArg).toBeUndefined();
    });

    it('each of the four personas reaches the service unchanged', async () => {
      const typed: PersonaId[] = [...ALLOWED_PERSONA_IDS];
      for (const persona of typed) {
        const fresh = new AIEngine();
        fresh.createAIFromProfile(1, { difficulty: 5, personaId: persona });
        await fresh.getAIMove(1, makeGameState());
      }
      const calls = (mockClient.getAIMove as jest.Mock).mock.calls;
      const personas = calls.slice(-typed.length).map((c) => c[6]);
      expect(personas.sort()).toEqual([...typed].sort());
    });
  });
});
