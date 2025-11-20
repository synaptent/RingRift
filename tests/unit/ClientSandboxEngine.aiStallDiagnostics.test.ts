import type { SandboxAIHooks } from '../../src/client/sandbox/sandboxAI';
import { GameState } from '../../src/shared/types/game';
import { createTestGameState } from '../utils/fixtures';

describe('ClientSandboxEngine sandbox AI stall diagnostics', () => {
  const originalEnv = process.env;

  afterEach(() => {
    process.env = originalEnv;
    // Clean up any window shim we created for this test.
    if ((global as any).window) {
      delete (global as any).window;
    }
  });

  it('emits ai_turn and stall entries into window.__RINGRIFT_SANDBOX_TRACE__ for consecutive no-op AI turns', async () => {
    // Enable sandbox AI stall diagnostics before importing the module so that
    // module-level flags (via envFlags) are initialised with the correct value.
    process.env = {
      ...originalEnv,
      RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS: '1',
    };

    // Provide a minimal window shim so sandboxAI can attach its trace buffer.
    (global as any).window = {};

    // Ensure we get a fresh copy of sandboxAI with the updated env applied.
    jest.resetModules();
    const { maybeRunAITurnSandbox } = await import('../../src/client/sandbox/sandboxAI');

    // Build a minimal GameState with a single AI player whose turn it is,
    // with an active game and ring_placement phase. We will provide hooks that
    // expose no legal placements or moves so that every AI tick is a no-op.
    const baseState: GameState = createTestGameState({ boardType: 'square8' });

    let currentState: GameState = {
      ...baseState,
      players: baseState.players.map((p) =>
        p.playerNumber === 1 ? { ...p, type: 'ai' } : { ...p, type: 'human' }
      ),
      currentPlayer: 1,
      gameStatus: 'active',
      currentPhase: 'ring_placement',
    };

    const hooks: SandboxAIHooks = {
      getPlayerStacks: () => [],
      hasAnyLegalMoveOrCaptureFrom: () => false,
      enumerateLegalRingPlacements: () => [],
      tryPlaceRings: () => false,
      enumerateCaptureSegmentsFrom: () => [],
      enumerateSimpleMovementLandings: () => [],
      maybeProcessForcedEliminationForCurrentPlayer: () => false,
      handleMovementClick: async () => {
        // no-op
      },
      appendHistoryEntry: () => {
        // no-op for this diagnostics test
      },
      getGameState: () => currentState,
      setGameState: (state: GameState) => {
        currentState = state;
      },
      setLastAIMove: () => {
        // tracked internally by sandboxAI; not needed here
      },
      setSelectedStackKey: () => {
        // selection not relevant for this test
      },
      getMustMoveFromStackKey: () => undefined,
      applyCanonicalMove: async () => {
        // no-op: our hooks are constructed so canonical moves are never emitted
      },
    };

    // Invoke the sandbox AI for several consecutive no-op turns. Because there
    // are no legal placements or moves, each call should leave the GameState
    // hash unchanged, while the same AI player remains to move in an active
    // game. This should cause the internal consecutive-noop counter to
    // increase and, after the threshold, emit a 'stall' entry.
    const INVOCATIONS = 8;
    for (let i = 0; i < INVOCATIONS; i += 1) {
      await maybeRunAITurnSandbox(hooks);
    }

    const trace = ((global as any).window.__RINGRIFT_SANDBOX_TRACE__ ?? []) as any[];

    expect(trace.length).toBeGreaterThan(0);
    expect(trace.some((entry: any) => entry.kind === 'ai_turn')).toBe(true);
    expect(trace.some((entry: any) => entry.kind === 'stall')).toBe(true);
  });
});