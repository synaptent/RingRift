import { GameSession } from '../../src/server/game/GameSession';
import type { Player } from '../../src/shared/types/game';
import {
  orchestratorRollout,
  EngineSelection,
} from '../../src/server/services/OrchestratorRolloutService';

jest.mock('../../src/server/services/OrchestratorRolloutService', () => {
  const actual = jest.requireActual('../../src/server/services/OrchestratorRolloutService');
  return {
    ...actual,
    orchestratorRollout: {
      selectEngine: jest.fn(),
      recordSuccess: jest.fn(),
      recordError: jest.fn(),
      isCircuitBreakerOpen: jest.fn().mockReturnValue(false),
      resetCircuitBreaker: jest.fn(),
      getCircuitBreakerState: jest.fn().mockReturnValue({
        isOpen: false,
        errorCount: 0,
        requestCount: 0,
        windowStart: Date.now(),
      }),
      getErrorRate: jest.fn().mockReturnValue(0),
    },
  };
});

describe('GameSession orchestrator engine selection', () => {
  function createSessionWithFakeEngine() {
    const io = {
      to: jest.fn().mockReturnThis(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient = {} as any;
    const session = new GameSession('game-1', io, pythonClient, new Map());

    let adapterEnabled = false;
    const fakeEngine = {
      enableOrchestratorAdapter: jest.fn(() => {
        adapterEnabled = true;
      }),
      disableOrchestratorAdapter: jest.fn(() => {
        adapterEnabled = false;
      }),
      isOrchestratorAdapterEnabled: jest.fn(() => adapterEnabled),
    };

    (session as any).gameEngine = fakeEngine;

    const players: Player[] = [
      {
        id: 'user-1',
        username: 'Player 1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: 600_000,
        ringsInHand: 18,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];

    return { session, fakeEngine, players };
  }

  it('selects LEGACY engine without mutating adapter enablement', () => {
    (orchestratorRollout.selectEngine as jest.Mock).mockReturnValue({
      engine: EngineSelection.LEGACY,
      reason: 'test-legacy',
    });

    const { session, fakeEngine, players } = createSessionWithFakeEngine();

    (session as any).configureEngineSelection(players);

    // Phase A: adapter enablement is controlled globally, not per session.
    expect(fakeEngine.disableOrchestratorAdapter).not.toHaveBeenCalled();
    expect(fakeEngine.enableOrchestratorAdapter).not.toHaveBeenCalled();
    expect(fakeEngine.isOrchestratorAdapterEnabled()).toBe(false);
    expect((session as any).engineSelection).toBe(EngineSelection.LEGACY);
  });

  it('selects ORCHESTRATOR engine without mutating adapter enablement', () => {
    (orchestratorRollout.selectEngine as jest.Mock).mockReturnValue({
      engine: EngineSelection.ORCHESTRATOR,
      reason: 'test-orchestrator',
    });

    const { session, fakeEngine, players } = createSessionWithFakeEngine();

    (session as any).configureEngineSelection(players);

    // Phase A: adapter enablement is controlled globally, not per session.
    expect(fakeEngine.enableOrchestratorAdapter).not.toHaveBeenCalled();
    expect(fakeEngine.disableOrchestratorAdapter).not.toHaveBeenCalled();
    expect(fakeEngine.isOrchestratorAdapterEnabled()).toBe(false);
    expect((session as any).engineSelection).toBe(EngineSelection.ORCHESTRATOR);
  });
});
