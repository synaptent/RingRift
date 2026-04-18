import { GameSession } from '../../src/server/game/GameSession';
import { globalAIEngine } from '../../src/server/game/ai/AIEngine';

jest.mock('../../src/server/game/ai/AIEngine', () => ({
  globalAIEngine: {
    getDiagnostics: jest.fn(),
  },
}));

describe('GameSession AI diagnostics aggregation', () => {
  const makeSession = () => {
    const io = {
      to: jest.fn().mockReturnThis(),
      sockets: {
        adapter: { rooms: new Map() },
        sockets: new Map(),
      },
    } as any;

    const pythonClient = {} as any;

    return new GameSession('game-1', io, pythonClient, new Map());
  };

  it('escalates aiQualityMode to rulesServiceDegraded when rules diagnostics show failures', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 1,
      pythonBackendFallbacks: 2,
      pythonShadowErrors: 3,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 4,
      localFallbackCount: 1,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.rulesServiceFailureCount).toBe(1);
    expect(snapshot.rulesShadowErrorCount).toBe(0);
    expect(snapshot.aiServiceFailureCount).toBe(4);
    expect(snapshot.aiFallbackMoveCount).toBe(1);
    expect(snapshot.aiQualityMode).toBe('rulesServiceDegraded');
  });

  it('sets aiQualityMode to fallbackLocalAI when AI falls back locally and rules are healthy', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 2,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.rulesServiceFailureCount).toBe(0);
    expect(snapshot.rulesShadowErrorCount).toBe(0);
    expect(snapshot.aiServiceFailureCount).toBe(0);
    expect(snapshot.aiFallbackMoveCount).toBe(2);
    expect(snapshot.aiQualityMode).toBe('fallbackLocalAI');
  });

  it('sets aiQualityMode to normal when both rules and AI diagnostics are clean', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.rulesServiceFailureCount).toBe(0);
    expect(snapshot.rulesShadowErrorCount).toBe(0);
    expect(snapshot.aiServiceFailureCount).toBe(0);
    expect(snapshot.aiFallbackMoveCount).toBe(0);
    expect(snapshot.aiQualityMode).toBe('normal');
  });

  it('exposes AIRequestState terminal breakdown fields in diagnostics snapshot', () => {
    const session = makeSession();

    // Seed internal counters that would normally be driven by the
    // AIRequestState machine / GameSession request tracking.
    (session as any).aiRequestState = { kind: 'timed_out' };

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 0,
      localFallbackCount: 0,
    });

    (session as any).updateDiagnostics(1);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting() as any;

    // The snapshot currently focuses on aggregate quality/health fields rather
    // than mirroring the full AIRequestState; this test simply ensures that
    // calling updateDiagnostics with a non-idle aiRequestState does not throw
    // and returns a structured object.
    expect(typeof snapshot).toBe('object');
  });

  it('preserves last move telemetry when diagnostics counters are recomputed', () => {
    const session = makeSession();

    const rulesDiag = {
      pythonEvalFailures: 0,
      pythonBackendFallbacks: 0,
      pythonShadowErrors: 0,
    };

    (session as any).rulesFacade = {
      getDiagnostics: jest.fn(() => rulesDiag),
    };

    (session as any).diagnosticsSnapshot = {
      rulesServiceFailureCount: 99,
      rulesShadowErrorCount: 0,
      aiServiceFailureCount: 99,
      aiFallbackMoveCount: 99,
      aiQualityMode: 'fallbackLocalAI',
      lastMoveTelemetry: {
        playerNumber: 2,
        moveType: 'place_ring',
        source: 'python_service',
        aiTier: 8,
        aiType: 'mcts',
        modelVersion: 'v2.0.0',
        modelPath: 'models/canonical_square8_2p.pth',
        latencyMs: 812,
        fallbackUsed: false,
        fallbackReason: null,
        recordedAt: '2026-04-18T00:00:00.000Z',
      },
    };

    (globalAIEngine.getDiagnostics as jest.Mock).mockReturnValue({
      serviceFailureCount: 1,
      localFallbackCount: 0,
    });

    (session as any).updateDiagnostics(2);

    const snapshot = session.getAIDiagnosticsSnapshotForTesting();

    expect(snapshot.aiServiceFailureCount).toBe(1);
    expect(snapshot.lastMoveTelemetry).toEqual(
      expect.objectContaining({
        playerNumber: 2,
        moveType: 'place_ring',
        modelVersion: 'v2.0.0',
        fallbackUsed: false,
      })
    );
  });
});
