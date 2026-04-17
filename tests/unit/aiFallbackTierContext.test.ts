/**
 * Test for #86: getLocalFallbackMove labels ai_type and difficulty.
 *
 * Before the fix, every service_degraded fallback emitted
 *   ai_fallback_moves_total{reason="service_degraded", ai_type="unknown", difficulty="unknown"}
 * so per-tier drill-down on D5 alerts was blinded.
 *
 * The fix: look up AIConfig from aiConfigs by playerNumber and emit the
 * real tier labels. Falls back to 'unknown' only when no config exists
 * (unconfigured seat).
 */

import { AIEngine, AIType } from '../../src/server/game/ai/AIEngine';
import { aiFallbackMovesCounter } from '../../src/server/utils/rulesParityMetrics';
import type { GameState } from '../../src/shared/types/game';

// Construct the minimal GameState shape that the fallback move selector
// needs (it calls into the rules engine for legal moves — we intercept
// by mocking the rules engine at the constructor level).
jest.mock('../../src/server/game/RuleEngine', () => {
  return {
    RuleEngine: jest.fn().mockImplementation(() => ({
      getValidMoves: () => [
        {
          id: 'place-test',
          type: 'place',
          player: 1,
          to: { x: 0, y: 0 },
          ringsToPlace: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
        {
          id: 'place-test-2',
          type: 'place',
          player: 1,
          to: { x: 1, y: 0 },
          ringsToPlace: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ],
    })),
  };
});

function makeMinimalGameState(): GameState {
  return {
    id: 'test-game',
    status: 'active',
    boardType: 'square8',
    currentPlayer: 1,
    players: [{ playerNumber: 1, name: 'p1' } as any, { playerNumber: 2, name: 'p2' } as any],
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

async function readLabelSample(
  metric: typeof aiFallbackMovesCounter,
  labels: Record<string, string>
): Promise<number> {
  const collected = await metric.get();
  const match = collected.values.find((v: any) => {
    if (!v.labels) return false;
    for (const [k, want] of Object.entries(labels)) {
      if (v.labels[k] !== want) return false;
    }
    return true;
  });
  return match ? match.value : 0;
}

describe('getLocalFallbackMove tier context (#86)', () => {
  it('uses configured ai_type + difficulty when the player has an AI config', async () => {
    const engine = new AIEngine();
    // D8 tier maps to MCTS per AI_DIFFICULTY_PRESETS (see AIEngine.ts).
    engine.createAI(1, 8);

    const before = await readLabelSample(aiFallbackMovesCounter, {
      reason: 'service_degraded',
      ai_type: String(AIType.MCTS),
      difficulty: '8',
    });

    const state = makeMinimalGameState();
    const move = engine.getLocalFallbackMove(1, state);
    expect(move).not.toBeNull();

    const after = await readLabelSample(aiFallbackMovesCounter, {
      reason: 'service_degraded',
      ai_type: String(AIType.MCTS),
      difficulty: '8',
    });

    expect(after).toBeGreaterThan(before);
  });

  it('emits the "unknown" labels when the player has no AI config', async () => {
    const engine = new AIEngine();
    // Intentionally do NOT call createAI — the player has no config.

    const before = await readLabelSample(aiFallbackMovesCounter, {
      reason: 'service_degraded',
      ai_type: 'unknown',
      difficulty: 'unknown',
    });

    const state = makeMinimalGameState();
    engine.getLocalFallbackMove(1, state);

    const after = await readLabelSample(aiFallbackMovesCounter, {
      reason: 'service_degraded',
      ai_type: 'unknown',
      difficulty: 'unknown',
    });

    expect(after).toBeGreaterThan(before);
  });
});
