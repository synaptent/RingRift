/**
 * Tests for D5 (plan / issue #82): silent-fallback observability.
 *
 * Verifies:
 * - ai_fallback_moves_total counter is registered with the (reason,
 *   ai_type, difficulty) labels and increments cleanly
 * - ai_circuit_breaker_state gauge is registered and settable
 * - ai_circuit_breaker_transitions_total counter is registered with
 *   (from_state, to_state) labels
 *
 * Intentionally lightweight: does not spin up AIEngine or mock game state.
 * The existing tests/unit/AIEngine.fallback.test.ts covers the higher-level
 * behaviour.
 */

import {
  aiFallbackMovesCounter,
  aiCircuitBreakerStateGauge,
  aiCircuitBreakerTransitionsCounter,
} from '../../src/server/utils/rulesParityMetrics';

async function promValue(metric: any, labels?: Record<string, string>): Promise<number> {
  const collected = await metric.get();
  if (!collected.values || collected.values.length === 0) return 0;
  if (!labels) {
    // Gauge with no labels: pick the single sample.
    const unlabeled = collected.values.find(
      (v: any) => !v.labels || Object.keys(v.labels).length === 0
    );
    return unlabeled ? unlabeled.value : 0;
  }
  const match = collected.values.find((v: any) => {
    if (!v.labels) return false;
    for (const [k, want] of Object.entries(labels)) {
      if (v.labels[k] !== want) return false;
    }
    return true;
  });
  return match ? match.value : 0;
}

describe('D5 fallback telemetry metrics', () => {
  describe('aiFallbackMovesCounter', () => {
    it('has the expected (reason, ai_type, difficulty) label names', () => {
      expect((aiFallbackMovesCounter as any).labelNames).toEqual([
        'reason',
        'ai_type',
        'difficulty',
      ]);
    });

    it('increments with full label set', async () => {
      const before = await promValue(aiFallbackMovesCounter, {
        reason: 'timeout',
        ai_type: 'gumbel_mcts',
        difficulty: '10',
      });
      aiFallbackMovesCounter.labels('timeout', 'gumbel_mcts', '10').inc();
      const after = await promValue(aiFallbackMovesCounter, {
        reason: 'timeout',
        ai_type: 'gumbel_mcts',
        difficulty: '10',
      });
      expect(after).toBe(before + 1);
    });

    it('accepts "unknown" label values', () => {
      // Used by the getLocalFallbackMove path where tier context is absent.
      expect(() =>
        aiFallbackMovesCounter.labels('service_degraded', 'unknown', 'unknown').inc()
      ).not.toThrow();
    });

    it('distinguishes by difficulty tier for the same reason', async () => {
      const tier3Before = await promValue(aiFallbackMovesCounter, {
        reason: 'circuit_open',
        ai_type: 'minimax',
        difficulty: '3',
      });
      const tier7Before = await promValue(aiFallbackMovesCounter, {
        reason: 'circuit_open',
        ai_type: 'mcts',
        difficulty: '7',
      });

      aiFallbackMovesCounter.labels('circuit_open', 'minimax', '3').inc();
      aiFallbackMovesCounter.labels('circuit_open', 'mcts', '7').inc(2);

      const tier3After = await promValue(aiFallbackMovesCounter, {
        reason: 'circuit_open',
        ai_type: 'minimax',
        difficulty: '3',
      });
      const tier7After = await promValue(aiFallbackMovesCounter, {
        reason: 'circuit_open',
        ai_type: 'mcts',
        difficulty: '7',
      });

      expect(tier3After - tier3Before).toBe(1);
      expect(tier7After - tier7Before).toBe(2);
    });
  });

  describe('aiCircuitBreakerStateGauge', () => {
    it('round-trips 0 / 0.5 / 1 for closed / half-open / open', async () => {
      aiCircuitBreakerStateGauge.set(0);
      expect(await promValue(aiCircuitBreakerStateGauge)).toBe(0);
      aiCircuitBreakerStateGauge.set(0.5);
      expect(await promValue(aiCircuitBreakerStateGauge)).toBe(0.5);
      aiCircuitBreakerStateGauge.set(1);
      expect(await promValue(aiCircuitBreakerStateGauge)).toBe(1);
    });
  });

  describe('aiCircuitBreakerTransitionsCounter', () => {
    it('has the expected (from_state, to_state) label names', () => {
      expect((aiCircuitBreakerTransitionsCounter as any).labelNames).toEqual([
        'from_state',
        'to_state',
      ]);
    });

    it('increments on a transition label pair', async () => {
      const before = await promValue(aiCircuitBreakerTransitionsCounter, {
        from_state: 'closed',
        to_state: 'open',
      });
      aiCircuitBreakerTransitionsCounter.labels('closed', 'open').inc();
      const after = await promValue(aiCircuitBreakerTransitionsCounter, {
        from_state: 'closed',
        to_state: 'open',
      });
      expect(after).toBe(before + 1);
    });
  });
});
