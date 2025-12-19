import client from 'prom-client';
import { MetricsService, getMetricsService } from '../../src/server/services/MetricsService';

/**
 * Tests for orchestrator shadow metrics bridge.
 *
 * STATUS: DEPRECATED (December 2025)
 *
 * These tests were designed for a shadow comparator feature that was never
 * implemented. The feature would have compared legacy orchestration with the
 * new FSM orchestration to detect mismatches during migration.
 *
 * Since RR-CANON-R070 established FSM as the canonical orchestrator and
 * Phase 4 migration is complete, shadow comparison is no longer needed.
 * This test file is preserved for reference but tests remain skipped.
 *
 * Decision: Remove this file during next test cleanup sprint if FSM
 * migration remains stable.
 */
describe('MetricsService orchestrator shadow metrics bridge [DEPRECATED]', () => {
  beforeEach(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  // DEPRECATED: Shadow comparator feature was never implemented and is now
  // obsolete following FSM migration completion. Test preserved for reference.
  it.skip('exposes shadow comparator gauges in /metrics output', async () => {
    const metrics = getMetricsService();

    // Force a refresh so gauges are registered and populated at least once
    metrics.refreshOrchestratorShadowMetrics();

    const output = await metrics.getMetrics();

    expect(output).toContain('ringrift_orchestrator_shadow_comparisons_current');
    expect(output).toContain('ringrift_orchestrator_shadow_mismatches_current');
    expect(output).toContain('ringrift_orchestrator_shadow_mismatch_rate');
    expect(output).toContain('ringrift_orchestrator_shadow_orchestrator_errors_current');
    expect(output).toContain('ringrift_orchestrator_shadow_orchestrator_error_rate');
    expect(output).toContain('ringrift_orchestrator_shadow_avg_legacy_latency_ms');
    expect(output).toContain('ringrift_orchestrator_shadow_avg_orchestrator_latency_ms');
  });
});
