import client from 'prom-client';
import { MetricsService, getMetricsService } from '../../src/server/services/MetricsService';

describe('MetricsService orchestrator shadow metrics bridge', () => {
  beforeEach(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  it('exposes shadow comparator gauges in /metrics output', async () => {
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
