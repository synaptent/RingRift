import { test, expect } from '@playwright/test';
import { waitForApiReady } from './helpers/test-utils';

/**
 * E2E Test Suite: Metrics & Observability Smoke
 * ============================================================================
 *
 * Verifies that the backend metrics endpoint is exposed and that key
 * orchestrator-related metrics are present. This is a lightweight guardrail
 * to catch regressions in observability wiring (for example metrics being
 * disabled or the endpoint being moved) that would otherwise only surface
 * when dashboards or alerts fail silently.
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - Backend API reachable at E2E_API_BASE_URL or http://localhost:3000
 * - Metrics enabled (ENABLE_METRICS not explicitly set to false)
 *
 * RUN COMMAND: npm run test:e2e -- metrics.e2e.spec.ts
 */

test.describe('Metrics & observability E2E', () => {
  test.setTimeout(60_000);

  test('exposes /metrics with orchestrator gauges', async ({ page }) => {
    // Ensure the API is up and ready before scraping metrics.
    await waitForApiReady(page);

    const apiBaseUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';
    const metricsUrl = `${apiBaseUrl.replace(/\/$/, '')}/metrics`;

    const response = await page.request.get(metricsUrl);
    expect(response.ok()).toBe(true);

    const body = await response.text();

    // Basic sanity: Prometheus format and at least one metric line.
    expect(body).toContain('# HELP');
    expect(body).toContain('\n');

    // Orchestrator-specific metrics that dashboards and alerts rely on.
    expect(body).toContain('ringrift_orchestrator_error_rate');
    expect(body).toContain('ringrift_orchestrator_rollout_percentage');
    expect(body).toContain('ringrift_orchestrator_circuit_breaker_state');
  });
});

