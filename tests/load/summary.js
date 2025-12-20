import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

/**
 * Shared handleSummary helper for RingRift k6 load tests.
 *
 * Goals for Wave 3.1:
 * - Emit a compact JSON summary per scenario under results/load/ that can be
 *   compared against SLO tables in STRATEGIC_ROADMAP.md and PROJECT_GOALS.md.
 * - Capture:
 *   - Scenario name and environment label
 *   - p95/p99 latencies for key HTTP / WebSocket / move metrics
 *   - Error and classification counters (contract vs capacity vs ID lifecycle)
 *
 * This helper is intentionally minimal. It does not try to mirror the full k6
 * JSON export format; instead it focuses on the small set of metrics that the
 * SLO docs and ALERTING_THRESHOLDS.md reference directly.
 */

/**
 * Extract p95/p99 percentiles for a Trend-like metric from the k6 summary.
 *
 * @param {string} metricName
 * @param {any} data - k6 summary object passed to handleSummary
 * @returns {{ p95: number, p99: number } | null}
 */
function extractPercentiles(metricName, data) {
  const metric = data && data.metrics && data.metrics[metricName];
  if (!metric || !metric.values) {
    return null;
  }

  const values = metric.values;
  const count =
    typeof values.count === 'number'
      ? values.count
      : null;
  const hasSamples = count === null ? null : count > 0;

  const rawP95 = values['p(95)'];
  const rawP99 = values['p(99)'];

  // When there are no samples, explicitly report null percentiles and a
  // hasSamples=false flag so consumers can distinguish "no data" from "0ms".
  if (hasSamples === false) {
    return {
      p95: null,
      p99: null,
      hasSamples,
    };
  }

  const p95 = typeof rawP95 === 'number' ? rawP95 : null;
  const p99 = typeof rawP99 === 'number' ? rawP99 : null;

  if (p95 === null && p99 === null && hasSamples !== null) {
    return {
      p95: null,
      p99: null,
      hasSamples,
    };
  }

  if (p95 === null && p99 === null) {
    return null;
  }

  return {
    p95,
    p99,
    hasSamples,
  };
}

/**
 * Extract simple count/rate information for a Counter/Rate metric.
 *
 * @param {string} metricName
 * @param {any} data
 * @returns {{ count?: number, rate?: number } | null}
 */
function extractCounterSummary(metricName, data) {
  const metric = data && data.metrics && data.metrics[metricName];
  if (!metric || !metric.values) {
    return null;
  }

  const values = metric.values;
  const hasCount = typeof values.count === 'number';
  const hasRate = typeof values.rate === 'number';

  if (!hasCount && !hasRate) {
    return null;
  }

  return {
    count: hasCount ? values.count : undefined,
    rate: hasRate ? values.rate : undefined,
  };
}

/**
 * Extract a simple rate value for Rate metrics such as success_rate.
 *
 * @param {string} metricName
 * @param {any} data
 * @returns {number | null}
 */
function extractRate(metricName, data) {
  const metric = data && data.metrics && data.metrics[metricName];
  if (!metric || !metric.values || typeof metric.values.rate !== 'number') {
    return null;
  }
  return metric.values.rate;
}

/**
 * Compute a summary breakdown of errors by classification.
 *
 * Returns:
 * - auth_expired_count: number of auth token expired errors (401)
 * - rate_limit_count: number of rate limit hits (429)
 * - true_error_count: number of real application errors
 * - excluded_count: total auth + rate limit (noise to exclude)
 *
 * @param {Object|null} authStats - Counter stats for auth_token_expired_total
 * @param {Object|null} rateLimitStats - Counter stats for rate_limit_hit_total
 * @param {Object|null} trueErrorStats - Counter stats for true_errors_total
 * @returns {Object} Computed breakdown
 */
function computeErrorBreakdown(authStats, rateLimitStats, trueErrorStats) {
  const authCount = (authStats && typeof authStats.count === 'number') ? authStats.count : 0;
  const rateLimitCount = (rateLimitStats && typeof rateLimitStats.count === 'number') ? rateLimitStats.count : 0;
  const trueErrorCount = (trueErrorStats && typeof trueErrorStats.count === 'number') ? trueErrorStats.count : 0;

  const excludedCount = authCount + rateLimitCount;
  const totalClassified = excludedCount + trueErrorCount;

  return {
    auth_expired_count: authCount,
    rate_limit_count: rateLimitCount,
    true_error_count: trueErrorCount,
    excluded_count: excludedCount,
    total_classified: totalClassified,
    // Human-readable summary for logs
    summary: `Auth expired: ${authCount}, Rate limited: ${rateLimitCount}, True errors: ${trueErrorCount} (${excludedCount} excluded from SLO)`,
  };
}

/**
 * Parse a k6 threshold expression such as "p(95)<800" or "rate<0.01".
 *
 * @param {string} expr
 * @returns {{ statistic: string | null, comparison: string | null, limit: number | null }}
 */
export function parseThresholdExpression(expr) {
  if (typeof expr !== 'string') {
    return { statistic: null, comparison: null, limit: null };
  }

  const trimmed = expr.trim();
  const match = trimmed.match(/^(p\(\d+\)|rate|count|max)\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)$/);

  if (!match) {
    return { statistic: null, comparison: null, limit: null };
  }

  return {
    statistic: match[1],
    comparison: match[2],
    limit: Number(match[3]),
  };
}

/**
 * Resolve the metric value used for a given statistic from a k6 metric.values block.
 *
 * @param {any} values
 * @param {string | null} statistic
 * @returns {number | null}
 */
function resolveMetricValueFromStatistic(values, statistic) {
  if (!values || !statistic) {
    return null;
  }

  if (statistic === 'rate' || statistic === 'count' || statistic === 'max') {
    const v = values[statistic];
    return typeof v === 'number' ? v : null;
  }

  const v = values[statistic];
  return typeof v === 'number' ? v : null;
}

/**
 * Compute per-metric SLO / threshold status for a k6 summary.
 *
 * This walks all metrics that have thresholds configured in the k6 run and
 * records:
 *   - metric name
 *   - raw threshold expression (e.g. "p(95)<800")
 *   - parsed statistic / comparator / limit
 *   - actual measured value used by k6 for the threshold evaluation
 *   - pass/fail
 *
 * @param {string} scenarioName
 * @param {string} environmentLabel
 * @param {any} data
 * @returns {{
 *   scenario: string;
 *   environment: string;
 *   thresholds: Array<{
 *     metric: string;
 *     threshold: string;
 *     statistic: string | null;
 *     comparison: string | null;
 *     limit: number | null;
 *     value: number | null;
 *     passed: boolean;
 *   }>;
 *   overallPass: boolean;
 * }}
 */
export function computeScenarioSloStatus(scenarioName, environmentLabel, data) {
  const metrics = (data && data.metrics) || {};
  /** @type {Array<{
   *   metric: string;
   *   threshold: string;
   *   statistic: string | null;
   *   comparison: string | null;
   *   limit: number | null;
   *   value: number | null;
   *   passed: boolean;
   *   hasSamples: boolean | null;
   * }>} */
  const thresholdStatuses = [];

  for (const [metricName, metric] of Object.entries(metrics)) {
    if (!metric || !metric.thresholds) {
      continue;
    }

    const metricThresholds = metric.thresholds;
    const values = metric.values || {};

    const sampleCount =
      typeof values.count === 'number'
        ? values.count
        : null;
    const hasSamplesForMetric = sampleCount === null ? null : sampleCount > 0;

    for (const [expr, result] of Object.entries(metricThresholds)) {
      const parsed = parseThresholdExpression(expr);
      const valueFromThreshold =
        result && typeof result.actual === 'number' ? result.actual : null;
      const metricValue =
        valueFromThreshold !== null
          ? valueFromThreshold
          : resolveMetricValueFromStatistic(values, parsed.statistic);

      let value = typeof metricValue === 'number' ? metricValue : null;
      let passed = Boolean(result && result.ok);

      // When a metric has no samples at all, treat its thresholds as "not
      // evaluated" and force passed=false with value=null. This prevents
      // zero-sample metrics from silently "passing" SLOs with p95=0.
      if (hasSamplesForMetric === false) {
        value = null;
        passed = false;
      }

      thresholdStatuses.push({
        metric: metricName,
        threshold: expr,
        statistic: parsed.statistic,
        comparison: parsed.comparison,
        limit: parsed.limit,
        value,
        passed,
        hasSamples: hasSamplesForMetric,
      });
    }
  }

  const metricsWithNoSamples = Array.from(
    new Set(
      thresholdStatuses
        .filter((t) => t.hasSamples === false)
        .map((t) => t.metric)
    )
  );
  const anyNoSamples = metricsWithNoSamples.length > 0;

  const overallPass =
    thresholdStatuses.length > 0 && !anyNoSamples
      ? thresholdStatuses.every((t) => t.passed)
      : false;

  return {
    scenario: scenarioName,
    environment: environmentLabel,
    thresholds: thresholdStatuses,
    overallPass,
    noSamplesForMetrics: metricsWithNoSamples,
    overallReason: anyNoSamples ? 'no_samples_for_key_metric' : null,
  };
}

/**
 * Build the compact JSON summary object shared across all scenarios.
 *
 * @param {string} scenarioName
 * @param {string} environmentLabel - human-readable environment (for reporting)
 * @param {string} thresholdsEnvLabel - thresholds.json environment key used for this run
 * @param {any} data
 */
function buildSummaryObject(scenarioName, environmentLabel, thresholdsEnvLabel, data) {
  const http = {
    // Aggregate HTTP timings across all requests in the scenario.
    http_req_duration: extractPercentiles('http_req_duration', data),
    // Scenario-specific trends, present where applicable:
    game_creation_latency_ms: extractPercentiles('game_creation_latency_ms', data),
    // For move scenarios:
    move_submission_latency_ms: extractPercentiles('move_submission_latency_ms', data),
    turn_processing_latency_ms: extractPercentiles('turn_processing_latency_ms', data),
  };

  const websocket = {
    websocket_message_latency_ms: extractPercentiles('websocket_message_latency_ms', data),
    websocket_connection_duration_ms: extractPercentiles(
      'websocket_connection_duration_ms',
      data
    ),
    websocket_connection_success_rate: extractRate('websocket_connection_success_rate', data),
    websocket_handshake_success_rate: extractRate('websocket_handshake_success_rate', data),

    // Gameplay-focused WebSocket metrics (present when scenarios emit them).
    ws_move_rtt_ms: extractPercentiles('ws_move_rtt_ms', data),
    ws_move_success_rate: extractRate('ws_move_success_rate', data),
    ws_moves_attempted_total: extractCounterSummary('ws_moves_attempted_total', data),
    ws_move_stalled_total: extractCounterSummary('ws_move_stalled_total', data),
  };

  const ai = {
    // Kept for future extension; current scenarios primarily validate AI via
    // Prometheus metrics rather than direct k6 metrics.
    ai_move_latency_ms: extractPercentiles('ai_move_latency_ms', data),
  };

  const classifications = {
    // Shared classification counters across all scenarios:
    contract_failures_total: extractCounterSummary('contract_failures_total', data),
    id_lifecycle_mismatches_total: extractCounterSummary(
      'id_lifecycle_mismatches_total',
      data
    ),
    capacity_failures_total: extractCounterSummary('capacity_failures_total', data),

    // Scenario-specific error counters, present where defined:
    game_state_errors: extractCounterSummary('game_state_errors', data),
    move_processing_errors: extractCounterSummary('move_processing_errors', data),
    stalled_moves_total: extractCounterSummary('stalled_moves_total', data),
    websocket_connection_errors: extractCounterSummary('websocket_connection_errors', data),
    websocket_protocol_errors: extractCounterSummary('websocket_protocol_errors', data),

    // Auth lifecycle classifications used by long-running load tests.
    // These are expected during long runs and should NOT count toward true error rate.
    auth_token_expired_total: extractCounterSummary('auth_token_expired_total', data),

    // Rate limiting - expected during capacity stress tests.
    // These are infrastructure-level responses, NOT application errors.
    rate_limit_hit_total: extractCounterSummary('rate_limit_hit_total', data),

    // True errors - all errors EXCLUDING auth (401) and rate-limit (429).
    // This is the key SLO metric for application health.
    true_errors_total: extractCounterSummary('true_errors_total', data),
  };

  // Compute derived error breakdown for SLO summary
  const totalErrorStats = extractCounterSummary('http_req_failed', data);
  const authExpiredStats = classifications.auth_token_expired_total;
  const rateLimitStats = classifications.rate_limit_hit_total;
  const trueErrorStats = classifications.true_errors_total;

  const errorBreakdown = {
    // Total HTTP request failures (from k6 built-in)
    total_http_failures: totalErrorStats,
    // Auth token expired (401) - expected noise, not app errors
    auth_token_expired: authExpiredStats,
    // Rate limit hit (429) - capacity signal, not app errors
    rate_limit_hit: rateLimitStats,
    // True errors - real application errors for SLO
    true_errors: trueErrorStats,
    // Computed true error rate if counts are available
    computed_breakdown: computeErrorBreakdown(authExpiredStats, rateLimitStats, trueErrorStats),
  };

  const raw = {
    scenario: scenarioName,
    environment: environmentLabel,
    // Mirror the THRESHOLD_ENV used by thresholds.json; this is usually an
    // environment label rather than a full deployment identifier.
    thresholdsEnv: thresholdsEnvLabel,
    http,
    websocket,
    ai,
    classifications,
  };

  // For SLO evaluation we care about the thresholds environment label, since it
  // encodes which block of thresholds.json was in effect.
  const slo = computeScenarioSloStatus(
    scenarioName,
    thresholdsEnvLabel || environmentLabel,
    data
  );
  const runTimestamp = new Date().toISOString();

  return {
    // Top-level identity and SLO summary – primary go/no-go signal.
    scenario: scenarioName,
    environment: environmentLabel,
    runTimestamp,
    overallPass: slo.overallPass,
    thresholds: slo.thresholds,
    slo,

    // Error breakdown showing auth/rate-limit exclusions for "true error rate" SLO.
    // This is the key metric distinguishing infrastructure noise from real app errors.
    errorBreakdown,

    // Preserve the existing compact metric shape for backwards compatibility.
    thresholdsEnv: raw.thresholdsEnv,
    http,
    websocket,
    ai,
    classifications,

    // Explicit raw block for consumers that prefer a nested structure.
    raw,
  };
}

/**
 * Factory for per-file handleSummary implementations.
 *
 * Usage in a scenario file (e.g. game-creation.js):
 *
 *   import { makeHandleSummary } from '../summary.js';
 *   export const handleSummary = makeHandleSummary('game-creation');
 *
 * This will:
 *   - Write a JSON file under results/load/<scenario>.<env>.summary.json
 *   - Keep the standard text summary on stdout unless K6_DISABLE_STDOUT_SUMMARY=1
 *
 * Environment variables:
 *   - THRESHOLD_ENV: environment label used for thresholds (default: "staging")
 *   - K6_SUMMARY_DIR: override output directory (default: "results/load")
 *   - K6_DISABLE_STDOUT_SUMMARY=1 to turn off the human-readable text summary
 *
 * @param {string} scenarioName
 * @returns {(data: any) => Record<string, string>}
 */
export function makeHandleSummary(scenarioName) {
  return function handleSummary(data) {
    // Determine the thresholds environment used by k6 for this run. This drives
    // thresholds.json selection and the filename suffix.
    let thresholdsEnv = 'staging';
    if (typeof __ENV !== 'undefined' && __ENV.THRESHOLD_ENV) {
      thresholdsEnv = String(__ENV.THRESHOLD_ENV);
    } else if (typeof process !== 'undefined' && process.env && process.env.THRESHOLD_ENV) {
      thresholdsEnv = String(process.env.THRESHOLD_ENV);
    }

    // Determine the human-readable environment label for reporting. Prefer the
    // deployment-level RINGRIFT_ENV when available, then fall back to any
    // explicit ENVIRONMENT label, and finally to the thresholds environment.
    let environmentLabel = thresholdsEnv;
    if (typeof process !== 'undefined' && process.env && process.env.RINGRIFT_ENV) {
      environmentLabel = String(process.env.RINGRIFT_ENV);
    } else if (typeof __ENV !== 'undefined' && __ENV.ENVIRONMENT) {
      environmentLabel = String(__ENV.ENVIRONMENT);
    }

    // Determine output directory, allowing both k6 (__ENV) and Node (process)
    // environment variables to override the default.
    let baseDir = 'results/load';
    if (typeof __ENV !== 'undefined' && __ENV.K6_SUMMARY_DIR) {
      baseDir = String(__ENV.K6_SUMMARY_DIR);
    } else if (typeof process !== 'undefined' && process.env && process.env.K6_SUMMARY_DIR) {
      baseDir = String(process.env.K6_SUMMARY_DIR);
    }

    const summary = buildSummaryObject(scenarioName, environmentLabel, thresholdsEnv, data);
    const filePath = `${baseDir}/${scenarioName}.${thresholdsEnv}.summary.json`;

    const outputs = {
      [filePath]: JSON.stringify(summary, null, 2),
    };

    // Preserve the default k6 behaviour of printing a human-readable summary to
    // stdout unless explicitly disabled via env var. This keeps local/dev usage
    // friendly while enabling machine-readable JSON for SLO validation.
    const disableStdout =
      typeof __ENV !== 'undefined' && __ENV.K6_DISABLE_STDOUT_SUMMARY === '1';

    if (!disableStdout) {
      outputs.stdout = textSummary(data, { indent: ' ', enableColors: true });
    }

    return outputs;
  };
}