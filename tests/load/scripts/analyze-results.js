#!/usr/bin/env node

/**
 * RingRift Baseline Load Test Results Analyzer
 *
 * Analyzes k6 JSON output and generates a capacity report.
 * Compares results against SLO targets defined in thresholds.json and PROJECT_GOALS.md.
 *
 * Usage:
 *   node analyze-results.js <results.json> [output-summary.json]
 *
 * The analyzer reads the k6 JSON stream output and computes:
 *   - Latency percentiles (p50, p95, p99, max)
 *   - Error rates and failure classification
 *   - Throughput metrics
 *   - SLO compliance status
 */

const fs = require('fs');
const path = require('path');

// SLO targets from PROJECT_GOALS.md and thresholds.json
const SLO_TARGETS = {
  concurrent_games: 100,
  concurrent_players: 300,
  p95_latency_ms: 500,
  p99_latency_ms: 2000,
  error_rate_percent: 1.0,
  ws_connection_success_percent: 99.0,
};

// Get command line arguments
const args = process.argv.slice(2);
const resultsFile = args[0];
const outputFile = args[1];

if (!resultsFile) {
  console.error('Usage: node analyze-results.js <results.json> [output-summary.json]');
  process.exit(1);
}

if (!fs.existsSync(resultsFile)) {
  console.error(`Results file not found: ${resultsFile}`);
  process.exit(1);
}

/**
 * Parse k6 JSON stream output (newline-delimited JSON)
 */
function parseK6JsonStream(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n').filter((line) => line.trim());

  const entries = [];
  for (const line of lines) {
    try {
      entries.push(JSON.parse(line));
    } catch (e) {
      // Skip malformed lines
    }
  }
  return entries;
}

/**
 * Extract metrics from k6 data points
 */
function extractMetrics(entries) {
  const metrics = {
    http_req_duration: [],
    http_req_failed: [],
    game_creation_latency: [],
    game_state_check: [],
    concurrent_games: [],
    ws_connection_success: [],
    contract_failures: 0,
    capacity_failures: 0,
    lifecycle_mismatches: 0,
    total_requests: 0,
    failed_requests: 0,
    startTime: null,
    endTime: null,
  };

  for (const entry of entries) {
    if (entry.type !== 'Point') continue;

    const { metric, data } = entry;
    if (!data) continue;

    // Track time range
    const timestamp = new Date(data.time);
    if (!metrics.startTime || timestamp < metrics.startTime) {
      metrics.startTime = timestamp;
    }
    if (!metrics.endTime || timestamp > metrics.endTime) {
      metrics.endTime = timestamp;
    }

    // Process different metric types
    switch (metric) {
      case 'http_req_duration':
        if (typeof data.value === 'number') {
          metrics.http_req_duration.push(data.value);
          metrics.total_requests++;
        }
        break;

      case 'http_req_failed':
        if (data.value === 1) {
          metrics.failed_requests++;
        }
        break;

      case 'game_creation_latency_ms':
        if (typeof data.value === 'number') {
          metrics.game_creation_latency.push(data.value);
        }
        break;

      case 'game_state_check_success':
        if (typeof data.value === 'number') {
          metrics.game_state_check.push(data.value);
        }
        break;

      case 'concurrent_active_games':
        if (typeof data.value === 'number') {
          metrics.concurrent_games.push(data.value);
        }
        break;

      case 'websocket_connection_success_rate':
        if (typeof data.value === 'number') {
          metrics.ws_connection_success.push(data.value);
        }
        break;

      case 'contract_failures_total':
        if (typeof data.value === 'number') {
          metrics.contract_failures = Math.max(metrics.contract_failures, data.value);
        }
        break;

      case 'capacity_failures_total':
        if (typeof data.value === 'number') {
          metrics.capacity_failures = Math.max(metrics.capacity_failures, data.value);
        }
        break;

      case 'id_lifecycle_mismatches_total':
        if (typeof data.value === 'number') {
          metrics.lifecycle_mismatches = Math.max(metrics.lifecycle_mismatches, data.value);
        }
        break;
    }
  }

  return metrics;
}

/**
 * Calculate percentile from sorted array
 */
function percentile(sortedArray, p) {
  if (sortedArray.length === 0) return 0;
  const index = Math.ceil((p / 100) * sortedArray.length) - 1;
  return sortedArray[Math.max(0, index)];
}

/**
 * Calculate statistics from raw metrics
 */
function calculateStatistics(metrics) {
  const sortedLatencies = [...metrics.http_req_duration].sort((a, b) => a - b);
  const sortedGameCreation = [...metrics.game_creation_latency].sort((a, b) => a - b);

  const duration_ms =
    metrics.startTime && metrics.endTime ? metrics.endTime - metrics.startTime : 0;

  const stats = {
    duration: {
      ms: duration_ms,
      minutes: (duration_ms / 1000 / 60).toFixed(2),
      formatted: formatDuration(duration_ms),
    },

    requests: {
      total: metrics.total_requests,
      failed: metrics.failed_requests,
      success_rate:
        metrics.total_requests > 0
          ? ((metrics.total_requests - metrics.failed_requests) / metrics.total_requests) * 100
          : 0,
      error_rate:
        metrics.total_requests > 0 ? (metrics.failed_requests / metrics.total_requests) * 100 : 0,
    },

    latency: {
      samples: sortedLatencies.length,
      min: sortedLatencies.length > 0 ? sortedLatencies[0] : 0,
      max: sortedLatencies.length > 0 ? sortedLatencies[sortedLatencies.length - 1] : 0,
      avg:
        sortedLatencies.length > 0
          ? sortedLatencies.reduce((a, b) => a + b, 0) / sortedLatencies.length
          : 0,
      p50: percentile(sortedLatencies, 50),
      p95: percentile(sortedLatencies, 95),
      p99: percentile(sortedLatencies, 99),
    },

    game_creation: {
      samples: sortedGameCreation.length,
      p95: percentile(sortedGameCreation, 95),
      p99: percentile(sortedGameCreation, 99),
    },

    capacity: {
      max_concurrent_games:
        metrics.concurrent_games.length > 0 ? Math.max(...metrics.concurrent_games) : 0,
      avg_concurrent_games:
        metrics.concurrent_games.length > 0
          ? metrics.concurrent_games.reduce((a, b) => a + b, 0) / metrics.concurrent_games.length
          : 0,
    },

    throughput: {
      rps: duration_ms > 0 ? (metrics.total_requests / (duration_ms / 1000)).toFixed(2) : 0,
    },

    classification: {
      contract_failures: metrics.contract_failures,
      capacity_failures: metrics.capacity_failures,
      lifecycle_mismatches: metrics.lifecycle_mismatches,
    },
  };

  return stats;
}

/**
 * Format duration in human-readable form
 */
function formatDuration(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

/**
 * Evaluate SLO compliance
 */
function evaluateSLOs(stats) {
  const sloResults = {
    p95_latency: {
      target_ms: SLO_TARGETS.p95_latency_ms,
      actual_ms: stats.latency.p95,
      passed: stats.latency.p95 < SLO_TARGETS.p95_latency_ms,
      margin_pct:
        stats.latency.p95 > 0
          ? (((SLO_TARGETS.p95_latency_ms - stats.latency.p95) / SLO_TARGETS.p95_latency_ms) * 100).toFixed(1)
          : 100,
    },

    p99_latency: {
      target_ms: SLO_TARGETS.p99_latency_ms,
      actual_ms: stats.latency.p99,
      passed: stats.latency.p99 < SLO_TARGETS.p99_latency_ms,
      margin_pct:
        stats.latency.p99 > 0
          ? (((SLO_TARGETS.p99_latency_ms - stats.latency.p99) / SLO_TARGETS.p99_latency_ms) * 100).toFixed(1)
          : 100,
    },

    error_rate: {
      target_percent: SLO_TARGETS.error_rate_percent,
      actual_percent: stats.requests.error_rate,
      passed: stats.requests.error_rate < SLO_TARGETS.error_rate_percent,
    },

    contract_failures: {
      target: 0,
      actual: stats.classification.contract_failures,
      passed: stats.classification.contract_failures === 0,
    },
  };

  sloResults.all_passed = Object.values(sloResults)
    .filter((v) => typeof v === 'object' && 'passed' in v)
    .every((r) => r.passed);

  return sloResults;
}

/**
 * Print formatted report to console
 */
function printReport(stats, sloResults) {
  const passIcon = (passed) => (passed ? '✅ PASS' : '❌ FAIL');

  console.log(`
╔════════════════════════════════════════════════════════════════════╗
║             RingRift Baseline Capacity Report                      ║
╠════════════════════════════════════════════════════════════════════╣
║  Duration:        ${stats.duration.formatted.padEnd(20)}                       ║
║  Total Requests:  ${String(stats.requests.total).padEnd(20)}                       ║
║  Failed Requests: ${String(stats.requests.failed).padEnd(20)}                       ║
║  Error Rate:      ${stats.requests.error_rate.toFixed(2)}%                                           ║
╠════════════════════════════════════════════════════════════════════╣
║  LATENCY (ms)                                                      ║
║    p50:  ${String(stats.latency.p50.toFixed(0)).padEnd(10)} p95:  ${String(stats.latency.p95.toFixed(0)).padEnd(10)} p99:  ${stats.latency.p99.toFixed(0).padEnd(10)}  ║
║    max:  ${String(stats.latency.max.toFixed(0)).padEnd(10)} avg:  ${stats.latency.avg.toFixed(0).padEnd(10)}                      ║
╠════════════════════════════════════════════════════════════════════╣
║  CAPACITY                                                          ║
║    Max Concurrent Games:  ${String(stats.capacity.max_concurrent_games).padEnd(10)}                         ║
║    Avg Concurrent Games:  ${stats.capacity.avg_concurrent_games.toFixed(1).padEnd(10)}                         ║
║    Throughput:            ${stats.throughput.rps} requests/second               ║
╠════════════════════════════════════════════════════════════════════╣
║  FAILURE CLASSIFICATION                                            ║
║    Contract Failures:     ${String(stats.classification.contract_failures).padEnd(10)}                         ║
║    Capacity Failures:     ${String(stats.classification.capacity_failures).padEnd(10)}                         ║
║    Lifecycle Mismatches:  ${String(stats.classification.lifecycle_mismatches).padEnd(10)}                         ║
╠════════════════════════════════════════════════════════════════════╣
║  SLO COMPLIANCE                                                    ║
║    p95 < ${SLO_TARGETS.p95_latency_ms}ms:           ${passIcon(sloResults.p95_latency.passed).padEnd(12)} (${stats.latency.p95.toFixed(0)}ms)             ║
║    p99 < ${SLO_TARGETS.p99_latency_ms}ms:          ${passIcon(sloResults.p99_latency.passed).padEnd(12)} (${stats.latency.p99.toFixed(0)}ms)             ║
║    Error Rate < ${SLO_TARGETS.error_rate_percent}%:     ${passIcon(sloResults.error_rate.passed).padEnd(12)} (${stats.requests.error_rate.toFixed(2)}%)            ║
║    Contract Failures = 0: ${passIcon(sloResults.contract_failures.passed).padEnd(12)} (${stats.classification.contract_failures})                ║
╠════════════════════════════════════════════════════════════════════╣
║  OVERALL:  ${sloResults.all_passed ? '✅ ALL SLOs PASSED' : '❌ SLO VIOLATIONS DETECTED'}                                    ║
╚════════════════════════════════════════════════════════════════════╝
`);
}

/**
 * Build summary object for JSON output
 */
function buildSummary(stats, sloResults, resultsFile) {
  return {
    timestamp: new Date().toISOString(),
    source_file: path.basename(resultsFile),
    duration_minutes: parseFloat(stats.duration.minutes),

    requests: stats.requests,
    latency: stats.latency,
    capacity: stats.capacity,
    throughput: stats.throughput,
    classification: stats.classification,

    slo: sloResults,
    passed: sloResults.all_passed,

    targets: SLO_TARGETS,

    recommendations: generateRecommendations(stats, sloResults),
  };
}

/**
 * Generate actionable recommendations based on results
 */
function generateRecommendations(stats, sloResults) {
  const recommendations = [];

  if (!sloResults.p95_latency.passed) {
    recommendations.push({
      severity: 'high',
      area: 'latency',
      message: `p95 latency (${stats.latency.p95.toFixed(0)}ms) exceeds target (${SLO_TARGETS.p95_latency_ms}ms). Investigate slow endpoints.`,
    });
  }

  if (!sloResults.error_rate.passed) {
    recommendations.push({
      severity: 'high',
      area: 'reliability',
      message: `Error rate (${stats.requests.error_rate.toFixed(2)}%) exceeds target (${SLO_TARGETS.error_rate_percent}%). Check server logs for errors.`,
    });
  }

  if (stats.classification.contract_failures > 0) {
    recommendations.push({
      severity: 'critical',
      area: 'contract',
      message: `${stats.classification.contract_failures} contract failures detected. These indicate API contract violations that must be fixed.`,
    });
  }

  if (stats.classification.lifecycle_mismatches > 0) {
    recommendations.push({
      severity: 'medium',
      area: 'lifecycle',
      message: `${stats.classification.lifecycle_mismatches} game ID lifecycle mismatches. Check game cleanup and expiration logic.`,
    });
  }

  if (stats.capacity.max_concurrent_games < SLO_TARGETS.concurrent_games * 0.5) {
    recommendations.push({
      severity: 'medium',
      area: 'capacity',
      message: `Max concurrent games (${stats.capacity.max_concurrent_games}) is below target (${SLO_TARGETS.concurrent_games}). Consider scaling resources.`,
    });
  }

  if (recommendations.length === 0) {
    recommendations.push({
      severity: 'info',
      area: 'general',
      message: 'All SLOs passed. System is performing within expected parameters.',
    });
  }

  return recommendations;
}

// Main execution
try {
  console.log(`\nAnalyzing results from: ${resultsFile}\n`);

  const entries = parseK6JsonStream(resultsFile);
  console.log(`Parsed ${entries.length} data points`);

  const metrics = extractMetrics(entries);
  const stats = calculateStatistics(metrics);
  const sloResults = evaluateSLOs(stats);

  printReport(stats, sloResults);

  // Write summary if output file specified
  const summaryPath = outputFile || resultsFile.replace('.json', '_summary.json');
  const summary = buildSummary(stats, sloResults, resultsFile);
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  console.log(`Summary written to: ${summaryPath}`);

  // Exit with error code if SLOs failed
  process.exit(sloResults.all_passed ? 0 : 1);
} catch (error) {
  console.error('Error analyzing results:', error.message);
  process.exit(1);
}