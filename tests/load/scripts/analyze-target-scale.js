#!/usr/bin/env node

/**
 * RingRift Target Scale Load Test Results Analyzer
 *
 * Analyzes k6 JSON output and validates against production targets:
 *   - 100 concurrent games
 *   - 300 concurrent players
 *   - p95 latency < 500ms
 *   - Error rate < 1%
 *
 * Usage:
 *   node analyze-target-scale.js <results.json> [output-summary.json]
 *
 * Exit Codes:
 *   0 - All targets met
 *   1 - One or more targets failed
 *   2 - Analysis error
 */

const fs = require('fs');
const path = require('path');

// Target scale targets from PROJECT_GOALS.md
const TARGET_SCALE = {
  concurrent_games: 100,
  concurrent_players: 300,
  players_per_game: 3,
};

// SLO thresholds
const SLO_TARGETS = {
  p95_latency_ms: 500,
  p99_latency_ms: 2000,
  error_rate_percent: 1.0,
  ws_connection_success_percent: 99.0,
  contract_failures_max: 0,
};

// Get command line arguments
const args = process.argv.slice(2);
const resultsFile = args[0];
const outputFile = args[1];

if (!resultsFile) {
  console.error('Usage: node analyze-target-scale.js <results.json> [output-summary.json]');
  process.exit(2);
}

if (!fs.existsSync(resultsFile)) {
  console.error(`Results file not found: ${resultsFile}`);
  process.exit(2);
}

// Load target configuration
let configPath = path.join(__dirname, '../configs/target-scale.json');
let config = {};
try {
  if (fs.existsSync(configPath)) {
    config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  }
} catch (e) {
  console.warn(`Warning: Could not load config from ${configPath}`);
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
    } catch {
      // Skip malformed lines
    }
  }
  return entries;
}

/**
 * Extract metrics from k6 data points with phase awareness
 */
function extractMetrics(entries) {
  const metrics = {
    http_req_duration: [],
    http_req_failed: [],
    game_creation_latency: [],
    game_state_check: [],
    concurrent_games: [],
    ws_connection_success: [],
    vus: [],
    contract_failures: 0,
    capacity_failures: 0,
    lifecycle_mismatches: 0,
    total_requests: 0,
    failed_requests: 0,
    startTime: null,
    endTime: null,
    // Phase-specific metrics
    phases: {
      warmup: { start: null, end: null, latencies: [], vus: [] },
      rampToHalf: { start: null, end: null, latencies: [], vus: [] },
      steadyHalf: { start: null, end: null, latencies: [], vus: [] },
      rampToFull: { start: null, end: null, latencies: [], vus: [] },
      steadyFull: { start: null, end: null, latencies: [], vus: [] },
      rampDown: { start: null, end: null, latencies: [], vus: [] },
    },
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
          metrics.http_req_duration.push({ value: data.value, time: timestamp });
          metrics.total_requests++;
        }
        break;

      case 'http_req_failed':
        if (data.value === 1) {
          metrics.failed_requests++;
        }
        break;

      case 'vus':
        if (typeof data.value === 'number') {
          metrics.vus.push({ value: data.value, time: timestamp });
        }
        break;

      case 'concurrent_active_games':
        if (typeof data.value === 'number') {
          metrics.concurrent_games.push({ value: data.value, time: timestamp });
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
  const latencyValues = metrics.http_req_duration.map((d) => d.value);
  const sortedLatencies = [...latencyValues].sort((a, b) => a - b);
  const vusValues = metrics.vus.map((d) => d.value);
  const concurrentGamesValues = metrics.concurrent_games.map((d) => d.value);

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

    capacity: {
      max_concurrent_vus: vusValues.length > 0 ? Math.max(...vusValues) : 0,
      avg_concurrent_vus:
        vusValues.length > 0
          ? vusValues.reduce((a, b) => a + b, 0) / vusValues.length
          : 0,
      max_concurrent_games:
        concurrentGamesValues.length > 0 ? Math.max(...concurrentGamesValues) : 0,
      avg_concurrent_games:
        concurrentGamesValues.length > 0
          ? concurrentGamesValues.reduce((a, b) => a + b, 0) / concurrentGamesValues.length
          : 0,
      // Explicit sample flags so callers can distinguish "metric missing" from "0".
      has_concurrent_vus_samples: vusValues.length > 0,
      has_concurrent_games_samples: concurrentGamesValues.length > 0,
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
 * Validate against target scale requirements
 */
function validateTargetScale(stats) {
  const hasConcurrentVusSamples =
    typeof stats.capacity.has_concurrent_vus_samples === 'boolean'
      ? stats.capacity.has_concurrent_vus_samples
      : stats.capacity.max_concurrent_vus > 0;

  const hasConcurrentGamesSamples =
    typeof stats.capacity.has_concurrent_games_samples === 'boolean'
      ? stats.capacity.has_concurrent_games_samples
      : stats.capacity.max_concurrent_games > 0;

  const validation = {
    concurrent_players: {
      target: TARGET_SCALE.concurrent_players,
      actual: stats.capacity.max_concurrent_vus,
      passed:
        hasConcurrentVusSamples &&
        stats.capacity.max_concurrent_vus >= TARGET_SCALE.concurrent_players,
      margin_percent:
        ((stats.capacity.max_concurrent_vus - TARGET_SCALE.concurrent_players) /
          TARGET_SCALE.concurrent_players) *
        100,
      has_samples: hasConcurrentVusSamples,
      reason: hasConcurrentVusSamples
        ? stats.capacity.max_concurrent_vus >= TARGET_SCALE.concurrent_players
          ? 'target_met'
          : 'target_not_met'
        : 'missing_metric',
    },

    concurrent_games: {
      target: TARGET_SCALE.concurrent_games,
      actual: stats.capacity.max_concurrent_games,
      passed:
        hasConcurrentGamesSamples &&
        stats.capacity.max_concurrent_games >= TARGET_SCALE.concurrent_games,
      margin_percent:
        ((stats.capacity.max_concurrent_games - TARGET_SCALE.concurrent_games) /
          TARGET_SCALE.concurrent_games) *
        100,
      has_samples: hasConcurrentGamesSamples,
      reason: hasConcurrentGamesSamples
        ? stats.capacity.max_concurrent_games >= TARGET_SCALE.concurrent_games
          ? 'target_met'
          : 'target_not_met'
        : 'missing_metric',
    },

    p95_latency: {
      target_ms: SLO_TARGETS.p95_latency_ms,
      actual_ms: stats.latency.p95,
      passed: stats.latency.p95 < SLO_TARGETS.p95_latency_ms,
      margin_percent:
        stats.latency.p95 > 0
          ? ((SLO_TARGETS.p95_latency_ms - stats.latency.p95) / SLO_TARGETS.p95_latency_ms) * 100
          : 100,
    },

    p99_latency: {
      target_ms: SLO_TARGETS.p99_latency_ms,
      actual_ms: stats.latency.p99,
      passed: stats.latency.p99 < SLO_TARGETS.p99_latency_ms,
      margin_percent:
        stats.latency.p99 > 0
          ? ((SLO_TARGETS.p99_latency_ms - stats.latency.p99) / SLO_TARGETS.p99_latency_ms) * 100
          : 100,
    },

    error_rate: {
      target_percent: SLO_TARGETS.error_rate_percent,
      actual_percent: stats.requests.error_rate,
      passed: stats.requests.error_rate < SLO_TARGETS.error_rate_percent,
    },

    contract_failures: {
      target: SLO_TARGETS.contract_failures_max,
      actual: stats.classification.contract_failures,
      passed: stats.classification.contract_failures <= SLO_TARGETS.contract_failures_max,
    },
  };

  validation.all_passed = Object.values(validation)
    .filter(
      (v) =>
        v &&
        typeof v === 'object' &&
        Object.prototype.hasOwnProperty.call(v, 'passed')
    )
    .every((r) => r.passed);

  // Classify overall result
  validation.overall_status = validation.all_passed
    ? 'TARGET_SCALE_VALIDATED'
    : 'SCALE_VALIDATION_FAILED';

  return validation;
}

/**
 * Print formatted target scale report to console
 */
function printReport(stats, validation) {
  const passIcon = (passed) => (passed ? '✅ PASS' : '❌ FAIL');
  const marginStr = (margin) => (margin >= 0 ? `+${margin.toFixed(1)}%` : `${margin.toFixed(1)}%`);

  console.log(`
╔════════════════════════════════════════════════════════════════════╗
║           RingRift TARGET SCALE Capacity Report                    ║
╠════════════════════════════════════════════════════════════════════╣
║  Target: 100 concurrent games / 300 concurrent players              ║
╠════════════════════════════════════════════════════════════════════╣
║  Duration:        ${stats.duration.formatted.padEnd(20)}                       ║
║  Total Requests:  ${String(stats.requests.total).padEnd(20)}                       ║
║  Failed Requests: ${String(stats.requests.failed).padEnd(20)}                       ║
║  Error Rate:      ${stats.requests.error_rate.toFixed(2)}%                                           ║
╠════════════════════════════════════════════════════════════════════╣
║  CAPACITY TARGETS                                                  ║
║    Concurrent Players: ${String(stats.capacity.max_concurrent_vus).padEnd(5)}/${TARGET_SCALE.concurrent_players}  ${passIcon(validation.concurrent_players.passed).padEnd(10)}             ║
║    Concurrent Games:   ${String(stats.capacity.max_concurrent_games).padEnd(5)}/${TARGET_SCALE.concurrent_games}   ${passIcon(validation.concurrent_games.passed).padEnd(10)}             ║
╠════════════════════════════════════════════════════════════════════╣
║  LATENCY (ms)                                                      ║
║    p50:  ${String(stats.latency.p50.toFixed(0)).padEnd(10)} p95:  ${String(stats.latency.p95.toFixed(0)).padEnd(10)} p99:  ${stats.latency.p99.toFixed(0).padEnd(10)}  ║
║    max:  ${String(stats.latency.max.toFixed(0)).padEnd(10)} avg:  ${stats.latency.avg.toFixed(0).padEnd(10)}                      ║
╠════════════════════════════════════════════════════════════════════╣
║  THROUGHPUT                                                        ║
║    RPS: ${stats.throughput.rps} requests/second                                 ║
╠════════════════════════════════════════════════════════════════════╣
║  FAILURE CLASSIFICATION                                            ║
║    Contract Failures:     ${String(stats.classification.contract_failures).padEnd(10)}                         ║
║    Capacity Failures:     ${String(stats.classification.capacity_failures).padEnd(10)}                         ║
║    Lifecycle Mismatches:  ${String(stats.classification.lifecycle_mismatches).padEnd(10)}                         ║
╠════════════════════════════════════════════════════════════════════╣
║  TARGET SCALE VALIDATION                                           ║
║    Concurrent Players: ${stats.capacity.max_concurrent_vus}/${TARGET_SCALE.concurrent_players} ......... ${passIcon(validation.concurrent_players.passed)}        ║
║    Concurrent Games:   ${stats.capacity.max_concurrent_games}/${TARGET_SCALE.concurrent_games} ........... ${passIcon(validation.concurrent_games.passed)}        ║
║    p95 Latency:        ${stats.latency.p95.toFixed(0)}ms < ${SLO_TARGETS.p95_latency_ms}ms ..... ${passIcon(validation.p95_latency.passed)}        ║
║    p99 Latency:        ${stats.latency.p99.toFixed(0)}ms < ${SLO_TARGETS.p99_latency_ms}ms ... ${passIcon(validation.p99_latency.passed)}        ║
║    Error Rate:         ${stats.requests.error_rate.toFixed(2)}% < ${SLO_TARGETS.error_rate_percent}% ..... ${passIcon(validation.error_rate.passed)}        ║
║    Contract Failures:  ${stats.classification.contract_failures} == 0 ........... ${passIcon(validation.contract_failures.passed)}        ║
╠════════════════════════════════════════════════════════════════════╣
║  OVERALL: ${validation.all_passed ? '✅ TARGET SCALE VALIDATED' : '❌ SCALE VALIDATION FAILED'}                                   ║
╚════════════════════════════════════════════════════════════════════╝
`);

  // Print recommendations if failed
  if (!validation.all_passed) {
    console.log('\n🔧 RECOMMENDATIONS:\n');
    if (!validation.concurrent_players.passed) {
      console.log(
        `  • Concurrent players (${stats.capacity.max_concurrent_vus}) did not reach target (${TARGET_SCALE.concurrent_players}).`
      );
      console.log('    → Check if VU count in k6 is configured correctly.');
      console.log('    → Ensure server can handle more connections.');
    }
    if (!validation.concurrent_games.passed) {
      if (validation.concurrent_games.has_samples === false) {
        console.log(
          '  • concurrent_active_games metric was missing from k6 output; capacity analysis could not determine max concurrent games.'
        );
        console.log('    → Ensure the load scenario emits the concurrent_active_games gauge.');
        console.log(
          '    → Verify that the intended scenario (e.g. concurrent-games or websocket-gameplay) was actually executed.'
        );
      } else {
        console.log(
          `  • Concurrent games (${stats.capacity.max_concurrent_games}) did not reach target (${TARGET_SCALE.concurrent_games}).`
        );
        console.log('    → Check game creation rate and lifecycle.');
        console.log('    → Investigate if games are being cleaned up too quickly.');
      }
    }
    if (!validation.p95_latency.passed) {
      console.log(
        `  • p95 latency (${stats.latency.p95.toFixed(0)}ms) exceeded target (${SLO_TARGETS.p95_latency_ms}ms).`
      );
      console.log('    → Profile server endpoints for bottlenecks.');
      console.log('    → Check database query performance.');
      console.log('    → Consider caching frequently accessed data.');
    }
    if (!validation.error_rate.passed) {
      console.log(
        `  • Error rate (${stats.requests.error_rate.toFixed(2)}%) exceeded target (${SLO_TARGETS.error_rate_percent}%).`
      );
      console.log('    → Check server logs for error patterns.');
      console.log('    → Verify connection pool sizes.');
      console.log('    → Check for rate limiting issues.');
    }
    if (!validation.contract_failures.passed) {
      console.log(`  • Contract failures (${stats.classification.contract_failures}) detected.`);
      console.log('    → These indicate API contract violations that must be fixed.');
      console.log('    → Review error responses for schema mismatches.');
    }
    console.log('\n');
  }
}

/**
 * Build summary object for JSON output
 */
function buildSummary(stats, validation, resultsFile) {
  return {
    timestamp: new Date().toISOString(),
    test_type: 'target-scale',
    source_file: path.basename(resultsFile),
    duration_minutes: parseFloat(stats.duration.minutes),

    targets: TARGET_SCALE,
    slo_targets: SLO_TARGETS,

    requests: stats.requests,
    latency: stats.latency,
    capacity: stats.capacity,
    throughput: stats.throughput,
    classification: stats.classification,

    validation: validation,
    all_targets_met: validation.all_passed,
    overall_status: validation.overall_status,

    recommendations: generateRecommendations(stats, validation),
  };
}

/**
 * Generate actionable recommendations based on results
 */
function generateRecommendations(stats, validation) {
  const recommendations = [];

  if (!validation.concurrent_players.passed) {
    recommendations.push({
      severity: 'high',
      area: 'capacity',
      target: 'concurrent_players',
      message: `Max concurrent players (${stats.capacity.max_concurrent_vus}) did not reach target (${TARGET_SCALE.concurrent_players}). Verify k6 VU configuration and server connection limits.`,
    });
  }

  if (!validation.concurrent_games.passed) {
    if (validation.concurrent_games.has_samples === false) {
      recommendations.push({
        severity: 'high',
        area: 'capacity',
        target: 'concurrent_games',
        message:
          'Metric concurrent_active_games was missing from k6 output; verify load harness instrumentation and that the intended scenario emitted this gauge.',
      });
    } else {
      recommendations.push({
        severity: 'high',
        area: 'capacity',
        target: 'concurrent_games',
        message: `Max concurrent games (${stats.capacity.max_concurrent_games}) did not reach target (${TARGET_SCALE.concurrent_games}). Check game lifecycle and cleanup timing.`,
      });
    }
  }

  if (!validation.p95_latency.passed) {
    recommendations.push({
      severity: 'high',
      area: 'latency',
      target: 'p95_latency',
      message: `p95 latency (${stats.latency.p95.toFixed(0)}ms) exceeded target (${SLO_TARGETS.p95_latency_ms}ms). Profile slow endpoints and optimize database queries.`,
    });
  }

  if (!validation.p99_latency.passed) {
    recommendations.push({
      severity: 'medium',
      area: 'latency',
      target: 'p99_latency',
      message: `p99 latency (${stats.latency.p99.toFixed(0)}ms) exceeded target (${SLO_TARGETS.p99_latency_ms}ms). Investigate tail latency causes.`,
    });
  }

  if (!validation.error_rate.passed) {
    recommendations.push({
      severity: 'high',
      area: 'reliability',
      target: 'error_rate',
      message: `Error rate (${stats.requests.error_rate.toFixed(2)}%) exceeded target (${SLO_TARGETS.error_rate_percent}%). Check server logs for error patterns.`,
    });
  }

  if (!validation.contract_failures.passed) {
    recommendations.push({
      severity: 'critical',
      area: 'contract',
      target: 'contract_failures',
      message: `${stats.classification.contract_failures} contract failures detected. These indicate API contract violations that must be fixed.`,
    });
  }

  if (stats.classification.lifecycle_mismatches > 0) {
    recommendations.push({
      severity: 'medium',
      area: 'lifecycle',
      target: 'lifecycle_mismatches',
      message: `${stats.classification.lifecycle_mismatches} game ID lifecycle mismatches. Check game cleanup and expiration logic.`,
    });
  }

  if (recommendations.length === 0) {
    recommendations.push({
      severity: 'info',
      area: 'general',
      target: 'all',
      message: 'All target scale requirements met. System is validated for production load.',
    });
  }

  return recommendations;
}

// Main execution
try {
  console.log(`\nAnalyzing target scale results from: ${resultsFile}\n`);

  const entries = parseK6JsonStream(resultsFile);
  console.log(`Parsed ${entries.length} data points`);

  const metrics = extractMetrics(entries);
  const stats = calculateStatistics(metrics);
  const validation = validateTargetScale(stats);

  printReport(stats, validation);

  // Write summary
  const summaryPath = outputFile || resultsFile.replace('.json', '_summary.json');
  const summary = buildSummary(stats, validation, resultsFile);
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  console.log(`Summary written to: ${summaryPath}`);

  // Exit with appropriate code
  if (validation.all_passed) {
    console.log('\n✅ Target scale validation PASSED\n');
    process.exit(0);
  } else {
    console.log('\n❌ Target scale validation FAILED\n');
    process.exit(1);
  }
} catch (error) {
  console.error('Error analyzing results:', error.message);
  console.error(error.stack);
  process.exit(2);
}