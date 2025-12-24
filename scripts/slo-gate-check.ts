#!/usr/bin/env npx ts-node
/**
 * RingRift SLO Gate Check Script
 *
 * A standalone SLO gate validation script for CI/CD integration.
 * Analyzes load test results against SLO thresholds and produces
 * structured output suitable for automated decision making.
 *
 * Usage:
 *   npx ts-node scripts/slo-gate-check.ts --results-file <path> [options]
 *
 * Options:
 *   --results-file <path>    Path to k6 JSON output file (required)
 *   --env <environment>      SLO environment: staging | production (default: staging)
 *   --format <format>        Output format: console | json | summary (default: console)
 *   --fail-on-breach         Exit with code 1 if any critical SLO is breached
 *   --gate-type <type>       Gate type: staging-promotion | production-readiness
 *   --output-file <path>     Write JSON report to file
 *   --help                   Show help message
 *
 * Exit Codes:
 *   0 - All critical SLOs passed (gate approved)
 *   1 - One or more critical SLOs breached (gate blocked)
 *   2 - Error (file not found, parse error, etc.)
 *
 * Examples:
 *   npx ts-node scripts/slo-gate-check.ts --results-file tests/load/results/baseline.json
 *   npx ts-node scripts/slo-gate-check.ts --results-file results.json --env production --fail-on-breach
 *   npx ts-node scripts/slo-gate-check.ts --results-file results.json --format json --output-file slo-report.json
 *
 * @see docs/production/PRODUCTION_VALIDATION_GATE.md
 * @see docs/planning/SLO_THRESHOLD_ALIGNMENT_AUDIT.md
 */

import * as fs from 'fs';

// SLO definitions with environment-specific overrides
interface SLODefinition {
  name: string;
  target: number;
  unit: 'percent' | 'ms' | 'count' | 'games' | 'players';
  priority: 'critical' | 'high' | 'medium';
  measurement: string;
  operator: '<=' | '>=' | '<' | '>' | '=';
}

interface SLOResult {
  name: string;
  target: number;
  actual: number;
  unit: string;
  passed: boolean;
  priority: string;
  note?: string;
  margin?: number; // percentage margin from target
}

interface GateReport {
  timestamp: string;
  environment: string;
  gateType: string;
  resultsFile: string;
  overall: {
    passed: boolean;
    passedCount: number;
    totalCount: number;
    criticalBreaches: number;
    highBreaches: number;
    mediumBreaches: number;
  };
  decision: {
    gateStatus: 'APPROVED' | 'BLOCKED' | 'CONDITIONAL';
    reason: string;
    recommendations: string[];
  };
  slos: Record<string, SLOResult>;
  metrics: {
    totalRequests: number;
    maxVUs: number;
    testDuration?: number;
  };
}

// Default SLO definitions (aligned with tests/load/configs/slo-definitions.json)
const getSLODefinitions = (env: string): Record<string, SLODefinition> => {
  const isProduction = env === 'production';

  return {
    availability: {
      name: 'Service Availability',
      target: 99.9,
      unit: 'percent',
      priority: 'critical',
      measurement: 'calculated from true_errors',
      operator: '>=',
    },
    error_rate: {
      name: 'Error Rate',
      target: isProduction ? 0.5 : 1.0,
      unit: 'percent',
      priority: 'critical',
      measurement: 'true_errors_total / total_requests',
      operator: '<=',
    },
    true_error_rate: {
      name: 'True Error Rate',
      target: isProduction ? 0.2 : 0.5,
      unit: 'percent',
      priority: 'critical',
      measurement: 'true_errors_total (excludes 401/429)',
      operator: '<=',
    },
    latency_api_p95: {
      name: 'API Latency (p95)',
      target: isProduction ? 500 : 800,
      unit: 'ms',
      priority: 'high',
      measurement: 'http_req_duration p95',
      operator: '<=',
    },
    latency_api_p99: {
      name: 'API Latency (p99)',
      target: 2000,
      unit: 'ms',
      priority: 'medium',
      measurement: 'http_req_duration p99',
      operator: '<=',
    },
    latency_move_e2e: {
      name: 'Move Latency E2E (p95)',
      target: isProduction ? 200 : 300,
      unit: 'ms',
      priority: 'high',
      measurement: 'move_latency p95',
      operator: '<=',
    },
    latency_ai_response: {
      name: 'AI Response Time (p95)',
      target: isProduction ? 1000 : 1500,
      unit: 'ms',
      priority: 'high',
      measurement: 'ai_response_time p95',
      operator: '<=',
    },
    contract_failures: {
      name: 'Contract Failures',
      target: 0,
      unit: 'count',
      priority: 'critical',
      measurement: 'contract_failures_total',
      operator: '<=',
    },
    lifecycle_mismatches: {
      name: 'Lifecycle Mismatches',
      target: 0,
      unit: 'count',
      priority: 'critical',
      measurement: 'id_lifecycle_mismatches_total',
      operator: '<=',
    },
    websocket_connection_success: {
      name: 'WebSocket Connection Success',
      target: isProduction ? 99.5 : 99.0,
      unit: 'percent',
      priority: 'high',
      measurement: 'ws_connection_success_rate',
      operator: '>=',
    },
    move_stall_rate: {
      name: 'Move Stall Rate',
      target: isProduction ? 0.2 : 0.5,
      unit: 'percent',
      priority: 'high',
      measurement: 'stalled_moves / total_moves',
      operator: '<=',
    },
    concurrent_games: {
      name: 'Concurrent Games',
      target: isProduction ? 100 : 20,
      unit: 'games',
      priority: 'high',
      measurement: 'max concurrent_active_games',
      operator: '>=',
    },
    concurrent_players: {
      name: 'Concurrent Players',
      target: isProduction ? 300 : 60,
      unit: 'players',
      priority: 'high',
      measurement: 'max concurrent VUs',
      operator: '>=',
    },
  };
};

// Parse command line arguments
interface CLIArgs {
  resultsFile?: string;
  env: string;
  format: 'console' | 'json' | 'summary';
  failOnBreach: boolean;
  gateType: string;
  outputFile?: string;
  help: boolean;
}

function parseArgs(args: string[]): CLIArgs {
  const result: CLIArgs = {
    env: 'staging',
    format: 'console',
    failOnBreach: false,
    gateType: 'staging-promotion',
    help: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--results-file':
        result.resultsFile = args[++i];
        break;
      case '--env':
        result.env = args[++i];
        break;
      case '--format':
        result.format = args[++i] as 'console' | 'json' | 'summary';
        break;
      case '--fail-on-breach':
        result.failOnBreach = true;
        break;
      case '--gate-type':
        result.gateType = args[++i];
        break;
      case '--output-file':
        result.outputFile = args[++i];
        break;
      case '--help':
      case '-h':
        result.help = true;
        break;
    }
  }

  return result;
}

// Parse k6 JSON results
interface ExtractedMetrics {
  latencies: number[];
  http_req_duration_p95?: number;
  http_req_duration_p99?: number;
  http_req_failed_rate?: number;
  failed_requests: number;
  total_requests: number;
  max_vus: number;
  ws_connecting_times: number[];
  move_latencies: number[];
  ai_response_times: number[];
  contract_failures: number;
  lifecycle_mismatches: number;
  stalled_moves: number;
  concurrent_games_max: number;
  true_errors_total: number;
  ws_connection_successes: number;
  ws_connection_attempts: number;
  has_true_errors_metric: boolean;
}

function parseResults(filePath: string): unknown[] {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n').filter((line) => line.trim());

  const results: unknown[] = [];
  let parseErrors = 0;

  for (const line of lines) {
    try {
      results.push(JSON.parse(line));
    } catch {
      parseErrors++;
    }
  }

  // If too many parse errors, try as single JSON object
  if (parseErrors > lines.length / 2) {
    try {
      return [JSON.parse(content)];
    } catch (error) {
      console.error('Error parsing results file:', (error as Error).message);
      process.exit(2);
    }
  }

  return results;
}

function extractMetrics(results: unknown[]): ExtractedMetrics {
  const metrics: ExtractedMetrics = {
    latencies: [],
    failed_requests: 0,
    total_requests: 0,
    max_vus: 0,
    ws_connecting_times: [],
    move_latencies: [],
    ai_response_times: [],
    contract_failures: 0,
    lifecycle_mismatches: 0,
    stalled_moves: 0,
    concurrent_games_max: 0,
    true_errors_total: 0,
    ws_connection_successes: 0,
    ws_connection_attempts: 0,
    has_true_errors_metric: false,
  };

  for (const entry of results) {
    const e = entry as Record<string, unknown>;

    // Handle k6 Point metrics
    if (e.type === 'Point') {
      const metric = e.metric as string;
      const data = e.data as Record<string, unknown> | undefined;
      const value = data?.value as number | undefined;

      if (value === undefined) continue;

      switch (metric) {
        case 'http_req_duration':
          metrics.latencies.push(value);
          metrics.total_requests++;
          break;
        case 'http_req_failed':
          if (value === 1) metrics.failed_requests++;
          break;
        case 'vus':
          if (value > metrics.max_vus) metrics.max_vus = value;
          break;
        case 'ws_connecting':
          metrics.ws_connecting_times.push(value);
          break;
        case 'move_latency':
        case 'ws_move_rtt_ms':
          metrics.move_latencies.push(value);
          break;
        case 'ai_response_time':
        case 'ai_move_latency_ms':
          metrics.ai_response_times.push(value);
          break;
        case 'contract_failures_total':
          metrics.contract_failures = Math.max(metrics.contract_failures, value);
          break;
        case 'id_lifecycle_mismatches_total':
          metrics.lifecycle_mismatches = Math.max(metrics.lifecycle_mismatches, value);
          break;
        case 'stalled_moves_total':
        case 'ws_move_stalled_total':
          metrics.stalled_moves = Math.max(metrics.stalled_moves, value);
          break;
        case 'true_errors_total':
          metrics.true_errors_total += value;
          metrics.has_true_errors_metric = true;
          break;
        case 'concurrent_active_games':
        case 'concurrent_games':
          if (value > metrics.concurrent_games_max) {
            metrics.concurrent_games_max = value;
          }
          break;
        case 'websocket_connection_success':
          metrics.ws_connection_successes++;
          metrics.ws_connection_attempts++;
          break;
      }
    }

    // Handle k6 summary format (from handleSummary)
    const entryMetrics = e.metrics as Record<string, Record<string, unknown>> | undefined;
    if (entryMetrics) {
      if (entryMetrics.http_req_duration?.values) {
        const v = entryMetrics.http_req_duration.values as Record<string, number>;
        metrics.http_req_duration_p95 = v['p(95)'];
        metrics.http_req_duration_p99 = v['p(99)'];
      }

      if (entryMetrics.http_req_failed?.values) {
        const v = entryMetrics.http_req_failed.values as Record<string, number>;
        metrics.http_req_failed_rate = v.rate;
      }

      if (entryMetrics.vus?.values) {
        const v = entryMetrics.vus.values as Record<string, number>;
        metrics.max_vus = Math.max(metrics.max_vus, v.max || 0);
      }

      if (entryMetrics.true_errors_total?.values && metrics.true_errors_total === 0) {
        const v = entryMetrics.true_errors_total.values as Record<string, number>;
        metrics.true_errors_total = v.count || 0;
        metrics.has_true_errors_metric = true;
      }
    }
  }

  return metrics;
}

function percentile(arr: number[], p: number): number {
  if (!arr || arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor(sorted.length * p);
  return sorted[Math.min(idx, sorted.length - 1)] || 0;
}

function evaluateSLO(key: string, definition: SLODefinition, metrics: ExtractedMetrics): SLOResult {
  let actual: number;
  let note: string | undefined;

  switch (key) {
    case 'availability': {
      const errorRate =
        metrics.total_requests > 0 ? (metrics.true_errors_total / metrics.total_requests) * 100 : 0;
      actual = 100 - errorRate;
      if (metrics.total_requests === 0) note = 'No request data collected';
      break;
    }
    case 'error_rate':
    case 'true_error_rate': {
      actual =
        metrics.total_requests > 0 ? (metrics.true_errors_total / metrics.total_requests) * 100 : 0;
      if (!metrics.has_true_errors_metric) note = 'true_errors_total not reported';
      break;
    }
    case 'latency_api_p95':
      actual = metrics.http_req_duration_p95 || percentile(metrics.latencies, 0.95);
      break;
    case 'latency_api_p99':
      actual = metrics.http_req_duration_p99 || percentile(metrics.latencies, 0.99);
      break;
    case 'latency_move_e2e':
      actual = percentile(metrics.move_latencies, 0.95);
      if (metrics.move_latencies.length === 0) note = 'No move data collected';
      break;
    case 'latency_ai_response':
      actual = percentile(metrics.ai_response_times, 0.95);
      if (metrics.ai_response_times.length === 0) note = 'No AI data collected';
      break;
    case 'contract_failures':
      actual = metrics.contract_failures;
      break;
    case 'lifecycle_mismatches':
      actual = metrics.lifecycle_mismatches;
      break;
    case 'websocket_connection_success':
      actual =
        metrics.ws_connection_attempts > 0
          ? (metrics.ws_connection_successes / metrics.ws_connection_attempts) * 100
          : 100;
      if (metrics.ws_connection_attempts === 0) note = 'No WS connection data';
      break;
    case 'move_stall_rate': {
      const totalMoves = metrics.move_latencies.length;
      actual = totalMoves > 0 ? (metrics.stalled_moves / totalMoves) * 100 : 0;
      if (totalMoves === 0) note = 'No move data collected';
      break;
    }
    case 'concurrent_games':
      actual = metrics.concurrent_games_max;
      if (actual === 0) note = 'No concurrent games data';
      break;
    case 'concurrent_players':
      actual = metrics.max_vus;
      if (actual === 0) note = 'No VU data collected';
      break;
    default:
      actual = 0;
      note = 'Unknown metric';
  }

  // Evaluate pass/fail based on operator
  let passed: boolean;
  switch (definition.operator) {
    case '>=':
      passed = actual >= definition.target || (note !== undefined && actual === 0);
      break;
    case '>':
      passed = actual > definition.target;
      break;
    case '<=':
      passed = actual <= definition.target;
      break;
    case '<':
      passed = actual < definition.target;
      break;
    case '=':
      passed = actual === definition.target;
      break;
    default:
      passed = actual <= definition.target;
  }

  // Calculate margin from target
  let margin: number;
  if (definition.target === 0) {
    margin = actual === 0 ? 100 : -100;
  } else if (definition.operator === '>=' || definition.operator === '>') {
    margin = ((actual - definition.target) / definition.target) * 100;
  } else {
    margin = ((definition.target - actual) / definition.target) * 100;
  }

  const result: SLOResult = {
    name: definition.name,
    target: definition.target,
    actual: Math.round(actual * 1000) / 1000, // Round to 3 decimal places
    unit: definition.unit,
    passed,
    priority: definition.priority,
    margin: Math.round(margin * 10) / 10,
  };

  if (note !== undefined) {
    result.note = note;
  }

  return result;
}

function generateReport(
  args: CLIArgs,
  sloResults: Record<string, SLOResult>,
  metrics: ExtractedMetrics
): GateReport {
  const results = Object.values(sloResults);
  const passedCount = results.filter((r) => r.passed).length;
  const criticalBreaches = results.filter((r) => !r.passed && r.priority === 'critical').length;
  const highBreaches = results.filter((r) => !r.passed && r.priority === 'high').length;
  const mediumBreaches = results.filter((r) => !r.passed && r.priority === 'medium').length;

  // Determine gate status
  let gateStatus: 'APPROVED' | 'BLOCKED' | 'CONDITIONAL';
  let reason: string;
  const recommendations: string[] = [];

  if (criticalBreaches > 0) {
    gateStatus = 'BLOCKED';
    reason = `${criticalBreaches} critical SLO(s) breached - gate blocked`;
    recommendations.push('Investigate and fix critical SLO breaches before deployment');
    recommendations.push('Check error logs for root cause analysis');
  } else if (highBreaches > 0) {
    gateStatus = 'CONDITIONAL';
    reason = `${highBreaches} high-priority SLO(s) breached - manual approval required`;
    recommendations.push('Review high-priority breaches with team lead');
    recommendations.push('Consider performance optimization before production');
  } else {
    gateStatus = 'APPROVED';
    reason = 'All critical and high-priority SLOs passed';
    if (mediumBreaches > 0) {
      recommendations.push(`${mediumBreaches} medium-priority SLO(s) should be reviewed`);
    }
  }

  return {
    timestamp: new Date().toISOString(),
    environment: args.env,
    gateType: args.gateType,
    resultsFile: args.resultsFile || 'unknown',
    overall: {
      passed: criticalBreaches === 0,
      passedCount,
      totalCount: results.length,
      criticalBreaches,
      highBreaches,
      mediumBreaches,
    },
    decision: {
      gateStatus,
      reason,
      recommendations,
    },
    slos: sloResults,
    metrics: {
      totalRequests: metrics.total_requests,
      maxVUs: metrics.max_vus,
    },
  };
}

function formatConsole(report: GateReport): void {
  const statusIcon =
    report.decision.gateStatus === 'APPROVED'
      ? '✅'
      : report.decision.gateStatus === 'BLOCKED'
        ? '❌'
        : '⚠️';

  console.log(`
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       RingRift SLO Gate Check Report                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Gate Type:     ${report.gateType.padEnd(20)} Environment: ${report.environment.padEnd(15)} ║
║  Status:        ${statusIcon} ${report.decision.gateStatus.padEnd(26)}                        ║
║  SLOs Passed:   ${report.overall.passedCount}/${report.overall.totalCount}                                                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣`);

  // Group by priority
  const priorities = ['critical', 'high', 'medium'];
  for (const priority of priorities) {
    const slos = Object.entries(report.slos).filter(([, slo]) => slo.priority === priority);
    if (slos.length === 0) continue;

    console.log(`║  [${priority.toUpperCase()}]`.padEnd(80) + '║');

    for (const [, slo] of slos) {
      const status = slo.passed ? '✅' : '❌';
      const value = formatValue(slo.actual, slo.unit);
      const target = formatValue(slo.target, slo.unit);
      const marginStr = slo.margin !== undefined && slo.passed ? ` (+${slo.margin}%)` : '';
      const line = `║    ${status} ${slo.name.substring(0, 28).padEnd(28)} ${value.padStart(10)} / ${target.padStart(10)}${marginStr}`;
      console.log(line.padEnd(80) + '║');
    }
  }

  console.log(`╠═══════════════════════════════════════════════════════════════════════════════╣`);
  console.log(`║  Decision: ${report.decision.reason.substring(0, 65).padEnd(65)} ║`);

  if (report.decision.recommendations.length > 0) {
    console.log(
      `║  Recommendations:                                                              ║`
    );
    for (const rec of report.decision.recommendations) {
      console.log(`║    • ${rec.substring(0, 70).padEnd(70)} ║`);
    }
  }

  console.log(`╚═══════════════════════════════════════════════════════════════════════════════╝`);
}

function formatValue(value: number, unit: string): string {
  switch (unit) {
    case 'percent':
      return `${value}%`;
    case 'ms':
      return `${value}ms`;
    case 'games':
      return `${value} games`;
    case 'players':
      return `${value} VUs`;
    case 'count':
      return `${value}`;
    default:
      return `${value} ${unit}`;
  }
}

function formatSummary(report: GateReport): void {
  const status = report.decision.gateStatus;
  const icon = status === 'APPROVED' ? '✅' : status === 'BLOCKED' ? '❌' : '⚠️';

  console.log(`${icon} SLO Gate: ${status}`);
  console.log(`   Environment: ${report.environment}`);
  console.log(`   SLOs: ${report.overall.passedCount}/${report.overall.totalCount} passed`);

  if (report.overall.criticalBreaches > 0) {
    console.log(`   ❌ Critical breaches: ${report.overall.criticalBreaches}`);
  }
  if (report.overall.highBreaches > 0) {
    console.log(`   ⚠️  High breaches: ${report.overall.highBreaches}`);
  }

  console.log(`   Reason: ${report.decision.reason}`);
}

function showHelp(): void {
  console.log(`
RingRift SLO Gate Check Script

A standalone SLO gate validation script for CI/CD integration.

Usage:
  npx ts-node scripts/slo-gate-check.ts --results-file <path> [options]

Options:
  --results-file <path>    Path to k6 JSON output file (required)
  --env <environment>      SLO environment: staging | production (default: staging)
  --format <format>        Output format: console | json | summary (default: console)
  --fail-on-breach         Exit with code 1 if any critical SLO is breached
  --gate-type <type>       Gate type: staging-promotion | production-readiness
  --output-file <path>     Write JSON report to file
  --help                   Show help message

Exit Codes:
  0 - All critical SLOs passed (gate approved)
  1 - One or more critical SLOs breached (gate blocked)
  2 - Error (file not found, parse error, etc.)

Examples:
  npx ts-node scripts/slo-gate-check.ts --results-file tests/load/results/baseline.json
  npx ts-node scripts/slo-gate-check.ts --results-file results.json --env production --fail-on-breach
  npx ts-node scripts/slo-gate-check.ts --results-file results.json --format json --output-file slo-report.json
`);
}

// Main execution
function main(): void {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    showHelp();
    process.exit(0);
  }

  if (!args.resultsFile) {
    console.error('Error: --results-file is required');
    showHelp();
    process.exit(2);
  }

  if (!fs.existsSync(args.resultsFile)) {
    console.error(`Error: Results file not found: ${args.resultsFile}`);
    process.exit(2);
  }

  // Parse results and extract metrics
  const rawResults = parseResults(args.resultsFile);
  const metrics = extractMetrics(rawResults);

  // Get SLO definitions for environment
  const sloDefinitions = getSLODefinitions(args.env);

  // Evaluate each SLO
  const sloResults: Record<string, SLOResult> = {};
  for (const [key, definition] of Object.entries(sloDefinitions)) {
    sloResults[key] = evaluateSLO(key, definition, metrics);
  }

  // Generate report
  const report = generateReport(args, sloResults, metrics);

  // Output based on format
  switch (args.format) {
    case 'json':
      console.log(JSON.stringify(report, null, 2));
      break;
    case 'summary':
      formatSummary(report);
      break;
    default:
      formatConsole(report);
  }

  // Write to output file if specified
  if (args.outputFile) {
    fs.writeFileSync(args.outputFile, JSON.stringify(report, null, 2));
    console.log(`\n📄 Report saved to: ${args.outputFile}`);
  }

  // Exit with appropriate code
  if (args.failOnBreach && report.overall.criticalBreaches > 0) {
    process.exit(1);
  }

  process.exit(0);
}

main();
