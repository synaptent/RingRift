#!/usr/bin/env node

/**
 * RingRift SLO Verification Framework
 * 
 * Validates all Service Level Objectives from load test results.
 * Reads k6 JSON output and compares metrics against SLO definitions.
 * 
 * Usage:
 *   node verify-slos.js <results.json> [console|json|markdown] [--env staging|production]
 * 
 * Examples:
 *   node verify-slos.js results/baseline_staging_20251207.json
 *   node verify-slos.js results/baseline_staging_20251207.json json
 *   node verify-slos.js results/baseline_staging_20251207.json markdown --env production
 * 
 * Exit codes:
 *   0 - All SLOs passed
 *   1 - One or more SLOs breached
 *   2 - Error (file not found, parse error, etc.)
 */

const fs = require('fs');
const path = require('path');

// Load SLO definitions
const sloPath = path.join(__dirname, '../configs/slo-definitions.json');
let sloConfig;

try {
  sloConfig = JSON.parse(fs.readFileSync(sloPath, 'utf8'));
} catch (error) {
  console.error(`Error loading SLO definitions from ${sloPath}:`, error.message);
  process.exit(2);
}

// Parse command line arguments
const args = process.argv.slice(2);
const resultsFile = args[0];
const outputFormat = args.find(a => ['console', 'json', 'markdown'].includes(a)) || 'console';
const envArg = args.find(a => a.startsWith('--env'));
const environment = envArg ? args[args.indexOf(envArg) + 1] : 'staging';

if (!resultsFile || args.includes('--help') || args.includes('-h')) {
  console.log(`
RingRift SLO Verification Framework

Usage:
  node verify-slos.js <results.json> [format] [--env environment]

Arguments:
  results.json    Path to k6 JSON output file
  format          Output format: console (default), json, markdown
  --env           Environment: staging (default), production

Examples:
  node verify-slos.js results/baseline_staging_20251207.json
  node verify-slos.js results/baseline_staging_20251207.json json
  node verify-slos.js results/baseline_staging_20251207.json markdown --env production
`);
  process.exit(resultsFile ? 0 : 2);
}

if (!fs.existsSync(resultsFile)) {
  console.error(`Error: Results file not found: ${resultsFile}`);
  process.exit(2);
}

/**
 * Read and parse k6 JSON results (newline-delimited JSON or standard JSON)
 */
function parseResults(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n').filter(line => line.trim());
  
  // Try parsing as newline-delimited JSON (k6 default output)
  const results = [];
  let parseErrors = 0;
  
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      results.push(parsed);
    } catch {
      parseErrors++;
    }
  }
  
  // If too many parse errors, try as single JSON object
  if (parseErrors > lines.length / 2) {
    try {
      return [JSON.parse(content)];
    } catch (error) {
      console.error('Error parsing results file:', error.message);
      process.exit(2);
    }
  }
  
  return results;
}

/**
 * Extract metrics from k6 results
 */
function extractMetrics(results) {
  const metrics = {
    latencies: [],
    failed_requests: 0,
    total_requests: 0,
    max_vus: 0,
    ws_connecting_times: [],
    move_latencies: [],
    game_creation_times: [],
    ai_response_times: [],
    server_processing_times: [],
    contract_failures: 0,
    lifecycle_mismatches: 0,
    stalled_moves: 0,
    ws_connection_successes: 0,
    ws_connection_attempts: 0,
    concurrent_games_max: 0
  };

  for (const entry of results) {
    // Handle k6 Point metrics
    if (entry.type === 'Point') {
      const { metric, data } = entry;
      const value = data?.value;
      
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
        case 'game_creation_time':
        case 'game_creation_latency_ms':
          metrics.game_creation_times.push(value);
          break;
        case 'ai_response_time':
        case 'ai_move_latency_ms':
          metrics.ai_response_times.push(value);
          break;
        case 'server_processing_latency':
        case 'turn_processing_latency_ms':
          metrics.server_processing_times.push(value);
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
        case 'concurrent_active_games':
        case 'concurrent_games':
          if (value > metrics.concurrent_games_max) {
            metrics.concurrent_games_max = value;
          }
          break;
        case 'websocket_connection_success_rate':
        case 'ws_connection_success_rate':
          metrics.ws_connection_successes++;
          metrics.ws_connection_attempts++;
          break;
      }
    }
    
    // Handle k6 summary format (from handleSummary)
    if (entry.metrics) {
      const m = entry.metrics;
      
      if (m.http_req_duration?.values) {
        const v = m.http_req_duration.values;
        // Store percentiles directly if available
        metrics.http_req_duration_p95 = v['p(95)'];
        metrics.http_req_duration_p99 = v['p(99)'];
      }
      
      if (m.http_req_failed?.values) {
        metrics.http_req_failed_rate = m.http_req_failed.values.rate;
      }
      
      if (m.vus?.values) {
        metrics.max_vus = Math.max(metrics.max_vus, m.vus.values.max || 0);
      }
    }
  }

  return metrics;
}

/**
 * Calculate percentile value from sorted array
 */
function percentile(arr, p) {
  if (!arr || arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor(sorted.length * p);
  return sorted[Math.min(idx, sorted.length - 1)] || 0;
}

/**
 * Get SLO target with environment overrides applied
 */
function getTarget(sloKey, env) {
  const baseSlo = sloConfig.slos[sloKey];
  if (!baseSlo) return null;
  
  const override = sloConfig.environments[env]?.overrides?.[sloKey];
  return override?.target !== undefined ? override.target : baseSlo.target;
}

/**
 * Verify each SLO against extracted metrics
 */
function verifySLOs(metrics, env) {
  const results = {};
  const slos = sloConfig.slos;

  // Availability (inverse of error rate)
  const errorRate = metrics.total_requests > 0 
    ? (metrics.failed_requests / metrics.total_requests) * 100 
    : 0;
  const availability = 100 - errorRate;
  const availabilityTarget = getTarget('availability', env);

  results.availability = {
    name: slos.availability.name,
    target: availabilityTarget,
    actual: parseFloat(availability.toFixed(3)),
    unit: slos.availability.unit,
    passed: availability >= availabilityTarget,
    priority: slos.availability.priority
  };

  // API Latency p95
  const apiP95 = metrics.http_req_duration_p95 || percentile(metrics.latencies, 0.95);
  const apiP95Target = getTarget('latency_api_p95', env);
  results.latency_api_p95 = {
    name: slos.latency_api_p95.name,
    target: apiP95Target,
    actual: Math.round(apiP95),
    unit: slos.latency_api_p95.unit,
    passed: apiP95 <= apiP95Target,
    priority: slos.latency_api_p95.priority
  };

  // API Latency p99
  const apiP99 = metrics.http_req_duration_p99 || percentile(metrics.latencies, 0.99);
  const apiP99Target = getTarget('latency_api_p99', env);
  results.latency_api_p99 = {
    name: slos.latency_api_p99.name,
    target: apiP99Target,
    actual: Math.round(apiP99),
    unit: slos.latency_api_p99.unit,
    passed: apiP99 <= apiP99Target,
    priority: slos.latency_api_p99.priority
  };

  // WebSocket Connect Time p95
  const wsP95 = percentile(metrics.ws_connecting_times, 0.95);
  const wsConnectTarget = getTarget('latency_websocket_connect', env);
  results.latency_websocket_connect = {
    name: slos.latency_websocket_connect.name,
    target: wsConnectTarget,
    actual: Math.round(wsP95),
    unit: slos.latency_websocket_connect.unit,
    passed: wsP95 <= wsConnectTarget || metrics.ws_connecting_times.length === 0,
    priority: slos.latency_websocket_connect.priority,
    note: metrics.ws_connecting_times.length === 0 ? 'No WebSocket data collected' : null
  };

  // Move Latency p95
  const moveP95 = percentile(metrics.move_latencies, 0.95);
  const moveTarget = getTarget('latency_move_e2e', env);
  results.latency_move_e2e = {
    name: slos.latency_move_e2e.name,
    target: moveTarget,
    actual: Math.round(moveP95),
    unit: slos.latency_move_e2e.unit,
    passed: moveP95 <= moveTarget || metrics.move_latencies.length === 0,
    priority: slos.latency_move_e2e.priority,
    note: metrics.move_latencies.length === 0 ? 'No move data collected' : null
  };

  // Server Processing Latency p95
  const serverP95 = percentile(metrics.server_processing_times, 0.95);
  const serverTarget = getTarget('latency_move_server', env);
  results.latency_move_server = {
    name: slos.latency_move_server.name,
    target: serverTarget,
    actual: Math.round(serverP95),
    unit: slos.latency_move_server.unit,
    passed: serverP95 <= serverTarget || metrics.server_processing_times.length === 0,
    priority: slos.latency_move_server.priority,
    note: metrics.server_processing_times.length === 0 ? 'No server processing data collected' : null
  };

  // AI Response Time p95
  const aiP95 = percentile(metrics.ai_response_times, 0.95);
  const aiTarget = getTarget('latency_ai_response', env);
  results.latency_ai_response = {
    name: slos.latency_ai_response.name,
    target: aiTarget,
    actual: Math.round(aiP95),
    unit: slos.latency_ai_response.unit,
    passed: aiP95 <= aiTarget || metrics.ai_response_times.length === 0,
    priority: slos.latency_ai_response.priority,
    note: metrics.ai_response_times.length === 0 ? 'No AI data collected' : null
  };

  // Game Creation Time p95
  const gameCreateP95 = percentile(metrics.game_creation_times, 0.95);
  const gameCreateTarget = getTarget('latency_game_creation', env);
  results.latency_game_creation = {
    name: slos.latency_game_creation.name,
    target: gameCreateTarget,
    actual: Math.round(gameCreateP95),
    unit: slos.latency_game_creation.unit,
    passed: gameCreateP95 <= gameCreateTarget || metrics.game_creation_times.length === 0,
    priority: slos.latency_game_creation.priority,
    note: metrics.game_creation_times.length === 0 ? 'No game creation data collected' : null
  };

  // Error Rate
  const errorRateTarget = getTarget('error_rate', env);
  results.error_rate = {
    name: slos.error_rate.name,
    target: errorRateTarget,
    actual: parseFloat(errorRate.toFixed(3)),
    unit: slos.error_rate.unit,
    passed: errorRate <= errorRateTarget,
    priority: slos.error_rate.priority
  };

  // Concurrent Games Capacity
  const gamesTarget = getTarget('concurrent_games', env);
  results.concurrent_games = {
    name: slos.concurrent_games.name,
    target: gamesTarget,
    actual: metrics.concurrent_games_max,
    unit: slos.concurrent_games.unit,
    passed: metrics.concurrent_games_max >= gamesTarget || metrics.concurrent_games_max === 0,
    priority: slos.concurrent_games.priority,
    note: metrics.concurrent_games_max === 0 ? 'No concurrent games data collected' : null
  };

  // Concurrent Players Capacity
  const playersTarget = getTarget('concurrent_players', env);
  results.concurrent_players = {
    name: slos.concurrent_players.name,
    target: playersTarget,
    actual: metrics.max_vus,
    unit: slos.concurrent_players.unit,
    passed: metrics.max_vus >= playersTarget || metrics.max_vus === 0,
    priority: slos.concurrent_players.priority,
    note: metrics.max_vus === 0 ? 'No VU data collected' : null
  };

  // Contract Failures (must be 0)
  const contractTarget = getTarget('contract_failures', env);
  results.contract_failures = {
    name: slos.contract_failures.name,
    target: contractTarget,
    actual: metrics.contract_failures,
    unit: slos.contract_failures.unit,
    passed: metrics.contract_failures <= contractTarget,
    priority: slos.contract_failures.priority
  };

  // Lifecycle Mismatches (must be 0)
  const lifecycleTarget = getTarget('lifecycle_mismatches', env);
  results.lifecycle_mismatches = {
    name: slos.lifecycle_mismatches.name,
    target: lifecycleTarget,
    actual: metrics.lifecycle_mismatches,
    unit: slos.lifecycle_mismatches.unit,
    passed: metrics.lifecycle_mismatches <= lifecycleTarget,
    priority: slos.lifecycle_mismatches.priority
  };

  // Move Stall Rate
  const totalMoves = metrics.move_latencies.length;
  const stallRate = totalMoves > 0 
    ? (metrics.stalled_moves / totalMoves) * 100 
    : 0;
  const stallTarget = getTarget('move_stall_rate', env);
  results.move_stall_rate = {
    name: slos.move_stall_rate.name,
    target: stallTarget,
    actual: parseFloat(stallRate.toFixed(3)),
    unit: slos.move_stall_rate.unit,
    passed: stallRate <= stallTarget || totalMoves === 0,
    priority: slos.move_stall_rate.priority,
    note: totalMoves === 0 ? 'No move data collected' : null
  };

  return results;
}

/**
 * Format console output
 */
function formatConsole(results, env) {
  const passed = Object.values(results).filter(r => r.passed).length;
  const total = Object.keys(results).length;
  const criticalBreaches = Object.values(results).filter(
    r => r.priority === 'critical' && !r.passed
  );
  const highBreaches = Object.values(results).filter(
    r => r.priority === 'high' && !r.passed
  );
  
  const statusIcon = criticalBreaches.length > 0 ? '❌' : 
                     highBreaches.length > 0 ? '⚠️' : '✅';
  const statusMsg = criticalBreaches.length > 0 
    ? `CRITICAL BREACHES: ${criticalBreaches.length}` 
    : highBreaches.length > 0
    ? `HIGH PRIORITY BREACHES: ${highBreaches.length}`
    : 'All SLOs Met';

  console.log(`
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      RingRift SLO Verification Report                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Environment: ${env.padEnd(12)} Overall: ${passed}/${total} SLOs PASSED                            ║
║  ${statusIcon} ${statusMsg.padEnd(70)} ║
╠═══════════════════════════════════════════════════════════════════════════════╣`);

  // Group by priority
  const priorities = ['critical', 'high', 'medium'];
  
  for (const priority of priorities) {
    const slosInPriority = Object.entries(results).filter(([_, slo]) => slo.priority === priority);
    if (slosInPriority.length === 0) continue;
    
    console.log(`║  [${priority.toUpperCase()}]`.padEnd(80) + '║');
    
    for (const [key, slo] of slosInPriority) {
      const status = slo.passed ? '✅' : '❌';
      const value = formatValue(slo.actual, slo.unit);
      const target = formatValue(slo.target, slo.unit);
      const note = slo.note ? ` (${slo.note})` : '';
      
      const line = `║  ${status} ${slo.name.padEnd(32)} ${value.padStart(12)} / ${target.padStart(12)}${note}`;
      console.log(line.padEnd(80) + '║');
    }
    console.log('║' + ' '.repeat(79) + '║');
  }

  console.log(`╚═══════════════════════════════════════════════════════════════════════════════╝`);
  
  return passed === total && criticalBreaches.length === 0;
}

/**
 * Format value with unit
 */
function formatValue(value, unit) {
  if (unit === 'percent') return `${value}%`;
  if (unit === 'ms') return `${value}ms`;
  if (unit === 'games') return `${value} games`;
  if (unit === 'players') return `${value} players`;
  if (unit === 'count') return `${value}`;
  return `${value} ${unit}`;
}

/**
 * Format JSON output
 */
function formatJSON(results, env, sourceFile) {
  const summary = {
    timestamp: new Date().toISOString(),
    environment: env,
    source_file: sourceFile,
    slo_definitions_version: sloConfig.version,
    overall_passed: Object.values(results).every(r => r.passed),
    passed_count: Object.values(results).filter(r => r.passed).length,
    total_count: Object.keys(results).length,
    breaches_by_priority: {
      critical: Object.values(results).filter(r => r.priority === 'critical' && !r.passed).length,
      high: Object.values(results).filter(r => r.priority === 'high' && !r.passed).length,
      medium: Object.values(results).filter(r => r.priority === 'medium' && !r.passed).length
    },
    critical_breaches: Object.entries(results)
      .filter(([_, r]) => r.priority === 'critical' && !r.passed)
      .map(([key, r]) => ({ slo: key, ...r })),
    slos: results
  };
  
  console.log(JSON.stringify(summary, null, 2));
  return summary.overall_passed;
}

/**
 * Format Markdown output
 */
function formatMarkdown(results, env) {
  const passed = Object.values(results).filter(r => r.passed).length;
  const total = Object.keys(results).length;
  
  let md = `# SLO Verification Report\n\n`;
  md += `**Date:** ${new Date().toISOString()}\n`;
  md += `**Environment:** ${env}\n`;
  md += `**Overall:** ${passed}/${total} SLOs Passed\n\n`;
  
  // Summary by priority
  const priorities = ['critical', 'high', 'medium'];
  for (const priority of priorities) {
    const breaches = Object.values(results).filter(r => r.priority === priority && !r.passed);
    if (breaches.length > 0) {
      md += `⚠️ **${breaches.length} ${priority.toUpperCase()} breach(es)**\n`;
    }
  }
  md += '\n';
  
  md += `## Detailed Results\n\n`;
  md += `| SLO | Target | Actual | Status | Priority |\n`;
  md += `|-----|--------|--------|--------|----------|\n`;
  
  for (const [key, slo] of Object.entries(results)) {
    const status = slo.passed ? '✅ Pass' : '❌ Fail';
    const value = formatValue(slo.actual, slo.unit);
    const target = formatValue(slo.target, slo.unit);
    const note = slo.note ? ` ⚠️` : '';
    md += `| ${slo.name}${note} | ${target} | ${value} | ${status} | ${slo.priority} |\n`;
  }
  
  md += `\n---\n`;
  md += `*Generated by RingRift SLO Verification Framework v${sloConfig.version}*\n`;
  
  console.log(md);
  return passed === total;
}

// Main execution
const rawResults = parseResults(resultsFile);
const metrics = extractMetrics(rawResults);
const sloResults = verifySLOs(metrics, environment);

let allPassed;
switch (outputFormat) {
  case 'json':
    allPassed = formatJSON(sloResults, environment, resultsFile);
    break;
  case 'markdown':
    allPassed = formatMarkdown(sloResults, environment);
    break;
  default:
    allPassed = formatConsole(sloResults, environment);
}

// Write results to file
const outputFile = resultsFile.replace('.json', '_slo_report.json');
const reportData = {
  timestamp: new Date().toISOString(),
  environment: environment,
  source_file: resultsFile,
  slo_definitions_version: sloConfig.version,
  all_passed: allPassed,
  passed_count: Object.values(sloResults).filter(r => r.passed).length,
  total_count: Object.keys(sloResults).length,
  slos: sloResults
};

try {
  fs.writeFileSync(outputFile, JSON.stringify(reportData, null, 2));
  console.log(`\n📄 SLO report saved to: ${outputFile}`);
} catch (error) {
  console.error(`Warning: Could not write report file: ${error.message}`);
}

process.exit(allPassed ? 0 : 1);