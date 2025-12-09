#!/usr/bin/env node

/**
 * RingRift BCAP v1 Two-Run Consistency Checker
 *
 * Compares two SLO report JSON files (outputs from verify-slos.js) for the
 * BCAP_SQ8_3P_TARGET_100G_300P scenario and checks:
 *
 *   1. Both runs have overall_passed === true.
 *   2. API p95 latency (latency_api_p95.actual) is within ±10% between runs.
 *   3. AI p95 latency (latency_ai_response.actual) is within ±10% between runs.
 *
 * Usage:
 *   node compare-runs.js <run1_slo_report.json> <run2_slo_report.json> [--scenario-id ID]
 *
 * Example:
 *   node tests/load/scripts/compare-runs.js \
 *     tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_staging_run1_slo_report.json \
 *     tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_staging_run2_slo_report.json
 *
 * Exit codes:
 *   0 - Consistency check passed
 *   1 - Consistency check failed (or metrics missing)
 *   2 - Usage / file errors
 */

const fs = require('fs');
const path = require('path');

const DEFAULT_SCENARIO_ID = 'BCAP_SQ8_3P_TARGET_100G_300P';
const MAX_REL_DELTA = 0.10; // 10%

function printUsage() {
  console.log(`
RingRift BCAP v1 Two-Run Consistency Checker

Usage:
  node compare-runs.js <run1_slo_report.json> <run2_slo_report.json> [--scenario-id ID]

Arguments:
  run1_slo_report.json   First SLO report (from verify-slos.js)
  run2_slo_report.json   Second SLO report (from verify-slos.js)
  --scenario-id          Optional scenario ID (default: ${DEFAULT_SCENARIO_ID})

Example:
  node tests/load/scripts/compare-runs.js \\
    tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_staging_run1_slo_report.json \\
    tests/load/results/BCAP_SQ8_3P_TARGET_100G_300P_staging_run2_slo_report.json
`);
}

function parseCli(argv) {
  const args = argv.slice(2);
  const positional = [];
  let scenarioId = DEFAULT_SCENARIO_ID;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];

    if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    }

    if (arg === '--scenario-id' || arg === '--scenario') {
      const val = args[i + 1];
      i += 1;
      if (!val) {
        console.error('Missing value for --scenario-id');
        process.exit(2);
      }
      scenarioId = val;
      continue;
    }

    if (arg.startsWith('--scenario-id=')) {
      scenarioId = arg.split('=', 2)[1] || scenarioId;
      continue;
    }

    if (arg.startsWith('--')) {
      console.error(`Unknown option: ${arg}`);
      printUsage();
      process.exit(2);
    }

    positional.push(arg);
  }

  if (positional.length !== 2) {
    printUsage();
    process.exit(2);
  }

  return {
    scenarioId,
    run1Path: positional[0],
    run2Path: positional[1],
  };
}

function loadReport(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`Error: File not found: ${filePath}`);
    process.exit(2);
  }

  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    console.error(`Error parsing JSON from ${filePath}:`, err.message);
    process.exit(2);
  }
}

function relativeDelta(v1, v2) {
  if (typeof v1 !== 'number' || typeof v2 !== 'number' || !isFinite(v1) || v1 === 0) {
    return null;
  }
  return Math.abs(v2 - v1) / Math.abs(v1);
}

(function main() {
  const cli = parseCli(process.argv);

  const report1 = loadReport(cli.run1Path);
  const report2 = loadReport(cli.run2Path);

  const env1 = report1.environment || 'unknown';
  const env2 = report2.environment || 'unknown';
  const scenarioId1 = report1.scenario_id || report1.scenarioId || null;
  const scenarioId2 = report2.scenario_id || report2.scenarioId || null;

  const slo1 = report1.slos || {};
  const slo2 = report2.slos || {};

  const api1 = slo1.latency_api_p95 && slo1.latency_api_p95.actual;
  const api2 = slo2.latency_api_p95 && slo2.latency_api_p95.actual;
  const ai1 = slo1.latency_ai_response && slo1.latency_ai_response.actual;
  const ai2 = slo2.latency_ai_response && slo2.latency_ai_response.actual;

  const apiDelta = relativeDelta(api1, api2);
  const aiDelta = relativeDelta(ai1, ai2);

  const overall1 = !!(report1.all_passed ?? report1.overall_passed);
  const overall2 = !!(report2.all_passed ?? report2.overall_passed);

  const checks = {
    both_runs_passed: overall1 && overall2,
    api_latency_present: typeof api1 === 'number' && typeof api2 === 'number',
    ai_latency_present: typeof ai1 === 'number' && typeof ai2 === 'number',
    api_latency_within_10_percent: apiDelta !== null && apiDelta <= MAX_REL_DELTA,
    ai_latency_within_10_percent: aiDelta !== null && aiDelta <= MAX_REL_DELTA,
  };

  const passed = Object.values(checks).every(Boolean);

  const summary = {
    scenario_expected: cli.scenarioId,
    scenario_run1: scenarioId1,
    scenario_run2: scenarioId2,
    environment_run1: env1,
    environment_run2: env2,
    run1_file: path.resolve(cli.run1Path),
    run2_file: path.resolve(cli.run2Path),
    run1_overall_passed: overall1,
    run2_overall_passed: overall2,
    latencies: {
      api_p95: {
        run1_ms: api1,
        run2_ms: api2,
        rel_delta: apiDelta !== null ? Number(apiDelta.toFixed(4)) : null,
        max_rel_delta_allowed: MAX_REL_DELTA,
        within_tolerance: checks.api_latency_within_10_percent,
      },
      ai_response_p95: {
        run1_ms: ai1,
        run2_ms: ai2,
        rel_delta: aiDelta !== null ? Number(aiDelta.toFixed(4)) : null,
        max_rel_delta_allowed: MAX_REL_DELTA,
        within_tolerance: checks.ai_latency_within_10_percent,
      },
    },
    checks,
    passed,
    verdict: passed
      ? 'CONSISTENT: both runs passed and API/AI p95 latencies are within ±10%'
      : 'INCONSISTENT: see checks for details',
  };

  // Human-readable summary
  console.log('╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║       BCAP v1 Two-Run Consistency Check (Target Scale Scenario)     ║');
  console.log('╠══════════════════════════════════════════════════════════════════════╣');
  console.log(
    `║  Scenario (expected): ${cli.scenarioId.padEnd(45)}║`
  );
  console.log(
    `║  Run 1: ${path.basename(cli.run1Path).padEnd(59)}║`
  );
  console.log(
    `║         env=${String(env1).padEnd(10)} scenario_id=${String(scenarioId1 || 'n/a').padEnd(24)}║`
  );
  console.log(
    `║  Run 2: ${path.basename(cli.run2Path).padEnd(59)}║`
  );
  console.log(
    `║         env=${String(env2).padEnd(10)} scenario_id=${String(scenarioId2 || 'n/a').padEnd(24)}║`
  );
  console.log('╠══════════════════════════════════════════════════════════════════════╣');
  console.log(
    `║  Both runs overall passed: ${String(checks.both_runs_passed).padEnd(38)}║`
  );
  console.log(
    `║  API p95 within ±10%:     ${String(checks.api_latency_within_10_percent).padEnd(38)}║`
  );
  console.log(
    `║  AI p95 within ±10%:      ${String(checks.ai_latency_within_10_percent).padEnd(38)}║`
  );
  console.log('╠══════════════════════════════════════════════════════════════════════╣');
  console.log(
    `║  Verdict: ${summary.verdict.padEnd(60)}║`
  );
  console.log('╚══════════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(JSON.stringify(summary, null, 2));

  process.exit(passed ? 0 : 1);
})();