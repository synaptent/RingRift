#!/usr/bin/env ts-node
/**
 * RingRift SSoT Enforcement Entrypoint
 *
 * This script aggregates all Single Source of Truth (SSoT) drift checks
 * into a single CLI. It is intended to be wired into CI (and exposed
 * via `npm run ssot-check`) so that rules/docs/hosts/config cannot drift
 * silently from their canonical sources.
 *
 * Individual checks live in neighbouring modules and focus on narrow
 * domains (rules semantics, lifecycle/API, Python parity/contracts,
 * CI/config/docs, and documentation banners).
 */

import { runRulesSsotCheck } from './rules-ssot-check';
import { runLifecycleSsotCheck } from './lifecycle-ssot-check';
import { runPythonParitySsotCheck } from './python-parity-ssot-check';
import { runCiAndConfigSsotCheck } from './ci-config-ssot-check';
import { runDocsBannerSsotCheck } from './docs-banner-ssot-check';
import { runEnvDocSsotCheck } from './env-doc-ssot-check';
import { runSecretsDocSsotCheck } from './secrets-doc-ssot-check';
import { runApiDocSsotCheck } from './api-doc-ssot-check';
import { runDocsLinkSsotCheck } from './docs-link-ssot-check';
import { runApiEndpointsSsotCheck } from './api-endpoints-ssot-check';
import { runParityProtectionSsotCheck } from './parity-protection-ssot-check';
import { runPhaseMoveContractSsotCheck } from './phase-move-contract-ssot-check';

interface CheckResult {
  name: string;
  passed: boolean;
  details?: string;
}

async function main() {
  const checks: Array<() => Promise<CheckResult>> = [
    runRulesSsotCheck,
    runLifecycleSsotCheck,
    runPhaseMoveContractSsotCheck,
    runPythonParitySsotCheck,
    runCiAndConfigSsotCheck,
    runDocsBannerSsotCheck,
    runEnvDocSsotCheck,
    runSecretsDocSsotCheck,
    runApiDocSsotCheck,
    runApiEndpointsSsotCheck,
    runDocsLinkSsotCheck,
    runParityProtectionSsotCheck,
  ];

  const results: CheckResult[] = [];

  for (const run of checks) {
    try {
      // Each runner is responsible for catching and surfacing its own
      // domain-specific errors and returning a structured result.
      // We still guard here to avoid a single throw aborting the entire
      // suite without a clear report.

      const result = await run();
      results.push(result);
    } catch (err) {
      const error = err as Error;
      results.push({
        name: run.name || 'unknown-check',
        passed: false,
        details: error.message ?? String(error),
      });
    }
  }

  const failed = results.filter((r) => !r.passed);

  // Simple textual summary for now; if needed, this can be upgraded to a
  // markdown or JSON report later.
  /* eslint-disable no-console */
  console.log('\nRingRift SSoT Check Summary');
  console.log('='.repeat(32));
  for (const r of results) {
    const status = r.passed ? 'PASS' : 'FAIL';
    console.log(`- [${status}] ${r.name}`);
    if (!r.passed && r.details) {
      console.log(`    → ${r.details.split('\n').join('\n      ')}`);
    }
  }
  console.log('');
  /* eslint-enable no-console */

  if (failed.length > 0) {
    process.exitCode = 1;
  }
}

// Execute immediately when invoked via `ts-node` / `node`.

main().catch((err) => {
  console.error('Unexpected error running SSoT checks:', err);
  process.exitCode = 1;
});

export type { CheckResult };
