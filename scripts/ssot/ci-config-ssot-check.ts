#!/usr/bin/env ts-node
/**
 * CI/config vs docs SSoT check (initial version)
 *
 * Verifies that the core CI jobs present in .github/workflows/ci.yml are
 * also mentioned in the supply-chain/security documentation, and that
 * the key operational configs exist.
 *
 * This is a conservative drift guard: adding/removing CI jobs or core
 * monitoring/compose files without updating docs will start failing this
 * check.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function expectFile(relativePath: string, projectRoot: string, problems: string[]): void {
  const fullPath = path.join(projectRoot, relativePath);
  if (!fs.existsSync(fullPath)) {
    problems.push(`Expected file not found: ${relativePath}`);
  }
}

export async function runCiAndConfigSsotCheck(): Promise<CheckResult> {
  try {
    const projectRoot = path.resolve(__dirname, '..', '..');

    const ciPath = path.join(projectRoot, '.github/workflows/ci.yml');
    const docPath = path.join(projectRoot, 'docs/SUPPLY_CHAIN_AND_CI_SECURITY.md');

    const problems: string[] = [];

    // Existence checks for core operational configs
    expectFile('docker-compose.yml', projectRoot, problems);
    expectFile('docker-compose.staging.yml', projectRoot, problems);
    expectFile('Dockerfile', projectRoot, problems);
    expectFile('monitoring/prometheus/alerts.yml', projectRoot, problems);
    expectFile('monitoring/prometheus/prometheus.yml', projectRoot, problems);
    expectFile('monitoring/alertmanager/alertmanager.yml', projectRoot, problems);

    // CI workflow + doc existence
    if (!fs.existsSync(ciPath)) {
      problems.push('CI workflow .github/workflows/ci.yml is missing');
    }
    if (!fs.existsSync(docPath)) {
      problems.push('docs/SUPPLY_CHAIN_AND_CI_SECURITY.md is missing');
    }

    if (problems.length === 0) {
      const ciContent = readFileSafe(ciPath);
      const docContent = readFileSafe(docPath);

      // Core CI job names as they appear under `name:` in ci.yml and as
      // human-readable labels in the supply-chain doc.
      const ciJobNames = [
        'Lint and Type Check',
        'Run Tests',
        'TS Rules Engine (rules-level)',
        'Build Application',
        'Security Scan',
        'Docker Build Test',
        'Python Rules Parity (fixture-based)',
        'Python Dependency Audit',
        'Playwright E2E Tests',
      ];

      for (const jobName of ciJobNames) {
        if (!ciContent.includes(`name: ${jobName}`)) {
          problems.push(`CI workflow is missing expected job name: "${jobName}"`);
        }
        if (!docContent.includes(jobName)) {
          problems.push(
            `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md does not mention CI job "${jobName}" (update doc or expected list)`
          );
        }
      }
    }

    if (problems.length === 0) {
      return {
        name: 'ci-config-ssot',
        passed: true,
        details: 'CI workflow jobs and core operational configs are present and documented.',
      };
    }

    return {
      name: 'ci-config-ssot',
      passed: false,
      details: problems.join('\n'),
    };
  } catch (error) {
    const err = error as Error;
    return {
      name: 'ci-config-ssot',
      passed: false,
      details: err.message ?? String(err),
    };
  }
}
