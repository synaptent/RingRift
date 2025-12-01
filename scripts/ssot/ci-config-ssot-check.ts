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
    const nightlyCiPath = path.join(
      projectRoot,
      '.github/workflows/orchestrator-soak-nightly.yml'
    );
    const docPath = path.join(projectRoot, 'docs/SUPPLY_CHAIN_AND_CI_SECURITY.md');
    const orchestratorPlanPath = path.join(projectRoot, 'docs/ORCHESTRATOR_ROLLOUT_PLAN.md');
    const orchestratorRunbookPath = path.join(
      projectRoot,
      'docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md'
    );

    const problems: string[] = [];

    // Existence checks for core operational configs
    expectFile('docker-compose.yml', projectRoot, problems);
    expectFile('docker-compose.staging.yml', projectRoot, problems);
    expectFile('Dockerfile', projectRoot, problems);
    expectFile('monitoring/prometheus/alerts.yml', projectRoot, problems);
    expectFile('monitoring/prometheus/prometheus.yml', projectRoot, problems);
    expectFile('monitoring/alertmanager/alertmanager.yml', projectRoot, problems);
    expectFile('.github/workflows/orchestrator-soak-nightly.yml', projectRoot, problems);
    expectFile('.github/workflows/ai-healthcheck-nightly.yml', projectRoot, problems);

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
      const nightlyCiContent = readFileSafe(nightlyCiPath);
      const orchestratorPlanContent = readFileSafe(orchestratorPlanPath);
      const orchestratorRunbookContent = readFileSafe(orchestratorRunbookPath);

      // Core CI job names as they appear under `name:` in ci.yml and as
      // human-readable labels in the supply-chain doc.
      const ciJobNames = [
        'Lint and Type Check',
        'Run Tests',
        'TS Rules Engine (rules-level)',
        'TS Orchestrator Parity (adapter-ON)',
        'TS Parity (trace-level & host parity)',
        'TS Integration (routes, WebSocket, AI integration)',
        'Orchestrator Invariant Soak (smoke)',
        'SSoT Drift Guards',
        'Orchestrator Invariant Soak (short)',
        'Orchestrator Parity (TS orchestrator + Python contracts)',
        'Build Application',
        'Security Scan',
        'Docker Build Test',
        'AI Service Docker Build & Torch Import Smoke',
        'Python Rules Parity (fixture-based)',
        'Python Core Tests (non-parity)',
        'Python AI Self-Play Healthcheck',
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

      // Nightly orchestrator soak lives in a separate workflow file but is
      // still part of the CI/SLO story. Ensure its job name and docs entry
      // stay in sync.
      const nightlyJobName = 'Orchestrator Invariant Soak (nightly)';
      if (!nightlyCiContent.includes(`name: ${nightlyJobName}`)) {
        problems.push(
          `Nightly workflow .github/workflows/orchestrator-soak-nightly.yml is missing expected job name: "${nightlyJobName}"`
        );
      }
      if (!docContent.includes(nightlyJobName)) {
        problems.push(
          `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md does not mention nightly CI job "${nightlyJobName}" (update doc or expected list)`
        );
      }

      // Nightly AI self-play healthcheck lives in its own workflow file and
      // should also be documented alongside the orchestrator nightly soak.
      const aiHealthcheckCiPath = path.join(
        projectRoot,
        '.github/workflows/ai-healthcheck-nightly.yml'
      );
      const aiHealthcheckCiContent = readFileSafe(aiHealthcheckCiPath);
      const aiNightlyJobName = 'AI Self-Play Healthcheck (Nightly)';
      if (!aiHealthcheckCiContent.includes(`name: ${aiNightlyJobName}`)) {
        problems.push(
          `Nightly workflow .github/workflows/ai-healthcheck-nightly.yml is missing expected job name: "${aiNightlyJobName}"`
        );
      }
      if (!docContent.includes(aiNightlyJobName)) {
        problems.push(
          `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md does not mention nightly CI job "${aiNightlyJobName}" (update doc or expected list)`
        );
      }

      // Orchestrator-specific gate commands should be documented both in the
      // supply-chain doc and in orchestrator rollout docs (plan and/or runbook).
      const orchestratorCommands = [
        'npm run test:orchestrator-parity:ts',
        'npm run soak:orchestrator:smoke',
        'npm run soak:orchestrator:short',
        'npm run soak:orchestrator:nightly',
      ];

      for (const cmd of orchestratorCommands) {
        if (!docContent.includes(cmd)) {
          problems.push(
            `docs/SUPPLY_CHAIN_AND_CI_SECURITY.md does not mention orchestrator gate command "${cmd}"`
          );
        }

        const mentionedInPlan = orchestratorPlanContent.includes(cmd);
        const mentionedInRunbook = orchestratorRunbookContent.includes(cmd);
        if (!mentionedInPlan && !mentionedInRunbook) {
          problems.push(
            `Neither docs/ORCHESTRATOR_ROLLOUT_PLAN.md nor docs/runbooks/ORCHESTRATOR_ROLLOUT_RUNBOOK.md mention orchestrator gate command "${cmd}"`
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
