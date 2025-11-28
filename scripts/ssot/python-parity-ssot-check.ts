#!/usr/bin/env ts-node
/**
 * Python parity & contracts SSoT check (stub)
 *
 * Initial implementation is intentionally lightweight: it only checks that
 * the v2 contract vector files exist in both the TS and Python runners,
 * and that the documented parity requirements file is present.
 *
 * This still provides value as a drift guard: removing/renaming vectors or
 * the Python runner without updating docs will fail this check.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';

function expectFile(relativePath: string, projectRoot: string, problems: string[]): void {
  const fullPath = path.join(projectRoot, relativePath);
  if (!fs.existsSync(fullPath)) {
    problems.push(`Expected file not found: ${relativePath}`);
  }
}

export async function runPythonParitySsotCheck(): Promise<CheckResult> {
  try {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const problems: string[] = [];

    // Contract vectors (v2) – TS side
    expectFile('tests/fixtures/contract-vectors/v2/placement.vectors.json', projectRoot, problems);
    expectFile('tests/fixtures/contract-vectors/v2/movement.vectors.json', projectRoot, problems);
    expectFile('tests/fixtures/contract-vectors/v2/capture.vectors.json', projectRoot, problems);
    expectFile(
      'tests/fixtures/contract-vectors/v2/line_detection.vectors.json',
      projectRoot,
      problems
    );
    expectFile('tests/fixtures/contract-vectors/v2/territory.vectors.json', projectRoot, problems);

    // TS runner
    expectFile('tests/contracts/contractVectorRunner.test.ts', projectRoot, problems);

    // Python runner
    expectFile('ai-service/tests/contracts/test_contract_vectors.py', projectRoot, problems);

    // Parity requirements doc
    expectFile('docs/PYTHON_PARITY_REQUIREMENTS.md', projectRoot, problems);

    if (problems.length === 0) {
      return {
        name: 'python-parity-ssot',
        passed: true,
        details: 'Contract vectors, TS/Python runners, and parity requirements doc are present.',
      };
    }

    return {
      name: 'python-parity-ssot',
      passed: false,
      details: problems.join('\n'),
    };
  } catch (error) {
    const err = error as Error;
    return {
      name: 'python-parity-ssot',
      passed: false,
      details: err.message ?? String(err),
    };
  }
}
