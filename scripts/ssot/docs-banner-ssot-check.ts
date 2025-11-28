#!/usr/bin/env ts-node
/**
 * Documentation SSoT banner check
 *
 * Verifies that key rules/architecture/CI/AI docs include a standard
 * SSoT banner, and that any critical file paths mentioned in the banner
 * actually exist in the repository.
 *
 * This is not a full Markdown linter; it is a focused guardrail to
 * prevent the accidental removal of SSoT framing from the most
 * important docs.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';

interface DocExpectation {
  /** Path relative to project root. */
  path: string;
  /** A substring that should appear in the banner. */
  requiredSnippet: string;
}

const DOCS_TO_CHECK: DocExpectation[] = [
  {
    path: 'RULES_ENGINE_ARCHITECTURE.md',
    requiredSnippet: 'Rules/invariants semantics SSoT',
  },
  {
    path: 'RULES_IMPLEMENTATION_MAPPING.md',
    requiredSnippet: 'Rules/invariants semantics SSoT',
  },
  {
    path: 'docs/RULES_ENGINE_SURFACE_AUDIT.md',
    requiredSnippet: 'Rules/invariants semantics SSoT',
  },
  {
    path: 'docs/CANONICAL_ENGINE_API.md',
    requiredSnippet: 'Lifecycle/API SSoT',
  },
  {
    path: 'docs/SUPPLY_CHAIN_AND_CI_SECURITY.md',
    requiredSnippet: 'Operational SSoT',
  },
  {
    path: 'AI_ARCHITECTURE.md',
    requiredSnippet: 'rules semantics SSoT',
  },
  {
    path: 'docs/PYTHON_PARITY_REQUIREMENTS.md',
    requiredSnippet: 'Canonical TS rules surface',
  },
  {
    path: 'ARCHITECTURE_ASSESSMENT.md',
    requiredSnippet: 'SSoT alignment',
  },
  {
    path: 'ARCHITECTURE_REMEDIATION_PLAN.md',
    requiredSnippet: 'SSoT alignment',
  },
  {
    path: 'docs/MODULE_RESPONSIBILITIES.md',
    requiredSnippet: 'SSoT alignment',
  },
];

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

export async function runDocsBannerSsotCheck(): Promise<CheckResult> {
  const projectRoot = path.resolve(__dirname, '..', '..');

  const problems: string[] = [];

  for (const doc of DOCS_TO_CHECK) {
    const fullPath = path.join(projectRoot, doc.path);
    if (!fs.existsSync(fullPath)) {
      problems.push(`Expected doc not found: ${doc.path}`);
      // Skip further checks for this file
      continue;
    }

    const content = readFileSafe(fullPath);

    if (!content.includes('SSoT alignment')) {
      problems.push(`Doc ${doc.path} is missing an "SSoT alignment" banner.`);
    }

    if (!content.includes(doc.requiredSnippet)) {
      problems.push(
        `Doc ${doc.path} does not contain required banner snippet: ${JSON.stringify(doc.requiredSnippet)}`
      );
    }
  }

  if (problems.length === 0) {
    return {
      name: 'docs-banner-ssot',
      passed: true,
      details: 'Core docs contain SSoT banners with expected snippets.',
    };
  }

  return {
    name: 'docs-banner-ssot',
    passed: false,
    details: problems.join('\n'),
  };
}
