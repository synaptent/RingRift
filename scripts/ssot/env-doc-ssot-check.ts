#!/usr/bin/env ts-node
/**
 * Environment variables documentation SSoT check
 *
 * Ensures that the canonical environment variable schema defined in
 * `src/server/config/env.ts` (EnvSchema) is reflected in
 * `docs/operations/ENVIRONMENT_VARIABLES.md`.
 *
 * This check is deliberately conservative:
 * - It requires that every EnvSchema key (minus an explicit ignore list)
 *   appears in the env-var doc.
 * - It does **not** currently fail if the doc mentions additional
 *   environment variables (e.g. test-only flags, client-only flags, or
 *   ai-service specific envs). Those may be validated by future,
 *   specialised checks.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';
import { EnvSchema } from '../../src/server/config/env';

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function getEnvSchemaKeys(): string[] {
  // Zod's ZodObject exposes a `shape` helper in TS which becomes a
  // runtime function returning the field map. We support both the
  // function form and a potential direct object for robustness.
  const anySchema: any = EnvSchema as any;
  let shape: Record<string, unknown> | undefined;

  if (typeof anySchema.shape === 'function') {
    shape = anySchema.shape();
  } else if (anySchema._def && typeof anySchema._def.shape === 'function') {
    shape = anySchema._def.shape();
  } else if (anySchema.shape && typeof anySchema.shape === 'object') {
    shape = anySchema.shape as Record<string, unknown>;
  }

  if (!shape) {
    throw new Error('Unable to introspect EnvSchema.shape – Zod API may have changed');
  }

  return Object.keys(shape);
}

export async function runEnvDocSsotCheck(): Promise<CheckResult> {
  const projectRoot = path.resolve(__dirname, '..', '..');
  // Canonical env-var reference lives under docs/operations/.
  const envDocPath = path.join(projectRoot, 'docs/operations/ENVIRONMENT_VARIABLES.md');

  const problems: string[] = [];

  if (!fs.existsSync(envDocPath)) {
    return {
      name: 'env-doc-ssot',
      passed: false,
      details:
        'docs/operations/ENVIRONMENT_VARIABLES.md is missing (cannot validate env docs against schema).',
    };
  }

  const docContent = readFileSafe(envDocPath);

  // Keys that are intentionally *not* documented as top-level env vars
  // in ENVIRONMENT_VARIABLES.md (internal / injected / implementation details).
  const ignoreSchemaKeys = new Set<string>([
    // npm injects this; not a user-configured env var.
    'npm_package_version',
  ]);

  const envKeys = getEnvSchemaKeys().filter((key) => !ignoreSchemaKeys.has(key));

  const missingInDoc: string[] = [];

  for (const key of envKeys) {
    // We primarily expect variables to appear as inline code, e.g. `NODE_ENV`.
    // Fall back to a plain substring check as a safety net.
    const codePattern = `\`${key}\``;
    if (!docContent.includes(codePattern) && !docContent.includes(key)) {
      missingInDoc.push(key);
    }
  }

  if (missingInDoc.length === 0) {
    return {
      name: 'env-doc-ssot',
      passed: true,
      details:
        'All server EnvSchema keys (minus internal ignores) are mentioned in docs/operations/ENVIRONMENT_VARIABLES.md.',
    };
  }

  problems.push(
    'The following EnvSchema keys are not mentioned in docs/operations/ENVIRONMENT_VARIABLES.md:'
  );
  for (const key of missingInDoc) {
    problems.push(`- ${key}`);
  }

  problems.push(
    '\nIf some of these are intentionally internal-only, add them to the ignore list in scripts/ssot/env-doc-ssot-check.ts. Otherwise, document them in docs/operations/ENVIRONMENT_VARIABLES.md as part of the canonical environment reference.'
  );

  return {
    name: 'env-doc-ssot',
    passed: false,
    details: problems.join('\n'),
  };
}
