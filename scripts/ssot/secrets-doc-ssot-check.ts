#!/usr/bin/env ts-node
/**
 * Secrets documentation SSoT check
 *
 * Ensures that the canonical secret definitions in
 * `src/server/utils/secretsValidation.ts` are reflected in
 * `docs/operations/SECRETS_MANAGEMENT.md`.
 *
 * This is intentionally conservative:
 * - Every SECRET_DEFINITIONS entry (minus an explicit ignore list)
 *   must be mentioned in the secrets doc.
 * - The doc is allowed to contain additional narrative content or
 *   non-canonical examples.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';
import { SECRET_DEFINITIONS } from '../../src/server/utils/secretsValidation';

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

export async function runSecretsDocSsotCheck(): Promise<CheckResult> {
  const projectRoot = path.resolve(__dirname, '..', '..');
  // Canonical secrets reference lives under docs/operations/.
  const secretsDocPath = path.join(projectRoot, 'docs/operations/SECRETS_MANAGEMENT.md');

  if (!fs.existsSync(secretsDocPath)) {
    return {
      name: 'secrets-doc-ssot',
      passed: false,
      details:
        'docs/operations/SECRETS_MANAGEMENT.md is missing (cannot validate secrets docs against canonical definitions).',
    };
  }

  const docContent = readFileSafe(secretsDocPath);
  const problems: string[] = [];

  // Secrets that are intentionally *not* documented in the public
  // secrets guide can be added here (for example, internal-only or
  // transient diagnostics secrets).
  const ignoreSecretNames = new Set<string>([]);

  const missingInDoc: string[] = [];

  for (const def of SECRET_DEFINITIONS) {
    const name = def.name;
    if (ignoreSecretNames.has(name)) continue;

    // Prefer matches where the secret name appears as inline code,
    // but fall back to a plain substring search to be tolerant of
    // formatting changes.
    const codePattern = `\`${name}\``;
    if (!docContent.includes(codePattern) && !docContent.includes(name)) {
      missingInDoc.push(name);
    }
  }

  if (missingInDoc.length === 0) {
    return {
      name: 'secrets-doc-ssot',
      passed: true,
      details:
        'All canonical SECRET_DEFINITIONS are mentioned in docs/operations/SECRETS_MANAGEMENT.md (minus any explicit ignores).',
    };
  }

  problems.push(
    'The following secrets from src/server/utils/secretsValidation.ts (SECRET_DEFINITIONS) are not mentioned in docs/operations/SECRETS_MANAGEMENT.md:'
  );
  for (const name of missingInDoc) {
    problems.push(`- ${name}`);
  }

  problems.push(
    '\nIf some of these are intentionally internal-only, add them to the ignore list in scripts/ssot/secrets-doc-ssot-check.ts. Otherwise, document them in docs/operations/SECRETS_MANAGEMENT.md as part of the canonical secrets reference.'
  );

  return {
    name: 'secrets-doc-ssot',
    passed: false,
    details: problems.join('\n'),
  };
}
