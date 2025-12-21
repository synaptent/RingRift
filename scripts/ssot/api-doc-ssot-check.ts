#!/usr/bin/env ts-node
/**
 * API docs ↔ error code SSoT check
 *
 * Ensures that the canonical error code catalog defined in
 * `src/server/errors/errorCodes.ts` is accurately reflected in
 * `docs/architecture/API_REFERENCE.md`.
 *
 * This check is intentionally conservative but not overbearing:
 * - It requires that every canonical ErrorCodes value (minus an
 *   explicit ignore list) appears at least once in the API reference.
 * - It also validates that every backtick‑wrapped ALL_CAPS_WITH_UNDERSCORES
 *   token in `API_REFERENCE.md` either:
 *     - Is a canonical ErrorCodes value, or
 *     - Is a known legacy code that normalizes to a canonical value via
 *       LegacyCodeMapping.
 * - It does **not** fail on extra prose that merely mentions codes without
 *   backticks.
 *
 * The goal is to keep the human‑facing error catalog in
 * `docs/architecture/API_REFERENCE.md` aligned with the executable SSoT while still
 * allowing some internal‑only codes to remain undocumented when appropriate.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';
import { ErrorCodes, LegacyCodeMapping, type ErrorCode } from '../../src/server/errors/errorCodes';

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function getCanonicalErrorCodes(): ErrorCode[] {
  // Object.values preserves the actual code strings, which are the
  // canonical surface we want docs to cover.
  return Object.values(ErrorCodes);
}

export async function runApiDocSsotCheck(): Promise<CheckResult> {
  const projectRoot = path.resolve(__dirname, '..', '..');
  // Canonical API reference lives under docs/architecture/.
  const apiDocPath = path.join(projectRoot, 'docs/architecture/API_REFERENCE.md');

  if (!fs.existsSync(apiDocPath)) {
    return {
      name: 'api-doc-ssot',
      passed: false,
      details:
        'docs/architecture/API_REFERENCE.md is missing (cannot validate API docs against error code catalog).',
    };
  }

  const docContent = readFileSafe(apiDocPath);

  // Error codes that are intentionally *not* required to appear in the
  // public API reference. These are typically internal, transport‑only,
  // or legacy codes that should not be advertised as stable public
  // contracts.
  const ignoreCanonicalCodes = new Set<ErrorCode>([
    // Example internal‑ish codes that are primarily wiring/transport
    // concerns rather than user‑facing API surface. We can revisit this
    // list as the documentation matures.
    // ErrorCodes.RESOURCE_ROUTE_NOT_FOUND,
  ]);

  const canonicalCodes = getCanonicalErrorCodes().filter((code) => !ignoreCanonicalCodes.has(code));

  const missingInDoc: ErrorCode[] = [];

  for (const code of canonicalCodes) {
    // Prefer backtick‑wrapped appearance (e.g. `AUTH_INVALID_CREDENTIALS`)
    // but fall back to a plain substring check as a safety net.
    const codePattern = `\`${code}\``;
    if (!docContent.includes(codePattern) && !docContent.includes(code)) {
      missingInDoc.push(code);
    }
  }

  // Collect all backtick‑wrapped ALL_CAPS_WITH_UNDERSCORES tokens from the doc
  const documentedCodeTokens = new Set<string>();
  const backtickCodeRegex = /`([A-Z0-9_]{3,})`/g;
  let match: RegExpExecArray | null;
  while ((match = backtickCodeRegex.exec(docContent)) !== null) {
    documentedCodeTokens.add(match[1]);
  }

  const canonicalSet = new Set<string>(canonicalCodes);
  const legacyKeys = new Set<string>(Object.keys(LegacyCodeMapping));
  const legacyValues = new Set<string>(Object.values(LegacyCodeMapping));

  // WebSocket error codes are governed by WebSocketErrorCode in
  // src/shared/types/websocket.ts rather than ErrorCodes, but
  // docs/architecture/API_REFERENCE.md legitimately documents them in the
  // WebSocket section. We treat them as additional allowed tokens
  // here so they are not flagged as unknown.
  const additionalAllowedTokens = new Set<string>([
    'INVALID_PAYLOAD',
    'RATE_LIMITED',
    'MOVE_REJECTED',
    'CHOICE_REJECTED',
    'DECISION_PHASE_TIMEOUT',
  ]);

  const unknownDocCodes: string[] = [];

  for (const token of documentedCodeTokens) {
    const isCanonical = canonicalSet.has(token);
    const isLegacyKey = legacyKeys.has(token);
    const isLegacyValue = legacyValues.has(token);
    const isAdditional = additionalAllowedTokens.has(token);

    if (!isCanonical && !isLegacyKey && !isLegacyValue && !isAdditional) {
      unknownDocCodes.push(token);
    }
  }

  // Build result
  if (missingInDoc.length === 0 && unknownDocCodes.length === 0) {
    return {
      name: 'api-doc-ssot',
      passed: true,
      details:
        'All canonical ErrorCodes are mentioned in docs/architecture/API_REFERENCE.md, and all documented backtick error codes map to known canonical or legacy codes.',
    };
  }

  const problems: string[] = [];

  if (missingInDoc.length > 0) {
    problems.push(
      'The following canonical ErrorCodes are not mentioned in docs/architecture/API_REFERENCE.md:'
    );
    for (const code of missingInDoc) {
      problems.push(`- ${code}`);
    }
    problems.push(
      '\nFor each of these codes, either (a) add a row to the HTTP error catalog in docs/architecture/API_REFERENCE.md, or (b) add the code to the ignore list in scripts/ssot/api-doc-ssot-check.ts if it is intentionally internal‑only.'
    );
  }

  if (unknownDocCodes.length > 0) {
    problems.push(
      'The following backtick‑wrapped error codes appear in docs/architecture/API_REFERENCE.md but are not part of ErrorCodes or LegacyCodeMapping:'
    );
    for (const token of unknownDocCodes) {
      problems.push(`- ${token}`);
    }
    problems.push(
      '\nThese are likely typos or outdated names. Either update them to canonical ErrorCodes values, map them via LegacyCodeMapping, or remove them from the public docs.'
    );
  }

  return {
    name: 'api-doc-ssot',
    passed: false,
    details: problems.join('\n'),
  };
}
