#!/usr/bin/env ts-node
/**
 * API endpoints ↔ OpenAPI SSoT check
 *
 * Ensures that the HTTP endpoints documented in docs/API_REFERENCE.md stay
 * aligned with the canonical OpenAPI specification generated from
 * src/server/openapi/config.ts.
 *
 * Behaviour (v1):
 * - Treat the OpenAPI spec (swaggerSpec.paths) as the canonical source of
 *   truth for REST endpoints.
 * - Parse docs/API_REFERENCE.md for endpoint tables and extract
 *   (method, path) pairs.
 * - Enforce **docs → OpenAPI** coverage: every documented endpoint must
 *   exist in the OpenAPI spec with a matching HTTP method.
 * - Enforce **OpenAPI → docs** coverage for public REST groups with
 *   stable surface (currently /auth, /users, /games): every such
 *   (method, path) pair in the spec must appear in the docs, unless it
 *   is explicitly ignored.
 *
 * Normalisation rules:
 * - Paths are compared after:
 *   - Ensuring a leading '/'.
 *   - Stripping an optional '/api' prefix (docs and routes use
 *     '/api' as a server prefix, while OpenAPI paths are unprefixed).
 *   - Converting Express-style segments ':id' to OpenAPI-style
 *     '{id}' so that `/games/:gameId` and `/games/{gameId}` match.
 * - HTTP methods are compared in lowercase.
 *
 * This check is intentionally conservative and explainable: failures
 * report the exact (method, path) pairs that are mismatched so that
 * authors can either:
 *   - Update docs/API_REFERENCE.md to match the spec, or
 *   - Update OpenAPI JSDoc to expose the intended endpoint, or
 *   - Add a temporary ignore entry in this script (documented
 *     technical debt) if an endpoint is deliberately undocumented.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';
// swaggerSpec is generated at module load time from route JSDoc.
// eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
const { swaggerSpec } = require('../../src/server/openapi/config');

type HttpMethod = 'get' | 'post' | 'put' | 'delete' | 'patch' | 'options' | 'head';

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function normalizePath(raw: string): string {
  let p = raw.trim();

  if (!p.startsWith('/')) {
    p = `/${p}`;
  }

  // Strip optional '/api' prefix so that '/api/auth/login' and
  // '/auth/login' are treated as the same canonical path.
  p = p.replace(/^\/api(?=\/)/, '');

  // Convert Express-style ":id" parameters to OpenAPI-style "{id}".
  p = p.replace(/:([A-Za-z0-9_]+)/g, '{$1}');

  return p;
}

function makeKey(method: HttpMethod, p: string): string {
  return `${method.toLowerCase()} ${normalizePath(p)}`;
}

function extractOpenApiEndpoints(): { all: Set<string>; publicAuthUsersGames: Set<string> } {
  const spec = swaggerSpec as { paths?: Record<string, Record<string, unknown>> };
  const pathsObj = spec.paths ?? {};

  const all = new Set<string>();
  const publicPrefixes = ['/auth', '/users', '/games'];
  const publicAuthUsersGames = new Set<string>();

  for (const [rawPath, pathItem] of Object.entries(pathsObj)) {
    const normPath = normalizePath(rawPath);
    const isPublicPrefix = publicPrefixes.some((prefix) => normPath.startsWith(prefix));

    for (const [method, op] of Object.entries(pathItem)) {
      // Only consider standard HTTP methods.
      const m = method.toLowerCase() as HttpMethod;
      if (!['get', 'post', 'put', 'delete', 'patch'].includes(m)) continue;
      if (!op) continue;

      const key = makeKey(m, normPath);
      all.add(key);
      if (isPublicPrefix) {
        publicAuthUsersGames.add(key);
      }
    }
  }

  return { all, publicAuthUsersGames };
}

function extractDocumentedEndpoints(docContent: string): Set<string> {
  const documented = new Set<string>();

  // Match table rows like:
  // | GET    | `/auth/login` | ... |
  const rowRegex = /\|\s*(GET|POST|PUT|DELETE|PATCH)\s*\|\s*`?([^`|]+?)`?\s*\|/g;
  let match: RegExpExecArray | null;

  while ((match = rowRegex.exec(docContent)) !== null) {
    const method = match[1].toLowerCase() as HttpMethod;
    const rawPath = match[2].trim();

    // Heuristic: only treat rows whose second column looks like a path.
    if (!rawPath.startsWith('/')) continue;

    const key = makeKey(method, rawPath);
    documented.add(key);
  }

  return documented;
}

// Endpoints that we intentionally *do not* require docs coverage for,
// even if they appear under /auth, /users, or /games in the OpenAPI
// spec. This should stay small and well-commented.
const openApiToDocsIgnore = new Set<string>([
  // Example (if we ever decide an internal admin-only endpoint should
  // remain undocumented):
  // makeKey('get', '/users/internal-debug'),
]);

export async function runApiEndpointsSsotCheck(): Promise<CheckResult> {
  const projectRoot = path.resolve(__dirname, '..', '..');
  const apiDocPath = path.join(projectRoot, 'docs/API_REFERENCE.md');

  if (!fs.existsSync(apiDocPath)) {
    return {
      name: 'api-endpoints-ssot',
      passed: false,
      details:
        'docs/API_REFERENCE.md is missing (cannot validate documented endpoints against OpenAPI spec).',
    };
  }

  const docContent = readFileSafe(apiDocPath);
  const documented = extractDocumentedEndpoints(docContent);
  const { all: specAll, publicAuthUsersGames } = extractOpenApiEndpoints();

  const missingInSpec: string[] = [];
  const undocumentedInDocs: string[] = [];

  // 1) docs → OpenAPI: every documented endpoint must exist in the spec.
  for (const key of documented) {
    if (!specAll.has(key)) {
      missingInSpec.push(key);
    }
  }

  // 2) OpenAPI → docs (for /auth, /users, /games only, minus ignore list).
  for (const key of publicAuthUsersGames) {
    if (openApiToDocsIgnore.has(key)) continue;
    if (!documented.has(key)) {
      undocumentedInDocs.push(key);
    }
  }

  if (missingInSpec.length === 0 && undocumentedInDocs.length === 0) {
    return {
      name: 'api-endpoints-ssot',
      passed: true,
      details:
        'All endpoints documented in docs/API_REFERENCE.md exist in the OpenAPI spec, and all public /auth, /users, /games endpoints in the spec are documented.',
    };
  }

  const problems: string[] = [];

  if (missingInSpec.length > 0) {
    problems.push(
      'The following documented endpoints do not exist in the OpenAPI spec (after normalisation):'
    );
    for (const key of missingInSpec) {
      problems.push(`- ${key}`);
    }
    problems.push(
      '\nFor each of these, either (a) update docs/API_REFERENCE.md to match the real route, (b) add or fix the corresponding JSDoc so it appears in swaggerSpec.paths, or (c) adjust the normalisation rules here if the path format is intentional.'
    );
  }

  if (undocumentedInDocs.length > 0) {
    problems.push(
      'The following public /auth, /users, /games endpoints exist in the OpenAPI spec but are not documented in docs/API_REFERENCE.md:'
    );
    for (const key of undocumentedInDocs) {
      problems.push(`- ${key}`);
    }
    problems.push(
      '\nFor each of these, either (a) add a row to the appropriate table in docs/API_REFERENCE.md, or (b) add the (method, path) pair to openApiToDocsIgnore in scripts/ssot/api-endpoints-ssot-check.ts if it is intentionally undocumented.'
    );
  }

  return {
    name: 'api-endpoints-ssot',
    passed: false,
    details: problems.join('\n'),
  };
}
