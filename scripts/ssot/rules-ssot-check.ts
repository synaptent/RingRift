#!/usr/bin/env ts-node
/**
 * Rules semantics SSoT check
 *
 * Verifies basic alignment between the canonical rules specification
 * (RULES_CANONICAL_SPEC.md) and the implementation mapping
 * (RULES_IMPLEMENTATION_MAPPING.md).
 *
 * Initial invariants:
 * - Every RR-CANON rule ID declared in RULES_CANONICAL_SPEC.md must appear
 *   somewhere in RULES_IMPLEMENTATION_MAPPING.md (at least as `RXYZ`).
 * - Every RXYZ reference in RULES_IMPLEMENTATION_MAPPING.md must correspond
 *   to a rule declared in RULES_CANONICAL_SPEC.md.
 *
 * This is intentionally a light-weight, text-based check; it does not
 * attempt to understand clusters or which engine modules/tests implement
 * a given rule yet. Those richer checks can be layered on later.
 */

import * as fs from 'fs';
import * as path from 'path';

interface CheckResult {
  name: string;
  passed: boolean;
  details?: string;
}

function readFileSafe(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function walkFiles(dir: string, exts: string[]): string[] {
  const results: string[] = [];
  if (!fs.existsSync(dir)) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkFiles(fullPath, exts));
    } else if (exts.some((ext) => entry.name.endsWith(ext))) {
      results.push(fullPath);
    }
  }

  return results;
}

function extractRuleIdsFromSpec(specContent: string): Set<string> {
  const ids = new Set<string>();
  const regex = /RR-CANON-(R\d{3})/g;
  let match: RegExpExecArray | null;
  // eslint-disable-next-line no-cond-assign
  while ((match = regex.exec(specContent)) !== null) {
    ids.add(match[1]); // e.g. "R001"
  }
  return ids;
}

function extractRuleIdsFromMapping(mappingContent: string): Set<string> {
  const ids = new Set<string>();
  const regex = /R(\d{3})/g;
  let match: RegExpExecArray | null;
  // eslint-disable-next-line no-cond-assign
  while ((match = regex.exec(mappingContent)) !== null) {
    ids.add(`R${match[1]}`);
  }
  return ids;
}

/**
 * Guardrail: prevent new production code from introducing additional
 * direct imports of legacy backend helpers such as `RuleEngine`.
 *
 * For now, the only sanctioned direct imports of `RuleEngine` are:
 * - `src/server/game/GameEngine.ts`
 * - `src/server/game/test-parity-cli.ts` (diagnostic CLI)
 *
 * Any other `from './RuleEngine'` imports under `src/server/game/**`
 * are treated as violations.
 */
function checkLegacyRuleEngineImports(projectRoot: string): string[] {
  const problems: string[] = [];
  const serverGameDir = path.join(projectRoot, 'src', 'server', 'game');

  const allowedImporters = new Set<string>([
    path.join(serverGameDir, 'GameEngine.ts'),
    path.join(serverGameDir, 'test-parity-cli.ts'),
  ]);

  const candidates = walkFiles(serverGameDir, ['.ts', '.tsx']);

  for (const filePath of candidates) {
    // Skip the legacy module itself; we only care about who imports it.
    if (filePath === path.join(serverGameDir, 'RuleEngine.ts')) continue;

    const content = readFileSafe(filePath);
    if (
      content.includes("from './RuleEngine'") ||
      content.includes('from "./RuleEngine"') ||
      content.includes("from './RuleEngine.ts'") ||
      content.includes('from "./RuleEngine.ts"')
    ) {
      if (!allowedImporters.has(filePath)) {
        const rel = path.relative(projectRoot, filePath);
        problems.push(
          `Unexpected import of './RuleEngine' in ${rel}. Only GameEngine.ts and test-parity-cli.ts may reference RuleEngine directly; new production code should route through TurnEngineAdapter + aggregates.`
        );
      }
    }
  }

  return problems;
}

/**
 * Guardrail: prevent diagnostics-only sandbox helpers from being imported
 * from production client hosts (GamePage, SandboxContext, ClientSandboxEngine,
 * or core sandbox hooks).
 *
 * - `localSandboxController` is a legacy experimental harness and must not be
 *   reintroduced as a rules host for `/sandbox`.
 * - `sandboxCaptureSearch` is an analysis-only capture-chain search helper
 *   and must not be wired into production sandbox flows.
 */
function checkDiagnosticsOnlyClientImports(projectRoot: string): string[] {
  const problems: string[] = [];
  const clientRoot = path.join(projectRoot, 'src', 'client');

  const guardedFiles = [
    path.join(clientRoot, 'pages', 'GamePage.tsx'),
    path.join(clientRoot, 'contexts', 'SandboxContext.tsx'),
    path.join(clientRoot, 'sandbox', 'ClientSandboxEngine.ts'),
    path.join(clientRoot, 'hooks', 'useSandboxInteractions.ts'),
  ];

  for (const filePath of guardedFiles) {
    if (!fs.existsSync(filePath)) continue;
    const content = readFileSafe(filePath);
    const rel = path.relative(projectRoot, filePath);

    if (
      content.includes("from '../sandbox/localSandboxController'") ||
      content.includes('from "../sandbox/localSandboxController"') ||
      content.includes("from './localSandboxController'") ||
      content.includes('from "./localSandboxController"')
    ) {
      problems.push(
        `Diagnostics-only legacy sandbox harness 'localSandboxController' must not be imported from ${rel}. /sandbox production flows should use ClientSandboxEngine + SandboxOrchestratorAdapter instead.`
      );
    }

    if (
      content.includes("from '../sandbox/sandboxCaptureSearch'") ||
      content.includes('from "../sandbox/sandboxCaptureSearch"') ||
      content.includes("from '../../client/sandbox/sandboxCaptureSearch'") ||
      content.includes('from "../../client/sandbox/sandboxCaptureSearch"')
    ) {
      problems.push(
        `Diagnostics-only capture search helper 'sandboxCaptureSearch' must not be imported from ${rel}. It is reserved for tests and CLI diagnostics, not for production sandbox hosts.`
      );
    }
  }

  return problems;
}

/**
 * Guardrail: sandbox-only helper modules must not be imported from
 * non-sandbox client hosts or any server code. These helpers exist to
 * support the /sandbox route, diagnostics, and Jest parity tests; they
 * are not general-purpose UI or backend utilities.
 *
 * Allowed import sites:
 * - src/client/sandbox/** (sandbox engine + helpers)
 * - tests/** (Jest suites)
 * - archive/** and docs/scripts that consume sandbox helpers for diagnostics
 *
 * Under src/** we enforce that only the sandbox subtree may import these
 * helpers; other client/server modules must rely on shared engine
 * aggregates instead.
 */
function checkSandboxHelperImportBoundaries(projectRoot: string): string[] {
  const problems: string[] = [];
  const srcRoot = path.join(projectRoot, 'src');
  if (!fs.existsSync(srcRoot)) {
    return problems;
  }

  const sandboxRoot = path.join(srcRoot, 'client', 'sandbox');
  const helpers = [
    'sandboxLines',
    'sandboxTerritory',
    'sandboxVictory',
    'sandboxGameEnd',
    'sandboxElimination',
    'sandboxCaptureSearch',
  ];

  const candidates = walkFiles(srcRoot, ['.ts', '.tsx']);

  for (const filePath of candidates) {
    // Allow imports from the sandbox subtree itself.
    if (filePath.startsWith(sandboxRoot)) continue;

    const content = readFileSafe(filePath);
    const rel = path.relative(projectRoot, filePath);

    for (const helper of helpers) {
      if (
        content.includes(`from './${helper}'`) ||
        content.includes(`from "../sandbox/${helper}"`) ||
        content.includes(`from '../sandbox/${helper}'`) ||
        content.includes(`from "../../client/sandbox/${helper}"`) ||
        content.includes(`from '../../client/sandbox/${helper}'`)
      ) {
        problems.push(
          `Sandbox-only helper '${helper}' must not be imported from ${rel}. It is reserved for src/client/sandbox hosts and tests; other modules should depend on shared engine aggregates instead.`
        );
      }
    }
  }

  return problems;
}

export async function runRulesSsotCheck(): Promise<CheckResult> {
  try {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const specPath = path.join(projectRoot, 'RULES_CANONICAL_SPEC.md');
    const mappingPath = path.join(projectRoot, 'RULES_IMPLEMENTATION_MAPPING.md');

    const specContent = readFileSafe(specPath);
    const mappingContent = readFileSafe(mappingPath);

    const specIds = extractRuleIdsFromSpec(specContent);
    const mappingIds = extractRuleIdsFromMapping(mappingContent);

    const missingInMapping: string[] = [];
    for (const id of specIds) {
      // We treat any occurrence of the bare RXYZ token as a mapping reference,
      // which covers both explicit IDs and ranges like "R001–R003".
      if (!mappingIds.has(id)) {
        missingInMapping.push(id);
      }
    }

    const missingInSpec: string[] = [];
    for (const id of mappingIds) {
      if (!specIds.has(id)) {
        missingInSpec.push(id);
      }
    }

    const problems: string[] = [];

    if (missingInMapping.length > 0) {
      problems.push(
        `Rules declared in RULES_CANONICAL_SPEC.md but not referenced in RULES_IMPLEMENTATION_MAPPING.md: ${missingInMapping
          .sort()
          .join(', ')}`
      );
    }

    if (missingInSpec.length > 0) {
      problems.push(
        `Rule IDs referenced in RULES_IMPLEMENTATION_MAPPING.md but not declared in RULES_CANONICAL_SPEC.md: ${missingInSpec
          .sort()
          .join(', ')}`
      );
    }

    // Enforce that no new production code introduces additional direct
    // imports of the legacy backend RuleEngine beyond the known, fenced
    // sites (GameEngine and test-parity-cli).
    const legacyImportProblems = checkLegacyRuleEngineImports(projectRoot);
    const diagnosticsImportProblems = checkDiagnosticsOnlyClientImports(projectRoot);
    const sandboxImportProblems = checkSandboxHelperImportBoundaries(projectRoot);
    problems.push(...legacyImportProblems, ...diagnosticsImportProblems, ...sandboxImportProblems);

    if (problems.length === 0) {
      return {
        name: 'rules-semantics-ssot',
        passed: true,
        details:
          'RR-CANON rule IDs are consistently referenced between spec and mapping, legacy RuleEngine imports are confined to approved callers, and diagnostics-only sandbox helpers are not imported from production hosts.',
      };
    }

    return {
      name: 'rules-semantics-ssot',
      passed: false,
      details: problems.join('\n'),
    };
  } catch (error) {
    const err = error as Error;
    return {
      name: 'rules-semantics-ssot',
      passed: false,
      details: err.message ?? String(err),
    };
  }
}
