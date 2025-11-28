#!/usr/bin/env ts-node
/**
 * docs-link-ssot
 *
 * Lightweight SSoT guard for high‑value documentation links.
 *
 * Responsibilities:
 * - For a curated set of markdown docs, verify that file‑relative links
 *   (e.g. `[text](../path/file.md)` or `[text](./file.md#anchor)`) resolve to
 *   existing files within the repo.
 * - For Prometheus alert `runbook_url` annotations in
 *   `monitoring/prometheus/alerts.yml`, verify that each GitHub URL pointing
 *   at `docs/runbooks/*.md` corresponds to a real runbook file.
 *
 * This check is intentionally conservative:
 * - It ignores external http/https/mailto links in markdown.
 * - It does not attempt to validate `#anchor` fragments; it only validates
 *   that the target file exists.
 * - It focuses on a curated list of markdown docs that are known to be
 *   operationally important rather than scanning the entire tree.
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

function fileExistsSafe(filePath: string): boolean {
  try {
    return fs.existsSync(filePath);
  } catch {
    return false;
  }
}

/**
 * Curated set of high‑value docs whose internal links should be kept healthy.
 * These are expressed as paths relative to the project root.
 */
const MARKDOWN_DOCS_TO_CHECK: string[] = [
  // AI & training
  'AI_ARCHITECTURE.md',
  'docs/AI_TRAINING_AND_DATASETS.md',
  'docs/AI_TRAINING_PREPARATION_GUIDE.md',
  'docs/AI_TRAINING_ASSESSMENT_FINAL.md',
  // API & lifecycle
  'docs/API_REFERENCE.md',
  'docs/STATE_MACHINES.md',
  'docs/CANONICAL_ENGINE_API.md',
  // Data & ops
  'docs/DATA_LIFECYCLE_AND_PRIVACY.md',
  'docs/ENVIRONMENT_VARIABLES.md',
  'docs/DEPLOYMENT_REQUIREMENTS.md',
  'docs/OPERATIONS_DB.md',
  'docs/ALERTING_THRESHOLDS.md',
  // Incidents & runbooks indices (they in turn link to concrete docs)
  'docs/incidents/INDEX.md',
  'docs/runbooks/INDEX.md',
];

/**
 * Basic markdown link extractor: finds `[label](target)` pairs.
 * This is deliberately simple and not a full markdown parser; it is
 * sufficient for our curated docs, which use standard link syntax.
 */
const MARKDOWN_LINK_REGEX = /\[([^\]]+)\]\(([^)]+)\)/g;

function isProbablyFileLink(target: string): boolean {
  // External links and mailto are ignored completely.
  if (target.startsWith('http://') || target.startsWith('https://')) return false;
  if (target.startsWith('mailto:')) return false;

  // Pure anchor links (e.g. `#section`) are ignored here; anchor validation
  // is brittle across markdown renderers and not worth enforcing right now.
  if (target.startsWith('#')) return false;

  // Absolute paths like `/api/docs` are API paths, not file paths.
  if (target.startsWith('/')) return false;

  // Everything else is treated as a candidate file‑relative link.
  return true;
}

function collectBrokenMarkdownLinks(projectRoot: string): string[] {
  const problems: string[] = [];

  for (const relativePath of MARKDOWN_DOCS_TO_CHECK) {
    const absolutePath = path.join(projectRoot, relativePath);

    if (!fileExistsSafe(absolutePath)) {
      problems.push(`- Doc missing: ${relativePath}`);
      continue;
    }

    const content = readFileSafe(absolutePath);
    const dir = path.dirname(absolutePath);

    const brokenLinks: string[] = [];
    let match: RegExpExecArray | null;

    MARKDOWN_LINK_REGEX.lastIndex = 0;
    while ((match = MARKDOWN_LINK_REGEX.exec(content)) !== null) {
      const rawTarget = match[2].trim();
      if (!isProbablyFileLink(rawTarget)) continue;

      // Strip any `#anchor` portion; we only care about the file itself.
      const [targetPath] = rawTarget.split('#', 1);
      const resolved = path.resolve(dir, targetPath);

      if (!fileExistsSafe(resolved)) {
        const relFromRoot = path.relative(projectRoot, resolved) || resolved;
        brokenLinks.push(`  • ${rawTarget} → ${relFromRoot} (missing)`);
      }
    }

    if (brokenLinks.length > 0) {
      problems.push(`- Broken links in ${relativePath}:`);
      problems.push(...brokenLinks);
    }
  }

  return problems;
}

/**
 * Verify that Prometheus alert `runbook_url` values in
 * `monitoring/prometheus/alerts.yml` point at existing runbook files under
 * `docs/runbooks/`.
 */
function collectMissingRunbookFiles(projectRoot: string): string[] {
  const alertsPath = path.join(projectRoot, 'monitoring/prometheus/alerts.yml');
  if (!fileExistsSafe(alertsPath)) {
    // If alerts.yml itself is missing, ci-config-ssot will already fail; no
    // need to double‑report here.
    return [];
  }

  const content = readFileSafe(alertsPath);
  const missing: string[] = [];

  const runbookRegex = /runbook_url:\s*"([^"]+)"/g;
  let match: RegExpExecArray | null;

  while ((match = runbookRegex.exec(content)) !== null) {
    const url = match[1];

    // We only care about GitHub URLs that point into docs/runbooks via
    // the standard blob/main pattern.
    const marker = '/blob/main/';
    const idx = url.indexOf(marker);
    if (idx === -1) continue;

    const relPath = url.slice(idx + marker.length);
    const absolute = path.join(projectRoot, relPath);

    if (!fileExistsSafe(absolute)) {
      missing.push(`  • ${url} → ${relPath} (missing runbook file)`);
    }
  }

  if (missing.length > 0) {
    return ['- Missing runbook files for Prometheus alert runbook_url links:', ...missing];
  }

  return [];
}

export async function runDocsLinkSsotCheck(): Promise<CheckResult> {
  const projectRoot = path.resolve(__dirname, '..', '..');

  const problems: string[] = [];

  // 1) Broken markdown links in curated docs
  problems.push(...collectBrokenMarkdownLinks(projectRoot));

  // 2) Missing runbook files referenced from Prometheus alerts
  problems.push(...collectMissingRunbookFiles(projectRoot));

  if (problems.length === 0) {
    return {
      name: 'docs-link-ssot',
      passed: true,
      details:
        'All curated markdown docs have resolvable file-relative links, and all Prometheus alert runbook_url links point at existing runbook files.',
    };
  }

  return {
    name: 'docs-link-ssot',
    passed: false,
    details:
      problems.join('\n') +
      '\n\nTo fix: update or remove broken links in the listed docs, or create the missing runbook files under docs/runbooks/.',
  };
}
