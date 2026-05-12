#!/usr/bin/env -S npx tsx
/**
 * Post-build security gate for the public client bundle.
 *
 * Scans `dist/client/**\/*.js` (and any source map shipped alongside it) for
 * credential-shaped strings that must never end up on the CDN. Mirrors the
 * `SECRET_SHAPED_PATTERNS` in `vite.config.ts`, but runs against the
 * *output* — so it catches leaks that Vite/Rollup substitutions produced
 * even if no value matched the input-side guard.
 *
 * History: 2026-04 a `startsWith('RINGRIFT_')` wildcard in
 * `vite.config.ts`'s `define` block baked `RINGRIFT_SLACK_WEBHOOK` into
 * the public client bundle (`assets/index-D1WcbuXm.js`). The fix narrowed
 * the allowlist; this scanner is the belt-and-braces second line.
 *
 * Usage:
 *   npx tsx scripts/check-bundle-secrets.ts
 *   npx tsx scripts/check-bundle-secrets.ts --dir dist/client
 *   npx tsx scripts/check-bundle-secrets.ts --json
 *
 * Exit codes:
 *   0  no findings
 *   1  at least one secret-shaped string found
 *   2  scan input was missing or unusable
 *
 * Wire into CI as a build gate so the build fails before the artifact
 * can be uploaded.
 */

import { readdirSync, readFileSync, statSync } from 'fs';
import { join, relative, extname } from 'path';

interface Pattern {
  name: string;
  re: RegExp;
  /**
   * Per-pattern allowlist of literal substrings that are known safe even
   * though they match the pattern (e.g. test fixtures, documentation
   * URLs). Keep this *narrow*.
   */
  knownSafe?: string[];
}

const PATTERNS: Pattern[] = [
  { name: 'Slack webhook URL', re: /hooks\.slack\.com\/services\/[A-Z0-9/]+/g },
  // AWS access key ids have a unique 20-char shape (AKIA + 16 base32-ish).
  // We deliberately do NOT scan for the 40-char secret access key shape:
  // minified JS contains long [A-Za-z0-9] identifier runs that produce
  // unavoidable false positives. The AKIA prefix is enough to detect leaks —
  // an access key id and its secret are always provisioned together, so
  // catching one is catching both.
  { name: 'AWS access key id', re: /\bAKIA[0-9A-Z]{16}\b/g },
  { name: 'postgres URL', re: /\bpostgres(?:ql)?:\/\/[^\s"'`<>]+/g },
  // JWT-shaped: 64+ hex chars with at least one digit AND at least one
  // a-f letter. Pure hex runs from minified bundle content (e.g. hashes
  // baked in by manualChunks) are unlikely to satisfy both conditions.
  { name: 'JWT-shaped 64+ hex token', re: /\b(?=[a-f0-9]*[a-f])(?=[a-f0-9]*\d)[a-f0-9]{64,}\b/g },
  { name: 'GitHub token', re: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}/g },
  { name: 'private key block', re: /-----BEGIN [A-Z ]*PRIVATE KEY-----/g },
];

interface Finding {
  file: string;
  pattern: string;
  match: string;
  index: number;
}

const MAX_FINDINGS_PER_PATTERN_PER_FILE = 5;

function isJsLike(filename: string): boolean {
  const ext = extname(filename).toLowerCase();
  return ext === '.js' || ext === '.mjs' || ext === '.cjs' || ext === '.map';
}

function walk(dir: string, out: string[] = []): string[] {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return out;
  }
  for (const entry of entries) {
    const full = join(dir, entry);
    let st;
    try {
      st = statSync(full);
    } catch {
      continue;
    }
    if (st.isDirectory()) {
      walk(full, out);
    } else if (st.isFile() && isJsLike(entry)) {
      out.push(full);
    }
  }
  return out;
}

function scanFile(path: string): Finding[] {
  const findings: Finding[] = [];
  let body: string;
  try {
    body = readFileSync(path, 'utf8');
  } catch (err) {
    process.stderr.write(`[check-bundle-secrets] failed to read ${path}: ${String(err)}\n`);
    return findings;
  }
  for (const pattern of PATTERNS) {
    let count = 0;
    for (const m of body.matchAll(pattern.re)) {
      const match = m[0];
      if (pattern.knownSafe && pattern.knownSafe.includes(match)) continue;
      findings.push({ file: path, pattern: pattern.name, match, index: m.index ?? -1 });
      count += 1;
      if (count >= MAX_FINDINGS_PER_PATTERN_PER_FILE) break;
    }
  }
  return findings;
}

function redact(s: string): string {
  if (s.length <= 12) return '***';
  return `${s.slice(0, 6)}…${s.slice(-4)}`;
}

function main(): never {
  const args = process.argv.slice(2);
  let dir = 'dist/client';
  let json = false;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--dir') {
      dir = args[++i] ?? dir;
    } else if (arg === '--json') {
      json = true;
    } else if (arg === '-h' || arg === '--help') {
      process.stdout.write(
        'Usage: check-bundle-secrets.ts [--dir <path>] [--json]\n' +
          'Scans built client artifacts for credential-shaped strings.\n'
      );
      process.exit(0);
    } else {
      process.stderr.write(`[check-bundle-secrets] unknown arg: ${arg}\n`);
      process.exit(2);
    }
  }

  const files = walk(dir);
  if (files.length === 0) {
    process.stderr.write(
      `[check-bundle-secrets] no .js/.mjs/.cjs/.map files found under ${dir}; ` +
        `did the build run?\n`
    );
    process.exit(2);
  }

  const findings: Finding[] = [];
  for (const f of files) findings.push(...scanFile(f));

  if (json) {
    process.stdout.write(
      JSON.stringify(
        {
          dir,
          filesScanned: files.length,
          findings: findings.map((f) => ({
            file: relative(process.cwd(), f.file),
            pattern: f.pattern,
            match: redact(f.match),
            index: f.index,
          })),
        },
        null,
        2
      ) + '\n'
    );
  } else if (findings.length === 0) {
    process.stdout.write(
      `[check-bundle-secrets] OK — scanned ${files.length} file(s) under ${dir}; ` +
        `no credential-shaped strings found.\n`
    );
  } else {
    process.stderr.write(
      `[check-bundle-secrets] FAIL — ${findings.length} finding(s) in ` +
        `${files.length} file(s) under ${dir}:\n`
    );
    for (const f of findings) {
      process.stderr.write(
        `  ${relative(process.cwd(), f.file)}: ${f.pattern} ` +
          `(@${f.index} -> ${redact(f.match)})\n`
      );
    }
    process.stderr.write(
      `\nIf any of these are intentional (test fixtures, etc.), add the\n` +
        `exact value to that pattern's \`knownSafe\` list in this script.\n` +
        `Otherwise, find what's leaking it into the bundle (check vite.config.ts\n` +
        `define.process.env and any shared code referencing process.env.<key>).\n`
    );
  }

  process.exit(findings.length > 0 ? 1 : 0);
}

main();
