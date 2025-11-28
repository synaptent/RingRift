#!/usr/bin/env ts-node
/**
 * Lifecycle/API SSoT check
 *
 * Verifies that the core lifecycle-facing enums and discriminants in the
 * executable SSoT (src/shared/types/game.ts) are at least mentioned in the
 * canonical API doc (docs/CANONICAL_ENGINE_API.md).
 *
 * Initial invariants:
 * - Every non-legacy MoveType literal in src/shared/types/game.ts must
 *   appear somewhere in CANONICAL_ENGINE_API.md.
 * - Every PlayerChoiceType literal must appear in CANONICAL_ENGINE_API.md.
 *
 * This is intentionally conservative and text-based: it does not try to
 * understand sections or headings, but it does ensure that adding a new
 * MoveType/PlayerChoiceType without touching the API doc will fail fast.
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

function extractLiteralsFromUnion(source: string, anchor: string): Set<string> {
  const start = source.indexOf(anchor);
  if (start === -1) {
    throw new Error(`Anchor "${anchor}" not found in source file`);
  }

  // Slice from the anchor until the following exported type/interface.
  const tail = source.slice(start);
  const endMatch = tail.match(/\nexport (type|interface) [A-Za-z0-9_]+/);
  const block = endMatch ? tail.slice(0, endMatch.index ?? undefined) : tail;

  const literals = new Set<string>();
  const regex = /'([^'\n]+)'/g;
  let match: RegExpExecArray | null;
  // eslint-disable-next-line no-cond-assign
  while ((match = regex.exec(block)) !== null) {
    literals.add(match[1]);
  }
  return literals;
}

export async function runLifecycleSsotCheck(): Promise<CheckResult> {
  try {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const typesPath = path.join(projectRoot, 'src/shared/types/game.ts');
    const apiDocPath = path.join(projectRoot, 'docs/CANONICAL_ENGINE_API.md');

    const typesContent = readFileSafe(typesPath);
    const apiDocContent = readFileSafe(apiDocPath);

    const moveTypes = extractLiteralsFromUnion(typesContent, 'export type MoveType');
    const playerChoiceTypes = extractLiteralsFromUnion(
      typesContent,
      'export type PlayerChoiceType'
    );

    // Known legacy/experimental MoveTypes that we do not require to be
    // documented in CANONICAL_ENGINE_API.md.
    const legacyMoveTypes = new Set<string>(['line_formation', 'territory_claim']);

    const missingInDoc: string[] = [];

    for (const moveType of moveTypes) {
      if (legacyMoveTypes.has(moveType)) continue;
      const pattern = new RegExp(`\\b${moveType.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`);
      if (!pattern.test(apiDocContent)) {
        missingInDoc.push(`MoveType '${moveType}'`);
      }
    }

    for (const choiceType of playerChoiceTypes) {
      const pattern = new RegExp(`\\b${choiceType.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`);
      if (!pattern.test(apiDocContent)) {
        missingInDoc.push(`PlayerChoiceType '${choiceType}'`);
      }
    }

    if (missingInDoc.length === 0) {
      return {
        name: 'lifecycle-api-ssot',
        passed: true,
        details:
          'MoveType and PlayerChoiceType literals are referenced in CANONICAL_ENGINE_API.md.',
      };
    }

    return {
      name: 'lifecycle-api-ssot',
      passed: false,
      details: `Lifecycle/API doc is missing references to: ${missingInDoc.join(', ')}`,
    };
  } catch (error) {
    const err = error as Error;
    return {
      name: 'lifecycle-api-ssot',
      passed: false,
      details: err.message ?? String(err),
    };
  }
}
