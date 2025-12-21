#!/usr/bin/env ts-node
/**
 * Python parity & contracts SSoT check
 *
 * Verifies that canonical enums and contracts remain aligned between:
 * - TS types (`src/shared/types/game.ts`)
 * - Python mirror (`ai-service/app/models/core.py`)
 * - Python legacy alias map (`ai-service/app/rules/legacy/move_type_aliases.py`)
 *
 * This prevents drift that breaks replay parity and canonical DB validation.
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

function expectFile(relativePath: string, projectRoot: string, problems: string[]): void {
  const fullPath = path.join(projectRoot, relativePath);
  if (!fs.existsSync(fullPath)) {
    problems.push(`Expected file not found: ${relativePath}`);
  }
}

function stripTsComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
}

function uniqueSorted(values: string[]): string[] {
  return Array.from(new Set(values)).sort();
}

function extractTsUnionValues(source: string, typeName: string): string[] {
  const regex = new RegExp(`export type ${typeName} =([\\s\\S]*?);`, 'm');
  const match = regex.exec(source);
  if (!match) {
    throw new Error(`Failed to locate TS union for ${typeName}`);
  }
  const body = stripTsComments(match[1]);
  const values: string[] = [];
  const valueRegex = /'([^']+)'/g;
  let valueMatch: RegExpExecArray | null;
  while ((valueMatch = valueRegex.exec(body)) !== null) {
    values.push(valueMatch[1]);
  }
  return uniqueSorted(values);
}

function extractPythonClassValues(source: string, className: string): string[] {
  const classRegex = new RegExp(`^class\\s+${className}\\b[\\s\\S]*?:`, 'm');
  const match = classRegex.exec(source);
  if (!match) {
    throw new Error(`Failed to locate Python enum ${className}`);
  }
  const rest = source.slice(match.index + match[0].length);
  const nextClassIndex = rest.search(/^\s*class\s+\w+/m);
  const block = nextClassIndex === -1 ? rest : rest.slice(0, nextClassIndex);
  const values: string[] = [];
  const valueRegex = /=\s*["']([^"']+)["']/g;
  let valueMatch: RegExpExecArray | null;
  while ((valueMatch = valueRegex.exec(block)) !== null) {
    values.push(valueMatch[1]);
  }
  return uniqueSorted(values);
}

function extractPythonLegacyMoveAliases(source: string): string[] {
  const mapStart = source.indexOf('LEGACY_TO_CANONICAL_MOVE_TYPE');
  if (mapStart === -1) {
    return [];
  }
  const braceStart = source.indexOf('{', mapStart);
  const braceEnd = source.indexOf('}', braceStart);
  if (braceStart === -1 || braceEnd === -1) {
    return [];
  }
  const body = source.slice(braceStart + 1, braceEnd);
  const values: string[] = [];
  const keyRegex = /["']([^"']+)["']\s*:/g;
  let keyMatch: RegExpExecArray | null;
  while ((keyMatch = keyRegex.exec(body)) !== null) {
    values.push(keyMatch[1].toLowerCase());
  }
  return uniqueSorted(values);
}

function setDiff(a: readonly string[], b: readonly string[]): string[] {
  const bSet = new Set(b);
  return a.filter((v) => !bSet.has(v));
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

    // Parity requirements doc now lives under docs/rules/.
    expectFile('docs/rules/PYTHON_PARITY_REQUIREMENTS.md', projectRoot, problems);

    const tsTypesPath = path.join(projectRoot, 'src', 'shared', 'types', 'game.ts');
    const pyModelsPath = path.join(projectRoot, 'ai-service', 'app', 'models', 'core.py');
    const pyLegacyAliasesPath = path.join(
      projectRoot,
      'ai-service',
      'app',
      'rules',
      'legacy',
      'move_type_aliases.py'
    );

    const tsTypesSource = readFileSafe(tsTypesPath);
    const pyModelsSource = readFileSafe(pyModelsPath);
    const pyLegacyAliasesSource = readFileSafe(pyLegacyAliasesPath);

    const tsGamePhases = extractTsUnionValues(tsTypesSource, 'GamePhase');
    const tsCanonicalMoveTypes = extractTsUnionValues(tsTypesSource, 'CanonicalMoveType');
    const tsLegacyMoveTypes = extractTsUnionValues(tsTypesSource, 'LegacyMoveType');
    const tsBoardTypes = extractTsUnionValues(tsTypesSource, 'BoardType');

    const pyGamePhases = extractPythonClassValues(pyModelsSource, 'GamePhase');
    const pyMoveTypes = extractPythonClassValues(pyModelsSource, 'MoveType');
    const pyBoardTypes = extractPythonClassValues(pyModelsSource, 'BoardType');

    const missingPhases = setDiff(tsGamePhases, pyGamePhases);
    const extraPhases = setDiff(pyGamePhases, tsGamePhases);
    if (missingPhases.length > 0) {
      problems.push(`Missing GamePhase values in Python: ${missingPhases.join(', ')}`);
    }
    if (extraPhases.length > 0) {
      problems.push(`Extra GamePhase values in Python: ${extraPhases.join(', ')}`);
    }

    const missingBoards = setDiff(tsBoardTypes, pyBoardTypes);
    const extraBoards = setDiff(pyBoardTypes, tsBoardTypes);
    if (missingBoards.length > 0) {
      problems.push(`Missing BoardType values in Python: ${missingBoards.join(', ')}`);
    }
    if (extraBoards.length > 0) {
      problems.push(`Extra BoardType values in Python: ${extraBoards.join(', ')}`);
    }

    const missingCanonicalMoves = setDiff(tsCanonicalMoveTypes, pyMoveTypes);
    if (missingCanonicalMoves.length > 0) {
      problems.push(
        `Missing canonical MoveType values in Python: ${missingCanonicalMoves.join(', ')}`
      );
    }

    const missingLegacyMoves = setDiff(tsLegacyMoveTypes, pyMoveTypes);
    if (missingLegacyMoves.length > 0) {
      problems.push(`Missing legacy MoveType values in Python: ${missingLegacyMoves.join(', ')}`);
    }

    const pythonLegacyAliases = extractPythonLegacyMoveAliases(pyLegacyAliasesSource);
    const allowedExtras = new Set([...tsLegacyMoveTypes, ...pythonLegacyAliases]);
    const extraPythonMoves = setDiff(pyMoveTypes, tsCanonicalMoveTypes);
    const unexpectedExtras = extraPythonMoves.filter((move) => !allowedExtras.has(move));
    if (unexpectedExtras.length > 0) {
      problems.push(`Unexpected Python MoveType values: ${unexpectedExtras.join(', ')}`);
    }

    if (problems.length === 0) {
      return {
        name: 'python-parity-ssot',
        passed: true,
        details:
          'Contract vectors, TS/Python runners, parity requirements doc, and enum parity are aligned.',
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
