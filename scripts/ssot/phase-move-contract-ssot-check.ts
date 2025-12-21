#!/usr/bin/env ts-node
/**
 * Phase↔MoveType contract SSoT check
 *
 * Ensures the canonical phase→MoveType mapping is kept in lockstep between:
 * - TS engine canonical contract (`src/shared/engine/phaseValidation.ts`)
 * - Python canonical history contract (`ai-service/app/rules/history_contract.py`)
 *
 * This is a drift guard against subtle contract skew that can break:
 * - canonical DB recording (write-time rejection),
 * - TS↔Python replay parity,
 * - history validation and downstream training pipelines.
 */

import * as path from 'path';
import { execFileSync } from 'child_process';

import type { CheckResult } from './ssot-check';

import { VALID_MOVES_BY_PHASE } from '../../src/shared/engine/phaseValidation';

function normalizeMoveList(raw: unknown): string[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter((v) => typeof v === 'string')
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .sort();
}

function setDiff(a: readonly string[], b: readonly string[]): string[] {
  const bSet = new Set(b);
  return a.filter((v) => !bSet.has(v));
}

function getPythonPhaseMoveContract(projectRoot: string): Record<string, string[]> {
  const pythonSnippet =
    'import json; from app.rules.history_contract import phase_move_contract; print(json.dumps(phase_move_contract()))';

  const candidates = [process.env.PYTHON, 'python3', 'python'].filter(Boolean) as string[];

  let lastError: unknown = undefined;
  for (const cmd of candidates) {
    try {
      const stdout = execFileSync(cmd, ['-c', pythonSnippet], {
        cwd: path.join(projectRoot, 'ai-service'),
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      return JSON.parse(stdout) as Record<string, string[]>;
    } catch (err) {
      lastError = err;
    }
  }

  const error = lastError as Error | undefined;
  throw new Error(error?.message ?? 'Failed to execute python to read phase_move_contract()');
}

export async function runPhaseMoveContractSsotCheck(): Promise<CheckResult> {
  try {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const problems: string[] = [];

    const pythonContractRaw = getPythonPhaseMoveContract(projectRoot);
    const pythonPhases = Object.keys(pythonContractRaw).sort();

    const tsContractRaw: Record<string, readonly string[]> = {};
    for (const [phase, moves] of Object.entries(VALID_MOVES_BY_PHASE)) {
      if (phase === 'game_over') {
        continue;
      }
      tsContractRaw[phase] = moves as readonly string[];
    }
    const tsPhases = Object.keys(tsContractRaw).sort();

    const missingInTs = pythonPhases.filter((p) => !tsPhases.includes(p));
    const extraInTs = tsPhases.filter((p) => !pythonPhases.includes(p));
    if (missingInTs.length > 0) {
      problems.push(`Missing phases in TS contract: ${missingInTs.join(', ')}`);
    }
    if (extraInTs.length > 0) {
      problems.push(`Extra phases in TS contract: ${extraInTs.join(', ')}`);
    }

    const phasesToCompare = Array.from(new Set([...pythonPhases, ...tsPhases])).sort();

    for (const phase of phasesToCompare) {
      const pyMoves = normalizeMoveList(pythonContractRaw[phase]);
      const tsMoves = normalizeMoveList(tsContractRaw[phase] ?? []);

      const missingMoves = setDiff(pyMoves, tsMoves);
      const extraMoves = setDiff(tsMoves, pyMoves);

      if (missingMoves.length > 0 || extraMoves.length > 0) {
        const parts: string[] = [];
        if (missingMoves.length > 0) {
          parts.push(`missing in TS: ${missingMoves.join(', ')}`);
        }
        if (extraMoves.length > 0) {
          parts.push(`extra in TS: ${extraMoves.join(', ')}`);
        }
        problems.push(`Phase '${phase}' mismatch (${parts.join(' | ')})`);
      }
    }

    if (problems.length === 0) {
      return {
        name: 'phase-move-contract-ssot',
        passed: true,
        details: 'TS and Python canonical phase→MoveType contracts match.',
      };
    }

    return {
      name: 'phase-move-contract-ssot',
      passed: false,
      details: problems.join('\n'),
    };
  } catch (error) {
    const err = error as Error;
    return {
      name: 'phase-move-contract-ssot',
      passed: false,
      details: err.message ?? String(err),
    };
  }
}
