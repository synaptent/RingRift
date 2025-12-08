/**
 * @deprecated Phase 4 legacy path test (trimmed)
 *
 * This file previously exercised legacy internal `GameEngine` helpers
 * (`processLineFormations()`, `processDisconnectedRegions()`, etc.) for a
 * grab-bag of RulesMatrix scenarios. Those flows have been replaced by:
 *
 *   - Orchestrator-backed multi-phase suites:
 *       • tests/scenarios/Orchestrator.Backend.multiPhase.test.ts
 *       • tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts
 *   - Per-domain RulesMatrix/FAQ scenario tests under tests/scenarios/*
 *   - v2 contract vectors in tests/fixtures/contract-vectors/v2, consumed
 *     by ai-service/tests/contracts/test_contract_vectors.py
 *
 * To avoid keeping legacy `GameEngine` internals alive, this file is now a
 * thin wrapper that sanity-checks the presence of a few canonical v2
 * contract-vector IDs corresponding to complex RulesMatrix axes (chain
 * capture, forced elimination, line+territory endgames, hex edge cases).
 */

import fs from 'fs';
import path from 'path';

import {
  importVectorBundle,
  type ContractTestVector,
} from '../../src/shared/engine/contracts/testVectorGenerator';

function loadBundle(fileName: string): ContractTestVector[] {
  const bundlePath = path.resolve(__dirname, '../fixtures/contract-vectors/v2', fileName);
  if (!fs.existsSync(bundlePath)) {
    throw new Error(`Contract vector bundle not found: ${bundlePath}`);
  }

  const json = fs.readFileSync(bundlePath, 'utf8');
  return importVectorBundle(json);
}

describe('RulesMatrix Comprehensive Scenarios – v2 contract vector coverage', () => {
  test('chain capture long-tail vectors include depth-3 square and hex cases', () => {
    const vectors = loadBundle('chain_capture_long_tail.vectors.json');
    const ids = new Set(vectors.map((v) => v.id));

    expect(ids.has('chain_capture.depth3.segment1.square8')).toBe(true);
    expect(ids.has('chain_capture.depth3.segment3.square8')).toBe(true);
    expect(ids.has('chain_capture.depth3.segment1.square19')).toBe(true);
    expect(ids.has('chain_capture.depth3.segment3.square19')).toBe(true);
    expect(ids.has('chain_capture.depth3.segment1.hexagonal')).toBe(true);
    expect(ids.has('chain_capture.depth3.segment3.hexagonal')).toBe(true);
  });

  test('forced-elimination vectors cover monotone chains and rotation/territory cases', () => {
    const vectors = loadBundle('forced_elimination.vectors.json');
    const ids = new Set(vectors.map((v) => v.id));

    expect(ids.has('forced_elimination.monotone_chain.step1.square8')).toBe(true);
    expect(ids.has('forced_elimination.monotone_chain.final.square8')).toBe(true);
    expect(ids.has('forced_elimination.rotation.skip_eliminated.square8')).toBe(true);
    expect(ids.has('forced_elimination.territory_explicit.square8')).toBe(true);
  });

  test('territory+line endgame vectors cover overlength lines and decision auto-exit', () => {
    const vectors = loadBundle('territory_line_endgame.vectors.json');
    const ids = new Set(vectors.map((v) => v.id));

    expect(ids.has('territory_line.overlong_line.step1.square8')).toBe(true);
    expect(ids.has('territory_line.single_point_swing.square19')).toBe(true);
    expect(ids.has('territory_line.decision_auto_exit.square8')).toBe(true);
  });

  test('hex edge-case vectors cover edge chains, corner regions, and 3p forced elimination', () => {
    const vectors = loadBundle('hex_edge_cases.vectors.json');
    const ids = new Set(vectors.map((v) => v.id));

    expect(ids.has('hex_edge_case.edge_chain.segment1.hexagonal')).toBe(true);
    expect(ids.has('hex_edge_case.corner_region.case1.hexagonal')).toBe(true);
    expect(ids.has('hex_edge_case.forced_elim_3p.hexagonal')).toBe(true);
  });

  test('multi-phase turn vectors cover line → territory flows across all board types', () => {
    const vectors = loadBundle('multi_phase_turn.vectors.json');
    const ids = new Set(vectors.map((v) => v.id));

    // Full-sequence turn including placement/capture/line/territory
    expect(ids.has('multi_phase.full_sequence_with_territory')).toBe(true);
    expect(ids.has('multi_phase.full_sequence_with_territory_square19')).toBe(true);
    expect(ids.has('multi_phase.full_sequence_with_territory_hex')).toBe(true);

    // Line → multi-region territory sequences by board type
    expect(ids.has('multi_phase.line_then_multi_region_territory.square8.step1_line')).toBe(true);
    expect(ids.has('multi_phase.line_then_multi_region_territory.square8.step2_regionB')).toBe(
      true
    );
    expect(ids.has('multi_phase.line_then_multi_region_territory.square8.step3_regionA')).toBe(
      true
    );

    expect(ids.has('multi_phase.line_then_multi_region_territory.square19.step1_line')).toBe(true);
    expect(ids.has('multi_phase.line_then_multi_region_territory.square19.step2_regionB')).toBe(
      true
    );
    expect(ids.has('multi_phase.line_then_multi_region_territory.square19.step3_regionA')).toBe(
      true
    );

    expect(ids.has('multi_phase.line_then_multi_region_territory.hex.step1_line')).toBe(true);
    expect(ids.has('multi_phase.line_then_multi_region_territory.hex.step2_regionB')).toBe(true);
    expect(ids.has('multi_phase.line_then_multi_region_territory.hex.step3_regionA')).toBe(true);
  });
});
