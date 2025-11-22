import { getScenarioById, TerritoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → mini-region territory scenario (square8, Q23 numeric invariant)
 *
 * This is a light-weight, data-only check that the compact Q23 mini-region
 * scenario is correctly encoded in the shared rules matrix and accessible via
 * getScenarioById. Numeric invariants for this geometry are asserted in the
 * dedicated rules-layer tests:
 *
 *   - tests/unit/territoryProcessing.rules.test.ts
 *   - tests/unit/sandboxTerritory.rules.test.ts
 *   - tests/unit/sandboxTerritoryEngine.rules.test.ts
 *
 * Scenario ID:
 *   Rules_12_2_Q23_mini_region_square8_numeric_invariant
 */

describe('RulesMatrix → TerritoryRuleScenario – Q23 mini-region numeric invariant (square8)', () => {
  it('exposes the compact Q23 mini-region scenario via getScenarioById', () => {
    const id = 'Rules_12_2_Q23_mini_region_square8_numeric_invariant';
    const scenario = getScenarioById(id) as TerritoryRuleScenario | undefined;

    expect(scenario).toBeDefined();
    if (!scenario) return;

    expect(scenario.kind).toBe('territory');
    expect(scenario.boardType).toBe('square8');
    expect(scenario.movingPlayer).toBe(1);
    expect(scenario.ref.id).toBe(id);
    expect(scenario.ref.rulesSections).toContain('§12.2');
    expect(scenario.ref.faqRefs).toContain('Q23');

    expect(scenario.regions.length).toBe(1);
    const [region] = scenario.regions;

    // Geometry: 2×2 mini-region at (2,2)–(3,3) containing victim stacks for player 2.
    const expectedSpaces = [
      { x: 2, y: 2 },
      { x: 2, y: 3 },
      { x: 3, y: 2 },
      { x: 3, y: 3 },
    ];

    expect(region.spaces).toEqual(expectedSpaces);
    expect(region.controllingPlayer).toBe(1);
    expect(region.victimPlayer).toBe(2);
    expect(region.movingPlayerHasOutsideStack).toBe(true);
    expect(region.outsideStackPosition).toEqual({ x: 0, y: 0 });
    expect(region.selfEliminationStackHeight).toBe(3);
  });
});
