/**
 * Test suite for src/shared/teaching/scenarioTelemetry.ts
 *
 * Tests the mapping between teaching scenarios, rules concepts, and telemetry contexts.
 */

import {
  getRulesUxContextForRulesConcept,
  getRulesUxContextForTeachingScenario,
  getRulesUxContextForScenarioId,
} from '../../../src/shared/teaching/scenarioTelemetry';
import {
  TEACHING_SCENARIOS,
  type RulesConcept,
  type TeachingScenarioMetadata,
} from '../../../src/shared/teaching/teachingScenarios';

describe('scenarioTelemetry', () => {
  describe('getRulesUxContextForRulesConcept', () => {
    it('should map anm_forced_elimination to anm_forced_elimination', () => {
      expect(getRulesUxContextForRulesConcept('anm_forced_elimination')).toBe(
        'anm_forced_elimination'
      );
    });

    it('should map territory_mini_region to territory_mini_region', () => {
      expect(getRulesUxContextForRulesConcept('territory_mini_region')).toBe(
        'territory_mini_region'
      );
    });

    it('should map territory_multi_region_budget to territory_multi_region', () => {
      expect(getRulesUxContextForRulesConcept('territory_multi_region_budget')).toBe(
        'territory_multi_region'
      );
    });

    it('should map line_vs_territory_multi_phase to territory_multi_region', () => {
      // This flow touches both line rewards and territory processing;
      // attributed to territory_multi_region for aggregation
      expect(getRulesUxContextForRulesConcept('line_vs_territory_multi_phase')).toBe(
        'territory_multi_region'
      );
    });

    it('should map capture_chain_mandatory to capture_chain_mandatory', () => {
      expect(getRulesUxContextForRulesConcept('capture_chain_mandatory')).toBe(
        'capture_chain_mandatory'
      );
    });

    it('should map landing_on_own_marker to landing_on_own_marker', () => {
      expect(getRulesUxContextForRulesConcept('landing_on_own_marker')).toBe(
        'landing_on_own_marker'
      );
    });

    it('should map structural_stalemate to structural_stalemate', () => {
      expect(getRulesUxContextForRulesConcept('structural_stalemate')).toBe('structural_stalemate');
    });

    it('should map last_player_standing to last_player_standing', () => {
      expect(getRulesUxContextForRulesConcept('last_player_standing')).toBe('last_player_standing');
    });

    it('should return undefined for unknown/unmapped concepts', () => {
      // recovery_marker_slide is defined but not mapped in the switch
      expect(getRulesUxContextForRulesConcept('recovery_marker_slide')).toBeUndefined();
    });

    it('should return undefined for unknown concept', () => {
      // Force an unknown concept as a type escape hatch
      expect(getRulesUxContextForRulesConcept('unknown_concept' as RulesConcept)).toBeUndefined();
    });
  });

  describe('getRulesUxContextForTeachingScenario', () => {
    it('should return telemetryRulesContext if explicitly set', () => {
      const scenario: TeachingScenarioMetadata = {
        scenarioId: 'test.explicit.context',
        rulesConcept: 'anm_forced_elimination',
        flowId: 'test_flow',
        stepIndex: 1,
        stepKind: 'guided',
        recommendedBoardType: 'square8',
        recommendedNumPlayers: 2,
        showInTeachingOverlay: true,
        showInSandboxPresets: true,
        showInTutorialCarousel: true,
        learningObjectiveShort: 'Test scenario',
        telemetryRulesContext: 'structural_stalemate', // Explicit override
      };

      expect(getRulesUxContextForTeachingScenario(scenario)).toBe('structural_stalemate');
    });

    it('should fall back to mapping rulesConcept when telemetryRulesContext is not set', () => {
      const scenario: TeachingScenarioMetadata = {
        scenarioId: 'test.fallback',
        rulesConcept: 'capture_chain_mandatory',
        flowId: 'test_flow',
        stepIndex: 1,
        stepKind: 'interactive',
        recommendedBoardType: 'square8',
        recommendedNumPlayers: 2,
        showInTeachingOverlay: true,
        showInSandboxPresets: true,
        showInTutorialCarousel: true,
        learningObjectiveShort: 'Test scenario',
        // telemetryRulesContext NOT set
      };

      expect(getRulesUxContextForTeachingScenario(scenario)).toBe('capture_chain_mandatory');
    });

    it('should return undefined when scenario has unmapped concept and no explicit context', () => {
      const scenario: TeachingScenarioMetadata = {
        scenarioId: 'test.unmapped',
        rulesConcept: 'recovery_marker_slide', // Unmapped in the switch
        flowId: 'test_flow',
        stepIndex: 1,
        stepKind: 'guided',
        recommendedBoardType: 'square8',
        recommendedNumPlayers: 2,
        showInTeachingOverlay: true,
        showInSandboxPresets: true,
        showInTutorialCarousel: true,
        learningObjectiveShort: 'Test scenario',
      };

      expect(getRulesUxContextForTeachingScenario(scenario)).toBeUndefined();
    });
  });

  describe('getRulesUxContextForScenarioId', () => {
    it('should return context for existing fe_loop scenario', () => {
      const context = getRulesUxContextForScenarioId('teaching.fe_loop.step_1');
      expect(context).toBe('anm_forced_elimination');
    });

    it('should return context for existing structural_stalemate scenario', () => {
      const context = getRulesUxContextForScenarioId('teaching.structural_stalemate.step_1');
      expect(context).toBe('structural_stalemate');
    });

    it('should return context for existing lps scenario', () => {
      const context = getRulesUxContextForScenarioId('teaching.lps.step_1');
      expect(context).toBe('last_player_standing');
    });

    it('should return context for existing mini_region scenario', () => {
      const context = getRulesUxContextForScenarioId('teaching.mini_region.step_1');
      expect(context).toBe('territory_mini_region');
    });

    it('should return context for existing capture_chain scenario', () => {
      const context = getRulesUxContextForScenarioId('teaching.capture_chain.step_1');
      expect(context).toBe('capture_chain_mandatory');
    });

    it('should return context for existing line_territory scenario', () => {
      const context = getRulesUxContextForScenarioId('teaching.line_territory.step_1');
      // Mapped via line_vs_territory_multi_phase -> territory_multi_region
      expect(context).toBe('line_vs_territory_multi_phase');
    });

    it('should return undefined for non-existent scenario', () => {
      expect(getRulesUxContextForScenarioId('non.existent.scenario')).toBeUndefined();
    });

    it('should return undefined for empty string', () => {
      expect(getRulesUxContextForScenarioId('')).toBeUndefined();
    });
  });

  describe('TEACHING_SCENARIOS integration', () => {
    it('should have all scenarios with valid scenarioIds', () => {
      TEACHING_SCENARIOS.forEach((scenario) => {
        expect(scenario.scenarioId).toBeDefined();
        expect(typeof scenario.scenarioId).toBe('string');
        expect(scenario.scenarioId.length).toBeGreaterThan(0);
      });
    });

    it('should have at least one scenario for each concept with a context mapping', () => {
      const conceptsWithMappings: RulesConcept[] = [
        'anm_forced_elimination',
        'territory_mini_region',
        'capture_chain_mandatory',
        'structural_stalemate',
        'last_player_standing',
        'line_vs_territory_multi_phase',
      ];

      for (const concept of conceptsWithMappings) {
        const scenariosForConcept = TEACHING_SCENARIOS.filter((s) => s.rulesConcept === concept);
        expect(scenariosForConcept.length).toBeGreaterThan(0);
      }
    });

    it('should have telemetryRulesContext set or derivable for all scenarios shown in overlay', () => {
      const overlayScenarios = TEACHING_SCENARIOS.filter((s) => s.showInTeachingOverlay);

      overlayScenarios.forEach((scenario) => {
        const context = getRulesUxContextForTeachingScenario(scenario);
        // Most overlay scenarios should have a context for telemetry
        // Only recovery_marker_slide is expected to be unmapped currently
        if (scenario.rulesConcept !== 'recovery_marker_slide') {
          expect(context).toBeDefined();
        }
      });
    });

    it('should have consistent flowId groupings', () => {
      const flowGroups = new Map<string, TeachingScenarioMetadata[]>();

      TEACHING_SCENARIOS.forEach((scenario) => {
        const existing = flowGroups.get(scenario.flowId) ?? [];
        existing.push(scenario);
        flowGroups.set(scenario.flowId, existing);
      });

      // Each flow should have scenarios with increasing stepIndex
      flowGroups.forEach((scenarios, flowId) => {
        const indices = scenarios.map((s) => s.stepIndex).sort((a, b) => a - b);
        for (let i = 0; i < indices.length; i++) {
          expect(indices[i]).toBe(i + 1);
        }
      });
    });

    it('should have valid recommendedBoardType values', () => {
      TEACHING_SCENARIOS.forEach((scenario) => {
        expect(['square8', 'square19', 'hex8', 'hexagonal']).toContain(
          scenario.recommendedBoardType
        );
      });
    });

    it('should have valid recommendedNumPlayers values', () => {
      TEACHING_SCENARIOS.forEach((scenario) => {
        expect([2, 3, 4]).toContain(scenario.recommendedNumPlayers);
      });
    });

    it('should have valid stepKind values', () => {
      TEACHING_SCENARIOS.forEach((scenario) => {
        expect(['guided', 'interactive']).toContain(scenario.stepKind);
      });
    });

    it('should have optional difficultyTag that is valid when defined', () => {
      TEACHING_SCENARIOS.forEach((scenario) => {
        if (scenario.difficultyTag !== undefined) {
          expect(['intro', 'intermediate', 'advanced']).toContain(scenario.difficultyTag);
        }
      });
    });
  });

  describe('Edge cases', () => {
    it('getRulesUxContextForScenarioId handles scenarios with explicit override', () => {
      // Find a scenario that has an explicit telemetryRulesContext
      const scenarioWithExplicit = TEACHING_SCENARIOS.find(
        (s) => s.telemetryRulesContext !== undefined
      );

      if (scenarioWithExplicit) {
        const context = getRulesUxContextForScenarioId(scenarioWithExplicit.scenarioId);
        expect(context).toBe(scenarioWithExplicit.telemetryRulesContext);
      }
    });

    it('getRulesUxContextForTeachingScenario prefers explicit over derived', () => {
      // Create a scenario where explicit differs from what would be derived
      const scenario: TeachingScenarioMetadata = {
        scenarioId: 'test.explicit.priority',
        rulesConcept: 'anm_forced_elimination', // Would map to anm_forced_elimination
        flowId: 'test_flow',
        stepIndex: 1,
        stepKind: 'guided',
        recommendedBoardType: 'square8',
        recommendedNumPlayers: 2,
        showInTeachingOverlay: true,
        showInSandboxPresets: true,
        showInTutorialCarousel: true,
        learningObjectiveShort: 'Test',
        telemetryRulesContext: 'last_player_standing', // Explicit override
      };

      // Explicit should win
      expect(getRulesUxContextForTeachingScenario(scenario)).toBe('last_player_standing');
    });
  });
});
