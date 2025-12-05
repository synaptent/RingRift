import curatedBundle from '../../public/scenarios/curated.json';
import type {
  CuratedScenarioBundle,
  ScenarioRulesConcept,
  LoadableScenario,
} from '../../src/client/sandbox/scenarioTypes';
import { RulesUxPhrases } from './rulesUxExpectations.testutil';

const bundle = curatedBundle as unknown as CuratedScenarioBundle;
const scenarios: LoadableScenario[] = bundle.scenarios;

// Mapping from rulesConcept -> canonical phrases that must appear in the
// player-facing description + rulesSnippet for that scenario.
const ConceptToPhrases: Partial<Record<ScenarioRulesConcept, readonly string[]>> = {
  movement_basic: RulesUxPhrases.movement.stackHeight,
  stack_height_mobility: RulesUxPhrases.movement.stackHeight,

  capture_basic: RulesUxPhrases.capture.basic,
  chain_capture_mandatory: RulesUxPhrases.capture.chainMandatory,

  lines_basic: RulesUxPhrases.lines.overlengthOption2,
  lines_overlength_option2: RulesUxPhrases.lines.overlengthOption2,

  territory_basic: RulesUxPhrases.territory.basicRegion,
  territory_multi_region: RulesUxPhrases.territory.basicRegion,
  territory_near_victory: RulesUxPhrases.victory.territory,

  victory_ring_elimination: RulesUxPhrases.victory.ringElimination,

  anm_forced_elimination: RulesUxPhrases.feAnm.forcedElimination,

  turn_multi_phase: ['movement or capture', 'line processing', 'Territory processing'],
  // board_intro_* and placement_basic / puzzle_capture concepts are intentionally
  // left without mandatory snippets here; they are covered by more general UX checks.
};

function escapeForRegex(snippet: string): string {
  return snippet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function toCaseInsensitiveRegex(snippet: string): RegExp {
  return new RegExp(escapeForRegex(snippet), 'i');
}

function getScenarioText(scenario: LoadableScenario): string {
  const description = typeof scenario.description === 'string' ? scenario.description : '';
  const rulesSnippet =
    typeof (scenario as any).rulesSnippet === 'string' ? (scenario as any).rulesSnippet : '';
  return `${description} ${rulesSnippet}`;
}

describe('Curated scenario rulesConcept - UX copy alignment', () => {
  it('keeps curated scenarios aligned with canonical rules semantics for their rulesConcept', () => {
    for (const scenario of scenarios) {
      const concept = scenario.rulesConcept as ScenarioRulesConcept | undefined;
      if (!concept) continue;

      const expectedSnippets = ConceptToPhrases[concept];
      if (!expectedSnippets || expectedSnippets.length === 0) {
        continue;
      }

      const blob = getScenarioText(scenario);

      expectedSnippets.forEach((snippet) => {
        expect(blob).toMatch(toCaseInsensitiveRegex(snippet));
      });
    }
  });

  it('links any territory_mini_region_q23 scenarios to Q23-style mini-region semantics when present', () => {
    const q23Scenarios = scenarios.filter((s) => s.rulesConcept === 'territory_mini_region_q23');

    if (q23Scenarios.length === 0) {
      // No Q23-focused curated scenarios yet; this test becomes a no-op
      // but will start asserting once such scenarios are added.
      return;
    }

    for (const scenario of q23Scenarios) {
      const blob = getScenarioText(scenario);

      // Copy must describe disconnected mini-regions and elimination inside the region.
      RulesUxPhrases.territory.miniRegionQ23.forEach((snippet) => {
        expect(blob).toMatch(toCaseInsensitiveRegex(snippet));
      });

      // When matrixScenarioId is present it should reference the Q23 edge-case family.
      if (typeof scenario.matrixScenarioId === 'string') {
        expect(scenario.matrixScenarioId).toMatch(/Q23/i);
      }
    }
  });
});
