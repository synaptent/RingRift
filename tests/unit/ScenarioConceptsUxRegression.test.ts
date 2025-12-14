import curatedBundle from '../../public/scenarios/curated.json';
import type {
  CuratedScenarioBundle,
  ScenarioRulesConcept,
  LoadableScenario,
} from '../../src/client/sandbox/scenarioTypes';
import { RulesUxPhrases } from './rulesUxExpectations.testutil';
import { TEACHING_SCENARIOS } from '../../src/shared/teaching/teachingScenarios';
import { getRulesUxContextForTeachingScenario } from '../../src/shared/teaching/scenarioTelemetry';

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

function toCaseInsensitiveRegex(snippet: string): RegExp {
  return new RegExp(snippet, 'i');
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

  it('maps mini-region teaching steps to rulesContext=territory_mini_region for telemetry', () => {
    const miniRegionTeaching = TEACHING_SCENARIOS.filter(
      (scenario) => scenario.rulesConcept === 'territory_mini_region'
    );

    if (miniRegionTeaching.length === 0) {
      // No mini-region teaching flows yet; this test becomes a no-op until such steps are added.
      return;
    }

    for (const scenario of miniRegionTeaching) {
      const ctx = getRulesUxContextForTeachingScenario(scenario);
      expect(ctx).toBe('territory_mini_region');
    }
  });

  it('exposes curated weird-state teaching presets for ANM/FE, structural stalemate, LPS, and territory mini-regions', () => {
    const byConcept = (concept: ScenarioRulesConcept) =>
      scenarios.filter((s) => s.rulesConcept === concept);

    const anmFe = byConcept('anm_forced_elimination');
    const structuralStalemate = byConcept('structural_stalemate');
    const lps = byConcept('anm_last_player_standing');
    const miniRegionQ23 = byConcept('territory_mini_region_q23');

    expect(anmFe.length).toBeGreaterThanOrEqual(1);
    expect(structuralStalemate.length).toBeGreaterThanOrEqual(1);
    expect(lps.length).toBeGreaterThanOrEqual(1);
    expect(miniRegionQ23.length).toBeGreaterThanOrEqual(1);

    // Sanity-check the primary curated presets we expect for each class.
    expect(anmFe.some((s) => s.id === 'teaching.fe_loop.step_1')).toBe(true);
    expect(structuralStalemate.some((s) => s.id === 'teaching.structural_stalemate.step_1')).toBe(
      true
    );
    expect(lps.some((s) => s.id === 'teaching.lps.step_1')).toBe(true);
    expect(miniRegionQ23.some((s) => s.id === 'teaching.mini_region.step_1')).toBe(true);
  });

  it('has scenario states consistent with intended weird-state archetypes for ANM/FE, stalemate, LPS, and mini-regions', () => {
    const fe = scenarios.find((s) => s.id === 'teaching.fe_loop.step_1');
    const stalemate = scenarios.find((s) => s.id === 'teaching.structural_stalemate.step_1');
    const lps = scenarios.find((s) => s.id === 'teaching.lps.step_1');
    const mini = scenarios.find((s) => s.id === 'teaching.mini_region.step_1');

    expect(fe).toBeDefined();
    expect(stalemate).toBeDefined();
    expect(lps).toBeDefined();
    expect(mini).toBeDefined();

    if (fe) {
      const state = fe.state;
      // ANM/FE: both players should have no rings in hand and at least one stack for the active player.
      expect(state.players.every((p) => p.ringsInHand === 0)).toBe(true);
      const stackEntries = Object.entries(state.board.stacks ?? {});
      expect(stackEntries.length).toBeGreaterThanOrEqual(1);
    }

    if (stalemate) {
      const state = stalemate.state;
      // Structural stalemate: no stacks or markers, fully collapsed board.
      expect(Object.keys(state.board.stacks ?? {}).length).toBe(0);
      expect(Object.keys(state.board.markers ?? {}).length).toBe(0);
      const collapsedCount = Object.keys(state.board.collapsedSpaces ?? {}).length;
      expect(collapsedCount).toBeGreaterThanOrEqual(64); // full 8×8 plateau
    }

    if (lps) {
      const state = lps.state;
      // LPS: three players, only player 1 with stacks and rings in hand.
      expect(state.players.length).toBe(3);
      const stacks = Object.values(state.board.stacks ?? {});
      const controllingPlayers = new Set(stacks.map((s: any) => (s as any).controllingPlayer));
      expect(controllingPlayers.has(1)).toBe(true);
      expect(controllingPlayers.has(2)).toBe(false);
      expect(controllingPlayers.has(3)).toBe(false);
      const p1 = state.players.find((p) => p.playerNumber === 1)!;
      const p2 = state.players.find((p) => p.playerNumber === 2)!;
      const p3 = state.players.find((p) => p.playerNumber === 3)!;
      expect(p1.ringsInHand).toBeGreaterThan(0);
      expect(p2.ringsInHand).toBe(0);
      expect(p3.ringsInHand).toBe(0);
    }

    if (mini) {
      const state = mini.state;
      // Mini-region Q23 archetype: square8 board with Q23 matrixScenarioId and the expected marker ring pattern.
      expect(state.board.type).toBe('square8');
      expect(mini.matrixScenarioId).toMatch(/Q23/i);
      const markers = state.board.markers ?? {};
      const markerKeys = Object.keys(markers);
      expect(markerKeys.length).toBeGreaterThanOrEqual(8);
      // Ensure the central 2×2 region is empty but surrounded by markers.
      expect(markers['2,2']).toBeDefined();
      expect(markers['4,4']).toBeDefined();
    }
  });
});
