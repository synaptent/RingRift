import {
  computeContextSummary,
  classifyHotspotSeverity,
  type HotspotSeverity,
} from '../../scripts/analyze_rules_ux_telemetry';
import type {
  RulesUxAggregatesRoot,
  RulesUxContextAggregate,
  RulesUxSourceAggregate,
} from '../../src/shared/telemetry/rulesUxHotspotTypes';

describe('analyze_rules_ux_telemetry core helpers', () => {
  function makeSourceAgg(
    source: string,
    events: RulesUxSourceAggregate['events']
  ): RulesUxSourceAggregate {
    return { source, events };
  }

  function makeContextAgg(
    rulesContext: string,
    sources: RulesUxSourceAggregate[]
  ): RulesUxContextAggregate {
    return { rulesContext, sources };
  }

  it('ranks contexts by help opens per 100 games and flags high reopen/resign rates as hotspots', () => {
    const aggregates: RulesUxAggregatesRoot = {
      board: 'square8',
      num_players: 2,
      window: {
        start: '2025-11-01T00:00:00Z',
        end: '2025-11-30T23:59:59Z',
      },
      games: {
        started: 1200,
        completed: 800,
      },
      contexts: [
        // High-volume, high-churn ANM / forced elimination context
        makeContextAgg('anm_forced_elimination', [
          makeSourceAgg('hud', {
            help_open: 80,
            help_reopen: 40,
            weird_state_banner_impression: 200,
            weird_state_details_open: 120,
            resign_after_weird_state: 60,
          }),
        ]),
        // Moderate structural stalemate context
        makeContextAgg('structural_stalemate', [
          makeSourceAgg('victory_modal', {
            help_open: 20,
            help_reopen: 2,
            weird_state_banner_impression: 40,
            weird_state_details_open: 15,
            resign_after_weird_state: 6,
          }),
        ]),
        // Low-volume territory mini-region context
        makeContextAgg('territory_mini_region', [
          makeSourceAgg('sandbox', {
            help_open: 4,
            help_reopen: 0,
            weird_state_banner_impression: 10,
            weird_state_details_open: 3,
            resign_after_weird_state: 0,
          }),
        ]),
      ],
    };

    const gamesCompleted = aggregates.games.completed;
    const contextSummaries = aggregates.contexts.map((ctx) =>
      computeContextSummary(ctx, gamesCompleted, 1)
    );

    expect(contextSummaries).toHaveLength(3);

    const byContext: Record<string, (typeof contextSummaries)[number]> = {};
    for (const ctx of contextSummaries) {
      byContext[ctx.rulesContext] = ctx;
    }

    const anm = byContext['anm_forced_elimination'];
    const stalemate = byContext['structural_stalemate'];
    const territory = byContext['territory_mini_region'];

    expect(anm).toBeDefined();
    expect(stalemate).toBeDefined();
    expect(territory).toBeDefined();

    // Sanity-check aggregate math for ANM context.
    // helpOpensPer100Games = (80 / 800) * 100 = 10
    expect(anm!.helpOpensPer100Games).toBeCloseTo(10, 5);
    // helpReopenRate = 40 / 80 = 0.5
    expect(anm!.maxHelpReopenRate).toBeCloseTo(0.5, 5);
    // resignAfterWeirdRate = 60 / 200 = 0.3
    expect(anm!.maxResignAfterWeirdRate).toBeCloseTo(0.3, 5);

    // Rank contexts by help opens per 100 games.
    const sortedByHelpOpens = [...contextSummaries].sort(
      (a, b) => b.helpOpensPer100Games - a.helpOpensPer100Games
    );
    const topK = 2;
    const topList = sortedByHelpOpens.slice(0, topK);
    const topSet = new Set(topList.map((c) => c.rulesContext));

    // Verify ranking: ANM should be the hottest context by help opens/100 games.
    expect(topList[0].rulesContext).toBe('anm_forced_elimination');

    // Classify severities using the same rule as the script.
    const severityByContext = new Map<string, HotspotSeverity>();
    for (const ctx of contextSummaries) {
      const severity = classifyHotspotSeverity(ctx, topSet.has(ctx.rulesContext));
      severityByContext.set(ctx.rulesContext, severity);
    }

    // High-volume + high-churn ANM context should be HIGH severity.
    expect(severityByContext.get('anm_forced_elimination')).toBe('HIGH');

    // Structural stalemate has moderate volume and resign-after-weird rate; should be at least MEDIUM.
    expect(severityByContext.get('structural_stalemate')).toBe('MEDIUM');

    // Territory mini-region is low volume with negligible churn; should be LOW.
    expect(severityByContext.get('territory_mini_region')).toBe('LOW');
  });
});
